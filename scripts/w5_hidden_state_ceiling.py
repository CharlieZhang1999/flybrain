#!/usr/bin/env python
"""W5 T9: is the decoder or the signal the bottleneck? Supervised nonlinear decoders on the v1
hidden-state dataset, against the linear baseline of PR #19 (0.713 balanced accuracy on calib).

Reused from scripts/w5_hidden_state_baseline.py (imported, not copied): the loaders (labels,
brain_inputs: decoder mask removed, 50 ms bins, 80 per episode), the training-fold-only
preprocessing (features), the logistic-regression row, and exactly the same 5 outer folds by
episode (folds(arange(120), 5, SEED=7500)). Inner splits use the baseline's inner seeds
(SEED + 10 + f); the nets use the first inner fold (1/4 of the training episodes) as the
validation split for early stopping and the grid.

Models (all supervised on the per-bin state label; per-bin output, 4 states)
  linear       the PR #19 row: sqrt counts, z-score, 100 PCs, logistic regression, C by inner
               4-fold CV (the reference; must give 0.713 +- 0.01 on calib).
  mlp          bin +- W_CTX bins (+-0.5 s, zero padded at the episode edges): a per-bin linear
               projection to d, the 21 x d window flattened, 2 hidden ReLU layers.
  gru_causal   per-bin projection to d (ReLU), 1-layer GRU forward in time over the 80 bins.
  gru_bi       the same, bidirectional.
  tf           NDT-style transformer from scratch: per-bin token = linear projection of the
               input vector + learned position embedding, TransformerEncoder (pre-norm),
               full (non-causal) attention over the 80 bins, per-bin linear classifier.
Inputs for the nets ('front'): 'full' = sqrt counts of the neurons with >= N_MIN training
spikes, z-scored on the training bins (all ~6-7k active unmasked neurons, the net's first layer
is the projection), or 'pca' = the baseline's 100 whitened PCs. Both are fitted on the
training episodes of the split only.
Training: AdamW (lr LR), batches of BATCH episodes, cross entropy, input dropout, dropout and
weight decay from the light grid GRID[model]; early stopping on the validation balanced
accuracy (patience PATIENCE, at most MAX_EP epochs). The grid point with the best validation
bacc is refitted on all training episodes of the outer fold for its best epoch count, then
predicts the held-out episodes.
Metrics (held-out bins pooled over the 5 folds): balanced accuracy, recall per state, per-fold
bacc; segment vote: each 2 s segment (bins 0-39, 40-79) gets the majority predicted state
(ties broken by the mean predicted probability) -> accuracy over the 240 segments (and its
balanced version). Note: segment edges are always at 2 s, so bidirectional / non-causal models
and the vote can use the whole segment; gru_causal is the online comparison.

Usage
  python scripts/w5_hidden_state_ceiling.py [--models linear,mlp,...] [--cfgs calib,shuffle]
Output (results/w5/hidden_state/ceiling/): folds/<model>_<cfg>_f<k>.npz (per-fold cache: a
re-run skips finished folds), row_<model>_<cfg>.json, summary.md, results.json, log.txt.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import w5_hidden_state_baseline as bl  # noqa: E402

OUT = bl.HS / 'ceiling'
STATES = ('baseline', 'hunger', 'thirst', 'arousal')
SEG = ((0, 40), (40, 80))
W_CTX = 10
LR, BATCH, MAX_EP, PATIENCE = 1e-3, 8, 150, 20
MODELS = ('linear', 'mlp', 'gru_causal', 'gru_bi', 'tf')
# light grid: front-end x regularisation (input dropout, dropout, weight decay) [x size]
_REG = (dict(in_drop=0.2, drop=0.3, wd=1e-3), dict(in_drop=0.5, drop=0.5, wd=1e-2))
GRID = {
    'mlp': [dict(front=f, d=64, h=128, **r) for f in ('pca', 'full') for r in _REG],
    'gru_causal': [dict(front=f, d=64, **r) for f in ('pca', 'full') for r in _REG],
    'gru_bi': [dict(front=f, d=64, **r) for f in ('pca', 'full') for r in _REG],
    'tf': [dict(front=f, d=64, layers=2, heads=4, **r) for f in ('pca', 'full') for r in _REG]
    + [dict(front='full', d=128, layers=4, heads=4, **_REG[1])],
}


# ---------------------------------------------------------------- preprocessing (numpy only)
def prep(X, tr, te, front):
    """Training-split-only preprocessing -> (Ftr (len(tr), NB, f), Fte (len(te), NB, f)).
    'pca' is the baseline's features(); 'full' is the same without the PCA step."""
    nb = X.shape[1]
    if front == 'pca':
        Ftr, Fte, _ = bl.features(X, tr, te, True)
    else:
        keep = np.flatnonzero(X[tr].sum((0, 1)) >= bl.N_MIN)
        Ftr = np.sqrt(X[tr][:, :, keep].reshape(-1, len(keep)))
        Fte = np.sqrt(X[te][:, :, keep].reshape(-1, len(keep)))
        mu, sd = Ftr.mean(0), Ftr.std(0)
        sd[sd == 0] = 1.0
        Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
    return (Ftr.reshape(len(tr), nb, -1).astype(np.float32), Fte.reshape(len(te), nb, -1).astype(np.float32))


def bacc(y, p, n=4):
    y, p = np.asarray(y).ravel(), np.asarray(p).ravel()
    return float(np.mean([(p[y == c] == c).mean() for c in range(n) if (y == c).any()]))


def segment_vote(pred, y, prob=None):
    """Majority predicted state per segment (ties -> highest mean probability, else lowest id).
    Returns (true per segment, voted per segment), each (n_ep * len(SEG),)."""
    yt, yv = [], []
    for k in range(pred.shape[0]):
        for a, b in SEG:
            seg = y[k, a:b]
            if not (seg == seg[0]).all():
                raise ValueError(f'episode row {k}: segment {a}-{b} mixes states')
            cnt = np.bincount(pred[k, a:b], minlength=4).astype(float)
            if prob is not None:
                cnt += 1e-3 * prob[k, a:b].mean(0)
            yt.append(int(seg[0]))
            yv.append(int(np.argmax(cnt)))
    return np.array(yt), np.array(yv)


def summarize(pred, y, prob=None):
    yt, pt = y.ravel(), pred.ravel()
    st_, sv = segment_vote(pred, y, prob)
    return dict(bacc=bacc(yt, pt), recall={s: float((pt[yt == c] == c).mean()) for c, s in enumerate(STATES)},
                seg_acc=float((st_ == sv).mean()), seg_bacc=bacc(st_, sv),
                seg_recall={s: float((sv[st_ == c] == c).mean()) for c, s in enumerate(STATES)})


# ---------------------------------------------------------------- models (torch)
def build(model, n_in, hp):
    import torch
    from torch import nn

    class Proj(nn.Module):
        def __init__(self, d, act=True):
            super().__init__()
            self.drop_in = nn.Dropout(hp['in_drop'])
            self.lin = nn.Linear(n_in, d)
            self.act = nn.ReLU() if act else nn.Identity()
            self.drop = nn.Dropout(hp['drop'])

        def forward(self, x):
            return self.drop(self.act(self.lin(self.drop_in(x))))

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            d, h, w = hp['d'], hp['h'], 2 * W_CTX + 1
            self.proj = Proj(d, act=False)
            self.net = nn.Sequential(nn.Linear(w * d, h), nn.ReLU(), nn.Dropout(hp['drop']),
                                     nn.Linear(h, h), nn.ReLU(), nn.Dropout(hp['drop']), nn.Linear(h, 4))

        def forward(self, x):                                   # x (B, T, n_in)
            z = self.proj(x)                                    # (B, T, d)
            z = nn.functional.pad(z, (0, 0, W_CTX, W_CTX))      # zero = training mean
            win = z.unfold(1, 2 * W_CTX + 1, 1)                 # (B, T, d, w)
            return self.net(win.flatten(2))

    class GRU(nn.Module):
        def __init__(self, bi):
            super().__init__()
            d = hp['d']
            self.proj = Proj(d)
            self.rnn = nn.GRU(d, d, batch_first=True, bidirectional=bi)
            self.drop = nn.Dropout(hp['drop'])
            self.out = nn.Linear(d * (2 if bi else 1), 4)

        def forward(self, x):
            return self.out(self.drop(self.rnn(self.proj(x))[0]))

    class TF(nn.Module):
        def __init__(self):
            super().__init__()
            d = hp['d']
            self.proj = Proj(d, act=False)
            self.pos = nn.Parameter(torch.zeros(1, bl.NB, d))
            nn.init.normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(d, hp['heads'], 2 * d, hp['drop'], batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, hp['layers'], enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.out = nn.Linear(d, 4)

        def forward(self, x):
            z = self.proj(x) + self.pos[:, :x.shape[1]]
            return self.out(self.norm(self.enc(z)))

    return {'mlp': MLP, 'gru_causal': lambda: GRU(False), 'gru_bi': lambda: GRU(True), 'tf': TF}[model]()


def train(model, hp, Ftr, ytr, Fva=None, yva=None, n_epochs=None, seed=0):
    """Fit; with a validation split: early stopping on its bacc, returns (best_bacc, best_epoch, None).
    Without: n_epochs epochs, returns the fitted net."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = build(model, Ftr.shape[2], hp)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=hp['wd'])
    Xt, Yt = torch.from_numpy(Ftr), torch.from_numpy(ytr.astype(np.int64))
    lossf = torch.nn.CrossEntropyLoss()
    best, best_ep, bad = -1.0, 0, 0
    for ep in range(1, (n_epochs or MAX_EP) + 1):
        net.train()
        for b in np.array_split(rng.permutation(len(Xt)), max(1, len(Xt) // BATCH)):
            opt.zero_grad()
            loss = lossf(net(Xt[b]).reshape(-1, 4), Yt[b].reshape(-1))
            loss.backward()
            opt.step()
        if Fva is None:
            continue
        p = predict(net, Fva).argmax(-1)
        ba = bacc(yva, p)
        if ba > best + 1e-4:
            best, best_ep, bad = ba, ep, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    return (best, best_ep, None) if Fva is not None else net


def predict(net, F):
    import torch
    net.eval()
    with torch.no_grad():
        return torch.softmax(net(torch.from_numpy(F)), -1).numpy()


# ---------------------------------------------------------------- CV
def run_fold(model, X, y, f, te, logf):
    tr = np.setdiff1d(np.arange(bl.N_EP), te)
    if model == 'linear':                                       # the PR #19 row, verbatim logic
        ps, ys = {C: [] for C in bl.C_GRID}, []
        for ite in bl.folds(tr, bl.N_INNER, bl.SEED + 10 + f):
            itr = np.setdiff1d(tr, ite)
            Ftr, Fte, _ = bl.features(X, itr, ite, True)
            for C in bl.C_GRID:
                ps[C].append(bl.logreg(Ftr, y[itr].ravel(), Fte, C))
            ys.append(y[ite].ravel())
        score = {C: bl.bacc(np.concatenate(ys), np.concatenate(ps[C])) for C in bl.C_GRID}
        C = max(bl.C_GRID, key=lambda c: (score[c], -c))
        from sklearn.linear_model import LogisticRegression
        Ftr, Fte, _ = bl.features(X, tr, te, True)
        prob = LogisticRegression(C=C, max_iter=3000).fit(Ftr, y[tr].ravel()).predict_proba(Fte)
        return prob.reshape(len(te), bl.NB, 4), dict(hp=dict(C=C), inner={str(k): v for k, v in score.items()})
    ite = bl.folds(tr, bl.N_INNER, bl.SEED + 10 + f)[0]          # validation split
    itr = np.setdiff1d(tr, ite)
    cache_front, scores = {}, []
    for g, hp in enumerate(GRID[model]):
        if hp['front'] not in cache_front:
            cache_front = {hp['front']: prep(X, itr, ite, hp['front'])}
        Ftr, Fva = cache_front[hp['front']]
        t0 = time.time()
        ba, be, _ = train(model, hp, Ftr, y[itr], Fva, y[ite], seed=bl.SEED + 100 * f + g)
        scores.append((ba, be))
        logf(f'    grid {g} {hp}: val bacc {ba:.3f} @ epoch {be} ({time.time() - t0:.0f} s)')
    g = max(range(len(scores)), key=lambda j: (scores[j][0], -j))
    hp, n_ep = GRID[model][g], max(1, scores[g][1])
    Ftr, Fte = prep(X, tr, te, hp['front'])
    net = train(model, hp, Ftr, y[tr], n_epochs=n_ep, seed=bl.SEED + 100 * f + g)
    return predict(net, Fte), dict(hp=hp, grid=g, val_bacc=scores[g][0], epochs=n_ep,
                                   val=[dict(bacc=a, epoch=b) for a, b in scores])


def run(model, cfg, X, y, logf):
    fd = OUT / 'folds'
    fd.mkdir(parents=True, exist_ok=True)
    prob = np.zeros((bl.N_EP, bl.NB, 4), np.float32)
    infos = []
    for f, te in enumerate(bl.folds(np.arange(bl.N_EP), bl.N_FOLD, bl.SEED)):
        cp = fd / f'{model}_{cfg}_f{f}.npz'
        if cp.exists():
            with np.load(cp, allow_pickle=False) as d:
                if not np.array_equal(d['te'], te):
                    raise RuntimeError(f'{cp}: cached fold has other episodes')
                prob[te], info = d['prob'], json.loads(str(d['info']))
            logf(f'  fold {f}: cached, bacc {info["bacc"]:.3f}')
        else:
            t0 = time.time()
            p, info = run_fold(model, X, y, f, te, logf)
            prob[te] = p
            info.update(fold=f, bacc=bacc(y[te], p.argmax(-1)), wall_s=time.time() - t0)
            np.savez_compressed(cp, prob=p, te=te, info=json.dumps(info))
            logf(f'  fold {f}: held-out bacc {info["bacc"]:.3f}, {info["hp"]} ({info["wall_s"]:.0f} s)')
        infos.append(info)
    pred = prob.argmax(-1).astype(np.int64)
    res = summarize(pred, y.astype(np.int64), prob)
    res.update(bacc_folds=[i['bacc'] for i in infos], folds=infos,
               wall_s=float(sum(i.get('wall_s', 0) for i in infos)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--cfgs', default=','.join(bl.CFGS))
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    import pandas as pd
    import torch
    from sim.cns import brain_mask
    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)

    def logf(m):
        print(m, flush=True)
        with open(OUT / 'log.txt', 'a') as fh:
            fh.write(m + '\n')

    meta = pd.read_parquet(ROOT / 'data' / 'malecns' / 'malecns_traced_meta.parquet')
    sets = json.loads((bl.HS / 'state_sets.json').read_text())
    mask = np.array(sets['decoder_mask']['neuron_idx'], np.int64)
    bm = brain_mask(meta).copy()
    bm[mask] = False
    inp_idx = np.flatnonzero(bm)
    y = bl.labels()
    logf(f'== T9 ceiling run {time.strftime("%Y-%m-%d %H:%M:%S")}: models {a.models}, cfgs {a.cfgs}')
    for cfg in a.cfgs.split(','):
        X = bl.brain_inputs(cfg, inp_idx, mask, len(meta))
        logf(f'[{cfg}] {X.shape[2]:,} active unmasked input neurons')
        for model in a.models.split(','):
            logf(f'[{model} / {cfg}]')
            res = run(model, cfg, X, y, logf)
            (OUT / f'row_{model}_{cfg}.json').write_text(json.dumps(res, indent=1, default=float))
            logf(f'  pooled bacc {res["bacc"]:.3f} (folds {", ".join(f"{b:.3f}" for b in res["bacc_folds"])}); '
                 f'recall {", ".join(f"{s} {v:.2f}" for s, v in res["recall"].items())}; '
                 f'segment vote {res["seg_acc"]:.3f}')
    write_summary(logf)


def write_summary(logf=print):
    rows, res = [], {}
    for model in MODELS:
        for cfg in bl.CFGS:
            p = OUT / f'row_{model}_{cfg}.json'
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            res[f'{model}_{cfg}'] = r
            rows.append(f'| {model} | {cfg} | {r["bacc"]:.3f} | {np.mean(r["bacc_folds"]):.3f} +- '
                        f'{np.std(r["bacc_folds"], ddof=1):.3f} | '
                        + ' | '.join(f'{r["recall"][s]:.2f}' for s in STATES)
                        + f' | {r["seg_acc"]:.3f} | {r["seg_bacc"]:.3f} | {r["wall_s"] / 60:.1f} |')
    head = ('| model | cfg | bacc (pooled) | fold mean +- sd | recall baseline | recall hunger | recall thirst '
            '| recall arousal | segment vote acc | segment vote bacc | train min |\n|' + '---|' * 12 + '\n')
    (OUT / 'results.json').write_text(json.dumps(res, indent=1, default=float))
    (OUT / 'summary.md').write_text('# W5 T9: supervised decoders on the v1 hidden-state dataset '
                                    '(4 states, 50 ms bins, the PR #19 5-fold CV by episode)\n\n'
                                    + head + '\n'.join(rows) + '\n')
    logf((OUT / 'summary.md').read_text())


if __name__ == '__main__':
    main()
