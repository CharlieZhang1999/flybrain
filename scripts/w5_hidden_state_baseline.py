#!/usr/bin/env python
"""W5 T7 Phase B, task 6: baselines on the hidden-state dataset.

Inputs (results/w5/hidden_state/dataset/, written by scripts/w5_hidden_state.py)
  activity/ep{k:03d}_{cfg}.npz   spikes, decoder-mask neurons already removed
  labels/ep{k:03d}.npz           state id per 10 ms bin (used only as the target)
  state_sets.json 'decoder_mask' the masked neurons (checked to be absent from the inputs)

Design (all choices fixed here)
  bins      50 ms, 80 per 4 s episode. Target = state id of the bin (the 10 ms label bins
            of a 50 ms bin always agree: segment edges are at 0, 2, 4 s).
  inputs    'brain': sim.cns.brain_mask neurons (brain, non-sensory, non-motor) minus the
            decoder mask (ISN + BiT + pC1/P1 male-specific + ER5, 153 neurons), spike count
            per bin, sqrt-transformed; neurons with >= N_MIN spikes over the training
            episodes of the fold; z-scored on the training bins; projected on K_PC principal
            components fitted on the training bins only (sklearn PCA, randomized, fixed seed).
            'words': the stimulus spec only, no activity: per bin, the rate (Hz) of each
            word of the vocabulary if it is on in that bin (else 0), the episode's word
            identities (one-hot, whole episode) and bg_hz; z-scored on the training bins.
  model     multinomial logistic regression (sklearn, lbfgs), C chosen per outer fold from
            C_GRID by inner 4-fold CV by episode on the training episodes (balanced accuracy).
  folds     5-fold CV by episode: the 120 episodes are permuted (seed SEED) and dealt
            round-robin; the same folds for every cfg and input set.
  metric    balanced accuracy (mean per-class recall over the 4 states) on the held-out bins
            pooled over the 5 folds; per-fold values too. Chance = 1 / 4 = 0.25 for balanced
            accuracy; empirical chance: the same 'brain' pipeline on calib with the label
            sequences permuted across episodes (seed SEED + 1), which keeps every label
            sequence and every activity file but breaks their pairing.
  configs   calib (real wiring) and shuffle (PR #14 shuffle control), fitted separately.

Usage
  python scripts/w5_hidden_state_baseline.py          # all rows (~10 min, 4 cores)
Output (results/w5/hidden_state/baseline/)
  summary.md, results.json, confusion_<row>.csv, pred_<row>.npz (held-out predictions),
  row_<row>.json (per-row cache: a re-run after a restart skips finished rows), log.txt (appended)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stimuli import states as st  # noqa: E402

HS = ROOT / 'results' / 'w5' / 'hidden_state'
DS = HS / 'dataset'
OUT = HS / 'baseline'
CFGS = ('calib', 'shuffle')
N_EP, T, BIN_S = 120, 4.0, 0.05
NB = int(round(T / BIN_S))
STEP = 1e-4
CLASSES = (0, 1, 2, 3)            # baseline, hunger, thirst, arousal (stimuli.states.STATES)
N_MIN = 5
K_PC = 100
C_GRID = (0.001, 0.01, 0.1, 1.0)
N_FOLD, N_INNER = 5, 4
SEED = 7500


def folds(eps, n, seed):
    p = np.random.default_rng(seed).permutation(eps)
    return [np.sort(p[j::n]) for j in range(n)]


def labels():
    y = np.zeros((N_EP, NB), np.int16)
    for k in range(N_EP):
        with np.load(DS / 'labels' / f'ep{k:03d}.npz') as d:
            sid, r = d['state_id'], int(round(BIN_S / float(d['bin_s'])))
        s = sid.reshape(NB, r)
        if not (s == s[:, :1]).all():
            raise ValueError(f'ep{k:03d}: a 50 ms bin mixes states')
        y[k] = s[:, 0]
    return y


def brain_inputs(cfg, inp_idx, mask, n):
    """(N_EP, NB, n_active) spike counts per 50 ms bin (float32) over the input neurons that
    spike at least once in this cfg (all episodes; a label-free filter that only drops silent
    columns, to keep memory small). n: number of model neurons."""
    col = -np.ones(n, np.int64)
    col[inp_idx] = np.arange(len(inp_idx))
    eps = []
    for k in range(N_EP):
        with np.load(DS / 'activity' / f'ep{k:03d}_{cfg}.npz') as d:
            t, i = d['t'], d['i']
        if np.isin(i, mask).any():
            raise RuntimeError(f'ep{k:03d}_{cfg}: decoder-mask neurons in the activity file')
        c = col[i.astype(np.int64)]
        ok = c >= 0
        eps.append((np.minimum(t[ok] // int(round(BIN_S / STEP)), NB - 1), c[ok]))
    active = np.unique(np.concatenate([c for _, c in eps]))
    pos = -np.ones(len(inp_idx), np.int64)
    pos[active] = np.arange(len(active))
    X = np.zeros((N_EP, NB, len(active)), np.float32)
    for k, (b, c) in enumerate(eps):
        np.add.at(X[k], (b, pos[c]), 1.0)
    return X


def word_inputs(specs):
    vocab = sorted({w['word'] for s in specs for w in s['words']})
    vi = {w: j for j, w in enumerate(vocab)}
    centre = (np.arange(NB) + 0.5) * BIN_S
    X = np.zeros((N_EP, NB, 2 * len(vocab) + 1), np.float32)
    for k, s in enumerate(specs):
        for w in s['words']:
            on = (centre >= w['onset']) & (centre < w['offset'])
            X[k, on, vi[w['word']]] = w['rate_hz']
            X[k, :, len(vocab) + vi[w['word']]] = 1.0
        X[k, :, -1] = s['bg_hz']
    return X, vocab


def features(X, tr, te, brain):
    """Training-fold-only preprocessing. brain: neurons with >= N_MIN training spikes,
    sqrt counts, z-score, K_PC PCs (whitened). words: z-score only.
    Returns (F_train (bins x f), F_test, n_inputs)."""
    from sklearn.decomposition import PCA
    keep = np.flatnonzero(X[tr].sum((0, 1)) >= N_MIN) if brain else np.arange(X.shape[2])
    Xtr = X[tr][:, :, keep].reshape(-1, len(keep))
    Xte = X[te][:, :, keep].reshape(-1, len(keep))
    if brain:
        Xtr, Xte = np.sqrt(Xtr), np.sqrt(Xte)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    if brain:
        p = PCA(n_components=min(K_PC, Xtr.shape[1]), svd_solver='randomized', random_state=SEED).fit(Xtr)
        Xtr, Xte = p.transform(Xtr), p.transform(Xte)
        s = Xtr.std(0); s[s == 0] = 1.0
        Xtr, Xte = Xtr / s, Xte / s
    return Xtr, Xte, len(keep)


def logreg(Ftr, ytr, Fte, C):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, max_iter=3000).fit(Ftr, ytr).predict(Fte)


def bacc(y, p):
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y, p))


def cv(X, y, brain, logf):
    """Outer 5-fold CV by episode with inner C selection. Returns held-out predictions (N_EP, NB)."""
    pred = np.full(y.shape, -1, np.int16)
    info = []
    for f, te in enumerate(folds(np.arange(N_EP), N_FOLD, SEED)):
        t0 = time.time()
        tr = np.setdiff1d(np.arange(N_EP), te)
        ps, ys = {C: [] for C in C_GRID}, []
        for ite in folds(tr, N_INNER, SEED + 10 + f):
            itr = np.setdiff1d(tr, ite)
            Ftr, Fte, _ = features(X, itr, ite, brain)
            for C in C_GRID:
                ps[C].append(logreg(Ftr, y[itr].ravel(), Fte, C))
            ys.append(y[ite].ravel())
        score = {C: bacc(np.concatenate(ys), np.concatenate(ps[C])) for C in C_GRID}
        C = max(C_GRID, key=lambda c: (score[c], -c))
        Ftr, Fte, n_in = features(X, tr, te, brain)
        pred[te] = logreg(Ftr, y[tr].ravel(), Fte, C).reshape(len(te), NB)
        ba = bacc(y[te].ravel(), pred[te].ravel())
        info.append(dict(fold=f, n_inputs=int(n_in), C=C, inner=score, bacc=ba))
        logf(f'  fold {f}: {n_in} inputs, C {C} (inner {", ".join(f"{c:g}:{v:.3f}" for c, v in score.items())}), '
             f'held-out bacc {ba:.3f} ({time.time() - t0:.0f} s)')
    return pred, info


def main():
    from sim.cns import brain_mask
    OUT.mkdir(parents=True, exist_ok=True)
    logp = OUT / 'log.txt'

    def logf(m):
        print(m, flush=True)
        with open(logp, 'a') as fh:
            fh.write(m + '\n')

    meta = pd.read_parquet(ROOT / 'data' / 'malecns' / 'malecns_traced_meta.parquet')
    sets = json.loads((HS / 'state_sets.json').read_text())
    mask = np.array(sets['decoder_mask']['neuron_idx'], np.int64)
    pos = {int(b): k for k, b in enumerate(meta.bodyId)}
    for s in sets['decoder_mask']['states']:
        if not set(pos[b] for b in sets['body_ids'][s]) <= set(mask.tolist()):
            raise RuntimeError(f'decoder mask misses {s} neurons')
    bm = brain_mask(meta).copy()
    bm[mask] = False
    inp_idx = np.flatnonzero(bm)
    logf(f'brain inputs: {len(inp_idx):,} neurons (brain_mask minus {len(mask)} decoder-mask neurons)')
    y = labels()
    specs = json.loads((DS / 'specs.json').read_text())
    idx = pd.read_csv(DS / 'index.csv')
    for cfg in CFGS:
        n = int((idx.cfg == cfg).sum())
        if n != N_EP:
            raise RuntimeError(f'{cfg}: {n} of {N_EP} episodes in index.csv')
    rows, res = [], {}
    Xw, vocab = word_inputs(specs)
    runs = [('calib', 'brain', False), ('calib', 'brain', True), ('shuffle', 'brain', False),
            ('words', 'words', False)]
    Xc = {}
    for cfg, inp, perm in runs:
        name = f'{cfg}_{inp}' + ('_permuted' if perm else '')
        logf(f'[{name}]')
        cache = OUT / f'row_{name}.json'
        if cache.exists() and (OUT / f'pred_{name}.npz').exists():      # resume after a restart
            c = json.loads(cache.read_text())
            res[name], rows = c['res'], rows + [c['row']]
            logf(f'  cached: pooled balanced accuracy {res[name]["bacc"]:.3f}')
            continue
        if inp == 'brain':
            if cfg not in Xc:
                Xc.clear()
                t0 = time.time()
                Xc[cfg] = brain_inputs(cfg, inp_idx, mask, len(meta))
                logf(f'  loaded {cfg} in {time.time() - t0:.0f} s: {Xc[cfg].shape[2]:,} input neurons spike at least once')
            X = Xc[cfg]
        else:
            X = Xw
        yy = y[np.random.default_rng(SEED + 1).permutation(N_EP)] if perm else y
        pred, info = cv(X, yy, inp == 'brain', logf)
        yt, pt = yy.ravel(), pred.ravel()
        cm = pd.crosstab(pd.Series([st.STATES[c] for c in yt], name='true'),
                         pd.Series([st.STATES[c] for c in pt], name='pred'))
        cm.to_csv(OUT / f'confusion_{name}.csv')
        rec = {st.STATES[c]: float((pt[yt == c] == c).mean()) for c in CLASSES}
        res[name] = dict(bacc=bacc(yt, pt), bacc_folds=[i['bacc'] for i in info], folds=info, recall=rec)
        np.savez_compressed(OUT / f'pred_{name}.npz', pred=pred, y=yy)
        rows.append(dict(row=name, inputs=inp, cfg=cfg if inp == 'brain' else 'n/a',
                         bacc=res[name]['bacc'], fold_mean=np.mean(res[name]['bacc_folds']),
                         fold_sd=np.std(res[name]['bacc_folds'], ddof=1),
                         **{f'recall_{s}': v for s, v in rec.items()}))
        cache.write_text(json.dumps(dict(res=res[name], row={k: (float(v) if isinstance(v, (float, np.floating)) else v)
                                                              for k, v in rows[-1].items()})))
        logf(f'  pooled balanced accuracy {res[name]["bacc"]:.3f}; recall ' +
             ', '.join(f'{s} {v:.2f}' for s, v in rec.items()))
    tab = pd.DataFrame(rows)
    tab = pd.concat([tab, pd.DataFrame([dict(row='chance (analytic)', inputs='-', cfg='-', bacc=1 / len(CLASSES))])])
    res['chance_analytic'] = 1 / len(CLASSES)
    res['design'] = dict(bin_s=BIN_S, n_min=N_MIN, k_pc=K_PC, c_grid=C_GRID, n_fold=N_FOLD, seed=SEED,
                         n_brain_inputs=int(len(inp_idx)), n_mask=int(len(mask)), word_vocab=vocab,
                         class_counts_bins={st.STATES[c]: int((y == c).sum()) for c in CLASSES})
    (OUT / 'results.json').write_text(json.dumps(res, indent=1))
    fmt = tab.copy()
    for c in fmt.columns:
        if fmt[c].dtype.kind == 'f':
            fmt[c] = fmt[c].map(lambda x: '' if pd.isna(x) else f'{x:.3f}')
    (OUT / 'summary.md').write_text(
        '# W5 T7 hidden-state baselines (balanced accuracy, 4 states, 50 ms bins, 5-fold CV by episode)\n\n'
        + fmt.to_markdown(index=False) + '\n')
    print(fmt.to_string(index=False))


if __name__ == '__main__':
    main()
