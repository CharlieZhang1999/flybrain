#!/usr/bin/env python
"""W5 T10: which neurons does the decoder read? Attribution by hop distance from the driven
cells, real wiring (calib) vs the degree-preserving shuffle, on the v1 hidden-state dataset
(PR #19), the same 5 outer folds as PR #19 / #20.

Stages (each caches to disk; a re-run skips what is done)
  hops   For cfg in (calib, shuffle): W = cns_model('cns', cfg, **W4_CALIB)['W'] (the W that
         produced the data). W is post x pre: lif_torch.run_trial adds column j of W to every
         row when neuron j spikes (Wd.colsum(pre)), and shuffle_W permutes the column index of
         the COO, so the shuffle keeps each row's (postsynaptic) entry count and redraws the
         presynaptic partners. Edges pre -> post = (col -> row) of the nonzeros of W, chemical
         synapses only (gap junctions and the sensory presyn rows are not part of W and are
         ignored). BFS from the driven set of each state (state_sets.json body_ids: hunger ISN,
         thirst BiT, arousal pC1/P1) to every neuron, two edge sets: 'any' (every nonzero) and
         'strong' (|w| >= the 90th percentile of |w| over the nonzeros of that W). Hop 0 = the
         driven cells (masked from the decoder input). -> hops/hops_<cfg>.npz,
         hop_histogram.csv, spotcheck.csv (a shortest path for 5 input neurons per cfg, every
         edge re-read from W, and the minimum hop over each neuron's presynaptic partners).
  train  Attention decoder (POYO / Perceiver style), per cfg and outer fold. Inputs: the
         decoder input neurons of baseline.brain_inputs (brain_mask minus decoder mask, active
         in that cfg) that have >= N_MIN spikes in the training episodes of the fold;
         x = sqrt(spike count per 50 ms bin). Token of neuron n in bin t = E[n] + phi(x[t, n]),
         E a learned neuron embedding, phi a piecewise-linear count embedding (learned vectors
         at x = 0, 1, ..., X_MAX, linear interpolation; differentiable in x). N_LAT learned
         latent queries cross-attend over the neuron tokens of each bin (N_HEAD heads, softmax
         over neurons), the N_LAT x d latent outputs are flattened and projected to d, a
         bidirectional GRU runs over the 80 bins, a linear layer gives the per-bin state.
         Training: the fold's training episodes are split by the PR #20 validation split
         (folds(tr, 4, SEED + 10 + f)[0]); AdamW, early stopping on the validation balanced
         accuracy, the best-epoch weights are kept (no refit). Test episodes are used only for
         the held-out predictions and the attributions below. -> train/<cfg>_f<k>.pt, .json
  attr   On the held-out episodes of each fold, with that fold's model:
         a) attention: cross-attention weight per neuron, mean over heads and latents, mean over
            test bins of each true state minus the mean over baseline bins; mass = |difference|.
         b) gradient x input: d(z_s - z_baseline)/dx * x per neuron (z = logits of the bin),
            mean over test bins of true state s; mass = |mean|.
         c) causal: x of all input neurons in hop group k (of state s, 'any' edges) set to
            their training mean; drop in the recall of state s and in overall balanced
            accuracy vs the intact model; mass = max(drop in recall of s, 0).
         -> attr/<cfg>_f<k>.npz
  report Heatmaps (rows hop 1, 2, 3, 4, >=5 [, unreachable]; columns hunger, thirst, arousal),
         raw share of mass and per-neuron mass (share / group size, renormalised), calib,
         shuffle and calib - shuffle panels, per method; H1-H3; CSVs and PNGs.

Usage
  python scripts/w5_hop_heatmap.py hops
  python scripts/w5_hop_heatmap.py train [--cfgs calib,shuffle] [--folds 0,1,2,3,4]
  python scripts/w5_hop_heatmap.py attr
  python scripts/w5_hop_heatmap.py report
Output: results/w5/hidden_state/hop_heatmap/
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

OUT = bl.HS / 'hop_heatmap'
CACHE = Path('/home/user/flybrain_cache/t10')     # local, rebuildable (W, X): not results
DRIVEN = ('hunger', 'thirst', 'arousal')
STATES = ('baseline', 'hunger', 'thirst', 'arousal')
EDGES = ('any', 'strong')
STRONG_Q = 90.0
HOP_ROWS = ('1', '2', '3', '4', '>=5', 'unreachable')
UNREACH = 10 ** 6
# attention decoder
D, N_LAT, N_HEAD, X_MAX = 32, 16, 1, 8
LR, WD, BATCH, MAX_EP, PATIENCE = 3e-3, 1e-2, 8, 150, 20
DROP, TOK_KEEP, TOK_DROP, LOGIT_SCALE = 0.3, 1.0, 0.2, 4.0   # TOK_KEEP < 1: random token subset per batch (off)


def logf(m):
    print(m, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'log.txt', 'a') as fh:
        fh.write(m + '\n')


def hop_group(h):
    """hop (int array) -> row index into HOP_ROWS (-1 for hop 0 = driven)."""
    g = np.where(h >= UNREACH, 5, np.minimum(h, 5) - 1)
    return np.where(h == 0, -1, g)


# ---------------------------------------------------------------- inputs
def setup():
    import pandas as pd
    from sim.cns import brain_mask
    meta = pd.read_parquet(ROOT / 'data' / 'malecns' / 'malecns_traced_meta.parquet')
    sets = json.loads((bl.HS / 'state_sets.json').read_text())
    mask = np.array(sets['decoder_mask']['neuron_idx'], np.int64)
    bm = brain_mask(meta).copy()
    bm[mask] = False
    pos = {int(b): i for i, b in enumerate(meta.bodyId)}
    driven = {s: np.array(sorted(pos[b] for b in sets['body_ids'][s]), np.int64) for s in DRIVEN}
    return meta, mask, np.flatnonzero(bm), driven


def active_idx(cfg, inp_idx, mask, n):
    """Model-neuron index of each column of bl.brain_inputs(cfg, ...) (same rule: inputs that
    spike at least once in this cfg)."""
    col = -np.ones(n, np.int64)
    col[inp_idx] = np.arange(len(inp_idx))
    seen = np.zeros(len(inp_idx), bool)
    for k in range(bl.N_EP):
        with np.load(bl.DS / 'activity' / f'ep{k:03d}_{cfg}.npz') as d:
            c = col[d['i'].astype(np.int64)]
        seen[c[c >= 0]] = True
    return inp_idx[seen]


def load_X(cfg, inp_idx, mask, n):
    """(X, idx): counts (N_EP, NB, n_active) and the model index of each column; cached."""
    cp = CACHE / f'X_{cfg}.npz'
    if cp.exists():
        with np.load(cp) as d:
            return d['X'], d['idx']
    X = bl.brain_inputs(cfg, inp_idx, mask, n)
    idx = active_idx(cfg, inp_idx, mask, n)
    if len(idx) != X.shape[2]:
        raise RuntimeError('active index mismatch')
    cp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cp, X=X.astype(np.uint8) if X.max() < 256 else X, idx=idx)
    return X, idx


# ---------------------------------------------------------------- stage: hops
def bfs(indptr, indices, src, n):
    """Multi-source BFS on a CSR adjacency (row = pre, indices = post). Returns (hop, parent)."""
    hop = np.full(n, UNREACH, np.int64)
    par = -np.ones(n, np.int64)
    hop[src] = 0
    front, h = np.asarray(src, np.int64), 0
    while len(front):
        h += 1
        starts, ends = indptr[front], indptr[front + 1]
        lens = ends - starts
        rep = np.repeat(front, lens)
        nb = indices[np.concatenate([np.arange(a, b) for a, b in zip(starts, ends)])] if lens.sum() else np.zeros(0, np.int64)
        new = hop[nb] == UNREACH
        nb, rep = nb[new], rep[new]
        u, first = np.unique(nb, return_index=True)
        hop[u] = h
        par[u] = rep[first]
        front = u
    return hop, par


def stage_hops():
    from scipy import sparse
    from sim.cns import cns_model, W4_CALIB
    meta, mask, inp_idx, driven = setup()
    n = len(meta)
    (OUT / 'hops').mkdir(parents=True, exist_ok=True)
    rows, spot = [], []
    for cfg in bl.CFGS:
        cp = OUT / 'hops' / f'hops_{cfg}.npz'
        if not cp.exists():
            t0 = time.time()
            W = cns_model('cns', cfg, **W4_CALIB)['W'].tocsc()
            logf(f'[hops {cfg}] W built ({time.time() - t0:.0f} s): {W.shape}, nnz {W.nnz:,}, '
                 f'shuffle_order {"fixed (default)" if cfg == "shuffle" else "n/a"}')
            a = np.abs(W.data)
            thr = float(np.percentile(a, STRONG_Q))
            out = dict(thr=thr, nnz=W.nnz)
            for e in EDGES:
                Wc = W.copy()
                if e == 'strong':
                    Wc.data[a < thr] = 0
                    Wc.eliminate_zeros()
                # CSC of W (post x pre): column j lists the posts of pre j -> CSR adjacency pre->post
                A = Wc.tocsc()
                A.sort_indices()
                out[f'n_edges_{e}'] = A.nnz
                for s in DRIVEN:
                    hop, par = bfs(A.indptr.astype(np.int64), A.indices.astype(np.int64), driven[s], n)
                    out[f'hop_{e}_{s}'] = hop
                    out[f'par_{e}_{s}'] = par
            sparse.save_npz(CACHE / f'W_{cfg}.npz', W)
            np.savez_compressed(cp, **out)
            logf(f'[hops {cfg}] strong threshold |w| >= {thr:.4g}; edges any {out["n_edges_any"]:,}, '
                 f'strong {out["n_edges_strong"]:,} ({time.time() - t0:.0f} s)')
        d = dict(np.load(cp))
        idx = active_idx(cfg, inp_idx, mask, n)
        for e in EDGES:
            for s in DRIVEN:
                g = hop_group(d[f'hop_{e}_{s}'][idx])
                cnt = np.bincount(g[g >= 0], minlength=6)
                rows.append(dict(cfg=cfg, edges=e, state=s, n_inputs=len(idx), n_driven=len(driven[s]),
                                 **{f'hop {r}': int(c) for r, c in zip(HOP_ROWS, cnt)},
                                 max_hop=int(d[f'hop_{e}_{s}'][idx][d[f'hop_{e}_{s}'][idx] < UNREACH].max())))
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'hop_histogram.csv', index=False)
    logf(df.to_string(index=False))
    spotcheck(meta, mask, inp_idx, driven)


def spotcheck(meta, mask, inp_idx, driven):
    """5 input neurons per cfg (any edges, hunger + arousal mix, different hops): a shortest
    path re-read from W (every edge must be a nonzero W[post, pre]) and the minimum hop of the
    neuron's presynaptic partners (must be hop - 1)."""
    from scipy import sparse
    import pandas as pd
    rng = np.random.default_rng(0)
    rows = []
    for cfg in bl.CFGS:
        d = dict(np.load(OUT / 'hops' / f'hops_{cfg}.npz'))
        if not (CACHE / f'W_{cfg}.npz').exists():
            from sim.cns import cns_model, W4_CALIB
            sparse.save_npz(CACHE / f'W_{cfg}.npz', cns_model('cns', cfg, **W4_CALIB)['W'].tocsc())
        W = sparse.load_npz(CACHE / f'W_{cfg}.npz').tocsr()   # row = post
        idx = active_idx(cfg, inp_idx, mask, len(meta))
        for j, s in enumerate(('hunger', 'thirst', 'arousal', 'hunger', 'thirst')):
            hop = d[f'hop_any_{s}']
            par = d[f'par_any_{s}']
            want = (2, 3, 1, 4, 2)[j]
            cand = idx[hop[idx] == want]
            if not len(cand):
                cand = idx[hop[idx] < UNREACH]
            v = int(rng.choice(cand))
            path = [v]
            while hop[path[-1]] > 0:
                path.append(int(par[path[-1]]))
            path = path[::-1]
            ok = all(W[b, a] != 0 for a, b in zip(path[:-1], path[1:]))
            pres = W[v].indices
            min_pre = int(hop[pres].min()) if len(pres) else -1
            rows.append(dict(cfg=cfg, state=s, neuron_idx=v, bodyId=int(meta.bodyId.iloc[v]),
                             type=str(meta.type.iloc[v]), hop=int(hop[v]),
                             path_bodyIds='>'.join(str(int(meta.bodyId.iloc[p])) for p in path),
                             path_edges_in_W=ok, min_hop_of_presyn_partners=min_pre,
                             source_in_driven=bool(path[0] in set(driven[s].tolist())),
                             check_pass=bool(ok and min_pre == hop[v] - 1 and path[0] in set(driven[s].tolist()))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'spotcheck.csv', index=False)
    logf('spot check:\n' + df.drop(columns='path_bodyIds').to_string(index=False))


# ---------------------------------------------------------------- attention decoder (torch)
def build_net(n_in, seed):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    dh = D // N_HEAD

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.E = nn.Parameter(torch.randn(n_in, D))          # neuron embedding
            self.C = nn.Parameter(torch.randn(X_MAX + 1, D))     # count embedding at x = 0..X_MAX
            self.Q = nn.Parameter(torch.randn(N_HEAD, N_LAT, dh))
            self.Wk = nn.Linear(D, D, bias=False)
            self.Wv = nn.Linear(D, D, bias=False)
            # learned logit scale per latent, initialised at LOGIT_SCALE: with near-uniform
            # attention over ~7k tokens the latent outputs barely change from bin to bin and
            # training stalls at chance; a sharper start makes each latent read ~tens of neurons
            self.ls = nn.Parameter(torch.full((N_HEAD, N_LAT, 1), float(LOGIT_SCALE)))
            self.mix = nn.Sequential(nn.LayerNorm(N_LAT * D), nn.Linear(N_LAT * D, D), nn.ReLU(), nn.Dropout(DROP))
            self.rnn = nn.GRU(D, D, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(DROP)
            self.out = nn.Linear(2 * D, 4)

        def attend(self, x, ids=None, drop_mask=None):
            """x (B, T, N) sqrt counts -> latents (B, T, N_LAT * D), attention (B, T, H, L, N).
            Token = E[n] + phi(x): keys / values are linear in the token, so the logits and
            values split into a neuron part and an interpolated count part (no B x T x N x D
            tensor is built)."""
            B, T, N = x.shape
            xi = x.clamp(0, X_MAX - 1e-4)
            lo = xi.detach().floor().long()
            fr = (xi - lo).unsqueeze(2).unsqueeze(2)                     # (B,T,1,1,N)
            E = self.E if ids is None else self.E[ids]
            KE = self.Wk(E).view(N, N_HEAD, dh).transpose(0, 1)          # (H,N,dh)
            KC = self.Wk(self.C).view(-1, N_HEAD, dh).transpose(0, 1)    # (H,X+1,dh)
            VE = self.Wv(E).view(N, N_HEAD, dh).transpose(0, 1)
            VC = self.Wv(self.C).view(-1, N_HEAD, dh).transpose(0, 1)
            SE = self.ls * torch.einsum('hld,hnd->hln', self.Q, KE) / dh ** 0.5   # (H,L,N)
            SC = self.ls * torch.einsum('hld,hkd->hlk', self.Q, KC) / dh ** 0.5   # (H,L,X+1)
            lo_ = lo.unsqueeze(2).unsqueeze(2).expand(B, T, N_HEAD, N_LAT, N)
            SCb = SC.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)
            s_lo = torch.gather(SCb, 4, lo_)
            s_hi = torch.gather(SCb, 4, lo_ + 1)
            logit = SE + (1 - fr) * s_lo + fr * s_hi                     # (B,T,H,L,N)
            if drop_mask is not None:                                    # token dropout (training)
                logit = logit.masked_fill(drop_mask[:, :, None, None, :], float('-inf'))
            a = torch.softmax(logit, -1)
            vE = torch.einsum('bthln,hnd->bthld', a, VE)
            M = torch.zeros(B, T, N_HEAD, N_LAT, X_MAX + 1, dtype=a.dtype)
            M = M.scatter_add(4, lo_, a * (1 - fr)).scatter_add(4, lo_ + 1, a * fr)
            vC = torch.einsum('bthlk,hkd->bthld', M, VC)
            z = (vE + vC).permute(0, 1, 3, 2, 4).reshape(B, T, N_LAT * D)
            return z, a

        def forward(self, x, ids=None, return_attn=False, drop_mask=None):
            z, a = self.attend(x, ids, drop_mask)
            h = self.rnn(self.mix(z))[0]
            out = self.out(self.drop(h))
            return (out, a) if return_attn else out

    return Net()


def fold_split(f):
    te = bl.folds(np.arange(bl.N_EP), bl.N_FOLD, bl.SEED)[f]
    tr = np.setdiff1d(np.arange(bl.N_EP), te)
    va = bl.folds(tr, bl.N_INNER, bl.SEED + 10 + f)[0]
    return tr, np.setdiff1d(tr, va), va, te


def fold_inputs(X, f):
    """Neurons with >= N_MIN spikes in the fold's training episodes (as PR #19 / #20);
    x = sqrt counts, float32. Returns keep (column index into X), x (N_EP, NB, n_keep)."""
    tr = fold_split(f)[0]
    keep = np.flatnonzero(X[tr].sum((0, 1), dtype=np.int64) >= bl.N_MIN)
    return keep, np.sqrt(X[:, :, keep].astype(np.float32))


def predict(net, x, bs=6):
    import torch
    net.eval()
    with torch.no_grad():
        return np.concatenate([torch.softmax(net(torch.from_numpy(x[i:i + bs])), -1).numpy()
                               for i in range(0, len(x), bs)])


def stage_train(cfgs, fold_list):
    import torch
    meta, mask, inp_idx, driven = setup()
    y = bl.labels().astype(np.int64)
    (OUT / 'train').mkdir(parents=True, exist_ok=True)
    for cfg in cfgs:
        X, idx = load_X(cfg, inp_idx, mask, len(meta))
        for f in fold_list:
            cp = OUT / 'train' / f'{cfg}_f{f}.json'
            if cp.exists():
                logf(f'[train {cfg} f{f}] cached: test bacc {json.loads(cp.read_text())["test_bacc"]:.3f}')
                continue
            t0 = time.time()
            tr, itr, iva, te = fold_split(f)
            keep, x = fold_inputs(X, f)
            seed = bl.SEED + 1000 + f
            net = build_net(len(keep), seed)
            rng = np.random.default_rng(seed)
            opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
            lossf = torch.nn.CrossEntropyLoss()
            Xt, Yt = torch.from_numpy(x[itr]), torch.from_numpy(y[itr])
            best, best_ep, bad, state, hist = -1.0, 0, 0, None, []
            for ep in range(1, MAX_EP + 1):
                net.train()
                tl = 0.0
                for b in np.array_split(rng.permutation(len(itr)), max(1, len(itr) // BATCH)):
                    opt.zero_grad()
                    if TOK_KEEP < 1:
                        ids = torch.from_numpy(np.sort(rng.permutation(len(keep))[:int(TOK_KEEP * len(keep))]))
                        o = net(Xt[b][:, :, ids], ids)
                    else:
                        o = net(Xt[b], drop_mask=torch.rand(len(b), bl.NB, len(keep)) < TOK_DROP)
                    loss = lossf(o.reshape(-1, 4), Yt[b].reshape(-1))
                    loss.backward()
                    opt.step()
                    tl += float(loss.detach()) * len(b)
                ba = bl.bacc(y[iva].ravel(), predict(net, x[iva]).argmax(-1).ravel())
                btr = bl.bacc(y[itr[:16]].ravel(), predict(net, x[itr[:16]]).argmax(-1).ravel())
                hist.append((ep, tl / len(itr), ba))
                if ba > best + 1e-4:
                    best, best_ep, bad = ba, ep, 0
                    state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                if ep % 5 == 0 or bad == 0:
                    logf(f'  [{cfg} f{f}] ep {ep}: loss {tl / len(itr):.3f}, val bacc {ba:.3f} '
                         f'(train-eval {btr:.3f}; best {best:.3f} @ {best_ep}, {time.time() - t0:.0f} s)')
                if bad >= PATIENCE:
                    break
            net.load_state_dict(state)
            prob = predict(net, x[te])
            tb = bl.bacc(y[te].ravel(), prob.argmax(-1).ravel())
            torch.save(state, OUT / 'train' / f'{cfg}_f{f}.pt')
            np.savez_compressed(OUT / 'train' / f'{cfg}_f{f}_pred.npz', prob=prob, te=te, keep_idx=idx[keep])
            info = dict(cfg=cfg, fold=f, test_bacc=tb, val_bacc=best, best_epoch=best_ep, n_inputs=len(keep),
                        te=te.tolist(), train=itr.tolist(), val=iva.tolist(), wall_s=time.time() - t0,
                        hp=dict(D=D, N_LAT=N_LAT, N_HEAD=N_HEAD, X_MAX=X_MAX, LR=LR, WD=WD, BATCH=BATCH,
                                DROP=DROP, TOK_KEEP=TOK_KEEP, TOK_DROP=TOK_DROP, LOGIT_SCALE=LOGIT_SCALE, PATIENCE=PATIENCE), history=hist)
            cp.write_text(json.dumps(info, default=float))
            logf(f'[train {cfg} f{f}] test bacc {tb:.3f} (val {best:.3f} @ ep {best_ep}, '
                 f'{len(keep)} inputs, {time.time() - t0:.0f} s)')



# ---------------------------------------------------------------- stage: attr
def stage_attr(cfgs, fold_list):
    import torch
    meta, mask, inp_idx, driven = setup()
    y = bl.labels().astype(np.int64)
    (OUT / 'attr').mkdir(parents=True, exist_ok=True)
    for cfg in cfgs:
        X, idx = load_X(cfg, inp_idx, mask, len(meta))
        hops = dict(np.load(OUT / 'hops' / f'hops_{cfg}.npz'))
        for f in fold_list:
            cp = OUT / 'attr' / f'{cfg}_f{f}.npz'
            if cp.exists():
                logf(f'[attr {cfg} f{f}] cached')
                continue
            t0 = time.time()
            tr, itr, iva, te = fold_split(f)
            keep, x = fold_inputs(X, f)
            kidx = idx[keep]
            net = build_net(len(keep), 0)
            net.load_state_dict(torch.load(OUT / 'train' / f'{cfg}_f{f}.pt'))
            net.eval()
            xte, yte = x[te], y[te]
            att = np.zeros((4, len(keep)))
            gi = np.zeros((3, len(keep)))
            nb = np.array([(yte == c).sum() for c in range(4)])
            for i in range(0, len(te), 4):
                xb = torch.from_numpy(xte[i:i + 4]).requires_grad_(True)
                yb = yte[i:i + 4]
                out, a = net(xb, return_attn=True)
                am = a.detach().mean((2, 3)).numpy()                     # (b, T, N)
                for c in range(4):
                    att[c] += am[yb == c].sum(0)
                for j, s in enumerate(DRIVEN):
                    c = STATES.index(s)
                    sel = torch.from_numpy(yb == c)
                    if not sel.any():
                        continue
                    tgt = (out[..., c] - out[..., 0])[sel].sum()
                    g, = torch.autograd.grad(tgt, xb, retain_graph=True)
                    gi[j] += (g * xb).detach().sum((0, 1)).numpy()
            att /= nb[:, None]
            gi /= nb[1:, None]
            att_diff = att[1:] - att[:1]
            # c) causal: mean-impute hop groups
            prob0 = predict(net, xte)
            p0 = prob0.argmax(-1)
            base_b = bl.bacc(yte.ravel(), p0.ravel())
            rec0 = np.array([(p0[yte == c] == c).mean() for c in range(4)])
            mu = x[tr].mean((0, 1))
            caus_rec = np.full((2, 3, 6), np.nan)                       # edges x state x hop row
            caus_bacc = np.full((2, 3, 6), np.nan)
            caus_n = np.zeros((2, 3, 6), np.int64)
            for ei, e in enumerate(EDGES):
                for j, s in enumerate(DRIVEN):
                    g = hop_group(hops[f'hop_{e}_{s}'][kidx])
                    c = STATES.index(s)
                    for r in range(6):
                        cols = np.flatnonzero(g == r)
                        caus_n[ei, j, r] = len(cols)
                        if not len(cols):
                            continue
                        xa = xte.copy()
                        xa[:, :, cols] = mu[cols]
                        pa = predict(net, xa).argmax(-1)
                        caus_rec[ei, j, r] = rec0[c] - (pa[yte == c] == c).mean()
                        caus_bacc[ei, j, r] = base_b - bl.bacc(yte.ravel(), pa.ravel())
            np.savez_compressed(cp, keep_idx=kidx, att=att, att_diff=att_diff, gi=gi, caus_rec=caus_rec,
                                caus_bacc=caus_bacc, caus_n=caus_n, base_bacc=base_b, rec0=rec0, te=te)
            logf(f'[attr {cfg} f{f}] done ({time.time() - t0:.0f} s); intact bacc {base_b:.3f}; '
                 f'causal any-edge drop in recall (rows hop 1..>=5,unr): '
                 + '; '.join(f'{s} ' + ' '.join(f'{v:+.3f}' for v in caus_rec[0, j]) for j, s in enumerate(DRIVEN)))



# ---------------------------------------------------------------- stage: report
METHODS = ('attention', 'grad_x_input', 'causal')


def fold_masses(cfg, f, edges):
    """-> dict method -> (raw (3 states, 6 rows), n per group (3, 6)) for one fold.
    raw = attribution mass summed over the group's neurons (causal: the group's drop)."""
    d = dict(np.load(OUT / 'attr' / f'{cfg}_f{f}.npz'))
    hops = np.load(OUT / 'hops' / f'hops_{cfg}.npz')
    ei = EDGES.index(edges)
    out = {}
    n = np.zeros((3, 6))
    raw = {m: np.zeros((3, 6)) for m in METHODS}
    for j, s in enumerate(DRIVEN):
        g = hop_group(hops[f'hop_{edges}_{s}'][d['keep_idx']])
        n[j] = np.bincount(g[g >= 0], minlength=6)
        if (g < 0).any():
            raise RuntimeError('a driven neuron among the decoder inputs')
        raw['attention'][j] = np.bincount(g, np.abs(d['att_diff'][j]), minlength=6)
        raw['grad_x_input'][j] = np.bincount(g, np.abs(d['gi'][j]), minlength=6)
        raw['causal'][j] = np.nan_to_num(np.clip(d['caus_rec'][ei, j], 0, None))
    for m in METHODS:
        out[m] = (raw[m], n)
    return out


def shares(raw, n, per_neuron):
    v = raw / np.maximum(n, 1) if per_neuron else raw.copy()
    v[n == 0] = 0
    tot = v.sum(1, keepdims=True)
    return np.divide(v, tot, out=np.full_like(v, np.nan), where=tot > 0)


def stage_report():
    import pandas as pd
    from scipy import stats
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figd = OUT / 'figures'
    figd.mkdir(parents=True, exist_ok=True)
    folds = range(bl.N_FOLD)
    rows, maps, hyp = [], {}, {}
    for edges in EDGES:
        for cfg in bl.CFGS:
            fm = [fold_masses(cfg, f, edges) for f in folds]
            for m in METHODS:
                for norm in ('raw_share', 'per_neuron'):
                    S = np.stack([shares(*fm[f][m], norm == 'per_neuron') for f in folds])   # (5, 3, 6)
                    maps[(edges, cfg, m, norm)] = S
                    mean = np.nanmean(S, 0)
                    for j, s in enumerate(DRIVEN):
                        for r, hr in enumerate(HOP_ROWS):
                            rows.append(dict(edges=edges, cfg=cfg, method=m, norm=norm, state=s, hop=hr,
                                             value=mean[j, r], sd_folds=np.nanstd(S[:, j, r], ddof=1),
                                             n_neurons=int(fm[0][m][1][j, r]),
                                             raw_mass=float(np.mean([fm[f][m][0][j, r] for f in folds]))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'heatmap_values.csv', index=False)
    # heatmaps: per edge set x norm x method: calib | shuffle | calib - shuffle
    has_unr = df[(df.hop == 'unreachable') & (df.n_neurons > 0)].shape[0] > 0
    nrow = 6 if has_unr else 5
    pngs = []
    for edges in EDGES:
        for norm in ('raw_share', 'per_neuron'):
            fig, ax = plt.subplots(3, 3, figsize=(10, 10.5), constrained_layout=True)
            for i, m in enumerate(METHODS):
                c = np.nanmean(maps[(edges, 'calib', m, norm)], 0).T[:nrow]
                sh = np.nanmean(maps[(edges, 'shuffle', m, norm)], 0).T[:nrow]
                for k, (M, title) in enumerate(((c, 'calib'), (sh, 'shuffle'), (c - sh, 'calib - shuffle'))):
                    a = ax[i, k]
                    if k < 2:
                        im = a.imshow(M, cmap='viridis', vmin=0, vmax=max(np.nanmax(c), np.nanmax(sh)))
                    else:
                        lim = np.nanmax(np.abs(M)) or 1
                        im = a.imshow(M, cmap='RdBu_r', vmin=-lim, vmax=lim)
                    for (r, q), v in np.ndenumerate(M):
                        a.text(q, r, f'{v:.2f}', ha='center', va='center', fontsize=8,
                               color='white' if k < 2 and v < 0.6 * im.norm.vmax else 'black')
                    a.set_xticks(range(3), DRIVEN)
                    a.set_yticks(range(nrow), [f'hop {h}' for h in HOP_ROWS[:nrow]])
                    a.set_title(f'{m}: {title}', fontsize=10)
                    fig.colorbar(im, ax=a, shrink=0.8)
            fig.suptitle(f'Attribution mass by hop from the driven cells ({norm.replace("_", " ")}, '
                         f'{edges} edges; mean of 5 folds)')
            p = figd / f'heatmap_{edges}_{norm}.png'
            fig.savefig(p, dpi=110)
            plt.close(fig)
            pngs.append(str(p))
    # hypotheses (primary: any edges)
    lines = []
    for edges in EDGES:
        for norm in ('raw_share', 'per_neuron'):
            for m in METHODS:
                Sc, Ss = maps[(edges, 'calib', m, norm)], maps[(edges, 'shuffle', m, norm)]
                near_s = np.nanmean(Ss[:, :, :2].sum(2), 1)      # per fold, mean over states
                near_c = np.nanmean(Sc[:, :, :2].sum(2), 1)
                deep_c = np.nanmean(Sc[:, :, 2:5].sum(2), 1)
                deep_s = np.nanmean(Ss[:, :, 2:5].sum(2), 1)
                t = stats.ttest_rel(deep_c, deep_s, alternative='greater')
                w = stats.wilcoxon(deep_c, deep_s, alternative='greater') if np.any(deep_c != deep_s) else None
                hyp[(edges, norm, m)] = dict(near12_shuffle=near_s.mean(), near12_calib=near_c.mean(),
                                             deep3_calib=deep_c.mean(), deep3_shuffle=deep_s.mean(),
                                             deep_diff_folds=(deep_c - deep_s).tolist(), t_p=t.pvalue,
                                             wilcoxon_p=None if w is None else w.pvalue,
                                             per_state_near12_shuffle=np.nanmean(Ss[:, :, :2].sum(2), 0).tolist(),
                                             per_state_deep3_calib=np.nanmean(Sc[:, :, 2:5].sum(2), 0).tolist(),
                                             per_state_deep3_shuffle=np.nanmean(Ss[:, :, 2:5].sum(2), 0).tolist())
                lines.append(dict(edges=edges, norm=norm, method=m, **{k: v for k, v in hyp[(edges, norm, m)].items()
                                                                     if not k.startswith('per_state') and k != 'deep_diff_folds'}))
    hdf = pd.DataFrame(lines)
    hdf.to_csv(OUT / 'hypotheses_h1_h2.csv', index=False)
    # H3: Spearman of hop-group vectors between methods (mean-of-folds maps), per cfg x state
    sp = []
    for edges in EDGES:
        for norm in ('raw_share', 'per_neuron'):
            for cfg in bl.CFGS:
                M = {m: np.nanmean(maps[(edges, cfg, m, norm)], 0) for m in METHODS}
                for j, s in enumerate(DRIVEN):
                    ok = M['attention'][j] == M['attention'][j]
                    nz = np.array([df[(df.edges == edges) & (df.cfg == cfg) & (df.method == 'attention')
                                      & (df.norm == norm) & (df.state == s) & (df.hop == h)].n_neurons.iloc[0] > 0
                                   for h in HOP_ROWS])
                    for a_, b_ in (('attention', 'grad_x_input'), ('attention', 'causal'), ('grad_x_input', 'causal')):
                        r = stats.spearmanr(M[a_][j][nz & ok], M[b_][j][nz & ok]).statistic
                        sp.append(dict(edges=edges, norm=norm, cfg=cfg, state=s, pair=f'{a_} vs {b_}', rho=r,
                                       n_groups=int((nz & ok).sum())))
    spdf = pd.DataFrame(sp)
    spdf.to_csv(OUT / 'hypothesis_h3_spearman.csv', index=False)
    (OUT / 'hypotheses.json').write_text(json.dumps({'|'.join(k): v for k, v in hyp.items()}, indent=1, default=float))
    # accuracy table vs PR #20
    acc = []
    ce = json.loads((bl.HS / 'ceiling' / 'results.json').read_text())
    for cfg in bl.CFGS:
        info = [json.loads((OUT / 'train' / f'{cfg}_f{f}.json').read_text()) for f in folds]
        prob = np.zeros((bl.N_EP, bl.NB, 4))
        for f in folds:
            with np.load(OUT / 'train' / f'{cfg}_f{f}_pred.npz') as d:
                prob[d['te']] = d['prob']
        y = bl.labels().astype(np.int64)
        pooled = bl.bacc(y.ravel(), prob.argmax(-1).ravel())
        best = max((k for k in ce if k.endswith('_' + cfg)), key=lambda k: ce[k]['bacc'])
        acc.append(dict(cfg=cfg, attention_bacc_pooled=pooled, attention_folds=[i['test_bacc'] for i in info],
                        best_pr20=best.rsplit('_', 1)[0], best_pr20_bacc=ce[best]['bacc'],
                        best_pr20_folds=ce[best]['bacc_folds'], gap=pooled - ce[best]['bacc'],
                        within_005=bool(pooled >= ce[best]['bacc'] - 0.05),
                        train_min=sum(i['wall_s'] for i in info) / 60))
    (OUT / 'accuracy.json').write_text(json.dumps(acc, indent=1, default=float))
    with pd.option_context('display.width', 250, 'display.max_columns', 30, 'display.float_format', '{:.3f}'.format):
        logf('accuracy:\n' + pd.DataFrame(acc).to_string(index=False))
        logf('H1/H2:\n' + hdf.to_string(index=False))
        logf('H3 (Spearman):\n' + spdf.groupby(['edges', 'norm', 'pair']).rho.describe()[['mean', 'min', '50%', 'max']].to_string())
    logf('figures: ' + ', '.join(pngs))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('hops', 'train', 'attr', 'report'))
    ap.add_argument('--cfgs', default=','.join(bl.CFGS))
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    if a.stage == 'hops':
        CACHE.mkdir(parents=True, exist_ok=True)
        stage_hops()
    elif a.stage == 'train':
        import torch
        torch.set_num_threads(a.threads)
        stage_train(a.cfgs.split(','), [int(v) for v in a.folds.split(',')])
    elif a.stage == 'attr':
        import torch
        torch.set_num_threads(a.threads)
        stage_attr(a.cfgs.split(','), [int(v) for v in a.folds.split(',')])
    else:
        stage_report()
