#!/usr/bin/env python
"""W4D6: first ridge baseline, brain activity -> VNC motor-pool rates 100 ms ahead.

Purpose
  Fit a linear (ridge) map from lagged brain-input rates to the 349 motor-pool rates
  (sim.cns.motor_pools) and the 42 coarse group x side rates, 100 ms ahead, on the Day 5
  regression set (results/w4/regression/day5/ep{000..199}_{calib,shuffle}.npz, the 200
  mixture episodes). Real (calib) and shuffled connectome are fitted separately, each with
  its own inputs and targets. Reported as R^2 per pool on held-out episodes.

Design (all choices fixed here, see the module constants)
  bins      10 ms (100 steps of dt 0.1 ms), 250 per 2.5 s episode; the first 100 ms
            (bins 0-9) are discarded as input and as target.
  samples   current bin t = 20..239 (220 per episode): inputs at bins t - l for
            l in LAGS (0, 1, ..., 10 bins = 0-100 ms, all 11 lags), target at bin t + 10.
  inputs    'all': brain_mask neurons (brain, non-sensory, non-motor) with >= N_MIN spikes
            in the training episodes of the fold (bins 10-249), z-scored on training bins,
            then projected on K_PC principal components fitted on the training bins of the
            fold only (torch.pca_lowrank, fixed seed); the K_PC scores at the 11 lags are the
            features. 'dn': the same, restricted to descending neurons (superclass contains
            'descending', inside brain_mask: superclass descending_neuron or
            efferent_descending; sensory_descending is excluded by brain_mask).
  targets   pool rate (spikes per neuron per s) at t + 100 ms, fine (349) and coarse
            (group | side, 42), the same pools and binning as scripts/w4d5_dataset_summary.py.
  fit       ridge in closed form (eigendecomposition of X'X), intercept by centring on the
            training fold; features standardised on the training fold. alpha is chosen per
            target by inner 4-fold CV by episode on the training episodes only (grid ALPHAS,
            pooled inner held-out SSE).
  folds     5-fold CV by episode over the 200 mixtures, stratified by (looming, wind): the
            episodes of each stratum are permuted (seed SEED) and dealt round-robin.
  controls  'time': all-brain inputs, but each episode's target series (the 220 target bins)
            is circularly shifted by a random offset of 30..190 bins (>= 300 ms either way
            round, seed SEED + 1), in training and test alike. 'mean': predict the training-
            fold mean.
  metrics   R^2 per target on the held-out bins pooled over the 5 folds (each episode is
            held out once): 1 - SSE / sum (y - mean_heldout)^2. Also on the held-out bins of
            the non-looming episodes only (same fitted models, evaluated on that subset).
            Summaries use the Day 5 variance floor (var > 1 Hz^2 over the 200 mixtures,
            results/w4/regression/day5/{pool,group}_variance.csv; var_noloom for the
            no-looming rows), per cfg.

Usage
  python scripts/w4d6_ridge_baseline.py            # all cfgs and input sets (~13 min, 4 cores, ~5 GB)
  python scripts/w4d6_ridge_baseline.py --check    # re-run calib 'all' fold 0 and compare
                                                   # its predictions with the saved run
  python scripts/w4d6_ridge_baseline.py --sens 500 # calib all / dn with 500 PCs -> sens_k500.txt
Output (results/w4/regression/ridge/)
  r2_per_pool.csv           one row per target (349 pools + 42 coarse): Day 5 variance,
                            floor flags, r2_<cfg>_<inputs>_<all|noloom>
  summary.md                tables: per cfg x inputs x subset x level
  log.txt                   neurons kept, PCA variance, alpha choices, timing per fold
  r2_calib_vs_shuffle.png   R^2 per pool, calib vs shuffle (all-brain inputs)
  pred_<cfg>_<inputs>.npz   held-out predictions (for --check and later analysis)
  log_check.txt, sens_k<K>.txt   outputs of --check and --sens
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.cns import cns_model, motor_pools, brain_mask, W4_CALIB  # noqa: E402

DAY5 = ROOT / 'results' / 'w4' / 'regression' / 'day5'
OUT = ROOT / 'results' / 'w4' / 'regression' / 'ridge'
CFGS = ('calib', 'shuffle')
INPUTS = ('all', 'dn', 'time', 'mean')
N_MIX = 200
DT = 1e-4
BIN = 100                    # steps per 10 ms bin
NB = 250                     # bins per 2.5 s episode
SKIP = 10                    # discard the first 100 ms
LAGS = tuple(range(11))      # 0, 10, ..., 100 ms
H = 10                       # target 100 ms ahead
T0, T1 = SKIP + max(LAGS), NB - H     # current bins t in [T0, T1) -> 220 per episode
N_MIN = 20                   # min spikes of an input neuron in the training episodes
K_PC = 200                   # principal components (all-brain and DN inputs)
N_FOLD, N_INNER = 5, 4
ALPHAS = np.logspace(0, 7, 15)
SEED = 0
SHIFT_MIN = 30               # bins (300 ms)
FLOOR = 1.0                  # Hz^2, Day 5 floor
R2_GOOD = 0.1

_LOG = []


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    _LOG.append(s)


def has_word(spec, w):
    return any(x['word'] == w for x in spec['words'])


def folds(eps_loom, eps_wind, n_fold, rng):
    """Fold id per episode: strata (looming, wind), permuted, dealt round-robin."""
    n = len(eps_loom)
    fid = np.full(n, -1)
    j = 0
    for lo in (True, False):
        for wi in (True, False):
            idx = np.flatnonzero((eps_loom == lo) & (eps_wind == wi))
            for e in rng.permutation(idx):
                fid[e] = j % n_fold
                j += 1
    return fid


def load_cfg(cfg, meta_bm, gid, gid_c, sizes, csizes, dn_mask):
    """Binned brain counts (superset: brain_mask neurons with >= N_MIN spikes over bins >= SKIP
    of all 200 episodes; per-fold selection happens later), pool and coarse rates, specs."""
    cnt = np.zeros(len(meta_bm), np.int64)
    raw = []
    for k in range(N_MIX):
        d = np.load(DAY5 / f'ep{k:03d}_{cfg}.npz')
        t, i = d['t'].astype(np.int64), d['i'].astype(np.int64)
        b = t // BIN
        spec = json.loads(str(d['spec']))
        raw.append((b, i, spec))
        k2 = b >= SKIP
        cnt += np.bincount(i[k2], minlength=len(meta_bm))
    sup = np.flatnonzero(meta_bm & (cnt >= N_MIN))
    col = np.full(len(meta_bm), -1)
    col[sup] = np.arange(len(sup))
    R = np.zeros((N_MIX, NB, len(sup)), np.float32)
    Y = np.zeros((N_MIX, NB, len(sizes)))
    Yc = np.zeros((N_MIX, NB, len(csizes)))
    loom, wind = np.zeros(N_MIX, bool), np.zeros(N_MIX, bool)
    for k, (b, i, spec) in enumerate(raw):
        c = col[i]; m = c >= 0
        np.add.at(R[k], (b[m], c[m]), 1)
        g = gid[i]; m = g >= 0
        np.add.at(Y[k], (b[m], g[m]), 1)
        g = gid_c[i]; m = g >= 0
        np.add.at(Yc[k], (b[m], g[m]), 1)
        loom[k], wind[k] = has_word(spec, 'looming'), has_word(spec, 'wind')
    Y /= sizes * BIN * DT
    Yc /= csizes * BIN * DT
    return dict(R=R, sup=sup, Y=np.concatenate([Y, Yc], 2), loom=loom, wind=wind,
                is_dn=dn_mask[sup])


def time_shift(Yall, rng):
    """Circular shift of each episode's target series (bins T0+H .. T1+H-1) by >= SHIFT_MIN."""
    Ys = Yall.copy()
    n = T1 - T0
    off = rng.integers(SHIFT_MIN, n - SHIFT_MIN + 1, size=len(Yall))
    for k in range(len(Yall)):
        seg = Yall[k, T0 + H:T1 + H]
        Ys[k, T0 + H:T1 + H] = np.roll(seg, off[k], axis=0)
    return Ys, off


def design(Z, eps):
    """Lagged features (n_eps*220, K*len(LAGS)) from per-bin scores Z (N_MIX, NB, K)."""
    ts = np.arange(T0, T1)
    return np.concatenate([np.concatenate([Z[e, ts - l] for l in LAGS], 1) for e in eps], 0)


def target(Y, eps):
    ts = np.arange(T0, T1) + H
    return np.concatenate([Y[e, ts] for e in eps], 0)


def pc_scores(R, cols, tr_eps, seed):
    """z-score + PCA on the training episodes' bins >= SKIP; scores for all episodes."""
    Xtr = R[np.ix_(tr_eps, np.arange(SKIP, NB), cols)].reshape(-1, len(cols))   # float32 copy
    mu = Xtr.mean(0, dtype=np.float64)
    sd = np.sqrt(((Xtr - mu.astype(np.float32)) ** 2).mean(0, dtype=np.float64))
    sd[sd == 0] = 1.0
    Xtr -= mu.astype(np.float32)
    Xtr /= sd.astype(np.float32)
    A = torch.from_numpy(Xtr)
    q = min(K_PC, len(cols))
    torch.manual_seed(seed)
    if q == len(cols):
        _, S, Vh = torch.linalg.svd(A, full_matrices=False)
        V = Vh.T
    else:
        _, S, V = torch.pca_lowrank(A, q=q, center=False, niter=4)
    V = V[:, :q].numpy().astype(np.float64)                 # (p, q)
    tot = sum(float((Xtr[j:j + 4096].astype(np.float64) ** 2).sum()) for j in range(0, len(Xtr), 4096))
    ev = float((S[:q].numpy().astype(np.float64) ** 2).sum()) / tot
    del A, Xtr
    Z = np.empty((N_MIX, NB, q))
    for e in range(N_MIX):
        Z[e] = ((R[e][:, cols] - mu) / sd) @ V
    return Z, ev


class Ridge:
    """Closed-form ridge via eigendecomposition of X'X (features standardised, centred)."""

    def __init__(self, X, Y):
        self.mx, self.sx = X.mean(0), X.std(0)
        self.sx[self.sx == 0] = 1.0
        self.my = Y.mean(0)
        Xs = (X - self.mx) / self.sx
        s2, self.V = np.linalg.eigh(Xs.T @ Xs)
        self.s2 = np.clip(s2, 0, None)
        self.B = self.V.T @ (Xs.T @ (Y - self.my))          # (p, T)

    def project(self, X):
        return ((X - self.mx) / self.sx) @ self.V

    def predict(self, A, alpha):
        """A = project(X); alpha scalar or per-target array."""
        alpha = np.broadcast_to(np.asarray(alpha, float), (self.B.shape[1],))
        out = np.empty((A.shape[0], self.B.shape[1]))
        for a in np.unique(alpha):
            j = alpha == a
            out[:, j] = A @ (self.B[:, j] / (self.s2 + a)[:, None])
        return out + self.my


def fit_predict(Z, Y, tr, te, loom, wind, seed):
    """Inner CV (by episode) picks alpha per target; refit on tr; predict te."""
    rng = np.random.default_rng(seed)
    ifold = folds(loom[tr], wind[tr], N_INNER, rng)
    sse = np.zeros((len(ALPHAS), Y.shape[2]))
    for f in range(N_INNER):
        itr, iva = tr[ifold != f], tr[ifold == f]
        m = Ridge(design(Z, itr), target(Y, itr))
        A, Yv = m.project(design(Z, iva)), target(Y, iva)
        for a, al in enumerate(ALPHAS):
            sse[a] += ((m.predict(A, al) - Yv) ** 2).sum(0)
    best = ALPHAS[sse.argmin(0)]
    m = Ridge(design(Z, tr), target(Y, tr))
    return m.predict(m.project(design(Z, te)), best), best


def r2(y, p):
    sst = ((y - y.mean(0)) ** 2).sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(sst > 0, 1 - ((y - p) ** 2).sum(0) / sst, np.nan)


def run(D, inputs, fid, seed_base, only_fold=None):
    """Held-out predictions (N_MIX, 220, n_targets) for one cfg and input set."""
    Y = D['Ytime'] if inputs == 'time' else D['Y']
    P = np.full((N_MIX, T1 - T0, Y.shape[2]), np.nan)
    fl = range(N_FOLD) if only_fold is None else [only_fold]
    for f in fl:
        t0 = time.time()
        tr, te = np.flatnonzero(fid != f), np.flatnonzero(fid == f)
        if inputs == 'mean':
            P[te] = target(Y, tr).mean(0)
            log(f'  fold {f}: mean-only')
            continue
        cnt = D['R'][tr][:, SKIP:].sum((0, 1))
        keep = cnt >= N_MIN
        if inputs == 'dn':
            keep &= D['is_dn']
        cols = np.flatnonzero(keep)
        Z, ev = pc_scores(D['R'], cols, tr, seed_base + f)
        pred, best = fit_predict(Z, Y, tr, te, D['loom'], D['wind'], seed_base + 100 + f)
        P[te] = pred.reshape(len(te), T1 - T0, -1)
        live = target(Y, tr).var(0) > 0                      # alpha is moot for constant targets
        ae = np.bincount(np.searchsorted(ALPHAS, best[live]), minlength=len(ALPHAS))
        log(f'  fold {f}: train {len(tr)} eps ({int(D["loom"][tr].sum())} looming), test {len(te)} '
            f'({int(D["loom"][te].sum())} looming); {len(cols)} input neurons, {Z.shape[2]} PCs '
            f'({ev:.1%} of z-scored variance), {Z.shape[2] * len(LAGS)} features; alpha median '
            f'{np.median(best):.3g}, at grid min {ae[0]}, at grid max {ae[-1]} of {int(live.sum())} non-constant targets (of {len(best)}); '
            f'{time.time() - t0:.0f} s')
    return P


def context():
    m = cns_model('cns', 'calib', verbose=False, **W4_CALIB)
    meta, vnc = m['meta'], m['vnc']
    del m
    ms = cns_model('cns', 'shuffle', verbose=False, **W4_CALIB)['meta']
    same = len(ms) == len(meta) and (ms.bodyId.to_numpy() == meta.bodyId.to_numpy()).all()
    log(f'meta: {len(meta)} neurons; shuffle meta has the same bodyId order: {same}')
    if not same:
        sys.exit('shuffle meta order differs from calib: indices would not match')
    bm = np.asarray(brain_mask(meta, vnc), bool)
    sc = meta.superclass.astype(str)
    dn = bm & sc.str.contains('descending').to_numpy()
    log(f'brain_mask: {int(bm.sum())} neurons; DN rule (superclass contains "descending", in '
        f'brain_mask): {int(dn.sum())} neurons, superclasses {sorted(sc[dn].unique())}')
    pools, pdf = motor_pools(meta)
    names = list(pools)
    n = len(meta)
    gid = np.full(n, -1)
    for k, nm in enumerate(names):
        gid[pools[nm]] = k
    sizes = np.array([len(pools[nm]) for nm in names], float)
    pdf = pdf.set_index('pool').loc[names].reset_index()
    pdf['coarse'] = pdf.group + ' | ' + pdf.side
    cnames = sorted(pdf.coarse.unique())
    cid = {c: k for k, c in enumerate(cnames)}
    gid_c = np.full(n, -1)
    for nm, c in zip(names, pdf.coarse):
        gid_c[pools[nm]] = cid[c]
    csizes = np.bincount(gid_c[gid_c >= 0], minlength=len(cnames)).astype(float)
    tg = pd.concat([pd.DataFrame(dict(level='fine', name=names, group=pdf.group, side=pdf.side,
                                      type=pdf.type, n=sizes)),
                    pd.DataFrame(dict(level='coarse', name=cnames, group=[c.split(' | ')[0] for c in cnames],
                                      side=[c.split(' | ')[1] for c in cnames], type='', n=csizes))],
                   ignore_index=True)
    return dict(bm=bm, dn=dn, gid=gid, gid_c=gid_c, sizes=sizes, csizes=csizes, tg=tg)


def floors(tg):
    pv = pd.read_csv(DAY5 / 'pool_variance.csv')
    gv = pd.read_csv(DAY5 / 'group_variance.csv')
    for cfg in CFGS:
        v = pd.concat([pv[pv.cfg == cfg].set_index('name')[['var', 'var_noloom']],
                       gv[gv.cfg == cfg].set_index('name')[['var', 'var_noloom']]])
        v = v.loc[tg.name]
        tg[f'var_{cfg}'] = v['var'].to_numpy()
        tg[f'var_noloom_{cfg}'] = v['var_noloom'].to_numpy()
        tg[f'floor_{cfg}'] = tg[f'var_{cfg}'] > FLOOR
        tg[f'floor_noloom_{cfg}'] = tg[f'var_noloom_{cfg}'] > FLOOR
    return tg


def stats(x):
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0, med=np.nan, q1=np.nan, q3=np.nan, good=0)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return dict(n=len(x), med=med, q1=q1, q3=q3, good=int((x > R2_GOOD).sum()))


def summarise(tg, timing):
    L = ['# W4D6 ridge baseline: brain -> motor-pool rates 100 ms ahead', '',
         'Generated by `scripts/w4d6_ridge_baseline.py` from `results/w4/regression/day5/` (200 mixture '
         'episodes per cfg). 10 ms bins, first 100 ms discarded, inputs at lags 0-100 ms (11 lags), '
         f'targets at t + 100 ms; {N_FOLD}-fold CV by episode stratified by (looming, wind); alpha per '
         f'target by inner {N_INNER}-fold CV on the training episodes; inputs = brain_mask neurons with '
         f'>= {N_MIN} spikes in the training episodes, z-scored, {K_PC} PCs fitted on the training '
         'episodes. R^2 on held-out bins pooled over folds. Pools: only those above the Day 5 floor '
         f'(var > {FLOOR:g} Hz^2 over the 200 mixtures; the no-looming rows use the no-looming variance).',
         '', 'Inputs: **all** = all brain inputs; **dn** = descending neurons only; **time** = all brain '
         'inputs, targets circularly shifted within each episode by >= 300 ms; **mean** = training-fold mean.',
         '']
    for level, lab in (('fine', 'pools (of 349)'), ('coarse', 'group x side (of 42)')):
        L += [f'## R^2, {level} level: {lab}', '',
              '| cfg | inputs | episodes | n above floor | median R^2 | IQR | R^2 > 0.1 |',
              '|---|---|---|---|---|---|---|']
        for cfg in CFGS:
            for inp in INPUTS:
                for sub in ('all', 'noloom'):
                    fl = tg[f'floor_{cfg}' if sub == 'all' else f'floor_noloom_{cfg}'] & (tg.level == level)
                    s = stats(tg.loc[fl, f'r2_{cfg}_{inp}_{sub}'].to_numpy())
                    L.append(f'| {cfg} | {inp} | {"all 200" if sub == "all" else "no looming"} | {s["n"]} | '
                             f'{s["med"]:.3f} | {s["q1"]:.3f} to {s["q3"]:.3f} | {s["good"]} |')
        L.append('')
    # paired comparisons on a common pool set
    L += ['## Paired comparisons (fine pools)', '']
    f = tg.level == 'fine'
    c = f & tg.floor_calib
    s_all = tg.loc[c, 'r2_calib_all_all']
    for inp in ('dn',):
        s_x = tg.loc[c, f'r2_calib_{inp}_all']
        good = c & (tg.r2_calib_all_all > R2_GOOD)
        ratio = (tg.loc[good, f'r2_calib_{inp}_all'] / tg.loc[good, 'r2_calib_all_all'])
        L.append(f'- calib, {inp} vs all, on the {int(c.sum())} calib pools above the floor: median R^2 '
                 f'{s_x.median():.3f} vs {s_all.median():.3f}; sum of positive R^2 '
                 f'{s_x.clip(lower=0).sum():.2f} vs {s_all.clip(lower=0).sum():.2f} '
                 f'({s_x.clip(lower=0).sum() / max(s_all.clip(lower=0).sum(), 1e-12):.0%}); on the '
                 f'{int(good.sum())} pools with all-brain R^2 > {R2_GOOD:g}, median ratio {ratio.median():.2f}.')
    cc = f & tg.floor_shuffle
    x = tg.loc[cc, 'r2_shuffle_dn_all']; y = tg.loc[cc, 'r2_shuffle_all_all']
    L.append(f'- shuffle, dn vs all, sum of positive R^2 on its {int(cc.sum())} floor pools: '
             f'{x.clip(lower=0).sum():.2f} vs {y.clip(lower=0).sum():.2f}.')
    sh_on_c = tg.loc[c, 'r2_shuffle_all_all']
    dfn = sh_on_c.notna()
    L.append(f'- calib vs shuffle on the {int(c.sum())} calib floor pools: in {int((~dfn).sum())} of them the '
             'shuffle pool has no spike in the held-out target bins (R^2 undefined; the shuffled network never '
             f'drives it); on the {int(dfn.sum())} where both are defined, median R^2 calib '
             f'{s_all[dfn].median():.3f} vs shuffle {sh_on_c[dfn].median():.3f}, calib higher in '
             f'{int((s_all[dfn] > sh_on_c[dfn]).sum())}.')
    both = f & tg.floor_calib & tg.floor_shuffle
    L.append(f'- pools above the floor in both cfgs: {int(both.sum())}; median R^2 calib '
             f'{tg.loc[both, "r2_calib_all_all"].median():.3f}, shuffle {tg.loc[both, "r2_shuffle_all_all"].median():.3f}.')
    nl = f & tg.floor_noloom_calib
    L.append(f'- calib, looming effect on the {int(nl.sum())} pools above the no-looming floor: median R^2 '
             f'all episodes {tg.loc[nl, "r2_calib_all_all"].median():.3f}, no-looming episodes '
             f'{tg.loc[nl, "r2_calib_all_noloom"].median():.3f}.')
    L += ['', '## Top calib pools (all-brain, all episodes)', '',
          '| pool | group | Day 5 var Hz^2 | R^2 all | R^2 no looming | R^2 dn | R^2 shuffle |', '|---|---|---|---|---|---|---|']
    top = tg[f].sort_values('r2_calib_all_all', ascending=False).head(15)
    for _, r in top.iterrows():
        L.append(f'| {r["name"]} | {r.group} | {r.var_calib:.1f} | {r.r2_calib_all_all:.3f} | '
                 f'{r.r2_calib_all_noloom:.3f} | {r.r2_calib_dn_all:.3f} | {r.r2_shuffle_all_all:.3f} |')
    L += ['', '## Run', ''] + [f'- {t}' for t in timing] + ['', 'Per-fold details: `log.txt`.', '']
    return L


def figure(tg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f = tg[tg.level == 'fine']
    lo = -0.2
    x = f.r2_calib_all_all.clip(lower=lo).fillna(lo)
    y = f.r2_shuffle_all_all.clip(lower=lo).fillna(lo)
    fig, ax = plt.subplots(figsize=(6, 6))
    cat = np.where(f.floor_calib & f.floor_shuffle, 'above floor in both',
                   np.where(f.floor_calib, 'above floor in calib only',
                            np.where(f.floor_shuffle, 'above floor in shuffle only', 'below floor in both')))
    for lab, col, s in (('below floor in both', '#bbbbbb', 8), ('above floor in shuffle only', '#d95f02', 18),
                        ('above floor in calib only', '#1b9e77', 18), ('above floor in both', '#7570b3', 24)):
        k = cat == lab
        ax.scatter(x[k], y[k], s=s, c=col, label=f'{lab} ({int(k.sum())})', alpha=0.8, edgecolors='none')
    ax.plot([lo, 1], [lo, 1], 'k:', lw=0.8)
    ax.set_xlim(lo - 0.02, 1); ax.set_ylim(lo - 0.02, 1)
    ax.set_xlabel('R$^2$ calib (real connectome)')
    ax.set_ylabel('R$^2$ shuffle (degree-preserving)')
    ax.set_title('Ridge, all-brain inputs, 100 ms ahead: R$^2$ per pool\n'
                 f'(held-out episodes, 5-fold; values < {lo} and undefined shown at {lo})', fontsize=9)
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(OUT / 'r2_calib_vs_shuffle.png', dpi=130)
    plt.close(fig)


def sensitivity(k):
    """Re-run calib 'all' and 'dn' with k PCs (same folds and seeds); medians on the floor pools."""
    global K_PC
    K_PC = k
    torch.set_num_threads(4)
    log(f'[w4d6_ridge --sens {k}] {time.strftime("%Y-%m-%d %H:%M:%S")} calib, K_PC={k}, otherwise as the main run')
    X = context()
    tg = floors(X['tg'].copy())
    D = load_cfg('calib', X['bm'], X['gid'], X['gid_c'], X['sizes'], X['csizes'], X['dn'])
    fid = folds(D['loom'], D['wind'], N_FOLD, np.random.default_rng(SEED))
    Yt = D['Y'][:, T0 + H:T1 + H]
    yy = Yt.reshape(-1, Yt.shape[2])
    nl = np.repeat(~D['loom'], T1 - T0)
    for ii, inp in enumerate(('all', 'dn')):
        t1 = time.time()
        log(f'[calib/{inp}, K={k}]')
        pp = run(D, inp, fid, 1000 + 10 * ii).reshape(-1, yy.shape[1])
        for sub, m, flc in (('all', slice(None), 'floor_calib'), ('noloom', nl, 'floor_noloom_calib')):
            for level in ('fine', 'coarse'):
                fl = (tg[flc] & (tg.level == level)).to_numpy()
                s = stats(r2(yy[m], pp[m])[fl])
                log(f'[calib/{inp}, K={k}] {level}, episodes {sub}: n {s["n"]}, median R^2 {s["med"]:.3f}, '
                    f'IQR {s["q1"]:.3f} to {s["q3"]:.3f}, R^2 > {R2_GOOD:g}: {s["good"]}')
        log(f'  {time.time() - t1:.0f} s')
    (OUT / f'sens_k{k}.txt').write_text('\n'.join(_LOG) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='re-run calib/all fold 0, compare with saved')
    ap.add_argument('--sens', type=int, metavar='K', help='sensitivity: calib all/dn with K PCs -> sens_k<K>.txt')
    args = ap.parse_args()
    if args.sens:
        return sensitivity(args.sens)
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log(f'[w4d6_ridge] {time.strftime("%Y-%m-%d %H:%M:%S")} N_MIN={N_MIN} K_PC={K_PC} LAGS={LAGS} H={H} '
        f'SKIP={SKIP} folds={N_FOLD}/{N_INNER} alphas={ALPHAS[0]:g}..{ALPHAS[-1]:g} ({len(ALPHAS)}) seed={SEED}')
    X = context()
    tg = floors(X['tg'].copy())
    timing = []
    cfgs = ('calib',) if args.check else CFGS
    for ci, cfg in enumerate(cfgs):
        t0 = time.time()
        D = load_cfg(cfg, X['bm'], X['gid'], X['gid_c'], X['sizes'], X['csizes'], X['dn'])
        fid = folds(D['loom'], D['wind'], N_FOLD, np.random.default_rng(SEED))
        D['Ytime'], off = time_shift(D['Y'], np.random.default_rng(SEED + 1))
        log(f'[{cfg}] loaded in {time.time() - t0:.0f} s: superset {len(D["sup"])} brain neurons '
            f'(>= {N_MIN} spikes over all 200 episodes; {int(D["is_dn"].sum())} DNs), '
            f'R {D["R"].nbytes / 1e9:.2f} GB; looming per fold {np.bincount(fid[D["loom"]], minlength=N_FOLD).tolist()}, '
            f'wind per fold {np.bincount(fid[D["wind"]], minlength=N_FOLD).tolist()}; time-shift offsets '
            f'{off.min()}-{off.max()} bins')
        Yt = D['Y'][:, T0 + H:T1 + H]
        if args.check:
            ref = np.load(OUT / 'pred_calib_all.npz')
            assert (ref['fid'] == fid).all(), 'fold assignment differs'
            P = run(D, 'all', fid, 1000 * (ci + 1), only_fold=0)
            te = fid == 0
            d = np.abs(P[te].astype(np.float32) - ref['P'][te]).max()    # saved as float32
            log(f'[check] calib/all fold 0: max |pred - saved| = {d:.3g}; identical: {d == 0}')
            (OUT / 'log_check.txt').write_text('\n'.join(_LOG) + '\n')
            return
        for ii, inp in enumerate(INPUTS):
            t1 = time.time()
            log(f'[{cfg}/{inp}]')
            P = run(D, inp, fid, 1000 * (ci + 1) + 10 * ii)
            Yref = D['Ytime'][:, T0 + H:T1 + H] if inp == 'time' else Yt
            np.savez_compressed(OUT / f'pred_{cfg}_{inp}.npz', P=P.astype(np.float32), fid=fid, loom=D['loom'])
            yy, pp = Yref.reshape(-1, Yref.shape[2]), P.reshape(-1, P.shape[2])
            nl = np.repeat(~D['loom'], T1 - T0)
            tg[f'r2_{cfg}_{inp}_all'] = r2(yy, pp)
            tg[f'r2_{cfg}_{inp}_noloom'] = r2(yy[nl], pp[nl])
            fl = tg[f'floor_{cfg}'] & (tg.level == 'fine')
            log(f'[{cfg}/{inp}] median R^2 on {int(fl.sum())} floor pools: '
                f'{np.nanmedian(tg.loc[fl, f"r2_{cfg}_{inp}_all"]):.3f}; {time.time() - t1:.0f} s')
            timing.append(f'{cfg}/{inp}: {time.time() - t1:.0f} s')
        del D
    tg.to_csv(OUT / 'r2_per_pool.csv', index=False)
    timing.append(f'total {time.time() - t_start:.0f} s')
    L = summarise(tg, timing)
    (OUT / 'summary.md').write_text('\n'.join(L))
    figure(tg)
    log('\n'.join(L))
    log(f'[w4d6_ridge] done in {time.time() - t_start:.0f} s')
    (OUT / 'log.txt').write_text('\n'.join(_LOG) + '\n')


if __name__ == '__main__':
    main()
