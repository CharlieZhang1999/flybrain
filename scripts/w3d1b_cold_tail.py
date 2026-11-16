#!/usr/bin/env python
"""Week 3 Day 1 follow-up: which neurons sustain the cold / humid tail?
Calibrated MaleCNS, cold or humid 0-0.5 s at 150 Hz, silence to 3.0 s, seed 0.
Reports tail neurons (firing in 2.5-3.0 s) by class/type, their rates, and the
strongly connected components of W restricted to the tail neurons.
Output: results/loop/w3d1b_tail_<word>.csv, printed summary."""
import sys
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from w3d1_common import calib_model, ROOT, DT
from sim.lif_numpy import run_trial
from scripts.ckpt2_ab import stim_idx

R, W, gap = calib_model()
for word in sys.argv[1:]:
    p = dict(R.p); p['t_run'] = 3.0
    idx = stim_idx(R.meta, R.pos, word)
    r = run_trial(W, idx, params=p, seed=0, apl=R.apl, t_stim=0.5, record=True, gap=gap)
    ts, si = r['t'] * DT, r['i']
    n = len(R.meta)
    for a in [0.5, 1.0, 1.5, 2.0, 2.5]:
        m = (ts >= a) & (ts < a + 0.5)
        print(f'{word} {a:.1f}-{a+0.5:.1f} s: active={len(np.unique(si[m]))}')
    m = ts >= 2.5
    c = np.bincount(si[m], minlength=n) / 0.5
    tail = np.flatnonzero(c > 0)
    d = R.meta.iloc[tail][['bodyId', 'type', 'instance', 'superclass', 'class', 'nt']].copy()
    d['rate_Hz'] = c[tail]
    # SCCs of the chemical graph among tail neurons (+ gap pairs as both directions)
    Ws = W[tail][:, tail].copy()
    G = (abs(Ws) > 0).astype(int)
    if gap is not None:
        gi = {g: k for k, g in enumerate(gap['idx'])}
        loc = np.array([gi.get(t, -1) for t in tail])
        ok = loc >= 0
        C = gap['C'].tocsr()[loc[ok]][:, loc[ok]]
        Gg = np.zeros((len(tail), len(tail)), int)
        sub = np.flatnonzero(ok)
        Cd = (C.toarray() > 0)
        Gg[np.ix_(sub, sub)] = Cd
        G = ((G.toarray() + Gg + Gg.T) > 0).astype(int)
    else:
        G = G.toarray()
    # excitatory-only graph for loops
    E = (Ws.toarray() > 0).astype(int)
    ncomp, lab = connected_components(E, directed=True, connection='strong')
    d['scc_exc'] = lab
    sizes = pd.Series(lab).value_counts()
    big = sizes[sizes > 1]
    print(f'{word}: tail neurons {len(tail)}; excitatory SCCs >1: {big.to_dict()}')
    d.to_csv(ROOT / 'results' / 'loop' / f'w3d1b_tail_{word}.csv', index=False)
    print(d.groupby(['superclass', 'class'], dropna=False).size().sort_values(ascending=False).head(12).to_string())
    print(d.groupby(['type', 'nt']).rate_Hz.agg(['size', 'mean']).sort_values('size', ascending=False).head(25).round(0).to_string())
    for s in big.index[:3]:
        dd = d[d.scc_exc == s]
        print(f'-- SCC {s} ({len(dd)} neurons):', dd.groupby('type').rate_Hz.agg(['size', 'mean']).round(0).sort_values('size', ascending=False).head(15).to_dict('index'))
