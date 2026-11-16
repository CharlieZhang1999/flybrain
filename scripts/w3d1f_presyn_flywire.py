#!/usr/bin/env python
"""Week 3 presyn step 3: FlyWire v783 taste check (Shiu model as published, plus
the Week 2 gap calibration) with and without presynaptic modulation.
Sensory = FlyWire super_class 'sensory'. Words: sugar, bitter, sugar+bitter,
water, none at 150 Hz for 1 s (Shiu protocol), seed 0 and 1. Reports MN9 (L) Hz.
Output: results/loop/w3d1f_flywire.csv"""
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from w3d1_common import ROOT
from sim.lif_numpy import load_network, run_trial, P
from sim.gap import build_gap
from sim.presyn import build_presyn
from stimuli.words import WORDS as FW, ID_MN9_LEFT

REPO = ROOT / 'third_party' / 'Drosophila_brain_model'
Ks = [float(k) for k in sys.argv[1:] if not k.startswith('--')] or [0.01]
ONLY = '--sugar-only' in sys.argv
ids, W = load_network(REPO / 'Completeness_783.csv', REPO / 'Connectivity_783.parquet')
n = len(ids); pos = pd.Series(np.arange(n), index=ids)
a = pd.read_csv(ROOT / 'data' / 'flywire_annot' / 'annot.tsv', sep='\t', low_memory=False,
                usecols=['root_id', 'super_class', 'cell_class'])
a = a[a.root_id.isin(pos.index)]
def mask(mm):
    out = np.zeros(n, bool); out[pos[a.root_id[mm]].to_numpy()] = True; return out
sens = mask(a.super_class == 'sensory')
alln, pn = mask(a.cell_class == 'ALLN'), mask(a.cell_class == 'ALPN')
ln = alln & (np.asarray(W.sum(axis=0)).ravel() > 0)
C = W.tocoo(); keep = ~(ln[C.col] & pn[C.row])
Wcut = sparse.csc_matrix((C.data[keep], (C.row[keep], C.col[keep])), shape=W.shape)
gap, _ = build_gap(abs(W) / P['w_syn'], ln, pn, 0.2, v_spike=0.002)
st = {w: np.sort(pos[[i for i in FW[w] if i in pos.index]].to_numpy()) for w in ['sugar', 'bitter', 'water']}
st['sugar+bitter'] = np.r_[st['sugar'], st['bitter']]; st['none'] = np.array([], int)
mn9 = int(pos[ID_MN9_LEFT])
print(f'sensory neurons: {sens.sum()}', flush=True)
conds = [] if ONLY else [('published', W, None, None), ('calib', Wcut, gap, None)]
for K in Ks:
    for name, Wb, gp in ([('published', W, None)] if ONLY else [('published', W, None), ('calib', Wcut, gap)]):
        W2, ps, info = build_presyn(Wb, sens, K)
        conds.append((f'{name}+presyn K={K}', W2, gp, ps))
p = dict(P); p['t_run'] = 1.0
rows = []
for name, Wc, gp, ps in conds:
    for word, idx in st.items():
        if ONLY and word not in ('sugar', 'sugar+bitter'):
            continue
        for seed in [0, 1]:
            r = run_trial(Wc, idx, params=p, seed=seed, record=True, gap=gp, presyn=ps)
            rows.append(dict(cond=name, word=word, seed=seed, mn9_Hz=float((r['i'] == mn9).sum() / 1.0),
                             active=len(np.unique(r['i']))))
            print(rows[-1], flush=True)
df = pd.DataFrame(rows); df.to_csv(ROOT / 'results' / 'loop' / ('w3d1f_flywire_Ksweep.csv' if ONLY else 'w3d1f_flywire.csv'), index=False)
print(df.groupby(['cond', 'word']).mn9_Hz.mean().unstack().round(1).to_string())
