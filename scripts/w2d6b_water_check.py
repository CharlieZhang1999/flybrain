#!/usr/bin/env python
"""Check water -> MN9 on FlyWire, base vs gap-junction calibrated, seeds 0-4.
Trial: water GRNs at 150 Hz for 0.5 s (rate over the stimulus window)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import sparse
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from sim.lif_numpy import load_network, run_trial, P
from sim.gap import build_gap
from stimuli.words import WORDS, ID_MN9_LEFT
REPO = ROOT / 'third_party' / 'Drosophila_brain_model'
ids, W = load_network(REPO / 'Completeness_783.csv', REPO / 'Connectivity_783.parquet')
n = len(ids); pos = pd.Series(np.arange(n), index=ids)
a = pd.read_csv(ROOT / 'data' / 'flywire_annot' / 'annot.tsv', sep='\t', low_memory=False); a = a[a.root_id.isin(pos.index)]
def mask(m):
    o = np.zeros(n, bool); o[pos[a.root_id[m]].to_numpy()] = True; return o
alln, pn = mask(a.cell_class == 'ALLN'), mask(a.cell_class == 'ALPN')
ln = alln & (np.asarray(W.sum(axis=0)).ravel() > 0)
C = W.tocoo(); keep = ~(ln[C.col] & pn[C.row])
Wc = sparse.csc_matrix((C.data[keep], (C.row[keep], C.col[keep])), shape=W.shape)
gap, _ = build_gap(abs(W) / P['w_syn'], ln, pn, c0=0.2, v_spike=0.002)
exc = pos[[i for i in WORDS['water'] if i in pos.index]].to_numpy(); m9 = pos[ID_MN9_LEFT]
p = dict(P); p['t_run'] = 0.5
for name, Wx, g in [('base', W, None), ('cut, no gap', Wc, None), ('calib', Wc, gap)]:
    r = [run_trial(Wx, exc, params=p, seed=s, gap=g)[m9] / 0.5 for s in range(5)]
    print(f'{name:12s} MN9 Hz per seed {r} mean {np.mean(r):.1f}', flush=True)
