#!/usr/bin/env python
"""Week 2 Day 1, D1: FlyWire v783 control for the post-odor persistent state.

Shiu et al. model exactly as published (Connectivity_783.parquet signs,
w_syn = 0.275 mV, monoamines excitatory, spiking APL, no extra scaling), run
with the same trial as C8: stimulus 0-1.0 s at 150 Hz Poisson, silence to 1.6 s.
Question: does FlyWire also keep firing after the stimulus ends?
Words use FlyWire cell_type labels (annot.tsv), matching stimuli/malecns_words.py:
  vinegar ORN_DM1+ORN_DM4, hot TRN_VP2, sound JO-A*/JO-B*, looming LPLC2+LC4,
  sugar = Shiu's 21 right labellar sugar GRNs (positive control), none.
Persistent run: > 20 neurons firing in the last 100 ms (same rule as C8).
Output: results/flywire/w2d1_flywire_control.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.lif_numpy import load_network, run_trial, P  # noqa: E402
from stimuli.words import WORDS as FW_TASTE, ID_MN9_LEFT  # noqa: E402

REPO = ROOT / 'third_party' / 'Drosophila_brain_model'
OUT = ROOT / 'results' / 'flywire'
OUT.mkdir(parents=True, exist_ok=True)
DT, T_END, BIN = P['dt'], 1.6, 0.005


def main():
    ids, W = load_network(REPO / 'Completeness_783.csv', REPO / 'Connectivity_783.parquet')
    pos = pd.Series(np.arange(len(ids)), index=ids)
    a = pd.read_csv(ROOT / 'data' / 'flywire_annot' / 'annot.tsv', sep='\t', low_memory=False)
    a = a[a.root_id.isin(pos.index)]
    ct = a.cell_type.fillna('')
    def idx(mask):
        return np.sort(pos[a.root_id[mask]].to_numpy())
    stim = {
        'vinegar': idx(ct.isin(['ORN_DM1', 'ORN_DM4'])),
        'hot': idx(ct == 'TRN_VP2'),
        'sound': idx(ct.str.match(r'^JO-[AB]')),
        'looming': idx(ct.isin(['LPLC2', 'LC4'])),
        'sugar': np.sort(pos[[i for i in FW_TASTE['sugar'] if i in pos.index]].to_numpy()),
        'none': np.array([], int),
    }
    al = np.zeros(len(ids), bool); al[idx(a.cell_class.isin(['ALLN', 'ALPN']))] = True
    lln1 = idx(ct == 'lLN1_bc')
    mn9 = int(pos[ID_MN9_LEFT])
    p = dict(P); p['t_run'] = T_END
    print(f'FlyWire v783: {len(ids):,} neurons, {W.nnz:,} edges; ' +
          ', '.join(f'{k}={len(v)}' for k, v in stim.items()), flush=True)

    rows = []
    for word in ['vinegar', 'hot', 'sound', 'looming', 'sugar', 'none']:
        for seed in [0, 1]:
            r = run_trial(W, stim[word], params=p, seed=seed, t_stim=1.0, record=True)
            ts, si = r['t'] * DT, r['i']
            def cnt(lo, hi):
                m = (ts >= lo) & (ts < hi)
                return np.bincount(si[m], minlength=len(ids))
            odor_c, last = cnt(0.5, 1.0), cnt(T_END - 0.1, T_END)
            b = r['t'] // int(round(BIN / DT))
            act = np.bincount(b, minlength=int(round(T_END / BIN)))  # spikes per bin
            nact = np.array([len(np.unique(si[b == k])) for k in range(int(round(T_END / BIN)))])
            ref = nact[int(0.9 / BIN):int(1.0 / BIN)].mean()
            below = np.flatnonzero(nact[int(1.0 / BIN):] < 0.1 * max(ref, 1))
            x = dict(word=word, seed=seed, n_stim=len(stim[word]),
                     stim_active=int((odor_c > 0).sum()), stim_AL=int((odor_c[al] > 0).sum()),
                     mn9_stim_Hz=float(odor_c[mn9] / 0.5),
                     lLN1_bc_stim_Hz=float(odor_c[lln1].mean() / 0.5),
                     tail_brain=int((last > 0).sum()), tail_AL=int((last[al] > 0).sum()),
                     lLN1_bc_tail_Hz=float(last[lln1].mean() / 0.1),
                     decay_ms=float(below[0] * 5) if len(below) else np.nan,
                     persistent=bool((last > 0).sum() > 20))
            rows.append(x)
            print(f'{word:8s} s{seed}: stim active={x["stim_active"]:6d} AL={x["stim_AL"]:4d} '
                  f'MN9={x["mn9_stim_Hz"]:5.1f} lLN1_bc={x["lLN1_bc_stim_Hz"]:5.1f}Hz | '
                  f'tail brain={x["tail_brain"]:5d} AL={x["tail_AL"]:4d} '
                  f'lLN1_bc={x["lLN1_bc_tail_Hz"]:5.1f}Hz decay={x["decay_ms"]} ms '
                  f'persistent={x["persistent"]}', flush=True)
            pd.DataFrame(rows).to_csv(OUT / 'w2d1_flywire_control.csv', index=False)
    print('DONE')


if __name__ == '__main__':
    main()
