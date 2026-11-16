#!/usr/bin/env python
"""Week 2 Days 2-3: the same eLN->PN gap-junction calibration on FlyWire v783.

Shiu model as published (signs, w_syn, monoamines excitatory, spiking APL).
eLN = FlyWire ALLN whose outgoing Shiu weights are excitatory. PN = ALPN.
Conditions: base, chemical eLN->PN 0, and chemical 0 + gap at c0 given on the
command line (default 0.01 0.02 0.04). Stimuli: vinegar, hot, sugar, none.
Output: results/flywire/w2d3_gap_flywire.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.lif_numpy import load_network, run_trial, P  # noqa: E402
from sim.gap import build_gap, pn_glomeruli  # noqa: E402
from stimuli.words import WORDS as FW_TASTE, ID_MN9_LEFT  # noqa: E402

REPO = ROOT / 'third_party' / 'Drosophila_brain_model'
OUT = ROOT / 'results' / 'flywire'
DT, T_END, BIN = P['dt'], 1.6, 0.005
TARGET = {'vinegar': {'DM1', 'DM4'}, 'hot': {'VP2'}}


def main(c0s):
    ids, W = load_network(REPO / 'Completeness_783.csv', REPO / 'Connectivity_783.parquet')
    n = len(ids)
    pos = pd.Series(np.arange(n), index=ids)
    a = pd.read_csv(ROOT / 'data' / 'flywire_annot' / 'annot.tsv', sep='\t', low_memory=False)
    a = a[a.root_id.isin(pos.index)]
    ct = a.cell_type.fillna('')
    def mask(m):
        out = np.zeros(n, bool); out[pos[a.root_id[m]].to_numpy()] = True; return out
    alln, pn = mask(a.cell_class == 'ALLN'), mask(a.cell_class == 'ALPN')
    outsum = np.asarray(W.sum(axis=0)).ravel()        # summed signed output per pre
    ln = alln & (outsum > 0)
    glom = pd.Series('', index=np.arange(n)); glom[pos[a.root_id].to_numpy()] = ct.to_numpy()
    targ = {w: pn & glom.map(lambda t: bool(pn_glomeruli(t) & g)).to_numpy() for w, g in TARGET.items()}
    stim = {
        'vinegar': np.flatnonzero(mask(ct.isin(['ORN_DM1', 'ORN_DM4']))),
        'hot': np.flatnonzero(mask(ct == 'TRN_VP2')),
        'sugar': np.sort(pos[[i for i in FW_TASTE['sugar'] if i in pos.index]].to_numpy()),
        'none': np.array([], int),
    }
    al = alln | pn
    lln1 = np.flatnonzero(mask(ct == 'lLN1_bc'))
    mn9 = int(pos[ID_MN9_LEFT])
    print(f'eLN={ln.sum()} (of {alln.sum()} ALLN), PN={pn.sum()}, target PNs '
          f'{ {w: int(m.sum()) for w, m in targ.items()} }', flush=True)
    C = W.tocoo()
    keep = ~(ln[C.col] & pn[C.row])
    Wcut = sparse.csc_matrix((C.data[keep], (C.row[keep], C.col[keep])), shape=W.shape)
    Wsyn = abs(W) / P['w_syn']
    conds = [('base', W, None), ('LN->PN chem 0', Wcut, None)]
    for c0 in c0s:
        gap, info = build_gap(Wsyn, ln, pn, c0)
        print(f'gap c0={c0}: {info}', flush=True)
        conds.append((f'gap c0={c0}', Wcut, gap))
    p = dict(P); p['t_run'] = T_END
    rows = []
    for name, Wc, gap in conds:
        for word in ['vinegar', 'hot', 'sugar', 'none']:
            for seed in ([0, 1] if word in ('vinegar', 'hot') else [0]):
                r = run_trial(Wc, stim[word], params=p, seed=seed, t_stim=1.0, record=True, gap=gap)
                ts, si = r['t'] * DT, r['i']
                def cnt(lo, hi):
                    m = (ts >= lo) & (ts < hi)
                    return np.bincount(si[m], minlength=n)
                odor_c, last = cnt(0.5, 1.0), cnt(T_END - 0.1, T_END)
                tg = targ.get(word, np.zeros(n, bool)); other = pn & ~tg
                x = dict(cond=name, word=word, seed=seed,
                         stim_active=int((odor_c > 0).sum()), odor_AL=int((odor_c[al] > 0).sum()),
                         target_PN_Hz=float(odor_c[tg].mean() / 0.5) if tg.any() else np.nan,
                         other_PN_Hz=float(odor_c[other].mean() / 0.5),
                         other_PN_active=int((odor_c[other] > 0).sum()),
                         eLN_odor_Hz=float(odor_c[ln].mean() / 0.5),
                         mn9_Hz=float(odor_c[mn9] / 0.5),
                         tail_brain=int((last > 0).sum()),
                         lLN1_bc_tail_Hz=float(last[lln1].mean() / 0.1),
                         persistent=bool((last > 0).sum() > 20))
                rows.append(x)
                print(f'[{name}] {word:8s} s{seed}: active={x["stim_active"]:5d} AL={x["odor_AL"]:4d} '
                      f'targetPN={x["target_PN_Hz"]:5.1f} otherPN={x["other_PN_Hz"]:5.1f}Hz '
                      f'({x["other_PN_active"]} active) eLN={x["eLN_odor_Hz"]:5.1f} MN9={x["mn9_Hz"]:4.0f} | '
                      f'tail={x["tail_brain"]:5d} lLN1_bc={x["lLN1_bc_tail_Hz"]:5.1f} pers={x["persistent"]}',
                      flush=True)
                pd.DataFrame(rows).to_csv(OUT / 'w2d3_gap_flywire.csv', index=False)
    print('DONE')


if __name__ == '__main__':
    main([float(x) for x in sys.argv[1:]] or [0.01, 0.02, 0.04])
