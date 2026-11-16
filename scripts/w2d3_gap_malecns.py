#!/usr/bin/env python
"""Week 2 Days 2-3: replace cholinergic eLN->PN chemical synapses with gap junctions (MaleCNS).

Biology (Yaksi & Wilson 2010): eLN->PN excitation survives Cd2+ and nicotinic
blockers and is removed in shakB2 -> electrical. PN->eLN is mixed
chemical + electrical; eLN->iLN mostly chemical. So:
  chemical cholinergic LN -> PN  set to 0
  PN -> LN, LN -> LN, LN -> GABA LN chemical kept
  gap junctions on eLN-PN contact pairs, c0 in {0.01, 0.02, 0.04} (sim/gap.py)
Base as C8: brain only, monoamines removed, w x 0.581, rate APL k = 0.1.
Trial: 0-1.0 s stimulus at 150 Hz, silence to 1.6 s; hot + vinegar, seeds 0, 1.
New specificity metric: mean rate of PNs of the stimulated glomeruli
(vinegar DM1, DM4; hot VP2) vs all other PNs, during 0.5-1.0 s.
Output: results/malecns/w2d3_gap_runs.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.lif_numpy import run_trial  # noqa: E402
from sim.lesion import select  # noqa: E402
from sim.malecns import load  # noqa: E402
from sim.gap import build_gap, pn_glomeruli  # noqa: E402
from scripts.ckpt2_ab import stim_idx  # noqa: E402
from scripts.c4_core_gain import Runner, DT, T_END, BIN, OUT  # noqa: E402
from scripts.ckpt5_stageA import ACH_LN, PN  # noqa: E402

TARGET = {'vinegar': {'DM1', 'DM4'}, 'hot': {'VP2'}}


def cut_ln_pn(W, ln, pn):
    C = W.tocoo(copy=True)
    keep = ~(ln[C.col] & pn[C.row])
    return sparse.csc_matrix((C.data[keep], (C.row[keep], C.col[keep])), shape=W.shape)


def main():
    R = Runner()
    ln, pn = select(R.meta, ACH_LN), select(R.meta, PN)
    glom = R.meta.type.map(pn_glomeruli)
    targ = {w: pn & glom.map(lambda s: bool(s & g)).to_numpy() for w, g in TARGET.items()}
    print('target PNs:', {w: int(m.sum()) for w, m in targ.items()}, flush=True)
    _, Wsyn = load(w_syn=1.0, monoamine='removed')     # synapse counts for contact sizes
    Wcut = cut_ln_pn(R.Wr, ln, pn)
    conds = [('base', R.Wr, None), ('LN->PN chem 0', Wcut, None)]
    for c0 in [0.01, 0.02, 0.04]:
        gap, info = build_gap(Wsyn, ln, pn, c0)
        print(f'gap c0={c0}: {info}', flush=True)
        conds.append((f'gap c0={c0}', Wcut, gap))
    rows = []
    for name, W, gap in conds:
        for word in ['hot', 'vinegar']:
            for seed in [0, 1]:
                r = run_trial(W, stim_idx(R.meta, R.pos, word), params=R.p, seed=seed, apl=R.apl,
                              t_stim=1.0, record=True, gap=gap)
                ts, si = r['t'] * DT, r['i']
                def cnt(lo, hi):
                    m = (ts >= lo) & (ts < hi)
                    return np.bincount(si[m], minlength=len(R.meta))
                odor_c, last = cnt(0.5, 1.0), cnt(T_END - 0.1, T_END)
                b = r['t'] // int(round(BIN / DT))
                nact = np.array([len(np.unique(si[b == k])) for k in range(int(round(T_END / BIN)))])
                ref = nact[int(0.9 / BIN):int(1.0 / BIN)].mean()
                below = np.flatnonzero(nact[int(1.0 / BIN):] < 0.1 * max(ref, 1))
                tg, other = targ[word], pn & ~targ[word]
                x = dict(cond=name, word=word, seed=seed,
                         odor_AL=int((odor_c[R.al] > 0).sum()),
                         target_PN_Hz=float(odor_c[tg].mean() / 0.5),
                         other_PN_Hz=float(odor_c[other].mean() / 0.5),
                         other_PN_active=int((odor_c[other] > 0).sum()),
                         eLN_odor_Hz=float(odor_c[ln].mean() / 0.5),
                         mn9_odor_Hz=float(odor_c[R.mn9[0]] / 0.5),
                         tail_brain=int((last > 0).sum()), tail_AL=int((last[R.al] > 0).sum()),
                         lLN1_bc_tail_Hz=float(last[R.lln1].mean() / 0.1),
                         decay_ms=float(below[0] * 5) if len(below) else np.nan,
                         persistent=bool((last > 0).sum() > 20))
                rows.append(x)
                print(f'[{name}] {word:8s} s{seed}: AL={x["odor_AL"]:4d} targetPN={x["target_PN_Hz"]:5.1f} '
                      f'otherPN={x["other_PN_Hz"]:5.1f}Hz ({x["other_PN_active"]} active) '
                      f'eLN={x["eLN_odor_Hz"]:5.1f} MN9={x["mn9_odor_Hz"]:4.0f} | tail={x["tail_brain"]:5d} '
                      f'lLN1_bc={x["lLN1_bc_tail_Hz"]:5.1f} decay={x["decay_ms"]} pers={x["persistent"]}',
                      flush=True)
                pd.DataFrame(rows).to_csv(OUT / 'w2d3_gap_runs.csv', index=False)
    print('DONE')


if __name__ == '__main__':
    main()
