#!/usr/bin/env python
"""Week 2 Day 3: all 17 words + sugar+bitter + none on MaleCNS with the
gap-junction calibration (chemical cholinergic LN->PN = 0, eLN-PN gap
junctions, charge-matched 2 mV spikelets), c0 in {0.1, 0.2}, plus base.
Seed 0. Output: results/malecns/w2d3c_allwords.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.lif_numpy import run_trial  # noqa: E402
from sim.lesion import select  # noqa: E402
from sim.malecns import load  # noqa: E402
from sim.gap import build_gap  # noqa: E402
from scripts.ckpt2_ab import stim_idx  # noqa: E402
from scripts.c4_core_gain import Runner, DT, T_END, OUT  # noqa: E402
from scripts.ckpt5_stageA import ACH_LN, PN, INPUTS  # noqa: E402
from scripts.w2d3_gap_malecns import cut_ln_pn  # noqa: E402


def main():
    R = Runner()
    ln, pn = select(R.meta, ACH_LN), select(R.meta, PN)
    _, Wsyn = load(w_syn=1.0, monoamine='removed')
    Wcut = cut_ln_pn(R.Wr, ln, pn)
    conds = [('base', R.Wr, None)]
    for c0 in [0.2, 0.1]:
        gap, _ = build_gap(Wsyn, ln, pn, c0); gap['v_spike'] = 0.002
        conds.append((f'gap c0={c0}', Wcut, gap))
    rows = []
    for name, W, gap in conds:
        for word in INPUTS + ['none']:
            exc = np.array([], int) if word == 'none' else stim_idx(R.meta, R.pos, word)
            r = run_trial(W, exc, params=R.p, seed=0, apl=R.apl, t_stim=1.0, record=True, gap=gap)
            ts, si = r['t'] * DT, r['i']
            def cnt(lo, hi):
                m = (ts >= lo) & (ts < hi)
                return np.bincount(si[m], minlength=len(R.meta))
            st, last = cnt(0.0, 1.0), cnt(T_END - 0.1, T_END)
            x = dict(cond=name, word=word, stim_active=int((st > 0).sum()),
                     stim_AL=int((st[R.al] > 0).sum()), mn9_Hz=float(st[R.mn9[0]] / 1.0),
                     tail_brain=int((last > 0).sum()), tail_AL=int((last[R.al] > 0).sum()),
                     persistent=bool((last > 0).sum() > 20))
            rows.append(x)
            print(f'[{name}] {word:13s} active={x["stim_active"]:6d} AL={x["stim_AL"]:4d} '
                  f'MN9={x["mn9_Hz"]:5.1f} | tail={x["tail_brain"]:5d} AL={x["tail_AL"]:4d} '
                  f'pers={x["persistent"]}', flush=True)
            pd.DataFrame(rows).to_csv(OUT / 'w2d3c_allwords.csv', index=False)
    print('DONE')


if __name__ == '__main__':
    main()
