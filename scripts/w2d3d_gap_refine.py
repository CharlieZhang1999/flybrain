#!/usr/bin/env python
"""Week 2: refine the gap-junction calibration so eLNs keep responding.

At c0 = 0.2 every eLN couples to ~210 PNs (sum c ~ 9), which shunts it silent.
Variants (all with chemical cholinergic LN->PN = 0 and 2 mV spikelets):
  cap1   : c0 = 0.2, total coupling per neuron capped at 1
  cap2   : c0 = 0.2, cap 2
  strong : c0 = 0.2, junctions only on contacts with >= 50 synapses (no cap)
Usage: w2d3d_gap_refine.py <variant>   (runs hot, vinegar x seeds 0, 1)
Output: results/malecns/w2d3d_<variant>.csv
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
from sim.gap import build_gap, pn_glomeruli  # noqa: E402
from scripts.ckpt2_ab import stim_idx  # noqa: E402
from scripts.c4_core_gain import Runner, DT, T_END, OUT  # noqa: E402
from scripts.ckpt5_stageA import ACH_LN, PN  # noqa: E402
from scripts.w2d3_gap_malecns import cut_ln_pn, TARGET  # noqa: E402

VARIANTS = {'cap1': dict(c0=0.2, cap=1.0), 'cap2': dict(c0=0.2, cap=2.0),
            'strong': dict(c0=0.2, min_syn=50)}


def main(variant):
    R = Runner()
    ln, pn = select(R.meta, ACH_LN), select(R.meta, PN)
    glom = R.meta.type.map(pn_glomeruli)
    targ = {w: pn & glom.map(lambda s: bool(s & g)).to_numpy() for w, g in TARGET.items()}
    _, Wsyn = load(w_syn=1.0, monoamine='removed')
    Wcut = cut_ln_pn(R.Wr, ln, pn)
    gap, info = build_gap(Wsyn, ln, pn, v_spike=0.002, **VARIANTS[variant])
    print(variant, info, flush=True)
    rows = []
    for word in ['hot', 'vinegar']:
        for seed in [0, 1]:
            r = run_trial(Wcut, stim_idx(R.meta, R.pos, word), params=R.p, seed=seed, apl=R.apl,
                          t_stim=1.0, record=True, gap=gap)
            ts, si = r['t'] * DT, r['i']
            def cnt(lo, hi):
                m = (ts >= lo) & (ts < hi)
                return np.bincount(si[m], minlength=len(R.meta))
            od, last = cnt(0.5, 1.0), cnt(T_END - 0.1, T_END)
            tg = targ[word]; other = pn & ~tg
            x = dict(variant=variant, word=word, seed=seed, odor_AL=int((od[R.al] > 0).sum()),
                     target_PN_Hz=float(od[tg].mean() / 0.5), other_PN_Hz=float(od[other].mean() / 0.5),
                     eLN_odor_Hz=float(od[ln].mean() / 0.5), eLN_active=int((od[ln] > 0).sum()),
                     tail_brain=int((last > 0).sum()), persistent=bool((last > 0).sum() > 20))
            rows.append(x); print(x, flush=True)
            pd.DataFrame(rows).to_csv(OUT / f'w2d3d_{variant}.csv', index=False)
    print('DONE')


if __name__ == '__main__':
    main(sys.argv[1])
