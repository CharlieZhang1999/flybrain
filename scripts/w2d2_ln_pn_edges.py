#!/usr/bin/env python
"""Week 2 Day 1, D2: cut the cholinergic LN <-> PN chemical edges on MaleCNS.

In the fly, excitatory (cholinergic) antennal-lobe LNs drive PNs mainly through
ShakB gap junctions, not nicotinic synapses (Yaksi & Wilson 2010; Huang 2010).
The model turns that coupling into chemical excitation. C8 left it intact.
Scaled edges (whole trial):
  LN->PN : pre cholinergic ALLN, post ALPN
  PN->LN : pre cholinergic ALPN, post cholinergic ALLN
Conditions: base (1.0), both at 0.5, both at 0, and 'eLN_off' = both at 0 plus
cholinergic LN -> cholinergic LN at 0 (no chemical coupling among eLNs and PNs).
Base: same as C8 (brain only, monoamines removed, w x 0.581, rate APL k = 0.1).
Trial: stimulus 0-1.0 s at 150 Hz, silence to 1.6 s. hot + vinegar, seeds 0, 1;
sound + looming seed 0 at base and 0. Persistent: > 20 brain neurons in last 100 ms.
Output: results/malecns/w2d2_runs.csv
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
from scripts.ckpt2_ab import stim_idx  # noqa: E402
from scripts.c4_core_gain import Runner, DT, T_END, BIN, OUT  # noqa: E402
from scripts.ckpt5_stageA import ACH_LN, PN  # noqa: E402

ACH_PN = "`class` == 'ALPN' and nt == 'acetylcholine'"


def scale(W, groups):
    """groups: list of (pre_mask, post_mask, factor)."""
    C = W.tocoo(copy=True)
    f = np.ones(len(C.data))
    for pre, post, g in groups:
        f[pre[C.col] & post[C.row]] = g
    return sparse.csc_matrix((C.data * f, (C.row, C.col)), shape=W.shape), f


def main():
    R = Runner()
    ln, pn, achpn = select(R.meta, ACH_LN), select(R.meta, PN), select(R.meta, ACH_PN)
    print(f'cholinergic LN={ln.sum()}, PN={pn.sum()} (cholinergic {achpn.sum()})', flush=True)
    conds = {
        'base': [],
        'LN<->PN 0.5': [(ln, pn, 0.5), (achpn, ln, 0.5)],
        'LN<->PN 0': [(ln, pn, 0.0), (achpn, ln, 0.0)],
        'eLN_off': [(ln, pn, 0.0), (achpn, ln, 0.0), (ln, ln, 0.0)],
    }
    rows = []
    for name, groups in conds.items():
        W, f = scale(R.Wr, groups) if groups else (R.Wr, None)
        if f is not None:
            print(f'[{name}] edges changed: {(f != 1).sum():,}', flush=True)
        trials = [(w, s) for w in ['hot', 'vinegar'] for s in [0, 1]]
        if name in ('base', 'LN<->PN 0'):
            trials += [('sound', 0), ('looming', 0)]
        for word, seed in trials:
            r = run_trial(W, stim_idx(R.meta, R.pos, word), params=R.p, seed=seed, apl=R.apl,
                          t_stim=1.0, record=True)
            ts, si = r['t'] * DT, r['i']
            def cnt(lo, hi):
                m = (ts >= lo) & (ts < hi)
                return np.bincount(si[m], minlength=len(R.meta))
            odor_c, last = cnt(0.5, 1.0), cnt(T_END - 0.1, T_END)
            b = r['t'] // int(round(BIN / DT))
            nact = np.array([len(np.unique(si[b == k])) for k in range(int(round(T_END / BIN)))])
            ref = nact[int(0.9 / BIN):int(1.0 / BIN)].mean()
            below = np.flatnonzero(nact[int(1.0 / BIN):] < 0.1 * max(ref, 1))
            x = dict(cond=name, word=word, seed=seed,
                     odor_AL=int((odor_c[R.al] > 0).sum()),
                     PN_odor_Hz=float(odor_c[pn].mean() / 0.5),
                     mn9_odor_Hz=float(odor_c[R.mn9[0]] / 0.5),
                     tail_brain=int((last > 0).sum()), tail_AL=int((last[R.al] > 0).sum()),
                     lLN1_bc_tail_Hz=float(last[R.lln1].mean() / 0.1),
                     PN_tail_Hz=float(last[pn].mean() / 0.1),
                     decay_ms=float(below[0] * 5) if len(below) else np.nan,
                     persistent=bool((last > 0).sum() > 20))
            rows.append(x)
            print(f'[{name}] {word:8s} s{seed}: odor AL={x["odor_AL"]:4d} PN={x["PN_odor_Hz"]:5.1f}Hz '
                  f'MN9={x["mn9_odor_Hz"]:5.1f} | tail brain={x["tail_brain"]:5d} AL={x["tail_AL"]:4d} '
                  f'lLN1_bc={x["lLN1_bc_tail_Hz"]:5.1f} PN={x["PN_tail_Hz"]:5.1f}Hz '
                  f'decay={x["decay_ms"]} persistent={x["persistent"]}', flush=True)
            pd.DataFrame(rows).to_csv(OUT / 'w2d2_runs.csv', index=False)
    print('DONE')


if __name__ == '__main__':
    main()
