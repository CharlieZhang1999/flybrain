#!/usr/bin/env python
"""C8: gains on the recurrent edges inside and between the LN modules.

Only LN->LN edges are scaled; LN->PN, LN->GABA, sensory input stay at 1.0.
  g1   lLN1_bc -> lLN1_bc              (29 core lLN1_bc)
  g2   otherLN -> otherLN              (51 other cholinergic core LNs)
  g12  lLN1_bc <-> otherLN, both directions
Base: Brian2-faithful simulator, brain-only, monoamines removed, w x 0.581,
rate APL k = 0.1, unclear signs as before (+1). Scaling for the whole trial.
Trial: odor 0-1.0 s, silence 1.0-1.6 s; hot + vinegar x seeds 0, 1.

Step 1  1-D sweeps of g1, g12, g2 at 0.9 ... 0.5 (others 1.0), plus baseline
Step 2  refine any transition (0.05 then 0.025 brackets)
Step 3  2-D (g1, g12) map: the 8 points from Stage_after_C5_C6.md
Step 4  only if nothing in steps 1-3 is stable: floor check, all three gains
        together at 0.25 and 0 (are LN->LN edges sufficient at all?)
Persistent run: > 20 brain neurons firing in the last 100 ms.
Outputs: results/malecns/c8_runs.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.lif_numpy import run_trial  # noqa: E402
from scripts.ckpt2_ab import stim_idx  # noqa: E402
from scripts.c4_core_gain import Runner, DT, T_END, BIN, OUT  # noqa: E402
from scripts.c5_gain_decomp import core_groups  # noqa: E402

ONE_D = [0.9, 0.8, 0.7, 0.6, 0.5]
TWO_D = [(0.75, 1.0), (0.5, 1.0), (1.0, 0.75), (1.0, 0.5), (0.75, 0.75), (0.5, 0.75), (0.75, 0.5)]


def module_gains(W, a, b, g1, g2, g12):
    """Scale edges pre->post: a->a by g1, b->b by g2, a<->b by g12."""
    C = W.tocoo(copy=True)
    pa, pb, qa, qb = a[C.col], b[C.col], a[C.row], b[C.row]
    f = np.ones(len(C.data))
    f[pa & qa] = g1
    f[pb & qb] = g2
    f[(pa & qb) | (pb & qa)] = g12
    return sparse.csc_matrix((C.data * f, (C.row, C.col)), shape=W.shape)


def main():
    R = Runner()
    G = core_groups(R)
    a, b = G['lLN1_bc'], G['otherLN']
    rows, done = [], {}

    def run(g1, g2, g12, step):
        key = (round(g1, 3), round(g2, 3), round(g12, 3))
        if key in done:
            return done[key]
        W = R.Wr if key == (1.0, 1.0, 1.0) else module_gains(R.Wr, a, b, *key)
        n_pers = 0
        for w in ['hot', 'vinegar']:
            for seed in [0, 1]:
                r = run_trial(W, stim_idx(R.meta, R.pos, w), params=R.p, seed=seed, apl=R.apl,
                              t_stim=1.0, record=True)
                ts, si = r['t'] * DT, r['i']
                def cnt(lo, hi):
                    m = (ts >= lo) & (ts < hi)
                    return np.bincount(si[m], minlength=len(R.meta))
                odor_c, last = cnt(0.5, 1.0), cnt(T_END - 0.1, T_END)
                bb = r['t'] // int(round(BIN / DT))
                act = np.array([len(np.unique(si[bb == k])) for k in range(int(round(T_END / BIN)))])
                ref = act[int(0.9 / BIN):int(1.0 / BIN)].mean()
                below = np.flatnonzero(act[int(1.0 / BIN):] < 0.1 * max(ref, 1))
                x = dict(step=step, g1=key[0], g2=key[1], g12=key[2], word=w, seed=seed,
                         odor_AL=int((odor_c[R.al] > 0).sum()), mn9_odor_Hz=float(odor_c[R.mn9[0]] / 0.5),
                         lLN1_bc_odor_Hz=float(odor_c[a].mean() / 0.5),
                         tail_brain=int((last > 0).sum()), tail_AL=int((last[R.al] > 0).sum()),
                         lLN1_bc_tail_Hz=float(last[a].mean() / 0.1), otherLN_tail_Hz=float(last[b].mean() / 0.1),
                         decay_ms=float(below[0] * 5) if len(below) else np.nan,
                         persistent=bool((last > 0).sum() > 20))
                rows.append(x); n_pers += x['persistent']
                print(f'[{step}] g1={key[0]:.3f} g2={key[1]:.3f} g12={key[2]:.3f} {w:8s} s{seed}: '
                      f'odor AL={x["odor_AL"]:4d} MN9={x["mn9_odor_Hz"]:5.1f} | tail brain={x["tail_brain"]:5d} '
                      f'lLN1_bc={x["lLN1_bc_tail_Hz"]:5.1f} other={x["otherLN_tail_Hz"]:5.1f} Hz '
                      f'decay={x["decay_ms"]}', flush=True)
        pd.DataFrame(rows).to_csv(OUT / 'c8_runs.csv', index=False)
        done[key] = n_pers
        return n_pers

    run(1.0, 1.0, 1.0, 'base')
    stable_any = False
    for name in ['g1', 'g12', 'g2']:
        prev = 1.0
        for v in ONE_D:
            g = dict(g1=1.0, g2=1.0, g12=1.0); g[name] = v
            n = run(g['g1'], g['g2'], g['g12'], f'1D-{name}')
            if n == 0:
                stable_any = True
                for step in (0.05, 0.025):            # refine between v (stable) and prev (persistent)
                    lo, hi = v, prev
                    for u in np.arange(lo + step, hi - 1e-9, step):
                        g[name] = round(u, 3)
                        if run(g['g1'], g['g2'], g['g12'], f'refine-{name}') == 0:
                            lo = round(u, 3)
                        else:
                            hi = round(u, 3); break
                    v, prev = lo, hi
                print(f'== {name}: transition between {v} (stable) and {prev} (persistent)', flush=True)
                break
            prev = v
        else:
            print(f'== {name}: persistent down to 0.5', flush=True)

    for g1, g12 in TWO_D:
        if run(g1, 1.0, g12, '2D') == 0:
            stable_any = True

    if not stable_any:
        print('== no stable point in steps 1-3; floor check', flush=True)
        for v in [0.25, 0.0]:
            run(v, v, v, 'floor')
    print('DONE')


if __name__ == '__main__':
    main()
