#!/usr/bin/env python
"""C5: gain decomposition of the causal core, as maintenance perturbations.

C4 scaled the 91 cholinergic core neurons for the whole trial and found no
stable alpha down to 0.5; C2 showed alpha = 0 (silencing, at 1.3 s) collapses
the state. C5 therefore first locates the maintenance transition, then splits
the core into groups and scales each one at that critical gain.

Base: Brian2-faithful simulator, brain-only, monoamines removed, w x 0.581,
rate APL k = 0.1. Trial: odor 0-1.0 s, silence 1.0-1.6 s. Outputs of the
scaled group multiplied by alpha from t = 1.3 s (state already established).

Step 1  all 91 cholinergic core neurons, alpha in {0.4, 0.3, 0.2, 0.1, 0.0},
        hot + vinegar x seeds 0, 1. alpha* = largest alpha with 0/4 persistent.
Step 2  groups (104-core members): lLN1_bc (29), other cholinergic LNs (51),
        cholinergic PNs (11), unclear (13), and the combinations
        lLN1_bc+otherLN, lLN1_bc+PN, otherLN+PN, all 91, all 104;
        alpha in {alpha*, 0}, hot + vinegar, seed 0.
Persistent run: > 20 brain neurons firing in the last 100 ms.
Collapse: ms after 1.3 s until brain activity per 5-ms bin < 10% of 1.2-1.3 s.
Outputs: results/malecns/c5_transition.csv, c5_groups.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.lif_numpy import run_trial  # noqa: E402
from scripts.ckpt2_ab import stim_idx  # noqa: E402
from scripts.c4_core_gain import Runner, scale_outputs, DT, T_END, BIN, OUT  # noqa: E402

T_SW = 1.3
STEP1 = [0.4, 0.3, 0.2, 0.1, 0.0]


def core_groups(R):
    core = pd.read_csv(OUT / 'c1_nodes_lpcut_core.csv')
    fvs = core[core.fvs_order >= 0]
    def mask(sel):
        m = np.zeros(len(R.meta), bool); m[[R.pos[b] for b in fvs.bodyId[sel]]] = True; return m
    ach = fvs.nt == 'acetylcholine'
    g = {'lLN1_bc': mask(fvs.type == 'lLN1_bc'),
         'otherLN': mask(ach & (fvs['class'] == 'ALLN') & (fvs.type != 'lLN1_bc')),
         'PN': mask(ach & (fvs['class'] == 'ALPN')),
         'unclear': mask(fvs.nt == 'unclear')}
    g['lLN1_bc+otherLN'] = g['lLN1_bc'] | g['otherLN']
    g['lLN1_bc+PN'] = g['lLN1_bc'] | g['PN']
    g['otherLN+PN'] = g['otherLN'] | g['PN']
    g['ach91'] = g['lLN1_bc'] | g['otherLN'] | g['PN']
    g['all104'] = g['ach91'] | g['unclear']
    assert g['ach91'].sum() == 91 and g['all104'].sum() == 104
    return g


def maint_trial(R, name, mask, alpha, word, seed):
    sw = (T_SW, scale_outputs(R.Wr, mask, alpha))
    r = run_trial(R.Wr, stim_idx(R.meta, R.pos, word), params=R.p, seed=seed, apl=R.apl,
                  t_stim=1.0, w_switch=sw, record=True)
    ts, si = r['t'] * DT, r['i']
    def cnt(a, b):
        m = (ts >= a) & (ts < b)
        return np.bincount(si[m], minlength=len(R.meta))
    pre, last = cnt(T_SW - 0.1, T_SW), cnt(T_END - 0.1, T_END)
    b = r['t'] // int(round(BIN / DT))
    act = np.array([len(np.unique(si[b == k])) for k in range(int(round(T_END / BIN)))])
    sb = int(round(T_SW / BIN))
    ref = act[sb - 20:sb].mean()
    below = np.flatnonzero(act[sb:] < 0.1 * max(ref, 1))
    return dict(group=name, n=int(mask.sum()), alpha=alpha, word=word, seed=seed,
                brain_pre=int((pre > 0).sum()), tail_brain=int((last > 0).sum()),
                tail_AL=int((last[R.al] > 0).sum()),
                core_pre_Hz=float(pre[R.core].mean() / 0.1), core_tail_Hz=float(last[R.core].mean() / 0.1),
                group_tail_Hz=float(last[mask].mean() / 0.1),
                collapse_ms=float(below[0] * 5) if len(below) else np.nan,
                persistent=bool((last > 0).sum() > 20))


def log(x):
    print(f'{x["group"]:16s} n={x["n"]:3d} a={x["alpha"]:.2f} {x["word"]:8s} s{x["seed"]}: '
          f'brain {x["brain_pre"]:5d} -> {x["tail_brain"]:5d}  AL={x["tail_AL"]:4d}  '
          f'core {x["core_pre_Hz"]:5.1f} -> {x["core_tail_Hz"]:5.1f} Hz  collapse={x["collapse_ms"]} ms',
          flush=True)


def main():
    R = Runner()
    G = core_groups(R)
    rows = []
    for a in STEP1:
        for w in ['hot', 'vinegar']:
            for s in [0, 1]:
                x = maint_trial(R, 'ach91', G['ach91'], a, w, s); rows.append(x); log(x)
        pd.DataFrame(rows).to_csv(OUT / 'c5_transition.csv', index=False)
    per = pd.DataFrame(rows).groupby('alpha').persistent.sum()
    stable = [a for a, n in per.items() if n == 0]
    a_star = max(stable) if stable else None
    print(f'== maintenance alpha* (ach91, 0/4 persistent) = {a_star}', flush=True)
    alphas = sorted({0.0, a_star} - {None}, reverse=True)

    grows = []
    for name, m in G.items():
        for a in alphas:
            for w in ['hot', 'vinegar']:
                x = maint_trial(R, name, m, a, w, 0); grows.append(x); log(x)
        pd.DataFrame(grows).to_csv(OUT / 'c5_groups.csv', index=False)
    print('DONE')


if __name__ == '__main__':
    main()
