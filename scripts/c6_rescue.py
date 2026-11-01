#!/usr/bin/env python
"""C6: rescue (add-back) from the stable condition found in C5.

C5: silencing the outputs of all 104 core neurons at 1.3 s extinguishes the
persistent state (brain 3,350 -> 0-2 neurons); no subset short of that does.
C6 starts from that silenced condition and restores one component at a time
to beta x its normal output strength; the other core neurons stay silenced.
A causal gain mechanism should bring persistence back once enough recurrent
excitation is restored.

Base: Brian2-faithful simulator, brain-only, monoamines removed, w x 0.581,
rate APL k = 0.1. Trial: odor 0-1.0 s, silence 1.0-1.6 s; perturbation from
t = 1.3 s. Restored components: lLN1_bc, otherLN, PN, unclear, and the pairs
lLN1_bc+otherLN, lLN1_bc+unclear, otherLN+unclear, ach91 (= all but unclear).
beta in {0.5, 1.0}; hot + vinegar, seed 0. Plus 'none' (all 104 silenced).
Outputs: results/malecns/c6_rescue.csv
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
from scripts.c5_gain_decomp import core_groups, T_SW  # noqa: E402

BETAS = [0.5, 1.0]


def scale_by(W, s):
    """Return a copy of W with the outputs of neuron j multiplied by s[j]."""
    C = W.tocoo(copy=True)
    return sparse.csc_matrix((C.data * s[C.col], (C.row, C.col)), shape=W.shape)


def rescue_trial(R, all104, name, restored, beta, word, seed):
    s = np.ones(len(R.meta)); s[all104] = 0.0; s[restored] = beta
    r = run_trial(R.Wr, stim_idx(R.meta, R.pos, word), params=R.p, seed=seed, apl=R.apl,
                  t_stim=1.0, w_switch=(T_SW, scale_by(R.Wr, s)), record=True)
    ts, si = r['t'] * DT, r['i']
    def cnt(a, b):
        m = (ts >= a) & (ts < b)
        return np.bincount(si[m], minlength=len(R.meta))
    pre, last = cnt(T_SW - 0.1, T_SW), cnt(T_END - 0.1, T_END)
    b = r['t'] // int(round(BIN / DT))
    act = np.array([len(np.unique(si[b == k])) for k in range(int(round(T_END / BIN)))])
    sb = int(round(T_SW / BIN))
    return dict(restored=name, n=int(restored.sum()), beta=beta, word=word, seed=seed,
                brain_pre=int((pre > 0).sum()), min_bin_after=int(act[sb:].min()),
                tail_brain=int((last > 0).sum()), tail_AL=int((last[R.al] > 0).sum()),
                restored_tail_Hz=float(last[restored].mean() / 0.1) if restored.any() else 0.0,
                core_tail_Hz=float(last[R.core].mean() / 0.1),
                persistent=bool((last > 0).sum() > 20))


def main():
    R = Runner()
    G = core_groups(R)
    all104 = G['all104']
    comps = {'none': np.zeros(len(R.meta), bool)}
    for k in ['lLN1_bc', 'otherLN', 'PN', 'unclear', 'lLN1_bc+otherLN']:
        comps[k] = G[k]
    comps['lLN1_bc+unclear'] = G['lLN1_bc'] | G['unclear']
    comps['otherLN+unclear'] = G['otherLN'] | G['unclear']
    comps['ach91'] = G['ach91']
    rows = []
    for name, m in comps.items():
        for beta in ([1.0] if name == 'none' else BETAS):
            for w in ['hot', 'vinegar']:
                x = rescue_trial(R, all104, name, m, beta, w, 0); rows.append(x)
                print(f'{name:16s} n={x["n"]:3d} b={beta:.1f} {w:8s}: brain {x["brain_pre"]:5d} -> '
                      f'min {x["min_bin_after"]:4d}/bin -> end {x["tail_brain"]:5d}  AL={x["tail_AL"]:4d}  '
                      f'restored={x["restored_tail_Hz"]:5.1f} Hz  persistent={x["persistent"]}', flush=True)
        pd.DataFrame(rows).to_csv(OUT / 'c6_rescue.csv', index=False)
    print('DONE')


if __name__ == '__main__':
    main()
