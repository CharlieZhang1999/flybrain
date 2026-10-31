#!/usr/bin/env python
"""C4: core gain sweep. Scale the outputs of the confirmed cholinergic members
of the 104-neuron causal core (C1/C2) by alpha; everything else unchanged.

Base: Brian2-faithful simulator, brain-only, monoamines removed, w x 0.581,
rate APL k = 0.1. Unclear-transmitter signs left as assigned (C3).
Trial: odor 0-1.0 s, silence 1.0-1.6 s. Scaling applied for the whole trial.

Step 1  coarse alpha in {1.0, 0.9, 0.8, 0.7, 0.6, 0.5}, hot + vinegar x seeds 0,1
Step 2  refine in 0.025 steps between the lowest persistent and highest stable alpha
Step 3  at alpha* (largest alpha with 0/4 persistent runs): all 18 inputs, seed 0,
        plus no-stimulus; compared with alpha = 1.0
Persistent run: > 20 brain neurons firing in the last 100 ms.
Decay time: ms after odor offset until brain activity per 5-ms bin falls below
10% of its level in the last 100 ms of the odor (NaN if it never does).
Outputs: results/malecns/c4_sweep.csv, c4_inputs.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.malecns import load  # noqa: E402
from sim.lif_numpy import run_trial, P  # noqa: E402
from sim.lesion import select  # noqa: E402
from scripts.ckpt2_ab import make_apl, stim_idx  # noqa: E402
from scripts.ckpt3_sweep_ablate import regions, mask_weights  # noqa: E402
from scripts.ckpt5_stageA import INPUTS, ACH_LN, LN, PN, SCALE, K_APL  # noqa: E402
from scripts.c3_sign_audit import with_sign  # noqa: E402,F401

OUT = ROOT / 'results' / 'malecns'
DT, T_END, BIN = P['dt'], 1.6, 0.005
COARSE = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


def scale_outputs(W, mask, alpha):
    C = W.tocoo(copy=True)
    data = C.data.copy(); data[mask[C.col]] *= alpha
    from scipy import sparse
    return sparse.csc_matrix((data, (C.row, C.col)), shape=W.shape)


class Runner:
    def __init__(self):
        meta, W1 = load(w_syn=1.0, monoamine='removed')
        vnc, _ = regions(meta)
        W = mask_weights(W1 * (P['w_syn'] * SCALE), lambda r, c: ~vnc[r] & ~vnc[c])
        self.Wr, kc, wkc, _ = make_apl(meta, W)
        self.apl = dict(kc_idx=kc, w_kc=wkc, k=K_APL, tau=0.01)
        self.meta, self.pos = meta, {b: i for i, b in enumerate(meta.bodyId)}
        core = pd.read_csv(OUT / 'c1_nodes_lpcut_core.csv')
        fvs = core[(core.fvs_order >= 0) & (core.nt == 'acetylcholine')]
        self.core = np.zeros(len(meta), bool)
        self.core[[self.pos[b] for b in fvs.bodyId]] = True
        self.al = select(meta, f'{LN} or {PN}')
        self.lln1 = select(meta, "type == 'lLN1_bc'")
        self.mn9 = np.flatnonzero(meta.type.values == 'MN9')
        self.p = dict(P); self.p['t_run'] = T_END
        print(f'scaled group: {self.core.sum()} cholinergic core neurons', flush=True)

    def trial(self, alpha, word, seed):
        W = self.Wr if alpha == 1.0 else scale_outputs(self.Wr, self.core, alpha)
        exc = np.array([], int) if word == 'none' else stim_idx(self.meta, self.pos, word)
        r = run_trial(W, exc, params=self.p, seed=seed, apl=self.apl, t_stim=1.0, record=True)
        ts, si = r['t'] * DT, r['i']
        def cnt(a, b):
            m = (ts >= a) & (ts < b)
            return np.bincount(si[m], minlength=len(self.meta))
        odor_c, last = cnt(0.5, 1.0), cnt(T_END - 0.1, T_END)
        b = (r['t'] // int(BIN / DT))
        nb = int(T_END / BIN)
        act = np.array([len(np.unique(si[b == k])) for k in range(nb)])
        ref = act[int(0.9 / BIN):int(1.0 / BIN)].mean()
        post = act[int(1.0 / BIN):]
        below = np.flatnonzero(post < 0.1 * max(ref, 1))
        return dict(alpha=alpha, word=word, seed=seed,
                    odor_brain=int((odor_c > 0).sum()), odor_AL=int((odor_c[self.al] > 0).sum()),
                    mn9_odor_Hz=float(odor_c[self.mn9[0]] / 0.5),
                    tail_brain=int((last > 0).sum()), tail_AL=int((last[self.al] > 0).sum()),
                    core_tail_Hz=float(last[self.core].mean() / 0.1),
                    lLN1_bc_odor_Hz=float(odor_c[self.lln1].mean() / 0.5),
                    decay_ms=float(below[0] * 5) if len(below) else np.nan,
                    persistent=bool((last > 0).sum() > 20))


def main():
    R = Runner()
    rows = []
    def sweep(alphas):
        for a in alphas:
            for odor in ['hot', 'vinegar']:
                for s in [0, 1]:
                    x = R.trial(a, odor, s); rows.append(x)
                    print(f'a={a:.3f} {odor:8s} s{s}: odor AL={x["odor_AL"]:4d} MN9={x["mn9_odor_Hz"]:5.1f} '
                          f'| tail brain={x["tail_brain"]:5d} core={x["core_tail_Hz"]:6.1f}Hz '
                          f'decay={x["decay_ms"]} ms', flush=True)
            pd.DataFrame(rows).to_csv(OUT / 'c4_sweep.csv', index=False)
    sweep(COARSE)
    df = pd.DataFrame(rows)
    per = df.groupby('alpha').persistent.sum()
    stable = sorted(a for a, n in per.items() if n == 0)
    if not stable:
        print('No stable alpha down to 0.5; stopping before refinement.', flush=True)
        return
    hi_stable = max(stable)
    above = sorted(a for a in per.index if a > hi_stable)
    if above:
        fine = [round(x, 3) for x in np.arange(hi_stable + 0.025, above[0] - 1e-9, 0.025)]
        sweep(fine)
    df = pd.DataFrame(rows)
    per = df.groupby('alpha').persistent.sum()
    a_star = max(a for a, n in per.items() if n == 0)
    print(f'== alpha* (largest alpha with 0/4 persistent runs) = {a_star}', flush=True)

    inp = []
    for a in [1.0, a_star]:
        for w in INPUTS + ['none']:
            x = R.trial(a, w, 0); inp.append(x)
            print(f'[inputs a={a}] {w:13s} odor brain={x["odor_brain"]:5d} AL={x["odor_AL"]:4d} '
                  f'MN9={x["mn9_odor_Hz"]:5.1f} tail={x["tail_brain"]:5d}', flush=True)
        pd.DataFrame(inp).to_csv(OUT / 'c4_inputs.csv', index=False)
    print('DONE')


if __name__ == '__main__':
    main()
