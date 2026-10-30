#!/usr/bin/env python
"""Checkpoint 2b: w_syn / 1.43 rescale A/B test on MaleCNS, rate-unit APL.

Arms: monoamine {excitatory (Shiu), removed}, both with rate APL and
w_syn = 0.275 mV / 1.43 (the MaleCNS/FAFB synapse-density ratio used by
gianlucamazza/flymsg).
Per arm: recalibrate APL gain k on odors (target 7.5% KC active), then run
all words: 1.0 s stimulus + 0.5 s silence; the tail window (last 0.3 s)
tests whether activity dies out after the stimulus.
Outputs: results/malecns/ckpt2b_calibration.csv, ckpt2b_grid.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.malecns import load  # noqa: E402
from sim.lif_numpy import run_trial, P  # noqa: E402
from stimuli.malecns_words import WORDS  # noqa: E402
from scripts.ckpt2_ab import make_apl, stim_idx, ODORS, TARGET  # noqa: E402

OUT = ROOT / 'results' / 'malecns'
SCALE = 1 / 1.43
K_GRID = [3e-3, 1e-2, 3e-2, 1e-1, 3e-1]


def main():
    cal, grid = [], []
    for mono in ['excitatory', 'removed']:
        meta, W = load(w_syn=P['w_syn'] * SCALE, monoamine=mono)
        pos = {b: i for i, b in enumerate(meta.bodyId)}
        mn9 = np.flatnonzero(meta.type.values == 'MN9')
        W_rate, kc_idx, w_kc, is_kc = make_apl(meta, W)

        P_half = dict(P); P_half['t_run'] = 0.5
        best = None
        for k in K_GRID:
            fr = [(run_trial(W_rate, stim_idx(meta, pos, w), params=P_half, seed=0,
                             apl=dict(kc_idx=kc_idx, w_kc=w_kc, k=k, tau=0.01)) > 0)[is_kc].mean()
                  for w in ODORS]
            m = float(np.mean(fr))
            cal.append(dict(monoamine=mono, k=k, kc_frac_odor=m))
            print(f'[cal] {mono:10s} k={k:.0e} odor KC frac={m:.3f}', flush=True)
            if best is None or abs(m - TARGET) < abs(best[1] - TARGET):
                best = (k, m)
        pd.DataFrame(cal).to_csv(OUT / 'ckpt2b_calibration.csv', index=False)
        apl = dict(kc_idx=kc_idx, w_kc=w_kc, k=best[0], tau=0.01)

        P_run = dict(P); P_run['t_run'] = 1.5
        acts = {}
        for w in list(WORDS) + ['sugar+bitter']:
            t0 = time.time()
            c, tail = run_trial(W_rate, stim_idx(meta, pos, w), params=P_run, seed=0,
                                apl=apl, t_stim=1.0, tail_from=1.2)
            a = c > 0
            acts[w] = a
            row = dict(monoamine=mono, word=w, k=best[0], active=int(a.sum()),
                       kc_frac=float(a[is_kc].mean()),
                       mn9_L=int(c[mn9[0]]), mn9_R=int(c[mn9[1]]),
                       tail_active=int((tail > 0).sum()))
            grid.append(row)
            print(f'[grid] {mono:10s} {w:13s} active={row["active"]:6d} KC={row["kc_frac"]:.3f} '
                  f'MN9={row["mn9_L"]}/{row["mn9_R"]} tail={row["tail_active"]} '
                  f'({time.time()-t0:.0f}s)', flush=True)
        ws = list(WORDS)
        J = [(acts[x] & acts[y]).sum() / max(1, (acts[x] | acts[y]).sum())
             for i, x in enumerate(ws) for y in ws[i + 1:]]
        grid.append(dict(monoamine=mono, word='_mean_jaccard', kc_frac=float(np.mean(J))))
        pd.DataFrame(grid).to_csv(OUT / 'ckpt2b_grid.csv', index=False)
    print('DONE')


if __name__ == '__main__':
    main()
