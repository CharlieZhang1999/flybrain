#!/usr/bin/env python
"""Checkpoint 2: 2x2 A/B test on MaleCNS.

Arms: monoamine {excitatory (Shiu), removed} x APL {spiking, rate}.
Step 1 (calibrate): for each monoamine setting, sweep the rate-APL gain k on
odor words and pick the k whose KC active fraction is closest to 7.5%
(imaging target 5-10%, Honegger et al. 2011; Lin et al. 2014).
Step 2 (grid): run all words in all 4 arms, 1 trial x 1 s, seed 0.
Outputs: results/malecns/ckpt2_calibration.csv, ckpt2_grid.csv
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.malecns import load  # noqa: E402
from sim.lif_numpy import run_trial, P  # noqa: E402
from stimuli.malecns_words import WORDS, select_ids  # noqa: E402

OUT = ROOT / 'results' / 'malecns'
OUT.mkdir(parents=True, exist_ok=True)
ODORS = ['vinegar', 'fruit', 'co2']
K_GRID = [1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]
TARGET = 0.075


def make_apl(meta, W):
    """Remove spiking APL from W; return (W_without_APL, apl config)."""
    is_kc = meta.type.fillna('').str.startswith('KC').to_numpy()
    apl = np.flatnonzero(meta.type.values == 'APL')
    kc_idx = np.flatnonzero(is_kc)
    # per-KC weight = summed |APL -> KC| weight across both APLs (volts/event)
    w_kc = -np.asarray(W[kc_idx][:, apl].sum(axis=1)).ravel()
    keep = np.ones(W.shape[0]); keep[apl] = 0
    D = sparse.diags(keep)
    W2 = (D @ W @ D).tocsc()          # APL in and out removed
    W2.eliminate_zeros()
    return W2, kc_idx, w_kc, is_kc


def stim_idx(meta, pos, word):
    return np.array([pos[b] for w in word.split('+') for b in select_ids(meta, w)])


def score(c, is_kc, mn9):
    a = c > 0
    return dict(active=int(a.sum()), kc_frac=float(a[is_kc].mean()),
                mn9_L=int(c[mn9[0]]), mn9_R=int(c[mn9[1]])), a


def main():
    cal_rows, grid_rows, k_best = [], [], {}
    for mono in ['excitatory', 'removed']:
        meta, W = load(monoamine=mono)
        pos = {b: i for i, b in enumerate(meta.bodyId)}
        mn9 = np.flatnonzero(meta.type.values == 'MN9')
        W_rate, kc_idx, w_kc, is_kc = make_apl(meta, W)

        # ---- step 1: calibrate k on odors (0.5 s trials) ----
        P_half = dict(P); P_half['t_run'] = 0.5
        best = None
        for k in K_GRID:
            fr = []
            for w in ODORS:
                c = run_trial(W_rate, stim_idx(meta, pos, w), params=P_half, seed=0,
                              apl=dict(kc_idx=kc_idx, w_kc=w_kc, k=k, tau=0.01))
                fr.append((c > 0)[is_kc].mean())
            m = float(np.mean(fr))
            cal_rows.append(dict(monoamine=mono, k=k, kc_frac_odor=m))
            print(f'[cal] mono={mono:10s} k={k:.0e} odor KC frac={m:.3f}', flush=True)
            if best is None or abs(m - TARGET) < abs(best[1] - TARGET):
                best = (k, m)
        k_best[mono] = best[0]
        pd.DataFrame(cal_rows).to_csv(OUT / 'ckpt2_calibration.csv', index=False)

        # ---- step 2: all words, both APL modes ----
        for apl_mode in ['spiking', 'rate']:
            acts = {}
            for w in list(WORDS) + ['sugar+bitter']:
                t0 = time.time()
                if apl_mode == 'spiking':
                    c = run_trial(W, stim_idx(meta, pos, w), seed=0)
                else:
                    c = run_trial(W_rate, stim_idx(meta, pos, w), seed=0,
                                  apl=dict(kc_idx=kc_idx, w_kc=w_kc,
                                           k=k_best[mono], tau=0.01))
                s, a = score(c, is_kc, mn9)
                acts[w] = a
                grid_rows.append(dict(monoamine=mono, apl=apl_mode, word=w, **s,
                                      k=k_best[mono] if apl_mode == 'rate' else None))
                print(f'[grid] {mono:10s} {apl_mode:7s} {w:13s} active={s["active"]:6d} '
                      f'KC={s["kc_frac"]:.3f} MN9={s["mn9_L"]}/{s["mn9_R"]} '
                      f'({time.time()-t0:.0f}s)', flush=True)
            ws = list(WORDS)
            J = [(acts[x] & acts[y]).sum() / max(1, (acts[x] | acts[y]).sum())
                 for i, x in enumerate(ws) for y in ws[i + 1:]]
            grid_rows.append(dict(monoamine=mono, apl=apl_mode, word='_mean_jaccard',
                                  kc_frac=float(np.mean(J))))
            pd.DataFrame(grid_rows).to_csv(OUT / 'ckpt2_grid.csv', index=False)
    json.dump(k_best, open(OUT / 'ckpt2_k_best.json', 'w'))
    print('DONE', k_best)


if __name__ == '__main__':
    main()
