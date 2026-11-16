#!/usr/bin/env python
"""Week 3 Day 1b: 5-word episodes in one continuous simulation (calibrated MaleCNS).

Each episode: 5 words drawn at random from the 17 (no repeats), 500 ms of 150 Hz
drive per word, `gap` ms of silence between words, 500 ms of silence at the end.
The same 40 word sequences are used for every spacing (paired comparison);
Poisson seed = 1000 + episode index, so no episode reuses a Week 2 trial seed.
Saved per episode: spike counts per neuron in every word window and every gap
window, and population spikes in 10 ms bins.
Usage: w3d1_episodes.py <gap_ms> <ep_start> <ep_end> [config]
       w3d1_episodes.py none 0 10      (no-drive trials for the 'none' class)
Output: results/loop/w3d1_ep_<gap>_<a>_<b>.npz
"""
import sys
import time
import numpy as np
from scipy import sparse
from w3d1_common import calib_model, ROOT, DT
from sim.lif_numpy import run_trial
from scripts.ckpt2_ab import stim_idx
from stimuli.malecns_words import WORDS

N_EP, N_W, T_W, RATE, T_TAIL, BIN = 40, 5, 0.5, 150.0, 0.5, 0.01
OUT = ROOT / 'results' / 'loop'


def sequences():
    rng = np.random.default_rng(2026)
    words = list(WORDS)
    return [[words[k] for k in rng.choice(len(words), N_W, replace=False)] for _ in range(N_EP)]


def windows(gap):
    """[(kind, t0, t1)] for word windows ('w') and gap windows ('g')."""
    out, t = [], 0.0
    for k in range(N_W):
        out.append(('w', t, t + T_W)); t += T_W
        if k < N_W - 1 and gap > 0:
            out.append(('g', t, t + gap)); t += gap
    out.append(('tail', t, t + T_TAIL))
    return out, t + T_TAIL


def main(gap_arg, a, b, config='calib'):
    ps = None
    if config.startswith('presyn'):          # e.g. presyn0.01 = calib + presynaptic modulation, K = 0.01
        from w3d1_common import presyn_model
        R, W, gap_m, ps, _ = presyn_model(float(config[6:]))
    else:
        R, W, gap_m = calib_model(config)
    n = len(R.meta)
    rows, pops, info = [], [], []
    if gap_arg == 'none':
        for seed in range(a, b):
            p = dict(R.p); p['t_run'] = 1.0
            r = run_trial(W, [], params=p, seed=seed, apl=R.apl, record=True, gap=gap_m, presyn=ps)
            ts, si = r['t'] * DT, r['i']
            rows.append(sparse.csr_matrix(np.bincount(si[ts < T_W], minlength=n).astype(np.int32)))
            info.append(('none', seed, 0, 'w', 'none', ''))
            print(f'none s{seed}: spikes={len(ts)}', flush=True)
        tag = f'none_{a}_{b}' + ('' if config == 'calib' else f'_{config}')
    else:
        gap = int(gap_arg) / 1000.0
        seqs = sequences()
        win, t_run = windows(gap)
        p = dict(R.p); p['t_run'] = t_run
        for ep in range(a, b):
            t0 = time.time()
            seq = seqs[ep]
            wins = [w for w in win if w[0] == 'w']
            blocks = [(w[1], w[2], stim_idx(R.meta, R.pos, word), RATE) for w, word in zip(wins, seq)]
            r = run_trial(W, [], params=p, seed=1000 + ep, apl=R.apl, record=True, gap=gap_m,
                          blocks=blocks, presyn=ps)
            ts, si = r['t'] * DT, r['i']
            k, prev = 0, ''
            for kind, w0, w1 in win:
                m = (ts >= w0 - 1e-9) & (ts < w1 - 1e-9)
                rows.append(sparse.csr_matrix(np.bincount(si[m], minlength=n).astype(np.int32)))
                if kind == 'w':
                    info.append((config, ep, k, 'w', seq[k], prev)); prev = seq[k]; k += 1
                else:
                    info.append((config, ep, k, kind, 'none', prev))
            nb = int(round(t_run / BIN))
            pops.append(np.bincount((ts / BIN).astype(int), minlength=nb)[:nb])
            print(f'gap {gap_arg} ep{ep} {"+".join(seq)}: spikes={len(ts)} '
                  f'tail active={rows[-1].nnz} ({time.time()-t0:.0f} s)', flush=True)
        tag = f'{gap_arg}_{a}_{b}' + ('' if config == 'calib' else f'_{config}')
    sparse.save_npz(OUT / f'w3d1_ep_{tag}_X.npz', sparse.vstack(rows).tocsr())
    np.savez_compressed(OUT / f'w3d1_ep_{tag}_info.npz', info=np.array(info, dtype=object),
                        pop=np.array(pops, dtype=object))
    print('DONE', flush=True)


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args[0], int(args[1]), int(args[2]), *(args[3:4] or []))
