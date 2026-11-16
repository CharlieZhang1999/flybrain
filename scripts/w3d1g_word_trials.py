#!/usr/bin/env python
"""Week 3: isolated word trials on calib + presynaptic modulation (as w2d4: drive
0-0.5 s at 150 Hz, silence to 1.0 s, 17 words x 10 seeds), for the readout.
Usage: w3d1g_word_trials.py <K> <word_start> <word_end> [calib|shuffle] [mirror]
(tag suffix: m = right-JO mirror, s = shuffled)
Output: results/loop/w3d1g_stim_presyn<K>_<a>_<b>.npz (+ _info.npz)"""
import sys
import numpy as np
from scipy import sparse
from w3d1_common import presyn_model, ROOT, DT
from sim.lif_numpy import run_trial
from scripts.ckpt2_ab import stim_idx
from stimuli.malecns_words import WORDS

K, a, b = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
CFG = sys.argv[4] if len(sys.argv) > 4 else 'calib'          # calib | shuffle
MIRROR = len(sys.argv) > 5 and sys.argv[5] == 'mirror'
R, W, gap, ps, _ = presyn_model(K, CFG, mirror=MIRROR)
p = dict(R.p); p['t_run'] = 1.0
n = len(R.meta); words = list(WORDS)
rows, labels, seeds = [], [], []
for wi in range(a, b):
    for seed in range(10):
        r = run_trial(W, stim_idx(R.meta, R.pos, words[wi]), params=p, seed=seed, apl=R.apl,
                      t_stim=0.5, record=True, gap=gap, presyn=ps)
        ts, si = r['t'] * DT, r['i']
        rows.append(sparse.csr_matrix(np.bincount(si[ts < 0.5], minlength=n).astype(np.int32)))
        labels.append(wi); seeds.append(seed)
        print(f'{words[wi]} s{seed}: active={rows[-1].nnz} post={len(np.unique(si[ts >= 0.5]))}', flush=True)
tag = f'presyn{K}{"m" if MIRROR else ""}{"s" if CFG == "shuffle" else ""}_{a}_{b}'
sparse.save_npz(ROOT / 'results' / 'loop' / f'w3d1g_stim_{tag}.npz', sparse.vstack(rows).tocsr())
np.savez(ROOT / 'results' / 'loop' / f'w3d1g_info_{tag}.npz', labels=np.array(labels), seeds=np.array(seeds))
print('DONE')
