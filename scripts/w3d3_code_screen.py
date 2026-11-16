#!/usr/bin/env python
"""Week 3 Day 3a: screen 8 candidate action codes on JO neurons (loop brain model:
calibrated MaleCNS + presynaptic modulation, K = 0.05). Each code: 500 ms of drive,
silence to 1.0 s, from rest, seeds 0-9.
Usage: w3d3_code_screen.py <code> [<code> ...]
Output: results/loop/w3d3_screen_<code>.npz (spike counts 0-0.5 s and 0.5-1.0 s)"""
import sys
from pathlib import Path
import numpy as np
from scipy import sparse
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from w3d1_common import presyn_model  # noqa: E402
from sim.lif_numpy import run_trial  # noqa: E402
from loop.actions import blocks  # noqa: E402

R, W, gap, ps, _ = presyn_model(0.05)
n = len(R.meta); p = dict(R.p); p['t_run'] = 1.0
for code in sys.argv[1:]:
    A, B = [], []
    for seed in range(10):
        r = run_trial(W, [], params=p, seed=seed, apl=R.apl, record=True, gap=gap, presyn=ps,
                      blocks=blocks(code, R.meta))
        ts, si = r['t'] * p['dt'], r['i']
        A.append(np.bincount(si[ts < 0.5], minlength=n)); B.append(np.bincount(si[ts >= 0.5], minlength=n))
        print(f'{code} s{seed}: active={int((A[-1] > 0).sum())} post={int((B[-1] > 0).sum())}', flush=True)
    np.savez_compressed(ROOT / 'results' / 'loop' / f'w3d3_screen_{code}.npz',
                        stim=sparse.csr_matrix(np.array(A)).toarray().astype(np.int32),
                        post=np.array(B, dtype=np.int32))
print('DONE')
