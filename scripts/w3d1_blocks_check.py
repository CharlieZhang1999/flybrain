#!/usr/bin/env python
"""Week 3 Day 1a regression: one-block `blocks=` schedule must match the normal call
spike for spike (calibrated MaleCNS, vinegar, 0.5 s drive, 1.0 s run)."""
import time
import numpy as np
from w3d1_common import calib_model
from sim.lif_numpy import run_trial
from scripts.ckpt2_ab import stim_idx

R, W, gap = calib_model()
p = dict(R.p); p['t_run'] = 1.0
idx = stim_idx(R.meta, R.pos, 'vinegar')
t0 = time.time()
a = run_trial(W, idx, params=p, seed=3, apl=R.apl, t_stim=0.5, record=True, gap=gap)
t1 = time.time()
b = run_trial(W, [], params=p, seed=3, apl=R.apl, record=True, gap=gap,
              blocks=[(0.0, 0.5, idx, 150.0)])
t2 = time.time()
same = np.array_equal(a['t'], b['t']) and np.array_equal(a['i'], b['i'])
print(f'spikes normal={len(a["t"])} blocks={len(b["t"])} identical={same}')
print(f'time per 1 s trial: {t1-t0:.0f} s / {t2-t1:.0f} s')
