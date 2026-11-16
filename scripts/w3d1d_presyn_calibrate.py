#!/usr/bin/env python
"""Week 3 presyn step 1: regression and K calibration (MaleCNS calibrated).
a) presyn with K = inf must equal 'no central input onto sensory' (rule A).
b) K sweep on vinegar (0-0.5 s): mean release gain of the stimulated ORNs
   (target ~0.5: Olsen & Wilson 2008 report roughly half to 85% suppression).
Output: results/loop/w3d1d_calibrate.csv"""
import numpy as np
import pandas as pd
from scipy import sparse
from w3d1_common import calib_model, ROOT, DT
from sim.lif_numpy import run_trial
from sim.presyn import build_presyn, sensory_mask_malecns
from scripts.ckpt2_ab import stim_idx

R, W, gap = calib_model()
sens = sensory_mask_malecns(R.meta)
p = dict(R.p); p['t_run'] = 0.5
rows = []
for word in ['vinegar', 'cold', 'sugar']:
    idx = stim_idx(R.meta, R.pos, word)
    for K in [np.inf, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3]:
        W2, ps, info = build_presyn(W, sens, K)
        r = run_trial(W2, idx, params=p, seed=0, apl=R.apl, t_stim=0.5, record=True, gap=gap, presyn=ps)
        pos_in = {g: k for k, g in enumerate(ps['idx'])}
        gs = r['presyn_mean_gain'][[pos_in[i] for i in idx]]
        rows.append(dict(word=word, K=K, spikes=len(r['t']), active=len(np.unique(r['i'])),
                         stim_gain_mean=float(gs.mean()), stim_gain_min=float(gs.min())))
        print(rows[-1], flush=True)
    if word == 'vinegar':
        print('presyn info', info)
pd.DataFrame(rows).to_csv(ROOT / 'results' / 'loop' / 'w3d1d_calibrate.csv', index=False)
