#!/usr/bin/env python
"""Week 3 Day 6: save/resume regression. Loop brain (presyn K=0.05 + right-JO mirror),
blocks vinegar 0-0.5 s, B-L code 0.5-1.0 s; one run to 1.2 s vs run to 0.6 s,
save, resume to 1.2 s. Spikes must match exactly."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from w3d1_common import presyn_model  # noqa: E402
from sim.lif_numpy import run_trial  # noqa: E402
from scripts.ckpt2_ab import stim_idx  # noqa: E402
from loop.actions import blocks as cb  # noqa: E402

R, W, gap, ps, _ = presyn_model(0.05, mirror=True)
bl = [(0.0, 0.5, stim_idx(R.meta, R.pos, 'vinegar'), 150.0)] + cb('B-L', R.meta, t0=0.5)
kw = dict(seed=5, apl=R.apl, record=True, gap=gap, presyn=ps, blocks=bl)
p = dict(R.p); p['t_run'] = 1.2
full = run_trial(W, [], params=p, **kw)
p1 = dict(R.p); p1['t_run'] = 0.6
a = run_trial(W, [], params=p1, return_state=True, **kw)
b = run_trial(W, [], params=p, state=a['state'], **kw)
t = np.r_[a['t'], b['t']]; i = np.r_[a['i'], b['i']]
print('spikes full', len(full['t']), 'split', len(t), 'identical', np.array_equal(full['t'], t) and np.array_equal(full['i'], i))
