#!/usr/bin/env python
"""Week 3 presyn step 2: MaleCNS regression on all 17 words + none.
Configs: calib (Week 2), ruleA (no central input onto sensory, K = inf),
presyn (K given). 0.5 s drive at 150 Hz, silence to 2.0 s, seed 0.
Reports active during stimulus, tail (neurons firing 1.5-2.0 s), MN9 rate,
target vs other PN rates for vinegar/hot.
Usage: w3d1e_presyn_words.py <config> [K]
Output: results/loop/w3d1e_words_<config>.csv"""
import sys
import numpy as np
import pandas as pd
from w3d1_common import calib_model, ROOT, DT
from sim.lif_numpy import run_trial
from sim.lesion import select
from sim.gap import pn_glomeruli
from sim.presyn import build_presyn, sensory_mask_malecns
from scripts.ckpt2_ab import stim_idx
from scripts.ckpt5_stageA import PN
from scripts.w2d3_gap_malecns import TARGET
from stimuli.malecns_words import WORDS

cfg = sys.argv[1]; K = float(sys.argv[2]) if len(sys.argv) > 2 else np.inf
R, W, gap = calib_model()
ps = None
if cfg != 'calib':
    W, ps, info = build_presyn(W, sensory_mask_malecns(R.meta), K); print(info, flush=True)
m = R.meta; n = len(m)
pn = select(m, PN); glom = m.type.map(pn_glomeruli)
targ = {w: pn & glom.map(lambda s: bool(s & g)).to_numpy() for w, g in TARGET.items()}
p = dict(R.p); p['t_run'] = 2.0
rows = []
for word in list(WORDS) + ['sugar+bitter', 'none']:
    idx = np.array([], int) if word == 'none' else stim_idx(m, R.pos, word)
    r = run_trial(W, idx, params=p, seed=0, apl=R.apl, t_stim=0.5, record=True, gap=gap, presyn=ps)
    ts, si = r['t'] * DT, r['i']
    c_st = np.bincount(si[ts < 0.5], minlength=n); c_tail = np.bincount(si[ts >= 1.5], minlength=n)
    c_late = np.bincount(si[(ts >= 0.25) & (ts < 0.5)], minlength=n)
    tg = targ.get(word)
    rows.append(dict(config=cfg, K=K, word=word, active_stim=int((c_st > 0).sum()),
                     tail=int((c_tail > 0).sum()), mn9_Hz=float(c_st[R.mn9].mean() / 0.5),
                     target_PN_Hz=float(c_late[tg].mean() / 0.25) if tg is not None else np.nan,
                     other_PN_Hz=float(c_late[pn & ~tg].mean() / 0.25) if tg is not None else np.nan))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(ROOT / 'results' / 'loop' / f'w3d1e_words_{cfg}{"" if cfg=="calib" else "_K"+str(K)}.csv', index=False)
print('DONE')
