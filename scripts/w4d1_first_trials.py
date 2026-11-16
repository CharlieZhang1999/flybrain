#!/usr/bin/env python
"""W4D1: first full-CNS trials. Does the VNC stay quiet at rest, stay stable
after a stimulus, and do known descending neurons reach the expected motor pools?

Conditions (drive 0-0.4 s at 150 Hz Poisson, run to 0.6 s):
  none, sugar, vinegar, looming       sensory words (Week 2 stimulus sets)
  DN:DNp01   giant fiber -> expected TTMn (jump) and DLM via PSI
  DN:MDN     moonwalker -> expected backward walking, leg MNs (hind legs)
  DN:DNg100  forward walking command (BDN2)
  DN:DNa02   steering, ipsilateral turning
Usage: python scripts/w4d1_first_trials.py [scope=cns] [config=calib] [device=cpu]
Output: results/w4/w4d1_first_<scope>_<config>.csv (per pool rates) and .txt (summary)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.cns import cns_model, motor_pools  # noqa: E402
from sim.lif_numpy import P  # noqa: E402
from sim import lif_torch  # noqa: E402
from scripts.ckpt2_ab import stim_idx  # noqa: E402

OUT = ROOT / 'results' / 'w4'
CONDS = ['none', 'sugar', 'vinegar', 'looming', 'DN:DNp01', 'DN:MDN', 'DN:DNg100', 'DN:DNa02']
T_STIM, T_RUN = 0.4, 0.6


def main(scope='cns', config='calib', device='cpu'):
    OUT.mkdir(parents=True, exist_ok=True)
    m = cns_model(scope, config)
    meta, vnc = m['meta'], m['vnc']
    pos = {b: i for i, b in enumerate(meta.bodyId)}
    pools, pdf = motor_pools(meta)
    p = dict(P); p['t_run'] = T_RUN
    txt = OUT / f'w4d1_first_{scope}_{config}.txt'
    rows = []
    with open(txt, 'w') as f:
        for c in CONDS:
            if c == 'none':
                exc = np.array([], int)
            elif c.startswith('DN:'):
                exc = np.flatnonzero(meta.type.astype(str).values == c[3:])
            else:
                exc = stim_idx(meta, pos, c)
            t0 = time.time()
            r = lif_torch.run_trial(m['W'], exc, params=p, seed=0, apl=m['apl'], gap=m['gap'],
                                    presyn=m['presyn'], t_stim=T_STIM, record=True,
                                    device=device, rng='numpy')
            wall = time.time() - t0
            ts, si = r['t'] * P['dt'], r['i']
            stim = np.bincount(si[ts < T_STIM], minlength=len(meta)) / T_STIM
            tail = np.bincount(si[ts >= T_STIM + 0.1], minlength=len(meta))
            act = stim > 0
            for name, ix in pools.items():
                rows.append(dict(cond=c, pool=name, rate_hz=stim[ix].mean(), tail_spikes=int(tail[ix].sum())))
            pr = pd.DataFrame([x for x in rows if x['cond'] == c]).merge(pdf, on='pool')
            grp = (pr.groupby(['subclass', 'group']).rate_hz.mean().sort_values(ascending=False))
            top = ', '.join(f'{s}/{g} {v:.1f}' for (s, g), v in grp.head(6).items() if v > 0) or 'none'
            s = (f'{c:10s} stim n={len(exc):4d} | active brain {int(act[~vnc].sum()):6d} '
                 f'VNC {int(act[vnc].sum()):5d} MN {int((act & meta.superclass.eq("vnc_motor").values).sum()):3d} '
                 f'| tail(0.5-0.6s) brain {int((tail[~vnc] > 0).sum()):5d} VNC {int((tail[vnc] > 0).sum()):5d} '
                 f'| {wall:.0f}s\n           top pool groups (Hz): {top}')
            print(s, flush=True); f.write(s + '\n')
    pd.DataFrame(rows).to_csv(OUT / f'w4d1_first_{scope}_{config}.csv', index=False)


if __name__ == '__main__':
    main(*sys.argv[1:])
