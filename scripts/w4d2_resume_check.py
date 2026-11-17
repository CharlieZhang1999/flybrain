#!/usr/bin/env python
"""W4D2 (science track): save/resume in the PyTorch simulator.

Full CNS (calibrated), blocks: looming 0-0.3 s, MDN 0.3-0.5 s, run to 0.7 s.
One uninterrupted run vs three chained segments (0-0.2, 0.2-0.45, 0.45-0.7 s,
splits inside both blocks). Spikes and binned motor-pool counts must match
exactly, for rng='numpy' (parity RNG) and rng='torch' (device RNG).
Usage: python scripts/w4d2_resume_check.py [device=cpu]
Output: results/w4/sci/w4d2_resume_check.txt
"""
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.cns import cns_model, motor_pools  # noqa: E402
from sim.lif_numpy import P  # noqa: E402
from sim import lif_torch  # noqa: E402
from scripts.ckpt2_ab import stim_idx  # noqa: E402

SPLITS = [0.2, 0.45, 0.7]


def main(device='cpu'):
    m = cns_model('cns', 'calib')
    meta = m['meta']
    pos = {b: i for i, b in enumerate(meta.bodyId)}
    pools, _ = motor_pools(meta)
    groups = list(pools.values())
    mdn = np.flatnonzero(meta.type.astype(str).values == 'MDN')
    bl = [(0.0, 0.3, stim_idx(meta, pos, 'looming'), 150.0), (0.3, 0.5, mdn, 150.0)]
    kw = dict(seed=3, apl=m['apl'], gap=m['gap'], presyn=m['presyn'], blocks=bl,
              record=True, device=device, bin_steps=100, bin_groups=groups)
    lines = []
    for rng in ['numpy', 'torch']:
        p = dict(P); p['t_run'] = SPLITS[-1]
        t0 = time.time()
        full = lif_torch.run_trial(m['W'], [], params=p, rng=rng, **kw)
        w_full = time.time() - t0
        st, ts, is_, bins = None, [], [], []
        t0 = time.time()
        for t_end in SPLITS:
            p = dict(P); p['t_run'] = t_end
            r = lif_torch.run_trial(m['W'], [], params=p, rng=rng, state=st, return_state=True, **kw)
            st = r['state']; ts.append(r['t']); is_.append(r['i']); bins.append(r['binned'])
        w_split = time.time() - t0
        t = np.concatenate(ts); i = np.concatenate(is_); b = np.concatenate(bins)
        same = (np.array_equal(full['t'], t) and np.array_equal(full['i'], i)
                and np.array_equal(full['binned'], b) and np.array_equal(full['counts'], r['counts']))
        s = (f'rng={rng:5s} spikes full {len(full["t"]):,} chained {len(t):,} | '
             f'MN pool bins {full["binned"].shape} | identical={same} | '
             f'wall {w_full:.0f}s / {w_split:.0f}s')
        print(s, flush=True); lines.append(s)
    out = ROOT / 'results' / 'w4' / 'sci'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'w4d2_resume_check.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:])
