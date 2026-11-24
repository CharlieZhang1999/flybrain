#!/usr/bin/env python
"""Week 4 T5b, step 1: measure spike_cap / syn_cap for sim/lif_torch_fast.py.

Runs the reference (sim/lif_torch.run_trial) on the bundle, looming stimulus,
rng='torch', float64, record=True, one full trial (P['t_run'] = 1.0 s, 10,000
steps) per seed, and logs per step:
  spikes      number of neurons that spike in the step
  syn_W       synapses gathered from W when those spikes are delivered
              (sum of the column degrees of W over the spiking neurons)
  syn_Wp      the same for the presyn matrix Wp
A step's spikes are delivered dly steps later as one gather, so the maximum of
syn_W / syn_Wp over steps is the largest gather of the trial.

Caps = ceil(max over seeds x 1.5). spike_cap uses max spikes; syn_cap uses the
max of syn_W (W and Wp share syn_cap in lif_torch_fast; Wp is gathered with
min(syn_cap, its own degree bound for spike_cap columns)).

Usage (repo root):
  python bench/w4_caps.py --bundle ~/flybench/flybrain/results/w4/cns_bundle_calib.npz
Output: perf/results/w4/<host>_<timestamp>_caps.json
"""
import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'bench'))
from sim import lif_torch  # noqa: E402
from sim.lif_numpy import P  # noqa: E402
import w4_overhead as t1  # noqa: E402  (bundle loader)

MARGIN = 1.5


def per_step(model, t, i, nsteps):
    degW = np.diff(model['W'].tocsc().indptr)
    degWp = np.diff(model['presyn']['Wp'].tocsc().indptr)
    spikes = np.bincount(t, minlength=nsteps)
    synW = np.bincount(t, weights=degW[i], minlength=nsteps).astype(np.int64)
    synWp = np.bincount(t, weights=degWp[i], minlength=nsteps).astype(np.int64)
    return spikes, synW, synWp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', default=str(ROOT / 'results' / 'w4' / 'cns_bundle_calib.npz'))
    ap.add_argument('--stim', default='looming')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--t-run', type=float, default=P['t_run'])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default=str(ROOT / 'perf' / 'results' / 'w4'))
    a = ap.parse_args()
    model = t1.load_bundle(a.bundle)
    p = dict(P); p['t_run'] = a.t_run
    nsteps = int(round(a.t_run / P['dt']))
    rows = []
    for seed in a.seeds:
        t0 = time.perf_counter()
        r = lif_torch.run_trial(model['W'], model['stim'][a.stim], params=p, seed=seed,
                                apl=model['apl'], gap=model['gap'], presyn=model['presyn'],
                                device=a.device, dtype=torch.float64, rng='torch', record=True)
        wall = time.perf_counter() - t0
        sp, sw, swp = per_step(model, np.asarray(r['t'], np.int64), np.asarray(r['i'], np.int64), nsteps)
        row = dict(seed=seed, steps=nsteps, wall_s=round(wall, 2), spikes_total=int(sp.sum()),
                   max_spikes=int(sp.max()), argmax_spikes_step=int(sp.argmax()),
                   max_syn_W=int(sw.max()), argmax_syn_W_step=int(sw.argmax()),
                   max_syn_Wp=int(swp.max()), argmax_syn_Wp_step=int(swp.argmax()),
                   p99_spikes=float(np.percentile(sp, 99)), mean_spikes=round(float(sp.mean()), 2),
                   p99_syn_W=float(np.percentile(sw, 99)), mean_syn_W=round(float(sw.mean()), 1))
        rows.append(row)
        print(json.dumps(row), flush=True)
    mx = {k: max(r[k] for r in rows) for k in ('max_spikes', 'max_syn_W', 'max_syn_Wp')}
    caps = dict(spike_cap=math.ceil(mx['max_spikes'] * MARGIN),
                syn_cap=math.ceil(mx['max_syn_W'] * MARGIN),
                syn_cap_Wp_needed=math.ceil(mx['max_syn_Wp'] * MARGIN))
    tag = f'{socket.gethostname()}_{time.strftime("%Y%m%d-%H%M")}'
    res = dict(tag=tag, env=dict(torch=torch.__version__,
                                 gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None),
               args=vars(a), margin=MARGIN, per_seed=rows, max_over_seeds=mx, caps=caps)
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f'{tag}_caps.json').write_text(json.dumps(res, indent=1))
    print('caps', json.dumps(caps), flush=True)
    print(f'saved {outdir / tag}_caps.json', flush=True)


if __name__ == '__main__':
    main()
