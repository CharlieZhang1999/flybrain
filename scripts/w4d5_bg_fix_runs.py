#!/usr/bin/env python
"""W4D5 (T2b): verification runs for the class-specific background (stimuli/mixtures.py).

Background-only episodes at 3 Hz, 1.0 s, seeds 0-2, cfg calib and shuffle
(cns_model('cns', cfg, **W4_CALIB)), CPU float32 rng='torch', blocks from
stimuli.mixtures.to_blocks(background_episode(...)):
  legacy    background_episode(3, seed, legacy=True): all 15,912 sensory at 3 Hz
            (= the diagnosis run dose_<cfg>_all_3Hz_s<seed>)
  abdSame   class_v1, bristle 0 Hz, abdominal 'same' (3 Hz)   [the new default]
  abd0      class_v1, bristle 0 Hz, abdominal 0 Hz
Runs one at a time, in one process.

Usage:  python scripts/w4d5_bg_fix_runs.py [calib|shuffle ...]     (default: calib shuffle)
        python scripts/w4d5_bg_fix_runs.py table                   (table from saved runs)
Output: results/w4/bg_fix/runs/<cfg>_<scheme>_3Hz_s<seed>.npz (t step, i neuron, spec JSON)
        results/w4/bg_fix/runs_log.txt, results/w4/bg_fix/verify.csv
Metrics: MNs active (>= 1 spike in 1 s), MN spikes, non-sensory active (>= 1 spike in 1 s), and
  non-sensory / VNC non-sensory / brain non-sensory active per 100 ms (mean of bins 0.1-1.0 s,
  as in the diagnosis), bristle spikes, driven-sensory rate.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.cns import cns_model, regions, brain_mask, W4_CALIB  # noqa: E402
from sim.lif_numpy import P  # noqa: E402
from sim import lif_torch  # noqa: E402
from stimuli.mixtures import background_episode, to_blocks, bg_class_masks  # noqa: E402

OUT = ROOT / 'results' / 'w4' / 'bg_fix'
RUNS = OUT / 'runs'
T, RATE, SEEDS = 1.0, 3.0, (0, 1, 2)
SCHEMES = {'legacy': dict(legacy=True), 'abdSame': dict(abdominal_bg='same'), 'abd0': dict(abdominal_bg=0.0)}
BIN = 1000                                   # steps per 100 ms


def log(msg):
    print(msg, flush=True)
    with open(OUT / 'runs_log.txt', 'a') as f:
        f.write(msg + '\n')


def groups(meta):
    vnc, _ = regions(meta)
    mk = bg_class_masks(meta)
    sens = mk['sensory']
    mn = (meta.superclass.astype(str) == 'vnc_motor').to_numpy()
    return dict(sens=sens, mn=mn, ns=~sens, vnc_ns=vnc & ~sens & ~mn, brain_ns=brain_mask(meta, vnc),
                bristle=mk['bristle'], abdominal=mk['abdominal'])


def metrics(g, t, i, driven):
    n = len(g['mn'])
    c = np.bincount(i, minlength=n)
    act = c > 0
    b = t // BIN

    def per100(mask):
        k = mask[i]
        return float(np.mean([len(np.unique(i[k & (b == j)])) for j in range(1, int(round(T / 0.1)))]))
    return dict(mn_active=int((act & g['mn']).sum()), mn_spikes=int(c[g['mn']].sum()),
                ns_active=int((act & g['ns']).sum()), vnc_ns_active=int((act & g['vnc_ns']).sum()),
                brain_ns_active=int((act & g['brain_ns']).sum()),
                ns_per100=round(per100(g['ns']), 1), vnc_ns_per100=round(per100(g['vnc_ns']), 1),
                brain_ns_per100=round(per100(g['brain_ns']), 1),
                n_driven=int(driven.sum()), driven_rate_hz=round(c[driven].sum() / max(driven.sum(), 1) / T, 3),
                bristle_spikes=int(c[g['bristle']].sum()), abdominal_spikes=int(c[g['abdominal']].sum()))


def run_cfg(cfg):
    t0 = time.time()
    m = cns_model('cns', cfg, verbose=False, **W4_CALIB)
    meta = m['meta']
    pos = {b: k for k, b in enumerate(meta.bodyId)}
    g = groups(meta)
    log(f'[{cfg}] model built in {time.time() - t0:.1f} s, torch threads {torch.get_num_threads()}')
    for sch, kw in SCHEMES.items():
        for s in SEEDS:
            name = f'{cfg}_{sch}_3Hz_s{s}'
            if (RUNS / f'{name}.npz').exists():
                log(f'[skip] {name} exists'); continue
            spec = background_episode(RATE, s, T=T, **kw)
            blocks = to_blocks(spec, meta, pos)
            driven = np.zeros(len(meta), bool)
            for b in blocks:
                driven[b[2]] = True
            p = dict(P); p['t_run'] = T
            t1 = time.time()
            r = lif_torch.run_trial(m['W'], [], params=p, seed=s, apl=m['apl'], gap=m['gap'],
                                    presyn=m['presyn'], blocks=blocks, record=True, device='cpu',
                                    rng='torch', dtype=torch.float32)
            wall = time.time() - t1
            ti, ii = r['t'].astype(np.int32), r['i'].astype(np.int32)
            np.savez_compressed(RUNS / f'{name}.npz', t=ti, i=ii, spec=json.dumps(dict(spec, cfg=cfg)))
            mt = metrics(g, ti, ii, driven)
            log(f'{name}: blocks {[(len(b[2]), b[3]) for b in blocks]}, '
                + ', '.join(f'{k}={v}' for k, v in mt.items()) + f', wall={wall:.1f}s')


def table():
    from sim.malecns import load
    meta, _ = load(w_syn=1.0, monoamine='removed')
    g = groups(meta)
    rows = []
    for f in sorted(RUNS.glob('*.npz')):
        d = np.load(f)
        spec = json.loads(str(d['spec']))
        cfg, sch, _, s = f.stem.split('_')
        driven = g['sens'].copy()
        if sch != 'legacy':
            driven &= ~g['bristle']
            if sch == 'abd0':
                driven &= ~g['abdominal']
        rows.append(dict(cfg=cfg, scheme=sch, seed=int(s[1:]), **metrics(g, d['t'], d['i'], driven)))
    df = pd.DataFrame(rows).sort_values(['cfg', 'scheme', 'seed'])
    df.to_csv(OUT / 'verify.csv', index=False)
    print(df.to_string(index=False))
    return df


if __name__ == '__main__':
    args = sys.argv[1:] or ['calib', 'shuffle']
    RUNS.mkdir(parents=True, exist_ok=True)
    if args == ['table']:
        table()
    else:
        for cfg in args:
            run_cfg(cfg)
        table()
