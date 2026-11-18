#!/usr/bin/env python
"""W4D4: 10-episode pilot for the brain -> motor regression dataset.

Builds 10 stimulus-mixture specs (stimuli/mixtures.sample_episode, spec seeds
SPEC_SEED0 + k) and 2 background-only specs (1 Hz and 5 Hz, the ends of the
background range). Runs every spec with cfg='calib' (real) and cfg='shuffle'
(degree-preserving control), both with sim.cns.W4_CALIB, the same spec and the
same Poisson seed, on CPU float32 rng='torch', 2 s each, one process.
Episodes 0-9 are mixtures, 10-11 background only.

Usage:
  python scripts/w4d4_pilot.py                 # full pilot (both cfgs)
  python scripts/w4d4_pilot.py calib           # one cfg
  python scripts/w4d4_pilot.py check <k> <cfg> # re-run episode k, compare to the saved spikes
  python scripts/w4d4_pilot.py bgoff [cfg]     # H1 test: 5 Hz background 0-1.5 s, then off to 2.0 s;
                                               # logs active non-sensory neurons per 100 ms
Output (results/w4/regression/pilot/):
  ep{k:03d}_{cfg}.npz  t (step, dt 0.1 ms), i (neuron), spec (JSON string)
  bgoff_{cfg}.npz      the bgoff run (same fields)
  index_{cfg}.csv      one row per episode: words, bg, seed, n spikes, wall s
  pilot_log.txt        printed log (appended)
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
from sim.cns import cns_model, W4_CALIB  # noqa: E402
from sim.lif_numpy import P  # noqa: E402
from sim import lif_torch  # noqa: E402
from stimuli.mixtures import sample_episode, background_episode, to_blocks  # noqa: E402

OUT = ROOT / 'results' / 'w4' / 'regression' / 'pilot'
SPEC_SEED0 = 4040
N_MIX = 10
BG_ONLY = [(1.0, 9001), (5.0, 9002)]      # (bg Hz, Poisson seed)
T = 2.0


def specs():
    # legacy=True: the Day 4 sampler and uniform background (stimuli/mixtures.py, Day 5 change)
    out = [sample_episode(SPEC_SEED0 + k, T=T, legacy=True) for k in range(N_MIX)]
    out += [background_episode(bg, seed, T=T, legacy=True) for bg, seed in BG_ONLY]
    return out


def log(msg):
    print(msg, flush=True)
    with open(OUT / 'pilot_log.txt', 'a') as f:
        f.write(msg + '\n')


def run_spec(m, pos, spec):
    p = dict(P); p['t_run'] = spec['T']
    t0 = time.time()
    r = lif_torch.run_trial(m['W'], [], params=p, seed=spec['seed'], apl=m['apl'], gap=m['gap'],
                            presyn=m['presyn'], blocks=to_blocks(spec, m['meta'], pos), record=True,
                            device='cpu', rng='torch', dtype=torch.float32)
    return r['t'].astype(np.int32), r['i'].astype(np.int32), time.time() - t0


def build(cfg):
    t0 = time.time()
    m = cns_model('cns', cfg, **W4_CALIB)
    pos = {b: i for i, b in enumerate(m['meta'].bodyId)}
    log(f'[{cfg}] model built in {time.time() - t0:.1f} s, torch threads {torch.get_num_threads()}')
    return m, pos


def run_cfg(cfg, sp):
    m, pos = build(cfg)
    rows = []
    for k, spec in enumerate(sp):
        ts, si, wall = run_spec(m, pos, spec)
        np.savez_compressed(OUT / f'ep{k:03d}_{cfg}.npz', t=ts, i=si, spec=json.dumps(spec))
        words = '+'.join(w['word'] for w in spec['words']) or '(background)'
        rows.append(dict(ep=k, cfg=cfg, kind='mix' if spec['words'] else 'bg_only', words=words,
                         n_words=len(spec['words']), bg_hz=spec['bg_hz'], seed=spec['seed'],
                         T=spec['T'], n_spikes=len(ts), wall_s=round(wall, 2),
                         wall_per_bio_s=round(wall / spec['T'], 2), file=f'ep{k:03d}_{cfg}.npz'))
        log(f'[{cfg}] ep{k:03d} {words} bg {spec["bg_hz"]:.2f} Hz: {len(ts):,} spikes, '
            f'{wall:.1f} s wall ({wall / spec["T"]:.1f} s per bio-s)')
        pd.DataFrame(rows).to_csv(OUT / f'index_{cfg}.csv', index=False)


def check(k, cfg):
    d = np.load(OUT / f'ep{int(k):03d}_{cfg}.npz')
    spec = json.loads(str(d['spec']))
    m, pos = build(cfg)
    ts, si, wall = run_spec(m, pos, spec)
    same = len(ts) == len(d['t']) and np.array_equal(ts, d['t']) and np.array_equal(si, d['i'])
    log(f'[check] ep{int(k):03d}_{cfg}: saved {len(d["t"]):,} spikes, re-run {len(ts):,}, '
        f'identical={same} ({wall:.1f} s wall)')
    return same


def bgoff(cfg='calib', bg=5.0, t_off=1.5, seed=9003):
    """Background on [0, t_off), nothing after: does non-sensory activity outlast the drive?"""
    from sim.presyn import sensory_mask_malecns
    m, pos = build(cfg)
    sens = sensory_mask_malecns(m['meta'])
    spec = background_episode(bg, seed, T=T, legacy=True)
    p = dict(P); p['t_run'] = T
    t0 = time.time()
    r = lif_torch.run_trial(m['W'], [], params=p, seed=seed, apl=m['apl'], gap=m['gap'],
                            presyn=m['presyn'], blocks=[(0.0, t_off, np.flatnonzero(sens), bg)],
                            record=True, device='cpu', rng='torch', dtype=torch.float32)
    ts, si = r['t'].astype(np.int32), r['i'].astype(np.int32)
    spec['bg_off'] = t_off
    np.savez_compressed(OUT / f'bgoff_{cfg}.npz', t=ts, i=si, spec=json.dumps(spec))
    k = ~sens[si]
    act = [len(np.unique(si[k & (ts // 1000 == b)])) for b in range(int(round(T / 0.1)))]
    log(f'[bgoff] {cfg} bg {bg:g} Hz on [0, {t_off}) s, off to {T} s ({time.time() - t0:.1f} s wall). '
        f'Active non-sensory per 100 ms: {act}')


def main(*args):
    OUT.mkdir(parents=True, exist_ok=True)
    if args and args[0] == 'check':
        sys.exit(0 if check(args[1], args[2]) else 1)
    if args and args[0] == 'bgoff':
        return bgoff(*args[1:])
    sp = specs()
    with open(OUT / 'specs.json', 'w') as f:
        json.dump(sp, f, indent=1)
    for cfg in (args or ('calib', 'shuffle')):
        run_cfg(cfg, sp)


if __name__ == '__main__':
    main(*sys.argv[1:])
