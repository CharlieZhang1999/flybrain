#!/usr/bin/env python
"""W4D5: the brain -> motor regression training set (T2).

Episodes (all T = 2.5 s, background scheme class_v2, the stimuli/mixtures.py default):
  k = 0..199    mixtures: sample_episode(SPEC_SEED0 + k, T=2.5, t_words=2.0), i.e.
                2-4 words with distinct modalities, rate 30-150 Hz, onset U(0, 1.7) s,
                offset clipped to 2.0 s (a >= 0.5 s word-free tail), looming in ~1/6,
                bg_hz U(1, 3). Spec seeds 50000-50199 (the T1 pilot used 4040-4049).
  k = 200..209  background only (rest reference): bg_hz = linspace(1, 3, 10),
                Poisson seeds BG_SEED0 + j.
Spec k depends only on k (order-stable), so chunks can run in any order.
Every spec runs with cfg 'calib' (real) and 'shuffle' (degree-preserving control),
sim.cns.W4_CALIB, the same spec and Poisson seed, CPU float32 rng='torch'.

Resumable: an episode whose npz exists and is valid (readable, same spec, spike
count consistent) is skipped. Files are written to a temporary name and renamed.
Run only one simulation process at a time.

Usage:
  python scripts/w4d5_dataset.py <k_start> <k_end> [cfg ...]   # episodes k_start <= k < k_end
                                                               # cfgs default: calib shuffle
  python scripts/w4d5_dataset.py check <k> <cfg>               # re-run k, compare with saved spikes
  python scripts/w4d5_dataset.py index                         # rebuild index.csv only
Output (results/w4/regression/day5/):
  ep{k:03d}_{cfg}.npz  t (step, dt 0.1 ms, int32), i (neuron, int32), spec (JSON string),
                       cfg, n_spikes, wall_s
  specs.json           all 210 specs
  index.csv            one row per saved episode (rebuilt from the npz files)
  run_log.txt          printed log (appended), per-episode wall time
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stimuli.mixtures import sample_episode, background_episode  # noqa: E402

OUT = ROOT / 'results' / 'w4' / 'regression' / 'day5'
CFGS = ('calib', 'shuffle')
T, T_WORDS = 2.5, 2.0
SPEC_SEED0 = 50000
N_MIX = 200
BG_SEED0 = 59000
BG_HZ = [round(float(x), 3) for x in np.linspace(1.0, 3.0, 10)]
N_EP = N_MIX + len(BG_HZ)


def spec_for(k):
    """Spec of episode k: depends only on k."""
    if 0 <= k < N_MIX:
        return sample_episode(SPEC_SEED0 + k, T=T, t_words=T_WORDS)
    if N_MIX <= k < N_EP:
        j = k - N_MIX
        return background_episode(BG_HZ[j], BG_SEED0 + j, T=T)
    raise ValueError(f'episode {k} out of range 0..{N_EP - 1}')


def specs():
    return [spec_for(k) for k in range(N_EP)]


def path(k, cfg):
    return OUT / f'ep{k:03d}_{cfg}.npz'


def log(msg):
    print(msg, flush=True)
    with open(OUT / 'run_log.txt', 'a') as f:
        f.write(msg + '\n')


def load_valid(k, cfg, spec=None):
    """The saved episode as a dict, or None if missing / unreadable / inconsistent."""
    f = path(k, cfg)
    if not f.exists():
        return None
    try:
        with np.load(f) as d:
            sp = json.loads(str(d['spec']))
            n = int(d['n_spikes'])
            if len(d['t']) != n or len(d['i']) != n or str(d['cfg']) != cfg:
                return None
            if sp != (spec_for(k) if spec is None else spec):
                return None
            return dict(spec=sp, n_spikes=n, wall_s=float(d['wall_s']))
    except Exception:
        return None


def save(k, cfg, spec, ts, si, wall):
    tmp = OUT / f'.ep{k:03d}_{cfg}.{os.getpid()}.tmp.npz'
    with open(tmp, 'wb') as fh:
        np.savez_compressed(fh, t=ts, i=si, spec=json.dumps(spec), cfg=cfg,
                            n_spikes=np.int64(len(ts)), wall_s=np.float64(wall))
    os.replace(tmp, path(k, cfg))


def write_index():
    rows = []
    for k in range(N_EP):
        for cfg in CFGS:
            r = load_valid(k, cfg)
            if r is None:
                continue
            sp = r['spec']
            rows.append(dict(episode=k, cfg=cfg, kind='mix' if sp['words'] else 'bg_only',
                             seed=sp['seed'], spec_seed=sp['spec_seed'],
                             words='+'.join(w['word'] for w in sp['words']) or '(background)',
                             n_words=len(sp['words']), bg_hz=sp['bg_hz'],
                             has_looming=any(w['word'] == 'looming' for w in sp['words']),
                             T=sp['T'], n_spikes=r['n_spikes'], wall_s=round(r['wall_s'], 2),
                             wall_per_bio_s=round(r['wall_s'] / sp['T'], 2), file=path(k, cfg).name))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'index.csv', index=False)
    return df


def write_specs():
    sp = specs()
    f = OUT / 'specs.json'
    if f.exists():                     # re-read before overwriting (shared mount)
        old = json.loads(f.read_text())
        if old == sp:
            return
        log(f'[specs] {f.name} differs from the generated specs: overwriting')
    f.write_text(json.dumps(sp, indent=1))


def build(cfg):
    import torch
    from sim.cns import cns_model, W4_CALIB
    t0 = time.time()
    m = cns_model('cns', cfg, **W4_CALIB)
    pos = {b: i for i, b in enumerate(m['meta'].bodyId)}
    log(f'[{cfg}] model built in {time.time() - t0:.1f} s, torch threads {torch.get_num_threads()}')
    return m, pos


def run_spec(m, pos, spec):
    import torch
    from sim.lif_numpy import P
    from sim import lif_torch
    from stimuli.mixtures import to_blocks
    p = dict(P); p['t_run'] = spec['T']
    t0 = time.time()
    r = lif_torch.run_trial(m['W'], [], params=p, seed=spec['seed'], apl=m['apl'], gap=m['gap'],
                            presyn=m['presyn'], blocks=to_blocks(spec, m['meta'], pos), record=True,
                            device='cpu', rng='torch', dtype=torch.float32)
    return r['t'].astype(np.int32), r['i'].astype(np.int32), time.time() - t0


def run(k0, k1, cfgs):
    todo = {cfg: [k for k in range(k0, k1) if load_valid(k, cfg) is None] for cfg in cfgs}
    log(f'[chunk] episodes {k0}..{k1 - 1}, cfgs {",".join(cfgs)}: '
        + ', '.join(f'{c} {len(todo[c])} to run' for c in cfgs) + f' ({time.strftime("%Y-%m-%d %H:%M:%S")})')
    for cfg in cfgs:
        if not todo[cfg]:
            continue
        m, pos = build(cfg)
        for k in todo[cfg]:
            spec = spec_for(k)
            ts, si, wall = run_spec(m, pos, spec)
            save(k, cfg, spec, ts, si, wall)
            words = '+'.join(w['word'] for w in spec['words']) or '(background)'
            log(f'[{cfg}] ep{k:03d} {words} bg {spec["bg_hz"]:.2f} Hz: {len(ts):,} spikes, '
                f'{wall:.1f} s wall ({wall / spec["T"]:.1f} s per bio-s)')
        del m
    df = write_index()
    log(f'[chunk] done ({time.strftime("%Y-%m-%d %H:%M:%S")}); index.csv has {len(df)} rows')


def check(k, cfg):
    k = int(k)
    d = np.load(path(k, cfg))
    spec = json.loads(str(d['spec']))
    m, pos = build(cfg)
    ts, si, wall = run_spec(m, pos, spec)
    same = len(ts) == len(d['t']) and np.array_equal(ts, d['t']) and np.array_equal(si, d['i'])
    log(f'[check] ep{k:03d}_{cfg}: saved {len(d["t"]):,} spikes, re-run {len(ts):,}, '
        f'identical={same} ({wall:.1f} s wall)')
    return same


def main(*args):
    OUT.mkdir(parents=True, exist_ok=True)
    if not args:
        sys.exit(__doc__)
    if args[0] == 'check':
        sys.exit(0 if check(args[1], args[2]) else 1)
    if args[0] == 'index':
        write_specs()
        print(f'index.csv: {len(write_index())} rows')
        return
    k0, k1 = int(args[0]), int(args[1])
    if not 0 <= k0 < k1 <= N_EP:
        sys.exit(f'need 0 <= k_start < k_end <= {N_EP}')
    cfgs = tuple(args[2:]) or CFGS
    bad = [c for c in cfgs if c not in CFGS]
    if bad:
        sys.exit(f'unknown cfg {bad}; use {CFGS}')
    write_specs()
    run(k0, k1, cfgs)


if __name__ == '__main__':
    main(*sys.argv[1:])
