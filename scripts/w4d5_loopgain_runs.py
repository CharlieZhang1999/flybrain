#!/usr/bin/env python
"""W4D5 T2e: verification runs for sim.cns LOOP_GAIN (LOOP5 internal synapses x 0.3).

Model: cns_model('cns', 'calib', **W4_CALIB) (now including loop_gain); 'old' variants use
the same call with loop_gain=None (= the W4_CALIB before LOOP_GAIN). CPU float32, rng='torch'.
Run ONE mode at a time (one simulation process at a time). Existing outputs are skipped.

Modes
  wind       Day 5 wind specs (exact spec + Poisson seed from results/w4/regression/day5/
             specs.json, T 2.5 s, class_v2 bg): the 12 in k 0-19 (saved Day 5 baselines) and
             the first 18 with k >= 20 (no baseline).                  -> wind__ep{k:03d}
  mix        non-wind Day 5 specs ep000, ep001, ep019 (vs saved Day 5) -> mix__ep{k:03d}
  bg         class_v2 background only, 3 Hz, seeds 0-2, T 1.0 s, as scripts/w4d5_bg_fix_runs.py
             (vs results/w4/bg_fix/runs/calib_v2_3Hz_s*.npz)          -> bg__v2_3Hz_s{s}
  loom       looming 0-0.4 s at 150 Hz, no background, T 1.0 s, seeds 0-2 (as
             scripts/w4d3_looming_tail.py, cns variant), old and new  -> loom__{old,new}_s{s}
  windalone  wind 0-0.5 s at 150 Hz, no background, T 1.5 s, seeds 0-2 (Week 3b word-check
             style), old and new                                      -> windalone__{old,new}_s{s}
Usage:  python scripts/w4d5_loopgain_runs.py <mode>
Output: results/w4/regression/loop_gain/runs/<mode>__<name>.npz (t: step, dt 0.1 ms; i: neuron;
        spec: JSON), runs_log.txt (wall time per run).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'results' / 'w4' / 'regression' / 'loop_gain'
RUNS = OUT / 'runs'
DAY5 = ROOT / 'results' / 'w4' / 'regression' / 'day5'
N_LOW, N_HIGH = 20, 18          # wind specs: all with k < 20, first 18 with k >= 20
MIX = (0, 1, 19)


def log(msg):
    print(msg, flush=True)
    with open(OUT / 'runs_log.txt', 'a') as f:
        f.write(msg + '\n')


def wind_eps(specs):
    w = [k for k in range(200) if any(x['word'] == 'wind' for x in specs[k]['words'])]
    return [k for k in w if k < N_LOW] + [k for k in w if k >= N_LOW][:N_HIGH]


def build(old=False):
    from sim.cns import cns_model, W4_CALIB
    kw = {k: v for k, v in W4_CALIB.items() if not (old and k == 'loop_gain')}
    t0 = time.time()
    m = cns_model('cns', 'calib', verbose=False, **kw)
    m['pos'] = {b: i for i, b in enumerate(m['meta'].bodyId)}
    log(f'[model] calib {"old (loop_gain=None)" if old else "new W4_CALIB"} built in {time.time() - t0:.1f} s, '
        f'keys {sorted(kw)}')
    return m


def run(m, T, seed, blocks=None, exc=(), t_stim=None):
    import torch
    from sim.lif_numpy import P
    from sim import lif_torch
    p = dict(P); p['t_run'] = T
    return lif_torch.run_trial(m['W'], list(exc), params=p, seed=seed, apl=m['apl'], gap=m['gap'],
                               presyn=m['presyn'], blocks=blocks, t_stim=t_stim, record=True,
                               device='cpu', rng='torch', dtype=torch.float32)


def path(mode, name):
    return RUNS / f'{mode}__{name}.npz'


def save(mode, name, r, spec, wall):
    RUNS.mkdir(parents=True, exist_ok=True)
    tmp = RUNS / f'.{mode}__{name}.tmp.npz'
    np.savez_compressed(tmp, t=np.asarray(r['t'], np.int32), i=np.asarray(r['i'], np.int32),
                        spec=json.dumps(spec))
    tmp.replace(path(mode, name))
    log(f'[{mode}] {name}: {len(r["t"]):,} spikes, {wall:.1f} s wall')


def day5(eps, mode):
    from stimuli.mixtures import to_blocks
    specs = json.loads((DAY5 / 'specs.json').read_text())
    todo = [k for k in eps if not path(mode, f'ep{k:03d}').exists()]
    log(f'[{mode}] episodes {eps}; to run {todo}')
    if not todo:
        return
    m = build()
    for k in todo:
        spec = specs[k]
        t0 = time.time()
        r = run(m, spec['T'], spec['seed'], blocks=to_blocks(spec, m['meta'], m['pos']))
        save(mode, f'ep{k:03d}', r, dict(spec, episode=k), time.time() - t0)


def wind():
    specs = json.loads((DAY5 / 'specs.json').read_text())
    day5(wind_eps(specs), 'wind')


def mix():
    day5(list(MIX), 'mix')


def bg():
    from stimuli.mixtures import background_episode, to_blocks
    todo = [s for s in (0, 1, 2) if not path('bg', f'v2_3Hz_s{s}').exists()]
    if not todo:
        return
    m = build()
    for s in todo:
        spec = background_episode(3.0, s, T=1.0, scheme='class_v2')
        t0 = time.time()
        r = run(m, 1.0, s, blocks=to_blocks(spec, m['meta'], m['pos']))
        save('bg', f'v2_3Hz_s{s}', r, spec, time.time() - t0)


def single_word(mode, word, t_stim, T):
    from stimuli.mixtures import word_idx
    for variant in ('old', 'new'):
        todo = [s for s in (0, 1, 2) if not path(mode, f'{variant}_s{s}').exists()]
        if not todo:
            continue
        m = build(old=variant == 'old')
        exc = word_idx(m['meta'], m['pos'], word)
        for s in todo:
            t0 = time.time()
            r = run(m, T, s, exc=exc, t_stim=t_stim)
            save(mode, f'{variant}_s{s}', r, dict(word=word, rate_hz=150.0, onset=0.0, offset=t_stim, T=T,
                                                  seed=s, model=variant, n_exc=len(exc), bg='none'),
                 time.time() - t0)
        del m


def loom():
    single_word('loom', 'looming', 0.4, 1.0)


def windalone():
    single_word('windalone', 'wind', 0.5, 1.5)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    modes = dict(wind=wind, mix=mix, bg=bg, loom=loom, windalone=windalone)
    if len(sys.argv) < 2 or sys.argv[1] not in modes:
        sys.exit(__doc__)
    log(f'[start] {sys.argv[1]} {time.strftime("%Y-%m-%d %H:%M:%S")}')
    modes[sys.argv[1]]()
    log(f'[end] {sys.argv[1]} {time.strftime("%Y-%m-%d %H:%M:%S")}')
