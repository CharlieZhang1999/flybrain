#!/usr/bin/env python
"""W4D1: PyTorch simulator vs NumPy simulator, and a speed benchmark.

  python scripts/w4d1_torch_parity.py parity [scope] [t_run]
      Same model, same seed, rng='numpy': counts must match spike-for-spike
      on CPU float64. Default: scope=brain, word=vinegar, 0.1 s.
  python scripts/w4d1_torch_parity.py bench [scope] [t_run] [device]
      Wall time per biological second for torch (and NumPy on CPU).
Output: results/w4/w4d1_<mode>_<scope>.txt (appended)
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.cns import cns_model  # noqa: E402
from sim.lif_numpy import P, run_trial as run_np  # noqa: E402
from sim import lif_torch  # noqa: E402
from scripts.ckpt2_ab import stim_idx  # noqa: E402

OUT = ROOT / 'results' / 'w4'
OUT.mkdir(parents=True, exist_ok=True)


def log(path, s):
    print(s, flush=True)
    with open(path, 'a') as f:
        f.write(s + '\n')


def main(mode='parity', scope='brain', t_run=0.1, device='cpu', word='vinegar'):
    t_run = float(t_run)
    m = cns_model(scope)
    meta = m['meta']
    pos = {b: i for i, b in enumerate(meta.bodyId)}
    exc = stim_idx(meta, pos, word)
    p = dict(P); p['t_run'] = t_run
    kw = dict(params=p, seed=0, apl=m['apl'], gap=m['gap'], presyn=m['presyn'])
    path = OUT / f'w4d1_{mode}_{scope}.txt'
    log(path, f'--- {time.strftime("%Y-%m-%d %H:%M")} mode={mode} scope={scope} word={word} '
              f't_run={t_run}s device={device} nnz={m["W"].nnz:,}')
    t0 = time.time()
    r_t = lif_torch.run_trial(m['W'], exc, device=device, rng='numpy' if mode == 'parity' else 'torch', **kw)
    wt = time.time() - t0
    log(path, f'torch {device}: {wt:.1f} s wall = {wt / t_run:.1f} s per bio-second, '
              f'{int(r_t.sum()):,} spikes, {int((r_t > 0).sum()):,} active neurons')
    if mode == 'parity' or device == 'cpu':
        t0 = time.time()
        r_n = run_np(m['W'], exc, presyn=m['presyn'], **{k: v for k, v in kw.items() if k != 'presyn'})
        wn = time.time() - t0
        log(path, f'numpy cpu: {wn:.1f} s wall = {wn / t_run:.1f} s per bio-second, '
                  f'{int(r_n.sum()):,} spikes')
        if mode == 'parity':
            d = np.abs(r_t - r_n)
            log(path, f'parity: neurons differing {int((d > 0).sum())}, max |dcount| {int(d.max())}, '
                      f'identical={bool((d == 0).all())}')


if __name__ == '__main__':
    main(*sys.argv[1:])
