#!/usr/bin/env python
"""Week 5 P1 design check: how similar are two Poisson realizations of the same trial?

A3 asks for rate correlation >= 0.999 between a batched trial and the reference
with the same seed. If the batch drew its own RNG stream (counter-based, or
draws on inactive drives), a batched trial would be a different Poisson
realization of the reference trial. This measures that case directly: the
reference (sim/lif_torch.run_trial, CUDA, float64, rng='torch', looming bundle,
1 s) for seeds 0, 1, 2, 1000, 1001, 1002, and the rate correlation on active
neurons between pairs of different seeds. Untimed.

Usage (repo root):
  python bench/w5_realizations.py --bundle ~/flybench/flybrain/results/w4/cns_bundle_calib.npz
Output: perf/results/w5/<host>_<timestamp>_realizations.json
"""
import argparse
import json
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
import w4_overhead as t1  # noqa: E402
import w4_graph as w4g  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', default=str(ROOT / 'results' / 'w4' / 'cns_bundle_calib.npz'))
    ap.add_argument('--out', default=str(ROOT / 'perf' / 'results' / 'w5'))
    a = ap.parse_args()
    m = t1.load_bundle(a.bundle)
    R = {}
    for s in (0, 1, 2, 1000, 1001, 1002):
        R[s] = np.asarray(lif_torch.run_trial(m['W'], m['stim']['looming'], params=dict(P), seed=s, apl=m['apl'],
                                              gap=m['gap'], presyn=m['presyn'], device='cuda',
                                              dtype=torch.float64, rng='torch'))
    pairs = {f'{x}_vs_{y}': w4g._corr(R[x], R[y]) for x, y in ((0, 1000), (1, 1001), (2, 1002), (0, 1), (1, 2))}
    res = dict(env=t1.env_info(), what='reference, CUDA float64 rng=torch, looming, 1 s; different seeds',
               pairs=pairs, min_rate_corr=min(p['rate_corr'] for p in pairs.values()),
               max_rate_corr=max(p['rate_corr'] for p in pairs.values()))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    f = out / f'{socket.gethostname()}_{time.strftime("%Y%m%d-%H%M")}_realizations.json'
    f.write_text(json.dumps(res, indent=1))
    print(json.dumps(pairs, indent=1), '\nsaved', f)


if __name__ == '__main__':
    main()
