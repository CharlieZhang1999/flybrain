#!/usr/bin/env python
"""W4D3b: do the day-3b brain changes (GLNO/LPsP sign fix, DNg33/IN19B040
clique gain) change the Week 3 word responses? Brain scope, 6 words, seed 0,
drive 0-0.5 s, run to 1.0 s: spike-count correlation old vs new, active
neurons during drive and after offset.
Usage: python scripts/w4d3b_word_check.py
Output: results/w4/sci/w4d3b_word_check.txt
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.cns import cns_model, SIGN_FIX, CLIQUE_GAIN  # noqa: E402
from sim.lif_numpy import P  # noqa: E402
from sim import lif_torch  # noqa: E402
from scripts.ckpt2_ab import stim_idx  # noqa: E402

WORDS = ['sugar', 'bitter', 'vinegar', 'looming', 'sound', 'hot']


def main():
    old = cns_model('brain', 'calib')
    new = cns_model('brain', 'calib', sign_fix=SIGN_FIX, clique_gain=CLIQUE_GAIN)
    meta = old['meta']
    pos = {b: i for i, b in enumerate(meta.bodyId)}
    p = dict(P); p['t_run'] = 1.0
    lines = ['# W4D3b word check (brain scope, seed 0): old = Week 3 model, new = + sign fix + clique gain', '',
             '| word | count corr (0-0.5 s) | active old / new (drive) | active old / new (0.6-1.0 s) |', '|---|---|---|---|']
    for w in WORDS:
        exc = stim_idx(meta, pos, w)
        res = []
        for m in (old, new):
            r = lif_torch.run_trial(m['W'], exc, params=p, seed=0, apl=m['apl'], gap=m['gap'], presyn=m['presyn'],
                                    t_stim=0.5, record=True, device='cpu', rng='numpy', dtype=torch.float64)
            ts = r['t'] * P['dt']
            res.append((np.bincount(r['i'][ts < 0.5], minlength=len(meta)), np.unique(r['i'][ts >= 0.6])))
        c = np.corrcoef(res[0][0], res[1][0])[0, 1]
        row = (f'| {w} | {c:.4f} | {(res[0][0] > 0).sum()} / {(res[1][0] > 0).sum()} | '
               f'{len(res[0][1])} / {len(res[1][1])} |')
        print(row, flush=True); lines.append(row)
    (ROOT / 'results' / 'w4' / 'sci' / 'w4d3b_word_check.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
