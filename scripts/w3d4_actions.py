#!/usr/bin/env python
"""Week 3 Day 4: action codes in the loop setting (with episode history).
Each trial, one continuous run: 5 random different words back to back (500 ms each,
150 Hz), then the action code (500 ms; 'none' = no drive), then 500 ms silence.
  train: 6 classes (5 actions + none) x 8 trials, seeds 500-547
  test : 37 trials, seeds 10-46, actions in a fixed random order (5 actions only)
Brain: calibrated MaleCNS + presynaptic modulation (K = 0.05) + right-JO mirror;
actions use loop.actions.ACTION_CODES_MIRROR; config 'shuffle' uses
the degree- and sign-preserving shuffled chemical matrix (as Week 2).
Usage: w3d4_actions.py <real|shuffle> <train|test> <i_start> <i_end>
Output: results/loop/w3d4_<config>_<part>_<a>_<b>.npz (action-window counts, all neurons)"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from w3d1_common import presyn_model  # noqa: E402
from sim.lif_numpy import run_trial  # noqa: E402
from scripts.ckpt2_ab import stim_idx  # noqa: E402
from stimuli.malecns_words import WORDS  # noqa: E402
from loop.actions import ACTION_CODES_MIRROR as ACTION_CODES, blocks as code_blocks  # noqa: E402

CLASSES = list(ACTION_CODES) + ['none']
T_W, N_W = 0.5, 5


def trials(part):
    if part == 'train':
        return [(CLASSES[k % 6], 500 + k) for k in range(48)]
    order = np.random.default_rng(37).choice(list(ACTION_CODES), 37)
    return [(a, 10 + k) for k, a in enumerate(order)]


def main(config, part, a, b):
    R, W, gap, ps, _ = presyn_model(0.05, 'calib' if config == 'real' else 'shuffle', mirror=True)
    n = len(R.meta); words = list(WORDS)
    t_code = N_W * T_W
    p = dict(R.p); p['t_run'] = t_code + 1.0
    rows, labels, seqs, tails = [], [], [], []
    for action, seed in trials(part)[a:b]:
        seq = [words[k] for k in np.random.default_rng(seed).choice(len(words), N_W, replace=False)]
        bl = [(k * T_W, (k + 1) * T_W, stim_idx(R.meta, R.pos, w), 150.0) for k, w in enumerate(seq)]
        if action != 'none':
            bl += code_blocks(ACTION_CODES[action], R.meta, t0=t_code)
        r = run_trial(W, [], params=p, seed=seed, apl=R.apl, record=True, gap=gap, presyn=ps, blocks=bl)
        ts, si = r['t'] * p['dt'], r['i']
        m = (ts >= t_code) & (ts < t_code + 0.5)
        rows.append(np.bincount(si[m], minlength=n).astype(np.int32))
        pre = (ts >= t_code - 0.1) & (ts < t_code)
        tails.append(len(np.unique(si[pre])))
        labels.append(action); seqs.append('+'.join(seq))
        print(f'[{config} {part}] seed {seed} {"+".join(seq)} -> {action}: active={int((rows[-1] > 0).sum())} '
              f'(firing just before code: {tails[-1]})', flush=True)
    np.savez_compressed(ROOT / 'results' / 'loop' / f'w3d4_{config}_{part}_{a}_{b}.npz', X=np.array(rows),
                        labels=np.array(labels), seqs=np.array(seqs), pre_active=np.array(tails))
    print('DONE', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
