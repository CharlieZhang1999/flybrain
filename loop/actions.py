"""LLM -> fly action codes on Johnston's organ (JO) hearing neurons (Week 3 Day 3).

Each candidate code is a list of drive blocks (t0, t1, neuron indices, rate) inside a
500 ms window. Side comes from the MaleCNS `instance` suffix (_L / _R). Left and
right groups are matched in size (JO-A: 18 per side, JO-B: 29 per side) by a fixed
random subsample of the larger side, so a left/right difference can't come from
neuron count alone.
"""
import numpy as np

T_CODE = 0.5
CANDIDATES = ['A-L', 'A-R', 'B-L', 'B-R', 'A+B', 'A-pulse', 'B-pulse', 'A-low']


def jo_groups(meta, seed=7):
    t = meta['type'].astype(str); side = meta['instance'].astype(str).str[-2:]
    rng = np.random.default_rng(seed)
    g = {}
    for sub in ['A', 'B']:
        isub = t.str.startswith(f'JO-{sub}').to_numpy()
        L = np.flatnonzero(isub & (side == '_L').to_numpy())
        R = np.flatnonzero(isub & (side == '_R').to_numpy())
        k = min(len(L), len(R))
        g[f'{sub}-L'] = np.sort(rng.choice(L, k, replace=False))
        g[f'{sub}-R'] = np.sort(rng.choice(R, k, replace=False))
        g[sub] = np.sort(np.r_[L, R])
    return g


def blocks(code, meta, t0=0.0):
    """Drive blocks for one code starting at t0."""
    g = jo_groups(meta)
    if code in ('A-L', 'A-R', 'B-L', 'B-R'):
        return [(t0, t0 + T_CODE, g[code], 150.0)]
    if code == 'A+B':
        return [(t0, t0 + T_CODE, np.r_[g['A'], g['B']], 150.0)]
    if code == 'A-low':
        return [(t0, t0 + T_CODE, g['A'], 50.0)]
    if code in ('A-pulse', 'B-pulse'):        # 20 ms on / 20 ms off at 300 Hz (pulse-song-like)
        idx = g[code[0]]
        return [(t0 + s, t0 + s + 0.02, idx, 300.0) for s in np.arange(0, T_CODE, 0.04)]
    raise ValueError(code)


# Day 3 choice (results/loop/w3d3_select.json): right-side JO neurons in MaleCNS have
# 15-30x fewer output synapses than left (incomplete reconstruction), and drive no
# descending neurons, so only left/bilateral codes are used. A-low (weakest, 9 DNs)
# is left out. The mapping to actions is arbitrary: no code means "left" to the fly.
ACTION_CODES = {'turn left': 'A-L', 'turn right': 'B-L', 'go forward': 'A+B',
                'stop': 'B-pulse', 'eat': 'A-pulse'}


# Day 4 (with the right-JO mirror, sim/jo_mirror.py): mirror-image codes for the two
# turns. B-L/B-R are used rather than A-L/A-R because A-R locks the DNg33 pair on
# (progress/week3/right_jo_fix.md). Mapping is still arbitrary.
ACTION_CODES_MIRROR = {'turn left': 'B-L', 'turn right': 'B-R', 'go forward': 'A+B',
                       'stop': 'B-pulse', 'eat': 'A-pulse'}
