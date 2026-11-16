#!/usr/bin/env python
"""Week 2 Days 4-5, part 2: decode the word from simulated activity.

For each config (base, calib, shuffle) and window (stimulus 0-0.5 s, post
0.5-1.0 s), fit the linear readout on (a) all non-sensory brain neurons and
(b) motor-only: descending neurons + brain motor neurons (superclass
descending_neuron, cb_motor). Chance = 1/17.
Output: results/readout/readout_summary.json, printed table.
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy import sparse
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from readout.linear import fit_eval  # noqa: E402
from stimuli.malecns_words import WORDS, select_ids  # noqa: E402
import pandas as pd  # noqa: E402

D = ROOT / 'results' / 'readout'


def main():
    meta = pd.read_parquet(ROOT / 'data' / 'malecns' / 'malecns_traced_meta.parquet')
    pos = {b: i for i, b in enumerate(meta.bodyId)}
    sens = np.zeros(len(meta), bool)
    for w in WORDS:
        sens[[pos[b] for b in select_ids(meta, w)]] = True
    allk = ~sens
    motor = meta.superclass.isin(['descending_neuron', 'cb_motor']).to_numpy() & ~sens
    out = {'words': list(WORDS), 'chance': 1 / len(WORDS), 'results': {}}
    for cfg in ['base', 'calib', 'shuffle']:
        f = D / f'stim_{cfg}.npz'
        if not f.exists():
            continue
        meta_npz = np.load(D / f'trials_{cfg}.npz', allow_pickle=True)
        y, seeds = meta_npz['labels'], meta_npz['seeds']
        for win in ['stim', 'post']:
            X = sparse.load_npz(D / f'{win}_{cfg}.npz').toarray()
            for name, keep in [('all', allk), ('motor', motor)]:
                r = fit_eval(X, y, seeds, keep)
                out['results'][f'{cfg}/{win}/{name}'] = r
                print(f'{cfg:8s} {win:5s} {name:6s} acc(7/3)={r["acc_split"]:.3f} '
                      f'acc(LOSO)={r["acc_loso"]:.3f} features={r["n_features"]}', flush=True)
    json.dump(out, open(D / 'readout_summary.json', 'w'))


if __name__ == '__main__':
    main()
