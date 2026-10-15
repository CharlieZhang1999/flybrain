#!/usr/bin/env python
"""Fly word demo: type a taste word, see which neurons spike.

Usage:
  python scripts/word_demo.py sugar
  python scripts/word_demo.py sugar+bitter       # mix stimuli
  python scripts/word_demo.py water --rate 200 --trials 3 --top 20

Runs the NumPy whole-brain LIF (139k neurons, FlyWire v783) with the chosen
GRNs Poisson-driven, and prints the top spiking neurons with annotations.
SEZ cell-type names come from the Shiu repo's sez_neurons.pickle.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPO = ROOT / 'third_party' / 'Drosophila_brain_model'

from sim.lif_numpy import load_network, run_exp, P  # noqa: E402
from stimuli.words import WORDS, KNOWN, ID_MN9_LEFT  # noqa: E402


def annotations():
    names = dict(KNOWN)
    pkl = REPO / 'sez_neurons.pickle'
    if pkl.exists():
        sez = pickle.load(open(pkl, 'rb'))
        for typ, ids in sez.items():
            for f in ids:
                names.setdefault(int(f), f'SEZ {typ}')
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('word', help='|'.join(WORDS) + " (combine with '+')")
    ap.add_argument('--rate', type=float, default=150.0, help='GRN Poisson rate (Hz)')
    ap.add_argument('--trials', type=int, default=3)
    ap.add_argument('--top', type=int, default=25)
    args = ap.parse_args()

    parts = [w.strip().lower() for w in args.word.split('+')]
    bad = [w for w in parts if w not in WORDS]
    if bad:
        sys.exit(f"unknown word(s) {bad}; choose from {sorted(WORDS)}")
    exc = sorted({f for w in parts for f in WORDS[w]})

    print(f">>> stimulus: {'+'.join(parts)}  ({len(exc)} GRNs at {args.rate:.0f} Hz, "
          f"{args.trials} trials x {P['t_run']:.1f} s)")
    t0 = time.time()
    ids, W = load_network(REPO / '2023_03_23_completeness_630_final.csv',
                          REPO / '2023_03_23_connectivity_630_final.parquet')
    print(f"    network loaded: {len(ids):,} neurons, {W.nnz:,} synapses "
          f"({time.time()-t0:.0f} s)")

    t0 = time.time()
    rates = run_exp(W, ids, exc, r_poi=args.rate, n_run=args.trials)
    print(f"    simulated in {time.time()-t0:.0f} s")

    names = annotations()
    df = rates[rates > 0].sort_values(ascending=False).to_frame()
    df['name'] = [names.get(f, '') for f in df.index]
    df['input'] = ['*' if f in set(exc) else '' for f in df.index]

    n_active = len(df)
    print(f"\n{n_active:,} of {len(ids):,} neurons spiked. Top {args.top} by rate:\n")
    pd.set_option('display.width', 120)
    print(df.head(args.top).to_string(
        header=['rate_Hz', 'annotation', 'stimulated'], float_format='%.1f'))

    mn9 = rates.get(ID_MN9_LEFT, 0.0)
    print(f"\nMN9 (proboscis extension) rate: {mn9:.1f} Hz "
          f"-> {'EXTEND (feed)' if mn9 > 10 else 'no feeding response'}")


if __name__ == '__main__':
    main()
