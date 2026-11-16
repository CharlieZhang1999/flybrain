#!/usr/bin/env python
"""Week 2 Days 4-5, part 1: simulate the word vocabulary for the readout.

Configs (MaleCNS brain only, monoamines removed, w x 0.581, rate APL k = 0.1):
  base     : C8 base model (has the post-odor attractor)
  calib    : gap-junction calibration (chemical cholinergic LN->PN = 0,
             eLN-PN gap junctions from sim/gap.py, 2 mV spikelets; variant given
             by GAP_KW below)
  shuffle  : calib with the chemical matrix shuffled. Each edge keeps its
             postsynaptic neuron; its (presynaptic neuron, weight) pair is
             permuted across all edges. So every neuron keeps its in-degree,
             out-degree and the sign of its outputs; the wiring is random.
             Gap junctions and stimulated neurons are unchanged.
Trial: drive 0-0.5 s at 150 Hz, silence to 1.0 s. 17 words x 10 seeds.
Saved per config: spike counts per neuron in 0-0.5 s and 0.5-1.0 s, and
population activity in 10 ms bins (for the artifact).
Usage: w2d4_word_trials.py <config>
Output: results/readout/trials_<config>.npz
"""
import sys
from pathlib import Path
import numpy as np
from scipy import sparse
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.lif_numpy import run_trial  # noqa: E402
from sim.lesion import select  # noqa: E402
from sim.malecns import load  # noqa: E402
from sim.gap import build_gap  # noqa: E402
from stimuli.malecns_words import WORDS  # noqa: E402
from scripts.ckpt2_ab import stim_idx  # noqa: E402
from scripts.c4_core_gain import Runner, DT  # noqa: E402
from scripts.ckpt5_stageA import ACH_LN, PN  # noqa: E402
from scripts.w2d3_gap_malecns import cut_ln_pn  # noqa: E402

GAP_KW = dict(c0=0.2)          # set from the w2d3d refinement
OUT = ROOT / 'results' / 'readout'
T_STIM, T_END, BIN = 0.5, 1.0, 0.01
SEEDS = range(10)


def shuffle_W(W, seed=12345):
    C = W.tocoo()
    perm = np.random.default_rng(seed).permutation(len(C.data))
    Ws = sparse.csc_matrix((C.data[perm], (C.row, C.col[perm])), shape=W.shape)
    Ws.sum_duplicates()
    return Ws


def main(config):
    R = Runner()
    R.p['t_run'] = T_END
    gap = None
    W = R.Wr
    if config in ('calib', 'shuffle'):
        ln, pn = select(R.meta, ACH_LN), select(R.meta, PN)
        _, Wsyn = load(w_syn=1.0, monoamine='removed')
        W = cut_ln_pn(R.Wr, ln, pn)
        gap, info = build_gap(Wsyn, ln, pn, v_spike=0.002, **GAP_KW)
        print('gap', info, flush=True)
    if config == 'shuffle':
        W = shuffle_W(W)
    n, nb = len(R.meta), int(round(T_END / BIN))
    words = list(WORDS)
    stim_c, post_c, pop, labels, seeds = [], [], [], [], []
    for wi, word in enumerate(words):
        for seed in SEEDS:
            r = run_trial(W, stim_idx(R.meta, R.pos, word), params=R.p, seed=seed, apl=R.apl,
                          t_stim=T_STIM, record=True, gap=gap)
            ts, si = r['t'] * DT, r['i']
            m1, m2 = ts < T_STIM, ts >= T_STIM
            stim_c.append(sparse.csr_matrix(np.bincount(si[m1], minlength=n).astype(np.int32)))
            post_c.append(sparse.csr_matrix(np.bincount(si[m2], minlength=n).astype(np.int32)))
            pop.append(np.bincount((ts / BIN).astype(int), minlength=nb)[:nb])
            labels.append(wi); seeds.append(seed)
            print(f'[{config}] {word:10s} s{seed}: stim spikes={int(m1.sum()):7d} '
                  f'active={stim_c[-1].nnz:6d} | post active={post_c[-1].nnz:6d}', flush=True)
    np.savez_compressed(OUT / f'trials_{config}.npz', words=np.array(words),
                        labels=np.array(labels), seeds=np.array(seeds),
                        stim=sparse.vstack(stim_c).toarray().astype(np.int16) if False else None,
                        pop=np.array(pop))
    sparse.save_npz(OUT / f'stim_{config}.npz', sparse.vstack(stim_c).tocsr())
    sparse.save_npz(OUT / f'post_{config}.npz', sparse.vstack(post_c).tocsr())
    print('DONE')


if __name__ == '__main__':
    main(sys.argv[1])
