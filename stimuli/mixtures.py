"""Stimulus-mixture episodes for the Week 4 brain -> motor regression dataset.

An episode is a plain dict (JSON-serialisable) that holds everything needed to
re-run it:
  T          episode length (s)
  seed       Poisson seed for lif_torch.run_trial
  spec_seed  the integer the episode was sampled from (None if a Generator was passed)
  bg_hz      background Poisson rate, same for every sensory neuron, [0, T)
  words      list of dict(word, modality, rate_hz, onset, offset)

Sampling rules (sample_episode):
  - 2-4 words, at most one per modality: the modalities are drawn uniformly
    without replacement from the modalities present in `words`, then one word
    uniformly within each modality (so every modality is equally likely,
    whatever its number of words);
  - rate uniform in [30, 150] Hz;
  - onset uniform in [0, T - 0.3] s, duration uniform in [0.2, 1.0] s,
    offset = min(onset + duration, T);
  - background uniform in [1, 5] Hz.
Times are rounded to the 0.1 ms simulator step so the spec is exactly what runs.

to_blocks(spec, meta, pos) -> run_trial blocks [(t0, t1, idx, rate_hz), ...]:
  first the background block (every sensory neuron, sim.presyn.sensory_mask_malecns),
  then one block per word (the word's neurons, stimuli.malecns_words.select_ids,
  same indices as scripts.ckpt2_ab.stim_idx). A word neuron that is also
  sensory receives background + word drive inside its window.

Usage:
  from stimuli.mixtures import sample_episode, to_blocks
  spec = sample_episode(123)
  blocks = to_blocks(spec, meta, pos)
"""
import numpy as np

from stimuli.malecns_words import WORDS, select_ids

RATE_HZ = (30.0, 150.0)
DUR_S = (0.2, 1.0)
ONSET_MARGIN_S = 0.3
BG_HZ = (1.0, 5.0)
N_WORDS = (2, 4)
DT = 1e-4


def _q(x):
    """Round a time to the simulator step (0.1 ms)."""
    return round(round(float(x) / DT) * DT, 4)


def sample_episode(rng, words=WORDS, T=2.0):
    """Sample one episode spec. rng: int seed or np.random.Generator."""
    spec_seed = None
    if not isinstance(rng, np.random.Generator):
        spec_seed = int(rng)
        rng = np.random.default_rng(spec_seed)
    mods = sorted({w['modality'] for w in words.values()})
    n = int(rng.integers(N_WORDS[0], N_WORDS[1] + 1))
    if n > len(mods):
        raise ValueError(f'{n} words need {n} modalities, only {len(mods)} available')
    picks = []
    for mod in rng.choice(mods, size=n, replace=False):
        cand = sorted(k for k, w in words.items() if w['modality'] == mod)
        word = cand[int(rng.integers(len(cand)))]
        rate = float(rng.uniform(*RATE_HZ))
        onset = _q(rng.uniform(0.0, T - ONSET_MARGIN_S))
        offset = _q(min(onset + rng.uniform(*DUR_S), T))
        picks.append(dict(word=word, modality=str(mod), rate_hz=round(rate, 3),
                          onset=onset, offset=offset))
    picks.sort(key=lambda d: (d['onset'], d['word']))
    return dict(T=float(T), seed=int(rng.integers(2 ** 31 - 1)), spec_seed=spec_seed,
                bg_hz=round(float(rng.uniform(*BG_HZ)), 3), words=picks)


def background_episode(bg_hz, seed, T=2.0):
    """Spec with background only (no words)."""
    return dict(T=float(T), seed=int(seed), spec_seed=None, bg_hz=float(bg_hz), words=[])


def word_idx(meta, pos, word):
    return np.array([pos[b] for b in select_ids(meta, word)], dtype=np.int64)


def to_blocks(spec, meta, pos):
    """run_trial blocks for a spec: background on all sensory neurons, then words."""
    from sim.presyn import sensory_mask_malecns
    sens = np.flatnonzero(sensory_mask_malecns(meta))
    blocks = []
    if spec['bg_hz'] > 0:
        blocks.append((0.0, spec['T'], sens, float(spec['bg_hz'])))
    for w in spec['words']:
        blocks.append((w['onset'], w['offset'], word_idx(meta, pos, w['word']), float(w['rate_hz'])))
    return blocks


def last_offset(spec):
    return max((w['offset'] for w in spec['words']), default=0.0)
