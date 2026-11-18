"""Stimulus-mixture episodes for the Week 4 brain -> motor regression dataset.

An episode is a plain dict (JSON-serialisable) that holds everything needed to
re-run it:
  T          episode length (s)
  seed       Poisson seed for lif_torch.run_trial
  spec_seed  the integer the episode was sampled from (None if a Generator was passed)
  bg_hz      the episode's background Poisson rate, [0, T)
  bg_scheme  (new specs only) per-class background, see below. A spec without
             this key is a legacy spec: bg_hz on every sensory neuron.
  words      list of dict(word, modality, rate_hz, onset, offset)

Background scheme (Day 5, T2b). The spec records its own scheme, so an old spec
(T1 pilot, no `bg_scheme`) still gives exactly its original blocks:
  bg_scheme = dict(name='class_v1', bristle_hz=<float>, abdominal_hz=<'same' | float>)
  - bristle   (BRISTLE: VNC sensory, subclass 'mechanosensory bristle', 2,206 neurons)
              gets bristle_hz (default 0 Hz, Charlie 2026-11-18);
  - abdominal (ABDOMINAL: class 'unknown_sensory', subclass 'abdomen', 1,040 neurons)
              gets bg_hz if abdominal_hz == 'same' (default, until the group is
              identified), else abdominal_hz;
  - every other sensory neuron gets bg_hz.
  The masks are in bg_class_masks(meta).

Sampling rules (sample_episode, default = Day 5 scheme):
  - 2-4 words, at most one per modality. The looming modality ('vision', the
    modality of the word 'looming') is included with probability looming_p
    (default 1/6); the remaining modalities are drawn uniformly without
    replacement from the other modalities. Then one word uniformly within each
    modality;
  - rate uniform in [30, 150] Hz;
  - onset uniform in [0, T - 0.3] s, duration uniform in [0.2, 1.0] s,
    offset = min(onset + duration, T);
  - background uniform in BG_HZ = [1, 3] Hz, with bg_scheme as above.
  legacy=True reproduces the Day 4 (T1 pilot) sampler draw-for-draw: modalities
  uniform (looming share 3/7), background [1, 5] Hz on all sensory neurons, no
  bg_scheme key.
Times are rounded to the 0.1 ms simulator step so the spec is exactly what runs.

to_blocks(spec, meta, pos) -> run_trial blocks [(t0, t1, idx, rate_hz), ...]:
  legacy spec: one background block (every sensory neuron,
    sim.presyn.sensory_mask_malecns), as on Day 4;
  class_v1 spec: one background block at bg_hz on every sensory neuron except
    bristles (and except the abdominal group if its rate differs), then a
    bristle block if bristle_hz > 0, then an abdominal block if its rate is
    separate and > 0. Neurons at 0 Hz are in no block.
  Then one block per word (the word's neurons, stimuli.malecns_words.select_ids,
  same indices as scripts.ckpt2_ab.stim_idx). A word neuron that is also
  sensory receives background + word drive inside its window.
  Timing: run_trial records a Poisson hit at step t as a spike at step t+1, so
  drive-only spikes of a block [t0, t1) fall in steps (t0, t1] (in 0.1 ms steps).

Usage:
  from stimuli.mixtures import sample_episode, to_blocks
  spec = sample_episode(123)                       # Day 5 scheme
  spec = sample_episode(123, abdominal_bg=0.0)     # abdominal group silent
  spec = sample_episode(4040, legacy=True)         # Day 4 / T1 pilot sampler
  blocks = to_blocks(spec, meta, pos)
"""
import numpy as np

from stimuli.malecns_words import WORDS, select_ids

RATE_HZ = (30.0, 150.0)
DUR_S = (0.2, 1.0)
ONSET_MARGIN_S = 0.3
BG_HZ = (1.0, 3.0)              # Day 5 (Charlie's pick)
LEGACY_BG_HZ = (1.0, 5.0)       # Day 4 / T1 pilot
N_WORDS = (2, 4)
DT = 1e-4

LOOMING_P = 1.0 / 6.0           # P(episode contains the looming modality)
BRISTLE_BG = 0.0                # Hz on mechanosensory bristles
ABDOMINAL_BG = 'same'           # 'same' (= bg_hz) or a rate in Hz
SCHEME = 'class_v1'


def _q(x):
    """Round a time to the simulator step (0.1 ms)."""
    return round(round(float(x) / DT) * DT, 4)


def _abdominal(x):
    if isinstance(x, str):
        if x != 'same':
            raise ValueError(f"abdominal_bg must be 'same' or a rate in Hz, got {x!r}")
        return x
    x = float(x)
    if x < 0:
        raise ValueError(f'abdominal_bg must be >= 0, got {x}')
    return x


def bg_scheme(bristle_bg=BRISTLE_BG, abdominal_bg=ABDOMINAL_BG):
    """The bg_scheme dict stored in a spec."""
    b = float(bristle_bg)
    if b < 0:
        raise ValueError(f'bristle_bg must be >= 0, got {b}')
    return dict(name=SCHEME, bristle_hz=b, abdominal_hz=_abdominal(abdominal_bg))


def _word(rng, words, mod, T):
    cand = sorted(k for k, w in words.items() if w['modality'] == mod)
    word = cand[int(rng.integers(len(cand)))]
    rate = float(rng.uniform(*RATE_HZ))
    onset = _q(rng.uniform(0.0, T - ONSET_MARGIN_S))
    offset = _q(min(onset + rng.uniform(*DUR_S), T))
    return dict(word=word, modality=str(mod), rate_hz=round(rate, 3), onset=onset, offset=offset)


def sample_episode(rng, words=WORDS, T=2.0, legacy=False, looming_p=LOOMING_P,
                   bg_range=BG_HZ, bristle_bg=BRISTLE_BG, abdominal_bg=ABDOMINAL_BG):
    """Sample one episode spec. rng: int seed or np.random.Generator.

    legacy=True: the Day 4 sampler, unchanged (looming_p, bg_range, bristle_bg and
    abdominal_bg are ignored).
    """
    spec_seed = None
    if not isinstance(rng, np.random.Generator):
        spec_seed = int(rng)
        rng = np.random.default_rng(spec_seed)
    mods = sorted({w['modality'] for w in words.values()})
    n = int(rng.integers(N_WORDS[0], N_WORDS[1] + 1))
    if legacy:
        if n > len(mods):
            raise ValueError(f'{n} words need {n} modalities, only {len(mods)} available')
        picks = [_word(rng, words, mod, T) for mod in rng.choice(mods, size=n, replace=False)]
        picks.sort(key=lambda d: (d['onset'], d['word']))
        return dict(T=float(T), seed=int(rng.integers(2 ** 31 - 1)), spec_seed=spec_seed,
                    bg_hz=round(float(rng.uniform(*LEGACY_BG_HZ)), 3), words=picks)

    scheme = bg_scheme(bristle_bg, abdominal_bg)
    loom = words['looming']['modality'] if 'looming' in words else None
    others = [m for m in mods if m != loom]
    with_loom = loom is not None and bool(rng.random() < looming_p)
    k = n - 1 if with_loom else n
    if k > len(others):
        raise ValueError(f'{n} words need {n} modalities, only {len(others) + with_loom} available')
    chosen = [str(m) for m in rng.choice(others, size=k, replace=False)] + ([loom] if with_loom else [])
    picks = [_word(rng, words, mod, T) for mod in chosen]
    picks.sort(key=lambda d: (d['onset'], d['word']))
    return dict(T=float(T), seed=int(rng.integers(2 ** 31 - 1)), spec_seed=spec_seed,
                bg_hz=round(float(rng.uniform(*bg_range)), 3), bg_scheme=scheme, words=picks)


def background_episode(bg_hz, seed, T=2.0, legacy=False, bristle_bg=BRISTLE_BG,
                       abdominal_bg=ABDOMINAL_BG):
    """Spec with background only (no words). legacy=True: bg_hz on all sensory neurons."""
    spec = dict(T=float(T), seed=int(seed), spec_seed=None, bg_hz=float(bg_hz), words=[])
    if not legacy:
        spec['bg_scheme'] = bg_scheme(bristle_bg, abdominal_bg)
    return spec


def word_idx(meta, pos, word):
    return np.array([pos[b] for b in select_ids(meta, word)], dtype=np.int64)


def bg_class_masks(meta):
    """Boolean masks (over meta rows) used by the class_v1 background scheme.

    sensory    sim.presyn.sensory_mask_malecns (superclass contains 'sensory')
    bristle    sensory & VNC (sim.cns.regions) & subclass == 'mechanosensory bristle'
    abdominal  sensory & class == 'unknown_sensory' & subclass == 'abdomen'
    """
    from sim.presyn import sensory_mask_malecns
    from sim.cns import regions
    sens = sensory_mask_malecns(meta)
    vnc, _ = regions(meta)
    cl = meta['class'].astype(object).where(meta['class'].notna(), '').astype(str).to_numpy()
    sub = meta['subclass'].astype(object).where(meta['subclass'].notna(), '').astype(str).to_numpy()
    bristle = sens & vnc & (sub == 'mechanosensory bristle')
    abdominal = sens & (cl == 'unknown_sensory') & (sub == 'abdomen')
    return dict(sensory=sens, bristle=bristle, abdominal=abdominal)


def bg_blocks(spec, meta):
    """Background blocks only (see module docstring)."""
    T, bg = spec['T'], float(spec['bg_hz'])
    scheme = spec.get('bg_scheme')
    if scheme is None:                                   # legacy: uniform on all sensory
        from sim.presyn import sensory_mask_malecns
        sens = np.flatnonzero(sensory_mask_malecns(meta))
        return [(0.0, T, sens, bg)] if bg > 0 else []
    if scheme.get('name') != SCHEME:
        raise ValueError(f'unknown bg_scheme {scheme!r}')
    mk = bg_class_masks(meta)
    b_hz = float(scheme['bristle_hz'])
    a = _abdominal(scheme['abdominal_hz'])
    a_hz = bg if a == 'same' else a
    rest = mk['sensory'] & ~mk['bristle']
    sep_abd = a_hz != bg
    if sep_abd:
        rest &= ~mk['abdominal']
    out = []
    for mask, r in ((rest, bg), (mk['bristle'], b_hz),
                    (mk['abdominal'] & ~mk['bristle'] if sep_abd else None, a_hz)):
        if mask is None or r <= 0 or not mask.any():
            continue
        out.append((0.0, T, np.flatnonzero(mask), float(r)))
    return out


def to_blocks(spec, meta, pos):
    """run_trial blocks for a spec: background blocks, then one block per word."""
    blocks = bg_blocks(spec, meta)
    for w in spec['words']:
        blocks.append((w['onset'], w['offset'], word_idx(meta, pos, w['word']), float(w['rate_hz'])))
    return blocks


def last_offset(spec):
    return max((w['offset'] for w in spec['words']), default=0.0)
