"""stimuli/mixtures.py: class-specific background, looming share, legacy specs.

Runs on a synthetic meta table (no MaleCNS data needed). The real-data checks
(class counts, T1 pilot specs.json) are in scripts/w4d5_bg_fix_checks.py.
"""
import numpy as np
import pandas as pd
import pytest

from stimuli import mixtures as mx
from stimuli.malecns_words import WORDS


def _day4_sample(seed, words=WORDS, T=2.0):
    """Frozen copy of the Day 4 sample_episode (git e3bbd01), for the legacy check."""
    rng = np.random.default_rng(seed)
    mods = sorted({w['modality'] for w in words.values()})
    n = int(rng.integers(2, 5))
    picks = []
    for mod in rng.choice(mods, size=n, replace=False):
        cand = sorted(k for k, w in words.items() if w['modality'] == mod)
        word = cand[int(rng.integers(len(cand)))]
        rate = float(rng.uniform(30.0, 150.0))
        onset = mx._q(rng.uniform(0.0, T - 0.3))
        offset = mx._q(min(onset + rng.uniform(0.2, 1.0), T))
        picks.append(dict(word=word, modality=str(mod), rate_hz=round(rate, 3), onset=onset, offset=offset))
    picks.sort(key=lambda d: (d['onset'], d['word']))
    return dict(T=float(T), seed=int(rng.integers(2 ** 31 - 1)), spec_seed=seed,
                bg_hz=round(float(rng.uniform(1.0, 5.0)), 3), words=picks)


@pytest.fixture(scope='module')
def meta():
    rows = (
        [dict(superclass='vnc_sensory', **{'class': 'mechanosensory_tactile'}, subclass='mechanosensory bristle',
              somaNeuromere=None, type='BR')] * 5
        + [dict(superclass='vnc_sensory', **{'class': 'unknown_sensory'}, subclass='mechanosensory bristle',
                somaNeuromere=None, type='BR2')] * 2
        + [dict(superclass='vnc_sensory', **{'class': 'unknown_sensory'}, subclass='abdomen',
                somaNeuromere=None, type=None)] * 4
        + [dict(superclass='sensory_ascending', **{'class': 'unknown_sensory'}, subclass='abdomen',
                somaNeuromere=None, type=None)] * 1
        + [dict(superclass='vnc_sensory', **{'class': 'chordotonal'}, subclass='chordotonal organ',
                somaNeuromere=None, type='CH')] * 3
        + [dict(superclass='cb_sensory', **{'class': 'mechanosensory'}, subclass=None,
                somaNeuromere=None, type='BM_InOm')] * 3          # head bristle: not in the mask
        + [dict(superclass='vpn', **{'class': None}, subclass=None, somaNeuromere=None, type='LC4')] * 2
        + [dict(superclass='vnc_motor', **{'class': None}, subclass=None, somaNeuromere='T1', type='MN')] * 2
    )
    m = pd.DataFrame(rows)
    m.insert(0, 'bodyId', np.arange(len(m)) + 100)
    return m


def test_masks(meta):
    mk = mx.bg_class_masks(meta)
    assert mk['sensory'].sum() == 18
    assert np.flatnonzero(mk['bristle']).tolist() == list(range(7))
    assert np.flatnonzero(mk['abdominal']).tolist() == list(range(7, 12))


@pytest.mark.parametrize('abd, n_blocks, abd_rate', [('same', 1, 2.5), (0, 1, 0.0), (1.0, 2, 1.0)])
def test_blocks_exclude_bristle(meta, abd, n_blocks, abd_rate):
    pos = {b: k for k, b in enumerate(meta.bodyId)}
    bl = mx.to_blocks(mx.background_episode(2.5, 0, abdominal_bg=abd), meta, pos)
    assert len(bl) == n_blocks
    rate = np.zeros(len(meta))
    for t0, t1, ix, r in bl:
        assert (t0, t1) == (0.0, 2.0)
        rate[ix] += r
    assert (rate[:7] == 0).all()                     # bristles
    assert np.allclose(rate[7:12], abd_rate)         # abdominal
    assert np.allclose(rate[12:18], 2.5)             # other sensory (incl. head BM)
    assert (rate[18:] == 0).all()                    # non-sensory


def test_legacy_blocks_unchanged(meta):
    pos = {b: k for k, b in enumerate(meta.bodyId)}
    spec = dict(T=2.0, seed=1, spec_seed=None, bg_hz=4.0, words=[])   # Day 4 spec: no bg_scheme
    bl = mx.to_blocks(spec, meta, pos)
    assert len(bl) == 1 and bl[0][2].tolist() == list(range(18)) and bl[0][3] == 4.0
    assert 'bg_scheme' not in mx.background_episode(4.0, 1, legacy=True)


def test_legacy_sampler_matches_day4():
    for s in range(300):
        assert mx.sample_episode(s, legacy=True) == _day4_sample(s)


def test_looming_share_and_range():
    eps = [mx.sample_episode(s) for s in range(5000)]
    share = np.mean([any(w['word'] == 'looming' for w in e['words']) for e in eps])
    assert abs(share - 1 / 6) <= 0.02
    bg = np.array([e['bg_hz'] for e in eps])
    assert bg.min() >= 1.0 and bg.max() <= 3.0
    assert all(2 <= len(e['words']) <= 4 for e in eps)
    assert all(len({w['modality'] for w in e['words']}) == len(e['words']) for e in eps)
    assert all(e['bg_scheme'] == dict(name='class_v1', bristle_hz=0.0, abdominal_hz='same') for e in eps)


def test_bad_scheme_values():
    with pytest.raises(ValueError):
        mx.sample_episode(0, abdominal_bg='half')
    with pytest.raises(ValueError):
        mx.background_episode(1.0, 0, bristle_bg=-1)
