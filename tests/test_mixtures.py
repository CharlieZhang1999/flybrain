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
        + [dict(superclass='sensory_ascending', **{'class': 'unknown_sensory'}, subclass='abdomen',
                somaNeuromere=None, type='SAxx01')] * 2                  # O2 (rows 7-8)
        + [dict(superclass='vnc_sensory', **{'class': 'unknown_sensory'}, subclass='abdomen',
                somaNeuromere=None, type='SNxx16')] * 1                  # O2 (row 9)
        + [dict(superclass='vnc_sensory', **{'class': 'unknown_sensory'}, subclass='abdomen',
                somaNeuromere=None, type=None)] * 4
        + [dict(superclass='sensory_ascending', **{'class': 'unknown_sensory'}, subclass='abdomen',
                somaNeuromere=None, type='SNxx03')] * 1
        + [dict(superclass='vnc_sensory', **{'class': 'chordotonal'}, subclass='chordotonal organ',
                somaNeuromere=None, type='CH')] * 3
        + [dict(superclass='cb_sensory', **{'class': 'mechanosensory'}, subclass=None,
                somaNeuromere=None, type='BM_InOm')] * 3          # head bristle: not in the mask
        + [dict(superclass='sensory_ascending', **{'class': None}, subclass=None,
                somaNeuromere=None, type='SAxx01')] * 1           # SAxx01 outside the abdominal mask: not O2
        + [dict(superclass='vpn', **{'class': None}, subclass=None, somaNeuromere=None, type='LC4')] * 2
        + [dict(superclass='vnc_motor', **{'class': None}, subclass=None, somaNeuromere='T1', type='MN')] * 2
    )
    m = pd.DataFrame(rows)
    m.insert(0, 'bodyId', np.arange(len(m)) + 100)
    return m


# rows: 0-6 bristle, 7-9 O2, 10-14 other abdominal, 15-21 other sensory, 22-25 non-sensory
BRI, O2, ABD, OTHER, NONSENS = range(0, 7), range(7, 10), range(10, 15), range(15, 22), range(22, 26)
N_SENS = 22


def _rates(bl, n):
    rate = np.zeros(n)
    for t0, t1, ix, r in bl:
        assert t0 == 0.0
        assert len(set(ix.tolist())) == len(ix)
        rate[ix] += r
    return rate


def test_masks(meta):
    mk = mx.bg_class_masks(meta)
    assert mk['sensory'].sum() == N_SENS
    assert np.flatnonzero(mk['bristle']).tolist() == list(BRI)
    assert np.flatnonzero(mk['abdominal']).tolist() == list(O2) + list(ABD)
    assert np.flatnonzero(mk['o2']).tolist() == list(O2)          # row 21 (SAxx01, not abdominal) excluded


@pytest.mark.parametrize('bg', [1.0, 1.7, 2.5, 3.0])
def test_default_v2_rates(meta, bg):
    """Default scheme: bristle 0, O2 0, other abdominal exactly 1 Hz at any episode rate, rest bg."""
    pos = {b: k for k, b in enumerate(meta.bodyId)}
    spec = mx.background_episode(bg, 0, T=1.0)
    assert spec['bg_scheme'] == dict(name='class_v2', bristle_hz=0.0, o2_hz=0.0, abdominal_hz=1.0)
    bl = mx.to_blocks(spec, meta, pos)
    rate = _rates(bl, len(meta))
    assert all(b[1] == 1.0 for b in bl)
    assert (rate[list(BRI)] == 0).all()
    assert (rate[list(O2)] == 0).all()
    assert (rate[list(ABD)] == 1.0).all()
    assert (rate[list(OTHER)] == bg).all()
    assert (rate[list(NONSENS)] == 0).all()
    assert len(bl) == (1 if bg == 1.0 else 2)        # abdominal merges into the bg block when bg == 1 Hz


@pytest.mark.parametrize('kw, rb, ro, ra', [
    (dict(abdominal_bg='same'), 0.0, 0.0, 2.5),
    (dict(abdominal_bg=0.0), 0.0, 0.0, 0.0),
    (dict(o2_bg=0.5, bristle_bg=0.2, abdominal_bg=2.0), 0.2, 0.5, 2.0),
    (dict(o2_bg=2.5), 0.0, 2.5, 1.0)])
def test_v2_options(meta, kw, rb, ro, ra):
    pos = {b: k for k, b in enumerate(meta.bodyId)}
    rate = _rates(mx.to_blocks(mx.background_episode(2.5, 0, **kw), meta, pos), len(meta))
    assert (rate[list(BRI)] == rb).all() and (rate[list(O2)] == ro).all() and (rate[list(ABD)] == ra).all()
    assert (rate[list(OTHER)] == 2.5).all() and (rate[list(NONSENS)] == 0).all()


def _v1_blocks(spec, meta):
    """Frozen copy of the class_v1 bg_blocks (git 9f5ef11), for the class_v1 check."""
    T, bg, sc = spec['T'], float(spec['bg_hz']), spec['bg_scheme']
    mk = mx.bg_class_masks(meta)
    b_hz = float(sc['bristle_hz'])
    a_hz = bg if sc['abdominal_hz'] == 'same' else float(sc['abdominal_hz'])
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


@pytest.mark.parametrize('abd, bri', [('same', 0.0), (0.0, 0.0), (1.0, 0.0), (2.0, 0.5)])
def test_class_v1_blocks_unchanged(meta, abd, bri):
    pos = {b: k for k, b in enumerate(meta.bodyId)}
    spec = dict(T=1.0, seed=0, spec_seed=None, bg_hz=2.5, words=[],
                bg_scheme=dict(name='class_v1', bristle_hz=bri, abdominal_hz=abd))   # as saved on 9f5ef11
    assert mx.background_episode(2.5, 0, T=1.0, scheme='class_v1', bristle_bg=bri, abdominal_bg=abd) == spec
    a, b = mx.to_blocks(spec, meta, pos), _v1_blocks(spec, meta)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x[0] == y[0] and x[1] == y[1] and x[3] == y[3] and np.array_equal(x[2], y[2])
    rate = _rates(a, len(meta))
    assert (rate[list(O2)] == rate[list(ABD)][0]).all()            # class_v1 has no O2 class


def test_legacy_blocks_unchanged(meta):
    pos = {b: k for k, b in enumerate(meta.bodyId)}
    spec = dict(T=2.0, seed=1, spec_seed=None, bg_hz=4.0, words=[])   # Day 4 spec: no bg_scheme
    bl = mx.to_blocks(spec, meta, pos)
    assert len(bl) == 1 and bl[0][2].tolist() == list(range(N_SENS)) and bl[0][3] == 4.0
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
    assert all(e['bg_scheme'] == dict(name='class_v2', bristle_hz=0.0, o2_hz=0.0, abdominal_hz=1.0) for e in eps)


def test_scheme_does_not_change_draws():
    for s in range(200):
        a, b = mx.sample_episode(s), mx.sample_episode(s, scheme='class_v1', abdominal_bg='same')
        assert {k: v for k, v in a.items() if k != 'bg_scheme'} == {k: v for k, v in b.items() if k != 'bg_scheme'}
        assert b['bg_scheme'] == dict(name='class_v1', bristle_hz=0.0, abdominal_hz='same')


def test_bad_scheme_values():
    with pytest.raises(ValueError):
        mx.sample_episode(0, abdominal_bg='half')
    with pytest.raises(ValueError):
        mx.background_episode(1.0, 0, bristle_bg=-1)
    with pytest.raises(ValueError):
        mx.background_episode(1.0, 0, o2_bg=-1)
    with pytest.raises(ValueError):
        mx.background_episode(1.0, 0, scheme='class_v3')


def test_t_words_default_unchanged():
    """t_words=None (default) gives the same specs as before: no new key, same draws."""
    for s in range(200):
        for kw in (dict(), dict(legacy=True), dict(scheme='class_v1', abdominal_bg='same')):
            a = mx.sample_episode(s, **kw)
            assert 't_words' not in a
            b = mx.sample_episode(s, t_words=2.0, **kw)          # t_words = T: same draws and times
            assert b.pop('t_words') == 2.0 and a == b


def test_t_words_window():
    """Day 5 dataset: T = 2.5 s, words in [0, 2.0] s, onset in [0, 1.7] s."""
    for s in range(2000):
        e = mx.sample_episode(s, T=2.5, t_words=2.0)
        assert e['T'] == 2.5 and e['t_words'] == 2.0
        for w in e['words']:
            assert 0.0 <= w['onset'] <= 1.7 and w['onset'] < w['offset'] <= 2.0
        assert mx.last_offset(e) <= 2.0
        # the same draws as a 2.0 s episode: only T and t_words differ
        f = mx.sample_episode(s, T=2.0)
        assert {k: v for k, v in e.items() if k not in ('T', 't_words')} == {k: v for k, v in f.items() if k != 'T'}


def test_t_words_bad_values():
    for tw in (0.0, 0.3, 2.6):
        with pytest.raises(ValueError):
            mx.sample_episode(0, T=2.5, t_words=tw)
