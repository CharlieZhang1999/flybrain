"""stimuli/states.py: state-set selection, segments, run_trial blocks, labels.

Synthetic annotation table and weights only (no data/, no torch).
"""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from stimuli import states as st


@pytest.fixture(scope='module')
def annot():
    rows = [
        (11, 'ISN', 'Traced', None), (12, 'ISN', 'Traced', None),
        (13, 'ISN', 'Orphan', None),                                  # not traced: out
        (21, 'BiT', 'Traced', None),
        (31, 'pC1_18b', 'Traced', 'male-specific'), (32, 'pC1a', 'Traced', 'male-specific'),
        (33, 'pC1a', 'Traced', 'potentially male-specific'),          # out
        (34, 'pC1c', 'Traced', 'sexually dimorphic'),                 # out
        (35, 'xpC1', 'Traced', 'male-specific'),                      # prefix rule: out
        (41, 'ER5', 'Traced', None), (42, 'ER5', 'Traced', None),
        (51, None, 'Traced', None),
    ]
    return pd.DataFrame(rows, columns=['bodyId', 'type', 'status', 'dimorphism'])


def test_state_sets(annot):
    s = st.state_sets(annot)
    assert s == {'hunger': [11, 12], 'thirst': [21], 'arousal': [31, 32], 'sleep': [41, 42]}
    with pytest.raises(ValueError):
        st.select_state_ids(annot.iloc[:0], 'hunger')
    with pytest.raises(ValueError):
        st.select_state_ids(annot, 'baseline')


def test_state_ids_fixed():
    assert st.STATES[0] == 'baseline' and st.STATE_ID['baseline'] == 0
    assert set(st.STATE_RULES) == set(st.STATES[1:])


def test_segments_validate():
    segs = st.check_segments([(2.0, 4.0, 'hunger', 50), dict(t_on=0, t_off=2.0, state='baseline')], T=4.0)
    assert [s['state'] for s in segs] == ['baseline', 'hunger']
    assert segs[1] == dict(t_on=2.0, t_off=4.0, state='hunger', level_hz=50.0)
    assert st.segment(0.12341, 1.0, 'sleep', 5)['t_on'] == 0.1234
    for bad in ([(0, 2, 'hunger', 10), (1.5, 3, 'thirst', 10)],      # overlap
                [(0, 5, 'hunger', 10)],                                # past T
                [(1, 1, 'hunger', 10)],                                # empty
                [(0, 1, 'hunger', -1)],                                # negative
                [(0, 1, 'baseline', 5)],                               # baseline with drive
                [(0, 1, 'joy', 5)]):                                   # unknown
        with pytest.raises(ValueError):
            st.check_segments(bad, T=4.0)


def test_state_blocks(annot):
    sets = st.state_sets(annot)
    pos = {b: k for k, b in enumerate(annot.bodyId)}
    segs = [(0, 1.0, 'baseline', 0), (1.0, 2.0, 'arousal', 20.0), (2.0, 3.0, 'sleep', 0.0),
            (3.0, 4.0, 'hunger', 100.0)]
    bl = st.state_blocks(segs, sets, pos)
    assert len(bl) == 2                                              # baseline and 0 Hz give no block
    assert bl[0][:2] == (1.0, 2.0) and bl[0][3] == 20.0
    np.testing.assert_array_equal(bl[0][2], [4, 5])
    np.testing.assert_array_equal(bl[1][2], [0, 1])
    assert bl[1][2].dtype == np.int64
    with pytest.raises(KeyError):
        st.state_blocks([(0, 1, 'thirst', 5)], sets, {11: 0})


def test_label_bins():
    sid, lev = st.label_bins([(0, 0.02, 'baseline', 0), (0.02, 0.05, 'thirst', 80.0)], T=0.06, bin_s=0.01)
    np.testing.assert_array_equal(sid, [0, 0, 2, 2, 2, 0])
    np.testing.assert_allclose(lev, [0, 0, 80, 80, 80, 0])
    assert sid.dtype == np.int16 and len(st.label_bins([], 4.0)[0]) == 400


def test_downstream_targets():
    # W[post, pre]; set = {0, 1}
    W = np.zeros((6, 6))
    W[2, 0], W[2, 1] = 3.0, 1.0          # target 2: 4
    W[3, 0] = -5.0                       # target 3: |-5| = 5 (inhibitory counts)
    W[4, 1] = 4.0                        # target 4: 4 (ties broken by index)
    W[1, 0] = 9.0                        # inside the set: excluded
    W[5, 2] = 9.0                        # not from the set
    for M in (W, sparse.csc_matrix(W)):
        idx, w = st.downstream_targets(M, [0, 1], n_top=3)
        np.testing.assert_array_equal(idx, [3, 2, 4])
        np.testing.assert_allclose(w, [5, 4, 4])
    idx, _ = st.downstream_targets(sparse.csc_matrix(W), [0, 1], n_top=50)
    assert len(idx) == 3
