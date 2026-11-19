"""sim.cns loop_gain edge selection (scale_set_edges), on a synthetic W and type
table (no data files, no pyarrow): every entry whose pre- AND postsynaptic
neuron types are both in the set is scaled, including cross-type and same-type
entries; every other entry is unchanged, and the sparsity pattern is kept."""
import numpy as np
import pytest
from scipy import sparse

from sim.cns import LOOP_GAIN, W4_CALIB, scale_set_edges


def _net(n=60, seed=7):
    rng = np.random.default_rng(seed)
    types = np.array(['A', 'B', 'C', 'X', 'Y'])[rng.integers(0, 5, n)]
    types[:5] = ['A', 'B', 'C', 'X', 'Y']                  # every type present
    post, pre = np.nonzero(rng.random((n, n)) < 0.3)
    w = rng.normal(0, 1e-3, len(post))
    w[w == 0] = 1e-4
    return sparse.csc_matrix((w, (post, pre)), shape=(n, n)), types


def test_scales_exactly_in_set_entries():
    W, types = _net()
    S = {'A', 'B', 'C'}
    W2 = scale_set_edges(W, types, S, 0.3)
    D, D2 = W.toarray(), W2.toarray()
    inS = np.isin(types, list(S))
    both = inS[:, None] & inS[None, :]                     # [post, pre]
    assert np.array_equal(D2[~both], D[~both])             # all others untouched (bitwise)
    np.testing.assert_array_equal(D2[both], D[both] * 0.3)
    # cross-type in-set entries exist and are scaled (not only same-type ones)
    cross = both & (types[:, None] != types[None, :]) & (D != 0)
    assert cross.sum() > 0
    np.testing.assert_array_equal(D2[cross], D[cross] * 0.3)
    # one endpoint in the set is not enough
    half = (inS[:, None] ^ inS[None, :]) & (D != 0)
    assert half.sum() > 0 and np.array_equal(D2[half], D[half])
    assert W2.nnz == W.nnz


def test_gain_one_is_identity_and_missing_type_raises():
    W, types = _net()
    assert (scale_set_edges(W, types, {'A', 'B'}, 1.0) != W).nnz == 0
    with pytest.raises(ValueError):
        scale_set_edges(W, types, {'A', 'nope'}, 0.3)


def test_w4_calib_constant():
    types, g = LOOP_GAIN
    assert set(types) == {'DNge027', 'GNG423', 'DNg21', 'DNg62', 'DNge044'} and g == 0.3
    assert W4_CALIB['loop_gain'] is LOOP_GAIN
