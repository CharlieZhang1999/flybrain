"""scripts/w5_behaviour_panel.py scoring helpers on synthetic data (no data files):
AUROC with ties, best-rank, GF silencing of W and gap, and the direction tests."""
import numpy as np
import pandas as pd
from scipy import sparse

from scripts.w5_behaviour_panel import CONDS, auroc, best_rank, hit_sets, score_net, silence_gf


def test_auroc_and_rank():
    s = np.array([5., 4., 3., 2., 1.])
    assert auroc(s, [1, 0, 0, 0, 0]) == 1.0
    assert auroc(s, [0, 0, 0, 0, 1]) == 0.0
    assert auroc(np.zeros(5), [1, 1, 0, 0, 0]) == 0.5        # all tied
    assert best_rank(s, [0, 0, 1, 1, 0]) == 3
    assert best_rank(np.array([1., 1., 0.]), [0, 1, 0]) == 1  # ties share the top rank


def test_silence_gf_zeroes_rows_cols_and_gap():
    n = 6
    meta = pd.DataFrame(dict(type=['DNp01', 'TTMn', 'x', 'DNp01', 'y', 'z']))
    rng = np.random.default_rng(0)
    W = sparse.csc_matrix(rng.random((n, n)))
    C = sparse.csr_matrix(np.array([[0, 0.4, 0], [0, 0, 0], [0.4, 0, 0]]))   # local idx 0,1,2
    m = dict(meta=meta, W=W, gap=dict(idx=np.array([1, 0, 4]), C=C, v_spike=0.002))
    out, info = silence_gf(m)
    D = out['W'].toarray()
    assert not D[[0, 3]].any() and not D[:, [0, 3]].any()
    keep = [1, 2, 4, 5]
    np.testing.assert_array_equal(D[np.ix_(keep, keep)], W.toarray()[np.ix_(keep, keep)])
    Cg = out['gap']['C'].toarray()                             # GF = global 0 = local 1
    assert not Cg[1].any() and not Cg[:, 1].any() and Cg[2, 0] == 0.4
    assert m['gap']['C'].nnz == 2 and m['W'].nnz == n * n       # original untouched


def test_score_net_directions():
    info = pd.DataFrame(dict(
        pool=['cb|L|MN9', 'cb|R|MN9', 'L|wm|TTMn', 'R|wm|TTMn', 'L|hl|a', 'L|fl|b', 'L|ml|c', 'L|ad|d'],
        kind=['cb', 'cb', 'vnc', 'vnc', 'vnc', 'vnc', 'vnc', 'vnc'],
        type=['MN9', 'MN9', 'TTMn', 'TTMn', 'a', 'b', 'c', 'd'],
        subclass=['cb', 'cb', 'wm', 'wm', 'hl', 'fl', 'ml', 'ad']))
    H = hit_sets(info)
    base = np.zeros(len(info))
    prof = {c: base.copy() for c in CONDS}
    for c, v in [('sugar_50', 1), ('sugar_100', 2), ('sugar', 3), ('sugar_200', 4), ('water', 2), ('sugar+bitter', 1)]:
        prof[c][:2] = v
    prof['looming'][2:4] = 10; prof['looming_GFsil'][2:4] = 1
    prof['MDN'][4] = 5; prof['DNg100'][4:7] = 2
    rows = [(c, s, p, prof[c][i]) for c in CONDS for s in (0, 1, 2) for i, p in enumerate(info.pool)]
    R = pd.DataFrame(rows, columns=['cond', 'seed', 'pool', 'rate_hz'])
    out = score_net(R, info, H)
    assert out['P1']['auroc'] == 1.0 and out['P1']['best_rank'] == 1
    assert out['P5']['auroc'] == 1.0 and out['P7']['auroc'] == 1.0 and out['P8']['auroc'] == 1.0
    assert out['P3']['passed'] and out['P4']['passed'] and out['P6']['passed'] and out['P9']['passed']
    assert out['P7']['hl_gt_flml']
    prof['sugar_200'][:2] = 3                                   # flat at the top: not monotonic
    R = pd.DataFrame([(c, s, p, prof[c][i]) for c in CONDS for s in (0, 1, 2) for i, p in enumerate(info.pool)],
                     columns=['cond', 'seed', 'pool', 'rate_hz'])
    assert not score_net(R, info, H)['P9']['passed']
