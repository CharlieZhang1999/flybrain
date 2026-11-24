"""sim.cns.presyn_mirror_shuffle: order of shuffle, build_presyn and the right-JO
mirror (T5b), on a synthetic network and type table (no data files, no pyarrow).

Bug (T5a): shuffle -> presyn -> mirror lets the mirror copy the SHUFFLED left JO
outputs, which point at random targets, so the right JO template becomes dense
(real model, seed 12345: nnz 46.8M vs 24.5M calib). Fixed order: mirror the real
W, then shuffle, so nnz stays at the calib value up to merged duplicates.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from sim.cns import presyn_mirror_shuffle
from sim.jo_mirror import mirror_jo
from sim.presyn import build_presyn

K = 0.05


def shuffle_W(W, seed=12345):
    """Copy of scripts.w2d4_word_trials.shuffle_W (that module needs pyarrow)."""
    C = W.tocoo()
    perm = np.random.default_rng(seed).permutation(len(C.data))
    Ws = sparse.csc_matrix((C.data[perm], (C.row, C.col[perm])), shape=W.shape)
    Ws.sum_duplicates()
    return Ws


def _net(seed=3, n_jo=40, n_types=80, per_side=2):
    """n_jo left + n_jo right JO-A1 afferents (sensory) and n_types central types
    with per_side neurons on each side. Each JO neuron contacts 6 central
    neurons of 3 fixed ipsilateral types (a sparse, stereotyped projection);
    central neurons are randomly wired (density 5%) and also synapse onto JO
    terminals (sensory rows, removed by build_presyn)."""
    rng = np.random.default_rng(seed)
    typ, side, sc = [], [], []
    for s in 'LR':
        typ += ['JO-A1'] * n_jo; side += [s] * n_jo; sc += ['sensory'] * n_jo
    for t in range(n_types):
        for s in 'LR':
            typ += [f'C{t}'] * per_side; side += [s] * per_side; sc += ['central'] * per_side
    n = len(typ)
    typ, side = np.array(typ), np.array(side)
    meta = pd.DataFrame(dict(type=typ, instance=[f'{t}_{s}' for t, s in zip(typ, side)],
                             somaSide=side, superclass=sc))
    jo = np.flatnonzero(typ == 'JO-A1')
    cen = np.flatnonzero(typ != 'JO-A1')
    rows, cols = [], []
    for j in jo:
        tgt = cen[np.isin(typ[cen], ['C0', 'C1', 'C2']) & (side[cen] == side[j])]
        r = rng.choice(tgt, 6, replace=False)
        rows += list(r); cols += [j] * 6
    post, pre = np.nonzero(rng.random((n, len(cen))) < 0.05)
    rows += list(post); cols += list(cen[pre])
    w = rng.normal(0, 1e-3, len(rows)); w[w == 0] = 1e-4
    W = sparse.csc_matrix((w, (rows, cols)), shape=(n, n))
    W.sum_duplicates()
    sens = (np.array(sc) == 'sensory')
    return W, meta, sens


def _run(W, meta, sens, config, order='fixed'):
    return presyn_mirror_shuffle(W, sens, K, mirror_fn=lambda X: mirror_jo(X, meta),
                                 shuffle_fn=shuffle_W if config == 'shuffle' else None, order=order)


def test_calib_path_is_presyn_then_mirror():
    W, meta, sens = _net()
    Wc, ps, _, _, _ = _run(W, meta, sens, 'calib')
    W0, ps0, _ = build_presyn(W, sens, K)
    W0, _ = mirror_jo(W0, meta)
    assert (Wc != W0).nnz == 0 and Wc.nnz == W0.nnz
    assert (ps['Wp'] != ps0['Wp']).nnz == 0 and np.array_equal(ps['idx'], ps0['idx'])


def test_fixed_order_keeps_nnz_legacy_inflates():
    W, meta, sens = _net()
    Wc = _run(W, meta, sens, 'calib')[0]
    Wf, _, _, _, ex = _run(W, meta, sens, 'shuffle', 'fixed')
    Wl = _run(W, meta, sens, 'shuffle', 'legacy')[0]
    # the order matters: legacy right JO template is dense (mean of random targets)
    assert Wl.nnz > 1.5 * Wc.nnz
    assert abs(Wf.nnz - Wc.nnz) <= 0.05 * Wc.nnz
    assert ex['mirror_sensory_row_entries'] == 0
    # fixed: the shuffle input is the mirrored real network (calib W) plus the
    # original sensory rows, and shuffling keeps every row's entry count
    again = _run(W, meta, sens, "shuffle", "fixed")
    assert ex['nnz_shuffle_input'] == Wc.nnz + int(sens[W.tocoo().row].sum())
    assert (again[0] != Wf).nnz == 0                       # deterministic


def test_fixed_shuffle_permutes_mirrored_real():
    """With an identity 'shuffle' the fixed order reproduces calib exactly."""
    W, meta, sens = _net()
    Wc, psc, _, _, _ = _run(W, meta, sens, 'calib')
    Wi, psi, _, _, _ = presyn_mirror_shuffle(W, sens, K, mirror_fn=lambda X: mirror_jo(X, meta),
                                             shuffle_fn=lambda X: X.copy(), order='fixed')
    assert (Wi != Wc).nnz == 0 and Wi.nnz == Wc.nnz
    assert (psi['Wp'] != psc['Wp']).nnz == 0


def test_bad_order_raises():
    W, meta, sens = _net()
    with pytest.raises(ValueError):
        _run(W, meta, sens, 'shuffle', 'nope')
