#!/usr/bin/env python
"""W4D5 T2e: static check of sim.cns LOOP_GAIN (no simulation).

Builds, in one process:
  old   W4_CALIB of the reference sim/cns.py (default: git HEAD copy given as argv[1],
        i.e. the module before LOOP_GAIN), calib and shuffle
  none  current cns_model(..., **W4_CALIB minus loop_gain)  (loop_gain=None)
  new   current cns_model(..., **W4_CALIB)
  t2c   old calib W with LOOP5 x 0.3 applied exactly as scripts/w4d5_burst_runs.loop_gain_W
Reports entry-level differences (same sparsity pattern, values compared bitwise).
Usage:  python scripts/w4d5_loopgain_wcheck.py <path to old cns.py>
Output: results/w4/regression/loop_gain/wcheck.txt
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'results' / 'w4' / 'regression' / 'loop_gain'
LINES = []


def log(s):
    print(s, flush=True); LINES.append(s)


def diff(A, B):
    """(pattern equal, n entries whose values differ, row/col of those) for two csc matrices."""
    A = A.tocsc(); B = B.tocsc(); A.sort_indices(); B.sort_indices()
    same_pat = (A.shape == B.shape and np.array_equal(A.indptr, B.indptr)
                and np.array_equal(A.indices, B.indices))
    if not same_pat:
        D = (A - B).tocoo(); D.eliminate_zeros()
        return False, D.nnz, D.row, D.col
    k = A.data != B.data
    col = np.repeat(np.arange(A.shape[1]), np.diff(A.indptr))
    return True, int(k.sum()), A.indices[k], col[k]


def main(old_path):
    from sim.cns import cns_model, W4_CALIB, LOOP_GAIN
    from scripts.w4d5_burst_runs import LOOP5, loop_gain_W
    spec = importlib.util.spec_from_file_location('cns_old', old_path)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    log(f'old module: {old_path}; old W4_CALIB keys {sorted(old.W4_CALIB)}; new keys {sorted(W4_CALIB)}')
    log(f'LOOP_GAIN = {sorted(LOOP_GAIN[0])} x {LOOP_GAIN[1]}; T2c LOOP5 = {sorted(LOOP5)}')
    kw_none = {k: v for k, v in W4_CALIB.items() if k != 'loop_gain'}
    for cfg in ('calib', 'shuffle'):
        mo = old.cns_model('cns', cfg, verbose=False, **old.W4_CALIB)
        mn = cns_model('cns', cfg, verbose=False, **kw_none)
        mw = cns_model('cns', cfg, verbose=False, **W4_CALIB)
        meta = mo['meta']
        ty = meta.type.fillna('?').astype(str).values
        L = np.isin(ty, LOOP5)
        Wo, Wn, Ww = mo['W'], mn['W'], mw['W']
        log(f'[{cfg}] nnz old {Wo.nnz:,} none {Wn.nnz:,} new {Ww.nnz:,}')
        p, n, _, _ = diff(Wo, Wn)
        log(f'[{cfg}] old vs loop_gain=None: same pattern {p}, differing entries {n}')
        p, n, r, c = diff(Wo, Ww)
        log(f'[{cfg}] old vs new W4_CALIB: same pattern {p}, differing entries {n}, '
            f'of which pre and post both LOOP5: {int((L[r] & L[c]).sum())}')
        # other model parts unchanged
        same_gap = all(np.array_equal(np.asarray(mo['gap'][k].toarray() if hasattr(mo['gap'][k], 'toarray') else mo['gap'][k]),
                                      np.asarray(mw['gap'][k].toarray() if hasattr(mw['gap'][k], 'toarray') else mw['gap'][k]))
                       for k in mo['gap'])
        log(f'[{cfg}] gap dict identical old vs new: {same_gap}')
        if cfg == 'calib':
            class _M: pass
            M = _M(); M.meta = meta; M.m = mo
            Wt, ne = loop_gain_W(M, 0.3)
            log(f'[calib] T2c loop_gain_W(0.3): {ne} entries scaled')
            p, n, _, _ = diff(Wt, Ww)
            log(f'[calib] T2c W vs new W4_CALIB W: same pattern {p}, differing entries {n}')
            Co = Wo.tocoo(); k = L[Co.row] & L[Co.col]
            pr = pd.crosstab(pd.Series(ty[Co.col[k]], name='pre'), pd.Series(ty[Co.row[k]], name='post'))
            log('[calib] LOOP5 entries by type (rows pre, cols post):\n' + pr.to_string())
            log(f'[calib] same-type entries {int((ty[Co.row[k]] == ty[Co.col[k]]).sum())}, '
                f'cross-type {int((ty[Co.row[k]] != ty[Co.col[k]]).sum())}')
            nw = Ww.tocsc()[:, np.flatnonzero(L)].tocsr()[np.flatnonzero(L)]
            no = Wo.tocsc()[:, np.flatnonzero(L)].tocsr()[np.flatnonzero(L)]
            log(f'[calib] sum of LOOP5 block: old {no.sum():.6g}, new {nw.sum():.6g}, ratio {nw.sum() / no.sum():.6f}')
        del mo, mn, mw
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'wcheck.txt').write_text('\n'.join(LINES) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
