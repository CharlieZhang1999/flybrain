#!/usr/bin/env python
"""Week 3 Day 3b: loop brain model (presyn_model(0.05)) + right-JO mirror (sim/jo_mirror.py).
1. Builds the mirror, checks that every non-right-JO column of W is unchanged and
   compares size-matched L/R JO-A, JO-B output weight before and after.
2. Runs A-L, A-R, B-L, B-R (150 Hz, 0-0.5 s; run 1.0 s), seeds 0-4, one process.
Output: results/loop/w3d3b_mirror_<code>.npz (counts 0-0.5, 0.5-0.7, 0.7-1.0 s),
        w3d3b_mirror_assign.csv, w3d3b_mirror_templates.csv, w3d3b_mirror_check.json"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from w3d1_common import presyn_model  # noqa: E402
from sim.lif_numpy import run_trial  # noqa: E402
from sim.jo_mirror import mirror_jo  # noqa: E402
from loop.actions import blocks, jo_groups  # noqa: E402

OUT = ROOT / 'results' / 'loop'
R, W, gap, ps, _ = presyn_model(0.05)
meta = R.meta; n = len(meta)
W2, info = mirror_jo(W, meta)

# --- structural checks
dst = info['assign'].idx.to_numpy()
other = np.ones(n, bool); other[dst] = False
D = (W2 - W).tocsc()
changed_cols = np.flatnonzero(np.diff(D.indptr) > 0)
chk = dict(n_right_jo=int(len(dst)), cols_changed=int(len(changed_cols)),
           cols_changed_outside_right_jo=int((~np.isin(changed_cols, dst)).sum()),
           max_abs_diff_other_cols=float(abs(D[:, np.flatnonzero(other)]).max()) if D.nnz else 0.0,
           nnz_before=int(W.nnz), nnz_after=int(W2.nnz),
           out_src_total=info['out_src_total'], out_dst_before=info['out_dst_before'],
           out_dst_after=info['out_dst_after'])
g = jo_groups(meta)
for k in ['A-L', 'A-R', 'B-L', 'B-R']:
    chk[f'out_{k}_before'] = float(W[:, g[k]].sum()); chk[f'out_{k}_after'] = float(W2[:, g[k]].sum())
    chk[f'partners_{k}_after'] = int((abs(W2[:, g[k]]).sum(1) > 0).sum())
chk['rule_counts'] = info['assign'].rule.value_counts().to_dict()
json.dump(chk, open(OUT / 'w3d3b_mirror_check.json', 'w'), indent=1)
A = info['assign'].copy(); A['bodyId'] = meta.bodyId.to_numpy()[A.idx]
A.to_csv(OUT / 'w3d3b_mirror_assign.csv', index=False)
pd.DataFrame(info['templates']).T.to_csv(OUT / 'w3d3b_mirror_templates.csv')
print(json.dumps(chk, indent=1), flush=True)

# --- simulations (one process)
p = dict(R.p); p['t_run'] = 1.0
for code in ['A-R', 'B-R', 'A-L', 'B-L']:
    S, M, T = [], [], []
    for seed in range(5):
        r = run_trial(W2, [], params=p, seed=seed, apl=R.apl, record=True, gap=gap, presyn=ps,
                      blocks=blocks(code, meta))
        ts, si = r['t'] * p['dt'], r['i']
        S.append(np.bincount(si[ts < 0.5], minlength=n))
        M.append(np.bincount(si[(ts >= 0.5) & (ts < 0.7)], minlength=n))
        T.append(np.bincount(si[ts >= 0.7], minlength=n))
        print(f'{code} s{seed}: active={int((S[-1] > 0).sum())} tail(0.7-1.0)={int((T[-1] > 0).sum())}', flush=True)
    np.savez_compressed(OUT / f'w3d3b_mirror_{code}.npz', stim=np.array(S, np.int32),
                        mid=np.array(M, np.int32), tail=np.array(T, np.int32))
print('DONE')
