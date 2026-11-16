#!/usr/bin/env python
"""Week 3 Day 3b: DN-only readout of the 8 JO codes; pick the 5 best-separated.
Features: log(1 + spike count) of the 1,314 descending neurons in 0-0.5 s (JO and
all other sensory neurons excluded by construction). L2 logistic regression,
leave-one-seed-out. For every 5-of-8 subset, LOSO accuracy is recomputed; the
best subset (ties: higher mean top-probability margin) is chosen.
Also: DNa01 / DNa02 (steering DNs) rates per code, left vs right.
Output: results/loop/w3d3_select.json, w3d3_dn_steering.csv"""
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT))  # noqa: E402,E702
from loop.actions import CANDIDATES  # noqa: E402

L = ROOT / 'results' / 'loop'
meta = pd.read_parquet(ROOT / 'data' / 'malecns' / 'malecns_traced_meta.parquet')
dn = np.flatnonzero((meta.superclass == 'descending_neuron').to_numpy())
X, y, s = [], [], []
for c in CANDIDATES:
    d = np.load(L / f'w3d3_screen_{c}.npz')['stim']
    X.append(d[:, dn]); y += [c] * len(d); s += list(range(len(d)))
X = np.log1p(np.vstack(X).astype(np.float32)); y = np.array(y); s = np.array(s)
print(f'DN features: {len(dn)}, active in any trial: {(X.sum(0) > 0).sum()}')


def loso(codes):
    m = np.isin(y, codes); Xm, ym, sm = X[m], y[m], s[m]
    pred, marg = np.empty(len(ym), dtype=object), np.zeros(len(ym))
    for k in np.unique(sm):
        tr, te = sm != k, sm == k
        act = Xm[tr].sum(0) > 0
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(Xm[tr][:, act], ym[tr])
        P = clf.predict_proba(Xm[te][:, act]); pred[te] = clf.classes_[P.argmax(1)]
        Ps = np.sort(P, 1); marg[te] = Ps[:, -1] - Ps[:, -2]
    return float((pred == ym).mean()), float(marg.mean()), pred, ym


acc8, marg8, pred8, y8 = loso(CANDIDATES)
cm8 = pd.crosstab(pd.Series(y8, name='true'), pd.Series(pred8, name='pred'))
print(f'all 8 codes: LOSO {acc8:.3f}, mean margin {marg8:.2f}\n{cm8.to_string()}')
res = []
for sub in itertools.combinations(CANDIDATES, 5):
    a, mg, _, _ = loso(list(sub)); res.append((a, mg, sub))
res.sort(key=lambda r: (-r[0], -r[1]))
for r in res[:8]:
    print(f'{r[0]:.3f} margin {r[1]:.2f} {r[2]}')
best = res[0]
# steering DNs
rows = []
for c in CANDIDATES:
    d = np.load(L / f'w3d3_screen_{c}.npz')['stim']
    for t in ['DNa01', 'DNa02', 'DNa03', 'DNg13']:
        for side in ['_L', '_R']:
            ix = np.flatnonzero((meta.type == t).to_numpy() & meta.instance.astype(str).str.endswith(side).to_numpy())
            rows.append(dict(code=c, dn=t, side=side[1], rate_Hz=float(d[:, ix].mean() / 0.5) if len(ix) else np.nan))
st = pd.DataFrame(rows).pivot_table(index=['dn', 'side'], columns='code', values='rate_Hz').round(1)
print(st.to_string())
st.to_csv(L / 'w3d3_dn_steering.csv')
json.dump(dict(all8=dict(acc=acc8, margin=marg8, confusion=cm8.to_dict()),
               ranking=[dict(acc=a, margin=m, codes=list(c)) for a, m, c in res],
               best=list(best[2])), open(L / 'w3d3_select.json', 'w'), indent=1)
