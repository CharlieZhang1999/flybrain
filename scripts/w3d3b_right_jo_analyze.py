#!/usr/bin/env python
"""Week 3 Day 3b: DN activity and DN-only readout for A-L/A-R/B-L/B-R, with the
right-JO mirror (w3d3b_mirror_<code>.npz) vs without (w3d3_screen_<code>.npz,
seeds 0-4 of the Day 3 screen). Readout as Day 3: log1p DN spike counts 0-0.5 s,
L2 logistic regression, leave-one-seed-out.
Output: results/loop/w3d3b_dn_summary.csv, w3d3b_dn_top.csv, w3d3b_steering.csv,
        w3d3b_readout.json"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
L = ROOT / 'results' / 'loop'
CODES = ['A-L', 'A-R', 'B-L', 'B-R']
meta = pd.read_parquet(ROOT / 'data' / 'malecns' / 'malecns_traced_meta.parquet')
dn = np.flatnonzero((meta.superclass == 'descending_neuron').to_numpy())
side = meta.instance.astype(str).str[-1].to_numpy()


def load(cfg, c):
    if cfg == 'mirror':
        d = np.load(L / f'w3d3b_mirror_{c}.npz'); return d['stim'], d['tail']
    d = np.load(L / f'w3d3_screen_{c}.npz'); return d['stim'][:5], None


rows, top, steer, readout = [], [], [], {}
for cfg in ['baseline', 'mirror']:
    X, y, s = [], [], []
    for c in CODES:
        st, tail = load(cfg, c)
        rate = st[:, dn].mean(0) / 0.5
        act = rate > 1
        rows.append(dict(config=cfg, code=c, dn_gt1Hz=int(act.sum()),
                         left=int((act & (side[dn] == 'L')).sum()), right=int((act & (side[dn] == 'R')).sum()),
                         brain_active=float((st > 0).sum(1).mean()),
                         tail_neurons_last0p3s=(float((tail > 0).sum(1).mean()) if tail is not None else np.nan),
                         tail_dn_last0p3s=(float((tail[:, dn] > 0).sum(1).mean()) if tail is not None else np.nan)))
        o = np.argsort(-rate)[:10]
        for k in o:
            if rate[k] > 1:
                top.append(dict(config=cfg, code=c, dn=meta.instance.iloc[dn[k]], rate_Hz=round(float(rate[k]), 1)))
        for t in ['DNa01', 'DNa02']:
            for sd in 'LR':
                ix = np.flatnonzero((meta.type == t).to_numpy() & (side == sd))
                steer.append(dict(config=cfg, code=c, dn=f'{t}_{sd}', rate_Hz=float(st[:, ix].mean() / 0.5)))
        X.append(st[:, dn]); y += [c] * len(st); s += list(range(len(st)))
    X = np.log1p(np.vstack(X).astype(np.float32)); y = np.array(y); s = np.array(s)
    pred = np.empty(len(y), dtype=object)
    for k in np.unique(s):
        tr, te = s != k, s == k
        a = X[tr].sum(0) > 0
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(X[tr][:, a], y[tr])
        pred[te] = clf.predict(X[te][:, a])
    cm = pd.crosstab(pd.Series(y, name='true'), pd.Series(pred, name='pred'))
    readout[cfg] = dict(acc=float((pred == y).mean()), confusion=cm.to_dict())
    print(f'{cfg}: LOSO {readout[cfg]["acc"]:.3f}\n{cm.to_string()}')

S = pd.DataFrame(rows); S.to_csv(L / 'w3d3b_dn_summary.csv', index=False)
T = pd.DataFrame(top); T.to_csv(L / 'w3d3b_dn_top.csv', index=False)
St = pd.DataFrame(steer).pivot_table(index='dn', columns=['config', 'code'], values='rate_Hz').round(1)
St.to_csv(L / 'w3d3b_steering.csv')
json.dump(readout, open(L / 'w3d3b_readout.json', 'w'), indent=1)
pd.set_option('display.width', 220)
print(S.to_string()); print(St.to_string())
for (cfg, c), g in T.groupby(['config', 'code']):
    print(cfg, c, ', '.join(f'{r.dn} {r.rate_Hz}' for r in g.itertuples()))
