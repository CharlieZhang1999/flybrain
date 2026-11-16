#!/usr/bin/env python
"""Week 3 Day 4: score the action readout. DN-only (1,314 descending neurons) and,
for reference, all non-sensory neurons. Train on the 48 with-history training
trials (5 actions + none, 8 each), test on the 37 held-out trials.
Output: results/loop/w3d4_score.json, printed table."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
ROOT = Path(__file__).resolve().parents[1]
L = ROOT / 'results' / 'loop'
meta = pd.read_parquet(ROOT / 'data' / 'malecns' / 'malecns_traced_meta.parquet')
dn = (meta.superclass == 'descending_neuron').to_numpy()
nonsens = ~meta.superclass.astype(str).str.contains('sensory').to_numpy()


def load(cfg, part):
    fs = sorted(L.glob(f'w3d4_{cfg}_{part}_*.npz'), key=lambda f: int(f.stem.split('_')[-2]))
    d = [np.load(f, allow_pickle=True) for f in fs]
    return (np.vstack([x['X'] for x in d]), np.concatenate([x['labels'] for x in d]),
            np.concatenate([x['seqs'] for x in d]), np.concatenate([x['pre_active'] for x in d]))


out = {}
for cfg in ['real', 'shuffle']:
    try:
        Xtr, ytr, _, _ = load(cfg, 'train'); Xte, yte, ste, pre = load(cfg, 'test')
    except ValueError:
        continue
    for fname, keep in [('DN', dn), ('all non-sensory', nonsens)]:
        A = np.log1p(Xtr[:, keep].astype(np.float32)); B = np.log1p(Xte[:, keep].astype(np.float32))
        act = A.sum(0) > 0
        clf = LogisticRegression(C=1.0, max_iter=5000).fit(A[:, act], ytr)
        P = clf.predict_proba(B[:, act]); pred = clf.classes_[P.argmax(1)]
        ok = pred == yte
        out[f'{cfg}/{fname}'] = dict(correct=int(ok.sum()), n=len(yte), features_active=int(act.sum()),
                                     errors=[f'{s}: {t} -> {p_}' for s, t, p_ in zip(ste[~ok], yte[~ok], pred[~ok])],
                                     confusion=pd.crosstab(pd.Series(yte, name='true'), pd.Series(pred, name='pred')).to_dict())
        print(f'{cfg:8s} {fname:16s} {ok.sum()}/{len(yte)}  (features {act.sum()})', flush=True)
        for e in out[f'{cfg}/{fname}']['errors']:
            print('    ', e)
    out[f'{cfg}/pre_active_median'] = float(np.median(pre))
json.dump(out, open(L / 'w3d4_score.json', 'w'), indent=1)
