"""Action readout for the loop (Week 3 Day 5): descending neurons only.
Trained on the Day 4 training trials (5 actions + none, 8 each, every one played
right after a 5-word episode), results/loop/w3d4_<cfg>_train_*.npz. On the 37
held-out Day 4 trials it scores 37/37 (real brain) and 36/37 (shuffled)."""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
L = ROOT / 'results' / 'loop'


def model_path(cfg):
    return L / f'action_readout_{cfg}.pkl'


def train(cfg='real'):
    meta = pd.read_parquet(ROOT / 'data' / 'malecns' / 'malecns_traced_meta.parquet')
    dn = np.flatnonzero((meta.superclass == 'descending_neuron').to_numpy())
    fs = sorted(L.glob(f'w3d4_{cfg}_train_*.npz'), key=lambda f: int(f.stem.split('_')[-2]))
    d = [np.load(f, allow_pickle=True) for f in fs]
    X = np.vstack([x['X'] for x in d])[:, dn]; y = np.concatenate([x['labels'] for x in d])
    act = X.sum(0) > 0; cols = dn[act]
    clf = LogisticRegression(C=1.0, max_iter=5000).fit(np.log1p(X[:, act].astype(np.float32)), y)
    pickle.dump(dict(clf=clf, cols=cols, cfg=cfg), open(model_path(cfg), 'wb'))
    return model_path(cfg), len(y), len(cols)


def load(cfg='real'):
    return pickle.load(open(model_path(cfg), 'rb'))


def decode(model, counts):
    P = model['clf'].predict_proba(np.log1p(np.asarray(counts)[None, model['cols']].astype(np.float32)))[0]
    k = P.argmax()
    return str(model['clf'].classes_[k]), float(P[k])
