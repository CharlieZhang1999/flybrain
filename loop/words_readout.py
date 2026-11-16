"""Word readout for the loop (Week 3 Day 2).

Trained once on isolated trials of the loop's brain model (calibrated MaleCNS +
presynaptic modulation, K = 0.05): 17 words x 10 seeds + 10 no-drive trials.
Features: log(1 + spike count) per non-sensory brain neuron in a 500 ms window,
L2 logistic regression (same as readout/linear.py). On the Day 1 episodes this
readout is 100% correct (results/loop/w3d1_carryover_presyn0.05.csv).
"""
import pickle
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
L = ROOT / 'results' / 'loop'
DEFAULT = 'presyn0.05'


def model_path(config=DEFAULT):
    return L / f'words_readout_{config}.pkl'
UNSURE = 0.5          # top probability below this -> 'unsure', left out of the prompt


def train(config=DEFAULT):
    import pandas as pd
    from stimuli.malecns_words import WORDS, select_ids
    meta = pd.read_parquet(ROOT / 'data' / 'malecns' / 'malecns_traced_meta.parquet')
    pos = {b: i for i, b in enumerate(meta.bodyId)}
    sens = np.zeros(len(meta), bool)
    for w in WORDS:
        sens[[pos[b] for b in select_ids(meta, w)]] = True
    fs = sorted(L.glob(f'w3d1g_stim_{config}_*.npz'), key=lambda f: int(f.stem.split('_')[-2]))
    X = sparse.vstack([sparse.load_npz(f) for f in fs]).tocsr()
    lab = np.concatenate([np.load(str(f).replace('_stim_', '_info_'))['labels'] for f in fs])
    y = [list(WORDS)[k] for k in lab]
    fn = list(L.glob(f'w3d1_ep_none_*_{config}_X.npz'))
    # no-drive trials are silent in every config (0 spikes), so zeros stand in when not simulated
    Xn = (sparse.vstack([sparse.load_npz(f) for f in fn]) if fn
          else sparse.csr_matrix((10, X.shape[1]), dtype=X.dtype))
    X = sparse.vstack([X, Xn]).tocsr(); y = np.array(y + ['none'] * Xn.shape[0])
    cols = np.flatnonzero(~sens & (np.asarray(X.sum(0)).ravel() > 0))
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(np.log1p(X[:, cols].toarray().astype(np.float32)), y)
    out = model_path(config)
    pickle.dump(dict(clf=clf, cols=cols, n=len(meta), config=config), open(out, 'wb'))
    return out, len(y), len(cols)


def load(config=DEFAULT):
    return pickle.load(open(model_path(config), 'rb'))


def decode(model, counts):
    """counts: (n_windows, n_neurons) spike counts. Returns [(word, p_top)]."""
    X = np.log1p(np.asarray(counts)[:, model['cols']].astype(np.float32))
    P = model['clf'].predict_proba(X)
    k = P.argmax(1)
    return [(str(model['clf'].classes_[i]), float(P[j, i])) for j, i in enumerate(k)]
