"""Linear readout for the word vocabulary (Week 2 Days 4-5).

Features: log(1 + spike count) per neuron in a window, excluding every
directly stimulated sensory neuron, restricted to neurons that fire in at
least one training trial. L2 logistic regression. Split: seeds 0-6 train,
7-9 test (plan.md Week 2 step 4), plus 10-fold leave-one-seed-out accuracy.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def fit_eval(X, y, seeds, keep, C=1.0):
    X = np.log1p(X[:, keep].astype(np.float32))
    def run(tr, te):
        active = X[tr].sum(axis=0) > 0
        if not active.any():
            return np.full(te.sum(), -1)
        clf = LogisticRegression(C=C, max_iter=3000)
        clf.fit(X[tr][:, active], y[tr])
        return clf.predict(X[te][:, active])
    tr, te = seeds < 7, seeds >= 7
    pred = run(tr, te)
    acc = float((pred == y[te]).mean())
    cm = confusion_matrix(y[te], pred, labels=np.unique(y)).tolist()
    loso = []
    for s in np.unique(seeds):
        p = run(seeds != s, seeds == s); loso.append(float((p == y[seeds == s]).mean()))
    return dict(acc_split=acc, acc_loso=float(np.mean(loso)), confusion=cm,
                n_features=int((X[tr].sum(axis=0) > 0).sum()))
