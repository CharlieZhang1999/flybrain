#!/usr/bin/env python
"""Week 3 Day 1b/1c: decode episode windows; apply the Day 1 decision rule.

Readout (readout/linear.py conventions): log(1 + count) per neuron in the
500 ms window, directly stimulated sensory neurons excluded, L2 logistic
regression (C = 1). Two feature sets: all non-sensory brain neurons, and
DNs + brain MNs. 18 classes: the 17 words + 'none' (10 no-drive trials).

A. isolated readout: trained on all Week 2 calibrated isolated trials
   (17 words x 10 seeds, results/readout/stim_calib.npz) + 'none'; tested on
   every episode word window for each spacing (0 / 250 / 500 ms), and on the
   500 ms gap windows (should read 'none').
B. episode-trained readout: isolated trials + word windows of episodes 0-27,
   tested on episodes 28-39, per spacing.
Also: accuracy by previous word, and the fraction of windows with top
probability < 0.5 ('unsure').
Output: results/loop/w3d1_carryover.csv, w3d1_by_prev.csv, w3d1_summary.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT))  # noqa: E402,E702
from stimuli.malecns_words import WORDS, select_ids  # noqa: E402

L, RD = ROOT / 'results' / 'loop', ROOT / 'results' / 'readout'
CLASSES = list(WORDS) + ['none']
GAPS = ['0', '250', '500']
N_TRAIN_EP = 28
sys.path.insert(0, str(ROOT / 'scripts'))
from w3d1_episodes import sequences  # noqa: E402
SEQS = sequences()
SEEN_CH = {e: {'cold', 'humid'} for e in range(len(SEQS))}


CONFIG = sys.argv[1] if len(sys.argv) > 1 else 'calib'
SUF = '' if CONFIG == 'calib' else f'_{CONFIG}'


def load_eps(tag_prefix):
    Xs, infos = [], []
    pat = f'w3d1_ep_{tag_prefix}_*_info.npz' if CONFIG == 'calib' else f'w3d1_ep_{tag_prefix}_*_{CONFIG}_info.npz'
    for f in sorted(L.glob(pat)):
        if CONFIG == 'calib' and f.name.count('_') != 5:   # skip other configs
            continue
        Xs.append(sparse.load_npz(str(f).replace('_info.npz', '_X.npz')))
        infos.append(np.load(f, allow_pickle=True)['info'])
    info = pd.DataFrame(np.concatenate(infos), columns=['config', 'ep', 'k', 'kind', 'word', 'prev'])
    info[['ep', 'k']] = info[['ep', 'k']].astype(int)
    return sparse.vstack(Xs).tocsr(), info


def fit(X, y):
    clf = LogisticRegression(C=1.0, max_iter=3000)
    clf.fit(np.log1p(X.astype(np.float32)), y)
    return clf


def predict(clf, X):
    P = clf.predict_proba(np.log1p(X.astype(np.float32)))
    return clf.classes_[P.argmax(1)], P.max(1)


def main():
    meta = pd.read_parquet(ROOT / 'data' / 'malecns' / 'malecns_traced_meta.parquet')
    pos = {b: i for i, b in enumerate(meta.bodyId)}
    sens = np.zeros(len(meta), bool)
    for w in WORDS:
        sens[[pos[b] for b in select_ids(meta, w)]] = True
    feats = {'all': ~sens,
             'motor': meta.superclass.isin(['descending_neuron', 'cb_motor']).to_numpy() & ~sens}

    if CONFIG == 'calib':
        tr = np.load(RD / 'trials_calib.npz', allow_pickle=True)
        Xiso = sparse.load_npz(RD / 'stim_calib.npz').tocsr(); lab = tr['labels']
    else:
        fs = sorted(L.glob(f'w3d1g_stim_{CONFIG}_*.npz'), key=lambda f: int(f.stem.split('_')[-2]))
        Xiso = sparse.vstack([sparse.load_npz(f) for f in fs]).tocsr()
        lab = np.concatenate([np.load(str(f).replace('_stim_', '_info_'))['labels'] for f in fs])
    yiso = np.array([list(WORDS)[k] for k in lab])
    Xn, _ = load_eps('none')
    Xiso = sparse.vstack([Xiso, Xn]).tocsr(); yiso = np.r_[yiso, ['none'] * Xn.shape[0]]
    print(f'isolated training set: {Xiso.shape[0]} trials; none trials total spikes={Xn.sum()}')

    rows, prev_rows, summary = [], [], {}
    eps = {g: load_eps(g) for g in GAPS if list(L.glob(f'w3d1_ep_{g}_*{SUF}_info.npz'))}
    for fname, keep in feats.items():
        # A: isolated readout, restricted to neurons active in its training set
        act = np.asarray(Xiso[:, keep].sum(0)).ravel() > 0
        cols = np.flatnonzero(keep)[act]
        clfA = fit(Xiso[:, cols].toarray(), yiso)
        for g, (X, info) in eps.items():
            w = (info.kind == 'w').to_numpy()
            pred, pmax = predict(clfA, X[w][:, cols].toarray())
            d = info[w].assign(pred=pred, pmax=pmax)
            d['ok'] = d.pred == d.word
            rec = dict(readout='isolated', features=fname, gap_ms=int(g), test='all 40 episodes',
                       n=len(d), acc=d.ok.mean(), acc_first=d[d.k == 0].ok.mean(),
                       acc_later=d[d.k > 0].ok.mean(), unsure=(d.pmax < 0.5).mean(),
                       acc_test_eps=d[d.ep >= N_TRAIN_EP].ok.mean())
            if g == '500':
                gw = (info.kind == 'g').to_numpy()
                gp, _ = predict(clfA, X[gw][:, cols].toarray())
                rec['gap_windows_read_none'] = float((gp == 'none').mean())
            rows.append(rec)
            for pv, dd in d[d.k > 0].groupby('prev'):
                prev_rows.append(dict(readout='isolated', features=fname, gap_ms=int(g), prev=pv,
                                      n=len(dd), acc=dd.ok.mean(),
                                      errors='; '.join(f'{a}->{b}' for a, b in
                                                       dd[~dd.ok][['word', 'pred']].values)))
            summary[f'isolated/{fname}/{g}/errors'] = [
                f'ep{e} k{k}: {a} (after {p or "-"}) -> {b} p={m:.2f}'
                for e, k, a, p, b, m in d[~d.ok][['ep', 'k', 'word', 'prev', 'pred', 'pmax']].values]
        # B: episode-trained readout, per spacing
        for g, (X, info) in eps.items():
            w = (info.kind == 'w').to_numpy()
            Xw, iw = X[w], info[w].reset_index(drop=True)
            trm = (iw.ep < N_TRAIN_EP).to_numpy()
            Xtr = sparse.vstack([Xiso, Xw[trm]]).tocsr(); ytr = np.r_[yiso, iw.word[trm]]
            act = np.asarray(Xtr[:, keep].sum(0)).ravel() > 0
            cols_b = np.flatnonzero(keep)[act]
            clfB = fit(Xtr[:, cols_b].toarray(), ytr)
            pred, pmax = predict(clfB, Xw[~trm][:, cols_b].toarray())
            d = iw[~trm].assign(pred=pred, pmax=pmax); d['ok'] = d.pred == d.word
            rows.append(dict(readout='episode-trained', features=fname, gap_ms=int(g),
                             test=f'episodes {N_TRAIN_EP}-39', n=len(d), acc=d.ok.mean(),
                             acc_first=d[d.k == 0].ok.mean(), acc_later=d[d.k > 0].ok.mean(),
                             unsure=(d.pmax < 0.5).mean(), acc_test_eps=d.ok.mean()))
            summary[f'episode/{fname}/{g}/errors'] = [
                f'ep{e} k{k}: {a} (after {p or "-"}) -> {b} p={m:.2f}'
                for e, k, a, p, b, m in d[~d.ok][['ep', 'k', 'word', 'prev', 'pred', 'pmax']].values]
        # C: episode-trained, 5-fold cross-validation over all 40 episodes (8 per fold)
        for g, (X, info) in eps.items():
            w = (info.kind == 'w').to_numpy()
            Xw, iw = X[w], info[w].reset_index(drop=True)
            preds = np.empty(len(iw), dtype=object)
            for f in range(5):
                te = (iw.ep % 5 == f).to_numpy()
                Xtr = sparse.vstack([Xiso, Xw[~te]]).tocsr(); ytr = np.r_[yiso, iw.word[~te]]
                act = np.asarray(Xtr[:, keep].sum(0)).ravel() > 0
                cc = np.flatnonzero(keep)[act]
                preds[te] = predict(fit(Xtr[:, cc].toarray(), ytr), Xw[te][:, cc].toarray())[0]
            d = iw.assign(pred=preds); d['ok'] = d.pred == d.word
            hot_after = d[(d.word == 'hot') & d.apply(
                lambda r: r.k > 0 and bool(SEEN_CH[r.ep] & set(SEQS[r.ep][:r.k])), axis=1)]
            rows.append(dict(readout='episode-trained 5-fold', features=fname, gap_ms=int(g),
                             test='all 40 episodes, 5-fold', n=len(d), acc=d.ok.mean(),
                             acc_first=d[d.k == 0].ok.mean(), acc_later=d[d.k > 0].ok.mean(),
                             acc_test_eps=d.ok.mean(),
                             hot_after_cold_humid=f'{int(hot_after.ok.sum())}/{len(hot_after)}'))
            summary[f'episode5fold/{fname}/{g}/errors'] = [
                f'ep{e} k{k}: {a} (after {p or "-"}) -> {b}'
                for e, k, a, p, b in d[~d.ok][['ep', 'k', 'word', 'prev', 'pred']].values]
    df = pd.DataFrame(rows)
    df.to_csv(L / f'w3d1_carryover{SUF}.csv', index=False)
    pd.DataFrame(prev_rows).to_csv(L / f'w3d1_by_prev{SUF}.csv', index=False)
    json.dump(summary, open(L / f'w3d1_summary{SUF}.json', 'w'), indent=1)
    pd.set_option('display.width', 200)
    print(df.round(3).to_string(index=False))
    for k, v in summary.items():
        print(k, len(v)); [print('   ', e) for e in v[:40]]


if __name__ == '__main__':
    main()
