#!/usr/bin/env python
"""Checkpoint 4: locate the self-sustaining brain attractor (observation only).

Setup (from ckpt3): brain-only network (VNC rows/cols zeroed), monoamines
removed, w_syn = 0.275 mV x 0.581, rate-unit APL k = 0.1.
Trial: 1.0 s stimulus at 150 Hz, then 0.5 s silence.

Part 1 controls (plan's runs A-C):
  A  fresh brain, zero stimulus for the whole trial
  B  'hot'; at offset stimulus stops AND all weights -> 0 (rate APL kept: inhibitory only)
  C  'hot'; at offset stimulus stops, recurrence unchanged
Part 2 fingerprint (plan's run D): every persistent word x 3 seeds, final
  100 ms; persistence probability per neuron, late-population overlap,
  rate-vector cosine.
Part 3 ignition: spike raster for 'hot'; ignition time and onset order of the
  common core.
Part 4 graph: core + precursors -> brain->brain subgraph with the exact
  simulation weights; strongly connected components (excitatory and signed),
  leading eigenvector loadings; NT audit of brain neurons and of core weight.
Outputs in results/malecns/ckpt4_*.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.malecns import load  # noqa: E402
from sim.lif_numpy import run_trial, P  # noqa: E402
from stimuli.malecns_words import WORDS  # noqa: E402
from scripts.ckpt2_ab import make_apl, stim_idx  # noqa: E402
from scripts.ckpt3_sweep_ablate import regions, mask_weights  # noqa: E402

OUT = ROOT / 'results' / 'malecns'
SCALE, K_APL = 0.581, 0.1
DT = P['dt']
SEEDS = [0, 1, 2]
LATE_FROM = 1.4          # final 100 ms of a 1.5 s trial


def setup():
    meta, W1 = load(w_syn=1.0, monoamine='removed')
    vnc, abd = regions(meta)
    W = mask_weights(W1 * (P['w_syn'] * SCALE), lambda r, c: ~vnc[r] & ~vnc[c])
    W_rate, kc_idx, w_kc, is_kc = make_apl(meta, W)
    apl = dict(kc_idx=kc_idx, w_kc=w_kc, k=K_APL, tau=0.01)
    pos = {b: i for i, b in enumerate(meta.bodyId)}
    return meta, vnc, W_rate, apl, pos, is_kc


def trial(W, apl, exc, seed=0, **kw):
    p = dict(P); p['t_run'] = 1.5
    return run_trial(W, exc, params=p, seed=seed, apl=apl, t_stim=1.0, **kw)


def part1(meta, W, apl, pos):
    print('=== Part 1: controls ===', flush=True)
    res = {}
    # A: zero stimulus (empty stim set)
    c, tail = trial(W, apl, np.array([], dtype=int), tail_from=1.2)
    res['A_zero_stim'] = dict(active=int((c > 0).sum()), tail=int((tail > 0).sum()))
    exc = stim_idx(meta, pos, 'hot')
    # C: reference
    r = trial(W, apl, exc, tail_from=1.2, record=True)
    post = r['t'] >= int(1.0 / DT)
    ext_after = int(np.isin(r['i'][post], exc).sum())
    res['C_offset_only'] = dict(active=int((r['counts'] > 0).sum()),
                               tail=int((r['tail'] > 0).sum()),
                               stimulated_neuron_spikes_after_offset=ext_after)
    # B: weights and APL to zero at offset
    Z = sparse.csc_matrix(W.shape)
    apl0 = dict(apl)
    c, tail = run_trial(W, exc, params={**P, 't_run': 1.5}, seed=0, apl=apl0,
                        t_stim=1.0, tail_from=1.2, w_switch=(1.0, Z))
    res['B_offset_and_W0'] = dict(active=int((c > 0).sum()), tail=int((tail > 0).sum()))
    for k, v in res.items():
        print(f'  {k}: {v}', flush=True)
    return res


def part2(meta, W, apl, pos, vnc):
    print('=== Part 2: fingerprint (final 100 ms, 3 seeds) ===', flush=True)
    late_cnt, rates, rows = {}, {}, []
    for w in list(WORDS) + ['sugar+bitter']:
        exc = stim_idx(meta, pos, w)
        for s in SEEDS:
            c, late = trial(W, apl, exc, seed=s, tail_from=LATE_FROM)
            late_cnt[(w, s)] = late
            rows.append(dict(word=w, seed=s, active=int((c > 0).sum()),
                             late=int((late > 0).sum())))
            print(f'  {w:13s} seed {s} active={rows[-1]["active"]:6d} late={rows[-1]["late"]}', flush=True)
    df = pd.DataFrame(rows)
    pers = [(k, v) for k, v in late_cnt.items() if (v > 0).sum() > 0]
    n = len(pers)
    P_any = np.zeros(len(meta))
    for _, v in pers:
        P_any += v > 0
    P_any /= max(n, 1)
    # per-word union over seeds for overlap, mean rate vector for cosine
    words_p = sorted({k[0] for k, _ in pers})
    sets = {w: np.any([late_cnt[(w, s)] > 0 for s in SEEDS], axis=0) for w in words_p}
    vecs = {w: np.mean([late_cnt[(w, s)] for s in SEEDS], axis=0) for w in words_p}
    J, C = [], []
    for i, a in enumerate(words_p):
        for b in words_p[i + 1:]:
            J.append((sets[a] & sets[b]).sum() / max(1, (sets[a] | sets[b]).sum()))
            va, vb = vecs[a], vecs[b]
            C.append(float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12)))
    summary = dict(persistent_trials=n, total_trials=len(late_cnt),
                   persistent_words=words_p,
                   late_overlap_mean=float(np.mean(J)) if J else None,
                   late_overlap_min=float(np.min(J)) if J else None,
                   late_cosine_mean=float(np.mean(C)) if C else None,
                   late_cosine_min=float(np.min(C)) if C else None)
    for th in [0.9, 0.8, 0.6, 0.4]:
        summary[f'core_n_P>={th}'] = int((P_any >= th).sum())
    print('  ', json.dumps(summary), flush=True)
    df.to_csv(OUT / 'ckpt4_fingerprint_trials.csv', index=False)
    np.save(OUT / 'ckpt4_P_any.npy', P_any)
    return P_any, summary


def part3(meta, W, apl, pos, core):
    print('=== Part 3: ignition order for hot ===', flush=True)
    r = trial(W, apl, stim_idx(meta, pos, 'hot'), tail_from=1.2, record=True)
    t, i = r['t'], r['i']
    bins = int(0.005 / DT)
    nb = int(1.5 / DT) // bins
    act = np.zeros(nb)
    for b in range(nb):
        m = (t >= b * bins) & (t < (b + 1) * bins)
        act[b] = len(np.unique(i[m]))
    # ignition = first 5-ms bin where distinct active neurons exceed half the late plateau
    plateau = np.median(act[int(1.4 / 0.005):])
    ig_bin = int(np.argmax(act >= 0.5 * plateau))
    t_ig = ig_bin * 0.005
    first = pd.Series(t).groupby(i).min() * DT
    core_idx = np.flatnonzero(core)
    onset = first.reindex(core_idx) - t_ig
    print(f'  late plateau ~{plateau:.0f} active neurons per 5 ms; ignition at {t_ig*1000:.0f} ms', flush=True)
    pd.DataFrame({'t_ms': np.arange(nb) * 5, 'active': act}).to_csv(OUT / 'ckpt4_hot_activity.csv', index=False)
    return t_ig, first, onset


def part4(meta, W, core, precursors, P_any, onset):
    print('=== Part 4: graph analysis ===', flush=True)
    # NT audit across brain neurons
    nt_tab = meta.groupby('nt').agg(neurons=('bodyId', 'size'), sign=('sign', 'first'))
    sel = core | precursors
    idx = np.flatnonzero(sel)
    S = W[idx][:, idx].tocsr()                       # post x pre within candidate set
    Sc = S.tocoo()
    pre_nt = meta.nt.values[idx][Sc.col]
    pos_w = Sc.data > 0
    w_by_nt = pd.Series(np.abs(Sc.data)).groupby(pre_nt).sum()
    pos_by_nt = pd.Series(Sc.data[pos_w]).groupby(pre_nt[pos_w]).sum()
    nt_tab['candidate_internal_|w|_share'] = (w_by_nt / w_by_nt.sum()).reindex(nt_tab.index).fillna(0).round(4)
    nt_tab['candidate_internal_E_share'] = (pos_by_nt / pos_by_nt.sum()).reindex(nt_tab.index).fillna(0).round(4)
    print(nt_tab.to_string(), flush=True)

    out = {}
    for label, M in [('excitatory', sparse.csr_matrix((Sc.data * pos_w, (Sc.row, Sc.col)), shape=S.shape)),
                     ('signed', S)]:
        M.eliminate_zeros()
        nc, lab = connected_components(M.T, directed=True, connection='strong')
        sizes = np.bincount(lab)
        comps = []
        for cid in np.argsort(-sizes)[:5]:
            members = idx[lab == cid]
            if len(members) < 2:
                continue
            sub = M[lab == cid][:, lab == cid]
            comps.append(dict(size=int(len(members)),
                              core_frac=float(core[members].mean()),
                              mean_P=float(P_any[members].mean()),
                              internal_E_mV=float(sub.data[sub.data > 0].sum() * 1e3),
                              internal_I_mV=float(-sub.data[sub.data < 0].sum() * 1e3),
                              top_types=meta.type.values[members].astype(str).tolist()))
        out[label] = comps
        print(f'  [{label}] {nc} SCCs; largest:', [(c['size'], round(c['core_frac'], 2)) for c in comps], flush=True)

    # leading eigenvector of the signed candidate matrix (ranking only)
    vals, vecs = eigs(S.astype(np.float64), k=1, which='LR')
    load_ = np.abs(vecs[:, 0])
    rank = pd.DataFrame({'bodyId': meta.bodyId.values[idx], 'type': meta.type.values[idx],
                         'class': meta['class'].values[idx], 'superclass': meta.superclass.values[idx],
                         'nt': meta.nt.values[idx], 'P_late': P_any[idx],
                         'onset_ms_vs_ignition': (onset.reindex(idx).values * 1e3),
                         'eig_loading': load_ / load_.max(),
                         'E_out_to_candidates_mV': np.asarray(
                             S.multiply(S > 0).sum(axis=0)).ravel() * 1e3})
    rank['score'] = rank.P_late * rank.eig_loading
    rank = rank.sort_values('score', ascending=False)
    rank.to_csv(OUT / 'ckpt4_candidates.csv', index=False)
    print(f'  leading eigenvalue (real part) {vals[0].real*1e3:.2f} mV-scale', flush=True)
    by_type = rank.head(200).groupby(['type', 'nt']).size().sort_values(ascending=False).head(25)
    print('  top-200 candidates by type:\n' + by_type.to_string(), flush=True)
    json.dump(dict(scc=out, leading_eig=float(vals[0].real)), open(OUT / 'ckpt4_graph.json', 'w'), default=str)
    nt_tab.to_csv(OUT / 'ckpt4_nt_audit.csv')
    return rank


def main(parts):
    meta, vnc, W, apl, pos, is_kc = setup()
    if 1 in parts:
        json.dump(part1(meta, W, apl, pos), open(OUT / 'ckpt4_controls.json', 'w'))
    if parts & {2, 3, 4}:
        P_any, summ = part2(meta, W, apl, pos, vnc)
        json.dump(summ, open(OUT / 'ckpt4_fingerprint.json', 'w'))
        core = P_any >= 0.8
        t_ig, first, onset = part3(meta, W, apl, pos, core)
        # precursors: brain neurons whose first spike precedes ignition by <= 50 ms
        pre_ids = first[(first >= t_ig - 0.05) & (first < t_ig)].index.values
        precursors = np.zeros(len(meta), bool); precursors[pre_ids] = True
        precursors &= ~vnc
        print(f'  core (P>=0.8) {core.sum()}  precursors {precursors.sum()}', flush=True)
        part4(meta, W, core, precursors, P_any, onset)
    print('DONE', flush=True)


if __name__ == '__main__':
    parts = {int(x) for x in sys.argv[1:]} or {1, 2, 3, 4}
    main(parts)
