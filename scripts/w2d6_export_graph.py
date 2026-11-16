#!/usr/bin/env python
"""Week 2 Days 6-7: export graph views for the Connectome Graph Explorer artifact.

One view = one simulated trial (seed 0) drawn on the connectome subgraph it uses.
  nodes : neurons that spiked (top NODE_CAP by spike count; stimulated sensory
          neurons capped at SENS_CAP), with type, superclass, transmitter,
          per-20-ms spike counts, and two precomputed layouts:
          layered (x = synaptic hops from the stimulated neurons) and force.
  edges : non-zero entries of the simulated W among those nodes: every node's
          strongest input, then the strongest remaining up to EDGE_CAP.
          Gap junctions (calibrated model) are included separately.
Usage:
  w2d6_export_graph.py malecns <base|calib|shuffle>
  w2d6_export_graph.py flywire <base|calib>
  w2d6_export_graph.py extras         (cell-type overview, attractor core, index)
Output: results/graph/<connectome>_<config>_<word>.json, overview.json, core.json
"""
import base64
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import breadth_first_order  # noqa: F401
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.lif_numpy import run_trial, P, load_network  # noqa: E402
from sim.gap import build_gap  # noqa: E402

OUT = ROOT / 'results' / 'graph'
OUT.mkdir(parents=True, exist_ok=True)
NODE_CAP, SENS_CAP, EDGE_CAP, GAP_CAP = 1200, 80, 4000, 800
T_STIM, T_END, BIN = 0.5, 1.0, 0.02
DT = P['dt']


def b64(a):
    return base64.b64encode(np.asarray(a, dtype=np.uint8).tobytes()).decode()


def force_layout(n, ei, ej, w, iters=150, seed=0):
    """Fruchterman-Reingold on an undirected weighted graph, numpy O(n^2)."""
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 2))
    k = 1.0 / np.sqrt(max(n, 1))
    t = 0.1
    w = w / (w.max() if len(w) else 1)
    for it in range(iters):
        d = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((d ** 2).sum(-1)) + 1e-9
        rep = (d / dist[..., None] ** 2 * k * k).sum(1)
        dd = pos[ei] - pos[ej]
        dl = np.sqrt((dd ** 2).sum(-1)) + 1e-9
        att = dd * (dl / k * (0.3 + w))[:, None]
        disp = rep.copy()
        np.add.at(disp, ei, -att); np.add.at(disp, ej, att)
        ln = np.sqrt((disp ** 2).sum(-1)) + 1e-9
        pos += disp / ln[:, None] * np.minimum(ln, t)[:, None]
        t *= 0.97
    pos -= pos.min(0); pos /= pos.max(0) + 1e-9
    return pos


def build_view(W, gap, exc, ids, typ, sup, nt, cls, stim_word, label, extra):
    n = W.shape[0]
    p = dict(P); p['t_run'] = T_END
    r = run_trial(W, exc, params=p, seed=0, apl=extra.get('apl'), t_stim=T_STIM, record=True, gap=gap)
    ts, si = r['t'] * DT, r['i']
    tot = np.bincount(si, minlength=n)
    is_stim = np.zeros(n, bool); is_stim[exc] = True
    sens = np.flatnonzero(is_stim & (tot > 0)); sens = sens[np.argsort(-tot[sens])][:SENS_CAP]
    rest = np.flatnonzero(~is_stim & (tot > 0)); rest = rest[np.argsort(-tot[rest])][:NODE_CAP - len(sens)]
    sel = np.concatenate([sens, rest])
    for must in list(extra.get('must', [])) + list(extra.get('always', [])):
        if must not in sel and (tot[must] > 0 or must in set(extra.get('always', []))):
            sel = np.append(sel, must)          # motor neurons are always shown, even silent
    m9 = set(extra.get('must', []))
    k = len(sel)
    loc = np.full(n, -1); loc[sel] = np.arange(k)
    sub = W[sel][:, sel].tocoo()                        # rows post, cols pre (local)
    # layers: BFS hops from stimulated neurons over excitatory edges
    A = sparse.csr_matrix((np.ones(sub.nnz), (sub.col, sub.row)), shape=(k, k))  # pre -> post
    exc_e = sub.data > 0
    Ae = sparse.csr_matrix((np.ones(exc_e.sum()), (sub.col[exc_e], sub.row[exc_e])), shape=(k, k))
    layer = np.full(k, -1); src = np.flatnonzero(is_stim[sel])
    layer[src] = 0; frontier = src; L = 0
    while len(frontier) and L < 8:
        L += 1
        nxt = np.unique(Ae[frontier].indices)
        nxt = nxt[layer[nxt] < 0]; layer[nxt] = L; frontier = nxt
    layer[layer < 0] = L + 1 if L < 8 else 9
    # first spike time for ordering
    first = np.full(n, T_END); m = np.unique(si, return_index=True)
    first[m[0]] = ts[m[1]]
    # layered layout
    lx = layer / max(layer.max(), 1)
    ly = np.zeros(k)
    for lv in np.unique(layer):
        idx = np.flatnonzero(layer == lv)
        order = idx[np.lexsort((first[sel[idx]], sup[sel[idx]]))]
        ly[order] = (np.arange(len(order)) + 0.5) / len(order)
    # edges
    aw = np.abs(sub.data)
    keep = np.zeros(sub.nnz, bool)
    best = pd.DataFrame(dict(r=sub.row, w=aw, i=np.arange(sub.nnz))).sort_values('w').groupby('r').tail(1).i
    keep[best.to_numpy()] = True
    order = np.argsort(-aw)
    for i in order:
        if keep.sum() >= EDGE_CAP:
            break
        keep[i] = True
    ei, ej, ew = sub.col[keep], sub.row[keep], sub.data[keep]          # pre, post, w
    fpos = force_layout(k, ei, ej, np.abs(ew))
    # gap edges among selected
    gedges = []
    if gap is not None:
        gi = np.asarray(gap['idx']); C = gap['C'].tocoo()
        a, b = loc[gi[C.row]], loc[gi[C.col]]
        ok = (a >= 0) & (b >= 0) & (a < b)
        a, b, c = a[ok], b[ok], C.data[ok]
        top = np.argsort(-c)[:GAP_CAP]
        gedges = [[int(x), int(y), round(float(z), 4)] for x, y, z in zip(a[top], b[top], c[top])]
    nb = int(round(T_END / BIN))
    bins = np.zeros((k, nb), np.uint8)
    mm = loc[si] >= 0
    np.add.at(bins, (loc[si[mm]], np.minimum((ts[mm] / BIN).astype(int), nb - 1)), 1)
    wmax = float(np.abs(ew).max()) if len(ew) else 1.0
    nodes = [dict(id=str(ids[g]), t=typ[g] or '', s=sup[g] or '', c=cls[g] or '', nt=nt[g] or '',
                  L=int(layer[i]), x=round(float(lx[i]), 4), y=round(float(ly[i]), 4),
                  fx=round(float(fpos[i, 0]), 4), fy=round(float(fpos[i, 1]), 4),
                  r=round(float(tot[g] / T_END), 1), st=int(is_stim[g]), a=b64(bins[i]),
                  **({'m9': 1} if g in m9 else {}))
             for i, g in enumerate(sel)]
    edges = [[int(a_), int(b_), round(float(w_ / wmax), 3)] for a_, b_, w_ in zip(ei, ej, ew)]
    pop = np.bincount(np.minimum((ts / BIN).astype(int), nb - 1), minlength=nb).tolist()
    tail = int(len(np.unique(si[ts >= T_END - 0.1])))
    view = dict(label=label, word=stim_word, bin_ms=int(BIN * 1000), t_stim=T_STIM, t_end=T_END,
                n_active=int((tot > 0).sum()), n_shown=k, tail_neurons=tail, pop=pop,
                nodes=nodes, edges=edges, gaps=gedges)
    return view


def malecns(config):
    from sim.lesion import select
    from sim.malecns import load
    from stimuli.malecns_words import WORDS
    from scripts.ckpt2_ab import stim_idx
    from scripts.c4_core_gain import Runner
    from scripts.ckpt5_stageA import ACH_LN, PN
    from scripts.w2d3_gap_malecns import cut_ln_pn
    from scripts.w2d4_word_trials import shuffle_W, GAP_KW
    R = Runner(); W, gap = R.Wr, None
    if config in ('calib', 'shuffle'):
        ln, pn = select(R.meta, ACH_LN), select(R.meta, PN)
        _, Wsyn = load(w_syn=1.0, monoamine='removed')
        W = cut_ln_pn(R.Wr, ln, pn)
        gap, _ = build_gap(Wsyn, ln, pn, v_spike=0.002, **GAP_KW)
    if config == 'shuffle':
        W = shuffle_W(W)
    m = R.meta
    ids, typ, sup, nt = m.bodyId.to_numpy(), m.type.fillna('').to_numpy(), m.superclass.fillna('').to_numpy(), m.nt.to_numpy()
    cls = m['class'].fillna('').to_numpy()
    for word in WORDS:
        v = build_view(W, gap, stim_idx(m, R.pos, word), ids, typ, sup, nt, cls, word,
                       f'MaleCNS {config}', dict(apl=R.apl, must=list(R.mn9),
                       always=list(np.flatnonzero(m.superclass.values == 'cb_motor'))))
        json.dump(v, open(OUT / f'malecns_{config}_{word}.json', 'w'), separators=(',', ':'))
        print(f'malecns {config} {word}: shown {v["n_shown"]}/{v["n_active"]} edges {len(v["edges"])} '
              f'gaps {len(v["gaps"])} tail {v["tail_neurons"]}', flush=True)


def flywire(config):
    from stimuli.words import WORDS as FW_TASTE, ID_MN9_LEFT
    REPO = ROOT / 'third_party' / 'Drosophila_brain_model'
    ids, W = load_network(REPO / 'Completeness_783.csv', REPO / 'Connectivity_783.parquet')
    n = len(ids); pos = pd.Series(np.arange(n), index=ids)
    a = pd.read_csv(ROOT / 'data' / 'flywire_annot' / 'annot.tsv', sep='\t', low_memory=False)
    a = a[a.root_id.isin(pos.index)]
    ct = a.cell_type.fillna('')
    def mask(mm):
        o = np.zeros(n, bool); o[pos[a.root_id[mm]].to_numpy()] = True; return o
    typ = np.full(n, '', object); typ[pos[a.root_id].to_numpy()] = ct.to_numpy()
    sup = np.full(n, '', object); sup[pos[a.root_id].to_numpy()] = a.super_class.fillna('').to_numpy()
    nt = np.full(n, '', object); nt[pos[a.root_id].to_numpy()] = a.top_nt.fillna('').to_numpy()
    cls = np.full(n, '', object); cls[pos[a.root_id].to_numpy()] = a.cell_class.fillna('').to_numpy()
    gap = None
    if config == 'calib':
        alln, pn = mask(a.cell_class == 'ALLN'), mask(a.cell_class == 'ALPN')
        ln = alln & (np.asarray(W.sum(axis=0)).ravel() > 0)
        C = W.tocoo(); keep = ~(ln[C.col] & pn[C.row])
        Wsyn = abs(W) / P['w_syn']
        W = sparse.csc_matrix((C.data[keep], (C.row[keep], C.col[keep])), shape=W.shape)
        gap, _ = build_gap(Wsyn, ln, pn, c0=0.2, v_spike=0.002)
    stim = {'vinegar': np.flatnonzero(mask(ct.isin(['ORN_DM1', 'ORN_DM4']))),
            'hot': np.flatnonzero(mask(ct == 'TRN_VP2')),
            'sound': np.flatnonzero(mask(ct.str.match(r'^JO-[AB]'))),
            'looming': np.flatnonzero(mask(ct.isin(['LPLC2', 'LC4']))),
            'sugar': np.sort(pos[[i for i in FW_TASTE['sugar'] if i in pos.index]].to_numpy())}
    taste = lambda w: np.sort(pos[[i for i in FW_TASTE[w] if i in pos.index]].to_numpy())
    for w in ['bitter', 'water', 'salt']:
        stim[w] = taste(w)
    stim['sugar+bitter'] = np.union1d(stim['sugar'], stim['bitter'])
    only = sys.argv[3].split(',') if len(sys.argv) > 3 else None
    for word, exc in stim.items():
        if only and word not in only:
            continue
        v = build_view(W, gap, exc, ids, typ, sup, nt, cls, word, f'FlyWire {config}',
                       dict(must=[int(pos[ID_MN9_LEFT])], always=list(np.flatnonzero(sup == 'motor'))))
        json.dump(v, open(OUT / f'flywire_{config}_{word}.json', 'w'), separators=(',', ':'))
        print(f'flywire {config} {word}: shown {v["n_shown"]}/{v["n_active"]} edges {len(v["edges"])} '
              f'tail {v["tail_neurons"]}', flush=True)


if __name__ == '__main__':
    {'malecns': lambda: malecns(sys.argv[2]), 'flywire': lambda: flywire(sys.argv[2])}[sys.argv[1]]()
