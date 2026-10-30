#!/usr/bin/env python
"""C1: recurrent-graph analysis of the persistent antennal-lobe core.

Simulator: Brian2-faithful refractory input ('block', fixed in C0).
Base: brain-only, monoamines removed, w x 0.581, rate APL k = 0.1.

Populations (AL neurons, i.e. class ALLN or ALPN, firing in the last 300 ms
of silence for BOTH 'hot' and 'vinegar', seed 0):
  base_tail : baseline network
  lpcut_core: cholinergic ALLN -> ALPN cut (the self-sustaining LN core, as B1)
For each population: induced directed graph with the exact simulation weights;
strongly connected components (SCCs) of the excitatory subgraph and of the
signed graph; per-SCC size, internal E and I, rates, near-ceiling count, input
from / output to the rest of the population; shortest strong cycles; cycle
participation; greedy feedback-vertex-set on the excitatory SCC.
Outputs: results/malecns/c1_*.csv, c1_summary.json
"""
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.malecns import load  # noqa: E402
from sim.lif_numpy import run_trial, P  # noqa: E402
from sim.lesion import apply, select  # noqa: E402
from scripts.ckpt2_ab import make_apl, stim_idx  # noqa: E402
from scripts.ckpt3_sweep_ablate import regions, mask_weights  # noqa: E402
from scripts.ckpt5_stageA import ACH_LN, LN, PN, SCALE, K_APL  # noqa: E402

OUT = ROOT / 'results' / 'malecns'
CEIL_HZ = 1 / (P['t_rfc'] + P['dt'])          # ~435 Hz refractory ceiling
STRONG_MV = 1.0                                # "high-weight" edge threshold for cycles


def tail_population(W, meta, pos, al):
    Wr, kc, wkc, _ = make_apl(meta, W)
    apl = dict(kc_idx=kc, w_kc=wkc, k=K_APL, tau=0.01)
    p = dict(P); p['t_run'] = 1.5
    tails = []
    for odor in ['hot', 'vinegar']:
        _, tl = run_trial(Wr, stim_idx(meta, pos, odor), params=p, seed=0, apl=apl,
                          t_stim=1.0, tail_from=1.2)
        tails.append(tl)
    tail = np.minimum(*tails)
    return Wr, (tail > 0) & al, tail / 0.3


def analyse(name, W, meta, pop, hz):
    idx = np.flatnonzero(pop)
    S = W[idx][:, idx].tocoo()                         # row = post, col = pre
    lab = lambda i: f'{meta.type.values[idx[i]]}#{i}'
    G_all, G_exc = nx.DiGraph(), nx.DiGraph()
    for k in range(len(idx)):
        G_all.add_node(k); G_exc.add_node(k)
    for r, c, w in zip(S.row, S.col, S.data):
        G_all.add_edge(c, r, w=w * 1e3)               # pre -> post, mV
        if w > 0:
            G_exc.add_edge(c, r, w=w * 1e3)
    summary = dict(population=name, n=int(len(idx)), edges=int(S.nnz),
                   E_mV=float(S.data[S.data > 0].sum() * 1e3),
                   I_mV=float(-S.data[S.data < 0].sum() * 1e3))
    rows = []
    for gname, G in [('excitatory', G_exc), ('signed', G_all)]:
        sccs = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
        for rank, comp in enumerate([c for c in sccs if len(c) > 1][:6]):
            comp = list(comp)
            inside = np.zeros(len(idx), bool); inside[comp] = True
            sub_r, sub_c = inside[S.row], inside[S.col]
            internal = sub_r & sub_c
            rates = hz[idx[comp]]
            rows.append(dict(population=name, graph=gname, scc=rank, size=len(comp),
                             E_int_mV=S.data[internal & (S.data > 0)].sum() * 1e3,
                             I_int_mV=-S.data[internal & (S.data < 0)].sum() * 1e3,
                             in_from_rest_mV=S.data[sub_r & ~sub_c].sum() * 1e3,
                             out_to_rest_mV=S.data[~sub_r & sub_c].sum() * 1e3,
                             mean_Hz=rates.mean(), max_Hz=rates.max(),
                             near_ceiling=int((rates > 0.9 * CEIL_HZ).sum()),
                             types=pd.Series(meta.type.values[idx[comp]]).value_counts().head(8).to_dict(),
                             nts=pd.Series(meta.nt.values[idx[comp]]).value_counts().to_dict()))
        summary[f'{gname}_n_nontrivial_scc'] = sum(len(c) > 1 for c in sccs)
    scc_df = pd.DataFrame(rows)

    # cycle analysis on the largest excitatory SCC, strong edges only
    big = max(nx.strongly_connected_components(G_exc), key=len)
    H = nx.DiGraph((u, v, d) for u, v, d in G_exc.subgraph(big).edges(data=True) if d['w'] >= STRONG_MV)
    two = [(u, v) for u, v in H.edges if H.has_edge(v, u) and u < v]
    part = np.zeros(len(idx))
    for u, v in two:
        part[u] += 1; part[v] += 1
    three = 0
    for u in H.nodes:                                  # count strong 3-cycles per node
        for v in H.successors(u):
            for x in H.successors(v):
                if x != u and H.has_edge(x, u):
                    part[u] += 1; three += 1
    # greedy feedback vertex set: repeatedly remove the node on the most strong
    # short cycles / highest strong in*out degree until no strong SCC > 1 remains
    Hc, fvs = H.copy(), []
    while True:
        comps = [c for c in nx.strongly_connected_components(Hc) if len(c) > 1]
        if not comps:
            break
        sub = Hc.subgraph(max(comps, key=len))
        score = {n: sub.in_degree(n, weight='w') * sub.out_degree(n, weight='w') for n in sub.nodes}
        n = max(score, key=score.get)
        fvs.append(n); Hc.remove_node(n)
    nodes = pd.DataFrame({'bodyId': meta.bodyId.values[idx], 'type': meta.type.values[idx],
                          'class': meta['class'].values[idx], 'nt': meta.nt.values[idx],
                          'tail_Hz': hz[idx], 'in_big_exc_scc': np.isin(np.arange(len(idx)), list(big)),
                          'strong_cycle_participation': part,
                          'fvs_order': [fvs.index(k) if k in fvs else -1 for k in range(len(idx))]})
    nodes.to_csv(OUT / f'c1_nodes_{name}.csv', index=False)
    summary.update(big_exc_scc=len(big), strong_edges_in_big=H.number_of_edges(),
                   strong_2cycles=len(two), strong_3cycles=three // 3, fvs_size=len(fvs),
                   fvs_types=pd.Series(meta.type.values[idx[fvs]]).value_counts().to_dict())
    return scc_df, summary


def main():
    meta, W1 = load(w_syn=1.0, monoamine='removed')
    vnc, _ = regions(meta)
    W = mask_weights(W1 * (P['w_syn'] * SCALE), lambda r, c: ~vnc[r] & ~vnc[c])
    W_cut, _ = apply(W, meta, [dict(op='scale_edges', pre=ACH_LN, post=PN, alpha=0.0)])
    pos = {b: i for i, b in enumerate(meta.bodyId)}
    al = select(meta, f'{LN} or {PN}')
    all_scc, summ = [], []
    for name, Wx in [('base_tail', W), ('lpcut_core', W_cut)]:
        Wr, pop, hz = tail_population(Wx, meta, pos, al)
        print(f'[{name}] persistent AL neurons in both odors: {pop.sum()}', flush=True)
        s_df, s = analyse(name, Wr, meta, pop, hz)
        all_scc.append(s_df); summ.append(s)
        print(json.dumps(s, default=str), flush=True)
        print(s_df[['graph', 'scc', 'size', 'E_int_mV', 'I_int_mV', 'in_from_rest_mV',
                    'out_to_rest_mV', 'mean_Hz', 'near_ceiling']].round(0).to_string(), flush=True)
    pd.concat(all_scc).to_csv(OUT / 'c1_scc.csv', index=False)
    json.dump(summ, open(OUT / 'c1_summary.json', 'w'), indent=1, default=str)
    print('DONE')


if __name__ == '__main__':
    main()
