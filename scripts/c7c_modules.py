#!/usr/bin/env python
"""C7c: are the C6 rescue modules graph-distinct, self-contained loops?

Groups (104-core members): lLN1_bc (29), otherLN (51), unclear (13), PN (11).
Part 1, graph (simulated weights Wr): per group, excitatory SCCs of the
  induced subgraph; for each member, the share of its excitatory input that
  comes from its own group, from the rest of the core, and from outside the
  core; and the share of the group's tail-period drive coming from outside.
Part 2, isolation test (simulation): as C6 (all 104 silenced at 1.3 s except
  the restored group, beta = 1.0), but additionally every edge INTO the group
  from outside the group is removed at 1.3 s. If the group still persists,
  its loop is self-contained; if it dies, it needs drive routed through other
  neurons. hot + vinegar, seed 0.
Outputs: results/malecns/c7c_graph.csv, c7c_isolation.csv
"""
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.lif_numpy import run_trial  # noqa: E402
from scripts.ckpt2_ab import stim_idx  # noqa: E402
from scripts.ckpt3_sweep_ablate import mask_weights  # noqa: E402
from scripts.c4_core_gain import Runner, DT, T_END, OUT  # noqa: E402
from scripts.c5_gain_decomp import core_groups, T_SW  # noqa: E402
from scripts.c6_rescue import scale_by  # noqa: E402

GROUPS = ['lLN1_bc', 'otherLN', 'unclear', 'PN']


def graph_part(R, G, tail_hz):
    Wc = R.Wr.tocsr()
    core = G['all104']
    rows = []
    for name in GROUPS:
        g = G[name]
        idx = np.flatnonzero(g)
        sub = Wc[idx][:, idx].tocoo()
        H = nx.DiGraph(); H.add_nodes_from(range(len(idx)))
        H.add_edges_from((c, r) for r, c, w in zip(sub.row, sub.col, sub.data) if w > 0)
        sccs = sorted((len(c) for c in nx.strongly_connected_components(H)), reverse=True)
        X = Wc[idx]
        E = X.multiply(X > 0).tocsc()
        tot = np.asarray(E.sum(axis=1)).ravel()
        own = np.asarray(E[:, g].sum(axis=1)).ravel()
        rest = np.asarray(E[:, core & ~g].sum(axis=1)).ravel()
        out = tot - own - rest
        # drive weighted by the tail-period rates of baseline (who actually feeds the group)
        dr = E.multiply(tail_hz[None, :]).tocsc()
        dtot = np.asarray(dr.sum(axis=1)).ravel()
        down = np.asarray(dr[:, g].sum(axis=1)).ravel()
        drest = np.asarray(dr[:, core & ~g].sum(axis=1)).ravel()
        rows.append(dict(group=name, n=len(idx), exc_edges=H.number_of_edges(),
                         largest_scc=sccs[0], n_nontrivial_scc=sum(s > 1 for s in sccs),
                         in_own=own.sum() / tot.sum(), in_rest_core=rest.sum() / tot.sum(),
                         in_outside=out.sum() / tot.sum(),
                         tail_drive_own=down.sum() / dtot.sum(),
                         tail_drive_rest_core=drest.sum() / dtot.sum(),
                         tail_drive_outside=1 - (down.sum() + drest.sum()) / dtot.sum()))
    return pd.DataFrame(rows)


def isolation_trial(R, G, name, isolate, word):
    g = G[name]
    s = np.ones(len(R.meta)); s[G['all104']] = 0.0; s[g] = 1.0
    W2 = scale_by(R.Wr, s)
    if isolate:
        W2 = mask_weights(W2, lambda r, c: ~(g[r] & ~g[c]))
    r = run_trial(R.Wr, stim_idx(R.meta, R.pos, word), params=R.p, seed=0, apl=R.apl,
                  t_stim=1.0, w_switch=(T_SW, W2), record=True)
    ts, si = r['t'] * DT, r['i']
    m = ts >= T_END - 0.1
    last = np.bincount(si[m], minlength=len(R.meta))
    return dict(group=name, isolated=isolate, word=word,
                tail_brain=int((last > 0).sum()), tail_AL=int((last[R.al] > 0).sum()),
                group_tail_Hz=float(last[g].mean() / 0.1),
                group_active=int((last[g] > 0).sum()), persistent=bool((last > 0).sum() > 20))


def main():
    R = Runner()
    G = core_groups(R)
    # baseline tail rates (no perturbation), hot, seed 0, last 300 ms
    r = run_trial(R.Wr, stim_idx(R.meta, R.pos, 'hot'), params=R.p, seed=0, apl=R.apl,
                  t_stim=1.0, record=True)
    m = r['t'] * DT >= T_END - 0.3
    tail_hz = np.bincount(r['i'][m], minlength=len(R.meta)) / 0.3
    gdf = graph_part(R, G, tail_hz)
    gdf.to_csv(OUT / 'c7c_graph.csv', index=False)
    print(gdf.round(3).to_string(), flush=True)

    rows = []
    for name in ['lLN1_bc', 'otherLN', 'unclear']:
        for iso in [False, True]:
            for w in ['hot', 'vinegar']:
                x = isolation_trial(R, G, name, iso, w); rows.append(x)
                print(f'{name:8s} isolated={iso!s:5s} {w:8s}: tail brain={x["tail_brain"]:5d} '
                      f'AL={x["tail_AL"]:4d} group {x["group_active"]}/{G[name].sum()} '
                      f'at {x["group_tail_Hz"]:5.1f} Hz persistent={x["persistent"]}', flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'c7c_isolation.csv', index=False)
    print('DONE')


if __name__ == '__main__':
    main()
