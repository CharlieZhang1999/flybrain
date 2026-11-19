#!/usr/bin/env python
"""W4D5 T2c (Q1): connectome view of the post-wind state (no simulation).

Purpose: (1) rank the excitatory inputs that hold each tail cell type, as a mean-field
estimate dV = tau_syn (5 ms) x sum_j W_ij x rate_j (rates = Day 5 ep009, 2.1-2.5 s; an
estimate, not a membrane potential); (2) shortest hop paths from the wind word (JO-C/E)
over excitatory edges >= 1 mV; (3) synapse counts inside the DNge027 loop.
Usage:  python scripts/w4d5_burst_loop.py
Output: results/w4/regression/burst_diag/analyze_loop.txt
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.w4d5_burst_common as C  # noqa: E402

TARGETS = [['DNge027'], ['GNG423'], ['DNg21'], ['DNg62'], ['DNge044'], ['DNge019'],
           ['DNg12_a', 'DNg12_c', 'DNg12_e', 'DNg12_f'], ['GNG282'], ['MeVC4a', 'MeVC4b', 'MeVC1'],
           ['MeVCMe1'], ['Mi18'], ['Dm17'], ['Pm8'], ['MDN']]


def main():
    from sim.cns import cns_model, W4_CALIB
    from sim.malecns import load
    from stimuli.mixtures import word_idx
    m = cns_model('cns', 'calib', verbose=False, **W4_CALIB)
    meta, W = m['meta'], m['W'].tocsr()
    ty = meta.type.fillna('?').astype(str).values
    sc = meta.superclass.fillna('?').astype(str).values
    sens = pd.Series(sc).str.contains('sensory').values
    t, i, _ = C.load_run(C.DAY5 / 'ep009_calib.npz')
    k = (t >= 21000) & ~sens[i]
    r = np.bincount(i[k], minlength=len(meta)) / 0.4
    out = ['## Inputs in the tail state (mean-field mV per target neuron; rates from ep009 2.1-2.5 s)', '']
    for tt in TARGETS:
        tg = np.flatnonzero(np.isin(ty, tt))
        s = pd.Series(np.asarray(W[tg].multiply(r[None, :]).sum(axis=0)).ravel() * 5.0 / len(tg))
        s = s[s != 0]
        g = s.groupby(ty[s.index]).sum().sort_values()
        out.append(f'{"+".join(tt)} (n {len(tg)}, tail {r[tg].mean():.0f} Hz): net {g.sum():.1f}; exc '
                   + ', '.join(f'{a} {b:.1f}' for a, b in g[::-1].head(6).items())
                   + ' | inh ' + ', '.join(f'{a} {b:.1f}' for a, b in g.head(3).items()))
    pos = {b: j for j, b in enumerate(meta.bodyId)}
    wind = word_idx(meta, pos, 'wind')
    A = W.T.tocsr().copy(); A.data = (A.data * 1e3 >= 1.0).astype(float); A.eliminate_zeros()
    dist = np.full(len(meta), -1); dist[wind] = 0; pred = np.full(len(meta), -1); fr = wind; h = 0
    while len(fr) and h < 8:
        h += 1; nb = A[fr].tocoo(); new = []
        for pre, post in zip(fr[nb.row], nb.col):
            if dist[post] < 0:
                dist[post] = h; pred[post] = pre; new.append(post)
        fr = np.array(sorted(set(new)), dtype=int)
    out += ['', '## Shortest paths from JO-C/E (wind) over excitatory edges >= 1 mV (one example path)', '']
    for tt in TARGETS:
        ix = np.flatnonzero(np.isin(ty, tt)); ix = ix[dist[ix] >= 0]
        if not len(ix):
            out.append(f'{tt}: unreached'); continue
        j = ix[np.argmin(dist[ix])]; p = [j]
        while pred[p[-1]] >= 0:
            p.append(pred[p[-1]])
        out.append(f'{"+".join(tt)}: {dist[j]} hops: ' + ' -> '.join(ty[q] for q in p[::-1]))
    _, W1 = load(w_syn=1.0, monoamine='removed'); W1 = W1.tocsr()
    d = np.flatnonzero(ty == 'DNge027')
    out += ['', '## DNge027 loop synapse counts (MaleCNS)', '',
            f'DNge027 L->R {W1[d[1], d[0]]:.0f}, R->L {W1[d[0], d[1]]:.0f} synapses; model weight '
            f'{W[d[1], d[0]] * 1e3:.1f} / {W[d[0], d[1]] * 1e3:.1f} mV per spike']
    for tt in ['GNG423', 'DNg21', 'DNg62', 'DNge044']:
        x = np.flatnonzero(ty == tt)
        out.append(f'{tt} -> DNge027 {W1[d][:, x].sum():.0f}, DNge027 -> {tt} {W1[x][:, d].sum():.0f} synapses')
    (C.OUT / 'analyze_loop.txt').write_text('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
