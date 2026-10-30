"""Config-driven, reversible weight-matrix lesions for any part of the network.

Every function returns a NEW matrix; the input matrix (and the cached base
network on disk) is never modified. An empty config returns an unchanged copy.

Neuron groups are selected with pandas query strings over the metadata table
(columns: type, class, superclass, nt, somaSide, somaNeuromere, ...), e.g.
    "type == 'lLN1_bc'"
    "`class` == 'ALLN' and nt == 'acetylcholine'"

Config: a list of operations applied in order, each a dict:
    {"op": "silence_out", "group": Q}            outputs of Q -> 0 (inputs kept)
    {"op": "silence_in",  "group": Q}            inputs of Q -> 0 (outputs kept)
    {"op": "scale_edges", "pre": Q1, "post": Q2, "alpha": a}
                                                 edges Q1 -> Q2 multiplied by a
    {"op": "set_sign", "group": Q, "sign": s}    outputs of Q forced to sign s
                                                 (+1 excitatory, -1 inhibitory, 0 none),
                                                 keeping |weight| = synapse count x w_syn
"""
import numpy as np
from scipy import sparse


def select(meta, query):
    """Boolean mask over neurons for a pandas query string."""
    m = np.zeros(len(meta), dtype=bool)
    m[meta.reset_index(drop=True).query(query, engine='python').index.to_numpy()] = True
    return m


def apply(W, meta, config):
    """Return (new W, log) after applying config ops to a copy of W."""
    C = W.tocoo(copy=True)
    data = C.data.astype(np.float64).copy()
    rows, cols = C.row, C.col                       # rows = post, cols = pre
    log = []
    for op in config or []:
        kind = op['op']
        if kind in ('silence_out', 'silence_in', 'set_sign'):
            g = select(meta, op['group'])
            hit = g[cols] if kind != 'silence_in' else g[rows]
            if kind == 'set_sign':
                data[hit] = np.abs(data[hit]) * op['sign']
            else:
                data[hit] = 0.0
            log.append(dict(op=kind, group=op['group'], neurons=int(g.sum()),
                            edges=int(hit.sum()), **({'sign': op['sign']} if kind == 'set_sign' else {})))
        elif kind == 'scale_edges':
            a, b = select(meta, op['pre']), select(meta, op['post'])
            hit = a[cols] & b[rows]
            data[hit] *= op['alpha']
            log.append(dict(op=kind, pre=op['pre'], post=op['post'], alpha=op['alpha'],
                            edges=int(hit.sum())))
        else:
            raise ValueError(f'unknown op {kind}')
    out = sparse.csc_matrix((data, (rows, cols)), shape=W.shape)
    out.eliminate_zeros()
    return out, log
