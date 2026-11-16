"""eLN-PN gap junctions for the Week 2 calibration.

The EM connectome shows chemical synapses only, so electrical coupling is
placed where an eLN and a PN have chemical contact (either direction), as a
proxy for membrane contact. Pair coupling scales with contact size:
    c_ij = c0 * min(s_ij / s_ref, 1),   s_ij = syn(i->j) + syn(j->i)
with s_ref = 50 synapses (90th percentile of MaleCNS eLN-PN contacts), so
strong pairs get c0. Yaksi & Wilson 2010 report small eLN-PN coupling
coefficients (Fig. 4C axis up to 0.04). Coupling is symmetric.
"""
import numpy as np
from scipy import sparse

S_REF = 50.0


def pn_glomeruli(t):
    """'VP1m+VP2_lvPN2' -> {'VP1m', 'VP2'}; non-uniglomerular 'M_...' -> set()."""
    if not isinstance(t, str) or '_' not in t or t.startswith('M_'):
        return set()
    return set(t.split('_')[0].split('+'))


def build_gap(W, ln_mask, pn_mask, c0, s_ref=S_REF, min_syn=1, cap=None, v_spike=0.04):
    """W: [post x pre] matrix whose |entries| are proportional to synapse counts
    (pass the unscaled synapse-count matrix). Returns gap dict for run_trial.
    min_syn: only contacts with >= min_syn synapses get a junction.
    cap: if set, scale each pair by min(1, cap / sum_c(i), cap / sum_c(j)) so no
         neuron's total coupling exceeds cap (keeps C symmetric)."""
    L, Pn = np.flatnonzero(ln_mask), np.flatnonzero(pn_mask)
    A = abs(W[Pn][:, L]).toarray()          # LN -> PN (rows PN, cols LN)
    B = abs(W[L][:, Pn]).toarray().T        # PN -> LN, as PN x LN
    S = A + B
    S[S < min_syn] = 0
    c = c0 * np.minimum(S / s_ref, 1.0)
    if cap is not None:
        sp, sl = c.sum(axis=1), c.sum(axis=0)          # per PN, per LN
        fp = np.minimum(1.0, cap / np.maximum(sp, 1e-12))
        fl = np.minimum(1.0, cap / np.maximum(sl, 1e-12))
        c = c * np.minimum(fp[:, None], fl[None, :])
    k = len(L) + len(Pn)
    idx = np.concatenate([L, Pn])
    top = sparse.csr_matrix(c)              # PN x LN block
    C = sparse.bmat([[None, top.T], [top, None]]).tocsr()
    C.eliminate_zeros()
    deg = np.asarray(C.sum(axis=1)).ravel()
    info = dict(pairs=int((c > 0).sum()), c0=c0, min_syn=min_syn, cap=cap,
                sum_c_PN_median=float(np.median(deg[len(L):])),
                sum_c_LN_median=float(np.median(deg[:len(L)])))
    return dict(idx=idx, C=C, v_spike=v_spike), info
