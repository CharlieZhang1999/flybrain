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


# --- Week 4: giant-fiber electrical synapses (rectifying) -------------------
# In the fly the giant fiber (GF, DNp01) drives the jump muscle motor neuron
# (TTMn) and the peripherally synapsing interneuron (PSI, -> DLMn flight power
# muscles) mainly through ShakB gap junctions, which rectify (GF -> target).
# EM counts only the chemical part. A rectifying junction is a one-way row in
# C: C[target, GF] = c, C[GF, target] = 0. With the simulator's gap update
# (dv_i = dt/tm * sum_j C_ij (v_j - v_i); spikelet C_ij * v_spike on a spike of
# j), the target is pulled toward GF voltage and gets a spikelet of
# c * v_spike when GF fires; GF itself is not loaded by the targets.

def build_gf_gap(meta, spikelet_mV, v_spike=0.002, targets=('TTMn', 'PSI')):
    """Rectifying GF -> TTMn / PSI junctions, ipsilateral pairs (by somaSide).

    spikelet_mV: depolarisation of the target per GF spike (sets c =
    spikelet / v_spike). Returns a gap dict on the global index plus info."""
    t = meta.type.astype(str).values
    side = meta.somaSide.astype(str).values
    gf = np.flatnonzero(t == 'DNp01')
    pairs = []
    for g in gf:
        for typ in targets:
            for j in np.flatnonzero((t == typ) & (side == side[g])):
                pairs.append((j, g))
    c = spikelet_mV * 1e-3 / v_spike
    idx = np.unique(np.array(pairs).ravel())
    pos = {k: i for i, k in enumerate(idx)}
    r = [pos[a] for a, _ in pairs]; q = [pos[b] for _, b in pairs]
    C = sparse.csr_matrix((np.full(len(pairs), c), (r, q)), shape=(len(idx), len(idx)))
    info = dict(pairs=len(pairs), c=c, spikelet_mV=spikelet_mV,
                names=[(t[a], side[a], t[b], side[b]) for a, b in pairs])
    return dict(idx=idx, C=C, v_spike=v_spike), info


def merge_gap(a, b):
    """Block-diagonal union of two gap dicts with the same v_spike and
    disjoint neuron sets."""
    if a is None:
        return b
    assert a.get('v_spike', 0.04) == b.get('v_spike', 0.04)
    assert not np.intersect1d(a['idx'], b['idx']).size
    C = sparse.block_diag([a['C'], b['C']]).tocsr()
    return dict(idx=np.concatenate([a['idx'], b['idx']]), C=C, v_spike=a.get('v_spike', 0.04))
