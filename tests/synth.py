"""Tiny synthetic network for CPU tests (no data files).

Layout of the default 300-neuron network (indices):
  0..19     sensory neurons (Poisson-driven; presynaptic modulation targets)
  20..49    Kenyon cells (APL feedback targets)
  50..69    gap-junction block
  70..299   other neurons
Weights are post x pre in volts, like lif_numpy.load_network:
  signed synapse count x w_syn x scale.
"""
import numpy as np
from scipy import sparse

from sim.lif_numpy import P
from sim.presyn import build_presyn


def make_network(n=300, density=0.05, frac_inh=0.2, scale=4.0, n_sens=20, n_kc=30,
                 n_gap=20, seed=1234):
    """Return a dict with W (csc, all rows), W_ps / presyn (sensory rows removed,
    presyn dict for run_trial), gap, apl, sens, kc, params. Fully seeded."""
    rng = np.random.default_rng(seed)
    nnz_mask = rng.random((n, n)) < density
    np.fill_diagonal(nnz_mask, False)
    post, pre = np.nonzero(nnz_mask)
    counts = rng.integers(1, 11, size=len(post)).astype(np.float64)
    sign = np.where(rng.random(len(post)) < frac_inh, -1.0, 1.0)
    W = sparse.csc_matrix((sign * counts * P['w_syn'] * scale, (post, pre)), shape=(n, n))

    sens = np.arange(n_sens)
    kc = np.arange(n_sens, n_sens + n_kc)
    g_idx = np.arange(n_sens + n_kc, n_sens + n_kc + n_gap)

    # gap junctions: symmetric random coupling inside the block
    c = np.triu(rng.random((n_gap, n_gap)) < 0.3, 1) * rng.uniform(0.05, 0.3, (n_gap, n_gap))
    C = sparse.csr_matrix(c + c.T)
    gap = dict(idx=g_idx, C=C, v_spike=0.002)

    # rate-APL feedback on the KCs (positive per-KC weights, volts per event)
    apl = dict(kc_idx=kc, w_kc=rng.uniform(2, 20, n_kc) * P['w_syn'], k=0.1, tau=0.01)

    # presynaptic modulation of the sensory terminals
    is_sens = np.zeros(n, bool); is_sens[sens] = True
    W_ps, presyn, _ = build_presyn(W, is_sens, K=0.05 * P['w_syn'] * 10)

    return dict(W=W, W_ps=W_ps, presyn=presyn, gap=gap, apl=apl, sens=sens, kc=kc,
                gap_idx=g_idx)


def params(t_run=0.2):
    p = dict(P)
    p['t_run'] = t_run
    return p
