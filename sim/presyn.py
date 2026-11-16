"""Presynaptic modulation of sensory terminals (Week 3).

Biology (fly): sensory neurons fire their spikes in the periphery (antenna,
labellum, legs, eye). Their axon terminals in the brain receive synapses from
central neurons, but these act on transmitter release, not on spiking.
  - ORN terminals carry GABA-A and GABA-B receptors. LN GABA gives a fast
    GABA-A phase and a long GABA-B phase (outward current lasting ~1-1.5 s);
    the inhibition scales with total ORN input (Olsen & Wilson 2008, Nature).
  - Sweet GRNs (not bitter GRNs) express GABA-B; GABA presynaptically inhibits
    their Ca2+ responses, and bitter stimuli suppress sweet output this way
    (Chu et al. 2014, Curr Biol).
  - No study shows central input making a sensory neuron spike. Cholinergic
    (axo-axonic) synapses onto sensory terminals exist in the connectome but
    their effect is not characterised, so they are given no effect here.
Model: remove every central -> sensory row from the chemical matrix (no effect
on the sensory membrane). Inhibitory inputs (negative weights) onto a sensory
neuron instead drive its release gain down (see lif_numpy._run_trial_brian2).
"""
import numpy as np
from scipy import sparse

TAU_A, TAU_B = 5e-3, 0.3


def sensory_mask_malecns(meta):
    return meta.superclass.astype(str).str.contains('sensory').to_numpy()


def build_presyn(W, sens, K, tauA=TAU_A, tauB=TAU_B):
    """W: (post x pre) signed weights in volts. sens: bool mask of sensory neurons.
    Returns (W without sensory rows, presyn dict, info)."""
    sens = np.asarray(sens, bool)
    W = W.tocsr()
    keep = sparse.diags((~sens).astype(np.float64))
    W_main = (keep @ W).tocsc(); W_main.eliminate_zeros()
    idx = np.flatnonzero(sens)
    Wp = W[idx].tocsr().copy()
    Wp.data = np.where(Wp.data < 0, -Wp.data, 0.0); Wp.eliminate_zeros()
    Wexc = W[idx].tocsr(); exc_sum = Wexc.data[Wexc.data > 0].sum()
    info = dict(n_sensory=int(sens.sum()), inhib_edges=int(Wp.nnz),
                inhib_sum_mV=float(Wp.sum() * 1e3), excit_removed_mV=float(exc_sum * 1e3), K=K)
    return W_main, dict(idx=idx, Wp=Wp.tocsc(), tauA=tauA, tauB=tauB, K=K), info
