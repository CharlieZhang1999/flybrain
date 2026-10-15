"""NumPy reimplementation of the Shiu et al. 2024 whole-brain LIF model.

Semantics copied from third_party/Drosophila_brain_model/model.py:
  dv/dt = (v_0 - v + g) / t_mbr   (unless refractory)
  dg/dt = -g / tau                 (unless refractory)
  spike: v > v_th  ->  v = v_rst, g = 0, refractory t_rfc
  synapse: on_pre g += w, delay t_dly, w = (Excitatory x Connectivity) * w_syn
  stimulation: Poisson N=1 at r_poi, weight w_syn * f_poi added to v,
               stimulated neurons have no refractory period.
Integration: exponential update per dt = 0.1 ms (Brian2 default clock).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

# Shiu et al. default parameters (SI units: volts, seconds)
P = dict(
    dt=1e-4, t_run=1.0,
    v_0=-52e-3, v_rst=-52e-3, v_th=-45e-3,
    t_mbr=20e-3, tau=5e-3, t_rfc=2.2e-3, t_dly=1.8e-3,
    w_syn=0.275e-3, f_poi=250,
)


def load_network(path_comp, path_con):
    """Return (flywire_ids ndarray, W csr matrix [post x pre] in volts)."""
    df_comp = pd.read_csv(path_comp, index_col=0)
    ids = df_comp.index.to_numpy(dtype=np.int64)
    n = len(ids)
    con = pd.read_parquet(path_con,
                          columns=['Presynaptic_Index', 'Postsynaptic_Index',
                                   'Excitatory x Connectivity'])
    W = sparse.csr_matrix(
        (con['Excitatory x Connectivity'].to_numpy(dtype=np.float64) * P['w_syn'],
         (con['Postsynaptic_Index'].to_numpy(), con['Presynaptic_Index'].to_numpy())),
        shape=(n, n)).tocsc()   # csc: fast column slicing by presynaptic spike
    return ids, W


def run_trial(W, exc_idx, r_poi=150.0, params=P, seed=0):
    """Simulate one trial. Returns spike-count vector (n,) over t_run."""
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    dt, nsteps = params['dt'], int(round(params['t_run'] / params['dt']))
    dly = int(round(params['t_dly'] / dt))
    rfc = int(round(params['t_rfc'] / dt))

    v = np.full(n, params['v_0'])
    g = np.zeros(n)
    refr = np.zeros(n, dtype=np.int32)         # steps of refractoriness left
    no_rfc = np.zeros(n, dtype=bool)
    no_rfc[exc_idx] = True                      # Poisson targets: rfc = 0

    decay_v = np.exp(-dt / params['t_mbr'])
    decay_g = np.exp(-dt / params['tau'])
    kick = params['w_syn'] * params['f_poi']    # Poisson weight -> v
    p_poi = r_poi * dt                          # Poisson prob per step
    exc_idx = np.asarray(exc_idx)

    ring = [np.empty(0, dtype=np.int64)] * dly  # delayed spike buffer
    counts = np.zeros(n, dtype=np.int64)

    for t in range(nsteps):
        active = refr == 0
        # state update (exact for g; exponential Euler for v with g held)
        g[active] *= decay_g
        vinf = params['v_0'] + g[active]
        v[active] = vinf + (v[active] - vinf) * decay_v
        refr[~active] -= 1

        # delayed synaptic input: g += sum of w from spikes at t - dly
        pre = ring[t % dly]
        if len(pre):
            upd = np.asarray(W[:, pre].sum(axis=1)).ravel()
            g += upd    # brian2: on_pre fires even into refractory neurons

        # Poisson drive straight to v (Brian2 PoissonInput target_var='v')
        hit = exc_idx[rng.random(len(exc_idx)) < p_poi]
        v[hit] += kick

        # threshold / reset
        spk = (v > params['v_th']) & active
        if spk.any():
            idx = np.flatnonzero(spk)
            counts[idx] += 1
            v[idx] = params['v_rst']
            g[idx] = 0.0
            refr[idx] = np.where(no_rfc[idx], 0, rfc)
            ring[t % dly] = idx
        else:
            ring[t % dly] = np.empty(0, dtype=np.int64)

    return counts


def run_exp(W, ids, exc_flyids, r_poi=150.0, n_run=5, params=P, seed0=0):
    """Average rates (Hz) over n_run trials. Returns pd.Series by flywire id."""
    pos = {f: i for i, f in enumerate(ids)}
    exc_idx = np.array([pos[f] for f in exc_flyids])
    acc = np.zeros(len(ids))
    for k in range(n_run):
        acc += run_trial(W, exc_idx, r_poi=r_poi, params=params, seed=seed0 + k)
    rates = acc / (n_run * params['t_run'])
    return pd.Series(rates, index=ids, name='rate_hz')
