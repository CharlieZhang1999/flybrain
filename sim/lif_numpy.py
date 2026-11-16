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


def run_trial(W, exc_idx, r_poi=150.0, params=P, seed=0, apl=None,
              t_stim=None, tail_from=None, w_switch=None, record=False,
              schedule='brian2', refractory_input='block', gap=None, blocks=None, presyn=None, state=None, return_state=False):
    """Simulate one trial. Returns spike-count vector (n,) over t_run.

    schedule: 'brian2' (default) reproduces Brian2's per-step order for the Shiu
      model exactly (see _run_trial_brian2). 'legacy' runs the original loop
      (approximate integrator, input applied before threshold), with
      `refractory_input` choosing 'block' or 'apply'; kept so earlier results
      can be reproduced. All other arguments behave the same in both.
    blocks: optional time-varying drive (Week 3), list of (t0, t1, idx, rate_Hz).
      Each block Poisson-drives `idx` for t0 <= t < t1 inside one continuous run,
      so network state carries over between blocks. Replaces exc_idx/r_poi/t_stim
      (pass exc_idx=[]). Every neuron in any block is a Poisson target (no
      refractory period) for the whole run, as exc_idx is in a normal trial. A
      single block (0, t_stim, exc_idx, r_poi) is bit-identical to the normal call.
      Needs schedule='brian2'.
    presyn: optional presynaptic modulation of sensory terminals (Week 3), from
      sim/presyn.py build_presyn(). Needs schedule='brian2'.
    state / return_state: save and resume (Week 3). With return_state=True the
      result (record=True required) has 'state', the full simulator state at
      t_run. Passing it back as state=... continues the same trial from there up to
      the new params['t_run'] (absolute time), with the same W, blocks (absolute
      times), apl, gap and presyn. The continuation is spike-for-spike identical to
      one uninterrupted run; only the new spikes are returned.
    """
    if presyn is not None and schedule != 'brian2':
        raise ValueError('presyn needs schedule="brian2"')
    if (state is not None or return_state) and schedule != 'brian2':
        raise ValueError('state needs schedule="brian2"')
    if blocks is not None or presyn is not None or state is not None or return_state:
        if schedule != 'brian2':
            raise ValueError('blocks need schedule="brian2"')
        return _run_trial_brian2(W, exc_idx, r_poi=r_poi, params=params, seed=seed, apl=apl,
                                 t_stim=t_stim, tail_from=tail_from, w_switch=w_switch,
                                 record=record, gap=gap, blocks=blocks, presyn=presyn,
                                 state=state, return_state=return_state)
    kw = dict(r_poi=r_poi, params=params, seed=seed, apl=apl, t_stim=t_stim,
              tail_from=tail_from, w_switch=w_switch, record=record)
    if schedule == 'brian2':
        return _run_trial_brian2(W, exc_idx, gap=gap, **kw)
    if gap is not None:
        raise ValueError('gap junctions need schedule="brian2"')
    return _run_trial_legacy(W, exc_idx, refractory_input=refractory_input, **kw)


def _run_trial_brian2(W, exc_idx, r_poi=150.0, params=P, seed=0, apl=None,
                      t_stim=None, tail_from=None, w_switch=None, record=False,
                      gap=None, blocks=None, presyn=None, state=None, return_state=False):
    """Brian2-faithful step order for the Shiu model (verified vs Brian2 2.10.1).

    Per 0.1-ms step t:
      1. not_refractory = (t - lastspike) >= t_rfc   (Poisson targets: rfc = 0)
      2. exact linear update of (v, g) for non-refractory neurons:
           g' = g e^{-dt/tau_s}
           v' = v0 + (v - v0 - A) e^{-dt/tau_m} + A e^{-dt/tau_s},
           A  = g tau_s / (tau_s - tau_m)
      3. threshold: spike if v > v_th and not refractory; spikers become refractory
      4. synapses slot: delayed g += w and Poisson v += kick, both dropped for
         refractory neurons (variables are `(unless refractory)`)
      5. reset spikers: v = v_rst, g = 0
    Optional rate-APL inhibition is applied in the synapses slot (step 4).

    Optional gap junctions (not in Shiu et al.; added in Week 2):
      gap = dict(idx=(k,) neuron indices, C=(k x k) sparse coupling matrix,
                 v_spike=spike amplitude in volts for spikelets)
      C[i, j] >= 0 is the coupling conductance from j onto i in units of the
      leak conductance (dimensionless). After step 2, each non-refractory
      coupled neuron gets   v_i += dt/tau_m * sum_j C[i, j] (v_j - v_i)
      (forward Euler on top of the exact LIF update). LIF has no spike
      waveform, so a spike of j also gives each non-refractory partner a
      spikelet  v_i += C[i, j] * v_spike  in the synapses slot (step 4).
      gap=None leaves the model exactly as before.

    Optional presynaptic modulation of sensory terminals (not in Shiu et al.;
    added in Week 3, see sim/presyn.py):
      presyn = dict(idx=(k,) sensory indices, Wp=(k x n) csc inhibitory input
                    magnitudes onto their terminals, tauA, tauB, K)
      W must already have the sensory rows removed, so central input never
      reaches a sensory neuron's membrane. Instead, each inhibitory spike onto a
      terminal adds x to a fast GABA-A trace IA (tau tauA) and x*tauA/tauB to a
      slow GABA-B trace IB (tau tauB) (equal steady-state share). The terminal's
      release gain is  gain = 1 / (1 + (IA + IB) / K),  and every outgoing
      chemical weight of that sensory neuron is multiplied by its gain at the
      moment the spike is delivered. presyn=None leaves the model as before.
    """
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    dt, nsteps = params['dt'], int(round(params['t_run'] / params['dt']))
    dly = int(round(params['t_dly'] / dt))
    rfc = int(round(params['t_rfc'] / dt))
    tm, ts = params['t_mbr'], params['tau']
    em, es = np.exp(-dt / tm), np.exp(-dt / ts)
    a_coef = ts / (ts - tm)
    v0 = params['v_0']

    v = np.full(n, v0)
    g = np.zeros(n)
    last = np.full(n, -10 ** 9, dtype=np.int64)
    rfc_n = np.full(n, rfc, dtype=np.int64)
    exc_idx = np.asarray(exc_idx, dtype=np.int64)
    rfc_n[exc_idx] = 0                                  # Poisson targets
    kick = params['w_syn'] * params['f_poi']
    p_poi = r_poi * dt
    if blocks is not None:                              # (s0, s1, idx, p) per block
        blk = [(int(round(b0 / dt)), int(round(b1 / dt)), np.asarray(ix, dtype=np.int64), r * dt)
               for b0, b1, ix, r in blocks]
        for _, _, ix, _ in blk:
            rfc_n[ix] = 0
        exc_idx = np.empty(0, dtype=np.int64)
    if presyn is not None:
        ps_idx = np.asarray(presyn['idx'], dtype=np.int64)
        ps_W = presyn['Wp'].tocsc()
        IA = np.zeros(len(ps_idx)); IB = np.zeros(len(ps_idx))
        eA, eB = np.exp(-dt / presyn['tauA']), np.exp(-dt / presyn['tauB'])
        rB, K = presyn['tauA'] / presyn['tauB'], presyn['K']
        gain = np.ones(n)
        gain_sum = np.zeros(len(ps_idx)); gain_n = 0

    ring = [np.empty(0, dtype=np.int64)] * dly
    counts = np.zeros(n, dtype=np.int64)
    tail = np.zeros(n, dtype=np.int64)
    stim_steps = nsteps if t_stim is None else int(round(t_stim / dt))
    tail_step = None if tail_from is None else int(round(tail_from / dt))
    sw_step = None if w_switch is None else int(round(w_switch[0] / dt))
    rec_t, rec_i = [], []
    if apl is not None:
        kc_idx = np.asarray(apl['kc_idx'])
        is_kc = np.zeros(n, dtype=bool); is_kc[kc_idx] = True
        w_kc = np.asarray(apl['w_kc'], dtype=np.float64)
        a_apl, kc_rate = 0.0, 0.0
        decay_a = np.exp(-dt / apl['tau'])

    if gap is not None:
        g_idx = np.asarray(gap['idx'], dtype=np.int64)
        g_C = gap['C'].tocsr()
        g_Cc = gap['C'].tocsc()
        g_deg = np.asarray(g_C.sum(axis=1)).ravel()
        g_pos = np.full(n, -1, dtype=np.int64); g_pos[g_idx] = np.arange(len(g_idx))
        g_k = dt / tm
        g_vs = gap.get('v_spike', 0.04)

    t_begin = 0
    if state is not None:                               # resume (see run_trial)
        t_begin = state['t']
        v[:] = state['v']; g[:] = state['g']; last[:] = state['last']
        ring = [x.copy() for x in state['ring']]; counts[:] = state['counts']
        rng.bit_generator.state = state['rng']
        if apl is not None:
            a_apl, kc_rate = state['a_apl'], state['kc_rate']
        if presyn is not None:
            IA[:] = state['IA']; IB[:] = state['IB']; gain[:] = state['gain']

    for t in range(t_begin, nsteps):
        if sw_step is not None and t == sw_step:
            W = w_switch[1]
        free = (t - last) >= rfc_n                      # 1
        A = g[free] * a_coef                            # 2
        v[free] = v0 + (v[free] - v0 - A) * em + A * es
        g[free] *= es
        if gap is not None:                             # 2b: gap-junction current
            vg = v[g_idx]
            dv = g_k * (g_C @ vg - g_deg * vg)
            fg = free[g_idx]
            v[g_idx[fg]] += dv[fg]
        spk = (v > params['v_th']) & free               # 3
        idx = np.flatnonzero(spk)
        if len(idx):
            last[idx] = t
            free[idx] = False
        pre = ring[t % dly]                             # 4: synapses slot
        if presyn is not None:
            IA *= eA; IB *= eB
            if len(pre):
                x = np.asarray(ps_W[:, pre].sum(axis=1)).ravel()
                IA += x; IB += x * rB
            gain[ps_idx] = 1.0 / (1.0 + (IA + IB) / K)
            gain_sum += gain[ps_idx]; gain_n += 1
        if len(pre):
            if presyn is None:
                upd = np.asarray(W[:, pre].sum(axis=1)).ravel()
            else:
                upd = np.asarray(W[:, pre] @ gain[pre]).ravel()
            g[free] += upd[free]
        if t < stim_steps and len(exc_idx):
            hit = exc_idx[rng.random(len(exc_idx)) < p_poi]
            hit = hit[free[hit]]
            v[hit] += kick
        if blocks is not None:
            for s0, s1, ix, pb in blk:
                if s0 <= t < s1 and len(ix):
                    hit = ix[rng.random(len(ix)) < pb]
                    hit = hit[free[hit]]
                    v[hit] += kick
        if gap is not None and len(idx):                # spikelets
            gs = g_pos[idx]; gs = gs[gs >= 0]
            if len(gs):
                sl = np.asarray(g_Cc[:, gs].sum(axis=1)).ravel() * g_vs
                fg = free[g_idx]
                v[g_idx[fg]] += sl[fg]
        if apl is not None:
            a_apl = a_apl * decay_a + (1 - decay_a) * apl['k'] * kc_rate
            g[kc_idx] -= np.where(free[kc_idx], w_kc * a_apl * dt, 0.0)
        if len(idx):                                    # 5: reset
            v[idx] = params['v_rst']
            g[idx] = 0.0
            counts[idx] += 1
            if record:
                rec_t.append(np.full(len(idx), t, dtype=np.int32)); rec_i.append(idx)
            if tail_step is not None and t >= tail_step:
                tail[idx] += 1
        ring[t % dly] = idx
        if apl is not None:
            kc_rate = is_kc[idx].sum() / dt if len(idx) else 0.0

    if record:
        out = dict(counts=counts, tail=tail,
                   t=np.concatenate(rec_t) if rec_t else np.empty(0, np.int32),
                   i=np.concatenate(rec_i) if rec_i else np.empty(0, np.int64))
        if presyn is not None:
            out['presyn_mean_gain'] = gain_sum / max(gain_n, 1)
        if return_state:
            st = dict(t=nsteps, v=v.copy(), g=g.copy(), last=last.copy(),
                      ring=[x.copy() for x in ring], counts=counts.copy(),
                      rng=rng.bit_generator.state)
            if apl is not None:
                st.update(a_apl=a_apl, kc_rate=kc_rate)
            if presyn is not None:
                st.update(IA=IA.copy(), IB=IB.copy(), gain=gain.copy())
            out['state'] = st
        return out
    return counts if tail_step is None else (counts, tail)


def _run_trial_legacy(W, exc_idx, r_poi=150.0, params=P, seed=0, apl=None,
                      t_stim=None, tail_from=None, w_switch=None, record=False,
                      refractory_input='block'):
    """Original loop (pre 2026-10-30), kept for reproducing earlier results.

    apl (optional): graded, non-spiking APL feedback onto Kenyon cells.
      dict(kc_idx, w_kc, k, tau)
        kc_idx : indices of Kenyon cells
        w_kc   : per-KC inhibitory weight (volts per APL "event"), from the real
                 APL->KC synapse counts x w_syn (positive numbers)
        k      : gain, APL events per KC spike (the one fitted parameter)
        tau    : APL time constant (s)
      APL level a(t) follows  da/dt = (-a + k * KC_spike_rate) / tau,
      and each step every KC gets  g -= w_kc * a * dt.  No refractory ceiling,
      so inhibition grows in proportion to total KC activity.
      The spiking APL neurons should be removed from W when this is used.
    t_stim (optional): stop the Poisson drive after t_stim seconds.
    tail_from (optional): also count spikes from tail_from seconds to the end;
      the function then returns (counts, tail_counts). Used to check that
      activity dies out after the stimulus ends.
    w_switch (optional): (t_switch, W_new) replace the weight matrix at
      t_switch seconds, leaving all membrane/synaptic state untouched.
    record (optional): if True, also return spike times as two arrays
      (step indices, neuron indices); the return value becomes a dict.
    refractory_input: 'block' (default) matches Brian2 / Shiu et al.: because v
      and g are declared `(unless refractory)`, synaptic writes `g += w` that
      arrive 1..21 steps after a spike are discarded; integration and input
      resume at step 22 (= t_rfc). Verified step-for-step against Brian2 2.10.1
      in scripts/c0_refractory_toy.py.
      'apply' reproduces this simulator's behaviour before 2026-10-30 (input
      during refractoriness was added to g and kept); results in
      results/malecns/ckpt2-5 and the Week 1 FlyWire numbers used 'apply'.
    """
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
    tail = np.zeros(n, dtype=np.int64)
    stim_steps = nsteps if t_stim is None else int(round(t_stim / dt))
    tail_step = None if tail_from is None else int(round(tail_from / dt))

    sw_step = None if w_switch is None else int(round(w_switch[0] / dt))
    rec_t, rec_i = [], []

    if apl is not None:
        kc_idx = np.asarray(apl['kc_idx'])
        is_kc = np.zeros(n, dtype=bool); is_kc[kc_idx] = True
        w_kc = np.asarray(apl['w_kc'], dtype=np.float64)
        a_apl, kc_rate = 0.0, 0.0
        decay_a = np.exp(-dt / apl['tau'])

    for t in range(nsteps):
        if sw_step is not None and t == sw_step:
            W = w_switch[1]
        if refractory_input == 'block':
            refr[refr > 0] -= 1                   # Brian2: free again at step t_rfc
            active = refr == 0
        else:                                     # legacy ordering
            active = refr == 0
            refr[~active] -= 1
        # state update (exact for g; exponential Euler for v with g held)
        g[active] *= decay_g
        vinf = params['v_0'] + g[active]
        v[active] = vinf + (v[active] - vinf) * decay_v

        # delayed synaptic input: g += sum of w from spikes at t - dly
        pre = ring[t % dly]
        if len(pre):
            upd = np.asarray(W[:, pre].sum(axis=1)).ravel()
            if refractory_input == 'block':
                g[active] += upd[active]          # writes to refractory neurons dropped
            else:
                g += upd                          # legacy: kept during refractoriness

        # graded APL feedback (rate unit) onto Kenyon cells
        if apl is not None:
            a_apl = a_apl * decay_a + (1 - decay_a) * apl['k'] * kc_rate
            if refractory_input == 'block':
                g[kc_idx] -= np.where(active[kc_idx], w_kc * a_apl * dt, 0.0)
            else:
                g[kc_idx] -= w_kc * a_apl * dt

        # Poisson drive straight to v (Brian2 PoissonInput target_var='v')
        if t < stim_steps:
            hit = exc_idx[rng.random(len(exc_idx)) < p_poi]
            v[hit] += kick

        # threshold / reset
        spk = (v > params['v_th']) & active
        if spk.any():
            idx = np.flatnonzero(spk)
            counts[idx] += 1
            if record:
                rec_t.append(np.full(len(idx), t, dtype=np.int32)); rec_i.append(idx)
            if tail_step is not None and t >= tail_step:
                tail[idx] += 1
            v[idx] = params['v_rst']
            g[idx] = 0.0
            refr[idx] = np.where(no_rfc[idx], 0, rfc)
            ring[t % dly] = idx
            if apl is not None:
                kc_rate = is_kc[idx].sum() / dt      # KC spikes per second
        else:
            ring[t % dly] = np.empty(0, dtype=np.int64)
            if apl is not None:
                kc_rate = 0.0

    if record:
        return dict(counts=counts, tail=tail,
                    t=np.concatenate(rec_t) if rec_t else np.empty(0, np.int32),
                    i=np.concatenate(rec_i) if rec_i else np.empty(0, np.int64))
    return counts if tail_step is None else (counts, tail)


def run_exp(W, ids, exc_flyids, r_poi=150.0, n_run=5, params=P, seed0=0):
    """Average rates (Hz) over n_run trials. Returns pd.Series by flywire id."""
    pos = {f: i for i, f in enumerate(ids)}
    exc_idx = np.array([pos[f] for f in exc_flyids])
    acc = np.zeros(len(ids))
    for k in range(n_run):
        acc += run_trial(W, exc_idx, r_poi=r_poi, params=params, seed=seed0 + k)
    rates = acc / (n_run * params['t_run'])
    return pd.Series(rates, index=ids, name='rate_hz')
