"""Week 4: PyTorch port of sim/lif_numpy._run_trial_brian2 for the full CNS.

Same equations, step order and options (blocks, gap, apl, presyn, t_stim) as the
NumPy simulator; see that docstring for the model. Differences:
  - Runs on 'cpu', 'cuda' or 'mps'. dtype float64 by default (float32 on MPS,
    which has no float64).
  - Synaptic input is event-driven: only the CSC columns of neurons that spiked
    t_dly ago are gathered and scattered (index_add_), so cost scales with
    spikes x out-degree, not with nnz.
  - rng='numpy' draws the Poisson drive from np.random.default_rng(seed) in the
    same order as lif_numpy, so on CPU/float64 the result is spike-for-spike
    identical to the NumPy simulator (scripts/w4d1_torch_parity.py).
    rng='torch' draws on the device (faster on GPUs; statistically equivalent).
  - Save/resume (Week 4 day 2): return_state=True adds 'state' to the result;
    state=... continues the same trial up to the new params['t_run'] (absolute
    time), spike-for-spike identical to one uninterrupted run (same contract as
    lif_numpy; scripts/w4d2_resume_check.py). The state keeps tensors on the
    device, so chained episodes never leave the GPU.
  - compile=True runs the elementwise LIF update (steps 1-2) through
    torch.compile; the event-driven parts have data-dependent shapes and stay
    eager.
"""
import numpy as np
import torch

from sim.lif_numpy import P


class _CSC:
    """CSC matrix on a device, with an event gather for a set of columns."""

    def __init__(self, M, device, dtype):
        M = M.tocsc(); M.sort_indices()
        self.n = M.shape[0]
        self.indptr = torch.as_tensor(M.indptr.astype(np.int64), device=device)
        self.indices = torch.as_tensor(M.indices.astype(np.int64), device=device)
        self.data = torch.as_tensor(M.data, dtype=dtype, device=device)

    def gather(self, cols):
        """rows, vals, col_pos (position of each entry's column in `cols`)."""
        starts = self.indptr[cols]
        lens = self.indptr[cols + 1] - starts
        tot = int(lens.sum())
        if tot == 0:
            return None
        pos = torch.repeat_interleave(torch.arange(len(cols), device=cols.device), lens,
                                      output_size=tot)
        first = torch.cumsum(lens, 0) - lens
        offs = starts[pos] + torch.arange(tot, device=cols.device) - first[pos]
        return self.indices[offs], self.data[offs], pos

    def colsum(self, cols, weights=None):
        out = torch.zeros(self.n, dtype=self.data.dtype, device=self.data.device)
        r = self.gather(cols)
        if r is not None:
            rows, vals, pos = r
            if weights is not None:
                vals = vals * weights[pos]
            out.index_add_(0, rows, vals)
        return out


def _decay(v, g, last, rfc_n, t, v0, a_coef, em, es):
    """Steps 1-2: refractory mask and exact linear (v, g) update."""
    free = (t - last) >= rfc_n
    A = torch.where(free, g * a_coef, 0.0)
    v = torch.where(free, v0 + (v - v0 - A) * em + A * es, v)
    g = torch.where(free, g * es, g)
    return v, g, free


_decay_compiled = None


def default_device():
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def run_trial(W, exc_idx, r_poi=150.0, params=P, seed=0, apl=None, t_stim=None,
              record=False, gap=None, blocks=None, presyn=None, device=None,
              dtype=None, rng='numpy', bin_steps=None, bin_groups=None, compile=False,
              state=None, return_state=False):
    """Simulate one trial. Returns counts (n,) or, with record=True, a dict with
    counts, t (step of each spike), i (neuron of each spike).

    bin_steps / bin_groups: optional on-device binning for long runs. With
    bin_steps=k, the result also has 'binned' = (n_bins, n) spike counts per
    k-step bin (or (n_bins, len(bin_groups)) summed over each group, if
    bin_groups is a list of index arrays). Avoids recording every spike. On resume, bins start at the resume step.

    state / return_state: see module docstring. counts are cumulative over the
    whole chained trial (as in lif_numpy); 't'/'i' and 'binned' hold only the
    new segment.
    """
    device = device or default_device()
    if dtype is None:
        dtype = torch.float32 if device == 'mps' else torch.float64
    tk = dict(device=device, dtype=dtype)
    n = W.shape[0]
    dt, nsteps = params['dt'], int(round(params['t_run'] / params['dt']))
    dly = int(round(params['t_dly'] / dt))
    rfc = int(round(params['t_rfc'] / dt))
    tm, ts = params['t_mbr'], params['tau']
    em, es = float(np.exp(-dt / tm)), float(np.exp(-dt / ts))
    a_coef = ts / (ts - tm)
    v0, v_th, v_rst = params['v_0'], params['v_th'], params['v_rst']
    kick = params['w_syn'] * params['f_poi']

    Wd = _CSC(W, device, dtype)
    v = torch.full((n,), v0, **tk)
    g = torch.zeros(n, **tk)
    last = torch.full((n,), -10 ** 9, dtype=torch.int64, device=device)
    rfc_n = torch.full((n,), rfc, dtype=torch.int64, device=device)
    exc_np = np.asarray(exc_idx, dtype=np.int64)
    stim_steps = nsteps if t_stim is None else int(round(t_stim / dt))
    drive = []                                   # (s0, s1, idx_np, idx_dev, p)
    if blocks is not None:
        for b0, b1, ix, r in blocks:
            ix = np.asarray(ix, dtype=np.int64)
            drive.append((int(round(b0 / dt)), int(round(b1 / dt)), ix,
                          torch.as_tensor(ix, device=device), r * dt))
    elif len(exc_np):
        drive.append((0, stim_steps, exc_np, torch.as_tensor(exc_np, device=device), r_poi * dt))
    for _, _, ix_np, ix, _ in drive:
        rfc_n[ix] = 0
    np_rng = np.random.default_rng(seed)
    t_gen = torch.Generator(device=device); t_gen.manual_seed(int(seed))

    if presyn is not None:
        ps_idx = torch.as_tensor(np.asarray(presyn['idx'], np.int64), device=device)
        ps_W = _CSC(presyn['Wp'], device, dtype)
        IA = torch.zeros(len(ps_idx), **tk); IB = torch.zeros(len(ps_idx), **tk)
        eA, eB = float(np.exp(-dt / presyn['tauA'])), float(np.exp(-dt / presyn['tauB']))
        rB, K = presyn['tauA'] / presyn['tauB'], presyn['K']
        gain = torch.ones(n, **tk)
    if gap is not None:
        g_idx = torch.as_tensor(np.asarray(gap['idx'], np.int64), device=device)
        C = gap['C'].tocsr()
        if device == 'cpu':   # sparse CSR: same summation order as scipy (exact parity)
            g_C = torch.sparse_csr_tensor(torch.as_tensor(C.indptr.astype(np.int64)),
                                          torch.as_tensor(C.indices.astype(np.int64)),
                                          torch.as_tensor(C.data), size=C.shape).to(**tk)
        else:                 # small (830 x 830): dense works on CUDA and MPS
            g_C = torch.as_tensor(C.toarray(), **tk)
        g_deg = torch.as_tensor(np.asarray(C.sum(axis=1)).ravel(), **tk)
        g_Cc = _CSC(gap['C'], device, dtype)
        g_pos = torch.full((n,), -1, dtype=torch.int64, device=device)
        g_pos[g_idx] = torch.arange(len(g_idx), device=device)
        g_k, g_vs = dt / tm, gap.get('v_spike', 0.04)
    if apl is not None:
        kc_idx = torch.as_tensor(np.asarray(apl['kc_idx'], np.int64), device=device)
        is_kc = torch.zeros(n, dtype=torch.bool, device=device); is_kc[kc_idx] = True
        w_kc = torch.as_tensor(np.asarray(apl['w_kc'], np.float64), **tk)
        a_apl, kc_rate = 0.0, 0.0
        decay_a = float(np.exp(-dt / apl['tau']))

    empty = torch.empty(0, dtype=torch.int64, device=device)
    ring = [empty] * dly
    counts = torch.zeros(n, dtype=torch.int64, device=device)
    rec_t, rec_i = [], []
    t_begin = 0
    if state is not None:                                    # resume
        t_begin = int(state['t'])
        _t = lambda x, dt_: torch.as_tensor(x, device=device, dtype=dt_).clone()
        v = _t(state['v'], dtype); g = _t(state['g'], dtype)
        last = _t(state['last'], torch.int64)
        ring = [_t(x, torch.int64) for x in state['ring']]
        counts = _t(state['counts'], torch.int64)
        if state.get('np_rng') is not None:
            np_rng.bit_generator.state = state['np_rng']
        if state.get('t_gen') is not None:
            t_gen.set_state(state['t_gen'])
        if apl is not None:
            a_apl, kc_rate = state['a_apl'], state['kc_rate']
        if presyn is not None:
            IA = _t(state['IA'], dtype); IB = _t(state['IB'], dtype)
            gain = _t(state['gain'], dtype)
    if bin_steps:
        nb = -(-(nsteps - t_begin) // bin_steps)
        if bin_groups is None:
            binned = torch.zeros((nb, n), dtype=torch.int32, device=device)
        else:
            gid = torch.full((n,), -1, dtype=torch.int64, device=device)
            for k_, ix in enumerate(bin_groups):
                gid[torch.as_tensor(np.asarray(ix, np.int64), device=device)] = k_
            binned = torch.zeros((nb, len(bin_groups)), dtype=torch.int32, device=device)

    global _decay_compiled
    if compile:
        if _decay_compiled is None:
            _decay_compiled = torch.compile(_decay)
        decay = _decay_compiled
    else:
        decay = _decay
    t_dev = torch.zeros((), dtype=torch.int64, device=device)

    for t in range(t_begin, nsteps):
        v, g, free = decay(v, g, last, rfc_n, t_dev.fill_(t), v0, a_coef, em, es)  # 1-2
        if gap is not None:                                          # 2b
            vg = v[g_idx]
            dv = g_k * (torch.mv(g_C, vg) - g_deg * vg)
            v[g_idx] = vg + torch.where(free[g_idx], dv, 0.0)
        spk = (v > v_th) & free                                      # 3
        idx = torch.nonzero(spk).flatten()
        has = idx.numel() > 0
        if has:
            last[idx] = t
            free[idx] = False
        pre = ring[t % dly]                                          # 4
        if presyn is not None:
            IA *= eA; IB *= eB
            if pre.numel():
                x = ps_W.colsum(pre)
                IA += x; IB += x * rB
            gain[ps_idx] = 1.0 / (1.0 + (IA + IB) / K)
        if pre.numel():
            upd = Wd.colsum(pre, None if presyn is None else gain[pre])
            g = g + torch.where(free, upd, 0.0)
        for s0, s1, ix_np, ix, p in drive:
            if s0 <= t < s1:
                if rng == 'numpy':
                    hit = ix_np[np_rng.random(len(ix_np)) < p]
                    if not len(hit):
                        continue
                    hit = torch.as_tensor(hit, device=device)
                else:
                    hit = ix[torch.rand(len(ix), generator=t_gen, device=device) < p]
                v[hit] += torch.where(free[hit], kick, 0.0)
        if gap is not None and has:                                  # spikelets
            gs = g_pos[idx]; gs = gs[gs >= 0]
            if gs.numel():
                sl = g_Cc.colsum(gs) * g_vs
                v[g_idx] += torch.where(free[g_idx], sl, 0.0)
        if apl is not None:
            a_apl = a_apl * decay_a + (1 - decay_a) * apl['k'] * kc_rate
            g[kc_idx] -= torch.where(free[kc_idx], w_kc * (a_apl * dt), 0.0)
        if has:                                                      # 5
            v[idx] = v_rst
            g[idx] = 0.0
            counts[idx] += 1
            if record:
                rec_t.append(torch.full((idx.numel(),), t, dtype=torch.int32, device=device))
                rec_i.append(idx)
            if bin_steps:
                if bin_groups is None:
                    binned[(t - t_begin) // bin_steps].index_add_(0, idx, torch.ones_like(idx, dtype=torch.int32))
                else:
                    gi = gid[idx]; gi = gi[gi >= 0]
                    binned[(t - t_begin) // bin_steps].index_add_(0, gi, torch.ones_like(gi, dtype=torch.int32))
        ring[t % dly] = idx if has else empty
        if apl is not None:
            kc_rate = float(is_kc[idx].sum()) / dt if has else 0.0

    st = None
    if return_state:
        st = dict(t=nsteps, v=v.clone(), g=g.clone(), last=last.clone(),
                  ring=[x.clone() for x in ring], counts=counts.clone(),
                  np_rng=np_rng.bit_generator.state, t_gen=t_gen.get_state())
        if apl is not None:
            st.update(a_apl=a_apl, kc_rate=kc_rate)
        if presyn is not None:
            st.update(IA=IA.clone(), IB=IB.clone(), gain=gain.clone())
    counts = counts.cpu().numpy()
    if not record and not bin_steps and not return_state:
        return counts
    out = dict(counts=counts)
    if st is not None:
        out['state'] = st
    if record:
        out['t'] = torch.cat(rec_t).cpu().numpy() if rec_t else np.empty(0, np.int32)
        out['i'] = torch.cat(rec_i).cpu().numpy() if rec_i else np.empty(0, np.int64)
    if bin_steps:
        out['binned'] = binned.cpu().numpy()
    return out
