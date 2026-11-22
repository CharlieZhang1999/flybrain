"""Week 4 (T5a): sync-free, static-shape version of sim/lif_torch.run_trial.

Same model, step order, options and return contract as sim/lif_torch.run_trial
(the reference, which stays untouched). The step loop body is one method,
_Sim.step(t, hits), over preallocated tensors and a device step counter, with
no host syncs and no data-dependent shapes, so that it can later be captured as
a CUDA graph (T5b). Python branches in step() depend only on static config
(which options are on, block windows by step number).

How each reference sync site is replaced (reference -> here):
  - torch.nonzero(spk) + idx.numel()/has: torch.nonzero_static(spk,
    size=spike_cap, fill_value=n). Spiking neurons ascending, padded with the
    sentinel n. Scatters by idx become masks: last = where(spk, t, last),
    free &= ~spk, v/g reset and counts += spk via where/add.
  - _CSC.gather's int(lens.sum()) + repeat_interleave: _PadCSC.colsum gathers a
    fixed syn_cap entries. Column n (the sentinel) has length 0 (indptr is
    extended by one repeated entry). Entry k belongs to column
    searchsorted(cumsum(lens), k, right=True); entries k >= total read a dummy
    entry (row n_rows, value 0.0) appended to indices/data, so padding adds
    exact zeros into a dummy output row that is dropped. Valid entries keep
    the reference order (columns in `cols` order, CSC order within a column).
  - pre.numel() (ring slot empty?): the ring is a (dly, spike_cap) tensor of
    padded index lists; an empty slot is all sentinels and gathers nothing.
    The presyn / synaptic updates always run and add exact zeros.
  - spikelet gs = g_pos[idx]; gs[gs >= 0]; gs.numel(): gs = where(g_pos[idx] >=
    0, g_pos[idx], m_gap) (m_gap = sentinel column of the gap matrix).
  - APL float(is_kc[idx].sum()): a_apl and kc_rate are 0-d tensors;
    kc_rate = (spk & is_kc).sum() / dt.
  - Poisson hits ix[rand < p] (bool-mask indexing): applied as a mask,
    v[ix] += where(free[ix] & (rand < p), kick, 0.0). With rng='numpy' the draw
    is still made by NumPy on the host and uploaded each step (a host round
    trip by construction; that mode is for parity only). rng='torch' is the
    sync-free mode.
  - Python `if` on tensor sizes: gone (see above).
  - Recording / binning: device buffers written at row (t_dev - t_begin), no
    Python lists; compacted once after the loop.

Caps and overflow. spike_cap bounds spikes per step, syn_cap the gathered
synapses per step (main matrix W and the presyn matrix Wp). Defaults:
  spike_cap = min(n, 4096)
  syn_cap   = sum of the spike_cap largest column degrees of W, i.e. the most
              synapses spike_cap spiking neurons can have; with the default it
              cannot overflow unless spike_cap does.
  Wp uses min(syn_cap, its own such bound); the gap matrix always uses its own
  bound (nnz of the m_gap x m_gap matrix), which cannot overflow.
T5b sets both caps from the measured looming burst. When a cap is exceeded a
device flag is set; it is read once after the loop (one sync per trial). On
overflow the trial is rerun with lif_torch.run_trial, a RuntimeWarning is
issued, FALLBACKS is incremented, and a dict result carries fallback=True
(fallback=False otherwise).

Recording: with record=True a (segment steps, spike_cap) buffer of padded
spike indices; 't'/'i' are compacted after the loop, in the reference order
(step ascending, neuron ascending). Memory is steps x spike_cap x 8 bytes.

Save/resume: same contract and the same state dict keys and formats as the
reference: t (int), v, g, last, counts (tensors), ring (list of dly int64
tensors, unpadded), np_rng, t_gen, plus a_apl, kc_rate (Python floats) with
apl and IA, IB, gain with presyn. States are interchangeable between the two
simulators. A saved ring slot longer than spike_cap counts as an overflow.
"""
import warnings

import numpy as np
import torch

from sim import lif_torch
from sim.lif_numpy import P
from sim.lif_torch import _decay, default_device

FALLBACKS = 0          # number of trials that fell back to lif_torch.run_trial


def _deg_bound(M, k):
    """Most entries k columns of M can hold: sum of the k largest column degrees."""
    deg = np.diff(M.tocsc().indptr)
    k = min(int(k), len(deg))
    return int(np.sort(deg)[::-1][:k].sum()) if k else 0


class _PadCSC:
    """CSC matrix on a device with a fixed-size event gather (no syncs)."""

    def __init__(self, M, device, dtype, cap):
        M = M.tocsc(); M.sort_indices()
        self.n = M.shape[0]
        self.nnz = int(M.nnz)
        self.cap = max(1, int(cap))
        ip = M.indptr.astype(np.int64)
        # column ncols (the sentinel) has length 0; entry nnz is a dummy
        # (row n, value 0) read by every padding slot
        self.indptr = torch.as_tensor(np.append(ip, ip[-1]), device=device)
        self.indices = torch.as_tensor(np.append(M.indices.astype(np.int64), self.n),
                                       device=device)
        self.data = torch.as_tensor(np.append(M.data, 0.0), dtype=dtype, device=device)
        self.k = torch.arange(self.cap, device=device)
        self.out = torch.zeros(self.n + 1, dtype=dtype, device=device)

    def colsum(self, cols, weights=None):
        """(column sum over `cols` (n,), total entries (0-d)). cols may hold the
        sentinel ncols. weights[j] scales the entries of cols[j]."""
        starts = self.indptr[cols]
        lens = self.indptr[cols + 1] - starts
        cend = torch.cumsum(lens, 0)
        total = cend[-1]
        first = cend - lens
        pos = torch.searchsorted(cend, self.k, right=True).clamp_(max=cols.shape[0] - 1)
        offs = torch.where(self.k < total, starts[pos] + self.k - first[pos], self.nnz)
        vals = self.data[offs]
        if weights is not None:
            vals = vals * weights[pos]
        out = self.out.zero_()
        out.index_add_(0, self.indices[offs], vals)
        return out[:self.n].clone(), total


class _Sim:
    """Preallocated state + step(). Built by _prepare; used by run_trial."""

    def step(self, t, hits=None):
        """One step. t: Python step number, used only for static branches
        (block windows). hits: rng='numpy' only, the uploaded Poisson masks of
        the drives active at t (in drive order). The device counter t_dev
        (== t) drives everything else and is incremented at the end."""
        s = self
        n, t_dev = s.n, s.t_dev
        v, g, free = s.decay(s.v, s.g, s.last, s.rfc_n, t_dev, s.v0, s.a_coef, s.em, s.es)
        if s.gap is not None:                                        # 2b
            vg = v[s.g_idx]
            dv = s.g_k * (torch.mv(s.g_C, vg) - s.g_deg * vg)
            v[s.g_idx] = vg + torch.where(free[s.g_idx], dv, 0.0)
        spk = (v > s.v_th) & free                                    # 3
        idxp = torch.nonzero_static(spk, size=s.spike_cap, fill_value=n).view(-1)
        s.ovf |= spk.sum() > s.spike_cap
        s.last.copy_(torch.where(spk, t_dev, s.last))
        free = free & ~spk
        slot = torch.remainder(t_dev, s.dly).view(1)                 # 4
        pre = s.ring.index_select(0, slot).view(-1)
        if s.presyn is not None:
            s.IA.mul_(s.eA); s.IB.mul_(s.eB)
            x, tot = s.ps_W.colsum(pre)
            s.ovf |= tot > s.ps_W.cap
            s.IA.add_(x); s.IB.add_(x * s.rB)
            s.gain[s.ps_idx] = 1.0 / (1.0 + (s.IA + s.IB) / s.K)
            upd, tot = s.Wd.colsum(pre, s.gain[pre])
        else:
            upd, tot = s.Wd.colsum(pre)
        s.ovf |= tot > s.Wd.cap
        g = g + torch.where(free, upd, 0.0)
        k = 0
        for s0, s1, ix, p in s.drive:
            if s0 <= t < s1:
                if s.rng == 'numpy':
                    m = hits[k]; k += 1
                else:
                    m = torch.rand(ix.shape[0], generator=s.t_gen, device=s.device) < p
                v[ix] = v[ix] + torch.where(free[ix] & m, s.kick, 0.0)
        if s.gap is not None:                                        # spikelets
            gp = s.g_pos[idxp]
            gs = torch.where(gp >= 0, gp, s.m_gap)
            sl, _ = s.g_Cc.colsum(gs)
            v[s.g_idx] = v[s.g_idx] + torch.where(free[s.g_idx], sl * s.g_vs, 0.0)
        if s.apl is not None:
            s.a_apl.copy_(s.a_apl * s.decay_a + s.c_apl * s.kc_rate)
            g[s.kc_idx] = g[s.kc_idx] - torch.where(free[s.kc_idx], s.w_kc * (s.a_apl * s.dt), 0.0)
        v = torch.where(spk, s.v_rst, v)                             # 5
        g = torch.where(spk, 0.0, g)
        s.counts += spk
        if s.record:
            s.rec.index_copy_(0, (t_dev - s.t_begin).view(1), idxp.view(1, -1))
        if s.bin_steps:
            row = torch.div(t_dev - s.t_begin, s.bin_steps, rounding_mode='floor').view(1)
            if s.gid is None:
                c = spk.to(torch.int32)
            else:
                c = torch.zeros(s.n_groups + 1, dtype=torch.int32, device=s.device)
                c.index_add_(0, s.gid, spk.to(torch.int32))
                c = c[:s.n_groups]
            s.binned.index_add_(0, row, c.view(1, -1))
        s.ring.index_copy_(0, slot, idxp.view(1, -1))
        if s.apl is not None:
            s.kc_rate.copy_((spk & s.is_kc).sum(dtype=torch.float64) / s.dt)
        s.v.copy_(v); s.g.copy_(g)
        t_dev += 1

    def draw(self, t):
        """rng='numpy': host Poisson draws for step t, uploaded (not sync-free)."""
        hits = []
        for (s0, s1, _, p), ix_np in zip(self.drive, self.drive_np):
            if s0 <= t < s1:
                hits.append(torch.as_tensor(self.np_rng.random(len(ix_np)) < p,
                                            device=self.device))
        return hits

    def advance(self, t):
        self.step(t, self.draw(t) if self.rng == 'numpy' else None)


def _prepare(W, exc_idx, r_poi=150.0, params=P, seed=0, apl=None, t_stim=None,
             record=False, gap=None, blocks=None, presyn=None, device=None,
             dtype=None, rng='numpy', bin_steps=None, bin_groups=None, compile=False,
             state=None, spike_cap=None, syn_cap=None):
    s = _Sim()
    device = device or default_device()
    if dtype is None:
        dtype = torch.float32 if device == 'mps' else torch.float64
    sdt = torch.float32 if device == 'mps' else torch.float64     # APL scalars
    tk = dict(device=device, dtype=dtype)
    n = W.shape[0]
    dt, nsteps = params['dt'], int(round(params['t_run'] / params['dt']))
    tm, ts = params['t_mbr'], params['tau']
    s.n, s.device, s.rng, s.dt, s.nsteps = n, device, rng, dt, nsteps
    s.dly = dly = int(round(params['t_dly'] / dt))
    rfc = int(round(params['t_rfc'] / dt))
    s.em, s.es = float(np.exp(-dt / tm)), float(np.exp(-dt / ts))
    s.a_coef = ts / (ts - tm)
    s.v0, s.v_th, s.v_rst = params['v_0'], params['v_th'], params['v_rst']
    s.kick = params['w_syn'] * params['f_poi']
    s.spike_cap = sc = min(n, 4096) if spike_cap is None else int(spike_cap)
    if syn_cap is None:
        syn_cap = _deg_bound(W, sc)

    s.Wd = _PadCSC(W, device, dtype, min(syn_cap, max(W.nnz, 1)))
    s.v = torch.full((n,), s.v0, **tk)
    s.g = torch.zeros(n, **tk)
    s.last = torch.full((n,), -10 ** 9, dtype=torch.int64, device=device)
    s.rfc_n = torch.full((n,), rfc, dtype=torch.int64, device=device)
    exc_np = np.asarray(exc_idx, dtype=np.int64)
    stim_steps = nsteps if t_stim is None else int(round(t_stim / dt))
    s.drive, s.drive_np = [], []                 # (s0, s1, idx_dev, p), idx_np
    if blocks is not None:
        for b0, b1, ix, r in blocks:
            ix = np.asarray(ix, dtype=np.int64)
            s.drive.append((int(round(b0 / dt)), int(round(b1 / dt)),
                            torch.as_tensor(ix, device=device), r * dt))
            s.drive_np.append(ix)
    elif len(exc_np):
        s.drive.append((0, stim_steps, torch.as_tensor(exc_np, device=device), r_poi * dt))
        s.drive_np.append(exc_np)
    for _, _, ix, _ in s.drive:
        s.rfc_n[ix] = 0
    s.np_rng = np.random.default_rng(seed)
    s.t_gen = torch.Generator(device=device); s.t_gen.manual_seed(int(seed))

    s.presyn, s.gap, s.apl = presyn, gap, apl
    if presyn is not None:
        s.ps_idx = torch.as_tensor(np.asarray(presyn['idx'], np.int64), device=device)
        Wp = presyn['Wp']
        s.ps_W = _PadCSC(Wp, device, dtype, min(syn_cap, max(_deg_bound(Wp, sc), 1)))
        s.IA = torch.zeros(len(s.ps_idx), **tk); s.IB = torch.zeros(len(s.ps_idx), **tk)
        s.eA, s.eB = float(np.exp(-dt / presyn['tauA'])), float(np.exp(-dt / presyn['tauB']))
        s.rB, s.K = presyn['tauA'] / presyn['tauB'], presyn['K']
        s.gain = torch.ones(n + 1, **tk)         # entry n: sentinel (weights padding)
    if gap is not None:
        s.g_idx = torch.as_tensor(np.asarray(gap['idx'], np.int64), device=device)
        C = gap['C'].tocsr()
        if device == 'cpu':   # same construction as the reference (exact parity)
            s.g_C = torch.sparse_csr_tensor(torch.as_tensor(C.indptr.astype(np.int64)),
                                            torch.as_tensor(C.indices.astype(np.int64)),
                                            torch.as_tensor(C.data), size=C.shape).to(**tk)
        else:
            s.g_C = torch.as_tensor(C.toarray(), **tk)
        s.g_deg = torch.as_tensor(np.asarray(C.sum(axis=1)).ravel(), **tk)
        s.m_gap = len(s.g_idx)
        s.g_Cc = _PadCSC(gap['C'], device, dtype, max(gap['C'].nnz, 1))
        s.g_pos = torch.full((n + 1,), -1, dtype=torch.int64, device=device)
        s.g_pos[s.g_idx] = torch.arange(len(s.g_idx), device=device)
        s.g_k, s.g_vs = dt / tm, gap.get('v_spike', 0.04)
    if apl is not None:
        s.kc_idx = torch.as_tensor(np.asarray(apl['kc_idx'], np.int64), device=device)
        s.is_kc = torch.zeros(n, dtype=torch.bool, device=device); s.is_kc[s.kc_idx] = True
        s.w_kc = torch.as_tensor(np.asarray(apl['w_kc'], np.float64), **tk)
        s.a_apl = torch.zeros((), dtype=sdt, device=device)
        s.kc_rate = torch.zeros((), dtype=sdt, device=device)
        s.decay_a = float(np.exp(-dt / apl['tau']))
        s.c_apl = (1 - s.decay_a) * apl['k']   # same evaluation order as the reference

    s.ring = torch.full((dly, sc), n, dtype=torch.int64, device=device)
    s.counts = torch.zeros(n, dtype=torch.int64, device=device)
    s.ovf = torch.zeros((), dtype=torch.bool, device=device)
    s.t_begin = 0
    if state is not None:                                    # resume
        s.t_begin = int(state['t'])
        _t = lambda x, dt_: torch.as_tensor(x, device=device, dtype=dt_).clone()
        s.v = _t(state['v'], dtype); s.g = _t(state['g'], dtype)
        s.last = _t(state['last'], torch.int64)
        for k, x in enumerate(state['ring']):
            x = _t(x, torch.int64)
            if x.shape[0] > sc:
                s.ovf.fill_(True); x = x[:sc]
            s.ring[k, :x.shape[0]] = x
        s.counts = _t(state['counts'], torch.int64)
        if state.get('np_rng') is not None:
            s.np_rng.bit_generator.state = state['np_rng']
        if state.get('t_gen') is not None:
            s.t_gen.set_state(state['t_gen'])
        if apl is not None:
            s.a_apl.fill_(float(state['a_apl'])); s.kc_rate.fill_(float(state['kc_rate']))
        if presyn is not None:
            s.IA = _t(state['IA'], dtype); s.IB = _t(state['IB'], dtype)
            s.gain[:n] = _t(state['gain'], dtype)
    s.record, s.bin_steps, s.gid = record, bin_steps, None
    nseg = max(nsteps - s.t_begin, 0)
    if record:
        s.rec = torch.full((nseg, sc), n, dtype=torch.int64, device=device)
    if bin_steps:
        nb = -(-nseg // bin_steps)
        if bin_groups is None:
            s.binned = torch.zeros((nb, n), dtype=torch.int32, device=device)
        else:
            s.n_groups = G = len(bin_groups)
            s.gid = torch.full((n,), G, dtype=torch.int64, device=device)   # G: no group
            for k_, ix in enumerate(bin_groups):
                s.gid[torch.as_tensor(np.asarray(ix, np.int64), device=device)] = k_
            s.binned = torch.zeros((nb, G), dtype=torch.int32, device=device)

    if compile:
        if lif_torch._decay_compiled is None:
            lif_torch._decay_compiled = torch.compile(_decay)
        s.decay = lif_torch._decay_compiled
    else:
        s.decay = _decay
    s.t_dev = torch.full((), s.t_begin, dtype=torch.int64, device=device)
    return s


def run_trial(W, exc_idx, r_poi=150.0, params=P, seed=0, apl=None, t_stim=None,
              record=False, gap=None, blocks=None, presyn=None, device=None,
              dtype=None, rng='numpy', bin_steps=None, bin_groups=None, compile=False,
              state=None, return_state=False, spike_cap=None, syn_cap=None):
    """Same arguments and result as lif_torch.run_trial, plus spike_cap /
    syn_cap (see module docstring). Dict results also carry 'fallback'."""
    global FALLBACKS
    ref_kw = dict(r_poi=r_poi, params=params, seed=seed, apl=apl, t_stim=t_stim,
                  record=record, gap=gap, blocks=blocks, presyn=presyn, device=device,
                  dtype=dtype, rng=rng, bin_steps=bin_steps, bin_groups=bin_groups,
                  compile=compile, state=state, return_state=return_state)
    s = _prepare(W, exc_idx, r_poi=r_poi, params=params, seed=seed, apl=apl,
                 t_stim=t_stim, record=record, gap=gap, blocks=blocks, presyn=presyn,
                 device=device, dtype=dtype, rng=rng, bin_steps=bin_steps,
                 bin_groups=bin_groups, compile=compile, state=state,
                 spike_cap=spike_cap, syn_cap=syn_cap)
    for t in range(s.t_begin, s.nsteps):
        s.advance(t)

    if bool(s.ovf):                              # the one sync per trial
        FALLBACKS += 1
        warnings.warn('lif_torch_fast: spike_cap/syn_cap exceeded '
                      f'(spike_cap={s.spike_cap}, syn_cap={s.Wd.cap}); '
                      'rerunning with lif_torch.run_trial', RuntimeWarning, stacklevel=2)
        out = lif_torch.run_trial(W, exc_idx, **ref_kw)
        if isinstance(out, dict):
            out['fallback'] = True
        return out

    n = s.n
    st = None
    if return_state:
        ring = [x[x < n].clone() for x in s.ring]
        st = dict(t=s.nsteps, v=s.v.clone(), g=s.g.clone(), last=s.last.clone(),
                  ring=ring, counts=s.counts.clone(),
                  np_rng=s.np_rng.bit_generator.state, t_gen=s.t_gen.get_state())
        if apl is not None:
            st.update(a_apl=float(s.a_apl), kc_rate=float(s.kc_rate))
        if presyn is not None:
            st.update(IA=s.IA.clone(), IB=s.IB.clone(), gain=s.gain[:n].clone())
    counts = s.counts.cpu().numpy()
    if not record and not bin_steps and not return_state:
        return counts
    out = dict(counts=counts, fallback=False)
    if st is not None:
        out['state'] = st
    if record:
        rec = s.rec.cpu().numpy()
        tt, kk = np.nonzero(rec < n)             # row-major: step, then neuron ascending
        out['t'] = (tt + s.t_begin).astype(np.int32)
        out['i'] = rec[tt, kk].astype(np.int64)
    if bin_steps:
        out['binned'] = s.binned.cpu().numpy()
    return out
