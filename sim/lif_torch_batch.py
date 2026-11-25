"""Week 5 (P1): B independent trials over one shared network, one step for the batch.

Same model, step order and options (gap, APL, presyn, blocks / Poisson drive,
t_stim, record, bin_steps / bin_groups) as sim/lif_torch.run_trial (the
reference, untouched) and sim/lif_torch_fast.run_trial (the sync-free single
trial, T5a/T5b). Each trial has its own seed and its own drive (blocks or
exc_idx). State is (B, n); the step (_Batch.step) has no host syncs and static
shapes, and the number of GPU kernels it launches does not depend on B except
for the RNG draws (one per drawing (trial, drive) pair, see RNG below).

How the single-trial fast step is batched:
  - v, g, last, rfc_n, counts: (B, n). last and rfc_n are int32 (step numbers
    and the refractory length fit; integer comparisons are exact either way).
    rfc_n is per trial because driven neurons have no refractory period and
    each trial drives its own neurons.
  - Spikes: one torch.nonzero_static over the flattened (B * n) spike mask with
    size B * spike_cap gives the spiking (trial, neuron) pairs in row-major
    order; each pair's rank inside its trial (position minus the number of
    spikes of earlier trials) places it in a (B, spike_cap) padded list
    (sentinel n), neurons ascending as in the single-trial path.
  - Event gather (_BatchCSC): one gather over the flattened (trial, spike) list:
    per trial, syn_cap entries are read from the shared CSC matrix (the T5a
    _PadCSC layout: column n has length 0, entry nnz is a dummy with row n and
    value 0), searchsorted is batched over the (B, spike_cap) cumulative column
    lengths, and one index_add_ scatters into a flattened (B, n + 1) output
    (row offset b * (n + 1); column n of each trial is the dropped dummy row).
    Within a trial the entries keep the single-trial order (columns in spike
    order, CSC order within a column), so on the CPU the sums are formed in the
    reference order.
  - Gap junctions: on the CPU the reference's sparse CSR torch.mv, one call per
    trial (exact parity); on CUDA a matmul of the dense matrix expanded to
    (B, m, m) with the (B, m, 1) voltages, which equalled torch.mv bitwise for
    every B tested (1-32) on the 5060 Ti (observed, not guaranteed); a plain
    (B, m) @ (m, m) matmul did not (1e-15 differences for B >= 2).
  - Drives: all (trial, drive, neuron) entries in one flat list with their
    window [s0, s1) in steps and probability p. A device mask
    (s0 <= t_dev < s1) selects the active entries, and one index_add_ adds the
    kick into v viewed as (B * n,). A neuron driven by two overlapping drives
    of one trial gets two adds of kick or 0, the same values in any order as
    the reference's two sequential adds. The kick is formed as the reference
    forms it (torch.where with Python scalars, i.e. float32, then promoted).
  - APL: a_apl and kc_rate are (B,) float64 tensors; a_apl * dt is cast to
    the run dtype before it scales w_kc, as the reference's Python float does.
  - Caps are per trial: every trial has its own spike_cap slots and syn_cap
    gathered entries (main matrix and presyn matrix). A (B,) overflow flag is
    set by a trial that exceeds a cap, and also by a trial whose spikes were cut
    from the flattened spike list because other trials overflowed (that can
    only happen when some trial overflowed). The flag is read once after the
    loop; flagged trials are rerun alone with lif_torch.run_trial (RuntimeWarning,
    FALLBACKS += 1, fallback=True), the others are kept.

RNG. Each trial has its own generators seeded with its seed, so its stream
depends only on (seed, its drive), never on B or its slot.
  - rng='numpy': np.random.default_rng(seed) per trial, drawn on the host in the
    reference order (per active drive, drive order) and uploaded once per step
    (parity mode; a host round trip per step, no CUDA graph).
  - rng='torch': one torch.Generator per trial, manual_seed(seed). With
    draws='active' (default) a trial draws torch.rand-equivalent numbers
    (Tensor.uniform_(0, 1, generator) on its slice of one buffer; equal to
    torch.rand bitwise, tested) only for its active drives, in drive order:
    exactly the reference's draws, so a batched trial gets the reference's
    Poisson input. With draws='always' every drive draws every step and the
    window mask discards inactive draws: a different stream than the reference
    (statistically equivalent only), but the set of RNG calls is the same at
    every step.
CUDA graph (run_batch(graph=True), rng='torch' on CUDA): the step is captured
with torch.cuda.graph and replayed once per step for the whole batch. Which
(trial, drive) pairs draw is the only host-side branch of the step, so one
graph is captured per distinct set of drawing pairs, lazily, the first time
it occurs (draws='always': one graph for any blocks). All trial generators are
registered with every graph (CUDAGraph.register_generator_state; PyTorch docs,
CUDA semantics > CUDA Graphs: RNG ops with several torch.Generator instances
are allowed in capture), and the graphs share one memory pool (they never run
concurrently and every value that outlives a step is in a tensor allocated
before capture). Three eager steps run on a side stream before the first
capture (the warm-up the docs ask for; they are steps of the trials).

Not supported (not needed by the batch use case): save/resume (state /
return_state), compile, MPS.
"""
import bisect
import warnings

import numpy as np
import torch

from sim import lif_torch
from sim.lif_numpy import P
from sim.lif_torch import _decay, default_device
from sim.lif_torch_fast import _PadCSC, _deg_bound

FALLBACKS = 0          # trials rerun alone with lif_torch.run_trial after an overflow
WARMUP = 3             # eager steps on a side stream before the first graph capture


class _BatchCSC:
    """Batched fixed-size event gather over one shared _PadCSC matrix."""

    def __init__(self, M, B, device):
        self.M, self.B, self.n, self.cap, self.nnz = M, B, M.n, M.cap, M.nnz
        self.k = torch.arange(self.cap, device=device).expand(B, -1).contiguous()
        self.bofs = (torch.arange(B, device=device) * (self.n + 1)).view(B, 1)
        self.out = torch.zeros(B * (self.n + 1), dtype=M.data.dtype, device=device)

    def colsum(self, cols, weights=None):
        """cols (B, c) column lists (sentinel = ncols allowed); weights (B, c)
        scales the entries of cols[b, j]. Returns ((B, n) sums, (B,) totals)."""
        M = self.M
        starts = M.indptr[cols]
        lens = M.indptr[cols + 1] - starts
        cend = torch.cumsum(lens, 1)
        total = cend[:, -1]
        first = cend - lens
        pos = torch.searchsorted(cend, self.k, right=True).clamp_(max=cols.shape[1] - 1)
        offs = torch.where(self.k < total.view(-1, 1),
                           starts.gather(1, pos) + self.k - first.gather(1, pos), self.nnz)
        vals = M.data[offs]
        if weights is not None:
            vals = vals * weights.gather(1, pos)
        out = self.out.zero_()
        out.index_add_(0, (M.indices[offs] + self.bofs).view(-1), vals.view(-1))
        return out.view(self.B, self.n + 1)[:, :self.n], total


class _Batch:
    """Preallocated (B, n) state + step(). Built by _prepare."""

    def key(self, t):
        """(trial, drive) pairs that draw at step t, in (trial, drive) order."""
        return self.seg_keys[bisect.bisect_right(self.seg_starts, t) - 1]

    def draw(self, key):
        """rng='numpy': host draws of the pairs in key, uploaded as one mask."""
        hits = np.zeros(self.n_ent, dtype=bool)
        for b, d in key:
            o, k, p = self.slices[b][d]
            hits[o:o + k] = self.np_rngs[b].random(k) < p
        return torch.as_tensor(hits, device=self.device)

    def advance(self, t):
        key = self.key(t)
        self.step(key, self.draw(key) if self.rng == 'numpy' else None)

    def step(self, key, hits=None):
        """One step of every trial. key: drawing pairs (static per graph);
        hits: rng='numpy' only, the uploaded draws. The device counter t_dev
        drives everything else and is incremented at the end."""
        s = self
        B, n, sc, t = s.B, s.n, s.spike_cap, s.t_dev
        mk = s.mark                          # optional phase hook (bench); None in normal runs
        if mk:
            mk('decay')
        t32 = t.to(torch.int32)
        v, g, free = _decay(s.v, s.g, s.last, s.rfc_n, t32, s.v0, s.a_coef, s.em, s.es)  # 1-2
        if s.gap is not None:                                                 # 2b
            if mk:
                mk('gap')
            vg = v[:, s.g_idx]
            dv = s.g_k * (s._gap_mv(vg) - s.g_deg * vg)
            v[:, s.g_idx] = vg + torch.where(free[:, s.g_idx], dv, 0.0)
        if mk:
            mk('threshold+lists')
        spk = (v > s.v_th) & free                                             # 3
        cnt = spk.sum(1)
        flat = torch.nonzero_static(spk.view(-1), size=B * sc, fill_value=B * n).view(-1)
        tb = torch.div(flat, n, rounding_mode='floor')                        # trial (B = padding)
        s.start[:B] = torch.cumsum(cnt, 0) - cnt
        rank = s.ar - s.start[tb]
        dest = torch.where((tb < B) & (rank < sc), tb * sc + rank, B * sc)
        idxb = s.idx_buf.fill_(n)
        idxb.index_put_((dest,), flat - tb * n)
        idxp = idxb[:B * sc].view(B, sc)                                      # (B, sc) padded
        s.ovf |= (cnt > sc) | (s.start[:B] + cnt > B * sc)
        s.last.copy_(torch.where(spk, t32, s.last))
        free = free & ~spk
        slot = torch.remainder(t, s.dly).view(1)                              # 4
        pre = s.ring.index_select(0, slot).view(B, sc)
        if s.presyn is not None:
            if mk:
                mk('presyn')
            s.IA.mul_(s.eA); s.IB.mul_(s.eB)
            x, tot = s.ps_W.colsum(pre)
            s.ovf |= tot > s.ps_W.cap
            s.IA.add_(x); s.IB.add_(x * s.rB)
            s.gain[:, s.ps_idx] = 1.0 / (1.0 + (s.IA + s.IB) / s.K)
            if mk:
                mk('synaptic')
            upd, tot = s.Wd.colsum(pre, s.gain.gather(1, pre))
        else:
            if mk:
                mk('synaptic')
            upd, tot = s.Wd.colsum(pre)
        s.ovf |= tot > s.Wd.cap
        g = g + torch.where(free, upd, 0.0)
        if s.n_ent:                                                           # Poisson drive
            if mk:
                mk('drive')
            if s.rng == 'numpy':
                hit = hits
            else:
                for b, d in key:
                    o, k, _ = s.slices[b][d]
                    s.u[o:o + k].uniform_(0, 1, generator=s.gens[b])
                hit = s.u < s.ent_p
            act = (s.ent_s0 <= t32) & (t32 < s.ent_s1)
            m = free.view(-1)[s.ent_idx] & act & hit
            v.view(-1).index_add_(0, s.ent_idx, torch.where(m, s.kick, 0.0).to(v.dtype))
        if s.gap is not None:                                                 # spikelets
            if mk:
                mk('spikelets')
            gp = s.g_pos[idxp]
            gs = torch.where(gp >= 0, gp, s.m_gap)
            sl, _ = s.g_Cc.colsum(gs)
            v[:, s.g_idx] = v[:, s.g_idx] + torch.where(free[:, s.g_idx], sl * s.g_vs, 0.0)
        if s.apl is not None:
            if mk:
                mk('apl')
            s.a_apl.copy_(s.a_apl * s.decay_a + s.c_apl * s.kc_rate)
            g[:, s.kc_idx] = g[:, s.kc_idx] - torch.where(
                free[:, s.kc_idx], s.w_kc * (s.a_apl.view(B, 1) * s.dt).to(s.w_kc.dtype), 0.0)
        if mk:
            mk('reset+record+ring')
        v = torch.where(spk, s.v_rst, v)                                      # 5
        g = torch.where(spk, 0.0, g)
        s.counts += spk
        if s.record:
            s.rec.index_copy_(0, (t - s.t_begin).view(1), idxp.to(torch.int32).view(1, B, sc))
        if s.bin_steps:
            row = torch.div(t - s.t_begin, s.bin_steps, rounding_mode='floor').view(1)
            if s.gid is None:
                c = spk.to(torch.int32)
            else:
                c = torch.zeros((B, s.n_groups + 1), dtype=torch.int32, device=s.device)
                c.index_add_(1, s.gid, spk.to(torch.int32))
                c = c[:, :s.n_groups]
            s.binned.index_add_(0, row, c.view(1, B, -1))
        s.ring.index_copy_(0, slot, idxp.view(1, B, sc))
        if s.apl is not None:
            s.kc_rate.copy_(spk[:, s.kc_idx].sum(1, dtype=torch.float64) / s.dt)
        s.v.copy_(v); s.g.copy_(g)
        t += 1
        if mk:
            mk('end')

    def _gap_mv(self, vg):
        if self.g_C.layout == torch.sparse_csr:            # CPU: the reference's mv per trial
            return torch.stack([torch.mv(self.g_C, vg[b]) for b in range(self.B)])
        return torch.matmul(self.g_C.expand(self.B, -1, -1), vg.unsqueeze(-1)).squeeze(-1)


def _drives(blocks, exc_idx, r_poi, t_stim, dt, nsteps):
    """The reference's drive list for one trial: [(s0, s1, idx_np, p)]."""
    out = []
    if blocks is not None:
        for b0, b1, ix, r in blocks:
            out.append((int(round(b0 / dt)), int(round(b1 / dt)), np.asarray(ix, np.int64), r * dt))
    else:
        exc = np.asarray(exc_idx, np.int64)
        if len(exc):
            stim_steps = nsteps if t_stim is None else int(round(t_stim / dt))
            out.append((0, stim_steps, exc, r_poi * dt))
    return out


def _prepare(W, seeds, blocks_list=None, exc_idx=(), r_poi=150.0, params=P, apl=None,
             t_stim=None, record=False, gap=None, presyn=None, device=None, dtype=None,
             rng='numpy', draws='active', bin_steps=None, bin_groups=None, spike_cap=None,
             syn_cap=None):
    assert rng in ('numpy', 'torch') and draws in ('active', 'always')
    s = _Batch()
    device = device or default_device()
    assert device != 'mps', 'lif_torch_batch: MPS not supported'
    dtype = torch.float64 if dtype is None else dtype
    tk = dict(device=device, dtype=dtype)
    seeds = [int(x) for x in seeds]
    s.B = B = len(seeds)
    if blocks_list is None:
        blocks_list = [None] * B
    assert len(blocks_list) == B, 'one blocks list (or None) per seed'
    n = W.shape[0]
    dt, nsteps = params['dt'], int(round(params['t_run'] / params['dt']))
    tm, ts = params['t_mbr'], params['tau']
    s.n, s.device, s.rng, s.dt, s.nsteps, s.t_begin = n, device, rng, dt, nsteps, 0
    s.dly = dly = int(round(params['t_dly'] / dt))
    rfc = int(round(params['t_rfc'] / dt))
    s.em, s.es = float(np.exp(-dt / tm)), float(np.exp(-dt / ts))
    s.a_coef = ts / (ts - tm)
    s.v0, s.v_th, s.v_rst = params['v_0'], params['v_th'], params['v_rst']
    s.kick = params['w_syn'] * params['f_poi']
    s.spike_cap = sc = min(n, 4096) if spike_cap is None else int(spike_cap)
    if syn_cap is None:
        syn_cap = _deg_bound(W, sc)
    s.Wd = _BatchCSC(_PadCSC(W, device, dtype, min(syn_cap, max(W.nnz, 1))), B, device)

    s.v = torch.full((B, n), s.v0, **tk)
    s.g = torch.zeros((B, n), **tk)
    s.last = torch.full((B, n), -10 ** 9, dtype=torch.int32, device=device)
    rfc_n = np.full((B, n), rfc, dtype=np.int32)
    # drive entries, flattened over (trial, drive, neuron)
    s.slices, e_idx, e_p, e_s0, e_s1, o = [], [], [], [], [], 0
    bounds = {0}
    for b, blocks in enumerate(blocks_list):
        sl = []
        for s0, s1, ix, p in _drives(blocks, exc_idx, r_poi, t_stim, dt, nsteps):
            rfc_n[b, ix] = 0
            sl.append((o, len(ix), p))
            e_idx.append(b * n + ix); e_p.append(np.full(len(ix), p))
            e_s0.append(np.full(len(ix), s0)); e_s1.append(np.full(len(ix), s1))
            o += len(ix)
            bounds.update((max(s0, 0), max(s1, 0)))
        s.slices.append(sl)
    s.n_ent = o
    cat = lambda xs, dt_: np.concatenate(xs).astype(dt_) if xs else np.zeros(0, dt_)  # noqa: E731
    s.ent_idx = torch.as_tensor(cat(e_idx, np.int64), device=device)
    s.ent_p = torch.as_tensor(cat(e_p, np.float64), device=device).to(torch.float32)  # rand < p in float32
    s.ent_s0 = torch.as_tensor(np.clip(cat(e_s0, np.int64), -2 ** 31, 2 ** 31 - 1).astype(np.int32), device=device)
    s.ent_s1 = torch.as_tensor(np.clip(cat(e_s1, np.int64), -2 ** 31, 2 ** 31 - 1).astype(np.int32), device=device)
    s.u = torch.ones(o, dtype=torch.float32, device=device)
    s.rfc_n = torch.as_tensor(rfc_n, device=device)
    # drawing pairs per step segment (the key is constant between window bounds)
    s.seg_starts = sorted(x for x in bounds if x < max(nsteps, 1))
    s.seg_keys = []
    for t0 in s.seg_starts:
        s.seg_keys.append(tuple((b, d) for b in range(B) for d, (s0, s1, _, _) in
                                enumerate(_drives(blocks_list[b], exc_idx, r_poi, t_stim, dt, nsteps))
                                if draws == 'always' or s0 <= t0 < s1))
    s.np_rngs = [np.random.default_rng(x) for x in seeds]
    s.gens = []
    for x in seeds:
        gen = torch.Generator(device=device); gen.manual_seed(x)
        s.gens.append(gen)

    s.presyn, s.gap, s.apl = presyn, gap, apl
    if presyn is not None:
        s.ps_idx = torch.as_tensor(np.asarray(presyn['idx'], np.int64), device=device)
        Wp = presyn['Wp']
        s.ps_W = _BatchCSC(_PadCSC(Wp, device, dtype, min(syn_cap, max(_deg_bound(Wp, sc), 1))), B, device)
        s.IA = torch.zeros((B, len(s.ps_idx)), **tk); s.IB = torch.zeros((B, len(s.ps_idx)), **tk)
        s.eA, s.eB = float(np.exp(-dt / presyn['tauA'])), float(np.exp(-dt / presyn['tauB']))
        s.rB, s.K = presyn['tauA'] / presyn['tauB'], presyn['K']
        s.gain = torch.ones((B, n + 1), **tk)    # column n: sentinel (weights padding)
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
        s.g_Cc = _BatchCSC(_PadCSC(gap['C'], device, dtype, max(gap['C'].nnz, 1)), B, device)
        s.g_pos = torch.full((n + 1,), -1, dtype=torch.int64, device=device)
        s.g_pos[s.g_idx] = torch.arange(len(s.g_idx), device=device)
        s.g_k, s.g_vs = dt / tm, gap.get('v_spike', 0.04)
    if apl is not None:
        s.kc_idx = torch.as_tensor(np.asarray(apl['kc_idx'], np.int64), device=device)
        s.w_kc = torch.as_tensor(np.asarray(apl['w_kc'], np.float64), **tk)
        s.a_apl = torch.zeros(B, dtype=torch.float64, device=device)
        s.kc_rate = torch.zeros(B, dtype=torch.float64, device=device)
        s.decay_a = float(np.exp(-dt / apl['tau']))
        s.c_apl = (1 - s.decay_a) * apl['k']   # same evaluation order as the reference

    s.ring = torch.full((dly, B, sc), n, dtype=torch.int64, device=device)
    s.idx_buf = torch.full((B * sc + 1,), n, dtype=torch.int64, device=device)
    s.ar = torch.arange(B * sc, device=device)
    s.start = torch.full((B + 1,), B * sc, dtype=torch.int64, device=device)  # entry B: padding
    s.counts = torch.zeros((B, n), dtype=torch.int32, device=device)
    s.ovf = torch.zeros(B, dtype=torch.bool, device=device)
    s.record, s.bin_steps, s.gid = record, bin_steps, None
    if record:
        s.rec = torch.full((nsteps, B, sc), n, dtype=torch.int32, device=device)
    if bin_steps:
        nb = -(-nsteps // bin_steps)
        if bin_groups is None:
            s.binned = torch.zeros((nb, B, n), dtype=torch.int32, device=device)
        else:
            s.n_groups = G = len(bin_groups)
            s.gid = torch.full((n,), G, dtype=torch.int64, device=device)   # G: no group
            for k_, ix in enumerate(bin_groups):
                s.gid[torch.as_tensor(np.asarray(ix, np.int64), device=device)] = k_
            s.binned = torch.zeros((nb, B, G), dtype=torch.int32, device=device)
    s.t_dev = torch.zeros((), dtype=torch.int64, device=device)
    s._t_host = 0                                # host copy of the step counter
    s.mark = None
    s.replays = 0
    return s


def eager_loop(s, stop=None, on_step=None):
    """Advance a prepared batch from its current step to `stop` (default nsteps)."""
    stop = s.nsteps if stop is None else stop
    for t in range(s._t_host, stop):
        if on_step is not None:
            on_step(t)
        s.advance(t)
    s._t_host = stop


def graph_loop(s, stop=None, on_step=None, graphs=None):
    """rng='torch' on CUDA: WARMUP eager steps on a side stream, then one
    captured graph per distinct set of drawing pairs, one replay per step
    (s.replays counts them). Returns the dict of graphs (keyed by drawing
    pairs); pass it back as `graphs` to continue without recapturing."""
    assert s.rng == 'torch' and s.device != 'cpu', "graph replay needs rng='torch' on CUDA"
    stop = s.nsteps if stop is None else stop
    t = s._t_host
    if graphs is None:
        graphs = {}
        cur = torch.cuda.current_stream()
        side = torch.cuda.Stream()
        side.wait_stream(cur)
        with torch.cuda.stream(side):
            for _ in range(min(WARMUP, stop - t)):
                if on_step is not None:
                    on_step(t)
                s.advance(t)
                t += 1
        cur.wait_stream(side)
    pool = next(iter(graphs.values())).pool() if graphs else None
    while t < stop:
        k = s.key(t)
        if k not in graphs:
            gr = torch.cuda.CUDAGraph()
            for gen in s.gens:
                gr.register_generator_state(gen)
            with torch.cuda.graph(gr, pool=pool):
                s.step(k)           # recorded, not executed
            pool = gr.pool() if pool is None else pool
            graphs[k] = gr
        if on_step is not None:
            on_step(t)
        graphs[k].replay()
        s.replays += 1
        t += 1
    s._t_host = stop
    return graphs


def run_batch(W, params=P, seeds=(0,), blocks_list=None, exc_idx=(), r_poi=150.0, apl=None,
              t_stim=None, record=False, gap=None, presyn=None, device=None, dtype=None,
              rng='numpy', draws='active', bin_steps=None, bin_groups=None, spike_cap=None,
              syn_cap=None, graph=None):
    """Run len(seeds) independent trials over one network W in one batch.

    seeds[b], blocks_list[b] (a blocks list as in lif_torch.run_trial, or None
    for the exc_idx / r_poi / t_stim drive) define trial b; every other option
    is shared. graph: replay a CUDA graph per step (default: True for
    rng='torch' on CUDA). Returns a list of B results in the format of
    lif_torch.run_trial (counts array, or with record / bin_steps a dict with
    counts, t, i, binned and fallback)."""
    global FALLBACKS
    s = _prepare(W, seeds, blocks_list, exc_idx=exc_idx, r_poi=r_poi, params=params, apl=apl,
                 t_stim=t_stim, record=record, gap=gap, presyn=presyn, device=device,
                 dtype=dtype, rng=rng, draws=draws, bin_steps=bin_steps, bin_groups=bin_groups,
                 spike_cap=spike_cap, syn_cap=syn_cap)
    if graph is None:
        graph = rng == 'torch' and s.device != 'cpu'
    if graph:
        graph_loop(s)
    else:
        eager_loop(s)
    ovf = s.ovf.cpu().numpy()                    # the one sync per batch
    counts = s.counts.cpu().numpy().astype(np.int64)
    rec = s.rec.cpu().numpy() if record else None
    binned = s.binned.cpu().numpy() if bin_steps else None
    n, out = s.n, []
    seeds = [int(x) for x in seeds]
    blocks_list = [None] * len(seeds) if blocks_list is None else blocks_list
    for b in range(s.B):
        if ovf[b]:
            FALLBACKS += 1
            warnings.warn(f'lif_torch_batch: trial {b} (seed {seeds[b]}) exceeded spike_cap/syn_cap '
                          f'(spike_cap={s.spike_cap}, syn_cap={s.Wd.cap}); rerunning it alone with '
                          'lif_torch.run_trial', RuntimeWarning, stacklevel=2)
            r = lif_torch.run_trial(W, [] if blocks_list[b] is not None else exc_idx, r_poi=r_poi,
                                    params=params, seed=seeds[b], apl=apl, t_stim=t_stim,
                                    record=record, gap=gap, blocks=blocks_list[b], presyn=presyn,
                                    device=s.device, dtype=dtype, rng=rng, bin_steps=bin_steps,
                                    bin_groups=bin_groups)
            if isinstance(r, dict):
                r['fallback'] = True
            out.append(r)
            continue
        if not record and not bin_steps:
            out.append(counts[b])
            continue
        r = dict(counts=counts[b], fallback=False)
        if record:
            rb = rec[:, b, :]
            tt, kk = np.nonzero(rb < n)          # row-major: step, then neuron ascending
            r['t'] = tt.astype(np.int32)
            r['i'] = rb[tt, kk].astype(np.int64)
        if bin_steps:
            r['binned'] = binned[:, b, :]
        out.append(r)
    return out
