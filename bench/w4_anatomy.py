#!/usr/bin/env python
"""Week 4 T3: anatomy of one steady GPU step of sim/lif_torch.py, phase by phase.

An instrumented copy of the lif_torch.run_trial step loop (bundle configuration
only: gap + APL + presyn, one Poisson drive, no blocks / record / bins / compile)
split into phases. The statements, their order and the sim helpers they call
(lif_torch._decay, lif_torch._CSC) are the same as main's sim/lif_torch.py
lines 201-256; only boundary markers are added between them.

Phases (segment -> source lines of sim/lif_torch.py on main):
  decay              202        _decay (refractory mask, v/g update)
  gap                203-206    gap junction current
  threshold+nonzero  207-209    spk, nonzero, has
  bookkeeping        210-212    last[idx] = t, free[idx] = False
  presyn             213-219    pre = ring[...], presyn IA/IB, gain
  synaptic           220-222    Wd.colsum(pre, gain[pre]) (_CSC.gather + index_add_), g update
  drive              223-232    Poisson drive
  spikelets          233-237    gap spikelets
  apl                238-240    APL decay and inhibition
                     255-256    + kc_rate = float(is_kc[idx].sum()) (end of step, same phase)
  reset              241-244    v/g reset, counts
  ring               254        ring[t % dly] = idx

Passes per run (each from the same warm state, so every pass simulates the same spikes):
  reference   lif_torch.run_trial itself, steps timed by wrapping lif_torch._decay (T1 method)
  host        instrumented copy, host perf_counter at every boundary only
              (the phase wall table comes from this pass: it adds ~1 us per boundary)
  full        host perf_counter + torch.cuda.Event.record + torch.cuda.nvtx range per phase
              (GPU span = event to event; the 13 event records per step add ~0.2-0.4 ms
              per step on WSL, so its walls are reported but not used for the table).
              GPU span tiles the stream timeline, so span/wall is ~1 by construction:
              while the host paces the GPU, a span holds the GPU's idle gaps too.
  gated       GPU execution time per aten op (--gated-steps, default 50 steps): before
              every aten op that touches the GPU, torch.cuda._sleep(--gate-ms, 1 ms) is
              enqueued, then an event; a second event follows the op. The host enqueues the
              op while the GPU sleeps, so the event span is the GPU time of the op's
              kernels, not paced by the host. Sync-forcing ops (nonzero,
              _local_scalar_dense, scalar index_put_, bool-mask index, H2D _to_copy) are
              summed separately: their span also holds the copy round trip and the host
              returning from the sync, so it is an upper bound.
              GPU busy per step is between "non-sync exec" and "non-sync exec + sync spans".
  counts      TorchDispatchMode (aten ops that launch GPU work, T1's classification) and
              torch.cuda.set_sync_debug_mode('warn') warnings, attributed to the phase
  util        nvidia-smi --query-gpu=timestamp,utilization.gpu,clocks.gr,pstate -lms 100 runs
              in the background for the whole run; the utilisation number uses the samples
              inside a window where lif_torch.run_trial steps for >= 2 s (setup excluded);
              clocks are also reported per pass

Drift guard (fails loudly, exit code 3): 300 steps from t = 0, seed 0, rng='numpy',
float64: the instrumented loop (full instrumentation) must give exactly the same
spike counts as lif_torch.run_trial. In addition, every instrumented repeat of the
warm window is compared with run_trial's counts for the same window.

Window: --warm-steps (1000) steps first (return_state=True), then --steps (200) +1
steps resumed from that state; steps 1..steps-1 are measured (same as T1). A step
slower than 5 ms is a stall; phase statistics use steady steps only.

Usage (repo root):
  python bench/w4_anatomy.py --bundle ~/flybench/flybrain/results/w4/cns_bundle_calib.npz [--runs 5]
Output: perf/results/w4/<host>_<timestamp>_anatomy.json and .md
  --nsys-window: run only the NVTX-instrumented window (for nsys profile), no output files:
  NSYS_NVTX_PROFILER_REGISTER_ONLY=0 nsys profile -t nvtx,osrt -s none --capture-range=nvtx       --nvtx-capture=window -o <out> python bench/w4_anatomy.py --nsys-window --bundle ...
"""
import argparse
import collections
import datetime
import json
import socket
import statistics
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'bench'))
from sim import lif_torch  # noqa: E402
from sim.lif_numpy import P  # noqa: E402
import w4_overhead as t1  # noqa: E402  (T1: bundle loader, StepClock, stats, op classes)

pc = time.perf_counter
STALL_MS = t1.STALL_MS
SEGS = ['decay', 'gap', 'threshold+nonzero', 'bookkeeping', 'presyn', 'synaptic', 'drive',
        'spikelets', 'apl', 'reset', 'ring', 'apl_rate']
PHASES = SEGS[:-1]
SEG_PHASE = {s: ('apl' if s == 'apl_rate' else s) for s in SEGS}
NB = len(SEGS) + 1          # boundaries per step


# ---------------------------------------------------------------- instrumented loop
class Inst:
    """What the loop records. mode: 'host' | 'full' | 'gated' | 'count' | 'nvtx'."""

    def __init__(self, mode, nsteps):
        self.mode = mode
        self.host = mode in ('host', 'full')
        self.events = mode == 'full'
        self.nvtx = mode in ('full', 'nvtx')
        self.H = np.zeros((nsteps, NB)) if self.host else None
        if self.events:
            self.E = [[torch.cuda.Event(enable_timing=True) for _ in range(NB)] for _ in range(nsteps)]
        self.track = mode in ('count', 'gated')
        self.cur = None          # current segment (count / gated modes)
        self.j = -1              # current step of the window (count / gated modes)


def instrumented_trial(model, stim, nsteps_total, device, rng, inst, seed=0, state=None):
    """Copy of lif_torch.run_trial for the bundle configuration, with phase markers.
    Returns the final counts as numpy (cumulative, like run_trial)."""
    W, exc_idx, apl, gap, presyn = model['W'], model['stim'][stim], model['apl'], model['gap'], model['presyn']
    params = dict(P); params['t_run'] = nsteps_total * P['dt']
    dtype = torch.float64
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

    Wd = lif_torch._CSC(W, device, dtype)
    v = torch.full((n,), v0, **tk)
    g = torch.zeros(n, **tk)
    last = torch.full((n,), -10 ** 9, dtype=torch.int64, device=device)
    rfc_n = torch.full((n,), rfc, dtype=torch.int64, device=device)
    exc_np = np.asarray(exc_idx, dtype=np.int64)
    stim_steps = nsteps
    drive = [(0, stim_steps, exc_np, torch.as_tensor(exc_np, device=device), 150.0 * dt)]
    for _, _, ix_np, ix, _ in drive:
        rfc_n[ix] = 0
    np_rng = np.random.default_rng(seed)
    t_gen = torch.Generator(device=device); t_gen.manual_seed(int(seed))

    ps_idx = torch.as_tensor(np.asarray(presyn['idx'], np.int64), device=device)
    ps_W = lif_torch._CSC(presyn['Wp'], device, dtype)
    IA = torch.zeros(len(ps_idx), **tk); IB = torch.zeros(len(ps_idx), **tk)
    eA, eB = float(np.exp(-dt / presyn['tauA'])), float(np.exp(-dt / presyn['tauB']))
    rB, K = presyn['tauA'] / presyn['tauB'], presyn['K']
    gain = torch.ones(n, **tk)
    g_idx = torch.as_tensor(np.asarray(gap['idx'], np.int64), device=device)
    C = gap['C'].tocsr()
    g_C = torch.as_tensor(C.toarray(), **tk)      # device != 'cpu' branch of run_trial
    g_deg = torch.as_tensor(np.asarray(C.sum(axis=1)).ravel(), **tk)
    g_Cc = lif_torch._CSC(gap['C'], device, dtype)
    g_pos = torch.full((n,), -1, dtype=torch.int64, device=device)
    g_pos[g_idx] = torch.arange(len(g_idx), device=device)
    g_k, g_vs = dt / tm, gap.get('v_spike', 0.04)
    kc_idx = torch.as_tensor(np.asarray(apl['kc_idx'], np.int64), device=device)
    is_kc = torch.zeros(n, dtype=torch.bool, device=device); is_kc[kc_idx] = True
    w_kc = torch.as_tensor(np.asarray(apl['w_kc'], np.float64), **tk)
    a_apl, kc_rate = 0.0, 0.0
    decay_a = float(np.exp(-dt / apl['tau']))

    empty = torch.empty(0, dtype=torch.int64, device=device)
    ring = [empty] * dly
    counts = torch.zeros(n, dtype=torch.int64, device=device)
    t_begin = 0
    if state is not None:                                    # resume (as run_trial)
        t_begin = int(state['t'])
        _t = lambda x, dt_: torch.as_tensor(x, device=device, dtype=dt_).clone()  # noqa: E731
        v = _t(state['v'], dtype); g = _t(state['g'], dtype)
        last = _t(state['last'], torch.int64)
        ring = [_t(x, torch.int64) for x in state['ring']]
        counts = _t(state['counts'], torch.int64)
        if state.get('np_rng') is not None:
            np_rng.bit_generator.state = state['np_rng']
        if state.get('t_gen') is not None:
            t_gen.set_state(state['t_gen'])
        a_apl, kc_rate = state['a_apl'], state['kc_rate']
        IA = _t(state['IA'], dtype); IB = _t(state['IB'], dtype)
        gain = _t(state['gain'], dtype)
    decay = lif_torch._decay
    t_dev = torch.zeros((), dtype=torch.int64, device=device)

    # ---- boundary marker: k = segment that starts (NB-1 = end of step)
    H, nv, ev, track = inst.H, inst.nvtx, inst.events, inst.track

    def B(j, k):
        if nv and k > 0:
            torch.cuda.nvtx.range_pop()
        if H is not None:
            H[j, k] = pc()
        if ev:
            inst.E[j][k].record()
        if nv:
            if k == 0:
                if j > 0:
                    torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push(f'step {j}')
            if k < NB - 1:
                torch.cuda.nvtx.range_push(SEGS[k])
        if track:
            inst.cur = SEGS[k] if k < NB - 1 else None
            inst.j = j

    for t in range(t_begin, nsteps):
        j = t - t_begin
        B(j, 0)
        v, g, free = decay(v, g, last, rfc_n, t_dev.fill_(t), v0, a_coef, em, es)  # 1-2
        B(j, 1)
        if gap is not None:                                          # 2b
            vg = v[g_idx]
            dv = g_k * (torch.mv(g_C, vg) - g_deg * vg)
            v[g_idx] = vg + torch.where(free[g_idx], dv, 0.0)
        B(j, 2)
        spk = (v > v_th) & free                                      # 3
        idx = torch.nonzero(spk).flatten()
        has = idx.numel() > 0
        B(j, 3)
        if has:
            last[idx] = t
            free[idx] = False
        B(j, 4)
        pre = ring[t % dly]                                          # 4
        if presyn is not None:
            IA *= eA; IB *= eB
            if pre.numel():
                x = ps_W.colsum(pre)
                IA += x; IB += x * rB
            gain[ps_idx] = 1.0 / (1.0 + (IA + IB) / K)
        B(j, 5)
        if pre.numel():
            upd = Wd.colsum(pre, None if presyn is None else gain[pre])
            g = g + torch.where(free, upd, 0.0)
        B(j, 6)
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
        B(j, 7)
        if gap is not None and has:                                  # spikelets
            gs = g_pos[idx]; gs = gs[gs >= 0]
            if gs.numel():
                sl = g_Cc.colsum(gs) * g_vs
                v[g_idx] += torch.where(free[g_idx], sl, 0.0)
        B(j, 8)
        if apl is not None:
            a_apl = a_apl * decay_a + (1 - decay_a) * apl['k'] * kc_rate
            g[kc_idx] -= torch.where(free[kc_idx], w_kc * (a_apl * dt), 0.0)
        B(j, 9)
        if has:                                                      # 5
            v[idx] = v_rst
            g[idx] = 0.0
            counts[idx] += 1
        B(j, 10)
        ring[t % dly] = idx if has else empty
        B(j, 11)
        if apl is not None:
            kc_rate = float(is_kc[idx].sum()) / dt if has else 0.0
        B(j, 12)
    if nv:
        torch.cuda.nvtx.range_pop()
    return counts.cpu().numpy()


# ---------------------------------------------------------------- passes
def run_ref(model, stim, nsteps_total, device, rng, state=None, seed=0, **kw):
    p = dict(P); p['t_run'] = nsteps_total * P['dt']
    return lif_torch.run_trial(model['W'], model['stim'][stim], params=p, seed=seed,
                               apl=model['apl'], gap=model['gap'], presyn=model['presyn'],
                               device=device, dtype=torch.float64, rng=rng, compile=False,
                               state=state, **kw)


class DriftError(RuntimeError):
    pass


def drift_guard(model, stim, device, steps=300):
    """Exact spike counts over `steps` steps from t = 0 (seed 0, rng='numpy', float64)."""
    ref = run_ref(model, stim, steps, device, 'numpy')
    inst = Inst('full', steps)
    got = instrumented_trial(model, stim, steps, device, 'numpy', inst)
    torch.cuda.synchronize()
    nd = int((ref != got).sum())
    out = dict(steps=steps, spikes_ref=int(ref.sum()), spikes_instrumented=int(got.sum()),
               neurons_differing=nd, passed=nd == 0)
    if nd:
        raise DriftError(f'DRIFT GUARD FAILED: {nd} neurons differ after {steps} steps '
                         f'(ref {int(ref.sum())} spikes, instrumented {int(got.sum())})')
    return out


def _seg_window(steps):
    return range(1, steps)          # steps 1..steps-1 (T1 window); step wall needs j+1


def time_reference(model, stim, device, rng, warm, steps, reps):
    walls = []
    for _ in range(reps):
        torch.cuda.synchronize()
        with t1.StepClock() as clk:
            run_ref(model, stim, warm['t'] + steps + 1, device, rng, state=warm)
        s = clk.stamps
        walls += [s[i + 1] - s[i] for i in _seg_window(steps)]
    return step_summary(walls)


def step_summary(walls_s):
    thr = STALL_MS / 1e3
    steady = [x for x in walls_s if x <= thr]
    stalls = [x for x in walls_s if x > thr]
    return dict(n_steps=len(walls_s), steady_median_ms=round(statistics.median(steady) * 1e3, 4),
                steady_mean_ms=round(statistics.fmean(steady) * 1e3, 4),
                steady_p90_ms=round(sorted(steady)[int(0.9 * (len(steady) - 1))] * 1e3, 4),
                all_mean_ms=round(statistics.fmean(walls_s) * 1e3, 4),
                stall_steps=len(stalls), stall_fraction=round(len(stalls) / len(walls_s), 4),
                stall_median_ms=round(statistics.median(stalls) * 1e3, 3) if stalls else None)


def time_instrumented(model, stim, device, rng, warm, steps, reps, mode, ref_counts):
    """host / full passes. Returns step summary and per-phase samples (steady steps)."""
    walls, ph_wall, ph_gpu, gpu_sum = [], collections.defaultdict(list), collections.defaultdict(list), []
    drift_ok = True
    for _ in range(reps):
        inst = Inst(mode, steps + 1)
        torch.cuda.synchronize()
        got = instrumented_trial(model, stim, warm['t'] + steps + 1, device, rng, inst, state=warm)
        torch.cuda.synchronize()
        drift_ok &= bool(np.array_equal(got, ref_counts))
        H = inst.H
        for j in _seg_window(steps):
            w = H[j + 1, 0] - H[j, 0]
            walls.append(w)
            if w > STALL_MS / 1e3:
                continue
            seg_w = np.diff(H[j])                 # NB-1 segments
            acc = collections.Counter()
            for k, s in enumerate(SEGS):
                acc[SEG_PHASE[s]] += seg_w[k]
            # time from the step's last boundary to the next step's start (loop overhead)
            acc['(loop)'] = H[j + 1, 0] - H[j, NB - 1]
            for ph, x in acc.items():
                ph_wall[ph].append(x)
            if mode == 'full':
                E = inst.E[j]
                accg = collections.Counter()
                for k, s in enumerate(SEGS):
                    accg[SEG_PHASE[s]] += E[k].elapsed_time(E[k + 1]) / 1e3
                accg['(loop)'] = E[NB - 1].elapsed_time(inst.E[j + 1][0]) / 1e3
                for ph, x in accg.items():
                    ph_gpu[ph].append(x)
                gpu_sum.append((E[0].elapsed_time(inst.E[j + 1][0]) / 1e3, w))
    out = dict(step=step_summary(walls), drift_ok=drift_ok,
               phase_wall_ms={ph: t1.stats(xs, 1e3) for ph, xs in ph_wall.items()})
    if mode == 'full':
        out['phase_gpu_span_ms'] = {ph: t1.stats(xs, 1e3) for ph, xs in ph_gpu.items()}
        r = [g_ / w for g_, w in gpu_sum]
        out['gpu_span_over_wall_per_step'] = dict(median=round(statistics.median(r), 4),
                                                 min=round(min(r), 4), max=round(max(r), 4))
    return out


def calibrate_sleep(ms):
    """cycles for torch.cuda._sleep to last ~ms on the GPU (measured with events)."""
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    cyc = 1_000_000
    for _ in range(4):
        torch.cuda.synchronize(); a.record(); torch.cuda._sleep(cyc); b.record(); torch.cuda.synchronize()
        cyc = max(1000, int(cyc * ms / a.elapsed_time(b)))
    torch.cuda.synchronize(); a.record(); torch.cuda._sleep(cyc); b.record(); torch.cuda.synchronize()
    return cyc, a.elapsed_time(b)


def _is_sync_op(base, args, kwargs):
    """The sync-forcing aten ops found in T1 (and checked per phase against the
    sync-debug counts of count_pass)."""
    if base in ('nonzero', '_local_scalar_dense'):
        return True
    if base == 'index_put_':
        return isinstance(args[2], torch.Tensor) and args[2].device.type == 'cpu'
    if base == 'index':
        return any(isinstance(i, torch.Tensor) and i.dtype == torch.bool for i in args[1])
    if base == '_to_copy':
        return args[0].device.type == 'cpu' and str(kwargs.get('device', '')).startswith('cuda')
    if base == 'copy_':
        return args[1].device.type == 'cpu' and args[0].device.type == 'cuda'
    return False


def time_gated(model, stim, device, rng, warm, steps, ref_counts, cyc, gate_ms):
    """GPU execution time per aten op: before every aten op that touches the GPU,
    torch.cuda._sleep(gate) is enqueued and an event recorded after it; a second
    event follows the op. While the GPU sleeps the host enqueues the op, so the
    event span is the GPU time of the op's kernels, not paced by the host. For the
    sync-forcing ops the span also contains the D2H/H2D round trip and the host
    returning from the sync, so they are reported separately (upper bound)."""
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten
    inst = Inst('gated', steps + 1)
    recs, host_dt = [], []
    E = lambda: torch.cuda.Event(enable_timing=True)  # noqa: E731

    class Gate(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if not (1 <= inst.j <= steps - 1) or inst.cur is None:
                return func(*args, **kwargs)
            base = str(func).split('.')[1] if str(func).startswith('aten.') else str(func)
            flat, _ = tree_flatten((args, kwargs))
            on_gpu = any(isinstance(x, torch.Tensor) and x.device.type == 'cuda' for x in flat) or                 str(kwargs.get('device', '')).startswith('cuda')
            if base in t1.VIEWLIKE or not on_gpu:
                return func(*args, **kwargs)
            sync_op = _is_sync_op(base, args, kwargs)
            h0 = pc()
            torch.cuda._sleep(cyc)
            s = E(); s.record()
            out = func(*args, **kwargs)
            h1 = pc()
            e = E(); e.record()
            recs.append((inst.j, SEG_PHASE[inst.cur], base, sync_op, s, e))
            if not sync_op:
                host_dt.append(h1 - h0)
            return out

    torch.cuda.synchronize()
    with Gate():
        got = instrumented_trial(model, stim, warm['t'] + steps + 1, device, rng, inst, state=warm)
    torch.cuda.synchronize()
    nw = steps - 1
    ex, sy = collections.defaultdict(list), collections.defaultdict(list)
    per_step = collections.defaultdict(lambda: [0.0, 0.0])
    n_sync = collections.Counter()
    by_op = collections.Counter()
    for j, ph, base, is_sync, s, e in recs:
        x = s.elapsed_time(e) / 1e3
        (sy if is_sync else ex)[(j, ph)].append(x)
        per_step[j][1 if is_sync else 0] += x
        if is_sync:
            n_sync[ph] += 1
        by_op[(ph, base, is_sync)] += x
    out_ph = {}
    for ph in PHASES:
        exs = [sum(ex.get((j, ph), [])) for j in range(1, steps)]
        sys_ = [sum(sy.get((j, ph), [])) for j in range(1, steps)]
        out_ph[ph] = dict(exec_nonsync=t1.stats(exs, 1e3), sync_ops_span=t1.stats(sys_, 1e3),
                          sync_ops_per_step=round(n_sync[ph] / nw, 3))
    st_ex = [per_step[j][0] for j in range(1, steps)]
    st_sy = [per_step[j][1] for j in range(1, steps)]
    gate_s = gate_ms / 1e3
    return dict(drift_ok=bool(np.array_equal(got, ref_counts)), phases=out_ph,
                gate_ms=gate_ms, host_enqueue_us=t1.stats(host_dt, 1e6),
                ops_host_enqueue_over_gate=round(sum(x > gate_s for x in host_dt) / len(host_dt), 4),
                step_exec_nonsync_ms=t1.stats(st_ex, 1e3), step_sync_ops_span_ms=t1.stats(st_sy, 1e3),
                step_upper_ms=t1.stats([a + b for a, b in zip(st_ex, st_sy)], 1e3),
                ms_by_phase_op_per_step={f'{ph} / {b}{" (sync)" if sflag else ""}': round(v / nw * 1e3, 4)
                                         for (ph, b, sflag), v in by_op.most_common()})


def count_pass(model, stim, device, rng, warm, steps):
    """aten ops that launch GPU work and syncs, per phase per step (window steps 1..steps-1)."""
    from torch.utils._python_dispatch import TorchDispatchMode
    inst = Inst('count', steps + 1)
    step = [-1]
    kern, ops, syncs = collections.Counter(), collections.Counter(), collections.Counter()
    sync_ops = collections.Counter()
    last_op = ['?']
    in_win = lambda: 1 <= step[0] <= steps - 1  # noqa: E731

    def on_warning(message, category, filename, lineno, file=None, line=None):
        if 'synchronizing CUDA operation' in str(message) and in_win() and inst.cur:
            syncs[SEG_PHASE[inst.cur]] += 1
            sync_ops[(SEG_PHASE[inst.cur], last_op[0])] += 1

    class Counter(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if in_win() and inst.cur:
                base, cls = t1._op_class(func, args, out)
                last_op[0] = base
                ph = SEG_PHASE[inst.cur]
                ops[ph] += 1
                if cls == 'kernel':
                    kern[ph] += 1
            return out

    # step counter: hook lif_torch._decay (called at the start of every step, as in T1)
    orig = lif_torch._decay

    def wrapped(*a):
        step[0] += 1
        return orig(*a)
    with warnings.catch_warnings():
        warnings.simplefilter('always')
        warnings.showwarning = on_warning
        lif_torch._decay = wrapped
        torch.cuda.set_sync_debug_mode('warn')
        try:
            with Counter():
                instrumented_trial(model, stim, warm['t'] + steps + 1, device, rng, inst, state=warm)
        finally:
            torch.cuda.set_sync_debug_mode(0)
            lif_torch._decay = orig
    nw = steps - 1
    return dict(kernels={p_: round(kern[p_] / nw, 3) for p_ in PHASES},
                aten_ops={p_: round(ops[p_] / nw, 3) for p_ in PHASES},
                syncs={p_: round(syncs[p_] / nw, 3) for p_ in PHASES},
                syncs_by_phase_op={f'{p_} / {o}': round(c / nw, 3) for (p_, o), c in sync_ops.most_common()},
                total=dict(kernels=round(sum(kern.values()) / nw, 2), aten_ops=round(sum(ops.values()) / nw, 2),
                           syncs=round(sum(syncs.values()) / nw, 2)))


class Sampler:
    """nvidia-smi --query-gpu=timestamp,utilization.gpu,clocks.gr,pstate -lms 100 in the
    background for a whole run; samples are later cut into pass windows (wall-clock)."""

    def __init__(self):
        cmd = ['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,clocks.gr,pstate',
               '--format=csv,noheader,nounits', '-lms', '100']
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.lines = []
        self.th = threading.Thread(target=lambda: [self.lines.append(ln) for ln in self.proc.stdout],
                                   daemon=True)
        self.th.start()
        self.windows = {}

    def mark(self, name, t0, t1):
        self.windows[name] = (t0, t1)

    def stop(self):
        time.sleep(0.3)
        self.proc.terminate(); self.proc.wait(timeout=5); self.th.join(timeout=5)
        rows = []
        for ln in self.lines:
            try:
                ts_s, u, clk, pst = [x.strip() for x in ln.split(',')]
                rows.append((datetime.datetime.strptime(ts_s, '%Y/%m/%d %H:%M:%S.%f').timestamp(),
                             float(u), float(clk), pst))
            except ValueError:
                continue
        out = {}
        for name, (t0, t1) in self.windows.items():
            # a sample covers the ~100 ms before its timestamp: keep samples fully inside the window
            w = [r for r in rows if t0 + 0.1 <= r[0] <= t1]
            if not w:
                out[name] = dict(n_samples=0)
                continue
            u = [r[1] for r in w]; c = [r[2] for r in w]
            out[name] = dict(n_samples=len(w), util_median_pct=statistics.median(u), util_min_pct=min(u),
                             util_max_pct=max(u), clock_median_mhz=statistics.median(c), clock_min_mhz=min(c),
                             clock_max_mhz=max(c), pstates=sorted({r[3] for r in w}), util=u, clock=c)
        return out


def gpu_util(model, stim, device, rng, warm, sampler, seconds=2.0, steps=None):
    """Window for the utilisation number: run_trial steps for >= `seconds` (steps after setup)."""
    steps = steps or int(seconds / 1.6e-3) + 200
    off = time.time() - pc()
    torch.cuda.synchronize()
    with t1.StepClock() as clk:
        run_ref(model, stim, warm['t'] + steps, device, rng, state=warm)
    torch.cuda.synchronize()
    t_end = pc()
    sampler.mark('util', clk.stamps[0] + off, t_end + off)
    return dict(window_s=round(t_end - clk.stamps[0], 3), steps=steps)


# ---------------------------------------------------------------- aggregation / report
def _rng(xs, nd=4):
    xs = [x for x in xs if x is not None]
    return dict(median=round(statistics.median(xs), nd), min=round(min(xs), nd), max=round(max(xs), nd),
                max_over_min=round(max(xs) / min(xs), 2) if min(xs) > 0 else None, runs=xs)


def one_run(model, a, warm, ref_window_counts, ref_gated_counts, cyc):
    r = {}
    smp = Sampler()
    time.sleep(1.0)                              # idle baseline
    smp.mark('idle', time.time() - 1.0, time.time())

    def timed(name, fn):
        t0 = time.time(); x = fn(); smp.mark(name, t0, time.time())
        return x
    r['reference'] = timed('reference', lambda: time_reference(model, a.stim, 'cuda', a.rng, warm, a.steps, a.reps))
    r['host'] = timed('host', lambda: time_instrumented(model, a.stim, 'cuda', a.rng, warm, a.steps, a.reps,
                                                        'host', ref_window_counts))
    r['full'] = timed('full', lambda: time_instrumented(model, a.stim, 'cuda', a.rng, warm, a.steps, a.reps,
                                                        'full', ref_window_counts))
    # If the host takes longer than the gate to enqueue an op, that op's span is host-paced:
    # repeat the pass (up to 3 attempts) while > 1% of the ops overran the gate; all attempts are kept.
    r['gated_attempts'] = []
    for _ in range(3):
        gx = timed('gated', lambda: time_gated(model, a.stim, 'cuda', a.rng, warm, a.gated_steps,
                                               ref_gated_counts, cyc, a.gate_ms_measured))
        r['gated_attempts'].append({k: v for k, v in gx.items() if k != 'ms_by_phase_op_per_step'})
        if gx['ops_host_enqueue_over_gate'] <= 0.01:
            break
    r['gated'] = gx
    r['counts'] = count_pass(model, a.stim, 'cuda', a.rng, warm, a.steps)
    r['util'] = gpu_util(model, a.stim, 'cuda', a.rng, warm, smp, a.util_seconds)
    r['nvsmi'] = smp.stop()
    u = r['nvsmi']['util']
    r['util'].update(n_samples=u['n_samples'], median_pct=u.get('util_median_pct'),
                     min_pct=u.get('util_min_pct'), max_pct=u.get('util_max_pct'),
                     clock_median_mhz=u.get('clock_median_mhz'),
                     idle_before_median_pct=r['nvsmi']['idle'].get('util_median_pct'))
    f, h = r['full'], r['host']
    ref_st = r['reference']['steady_median_ms']
    chk = {}
    for name, x in (('host', h), ('full', f)):
        st = x['step']['steady_median_ms']
        s = sum(x['phase_wall_ms'][p]['median'] for p in PHASES)
        chk[f'{name}_sum_phase_wall_medians_ms'] = round(s, 4)
        chk[f'{name}_steady_median_ms'] = st
        chk[f'{name}_sum_over_steady'] = round(s / st, 4)
        chk[f'{name}_sum_incl_loop_over_steady'] = round((s + x['phase_wall_ms']['(loop)']['median']) / st, 4)
        chk[f'{name}_over_reference'] = round(st / ref_st, 4)
    r['check'] = dict(
        **chk,
        gpu_exec_nonsync_over_wall=round(r['gated']['step_exec_nonsync_ms']['median'] / ref_st, 4),
        gpu_upper_over_wall=round(r['gated']['step_upper_ms']['median'] / ref_st, 4),
        sync_classification_matches_sync_debug=all(
            abs(r['gated']['phases'][p]['sync_ops_per_step'] - r['counts']['syncs'][p]) < 1e-9 for p in PHASES),
        drift_ok=bool(f['drift_ok'] and h['drift_ok'] and r['gated']['drift_ok']))
    print(json.dumps(dict(ref=r['reference']['steady_median_ms'], host=r['host']['step']['steady_median_ms'],
                          full=f['step']['steady_median_ms'], stall=h['step']['stall_fraction'],
                          util=r['util']['median_pct'],
                          **r['check'])), flush=True)
    if not r['check']['drift_ok']:
        raise DriftError('DRIFT: an instrumented repeat of the warm window differs from run_trial')
    return r


def aggregate(res):
    runs = res['runs']
    agg = dict(phases={}, variability=[])
    for p in PHASES + ['(loop)']:
        row = {}
        # wall: host-only pass (perf_counter only); wall_events: the pass with events + NVTX
        row['wall_median_ms'] = _rng([r['host']['phase_wall_ms'][p]['median'] for r in runs])
        row['wall_p90_ms'] = _rng([r['host']['phase_wall_ms'][p]['p90'] for r in runs])
        row['wall_events_median_ms'] = _rng([r['full']['phase_wall_ms'][p]['median'] for r in runs])
        row['gpu_span_median_ms'] = _rng([r['full']['phase_gpu_span_ms'][p]['median'] for r in runs])
        if p in PHASES:
            row['gpu_exec_median_ms'] = _rng([r['gated']['phases'][p]['exec_nonsync']['median'] for r in runs])
            row['sync_ops_span_median_ms'] = _rng([r['gated']['phases'][p]['sync_ops_span']['median'] for r in runs])
            row['sync_ops_classified'] = _rng([r['gated']['phases'][p]['sync_ops_per_step'] for r in runs], 3)
            row['syncs'] = _rng([r['counts']['syncs'][p] for r in runs], 3)
            row['kernels'] = _rng([r['counts']['kernels'][p] for r in runs], 3)
            row['aten_ops'] = _rng([r['counts']['aten_ops'][p] for r in runs], 3)
        row['wall_runs_flagged_2x_within'] = sum(r['host']['phase_wall_ms'][p]['varies_2x'] for r in runs)
        agg['phases'][p] = row
        for k in ('wall_median_ms', 'wall_p90_ms', 'wall_events_median_ms', 'gpu_span_median_ms', 'gpu_exec_median_ms',
                  'sync_ops_span_median_ms'):
            if k in row and row[k]['max_over_min'] is not None and row[k]['max_over_min'] > 2:
                agg['variability'].append(dict(what=f'{p} {k}', **{x: row[k][x] for x in (
                    'median', 'min', 'max', 'max_over_min')}))
    g = lambda f: _rng([f(r) for r in runs])  # noqa: E731
    agg['step'] = dict(
        reference_steady_median_ms=g(lambda r: r['reference']['steady_median_ms']),
        reference_steady_mean_ms=g(lambda r: r['reference']['steady_mean_ms']),
        reference_stall_fraction=g(lambda r: r['reference']['stall_fraction']),
        host_steady_median_ms=g(lambda r: r['host']['step']['steady_median_ms']),
        host_steady_mean_ms=g(lambda r: r['host']['step']['steady_mean_ms']),
        host_stall_fraction=g(lambda r: r['host']['step']['stall_fraction']),
        host_stall_median_ms=g(lambda r: r['host']['step']['stall_median_ms']),
        sum_phase_wall_medians_ms=g(lambda r: r['check']['host_sum_phase_wall_medians_ms']),
        sum_over_steady=g(lambda r: r['check']['host_sum_over_steady']),
        sum_incl_loop_over_steady=g(lambda r: r['check']['host_sum_incl_loop_over_steady']),
        host_over_reference=g(lambda r: r['check']['host_over_reference']),
        full_steady_median_ms=g(lambda r: r['full']['step']['steady_median_ms']),
        full_steady_mean_ms=g(lambda r: r['full']['step']['steady_mean_ms']),
        full_stall_fraction=g(lambda r: r['full']['step']['stall_fraction']),
        full_sum_over_steady=g(lambda r: r['check']['full_sum_over_steady']),
        full_over_reference=g(lambda r: r['check']['full_over_reference']),
        gpu_span_over_wall=g(lambda r: r['full']['gpu_span_over_wall_per_step']['median']),
        gpu_exec_nonsync_step_ms=g(lambda r: r['gated']['step_exec_nonsync_ms']['median']),
        gpu_sync_ops_span_step_ms=g(lambda r: r['gated']['step_sync_ops_span_ms']['median']),
        gpu_exec_nonsync_over_wall=g(lambda r: r['check']['gpu_exec_nonsync_over_wall']),
        gpu_upper_over_wall=g(lambda r: r['check']['gpu_upper_over_wall']),
        util_median_pct=g(lambda r: r['util']['median_pct']),
        util_idle_before_pct=g(lambda r: r['util']['idle_before_median_pct']),
        util_window_s=g(lambda r: r['util']['window_s']),
        util_clock_median_mhz=g(lambda r: r['util']['clock_median_mhz']),
        gated_clock_median_mhz=g(lambda r: r['nvsmi']['gated'].get('clock_median_mhz')),
        gated_clock_min_mhz=g(lambda r: r['nvsmi']['gated'].get('clock_min_mhz')),
        host_pass_clock_min_mhz=g(lambda r: r['nvsmi']['host'].get('clock_min_mhz')),
        kernels_total=g(lambda r: r['counts']['total']['kernels']),
        syncs_total=g(lambda r: r['counts']['total']['syncs']),
        aten_ops_total=g(lambda r: r['counts']['total']['aten_ops']),
    )
    for k, v in agg['step'].items():
        if v['max_over_min'] is None or v['max_over_min'] > 2:
            agg['variability'].append(dict(what=f'step {k}', **{x: v[x] for x in (
                'median', 'min', 'max', 'max_over_min')}))
    top = sorted(PHASES, key=lambda p: -agg['phases'][p]['wall_median_ms']['median'])
    agg['top_by_wall'] = top
    return agg


def _m(a, f='{:.3f}'):
    return f"{f.format(a['median'])} [{f.format(a['min'])}-{f.format(a['max'])}]"


def to_md(res):
    A = res['agg']; n = len(res['runs']); S = A['step']
    L = [f"# Week 4 T3 step anatomy ({res['tag']})", '',
         f"Env: {res['env'].get('gpu')} {res['env'].get('cc')}, torch {res['env']['torch']}, "
         f"driver {res['env'].get('driver')}, WSL={res['env']['wsl']}. rng={res['args']['rng']}, float64, "
         f"stim={res['args']['stim']}, warm {res['args']['warm_steps']} steps, window {res['args']['steps']} "
         f"steps x {res['args']['reps']} repeats, {n} runs.", '',
         f"Drift guard: {res['drift_guard']}", '',
         'Every value: median across runs [min-max across runs]; per run, each value is the median over '
         'steady steps (<= 5 ms) of the window. wall = host-only pass (perf_counter); "events pass" = '
         'perf_counter + CUDA events + NVTX. Ratios to wall use the uninstrumented run_trial steady median.', '',
         '| phase | wall ms | wall p90 ms | wall ms, events pass | GPU span ms (events pass) | GPU exec ms, non-sync ops (gated) | sync ops span ms (gated) | syncs | GPU aten ops | aten ops |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for p in PHASES + ['(loop)']:
        r = A['phases'][p]
        L.append(f"| {p} | {_m(r['wall_median_ms'])} | {_m(r['wall_p90_ms'])} | {_m(r['wall_events_median_ms'])} | "
                 f"{_m(r['gpu_span_median_ms'])} | "
                 + (f"{_m(r['gpu_exec_median_ms'])} | {_m(r['sync_ops_span_median_ms'])} | {r['syncs']['median']:g} | {r['kernels']['median']:g} | "
                    f"{r['aten_ops']['median']:g} |" if p in PHASES else '- | - | - | - | - |'))
    L += ['', '| step | value |', '|---|---|']
    for k, v in S.items():
        L.append(f'| {k} | {_m(v)} |')
    L += ['', f"Top 3 phases by wall median: {', '.join(A['top_by_wall'][:3])}", '',
          '## Variability (max/min across runs > 2)', '', '| value | median | min | max | max/min |',
          '|---|---|---|---|---|']
    L += [f"| {v['what']} | {v['median']} | {v['min']} | {v['max']} | {v['max_over_min']} |"
          for v in A['variability']] or ['| (none) | | | | |']
    L += ['', 'Syncs by phase / op (run 1, per step):', '', '```',
          json.dumps(res['runs'][0]['counts']['syncs_by_phase_op'], indent=1), '```']
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', default=str(ROOT / 'results' / 'w4' / 'cns_bundle_calib.npz'))
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--warm-steps', type=int, default=1000)
    ap.add_argument('--reps', type=int, default=10)
    ap.add_argument('--stim', default='looming')
    ap.add_argument('--rng', default='numpy', choices=('numpy', 'torch'))
    ap.add_argument('--drift-steps', type=int, default=300)
    ap.add_argument('--util-seconds', type=float, default=2.0)
    ap.add_argument('--gate-ms', type=float, default=1.0, help='GPU sleep before each op in the gated pass')
    ap.add_argument('--gated-steps', type=int, default=50)
    ap.add_argument('--nsys-window', action='store_true',
                    help='only run the NVTX-instrumented warm window (cudaProfilerApi range), for nsys')
    ap.add_argument('--out', default=str(ROOT / 'perf' / 'results' / 'w4'))
    a = ap.parse_args()
    assert torch.cuda.is_available(), 'needs CUDA'
    model = t1.load_bundle(a.bundle)
    warm = run_ref(model, a.stim, a.warm_steps, 'cuda', a.rng, return_state=True)['state']
    if a.nsys_window:
        # warm-up repeat, then the captured one: NVTX range 'window' is the nsys capture range
        # (nsys profile --capture-range=nvtx --nvtx-capture=window)
        for name in ('warmup', 'window'):
            inst = Inst('nvtx', a.steps + 1)
            torch.cuda.synchronize()
            with torch.cuda.nvtx.range(name):
                instrumented_trial(model, a.stim, warm['t'] + a.steps + 1, 'cuda', a.rng, inst, state=warm)
                torch.cuda.synchronize()
        print('nsys window done', flush=True)
        return
    tag = f'{socket.gethostname()}_{time.strftime("%Y%m%d-%H%M")}'
    res = dict(tag=tag, env=t1.env_info(), args=vars(a), runs=[], bundle_path=a.bundle)
    print(json.dumps(res['env'], indent=1), flush=True)
    try:
        res['drift_guard'] = drift_guard(model, a.stim, 'cuda', a.drift_steps)
    except DriftError as ex:
        print(str(ex), file=sys.stderr, flush=True)
        sys.exit(3)
    print('drift guard', res['drift_guard'], flush=True)
    ref_win = run_ref(model, a.stim, warm['t'] + a.steps + 1, 'cuda', a.rng, state=warm)
    res['window_spikes'] = int(ref_win.sum() - warm['counts'].sum().item())
    ref_gated = run_ref(model, a.stim, warm['t'] + a.gated_steps + 1, 'cuda', a.rng, state=warm)
    cyc, gate_ms = calibrate_sleep(a.gate_ms)
    res['gate'] = dict(cycles=cyc, measured_ms=round(gate_ms, 3))
    a.gate_ms_measured = gate_ms
    for i in range(a.runs):
        print(f'=== run {i + 1}/{a.runs}', flush=True)
        try:
            res['runs'].append(one_run(model, a, warm, ref_win, ref_gated, cyc))
        except DriftError as ex:
            print(str(ex), file=sys.stderr, flush=True)
            sys.exit(3)
    res['agg'] = aggregate(res)
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f'{tag}_anatomy.json').write_text(json.dumps(res, indent=1))
    (outdir / f'{tag}_anatomy.md').write_text(to_md(res))
    S = res['agg']['step']
    print('top3', res['agg']['top_by_wall'][:3], 'sum/steady', S['sum_over_steady'],
          'util', S['util_median_pct'], 'gpu exec/wall', S['gpu_exec_nonsync_over_wall'], flush=True)
    print(f'saved {outdir / tag}_anatomy.json and .md', flush=True)


if __name__ == '__main__':
    main()
