#!/usr/bin/env python
"""Week 4 T1: where the GPU step of sim/lif_torch.py goes (overhead budget).

Micro-benchmarks of host-side costs (launch, sync, small copies, Python dispatch)
plus per-step op / sync / copy counts in the real simulator, combined into a
predicted ms/step that is compared with the measured ms/step of the same steps.

Items (pick with --only, default all):
  1 launch    10,000 x.add_(1) on a 1-element tensor: per-op enqueue (no sync)
              and throughput (with a final sync)
  2 sync      torch.cuda.synchronize() on an idle GPU; .item() after one tiny op
  3 syncops   sync-forcing ops at simulator size (n = 165,122): torch.nonzero
              with 0/100/2,000 true, int(t.sum()); plus the two other sync-forcing
              patterns the simulator uses (boolean-mask index of a short vector,
              x[idx] = python scalar), found by item 6
  4 h2d       torch.as_tensor(np.int64 array of 10/100/1,000, device=...),
              pageable and pinned
  5 graph     50 tiny kernels one by one vs one CUDA-graph replay (cuda only)
  6 counts    per-step counts in lif_torch.run_trial on the bundle (looming,
              rng='torch' and 'numpy'): aten ops dispatched, ops that sync, H2D
              copies, by source line; and measured ms/step for the same steps
  7 budget    predicted ms/step = sum(count x unit cost) vs measured (cuda only)

The simulator is used unmodified. Steps are timed by wrapping
lif_torch._decay (called once at the start of every step) with a timestamp, so
setup (CSC upload) is excluded. Counts and timings use a warm network: 1000
steps (0.1 s) are run first with return_state=True, then the measured window
(--steps, default 200) resumes from that state, so every repeat simulates the
same spikes.

Usage (from the repo root, so this repo's sim/ is imported):
  python bench/w4_overhead.py --bundle ~/flybench/flybrain/results/w4/cns_bundle_calib.npz
Output: perf/results/w4/<host>_<timestamp>_overhead.json and .md
(results/ is gitignored, so committed results live under perf/results/).
"""
import argparse
import collections
import json
import platform
import socket
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim import lif_torch  # noqa: E402
from sim.lif_numpy import P  # noqa: E402

pc = time.perf_counter
N_SIM = 165122
STALL_MS = 5.0     # a step slower than this counts as a stall (typical step ~2 ms)


# ---------------------------------------------------------------- utilities
def sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def stats(samples_s, scale=1e6):
    """median / p90 / mean / min in us (scale=1e6) or ms (1e3), plus a 2x check:
    the repeats are split into 5 consecutive chunks; flag if the slowest chunk
    median is > 2x the fastest, or p90 > 2x median."""
    xs = sorted(samples_s)
    n = len(xs)
    med = statistics.median(xs)
    p90 = xs[int(0.9 * (n - 1))]
    k = max(1, n // 5)
    chunks = [statistics.median(samples_s[i:i + k]) for i in range(0, n - k + 1, k)]
    chunk_ratio = max(chunks) / min(chunks) if min(chunks) > 0 else float('inf')
    return dict(n=n, median=round(med * scale, 3), p90=round(p90 * scale, 3),
                mean=round(statistics.fmean(xs) * scale, 3), min=round(xs[0] * scale, 3),
                chunk_median_ratio=round(chunk_ratio, 2), p90_over_median=round(p90 / med, 2) if med else None,
                varies_2x=bool(chunk_ratio > 2 or (med and p90 / med > 2)))


def time_repeats(fn, device, reps, sync_before=True, sync_after=True, warmup=20):
    """Host wall time of fn() per repeat. sync_before: GPU idle at start."""
    for _ in range(warmup):
        fn()
    sync(device)
    ts = []
    for _ in range(reps):
        if sync_before:
            sync(device)
        t0 = pc()
        fn()
        if sync_after:
            sync(device)
        ts.append(pc() - t0)
    return ts


def _run(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=20)
        b = out.stdout
        s = b.decode('utf-16-le') if b[:2] in (b'\xff\xfe',) or b[1:2] == b'\x00' else b.decode()
        return s.replace('\x00', '').strip()
    except Exception as ex:  # noqa: BLE001
        return f'unavailable ({type(ex).__name__})'


def env_info():
    e = dict(host=socket.gethostname(), platform=platform.platform(), python=platform.python_version(),
             torch=torch.__version__, numpy=np.__version__, cpu_threads=torch.get_num_threads(),
             kernel=platform.release())
    if torch.cuda.is_available():
        pr = torch.cuda.get_device_properties(0)
        e.update(gpu=pr.name, cc=f'sm_{pr.major}{pr.minor}', cuda_runtime=torch.version.cuda,
                 cudnn=torch.backends.cudnn.version())
        e['driver'] = _run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'])
    e['wsl'] = 'microsoft' in platform.release().lower()
    if e['wsl']:
        e['wsl_version'] = _run(['/mnt/c/Windows/System32/wsl.exe', '--version'])
    try:
        import triton
        e['triton'] = triton.__version__
    except Exception:  # noqa: BLE001
        e['triton'] = None
    return e


# ---------------------------------------------------------------- items 1-5
def item1_launch(device, n_ops=10_000, batches=100):
    x = torch.zeros(1, device=device)
    for _ in range(1000):
        x.add_(1)
    sync(device)
    enq, thr, per_op = [], [], []
    for b in range(batches):
        sync(device)
        stamps = [0.0] * (n_ops + 1)
        t0 = pc()
        for i in range(n_ops):
            stamps[i] = pc()
            x.add_(1)
        t1 = pc()
        stamps[n_ops] = t1
        sync(device)
        t2 = pc()
        enq.append((t1 - t0) / n_ops)
        thr.append((t2 - t0) / n_ops)
        if b < 10:   # 100,000 individual per-op samples (incl. ~0.05 us timer cost)
            per_op += [stamps[i + 1] - stamps[i] for i in range(n_ops)]
    return dict(n_ops=n_ops, batches=batches,
                enqueue_us_per_op=stats(enq), throughput_us_per_op_with_sync=stats(thr),
                single_op_interval_us=stats(per_op))


def item2_sync(device):
    x = torch.zeros(1, device=device)
    out = {}
    if device == 'cuda':
        out['synchronize_idle_us'] = stats(time_repeats(torch.cuda.synchronize, device, 2000,
                                                        sync_after=False))
    out['add_then_item_us'] = stats(time_repeats(lambda: (x.add_(1), x.item()), device, 2000,
                                                 sync_after=False))
    out['item_only_us'] = stats(time_repeats(lambda: x.item(), device, 2000, sync_after=False))
    return out


def item3_syncops(device, reps=1000):
    out = {}
    g = torch.Generator(device='cpu').manual_seed(0)
    for k in (0, 100, 2000):
        m = torch.zeros(N_SIM, dtype=torch.bool)
        m[torch.randperm(N_SIM, generator=g)[:k]] = True
        m = m.to(device)
        out[f'nonzero_n{N_SIM}_true{k}_us'] = stats(time_repeats(lambda: torch.nonzero(m), device, reps,
                                                                 sync_after=False))
    t = torch.ones(N_SIM, dtype=torch.int64, device=device)
    out[f'int_sum_n{N_SIM}_us'] = stats(time_repeats(lambda: int(t.sum()), device, reps, sync_after=False))
    # extra patterns found by the item-6 sync scan (both sync on CUDA)
    for L in (20, 311):          # gap spikelet ids; looming stimulus size
        ix = torch.arange(L, device=device)
        r = torch.rand(L, device=device)
        out[f'mask_index_len{L}_us'] = stats(time_repeats(lambda: ix[r < 0.015], device, reps,
                                                          sync_after=False))
    v = torch.zeros(N_SIM, dtype=torch.float64, device=device)
    for k in (20, 200):
        idx = torch.randperm(N_SIM, device=device)[:k]
        out[f'setitem_scalar_n{N_SIM}_k{k}_us'] = stats(
            time_repeats(lambda: v.__setitem__(idx, 0.5), device, reps, sync_after=False))
        tv = torch.full((k,), 0.5, dtype=torch.float64, device=device)
        out[f'setitem_tensor_n{N_SIM}_k{k}_us'] = stats(
            time_repeats(lambda: v.__setitem__(idx, tv), device, reps, sync_after=False))
    return out


def item4_h2d(device, reps=1000):
    out = {}
    for L in (10, 100, 1000):
        a = np.arange(L, dtype=np.int64)
        out[f'pageable_as_tensor_len{L}_us'] = stats(
            time_repeats(lambda: torch.as_tensor(a, device=device), device, reps, sync_after=False))
        if device == 'cuda':
            out[f'pageable_as_tensor_len{L}_with_sync_us'] = stats(
                time_repeats(lambda: torch.as_tensor(a, device=device), device, reps, sync_after=True))
            buf = torch.empty(L, dtype=torch.int64).pin_memory()
            bnp = buf.numpy()

            def pinned():
                bnp[:] = a
                return buf.to(device, non_blocking=True)
            out[f'pinned_nonblocking_len{L}_enqueue_us'] = stats(
                time_repeats(pinned, device, reps, sync_after=False))
            out[f'pinned_nonblocking_len{L}_with_sync_us'] = stats(
                time_repeats(pinned, device, reps, sync_after=True))
    return out


def item5_graph(reps=1000, k=50):
    x = torch.zeros(1, device='cuda')
    out = dict(kernels=k)

    def eager():
        for _ in range(k):
            x.add_(1)
    out['eager_50_with_sync_us'] = stats(time_repeats(eager, 'cuda', reps))
    out['eager_50_enqueue_us'] = stats(time_repeats(eager, 'cuda', reps, sync_after=False))
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                eager()
        torch.cuda.current_stream().wait_stream(s)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            eager()
        before = x.item()
        gr.replay(); torch.cuda.synchronize()
        assert x.item() == before + k, 'graph replay did not add k'
        out['graph_replay_with_sync_us'] = stats(time_repeats(gr.replay, 'cuda', reps))
        out['graph_replay_enqueue_us'] = stats(time_repeats(gr.replay, 'cuda', reps, sync_after=False))
        out['capture_ok'] = True
    except Exception as ex:  # noqa: BLE001
        out['capture_ok'] = False
        out['error'] = f'{type(ex).__name__}: {ex}'[:500]
    return out


# ---------------------------------------------------------------- simulator
def load_bundle(path):
    """Same loader as the bench pack's bench/w4_bench.py."""
    z = np.load(path, allow_pickle=False)
    W = sparse.csc_matrix((z['W_data'], z['W_indices'], z['W_indptr']), shape=tuple(z['W_shape']))
    C = sparse.csr_matrix((z['gap_C_data'], z['gap_C_indices'], z['gap_C_indptr']),
                          shape=tuple(z['gap_C_shape']))
    Wp = sparse.csc_matrix((z['ps_Wp_data'], z['ps_Wp_indices'], z['ps_Wp_indptr']),
                           shape=tuple(z['ps_Wp_shape']))
    info = json.loads(str(z['info']))
    return dict(
        W=W,
        gap=dict(idx=z['gap_idx'], C=C, v_spike=float(z['gap_v_spike'])),
        presyn=dict(idx=z['ps_idx'], Wp=Wp, tauA=float(z['ps_tauA']), tauB=float(z['ps_tauB']),
                    K=float(z['ps_K'])),
        apl=dict(kc_idx=z['apl_kc_idx'], w_kc=z['apl_w_kc'], k=float(z['apl_k']), tau=float(z['apl_tau'])),
        stim={w: z[f'stim_{w}'] for w in info['stims']},
        info=info,
    )


def run(model, stim, t_run, device, rng, seed=0, **kw):
    p = dict(P); p['t_run'] = t_run
    return lif_torch.run_trial(model['W'], model['stim'][stim], params=p, seed=seed,
                               apl=model['apl'], gap=model['gap'], presyn=model['presyn'],
                               device=device, dtype=torch.float64, rng=rng, compile=False, **kw)


class StepClock:
    """Wraps lif_torch._decay (first call of every step) to timestamp steps."""

    def __init__(self, on_step=None):
        self.stamps = []
        self.on_step = on_step

    def __enter__(self):
        self.orig = lif_torch._decay

        def wrapped(*a):
            if self.on_step is not None:
                self.on_step()
            self.stamps.append(pc())
            return self.orig(*a)
        lif_torch._decay = wrapped
        return self

    def __exit__(self, *exc):
        lif_torch._decay = self.orig


VIEWLIKE = {'view', '_unsafe_view', 'as_strided', 'select', 'slice', 'expand', 'squeeze', 'unsqueeze',
            't', 'transpose', 'permute', 'alias', 'detach', 'lift_fresh', '_reshape_alias', 'empty',
            'empty_strided', 'scalar_tensor', 'unbind', 'split', 'flatten', 'reshape'}


def _op_class(func, args, out):
    """kernel (launches GPU work) / host (view, metadata, CPU-only tensors)."""
    name = func.__name__.split('.')[0] if hasattr(func, '__name__') else str(func)
    base = str(func).split('.')[1] if str(func).startswith('aten.') else name
    if base in VIEWLIKE:
        return base, 'host'
    outs = out if isinstance(out, (tuple, list)) else (out,)
    dev = None
    for o in outs:
        if isinstance(o, torch.Tensor):
            dev = o.device.type
            break
    if dev is None:          # e.g. _local_scalar_dense returns a python number
        for a in args:
            if isinstance(a, torch.Tensor):
                dev = a.device.type
                break
    return base, ('kernel' if dev == 'cuda' else 'host')


def count_ops_run(model, stim, device, rng, steps, warm_state):
    """Dispatch-level counts per step (TorchDispatchMode). Syncs are detected with
    torch.cuda.set_sync_debug_mode('warn') and attributed to the aten op and the
    lif_torch.py line that triggered them."""
    from torch.utils._python_dispatch import TorchDispatchMode
    sim_file = lif_torch.__file__
    step = [-1]
    per_op = collections.Counter()
    per_line = collections.Counter()
    sync_by = collections.Counter()
    h2d_by = collections.Counter()
    d2h_by = collections.Counter()

    def caller_line():
        f = sys._getframe(2)
        while f is not None:
            if f.f_code.co_filename == sim_file:
                return f'{f.f_code.co_name}:{f.f_lineno}'
            f = f.f_back
        return '?'

    last = [('?', '?')]
    in_win = lambda: 1 <= step[0] <= steps - 1   # window = steps 1 .. steps-1 (whole steps)  # noqa: E731

    # The sync-debug warning raised inside an aten op is buffered by the
    # outermost Python->C++ binding and reaches Python only when that call
    # returns, i.e. after the op that caused it. It is attributed to the last
    # aten op dispatched before it (checked by hand: nonzero, index with a bool
    # mask, _local_scalar_dense, index_put_ with a python scalar, _to_copy).
    def on_warning(message, category, filename, lineno, file=None, line=None):
        if 'synchronizing CUDA operation' in str(message) and in_win():
            sync_by[last[0]] += 1

    with warnings.catch_warnings():
        warnings.simplefilter('always')
        warnings.showwarning = on_warning

        class Counter(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                out = func(*args, **kwargs)
                if not in_win():
                    return out
                base, cls = _op_class(func, args, out)
                line = caller_line()
                last[0] = (base, line)
                per_op[(base, cls)] += 1
                per_line[line] += 1
                if base in ('_to_copy', 'copy_'):
                    src = args[1] if base == 'copy_' else args[0]
                    dst_dev = (args[0].device.type if base == 'copy_'
                               else (out.device.type if isinstance(out, torch.Tensor) else None))
                    if isinstance(src, torch.Tensor) and src.device.type != dst_dev:
                        (h2d_by if src.device.type == 'cpu' else d2h_by)[(base, line)] += 1
                if base == '_local_scalar_dense' and device == 'cuda':
                    d2h_by[(base, line)] += 1
                return out

        def on_step():
            step[0] += 1
        if device == 'cuda':
            torch.cuda.set_sync_debug_mode('warn')
        try:
            # run steps+1 steps so the window 1..steps-1 is complete, plus one closing step
            with StepClock(on_step), Counter():
                run(model, stim, (warm_state['t'] + steps + 1) * P['dt'], device, rng, state=warm_state)
        finally:
            if device == 'cuda':
                torch.cuda.set_sync_debug_mode(0)
    nwin = steps - 1
    return dict(steps_in_window=nwin, per_op=per_op, per_line=per_line, sync_by=sync_by,
                h2d_by=h2d_by, d2h_by=d2h_by)


def time_steps(model, stim, device, rng, steps, warm_state, reps):
    """Per-step host intervals between consecutive step starts (setup excluded)."""
    per_step, run_mean = [], []
    for _ in range(reps):
        sync(device)
        with StepClock() as clk:
            run(model, stim, (warm_state['t'] + steps + 1) * P['dt'], device, rng, state=warm_state)
        s = clk.stamps
        d = [s[i + 1] - s[i] for i in range(1, steps)]    # same window as counts: steps 1..steps-1
        per_step += d
        run_mean.append(statistics.fmean(d))
    # Stalls: on WSL some steps take ~10 ms instead of ~2 ms, in bursts that come
    # and go between runs. They are reported separately, not averaged away.
    thr = STALL_MS / 1e3
    steady = [x for x in per_step if x <= thr]
    stalls = [x for x in per_step if x > thr]
    return dict(per_step_ms=stats(per_step, 1e3), run_mean_ms=stats(run_mean, 1e3),
                stall_threshold_ms=STALL_MS, stall_steps=len(stalls),
                stall_fraction=round(len(stalls) / len(per_step), 4),
                stall_median_ms=round(statistics.median(stalls) * 1e3, 3) if stalls else None,
                steady_mean_ms=round(statistics.fmean(steady) * 1e3, 3),
                stall_ms_per_step=round((sum(per_step) - sum(steady)) / len(per_step) * 1e3, 3),
                per_step_raw_ms=[round(x * 1e3, 3) for x in per_step])


def sim_section(model, device, steps, reps, stim, warm_steps):
    out = dict(stim=stim, steps=steps, warm_steps=warm_steps, dtype='float64', compile=False)
    for rng in ('torch', 'numpy'):
        r = run(model, stim, warm_steps * P['dt'], device, rng, return_state=True)
        warm = r['state']
        spikes_warm = int(r['counts'].sum())
        res = dict(warm_spikes_total=spikes_warm)
        # spikes in the window (reference run without instrumentation)
        r2 = run(model, stim, (warm['t'] + steps + 1) * P['dt'], device, rng, state=warm, record=True)
        tt = r2['t']
        win = (tt >= warm['t'] + 1) & (tt < warm['t'] + steps)
        res['spikes_per_step_window'] = round(float(win.sum()) / (steps - 1), 2)
        res['timing'] = time_steps(model, stim, device, rng, steps, warm, reps)
        c = count_ops_run(model, stim, device, rng, steps, warm)
        nw = c['steps_in_window']
        res['counts_per_step'] = dict(
            aten_ops=round(sum(c['per_op'].values()) / nw, 2),
            kernel_ops=round(sum(v for (o, k), v in c['per_op'].items() if k == 'kernel') / nw, 2),
            host_ops=round(sum(v for (o, k), v in c['per_op'].items() if k == 'host') / nw, 2),
            syncs=round(sum(c['sync_by'].values()) / nw, 2),
            h2d_copies=round(sum(c['h2d_by'].values()) / nw, 2),
            d2h_copies=round(sum(c['d2h_by'].values()) / nw, 2),
            nonzero=round(sum(v for (o, k), v in c['per_op'].items() if o == 'nonzero') / nw, 2),
            local_scalar_dense=round(sum(v for (o, k), v in c['per_op'].items()
                                         if o == '_local_scalar_dense') / nw, 2),
        )
        res['ops_by_name_per_step'] = {f'{o} [{k}]': round(v / nw, 3)
                                       for (o, k), v in c['per_op'].most_common()}
        res['ops_by_line_per_step'] = {ln: round(v / nw, 3) for ln, v in c['per_line'].most_common()}
        res['syncs_by_op_line_per_step'] = {f'{o} @ {ln}': round(v / nw, 3)
                                            for (o, ln), v in c['sync_by'].most_common()}
        res['h2d_by_op_line_per_step'] = {f'{o} @ {ln}': round(v / nw, 3)
                                          for (o, ln), v in c['h2d_by'].most_common()}
        res['d2h_by_op_line_per_step'] = {f'{o} @ {ln}': round(v / nw, 3)
                                          for (o, ln), v in c['d2h_by'].most_common()}
        out[rng] = res
        print(f'  [{device}] rng={rng}', json.dumps(res['counts_per_step']),
              'ms/step', res['timing']['per_step_ms']['mean'], flush=True)
    return out


def bench_method_sim(model, device, stim, bio=0.2, repeats=3):
    """Re-run of the bench-pack measurement (bench/w4_bench.py sec_sim: float64,
    eager, rng='torch', 0.2 s, median of 3, setup included) with main's sim/."""
    out = {}
    for rng in ('torch', 'numpy'):
        run(model, stim, 0.005, device, rng)
        walls, steps_ms = [], []
        for r in range(repeats):
            sync(device)
            t0 = pc()
            with StepClock() as clk:
                run(model, stim, bio, device, rng, seed=r)
            sync(device)
            walls.append(pc() - t0)
            s = clk.stamps
            steps_ms += [(s[i + 1] - s[i]) * 1e3 for i in range(len(s) - 1)]
        out[rng] = dict(bio_s=bio, repeats=repeats,
                        ms_per_step_incl_setup=round(statistics.median(walls) / (bio / P['dt']) * 1e3, 3),
                        walls_s=[round(w, 3) for w in walls],
                        per_step_interval_ms=stats([x / 1e3 for x in steps_ms], 1e3))
        print(f'  [{device}] bench-method rng={rng}', out[rng]['ms_per_step_incl_setup'], flush=True)
    return out


# ---------------------------------------------------------------- budget
def budget(res, rng):
    """Predicted ms/step = sum over aten ops of a unit cost, split by category.

    Every aten op pays the host dispatch cost c_disp (item 1 on cpu: Python ->
    aten -> tiny CPU kernel). Ops that launch GPU work also pay
    c_launch = c_enq - c_disp, where c_enq is item 1 on cuda with a final sync
    (10,000 tiny kernels: the rate at which the GPU retires them, which is what a
    step pays because every step ends its kernel chains in a sync). Sync-forcing /
    copy ops pay, on top, their measured micro cost minus what dispatch+launch
    already covers:
      nonzero                     -> item 3 nonzero (n=165,122, 100 true) - c_enq
      index (bool mask, short)    -> item 3 mask_index_len311 - 2 c_enq (lt + index);
                                     used for both mask sites (drive, spikelets)
      _local_scalar_dense         -> item 2 add_then_item - c_enq - c_disp (D2H + sync)
      index_put_ with CPU scalar  -> item 3 setitem_scalar_k20 - setitem_tensor_k20
      _to_copy H2D (as_tensor)    -> item 4 pageable_as_tensor_len10 - c_enq - c_disp
    Not modelled: GPU execution time beyond the tiny-kernel rate, and stalls
    (steps > STALL_MS); the measured side is reported both with and without them.
    """
    m = res['micro']['cuda']
    mc = res['micro']['cpu']
    c_disp = mc['1_launch']['enqueue_us_per_op']['median']
    c_enq = m['1_launch']['throughput_us_per_op_with_sync']['median']   # per tiny kernel, GPU-paced
    c_launch = max(c_enq - c_disp, 0.0)
    s3, s2, s4 = m['3_syncops'], m['2_sync'], m['4_h2d']
    extra = {
        'nonzero': s3[f'nonzero_n{N_SIM}_true100_us']['median'] - c_enq,
        'index': s3['mask_index_len311_us']['median'] - 2 * c_enq,   # lt + index
        '_local_scalar_dense': s2['add_then_item_us']['median'] - c_enq - c_disp,
        'index_put_': (s3[f'setitem_scalar_n{N_SIM}_k20_us']['median']
                       - s3[f'setitem_tensor_n{N_SIM}_k20_us']['median']),
        '_to_copy': s4['pageable_as_tensor_len10_us']['median'] - c_enq - c_disp,
    }
    sim = res['sim']['cuda'][rng]
    c = sim['counts_per_step']
    # sync / copy op counts from the attribution tables
    syncs = collections.Counter()
    for k, v in sim['syncs_by_op_line_per_step'].items():
        syncs[k.split(' @ ')[0]] += v
    h2d_n = sum(v for k, v in sim['h2d_by_op_line_per_step'].items())
    cats = collections.OrderedDict()
    cats['python+dispatch'] = c['aten_ops'] * c_disp
    cats['kernel launch'] = c['kernel_ops'] * c_launch
    sync_us = 0.0
    sync_detail = {}
    for op, n in syncs.items():
        if op == '_to_copy':
            continue                      # costed as H2D below
        if op == 'index_put_':
            continue                      # scalar index_put_: hidden H2D + sync, own category
        u = extra.get(op, extra['_local_scalar_dense'])
        sync_detail[op] = dict(per_step=n, unit_us=round(u, 2))
        sync_us += n * u
    cats['sync round trips (D2H)'] = sync_us
    sip = syncs.get('index_put_', 0.0)
    cats['scalar index_put_ (hidden H2D+sync)'] = sip * extra['index_put_']
    cats['as_tensor H2D (pageable)'] = h2d_n * extra['_to_copy'] if h2d_n else 0.0
    # avoid double counting: hidden H2D of scalar index_put_ is seen as a sync, not a _to_copy
    pred_ms = sum(cats.values()) / 1e3
    tm = sim['timing']
    meas, steady = tm['per_step_ms']['mean'], tm['steady_mean_ms']
    return dict(rng=rng,
                unit_costs_us=dict(c_disp=round(c_disp, 3), c_enq=round(c_enq, 3),
                                   c_launch=round(c_launch, 3),
                                   **{f'extra_{k}': round(v, 3) for k, v in extra.items()}),
                counts_per_step=dict(aten_ops=c['aten_ops'], kernel_ops=c['kernel_ops'],
                                     syncs_by_op=dict(syncs), h2d=h2d_n),
                sync_detail=sync_detail,
                category_ms={k: round(v / 1e3, 4) for k, v in cats.items()},
                predicted_ms=round(pred_ms, 3), measured_ms_mean=meas,
                measured_ms_median=tm['per_step_ms']['median'],
                measured_ms_steady_mean=steady, stall_ms_per_step=tm['stall_ms_per_step'],
                stall_fraction=tm['stall_fraction'],
                ratio_pred_over_meas=round(pred_ms / meas, 3),
                within_20pct=bool(abs(pred_ms / meas - 1) <= 0.20),
                ratio_pred_over_steady=round(pred_ms / steady, 3),
                within_20pct_steady=bool(abs(pred_ms / steady - 1) <= 0.20))


# ---------------------------------------------------------------- report
def to_md(res):
    L = [f"# Week 4 T1 overhead budget: {res['tag']}", '',
         '```', json.dumps(res['env'], indent=1), '```', '']
    for d, m in res.get('micro', {}).items():
        L += [f'## Micro-benchmarks ({d})', '', '| item | measurement | n | median | p90 | 2x flag |',
              '|---|---|---|---|---|---|']
        for item, vals in m.items():
            for k, v in vals.items():
                if isinstance(v, dict) and 'median' in v:
                    L.append(f"| {item} | {k} | {v['n']} | {v['median']} | {v['p90']} | "
                             f"{'**yes**' if v['varies_2x'] else 'no'} |")
                elif not isinstance(v, dict):
                    L.append(f'| {item} | {k} | | {v} | | |')
        L.append('')
    for d, s in res.get('sim', {}).items():
        L += [f'## Simulator per-step counts and timing ({d})', '']
        for rng in ('torch', 'numpy'):
            if rng not in s:
                continue
            r = s[rng]
            L += [f'### rng={rng}', '', '```',
                  json.dumps(dict(spikes_per_step=r['spikes_per_step_window'],
                                  counts_per_step=r['counts_per_step'],
                                  timing={k: v for k, v in r['timing'].items() if k != 'per_step_raw_ms'}),
                             indent=1),
                  '```', '', 'Syncs by op @ line (per step):', '', '```',
                  json.dumps(r['syncs_by_op_line_per_step'], indent=1), '```',
                  '', 'H2D by op @ line (per step):', '', '```',
                  json.dumps(r['h2d_by_op_line_per_step'], indent=1), '```', '']
    if 'bench_method' in res:
        L += ['## Bench-pack method with main\'s sim/ (0.2 s, setup included)', '', '```',
              json.dumps(res['bench_method'], indent=1), '```', '']
    if 'budget' in res:
        L += ['## Budget (cuda)', '']
        for b in res['budget']:
            L += [f"### rng={b['rng']}: predicted {b['predicted_ms']} ms vs measured mean "
                  f"{b['measured_ms_mean']} ms (ratio {b['ratio_pred_over_meas']}), steady-state mean "
                  f"{b['measured_ms_steady_mean']} ms (ratio {b['ratio_pred_over_steady']}); stalls "
                  f"{b['stall_ms_per_step']} ms/step ({b['stall_fraction']:.1%} of steps)", '',
                  '| category | ms/step |', '|---|---|']
            L += [f'| {k} | {v} |' for k, v in b['category_ms'].items()]
            L += ['', '```', json.dumps(dict(unit_costs_us=b['unit_costs_us'],
                                             counts_per_step=b['counts_per_step']), indent=1), '```', '']
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', default=str(ROOT / 'results' / 'w4' / 'cns_bundle_calib.npz'))
    ap.add_argument('--devices', default='cuda,cpu')
    ap.add_argument('--only', default='1,2,3,4,5,6,7,bench')
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--warm-steps', type=int, default=1000)
    ap.add_argument('--reps', type=int, default=10, help='timed repeats of the --steps window')
    ap.add_argument('--stim', default='looming')
    ap.add_argument('--out', default=str(ROOT / 'perf' / 'results' / 'w4'))
    a = ap.parse_args()
    only = set(a.only.split(','))
    devices = [d for d in a.devices.split(',') if d == 'cpu' or (d == 'cuda' and torch.cuda.is_available())]
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    tag = f'{socket.gethostname()}_{time.strftime("%Y%m%d-%H%M")}'
    res = dict(tag=tag, env=env_info(), args=vars(a), micro={}, sim={})
    print(json.dumps(res['env'], indent=1), flush=True)
    for d in devices:
        m = {}
        if '1' in only:
            m['1_launch'] = item1_launch(d)
        if '2' in only:
            m['2_sync'] = item2_sync(d)
        if '3' in only:
            m['3_syncops'] = item3_syncops(d)
        if '4' in only:
            m['4_h2d'] = item4_h2d(d)
        if '5' in only and d == 'cuda':
            m['5_graph'] = item5_graph()
        res['micro'][d] = m
        print(f'micro {d}:', json.dumps({k: {kk: (vv['median'] if isinstance(vv, dict) and 'median' in vv
                                                  else vv) for kk, vv in v.items()} for k, v in m.items()}),
              flush=True)
    if only & {'6', '7', 'bench'}:
        model = load_bundle(a.bundle)
        res['bundle'] = model['info']
        res['bundle_path'] = a.bundle
        res['bundle_sizes'] = dict(n=int(model['W'].shape[0]), nnz=int(model['W'].nnz),
                                   n_stim={w: int(len(v)) for w, v in model['stim'].items()},
                                   n_gap=int(len(model['gap']['idx'])),
                                   n_presyn=int(len(model['presyn']['idx'])),
                                   n_kc=int(len(model['apl']['kc_idx'])))
        if '6' in only or '7' in only:
            for d in devices:
                res['sim'][d] = sim_section(model, d, a.steps, a.reps, a.stim, a.warm_steps)
        if 'bench' in only and 'cuda' in devices:
            res['bench_method'] = bench_method_sim(model, 'cuda', a.stim)
            res['bench_method']['reference_ms_per_step_bench_pack'] = 3.327
    if '7' in only and 'cuda' in res['sim'] and 'cuda' in res['micro'] and 'cpu' in res['micro']:
        res['budget'] = [budget(res, rng) for rng in ('numpy', 'torch')]
        for b in res['budget']:
            print('budget', json.dumps(b), flush=True)
    (outdir / f'{tag}_overhead.json').write_text(json.dumps(res, indent=1))
    (outdir / f'{tag}_overhead.md').write_text(to_md(res))
    print(f'saved {outdir / tag}_overhead.json and .md', flush=True)


if __name__ == '__main__':
    main()
