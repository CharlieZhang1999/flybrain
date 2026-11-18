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
  python bench/w4_overhead.py --bundle ~/flybench/flybrain/results/w4/cns_bundle_calib.npz [--runs 5]
The whole suite runs --runs times (default 5) in one invocation; the report gives,
for every value, the median across runs and the min-max across runs.
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
    for k in (1, 20, 100):     # the simulator assigns to ~17 spiking neurons per step
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
def units(micro):
    """Unit costs (us) from one run's micro results, or from the across-run
    aggregate (same shape; every value has a 'median').

    Every aten op pays the host dispatch cost c_disp (item 1 on cpu: Python ->
    aten -> tiny CPU kernel). Ops that launch GPU work also pay
    c_launch = c_enq - c_disp, where c_enq is item 1 on cuda with a final sync
    (10,000 tiny kernels: the rate at which the GPU retires them). Sync-forcing /
    copy ops pay, on top, a directly measured total minus the dispatch + launch
    already charged to the same ops; these extras are clamped at 0.
      nonzero               median over the 3 sizes (0/100/2,000 true)       - c_enq
      index (bool mask)     median over lengths 20 and 311                    - 2 c_enq (lt + index)
      _local_scalar_dense   item 2 add_then_item                              - c_enq - c_disp
      index_put_ (scalar)   median of v[idx] = 0.5 for 1/20/100 idx           - c_enq - c_disp (lift_fresh)
      _to_copy H2D          median of pageable as_tensor of 10/100/1,000 ints - c_enq - c_disp
    """
    m, mc = micro['cuda'], micro['cpu']
    med = lambda xs: statistics.median(xs)  # noqa: E731
    c_disp = mc['1_launch']['enqueue_us_per_op']['median']
    c_enq = m['1_launch']['throughput_us_per_op_with_sync']['median']
    s2, s3, s4 = m['2_sync'], m['3_syncops'], m['4_h2d']
    tot = dict(
        nonzero=med([s3[f'nonzero_n{N_SIM}_true{k}_us']['median'] for k in (0, 100, 2000)]),
        index=med([s3[f'mask_index_len{L}_us']['median'] for L in (20, 311)]),
        _local_scalar_dense=s2['add_then_item_us']['median'],
        index_put_=med([s3[f'setitem_scalar_n{N_SIM}_k{k}_us']['median'] for k in (1, 20, 100)]),
        _to_copy=med([s4[f'pageable_as_tensor_len{L}_us']['median'] for L in (10, 100, 1000)]),
    )
    already = dict(nonzero=c_enq, index=2 * c_enq, _local_scalar_dense=c_enq + c_disp,
                   index_put_=c_enq + c_disp, _to_copy=c_enq + c_disp)
    extra = {k: max(tot[k] - already[k], 0.0) for k in tot}
    clamped = [k for k in tot if tot[k] - already[k] < 0]
    return dict(c_disp=c_disp, c_enq=c_enq, c_launch=max(c_enq - c_disp, 0.0),
                launch_clamped=c_enq < c_disp, total=tot, extra=extra, clamped=clamped)


def counts_of(sim_rng):
    syncs = collections.Counter()
    for k, v in sim_rng['syncs_by_op_line_per_step'].items():
        syncs[k.split(' @ ')[0]] += v
    c = sim_rng['counts_per_step']
    return dict(aten_ops=c['aten_ops'], kernel_ops=c['kernel_ops'], syncs_by_op=dict(syncs),
                h2d=sum(sim_rng['h2d_by_op_line_per_step'].values()))


def budget(u, c):
    """Predicted ms/step by category from unit costs u (us) and per-step counts c."""
    s = c['syncs_by_op']
    ex = u['extra']
    cats = collections.OrderedDict()
    cats['python+dispatch'] = c['aten_ops'] * u['c_disp']
    cats['kernel launch'] = c['kernel_ops'] * u['c_launch']
    # D2H round trips; index_put_ and _to_copy syncs are costed in their own categories
    cats['sync round trips (D2H)'] = sum(n * ex.get(op, ex['_local_scalar_dense'])
                                         for op, n in s.items() if op not in ('index_put_', '_to_copy'))
    cats['scalar index_put_ (hidden H2D+sync)'] = s.get('index_put_', 0.0) * ex['index_put_']
    cats['as_tensor H2D (pageable)'] = c['h2d'] * ex['_to_copy']
    cats = {k: round(v / 1e3, 4) for k, v in cats.items()}
    return cats, round(sum(cats.values()), 3)


# ---------------------------------------------------------------- aggregation
def _rng(xs):
    return dict(median=round(statistics.median(xs), 3), min=round(min(xs), 3), max=round(max(xs), 3),
                max_over_min=round(max(xs) / min(xs), 2) if min(xs) > 0 else None, runs=xs)


def aggregate(res):
    runs = res['runs']
    agg = dict(micro={}, sim={}, budget=[], variability=[])
    # micro: median across runs of each per-run median
    for d in runs[0]['micro']:
        agg['micro'][d] = {}
        for it, vals in runs[0]['micro'][d].items():
            agg['micro'][d][it] = {}
            for k, v in vals.items():
                if isinstance(v, dict) and 'median' in v:
                    xs = [r['micro'][d][it][k]['median'] for r in runs]
                    p90 = [r['micro'][d][it][k]['p90'] for r in runs]
                    a = _rng(xs)
                    a['p90_median'] = round(statistics.median(p90), 3)
                    a['runs_flagged_2x_within'] = sum(r['micro'][d][it][k]['varies_2x'] for r in runs)
                    agg['micro'][d][it][k] = a
                    if a['max_over_min'] and a['max_over_min'] > 2:
                        agg['variability'].append(dict(what=f'{d} {it} {k} (us)', **{x: a[x] for x in (
                            'median', 'min', 'max', 'max_over_min')}))
    # simulator timing + counts
    for d in runs[0]['sim']:
        agg['sim'][d] = {}
        for rng in ('torch', 'numpy'):
            rs = [r['sim'][d][rng] for r in runs]
            t = {k: _rng([x['timing'][k] if k in x['timing'] else x['timing']['per_step_ms'][k.split(':')[1]]
                          for x in rs])
                 for k in ('steady_mean_ms', 'stall_fraction', 'per_step_ms:mean', 'per_step_ms:median')}
            cnt = {k: _rng([x['counts_per_step'][k] for x in rs]) for k in rs[0]['counts_per_step']}
            agg['sim'][d][rng] = dict(timing=t, counts=cnt,
                                      spikes_per_step=_rng([x['spikes_per_step_window'] for x in rs]))
            for k, lab in (('stall_fraction', 'stall fraction'), ('per_step_ms:mean', 'all-steps mean ms/step'),
                           ('steady_mean_ms', 'steady-state mean ms/step')):
                a = t[k]
                if a['max_over_min'] is None or a['max_over_min'] > 2:
                    agg['variability'].append(dict(what=f'{d} rng={rng} {lab}', **{x: a[x] for x in (
                        'median', 'min', 'max', 'max_over_min')}))
    if 'bench_method' in runs[0]:
        agg['bench_method'] = {rng: _rng([r['bench_method'][rng]['ms_per_step_incl_setup'] for r in runs])
                               for rng in ('torch', 'numpy')}
        agg['bench_method']['reference_ms_per_step_bench_pack'] = 3.327
        for rng in ('torch', 'numpy'):
            a = agg['bench_method'][rng]
            if a['max_over_min'] > 2:
                agg['variability'].append(dict(what=f'bench-method rng={rng} ms/step (setup incl.)',
                                               **{x: a[x] for x in ('median', 'min', 'max', 'max_over_min')}))
    # budget: aggregated units x median counts, plus per-run predictions
    if 'cuda' in agg['sim'] and 'cpu' in agg['micro']:
        ua = units(agg['micro'])
        agg['units_us'] = {k: (round(v, 3) if isinstance(v, float) else
                               ({kk: round(vv, 3) for kk, vv in v.items()} if isinstance(v, dict) else v))
                           for k, v in ua.items()}
        for rng in ('numpy', 'torch'):
            per = []
            for r in runs:
                c = counts_of(r['sim']['cuda'][rng])
                cats, pred = budget(units(r['micro']), c)
                steady = r['sim']['cuda'][rng]['timing']['steady_mean_ms']
                per.append(dict(predicted_ms=pred, steady_ms=steady, ratio=round(pred / steady, 3),
                                categories=cats, counts=c))
            cs = [p['counts'] for p in per]
            cmed = dict(aten_ops=statistics.median([c['aten_ops'] for c in cs]),
                        kernel_ops=statistics.median([c['kernel_ops'] for c in cs]),
                        h2d=statistics.median([c['h2d'] for c in cs]),
                        syncs_by_op={op: statistics.median([c['syncs_by_op'].get(op, 0.0) for c in cs])
                                     for op in cs[0]['syncs_by_op']})
            cats, pred = budget(ua, cmed)
            st = agg['sim']['cuda'][rng]['timing']
            agg['budget'].append(dict(
                rng=rng, counts_median=cmed, category_ms_agg_units=cats, predicted_ms_agg_units=pred,
                predicted_ms_per_run=_rng([p['predicted_ms'] for p in per]),
                category_ms_per_run={k: _rng([p['categories'][k] for p in per]) for k in cats},
                steady_ms=st['steady_mean_ms'], all_steps_ms=st['per_step_ms:mean'],
                stall_fraction=st['stall_fraction'],
                ratio_pred_over_steady_per_run=_rng([p['ratio'] for p in per]),
                ratio_agg_units_over_steady_median=round(pred / st['steady_mean_ms']['median'], 3),
                negative_category=any(v < 0 for p in per for v in p['categories'].values())
                or any(v < 0 for v in cats.values())))
            r_ = agg['budget'][-1]['ratio_pred_over_steady_per_run']
            if r_['max_over_min'] > 2:
                agg['variability'].append(dict(what=f'budget rng={rng} predicted/steady ratio',
                                               **{x: r_[x] for x in ('median', 'min', 'max', 'max_over_min')}))
    return agg


# ---------------------------------------------------------------- report
def _mmm(a, f='{:g}'):
    return f"{f.format(a['median'])} [{f.format(a['min'])}-{f.format(a['max'])}]"


def to_md(res):
    agg = res['agg']
    n = len(res['runs'])
    L = [f"# Week 4 T1 overhead budget: {res['tag']} ({n} runs in one invocation)", '',
         'Every value: median across runs [min-max across runs]. Per run, each timing is itself the median '
         'of >= 1,000 repeats (item 1: 100 batches of 10,000 ops; simulator: 1,990 steps).', '',
         '```', json.dumps(res['env'], indent=1), '```', '']
    if agg.get('budget'):
        L += ['## Budget (cuda, ms per 0.1-ms step)', '',
              '| category | ' + ' | '.join(f"rng={b['rng']}" for b in agg['budget']) + ' |',
              '|---|' + '---|' * len(agg['budget'])]
        for k in agg['budget'][0]['category_ms_agg_units']:
            L.append(f'| {k} | ' + ' | '.join(f"{b['category_ms_agg_units'][k]:.2f}" for b in agg['budget']) + ' |')
        rows = [('**predicted, units = medians across runs**',
                 lambda b: f"**{b['predicted_ms_agg_units']:.2f}**"),
                ('predicted per run, median [min-max]', lambda b: _mmm(b['predicted_ms_per_run'], '{:.2f}')),
                ('measured steady state (steps <= 5 ms)', lambda b: _mmm(b['steady_ms'], '{:.2f}')),
                ('measured all steps', lambda b: _mmm(b['all_steps_ms'], '{:.2f}')),
                ('stall fraction', lambda b: _mmm(b['stall_fraction'], '{:.3f}')),
                ('predicted / steady, per run', lambda b: _mmm(b['ratio_pred_over_steady_per_run'], '{:.2f}')),
                ('any negative category', lambda b: str(b['negative_category']))]
        for lab, f in rows:
            L.append(f'| {lab} | ' + ' | '.join(f(b) for b in agg['budget']) + ' |')
        L += ['', 'Unit costs from the medians across runs (us):', '', '```',
              json.dumps(agg['units_us'], indent=1), '```', '']
    L += ['## Variability: values whose min-max across runs differs by more than 2x', '',
          '| value | median | min | max | max/min |', '|---|---|---|---|---|']
    L += [f"| {v['what']} | {v['median']} | {v['min']} | {v['max']} | {v['max_over_min']} |"
          for v in agg['variability']]
    L.append('')
    for d, m in agg['micro'].items():
        L += [f'## Micro-benchmarks ({d}), us', '',
              '| item | measurement | median | min | max | max/min | p90 (median) | runs >2x within |',
              '|---|---|---|---|---|---|---|---|']
        for it, vals in m.items():
            for k, a in vals.items():
                L.append(f"| {it} | {k} | {a['median']} | {a['min']} | {a['max']} | {a['max_over_min']} | "
                         f"{a['p90_median']} | {a['runs_flagged_2x_within']}/{n} |")
        L.append('')
    for d, s in agg['sim'].items():
        L += [f'## Simulator ({d})', '']
        for rng, a in s.items():
            L += [f'### rng={rng}', '', '```', json.dumps(dict(
                spikes_per_step=a['spikes_per_step'], counts={k: v['median'] for k, v in a['counts'].items()},
                timing={k: {x: v[x] for x in ('median', 'min', 'max')} for k, v in a['timing'].items()}),
                indent=1), '```', '']
        r0 = res['runs'][0]['sim'][d]
        for rng in ('torch', 'numpy'):
            if r0[rng]['syncs_by_op_line_per_step']:
                L += [f'Syncs by op @ line, rng={rng}, run 1 (per step):', '', '```',
                      json.dumps(r0[rng]['syncs_by_op_line_per_step'], indent=1), '```', '']
    if 'bench_method' in agg:
        L += ["## Bench-pack method with main's sim/ (0.2 s, float64 eager, setup included, median of 3)", '',
              f"rng='torch': {_mmm(agg['bench_method']['torch'], '{:.2f}')} ms/step; "
              f"rng='numpy': {_mmm(agg['bench_method']['numpy'], '{:.2f}')} ms/step; "
              f"2026-11-16 bench pack: 3.327 ms/step (rng='torch').", '']
    return '\n'.join(L)


def one_run(a, devices, only, model):
    run_res = dict(micro={}, sim={})
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
        run_res['micro'][d] = m
        print(f'micro {d}:', json.dumps({k: {kk: (vv['median'] if isinstance(vv, dict) and 'median' in vv
                                                  else vv) for kk, vv in v.items()} for k, v in m.items()}),
              flush=True)
    if model is not None:
        if '6' in only or '7' in only:
            for d in devices:
                run_res['sim'][d] = sim_section(model, d, a.steps, a.reps, a.stim, a.warm_steps)
        if 'bench' in only and 'cuda' in devices:
            run_res['bench_method'] = bench_method_sim(model, 'cuda', a.stim)
    return run_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', default=str(ROOT / 'results' / 'w4' / 'cns_bundle_calib.npz'))
    ap.add_argument('--devices', default='cuda,cpu')
    ap.add_argument('--only', default='1,2,3,4,5,6,7,bench')
    ap.add_argument('--runs', type=int, default=5, help='full repeats of the whole suite, aggregated')
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
    res = dict(tag=tag, env=env_info(), args=vars(a), runs=[])
    print(json.dumps(res['env'], indent=1), flush=True)
    model = None
    if only & {'6', '7', 'bench'}:
        model = load_bundle(a.bundle)
        res['bundle'] = model['info']
        res['bundle_path'] = a.bundle
        res['bundle_sizes'] = dict(n=int(model['W'].shape[0]), nnz=int(model['W'].nnz),
                                   n_stim={w: int(len(v)) for w, v in model['stim'].items()},
                                   n_gap=int(len(model['gap']['idx'])),
                                   n_presyn=int(len(model['presyn']['idx'])),
                                   n_kc=int(len(model['apl']['kc_idx'])))
    for i in range(a.runs):
        print(f'=== run {i + 1}/{a.runs}', flush=True)
        res['runs'].append(one_run(a, devices, only, model))
    res['agg'] = aggregate(res)
    for b in res['agg']['budget']:
        print('budget', b['rng'], 'pred(agg units)', b['predicted_ms_agg_units'],
              'pred per run', b['predicted_ms_per_run']['median'], 'steady', b['steady_ms']['median'],
              'ratio per run', b['ratio_pred_over_steady_per_run'], flush=True)
    (outdir / f'{tag}_overhead.json').write_text(json.dumps(res, indent=1))
    (outdir / f'{tag}_overhead.md').write_text(to_md(res))
    print(f'saved {outdir / tag}_overhead.json and .md', flush=True)


if __name__ == '__main__':
    main()
