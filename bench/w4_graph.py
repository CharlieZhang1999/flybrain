#!/usr/bin/env python
"""Week 4 T5b: CUDA graph + GPU timing of the sync-free step (sim/lif_torch_fast.py).

Graph replay (graph_trial): sim/lif_torch_fast._Sim.step is captured with
torch.cuda.graph and replayed once per step. The step already works on static
shapes with the step counter as a device tensor (t_dev, incremented inside the
graph), so the captured graph is replayed unchanged for every step. The only
host-side branch in step() is which drives are active at step t (block
windows; the bundle's looming trial has one drive that is always on), so one
graph is captured per active-drive combination, lazily, the first time that
combination occurs. Before the first capture, 3 real steps run eagerly on a
side stream (the warm-up the PyTorch docs ask for; they are steps of the trial,
not extra steps). The torch generator is registered with every graph
(CUDAGraph.register_generator_state) so that each replay draws fresh numbers
and advances the generator as eager calls would. Without it, capture fails
with "Attempt to increase offset for a CUDA generator not in capture mode".
Docs: PyTorch "CUDA semantics" notes, CUDA Graphs > Constraints
(https://docs.pytorch.org/docs/stable/notes/cuda.html#capture-constraints):
CUDA RNG ops are permitted, also with several torch.Generator instances, and
Generator.get_state / set_state must not be used during capture
(graphsafe_get_state / graphsafe_set_state instead; this script uses neither).
The docs page does not describe register_generator_state itself (its
docstring in torch 2.11 is only the signature); that it is needed for a
non-default generator, and sufficient, is shown here by the capture error
without it and by graph == eager spike counts with it.

Rows (looming, seed 0, rng='torch', caps from bench/w4_caps.py):
  ref eager     cuda float64  sim/lif_torch.run_trial (steps timed via lif_torch._decay, T1 method)
  fast eager    cuda float64  lif_torch_fast, _Sim.advance per step
  fast graph    cuda float64  graph_trial
  fast eager    cuda float64, default caps (spike_cap = 4096, syn_cap = degree bound)
  fast eager / fast graph cuda float32
  fast eager    cpu  float64
  ref eager     cpu  float64  (re-measures the CPU baseline, 0.6 ms in T1)

Timing, per row and run: one trial of --warm + --steps steps from t = 0. After
the warm steps the device is synchronized; the window ends with a synchronize.
Window ms/step = synchronized host wall / steps (stalls included).
Per-step durations: cuda rows record a CUDA event at the start of every window
step (event to event = the step on the GPU timeline, right also when the host
runs ahead of the GPU, as in the sync-free rows); the spans are rescaled so that
they sum to the host wall of the same window, because CUDA event times on this
WSL machine read 4-6% high against the host clock. cpu rows: host
perf_counter. Steady = steps <= 5 ms (T1/T3 rule), stall fraction separate;
ms/step = steady mean, as in T1/T3. cuda rows run a second trial without
events for the window ms/step without the event-record overhead.
Peak GPU memory: torch.cuda.max_memory_allocated over setup + trial, minus
the memory allocated before the row (the bundle on the host is not counted).
Rows marked deterministic run under torch.use_deterministic_algorithms(True).

Checks: overflow flag of every fast row (must be 0); graph vs eager counts on
the same seed in every run (must be identical); --parity: full trials
(P['t_run'] = 1 s, seeds 0-2): graph vs eager, fast vs reference on GPU (rate
correlation over active neurons, share of differing neurons) and reference vs
reference; sync check: the fast eager steady loop under
torch.cuda.set_sync_debug_mode('error') plus a positive control (the
reference must raise under the same mode).

Usage (repo root):
  python bench/w4_graph.py --bundle ~/flybench/flybrain/results/w4/cns_bundle_calib.npz [--runs 5]
Output: perf/results/w4/<host>_<timestamp>_graph.json and .md
"""
import argparse
import gc
import json
import socket
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'bench'))
from sim import lif_torch, lif_torch_fast  # noqa: E402
from sim.lif_numpy import P  # noqa: E402
import w4_overhead as t1  # noqa: E402  (bundle loader, StepClock, env_info)

pc = time.perf_counter
STALL_MS = t1.STALL_MS
CAPS = dict(spike_cap=60, syn_cap=44483)   # bench/w4_caps.py, 1.5 x max over 3 seeds (see perf/week04_day6.md)
WARMUP = 3                                  # eager steps on a side stream before the first capture


# ---------------------------------------------------------------- graph replay
def _key(s, t):
    return tuple(s0 <= t < s1 for s0, s1, _, _ in s.drive)


def graph_trial(s, stop=None, on_step=None):
    """Run the steps of a prepared lif_torch_fast._Sim (rng='torch', cuda) from
    s.t_begin to `stop` (default s.nsteps): WARMUP eager steps on a side stream,
    then one captured graph per active-drive combination, replayed per step.
    Returns the dict of graphs (keyed by active-drive tuple)."""
    assert s.rng == 'torch', "graph replay needs rng='torch' (rng='numpy' draws on the host)"
    stop = s.nsteps if stop is None else stop
    t = s.t_begin
    cur = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    side.wait_stream(cur)
    with torch.cuda.stream(side):
        for _ in range(min(WARMUP, stop - t)):
            if on_step is not None:
                on_step(t)
            s.step(t)
            t += 1
    cur.wait_stream(side)
    graphs = {}
    while t < stop:
        k = _key(s, t)
        if k not in graphs:
            g = torch.cuda.CUDAGraph()
            g.register_generator_state(s.t_gen)
            with torch.cuda.graph(g):
                s.step(t)           # recorded, not executed
            graphs[k] = g
        if on_step is not None:
            on_step(t)
        graphs[k].replay()
        t += 1
    return graphs


def eager_trial(s, stop=None, on_step=None):
    stop = s.nsteps if stop is None else stop
    for t in range(s.t_begin, stop):
        if on_step is not None:
            on_step(t)
        s.advance(t)


# ---------------------------------------------------------------- runs
def _params(nsteps):
    p = dict(P); p['t_run'] = nsteps * P['dt']
    return p


def model_kw(model):
    return dict(apl=model['apl'], gap=model['gap'], presyn=model['presyn'])


def prepare(model, stim, nsteps, device, dtype, caps, seed=0):
    return lif_torch_fast._prepare(model['W'], model['stim'][stim], params=_params(nsteps), seed=seed,
                                   device=device, dtype=dtype, rng='torch', **model_kw(model), **caps)


def _sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def _clean():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _summary(d, window_s, steps):
    """d: per-step durations (s); window_s: synchronized wall of the window."""
    thr = STALL_MS / 1e3
    steady = [x for x in d if x <= thr]
    stalls = [x for x in d if x > thr]
    return dict(steps=steps,
                steady_mean_ms=round(statistics.fmean(steady) * 1e3, 4) if steady else None,
                steady_median_ms=round(statistics.median(steady) * 1e3, 4) if steady else None,
                window_ms_per_step=round(window_s / steps * 1e3, 4),
                stall_fraction=round(len(stalls) / len(d), 4),
                stall_median_ms=round(statistics.median(stalls) * 1e3, 3) if stalls else None)


def time_row(model, stim, row, warm, steps, events):
    """One timed trial for a row. Returns timing, peak memory, overflow, counts.
    events (cuda): a CUDA event at the start of every window step; per-step
    duration = event to event on the GPU timeline (also right when the host
    runs ahead of the GPU). Otherwise host perf_counter stamps."""
    impl, device, dtype, mode, caps = row['impl'], row['device'], row['dtype'], row['mode'], row['caps']
    nsteps = warm + steps
    _clean()
    base = torch.cuda.memory_allocated() if device == 'cuda' else 0
    stamps = []
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(steps + 1)] if events else None

    def on_step(t):
        if t == warm:
            _sync(device)
        if t >= warm:
            stamps.append(pc())
            if ev is not None:
                ev[t - warm].record()

    t_setup = pc()
    det0 = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(row.get('det', False))
    try:
        out, counts = _time_body(model, stim, row, warm, steps, nsteps, on_step, ev, stamps, t_setup, base)
    finally:
        torch.use_deterministic_algorithms(det0)
    return out, counts


def _time_body(model, stim, row, warm, steps, nsteps, on_step, ev, stamps, t_setup, base):
    impl, device, dtype, mode, caps = row['impl'], row['device'], row['dtype'], row['mode'], row['caps']
    if impl == 'ref':
        class Hook:      # StepClock calls on_step() before each step's _decay; count steps
            k = 0

            def __call__(self):
                on_step(self.k)
                self.k += 1
        with t1.StepClock(on_step=Hook()):
            counts = lif_torch.run_trial(model['W'], model['stim'][stim], params=_params(nsteps), seed=0,
                                         device=device, dtype=dtype, rng='torch', **model_kw(model))
        if ev is not None:
            ev[steps].record()
        _sync(device)
        t_end = pc()
        ovf, ngraphs = None, None
        counts = np.asarray(counts)
    else:
        s = prepare(model, stim, nsteps, device, dtype, caps)
        if mode == 'graph':
            ngraphs = len(graph_trial(s, on_step=on_step))
        else:
            eager_trial(s, on_step=on_step)
            ngraphs = None
        if ev is not None:
            ev[steps].record()
        _sync(device)
        t_end = pc()
        ovf = int(bool(s.ovf))
        counts = s.counts.cpu().numpy()
        del s
    ratio = None
    if ev is not None:
        # CUDA event times on this WSL box run 4-6% fast against the host clock (a 3 s host sleep
        # between two events reads 3.11-3.18 s), so the per-step spans are rescaled to the host
        # clock: sum of spans = synchronized host wall of the window (same trial).
        d = [ev[i].elapsed_time(ev[i + 1]) / 1e3 for i in range(steps)]
        ratio = sum(d) / (t_end - stamps[0])
        d = [x / ratio for x in d]
    else:
        d = [stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)] + [t_end - stamps[-1]]
    out = _summary(d, t_end - stamps[0], steps)
    out.update(timer='cuda events rescaled to host clock' if ev is not None else 'host perf_counter',
               event_over_host_clock=None if ratio is None else round(ratio, 4),
               trial_wall_s=round(t_end - t_setup, 3), overflow=ovf, graphs=ngraphs,
               spikes=int(counts.sum()),
               peak_mem_mb=(round((torch.cuda.max_memory_allocated() - base) / 2 ** 20, 1) if device == 'cuda' else None),
               mem_before_mb=(round(base / 2 ** 20, 1) if device == 'cuda' else None))
    return out, counts


def rows_def(a):
    f64, f32 = torch.float64, torch.float32
    R = [dict(name='ref eager cuda f64', impl='ref', device='cuda', dtype=f64, mode='eager', caps={}),
         dict(name='fast eager cuda f64', impl='fast', device='cuda', dtype=f64, mode='eager', caps=CAPS),
         dict(name='fast graph cuda f64', impl='fast', device='cuda', dtype=f64, mode='graph', caps=CAPS),
         dict(name='fast eager cuda f64 default caps', impl='fast', device='cuda', dtype=f64, mode='eager',
              caps={}, steps=a.default_caps_steps),
         dict(name='fast eager cuda f32', impl='fast', device='cuda', dtype=f32, mode='eager', caps=CAPS),
         dict(name='fast graph cuda f32', impl='fast', device='cuda', dtype=f32, mode='graph', caps=CAPS),
         dict(name='fast graph cuda f64 deterministic', impl='fast', device='cuda', dtype=f64, mode='graph',
              caps=CAPS, det=True, steps=a.default_caps_steps),
         dict(name='fast graph cuda f32 deterministic', impl='fast', device='cuda', dtype=f32, mode='graph',
              caps=CAPS, det=True, steps=a.default_caps_steps),
         dict(name='fast eager cpu f64', impl='fast', device='cpu', dtype=f64, mode='eager', caps=CAPS),
         dict(name='ref eager cpu f64', impl='ref', device='cpu', dtype=f64, mode='eager', caps={})]
    if a.only:
        R = [r for r in R if r['name'] in a.only]
    return R


def one_run(model, a):
    res, cnt = {}, {}
    for row in rows_def(a):
        steps = row.get('steps', a.steps)
        cuda = row['device'] == 'cuda'
        r, c = time_row(model, a.stim, row, a.warm, steps, events=cuda)
        if cuda:     # second trial without events: window wall free of event-record overhead
            r2, c2 = time_row(model, a.stim, row, a.warm, steps, events=False)
            r['window_ms_per_step_no_events'] = r2['window_ms_per_step']
            r['same_counts_both_trials'] = bool((c == c2).all())
            r['overflow'] = None if r['overflow'] is None else max(r['overflow'], r2['overflow'])
        else:
            r['window_ms_per_step_no_events'] = r['window_ms_per_step']
        res[row['name']] = r
        cnt[row['name']] = c
        print(f"  {row['name']:34s} steady {r['steady_mean_ms']} ms  window {r['window_ms_per_step']} "
              f"/ no-ev {r['window_ms_per_step_no_events']} ms  stall {r['stall_fraction']}  "
              f"peak {r['peak_mem_mb']} MB  ovf {r['overflow']}", flush=True)
    checks = {}
    for dt_ in ('f64', 'f32'):
        e, g = cnt.get(f'fast eager cuda {dt_}'), cnt.get(f'fast graph cuda {dt_}')
        if e is not None and g is not None:
            checks[f'graph_vs_eager_{dt_}_neurons_differing'] = int((e != g).sum())
    return dict(rows=res, checks=checks)


# ---------------------------------------------------------------- parity
def _corr(a, b):
    act = (a > 0) | (b > 0)
    x, y = a[act].astype(float), b[act].astype(float)
    return dict(active=int(act.sum()), rate_corr=round(float(np.corrcoef(x, y)[0, 1]), 6),
                neurons_differing=int((a != b).sum()),
                share_differing_of_active=round(float((a != b).sum() / act.sum()), 4),
                spikes=[int(a.sum()), int(b.sum())])


def parity(model, stim, seeds, nsteps):
    out = []
    for seed in seeds:
        _clean()
        p = _params(nsteps)
        ref = lambda: np.asarray(lif_torch.run_trial(  # noqa: E731
            model['W'], model['stim'][stim], params=p, seed=seed, device='cuda',
            dtype=torch.float64, rng='torch', **model_kw(model)))
        r1, r2 = ref(), ref()
        row = dict(seed=seed, steps=nsteps)
        for dt_name, dt_ in (('f64', torch.float64), ('f32', torch.float32)):
            se = prepare(model, stim, nsteps, 'cuda', dt_, CAPS, seed=seed)
            eager_trial(se)
            sg = prepare(model, stim, nsteps, 'cuda', dt_, CAPS, seed=seed)
            graph_trial(sg)
            torch.cuda.synchronize()
            e, g = se.counts.cpu().numpy(), sg.counts.cpu().numpy()
            row[f'overflow_{dt_name}'] = [int(bool(se.ovf)), int(bool(sg.ovf))]
            row[f'graph_vs_eager_{dt_name}_neurons_differing'] = int((e != g).sum())
            row[f'spikes_fast_{dt_name}'] = int(e.sum())
            row[f'fast_{dt_name}_vs_ref_f64'] = _corr(e, r1)
            del se, sg
        # float32 with torch.use_deterministic_algorithms(True): index_add_ without float atomics
        torch.use_deterministic_algorithms(True)
        try:
            se = prepare(model, stim, nsteps, 'cuda', torch.float32, CAPS, seed=seed)
            eager_trial(se)
            sg = prepare(model, stim, nsteps, 'cuda', torch.float32, CAPS, seed=seed)
            graph_trial(sg)
            torch.cuda.synchronize()
            e, g = se.counts.cpu().numpy(), sg.counts.cpu().numpy()
            row['overflow_f32_det'] = [int(bool(se.ovf)), int(bool(sg.ovf))]
            row['graph_vs_eager_f32_det_neurons_differing'] = int((e != g).sum())
            row['fast_f32_det_vs_ref_f64'] = _corr(e, r1)
            del se, sg
        finally:
            torch.use_deterministic_algorithms(False)
        row['ref_vs_ref'] = _corr(r1, r2)
        out.append(row)
        print('  parity', json.dumps(row), flush=True)
    return out


# ---------------------------------------------------------------- sync check
def ops_per_step(s, t0, n):
    from torch.utils._python_dispatch import TorchDispatchMode
    cnt = dict(aten_ops=0, kernel_ops=0)

    class M(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            cnt['aten_ops'] += 1
            cnt['kernel_ops'] += t1._op_class(func, args, out)[1] == 'kernel'
            return out
    with M():
        for t in range(t0, t0 + n):
            s.advance(t)
    torch.cuda.synchronize()
    return {k: round(v / n, 2) for k, v in cnt.items()}


def sync_check(model, stim, warm=1000, steps=200):
    """Steady loop of the fast eager path under set_sync_debug_mode('error');
    positive control: one reference step under the same mode must raise.
    Then aten ops per steady step of the fast step (TorchDispatchMode; 'kernel'
    = ops with a cuda output, T1's classification), for the launch-bound
    question. (Graph capture of the step, done in every run, would also fail on
    a sync: CPU-GPU syncs are not allowed while a stream is capturing.)"""
    out = {}
    n_ops = 50
    s = prepare(model, stim, warm + steps + n_ops, 'cuda', torch.float64, CAPS)
    eager_trial(s, stop=warm)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode('error')
    try:
        for t in range(warm, warm + steps):
            s.advance(t)
        out['fast_steady_loop'] = dict(steps=steps, raised=None, passed=True)
    except Exception as ex:  # noqa: BLE001
        out['fast_steady_loop'] = dict(steps=steps, raised=f'{type(ex).__name__}: {ex}',
                                       trace=traceback.format_exc()[-2000:], passed=False)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()
    out['ops_per_step'] = ops_per_step(s, warm + steps, n_ops)
    out['overflow'] = int(bool(s.ovf))
    del s
    torch.cuda.set_sync_debug_mode('error')
    try:
        lif_torch.run_trial(model['W'], model['stim'][stim], params=_params(2), seed=0, device='cuda',
                            dtype=torch.float64, rng='torch', **model_kw(model))
        out['positive_control_reference'] = dict(raised=None, passed=False)
    except Exception as ex:  # noqa: BLE001
        out['positive_control_reference'] = dict(raised=f'{type(ex).__name__}: {str(ex)[:200]}', passed=True)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    out['passed'] = out['fast_steady_loop']['passed'] and out['positive_control_reference']['passed']
    print('  sync check', json.dumps({k: v for k, v in out.items() if k != 'trace'}), flush=True)
    return out


# ---------------------------------------------------------------- aggregate / report
def _rng(xs, nd=4):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return dict(median=round(statistics.median(xs), nd), min=round(min(xs), nd), max=round(max(xs), nd),
                max_over_min=round(max(xs) / min(xs), 2) if min(xs) > 0 else None, runs=xs)


def aggregate(res):
    runs = res['runs']
    agg = dict(rows={}, checks={}, variability=[])
    for name in runs[0]['rows']:
        rs = [r['rows'][name] for r in runs]
        a = {k: _rng([x[k] for x in rs]) for k in ('steady_mean_ms', 'steady_median_ms', 'window_ms_per_step', 'window_ms_per_step_no_events',
                                                    'stall_fraction', 'peak_mem_mb', 'mem_before_mb',
                                                    'event_over_host_clock')}
        a['overflow_total'] = sum(x['overflow'] or 0 for x in rs) if rs[0]['overflow'] is not None else None
        a['same_counts_both_trials'] = all(x.get('same_counts_both_trials', True) for x in rs)
        a['timer'] = rs[0]['timer']
        a['graphs'] = rs[0]['graphs']
        a['steps'] = rs[0]['steps']
        agg['rows'][name] = a
        for k in ('steady_mean_ms', 'window_ms_per_step_no_events', 'stall_fraction'):
            v = a[k]
            if v and v['max'] > 0 and (v['max_over_min'] is None or v['max_over_min'] > 2):
                agg['variability'].append(dict(what=f'{name} {k}', median=v['median'], min=v['min'],
                                               max=v['max'], max_over_min=v['max_over_min']))
    for k in runs[0]['checks']:
        agg['checks'][k] = [r['checks'][k] for r in runs]
    return agg


def _m(a, f='{:.3f}'):
    if a is None:
        return '-'
    return f"{f.format(a['median'])} [{f.format(a['min'])}-{f.format(a['max'])}]"


def to_md(res):
    agg = res['agg']
    L = [f"# Week 4 T5b: CUDA graph of the sync-free step: {res['tag']} ({len(res['runs'])} runs)", '',
         f"Looming, seed 0, rng='torch'. Window: {res['args']['steps']} steps after {res['args']['warm']} warm "
         f"steps, one trial per row per run. Caps: spike_cap={CAPS['spike_cap']}, syn_cap={CAPS['syn_cap']} "
         "(except the default-caps row). Every value: median across runs [min-max across runs].", '',
         'steady ms/step: mean of the steps <= 5 ms (cuda: CUDA event to event, rescaled to the host clock; '
         'cpu: host clock). '
         'Window ms/step: synchronized wall / steps, stalls included, with and without the per-step events '
         '(cuda; a second trial).', '',
         '| row | steady ms/step | stall fraction | window ms/step (events) | window ms/step (no events) '
         '| peak GPU MB above baseline | overflow | same counts both trials |',
         '|---|---|---|---|---|---|---|---|']
    for name, a in agg['rows'].items():
        steps = '' if a['steps'] == res['args']['steps'] else f" ({a['steps']} steps)"
        L.append(f"| {name}{steps} | {_m(a['steady_mean_ms'])} | {_m(a['stall_fraction'], '{:.4f}')} | "
                 f"{_m(a['window_ms_per_step'])} | {_m(a['window_ms_per_step_no_events'])} | "
                 f"{_m(a['peak_mem_mb'], '{:.0f}')} | "
                 f"{'-' if a['overflow_total'] is None else a['overflow_total']} | "
                 f"{a['same_counts_both_trials']} |")
    L += ['', '## Checks', '', '```', json.dumps(agg['checks'], indent=1), '```', '']
    if res.get('sync_check'):
        sc = {k: v for k, v in res['sync_check'].items()}
        if 'trace' in sc.get('fast_steady_loop', {}):
            sc['fast_steady_loop'] = {k: v for k, v in sc['fast_steady_loop'].items() if k != 'trace'}
        L += ['## Sync check', '', '```', json.dumps(sc, indent=1), '```', '']
    if res.get('parity'):
        L += ['## Parity (full trials)', '', '```', json.dumps(res['parity'], indent=1), '```', '']
    L += ['## Variability (> 2x across runs)', '']
    L += [f"- {v['what']}: {v['median']} [{v['min']}-{v['max']}], "
          f"{'min 0' if v['max_over_min'] is None else str(v['max_over_min']) + 'x'}"
          for v in agg['variability']] or ['- none']
    L += ['', '```', json.dumps(res['env'], indent=1), '```', '']
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', default=str(ROOT / 'results' / 'w4' / 'cns_bundle_calib.npz'))
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--warm', type=int, default=1000)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--default-caps-steps', type=int, default=200)
    ap.add_argument('--stim', default='looming')
    ap.add_argument('--only', nargs='*', help='row names to run (default: all)')
    ap.add_argument('--parity-seeds', type=int, nargs='*', default=[0, 1, 2])
    ap.add_argument('--parity-steps', type=int, default=int(round(P['t_run'] / P['dt'])))
    ap.add_argument('--no-parity', action='store_true')
    ap.add_argument('--no-sync-check', action='store_true')
    ap.add_argument('--sync-check-only', action='store_true')
    ap.add_argument('--out', default=str(ROOT / 'perf' / 'results' / 'w4'))
    a = ap.parse_args()
    assert torch.cuda.is_available(), 'needs CUDA'
    model = t1.load_bundle(a.bundle)
    tag = f'{socket.gethostname()}_{time.strftime("%Y%m%d-%H%M")}'
    res = dict(tag=tag, env=t1.env_info(), args=vars(a), caps=CAPS, warmup_eager_steps=WARMUP,
               torch_threads=torch.get_num_threads(), runs=[])
    print(json.dumps(res['env']), flush=True)
    if not a.no_sync_check:
        res['sync_check'] = sync_check(model, a.stim)
        if a.sync_check_only:
            outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
            (outdir / f'{tag}_synccheck.json').write_text(json.dumps(res, indent=1))
            print(f'saved {outdir / tag}_synccheck.json', flush=True)
            sys.exit(0 if res['sync_check']['passed'] else 4)
    for i in range(a.runs):
        print(f'=== run {i + 1}/{a.runs}', flush=True)
        res['runs'].append(one_run(model, a))
    res['agg'] = aggregate(res)
    if not a.no_parity:
        res['parity'] = parity(model, a.stim, a.parity_seeds, a.parity_steps)
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f'{tag}_graph.json').write_text(json.dumps(res, indent=1))
    (outdir / f'{tag}_graph.md').write_text(to_md(res))
    print(to_md(res), flush=True)
    print(f'saved {outdir / tag}_graph.json and .md', flush=True)


if __name__ == '__main__':
    main()
