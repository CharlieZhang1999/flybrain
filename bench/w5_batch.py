#!/usr/bin/env python
"""Week 5 P1: B trials in one CUDA graph (sim/lif_torch_batch.py), scaling B = 1..32.

Rows (looming bundle, gap + APL + presyn, rng='torch', caps spike_cap = 60 and
syn_cap = 44,483 per trial (bench/w4_graph.py CAPS), seeds 0..B-1, every trial
the looming drive):
  batch graph cuda f64 B=k   lif_torch_batch.graph_loop, k = 1, 2, 4, 8, 16, 32
  batch graph cuda f32 B=k   same in float32 (information only)
  fast graph cuda f64        single trial, bench/w4_graph.graph_trial (Week 4 T5b path)
  ref eager cpu f64          single trial, sim/lif_torch.run_trial on the PC CPU

Timing (Day 6 method, bench/w4_graph.py): per row and run, one trial of --warm
+ --steps steps from t = 0; after the warm steps the device is synchronized.
Per-step durations: a CUDA event at the start of every window step, spans
rescaled so that they sum to the synchronized host wall of the same window
(CUDA event times on this WSL machine read high against the host clock; the
ratio is recorded per row). Clock = CUDA events rescaled to the host clock,
chosen so that steady ms/step and all-steps ms/step are on one clock and
comparable with Week 4 Day 6 (0.49 ms/step for the fast graph). Steady = steps
<= 5 ms; stall share = share of steps > 5 ms; all-steps ms/step = synchronized
wall / steps from a second trial without per-step events. cpu row: host
perf_counter. Throughput per B from the all-steps ms/step:
trials per wall-second = B / (ms/step x 10,000 steps of a 1 s trial) and
bio-seconds per wall-second = B x 0.1 ms / ms/step (same numbers, two units);
also from the steady ms/step.
Peak GPU memory: torch.cuda.max_memory_allocated over setup + trial minus the
memory allocated before the row.

Sim-busy check: before every row, WSL `ps` (python processes other than this
one and its parents; the system's unattended-upgrade daemon excluded) and
Windows Get-Process python/pythonw (via powershell.exe). If anything is found
the row waits (polling every --busy-poll s) and is timed only when nothing
runs; every check is recorded in the JSON.

Checks:
  ops per step (eager step, TorchDispatchMode; 'kernel' = ops with a CUDA
    output, T1 classification) per B.
  phases: GPU execution time per phase per step, eager, gated (before every
    phase torch.cuda._sleep(--gate-ms) is enqueued, then an event; the host
    enqueues the phase while the GPU sleeps, so the event span is the phase's
    GPU time, not paced by the host; raw CUDA event clock), --phase-steps steps
    per B.
  A2 batch invariance: full looming trials (P['t_run'] = 1 s), B = 8 seeds
    0..7 and the same seeds reversed, B = 1 for seeds 0, 3, 7 (slots 0/7, 3/4,
    7/0): identical counts.
  A3 statistical parity: the B = 8 batch vs sim/lif_torch.run_trial on CUDA,
    same seeds: rate correlation on active neurons, share of differing neurons.
  A4 graph + syncs: B = 8 with different blocks per trial (looming / vinegar /
    sugar, different windows and rates, overlapping blocks): graphs captured,
    replays per step; set_sync_debug_mode('error') over 200 steady steps of
    the eager batch and of the graph replays; positive control: the reference
    raises. Also draws='always' (one graph). Overflow of every timed trial.

Resumable: results go to <out>/<tag>_batch.partial.json after every run;
--resume <that file> continues with the next run.

Usage (repo root):
  python bench/w5_batch.py --bundle ~/flybench/flybrain/results/w4/cns_bundle_calib.npz [--runs 5]
Output: perf/results/w5/<host>_<timestamp>_batch.json and .md
"""
import argparse
import gc
import json
import os
import socket
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'bench'))
from sim import lif_torch, lif_torch_batch, lif_torch_fast  # noqa: E402
from sim.lif_numpy import P  # noqa: E402
import w4_overhead as t1  # noqa: E402  (bundle loader, env_info, op classification)
import w4_graph as w4g  # noqa: E402  (Week 4 fast graph, _summary, _corr)
from w4_anatomy import calibrate_sleep  # noqa: E402  (cycles for torch.cuda._sleep)

pc = time.perf_counter
CAPS = w4g.CAPS
BS = (1, 2, 4, 8, 16, 32)
STEPS_PER_TRIAL = int(round(P['t_run'] / P['dt']))      # 10,000 steps = 1 s


# ---------------------------------------------------------------- sim-busy check
def busy_check():
    me = {os.getpid(), os.getppid()}
    wsl = []
    try:
        ps = subprocess.run(['ps', '-eo', 'pid=,ppid=,args='], capture_output=True, text=True, timeout=20).stdout
        for line in ps.splitlines():
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            pid, ppid, args = int(parts[0]), int(parts[1]), parts[2]
            if 'python' not in args or pid in me or 'unattended-upgrade' in args:
                continue
            if args.startswith(('ps ', 'bash ', '/bin/bash ')):
                continue
            wsl.append(f'{pid} {args[:160]}')
    except Exception as ex:  # noqa: BLE001
        wsl.append(f'check failed: {ex}')
    win, win_ok = [], True
    try:
        cmd = ('Get-Process python,pythonw -ErrorAction SilentlyContinue | '
               'ForEach-Object { "{0} {1}" -f $_.Id, $_.Path }')
        r = subprocess.run(['powershell.exe', '-NoProfile', '-Command', cmd], capture_output=True,
                           text=True, timeout=60)
        win = [x.strip() for x in r.stdout.splitlines() if x.strip()]
    except Exception as ex:  # noqa: BLE001
        win_ok = False
        win = [f'check failed: {ex}']
    return dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), wsl_python=wsl, windows_python=win,
                windows_check_ok=win_ok, busy=bool(wsl) or (win_ok and bool(win)))


def wait_idle(poll, log):
    checks = [busy_check()]
    while checks[-1]['busy']:
        print(f"  busy: {checks[-1]['wsl_python']} {checks[-1]['windows_python']}; waiting {poll} s", flush=True)
        log.append(checks[-1])
        time.sleep(poll)
        checks.append(busy_check())
    return checks[-1], len(checks) - 1


# ---------------------------------------------------------------- rows
def model_kw(model):
    return dict(apl=model['apl'], gap=model['gap'], presyn=model['presyn'])


def prep_batch(model, stim, B, nsteps, dtype, seeds=None, blocks_list=None, draws='active', record=False):
    seeds = list(range(B)) if seeds is None else seeds
    return lif_torch_batch._prepare(model['W'], seeds, blocks_list, exc_idx=model['stim'][stim],
                                    params=w4g._params(nsteps), device='cuda', dtype=dtype, rng='torch',
                                    draws=draws, record=record, **model_kw(model), **CAPS)


def rows_def(a):
    R = [dict(name=f'batch graph cuda f64 B={B}', impl='batch', B=B, dtype=torch.float64) for B in a.bs]
    R += [dict(name=f'batch graph cuda f32 B={B}', impl='batch', B=B, dtype=torch.float32) for B in a.bs_f32]
    R += [dict(name='fast graph cuda f64', impl='fast', B=1, dtype=torch.float64),
          dict(name='ref eager cpu f64', impl='ref', B=1, dtype=torch.float64)]
    if a.only:
        R = [r for r in R if r['name'] in a.only]
    return R


def time_row(model, a, row, events):
    warm, steps = a.warm, a.steps
    nsteps = warm + steps
    w4g._clean()
    cuda = row['impl'] != 'ref'
    base = torch.cuda.memory_allocated() if cuda else 0
    stamps = []
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(steps + 1)] if events else None

    def on_step(t):
        if t == warm:
            if cuda:
                torch.cuda.synchronize()
        if t >= warm:
            stamps.append(pc())
            if ev is not None:
                ev[t - warm].record()

    t_setup = pc()
    ovf, ngraphs, replays = None, None, None
    if row['impl'] == 'batch':
        s = prep_batch(model, a.stim, row['B'], nsteps, row['dtype'])
        ngraphs = len(lif_torch_batch.graph_loop(s, on_step=on_step))
        replays = s.replays
        if ev is not None:
            ev[steps].record()
        torch.cuda.synchronize()
        t_end = pc()
        ovf = s.ovf.cpu().numpy().astype(int).tolist()
        counts = s.counts.cpu().numpy()
        del s
    elif row['impl'] == 'fast':
        s = w4g.prepare(model, a.stim, nsteps, 'cuda', row['dtype'], CAPS)
        ngraphs = len(w4g.graph_trial(s, on_step=on_step))
        if ev is not None:
            ev[steps].record()
        torch.cuda.synchronize()
        t_end = pc()
        ovf = [int(bool(s.ovf))]
        counts = s.counts.cpu().numpy()[None]
        del s
    else:
        class Hook:      # StepClock calls on_step() before each step's _decay
            k = 0

            def __call__(self):
                on_step(self.k)
                self.k += 1
        with t1.StepClock(on_step=Hook()):
            counts = lif_torch.run_trial(model['W'], model['stim'][a.stim], params=w4g._params(nsteps), seed=0,
                                         device='cpu', dtype=row['dtype'], rng='torch', **model_kw(model))
        t_end = pc()
        counts = np.asarray(counts)[None]
    ratio = None
    if ev is not None:
        d = [ev[i].elapsed_time(ev[i + 1]) / 1e3 for i in range(steps)]
        ratio = sum(d) / (t_end - stamps[0])
        d = [x / ratio for x in d]
    else:
        d = [stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)] + [t_end - stamps[-1]]
    out = w4g._summary(d, t_end - stamps[0], steps)
    out['median_ms_all_steps'] = round(statistics.median(d) * 1e3, 4)   # no 5 ms threshold
    out.update(timer='cuda events rescaled to host clock' if ev is not None else 'host perf_counter',
               event_over_host_clock=None if ratio is None else round(ratio, 4),
               trial_wall_s=round(t_end - t_setup, 3), overflow=ovf, graphs=ngraphs, replays=replays,
               spikes=int(counts.sum()),
               peak_mem_mb=round((torch.cuda.max_memory_allocated() - base) / 2 ** 20, 1) if cuda else None)
    return out, counts


def one_run(model, a, busy_log):
    res = {}
    for row in rows_def(a):
        chk, waits = wait_idle(a.busy_poll, busy_log)
        cuda = row['impl'] != 'ref'
        r, c = time_row(model, a, row, events=cuda)
        if cuda:
            r2, c2 = time_row(model, a, row, events=False)
            r['window_ms_per_step_no_events'] = r2['window_ms_per_step']
            r['same_counts_both_trials'] = bool((c == c2).all())
            r['overflow'] = [max(x, y) for x, y in zip(r['overflow'], r2['overflow'])]
            r['peak_mem_mb'] = max(r['peak_mem_mb'], r2['peak_mem_mb'])
        else:
            r['window_ms_per_step_no_events'] = r['window_ms_per_step']
        after = busy_check()
        r['busy_check_before'] = chk
        r['busy_waits_before'] = waits
        r['busy_check_after'] = after
        res[row['name']] = r
        print(f"  {row['name']:28s} steady {r['steady_mean_ms']} ms  all-steps {r['window_ms_per_step_no_events']} ms  "
              f"stall {r['stall_fraction']}  peak {r['peak_mem_mb']} MB  ovf {sum(r['overflow'] or [0])}  "
              f"busy {chk['busy']}/{after['busy']}", flush=True)
    return dict(rows=res, time=time.strftime('%Y-%m-%d %H:%M:%S'))


# ---------------------------------------------------------------- ops and phases
def ops_per_step(model, a, B, dtype=torch.float64, warm=200, n=20):
    from torch.utils._python_dispatch import TorchDispatchMode
    cnt = dict(aten_ops=0, kernel_ops=0)

    class M(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            cnt['aten_ops'] += 1
            cnt['kernel_ops'] += t1._op_class(func, args, out)[1] == 'kernel'
            return out
    s = prep_batch(model, a.stim, B, warm + n, dtype)
    lif_torch_batch.eager_loop(s, stop=warm)
    with M():
        lif_torch_batch.eager_loop(s, stop=warm + n)
    torch.cuda.synchronize()
    del s
    return {k: round(v / n, 2) for k, v in cnt.items()}


def phases(model, a, B, dtype=torch.float64, warm=1000):
    """GPU time per phase per step (gated; raw CUDA event clock)."""
    n = a.phase_steps
    s = prep_batch(model, a.stim, B, warm + n, dtype)
    lif_torch_batch.eager_loop(s, stop=warm)
    torch.cuda.synchronize()
    cycles, gate_measured = calibrate_sleep(a.gate_ms)
    marks = []

    def mark(name):
        e_end = torch.cuda.Event(enable_timing=True); e_end.record()
        torch.cuda._sleep(cycles)
        e_start = torch.cuda.Event(enable_timing=True); e_start.record()
        marks.append((name, e_end, e_start))
    steps = []
    for t in range(warm, warm + n):
        marks.clear()
        s.mark = mark
        s.advance(t)
        s.mark = None
        s._t_host = t + 1
        torch.cuda.synchronize()
        ph = {}
        for (name, _, st), (_, en, _) in zip(marks[:-1], marks[1:]):
            ph[name] = st.elapsed_time(en)
        steps.append(ph)
    del s
    names = list(steps[0])
    med = {k: round(statistics.median(p[k] for p in steps), 4) for k in names}
    med['total'] = round(statistics.median(sum(p.values()) for p in steps), 4)
    return dict(B=B, steps=n, gate_ms=round(gate_measured, 3), clock='raw CUDA events', median_ms=med)


# ---------------------------------------------------------------- A2 / A3 / A4
def full_batch(model, a, seeds, blocks_list=None, draws='active'):
    s = prep_batch(model, a.stim, len(seeds), STEPS_PER_TRIAL, torch.float64, seeds=seeds,
                   blocks_list=blocks_list, draws=draws)
    graphs = lif_torch_batch.graph_loop(s)
    torch.cuda.synchronize()
    out = dict(counts=s.counts.cpu().numpy().astype(np.int64), ovf=s.ovf.cpu().numpy().astype(int).tolist(),
               graphs=len(graphs), replays=s.replays, steps=s.nsteps)
    del s, graphs
    w4g._clean()
    return out


def parity(model, a):
    seeds = list(range(8))
    fwd = full_batch(model, a, seeds)
    rev = full_batch(model, a, seeds[::-1])
    a2 = []
    for sd in (0, 3, 7):
        one = full_batch(model, a, [sd])
        i_f, i_r = seeds.index(sd), 7 - seeds.index(sd)
        a2.append(dict(seed=sd, slots_in_B8=[i_f, i_r],
                       neurons_differing_B1_vs_B8_slot=[int((one['counts'][0] != fwd['counts'][i_f]).sum()),
                                                        int((one['counts'][0] != rev['counts'][i_r]).sum())],
                       spikes=int(one['counts'][0].sum()), overflow=one['ovf']))
    a2_all = [int((fwd['counts'][i] != rev['counts'][7 - i]).sum()) for i in range(8)]
    a3 = []
    for i, sd in enumerate(seeds):
        ref = np.asarray(lif_torch.run_trial(model['W'], model['stim'][a.stim], params=dict(P), seed=sd,
                                             device='cuda', dtype=torch.float64, rng='torch', **model_kw(model)))
        c = w4g._corr(fwd['counts'][i], ref)
        c['seed'] = sd
        a3.append(c)
        print('  A3', json.dumps(c), flush=True)
    out = dict(steps=STEPS_PER_TRIAL,
               A2=dict(per_seed=a2, fwd_vs_rev_neurons_differing=a2_all,
                       overflow=dict(fwd=fwd['ovf'], rev=rev['ovf']),
                       passed=all(max(x['neurons_differing_B1_vs_B8_slot']) == 0 for x in a2)),
               A3=dict(per_trial=a3, min_rate_corr=min(x['rate_corr'] for x in a3),
                       max_share_differing=max(x['share_differing_of_active'] for x in a3),
                       passed=all(x['rate_corr'] >= 0.999 for x in a3)),
               graphs=dict(fwd=fwd['graphs'], rev=rev['graphs']))
    print('  A2', json.dumps(out['A2']), flush=True)
    return out


def a4_blocks(model):
    st = model['stim']
    T = P['t_run']
    lo, vi, su = st['looming'], st['vinegar'], st['sugar']
    return [[(0.0, T, lo, 150.0)],
            [(0.0, 0.5 * T, lo, 200.0), (0.3 * T, T, vi, 150.0)],
            [(0.1 * T, 0.9 * T, su, 250.0)],
            [(0.0, T, vi, 120.0), (0.2 * T, 0.4 * T, lo[:150], 300.0)],
            [(0.05 * T, 0.6 * T, lo, 150.0), (0.05 * T, 0.6 * T, lo[::2], 100.0)],
            [(0.0, 0.25 * T, su, 200.0), (0.25 * T, 0.75 * T, vi, 200.0), (0.75 * T, T, lo, 200.0)],
            [(0.4 * T, T, lo, 180.0)],
            [(0.0, T, lo[150:], 160.0), (0.0, T, su, 90.0)]]


def a4(model, a, warm=1000, steps=200):
    bl = a4_blocks(model)
    seeds = list(range(100, 108))
    out = {}
    # graph replays over a full trial with different blocks per trial
    for draws in ('active', 'always'):
        s = prep_batch(model, a.stim, 8, STEPS_PER_TRIAL, torch.float64, seeds=seeds, blocks_list=bl, draws=draws)
        graphs = lif_torch_batch.graph_loop(s)
        torch.cuda.synchronize()
        out[f'graph_{draws}'] = dict(steps=s.nsteps, eager_warmup_steps=lif_torch_batch.WARMUP,
                                     graphs=len(graphs), replays=s.replays,
                                     replays_per_graph_step=s.replays / (s.nsteps - lif_torch_batch.WARMUP),
                                     distinct_keys=len(set(s.seg_keys)),
                                     overflow=s.ovf.cpu().numpy().astype(int).tolist(),
                                     spikes_per_trial=s.counts.sum(1).cpu().numpy().tolist())
        del s, graphs
        w4g._clean()
    # sync debug: eager steady loop and graph replays
    for mode in ('eager', 'graph'):
        s = prep_batch(model, a.stim, 8, warm + steps, torch.float64, seeds=seeds, blocks_list=bl)
        graphs = None
        if mode == 'eager':
            lif_torch_batch.eager_loop(s, stop=warm)
        else:
            graphs = lif_torch_batch.graph_loop(s, stop=warm)
        torch.cuda.synchronize()
        n_graphs_before = len(graphs) if graphs is not None else None
        torch.cuda.set_sync_debug_mode('error')
        try:
            if mode == 'eager':
                lif_torch_batch.eager_loop(s, stop=warm + steps)
            else:
                lif_torch_batch.graph_loop(s, stop=warm + steps, graphs=graphs)
            r = dict(steps=steps, raised=None, passed=True)
        except Exception as ex:  # noqa: BLE001
            r = dict(steps=steps, raised=f'{type(ex).__name__}: {str(ex)[:300]}',
                     trace=traceback.format_exc()[-1500:], passed=False)
        finally:
            torch.cuda.set_sync_debug_mode(0)
        torch.cuda.synchronize()
        if mode == 'graph':
            r['graphs_before_window'] = n_graphs_before
            r['graphs_after_window'] = len(graphs)
        r['overflow'] = s.ovf.cpu().numpy().astype(int).tolist()
        out[f'sync_{mode}'] = r
        del s, graphs
        w4g._clean()
    torch.cuda.set_sync_debug_mode('error')
    try:
        lif_torch.run_trial(model['W'], model['stim'][a.stim], params=w4g._params(2), seed=0, device='cuda',
                            dtype=torch.float64, rng='torch', **model_kw(model))
        out['positive_control_reference'] = dict(raised=None, passed=False)
    except Exception as ex:  # noqa: BLE001
        out['positive_control_reference'] = dict(raised=f'{type(ex).__name__}: {str(ex)[:200]}', passed=True)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    out['passed'] = (out['sync_eager']['passed'] and out['sync_graph']['passed']
                     and out['positive_control_reference']['passed']
                     and out['graph_active']['replays_per_graph_step'] == 1.0
                     and out['graph_always']['graphs'] == 1)
    print('  A4', json.dumps({k: v for k, v in out.items()}, default=str)[:2000], flush=True)
    return out


# ---------------------------------------------------------------- aggregate / report
def aggregate(res):
    runs = res['runs']
    agg = dict(rows={})
    for name in runs[0]['rows']:
        rs = [r['rows'][name] for r in runs]
        a = {k: w4g._rng([x[k] for x in rs]) for k in ('steady_mean_ms', 'median_ms_all_steps', 'window_ms_per_step',
                                                         'window_ms_per_step_no_events', 'stall_fraction',
                                                         'peak_mem_mb', 'event_over_host_clock')}
        B = res['row_B'][name]
        for src, key in (('window_ms_per_step_no_events', 'all_steps'), ('steady_mean_ms', 'steady')):
            ms = [x[src] for x in rs if x[src] is not None]
            a[f'trials_per_s_{key}'] = w4g._rng([B / (m * STEPS_PER_TRIAL / 1e3) for m in ms])
            a[f'bio_s_per_wall_s_{key}'] = w4g._rng([B * P['dt'] * 1e3 / m for m in ms])
        a['overflow_total'] = sum(sum(x['overflow'] or [0]) for x in rs) if rs[0]['overflow'] is not None else None
        a['same_counts_both_trials'] = all(x.get('same_counts_both_trials', True) for x in rs)
        a['busy_before_any'] = any(x['busy_check_before']['busy'] for x in rs)
        a['busy_after_any'] = any(x['busy_check_after']['busy'] for x in rs)
        a['graphs'] = rs[0]['graphs']
        a['B'] = B
        agg['rows'][name] = a
    return agg


def _m(a, f='{:.3f}'):
    if a is None:
        return '-'
    return f"{f.format(a['median'])} [{f.format(a['min'])}-{f.format(a['max'])}]"


def to_md(res):
    agg = res['agg']
    L = [f"# Week 5 P1: B trials in one CUDA graph: {res['tag']} ({len(res['runs'])} runs)", '',
         f"Looming bundle, gap + APL + presyn, rng='torch', seeds 0..B-1. Window: {res['args']['steps']} steps "
         f"after {res['args']['warm']} warm steps, one trial per row per run. Caps per trial: "
         f"spike_cap={CAPS['spike_cap']}, syn_cap={CAPS['syn_cap']}. Every value: median across runs "
         "[min-max across runs]. Clock: CUDA events rescaled to the host clock (cuda rows), host clock (cpu).", '',
         'steady ms/step: mean of the steps <= 5 ms (Day 6 rule; none when every step is slower). '
         'median ms/step: median of all window steps, no threshold (events, rescaled). '
         'all-steps ms/step: synchronized wall / steps of a second trial without per-step events.', '',
         '| row | B | steady ms/step | stall share | median ms/step | all-steps ms/step | trials/s (all steps) '
         '| bio-s/wall-s (all steps) | trials/s (steady) | peak GPU MB | overflow | busy (before/after) |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name, a in agg['rows'].items():
        L.append(f"| {name} | {a['B']} | {_m(a['steady_mean_ms'])} | {_m(a['stall_fraction'], '{:.4f}')} | "
                 f"{_m(a['median_ms_all_steps'])} | {_m(a['window_ms_per_step_no_events'])} | "
                 f"{_m(a['trials_per_s_all_steps'], '{:.3f}')} | "
                 f"{_m(a['bio_s_per_wall_s_all_steps'], '{:.3f}')} | {_m(a['trials_per_s_steady'], '{:.3f}')} | "
                 f"{_m(a['peak_mem_mb'], '{:.0f}')} | {a['overflow_total']} | "
                 f"{a['busy_before_any']}/{a['busy_after_any']} |")
    for k in ('ops_per_step', 'phases', 'parity', 'a4'):
        if res.get(k):
            L += ['', f'## {k}', '', '```', json.dumps(res[k], indent=1, default=str), '```']
    L += ['', '## Environment', '', '```', json.dumps(res['env'], indent=1), '```', '']
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', default=str(ROOT / 'results' / 'w4' / 'cns_bundle_calib.npz'))
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--warm', type=int, default=1000)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--stim', default='looming')
    ap.add_argument('--bs', type=int, nargs='*', default=list(BS))
    ap.add_argument('--bs-f32', type=int, nargs='*', default=list(BS))
    ap.add_argument('--only', nargs='*', help='row names to run (default: all)')
    ap.add_argument('--phase-steps', type=int, default=30)
    ap.add_argument('--gate-ms', type=float, default=1.0)
    ap.add_argument('--busy-poll', type=float, default=120.0)
    ap.add_argument('--no-checks', action='store_true', help='skip ops / phases / A2-A4')
    ap.add_argument('--checks-only', action='store_true')
    ap.add_argument('--resume', help='a *_batch.partial.json to continue')
    ap.add_argument('--out', default=str(ROOT / 'perf' / 'results' / 'w5'))
    a = ap.parse_args()
    assert torch.cuda.is_available(), 'needs CUDA'
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    model = t1.load_bundle(a.bundle)
    if a.resume:
        res = json.loads(Path(a.resume).read_text())
        tag = res['tag']
        print(f"resuming {tag}: {len(res['runs'])} runs done", flush=True)
    else:
        tag = f'{socket.gethostname()}_{time.strftime("%Y%m%d-%H%M")}'
        res = dict(tag=tag, env=t1.env_info(), args=vars(a), caps=CAPS, warmup_eager_steps=lif_torch_batch.WARMUP,
                   torch_threads=torch.get_num_threads(), runs=[], busy_log=[],
                   row_B={r['name']: r['B'] for r in rows_def(a)})
        print(json.dumps(res['env']), flush=True)
    partial = outdir / f'{tag}_batch.partial.json'

    def save():
        partial.write_text(json.dumps(res, indent=1, default=str))
    if not a.no_checks and 'a4' not in res:
        res['ops_per_step'] = {f'B={B}': ops_per_step(model, a, B) for B in a.bs}
        print('  ops', json.dumps(res['ops_per_step']), flush=True)
        chk, waits = wait_idle(a.busy_poll, res['busy_log'])     # phases are timed too
        res['phases'] = dict(busy_check_before=chk, busy_waits=waits,
                             per_B=[phases(model, a, B) for B in a.bs], busy_check_after=busy_check())
        print('  phases', json.dumps(res['phases']['per_B']), flush=True)
        res['parity'] = parity(model, a)
        res['a4'] = a4(model, a)
        save()
    if not a.checks_only:
        while len(res['runs']) < a.runs:
            print(f"=== run {len(res['runs']) + 1}/{a.runs}", flush=True)
            res['runs'].append(one_run(model, a, res['busy_log']))
            save()
        res['agg'] = aggregate(res)
    (outdir / f'{tag}_batch.json').write_text(json.dumps(res, indent=1, default=str))
    (outdir / f'{tag}_batch.md').write_text(to_md(res) if 'agg' in res else json.dumps(res, indent=1, default=str))
    if 'agg' in res:
        print(to_md(res), flush=True)
    print(f'saved {outdir / tag}_batch.json and .md', flush=True)


if __name__ == '__main__':
    main()
