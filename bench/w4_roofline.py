#!/usr/bin/env python
"""Week 4 T4: compute roofline of the RTX 5060 Ti (matmul peaks per dtype).

Peaks: torch.matmul on square n x n matrices (randn inputs, preallocated
output, torch.matmul(a, b, out=c)) for
  fp64  float64
  fp32  float32, TF32 off (torch.backends.cuda.matmul.allow_tf32 = False)
  tf32  float32, TF32 on  (torch.backends.cuda.matmul.allow_tf32 = True)
  bf16  bfloat16 (torch defaults: fp32 compute type, reduced-precision
        split-K reductions allowed)
  fp16  float16  (same)
and n in --sizes (default 1024 2048 4096 8192; a size that raises OOM is
skipped and recorded). Per (dtype, n): --warmup untimed matmuls, then --reps
(>= 100) matmuls back to back, one CUDA event pair per matmul. TFLOP/s =
2 n^3 / t. Per size: median and p90 of t (p90 of t = the slow tail, i.e. the
p10 of TFLOP/s). The peak of a dtype in a run is the best size's median.

Clock: TFLOP/s use the CUDA event times as measured (not rescaled). T5b
rescaled event spans to the host clock because events read 4-6% high against
it. Here the host clock is the unstable one: the ratio host wall / event sum
of the same loop, and event / host over a 2 s idle sleep, change between
sessions and runs (seen: 0.92, 1.00, 1.05, 1.09), while the raw event time of
the same matmul repeats within 1% across runs and sessions. So events are the
measurement and the host clock is logged as diagnostics only: host_tflops =
2 n^3 reps / synchronized host wall of the loop, event_scale = wall /
sum(events) (which also includes launch gaps at small n), and
clock_event_over_host_sleep.

Copy bandwidth: every run also repeats the Day 2 copy measurement of
bench/w4_bench.py (torch copy_ of 512 MB float32, (read + write) bytes /
median time, CUDA events). The roofline itself uses the
Day 2 value 382 GB/s (BW_DAY2) as the brief asks; the same-session value is a
check.

LIF decay intensity: sim/lif_torch._decay (imported, not changed) is run once
on CPU under a TorchDispatchMode that counts, per aten op, the floating-point
operations (elementwise add/sub/mul/div/rsub = 1 FLOP per output element;
where, compare and integer ops = 0 FLOPs) and the bytes of every tensor
argument read and every tensor result written. This is the eager (unfused)
traffic, one kernel per op. The fused lower bound (each input read once,
each output written once) is computed from the same function's arguments
and results. Both per neuron, for float64 and float32.

Stop conditions (brief): a dtype whose matmul raises is recorded and skipped;
if a dtype's peak (median across runs) exceeds the vendor spec by > 10%, the
results are still written and the script exits with code 5.

Usage (repo root, WSL venv with CUDA torch):
  python bench/w4_roofline.py --runs 5
Output: perf/results/w4/<host>_<timestamp>_roofline.json and .md
Plot:   python bench/w4_roofline_plot.py perf/results/w4/<tag>_roofline.json  (needs matplotlib)
"""
import argparse
import json
import math
import socket
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'bench'))
from sim import lif_torch  # noqa: E402
import w4_overhead as t1  # noqa: E402  (env_info, _run)

pc = time.perf_counter
DTYPES = {'fp64': (torch.float64, False), 'fp32': (torch.float32, False), 'tf32': (torch.float32, True),
          'bf16': (torch.bfloat16, False), 'fp16': (torch.float16, False)}
BW_DAY2 = 382.0   # GB/s, cuda/torch_copy, bench/w4_bench.py, PC run DESKTOP-4C13T5V_20261116-2308 (WSL results/w4/bench/)

# Vendor numbers. NVIDIA publishes for the RTX 5060 Ti only: 4,608 CUDA cores, 2.57 GHz boost,
# 759 "AI TOPS" (no precision given), 448 GB/s (nvidia.com 50-series compare page). It publishes
# no FP64 / FP32 / tensor TFLOPS for it. The specs below are DERIVED from the RTX Blackwell GPU
# architecture whitepaper (Appendix C, Table 6, RTX 5070 = GB205, 48 SMs, 2,512 MHz boost:
# FP32 non-tensor 30.9, TF32 tensor 30.9, FP16/BF16 tensor with FP32 accumulate 61.7,
# FP16 tensor with FP16 accumulate 123.5 dense TFLOPS; FP64 = 1/64 of FP32, stated for GB202 on
# the GB202 section of the same paper, "Note: The GB202 GPU also includes 384 FP64 Cores"), scaled per SM per clock to the 5060 Ti: 36 SMs (4,608 / 128),
# 2.57 GHz. Check: 5070 FP4 sparse 987.8 = 8 x FP16-acc; 8 x 94.7 = 758 for the 5060 Ti vs the
# published 759 AI TOPS.
SPEC_SOURCES = dict(
    product='https://www.nvidia.com/en-us/geforce/graphics-cards/compare/#50-series (RTX 5060 Ti column: '
            '4608 CUDA cores, boost 2.57 GHz, 759 AI TOPS, 448 GB/sec)',
    whitepaper='https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf '
               '(Appendix C, Table 6, RTX 5070 column; FP64 = 1/64 FP32: GB202 section)')
SM, BOOST_GHZ = 36, 2.57
PER_SM_CLK = {'fp64': 4, 'fp32': 256, 'tf32': 256, 'bf16': 512, 'fp16': 512}   # dense FLOP / SM / clock
SPEC_TFLOPS = {k: round(v * SM * BOOST_GHZ * 1e9 / 1e12, 3) for k, v in PER_SM_CLK.items()}
SPEC_FP16_ACC_TFLOPS = round(1024 * SM * BOOST_GHZ * 1e9 / 1e12, 3)   # FP16 tensor, FP16 accumulate (not what torch uses)
SPEC_BW_GBPS = 448.0


def _q(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def smi():
    q = 'clocks.sm,clocks.max.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active'
    return t1._run(['nvidia-smi', f'--query-gpu={q}', '--format=csv,noheader'])


def calibrate_clock(sleep_s=2.0):
    """CUDA event time / host time over a host sleep (> 1: events read high)."""
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record(); torch.cuda.synchronize(); h0 = pc()
    time.sleep(sleep_s)
    e.record(); torch.cuda.synchronize(); h1 = pc()
    return (s.elapsed_time(e) / 1e3) / (h1 - h0)


def time_op(fn, reps, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
    h0 = pc()
    for s, e in ev:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    wall = pc() - h0
    raw = [s.elapsed_time(e) / 1e3 for s, e in ev]
    return raw, wall, wall / sum(raw)


def bandwidth(mb=512, reps=100):
    src = torch.empty(mb * 2 ** 20 // 4, dtype=torch.float32, device='cuda').normal_()
    dst = torch.empty_like(src)
    raw, _, scale = time_op(lambda: dst.copy_(src), reps, 10)
    ts = raw
    b = 2 * src.numel() * 4
    out = dict(GBps=round(b / statistics.median(ts) / 1e9, 1), p90_GBps=round(b / _q(ts, 0.9) / 1e9, 1), mb=mb,
               GBps_host=round(b * reps / (sum(raw) * scale) / 1e9, 1), event_scale=round(scale, 4))
    del src, dst
    torch.cuda.empty_cache()
    return out


def matmul_size(dtype, n, reps, warmup):
    a = torch.randn(n, n, device='cuda', dtype=dtype)
    b = torch.randn(n, n, device='cuda', dtype=dtype)
    c = torch.empty(n, n, device='cuda', dtype=dtype)
    # warm-up by time for large / slow sizes: at least `warmup` calls, or ~1 s
    torch.matmul(a, b, out=c); torch.cuda.synchronize()
    t0 = pc(); torch.matmul(a, b, out=c); torch.cuda.synchronize(); one = pc() - t0
    w = max(warmup, min(50, int(1.0 / max(one, 1e-6))))
    ts_raw, wall, scale = time_op(lambda: torch.matmul(a, b, out=c), reps, w)
    ts = ts_raw
    fl = 2.0 * n ** 3
    med, p90 = statistics.median(ts), _q(ts, 0.9)
    r = dict(n=n, reps=reps, warmup=w, median_ms=round(med * 1e3, 4), p90_ms=round(p90 * 1e3, 4),
             tflops_median=round(fl / med / 1e12, 3), tflops_p90=round(fl / p90 / 1e12, 3),
             tflops_median_hostscaled=round(fl / (statistics.median(ts_raw) * scale) / 1e12, 3),
             host_tflops=round(fl * reps / wall / 1e12, 3), event_scale=round(scale, 4),
             smi_after=smi())
    del a, b, c
    torch.cuda.empty_cache()
    return r


def one_run(a, clock):
    run = dict(clock_event_over_host_sleep=round(clock, 4), smi_start=smi(), bandwidth=bandwidth(), dtypes={})
    print(f"  sleep clock {clock:.4f}  copy {run['bandwidth']}", flush=True)
    for name in a.dtypes:
        dt, tf32 = DTYPES[name]
        torch.backends.cuda.matmul.allow_tf32 = tf32
        d = dict(allow_tf32=torch.backends.cuda.matmul.allow_tf32, sizes={}, skipped={}, error=None)
        for n in a.sizes:
            try:
                d['sizes'][str(n)] = r = matmul_size(dt, n, a.reps, a.warmup)
                print(f"  {name} n={n}: {r['tflops_median']} TFLOP/s (p90 {r['tflops_p90']}, host {r['host_tflops']}, scale {r['event_scale']})"
                      f"  smi {r['smi_after']}", flush=True)
            except torch.OutOfMemoryError as ex:
                d['skipped'][str(n)] = f'OOM: {str(ex)[:200]}'
                torch.cuda.empty_cache()
                print(f'  {name} n={n}: OOM, skipped', flush=True)
            except Exception as ex:  # noqa: BLE001  (stop condition: report, skip the dtype, continue)
                d['error'] = f'{type(ex).__name__}: {ex}'[:500]
                d['trace'] = traceback.format_exc()[-2000:]
                torch.cuda.empty_cache()
                print(f'  {name} n={n}: ERROR {d["error"]}', flush=True)
                break
        if d['sizes']:
            best = max(d['sizes'].values(), key=lambda r: r['tflops_median'])
            d['peak_tflops'], d['peak_n'] = best['tflops_median'], best['n']
        run['dtypes'][name] = d
    torch.backends.cuda.matmul.allow_tf32 = False
    run['smi_end'] = smi()
    return run


def tf32_check(n=1024):
    """Relative error vs a float64 product: shows whether the TF32 flag took effect."""
    g = torch.Generator(device='cuda').manual_seed(0)
    a = torch.randn(n, n, device='cuda', generator=g)
    b = torch.randn(n, n, device='cuda', generator=g)
    ref = a.double() @ b.double()
    out = {}
    for name in ('fp32', 'tf32'):
        torch.backends.cuda.matmul.allow_tf32 = DTYPES[name][1]
        out[name] = float(((a @ b).double() - ref).norm() / ref.norm())
    torch.backends.cuda.matmul.allow_tf32 = False
    return out


# ---------------------------------------------------------------- LIF decay intensity
FLOP_OPS = {'add', 'sub', 'mul', 'div', 'rsub'}


class _Count(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        ins = [x for x in list(args) + list(kwargs.values()) if isinstance(x, torch.Tensor)]
        outs = [x for x in (out if isinstance(out, (tuple, list)) else [out]) if isinstance(x, torch.Tensor)]
        base = func.overloadpacket.__name__.rstrip('_')
        o = outs[0]
        fl = o.numel() if (base in FLOP_OPS and o.is_floating_point()) else 0
        self.ops.append(dict(op=str(func), flops=fl,
                             read=sum(x.numel() * x.element_size() for x in ins),
                             write=sum(x.numel() * x.element_size() for x in outs)))
        return out


def decay_intensity(n=100_000):
    """FLOPs and bytes per neuron of sim/lif_torch._decay (bundle-style args, see run_trial)."""
    res = {}
    for name, dt in (('float64', torch.float64), ('float32', torch.float32)):
        g = torch.Generator().manual_seed(0)
        v = torch.rand(n, generator=g, dtype=dt)
        gg = torch.rand(n, generator=g, dtype=dt)
        last = torch.randint(-20, 0, (n,), generator=g, dtype=torch.int64)
        rfc_n = torch.full((n,), 20, dtype=torch.int64)
        t = torch.zeros((), dtype=torch.int64)
        args = (v, gg, last, rfc_n, t, -52.0, 0.5, 0.95, 0.9)
        with _Count() as c:
            outs = lif_torch._decay(*args)
        fl = sum(o['flops'] for o in c.ops)
        rd = sum(o['read'] for o in c.ops)
        wr = sum(o['write'] for o in c.ops)
        ins = [x for x in args if isinstance(x, torch.Tensor)]
        f_rd = sum(x.numel() * x.element_size() for x in ins)
        f_wr = sum(x.numel() * x.element_size() for x in outs)
        res[name] = dict(
            n=n, aten_ops=len(c.ops), ops=[o['op'] for o in c.ops],
            flops_per_neuron=round(fl / n, 4),
            eager_bytes_per_neuron=round((rd + wr) / n, 4),
            eager_read_per_neuron=round(rd / n, 4), eager_write_per_neuron=round(wr / n, 4),
            eager_flop_per_byte=round(fl / (rd + wr), 5),
            fused_bytes_per_neuron=round((f_rd + f_wr) / n, 4),
            fused_read_per_neuron=round(f_rd / n, 4), fused_write_per_neuron=round(f_wr / n, 4),
            fused_flop_per_byte=round(fl / (f_rd + f_wr), 5))
    return res


# ---------------------------------------------------------------- aggregate / report
def _rng(xs, nd=3):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return dict(median=round(statistics.median(xs), nd), min=round(min(xs), nd), max=round(max(xs), nd),
                max_over_min=round(max(xs) / min(xs), 2) if min(xs) > 0 else None, runs=xs)


def aggregate(res):
    runs, A = res['runs'], dict(dtypes={}, variability=[], over_spec=[])
    A['clock_event_over_host_sleep'] = _rng([r['clock_event_over_host_sleep'] for r in runs], 4)
    A['event_scale'] = _rng([s['event_scale'] for r in runs for d in r['dtypes'].values() for s in d['sizes'].values()]
                            + [r['bandwidth']['event_scale'] for r in runs], 4)
    A['copy_GBps'] = _rng([r['bandwidth']['GBps'] for r in runs], 1)
    for name in res['args']['dtypes']:
        ds = [r['dtypes'][name] for r in runs]
        a = dict(peak_tflops=_rng([d.get('peak_tflops') for d in ds]),
                 peak_n=[d.get('peak_n') for d in ds],
                 errors=[d['error'] for d in ds if d['error']],
                 skipped=sorted({k for d in ds for k in d['skipped']}), sizes={})
        for n in res['args']['sizes']:
            rs = [d['sizes'][str(n)] for d in ds if str(n) in d['sizes']]
            if rs:
                a['sizes'][str(n)] = {k: _rng([r[k] for r in rs]) for k in
                                      ('tflops_median', 'tflops_p90', 'tflops_median_hostscaled', 'host_tflops',
                                                                   'event_scale')}
        spec = SPEC_TFLOPS[name]
        if a['peak_tflops']:
            a['spec_tflops'] = spec
            a['pct_of_spec'] = round(100 * a['peak_tflops']['median'] / spec, 1)
            a['pct_of_spec_range'] = [round(100 * a['peak_tflops']['min'] / spec, 1),
                                      round(100 * a['peak_tflops']['max'] / spec, 1)]
            a['ridge_flop_per_byte_382'] = round(a['peak_tflops']['median'] * 1e12 / (BW_DAY2 * 1e9), 3)
            if a['peak_tflops']['median'] > 1.10 * spec:
                A['over_spec'].append(dict(dtype=name, peak=a['peak_tflops']['median'], spec=spec))
            a['runs_over_110pct'] = [x for x in a['peak_tflops']['runs'] if x > 1.10 * spec]
            for n, s in a['sizes'].items():
                v = s['tflops_median']
                if v['max_over_min'] and v['max_over_min'] > 2:
                    A['variability'].append(dict(what=f'{name} n={n} tflops_median', **{k: v[k] for k in
                                                 ('median', 'min', 'max', 'max_over_min')}))
            v = a['peak_tflops']
            if v['max_over_min'] and v['max_over_min'] > 2:
                A['variability'].append(dict(what=f'{name} peak', **{k: v[k] for k in ('median', 'min', 'max', 'max_over_min')}))
        A['dtypes'][name] = a
    p = A['dtypes']
    if p.get('fp64', {}).get('peak_tflops') and p.get('fp32', {}).get('peak_tflops'):
        A['fp32_over_fp64'] = round(p['fp32']['peak_tflops']['median'] / p['fp64']['peak_tflops']['median'], 1)
    return A


def _m(a, f='{:.2f}'):
    return '-' if a is None else f"{f.format(a['median'])} [{f.format(a['min'])}-{f.format(a['max'])}]"


def to_md(res):
    A, L = res['agg'], []
    L += [f"# Week 4 T4: compute roofline, {res['tag']} ({len(res['runs'])} runs)", '',
          f"torch.matmul, square n x n, sizes {res['args']['sizes']}, {res['args']['reps']} timed repeats per size "
          "(CUDA events, not rescaled; host clock in the diagnostics columns). Peak of a run = best size's median. "
          'Every value: median across runs [min-max across runs].', '',
          '| dtype | peak TFLOP/s | best n per run | spec TFLOP/s (derived) | % of spec | ridge FLOP/B at 382 GB/s |',
          '|---|---|---|---|---|---|']
    for name, a in A['dtypes'].items():
        if not a['peak_tflops']:
            L.append(f"| {name} | error: {a['errors'][:1]} | - | {SPEC_TFLOPS[name]} | - | - |")
            continue
        L.append(f"| {name} | {_m(a['peak_tflops'], '{:.3f}')} | {a['peak_n']} | {a['spec_tflops']:.2f} | "
                 f"{a['pct_of_spec']}% [{a['pct_of_spec_range'][0]}-{a['pct_of_spec_range'][1]}%] | "
                 f"{a['ridge_flop_per_byte_382']} |")
    L += ['', f"fp32 / fp64 peak: {A.get('fp32_over_fp64')}. FP16 tensor with FP16 accumulate (not used by "
          f"torch.matmul): {SPEC_FP16_ACC_TFLOPS} TFLOP/s.", '',
          '## Per size (TFLOP/s; median of the per-run medians [min-max]; p90 of time = slow tail; host = '
          'wall of the whole repeat loop)', '',
          '| dtype | n | median | p90 | median rescaled to host wall | host wall | host wall / event sum |', '|---|---|---|---|---|---|---|']
    for name, a in A['dtypes'].items():
        for n, s in a['sizes'].items():
            L.append(f"| {name} | {n} | {_m(s['tflops_median'], '{:.3f}')} | {_m(s['tflops_p90'], '{:.3f}')} | "
                     f"{_m(s['tflops_median_hostscaled'], '{:.3f}')} | {_m(s['host_tflops'], '{:.3f}')} | {_m(s['event_scale'], '{:.3f}')} |")
        if a['skipped']:
            L.append(f"| {name} | skipped {a['skipped']} (OOM) | | | | |")
        if a['errors']:
            L.append(f"| {name} | error | {a['errors'][0]} | | | |")
    L += ['', f"Host wall / event sum, all loops: {_m(A['event_scale'], '{:.4f}')}; "
          f"event / host over a 2 s idle sleep: {_m(A['clock_event_over_host_sleep'], '{:.4f}')}. "
          f"Copy bandwidth this session: {_m(A['copy_GBps'], '{:.1f}')} GB/s (Day 2: {BW_DAY2} GB/s; NVIDIA: "
          f"{SPEC_BW_GBPS} GB/s). TF32 check, relative error vs float64 at n=1024: {res['tf32_check']}.", '',
          '## LIF decay (sim/lif_torch._decay) per neuron', '',
          '| dtype | FLOPs | eager bytes (aten ops) | eager FLOP/B | fused bytes | fused FLOP/B |',
          '|---|---|---|---|---|---|']
    for name, d in res['decay'].items():
        L.append(f"| {name} | {d['flops_per_neuron']:g} | {d['eager_bytes_per_neuron']:g} ({d['aten_ops']} ops) | "
                 f"{d['eager_flop_per_byte']:.4f} | {d['fused_bytes_per_neuron']:g} | {d['fused_flop_per_byte']:.4f} |")
    L += ['', '## Variability (> 2x across runs)', '']
    L += [f"- {v['what']}: {v['median']} [{v['min']}-{v['max']}], {v['max_over_min']}x" for v in A['variability']] or ['- none']
    L += ['', '## Over spec by > 10% (stop condition)', '']
    L += [f"- {v['dtype']}: {v['peak']} vs {v['spec']}" for v in A['over_spec']] or ['- none (median peak)']
    L += [f"- single runs above 110% of spec, {k}: {a['runs_over_110pct']}" for k, a in A['dtypes'].items()
          if a.get('runs_over_110pct')]
    L += ['', '## Spec derivation', '', '```', json.dumps(res['spec'], indent=1), '```', '',
          '## Environment', '', '```', json.dumps(res['env'], indent=1), '```', '']
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--sizes', type=int, nargs='*', default=[1024, 2048, 4096, 8192])
    ap.add_argument('--dtypes', nargs='*', default=list(DTYPES), choices=list(DTYPES))
    ap.add_argument('--reps', type=int, default=100)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--out', default=str(ROOT / 'perf' / 'results' / 'w4'))
    a = ap.parse_args()
    assert torch.cuda.is_available(), 'needs CUDA'
    assert a.reps >= 100 or a.runs < 5, 'the brief asks for >= 100 repeats per size'
    tag = f'{socket.gethostname()}_{time.strftime("%Y%m%d-%H%M")}'
    env = t1.env_info()
    pr = torch.cuda.get_device_properties(0)
    env.update(sms=pr.multi_processor_count, vram_gb=round(pr.total_memory / 2 ** 30, 1),
               l2_mb=round(getattr(pr, 'L2_cache_size', 0) / 2 ** 20, 1),
               fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
               bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
               smi_clocks=t1._run(['nvidia-smi', '--query-gpu=clocks.max.sm,clocks.max.mem,power.limit',
                                   '--format=csv,noheader']))
    spec = dict(sources=SPEC_SOURCES, sms=SM, boost_ghz=BOOST_GHZ, dense_flop_per_sm_per_clock=PER_SM_CLK,
                tflops=SPEC_TFLOPS, fp16_acc_tflops=SPEC_FP16_ACC_TFLOPS, bandwidth_GBps=SPEC_BW_GBPS,
                note='Derived, not published for the 5060 Ti: whitepaper RTX 5070 rates per SM per clock x 36 SMs '
                     'x 2.57 GHz boost. FP64 = FP32 / 64 (whitepaper, GB202). torch fp16/bf16 matmul uses FP32 '
                     'accumulate.')
    res = dict(tag=tag, env=env, args=vars(a), spec=spec, bw_day2_GBps=BW_DAY2, runs=[])
    print(json.dumps(env), flush=True)
    res['decay'] = decay_intensity()
    res['tf32_check'] = tf32_check()
    print('decay', json.dumps({k: {kk: v[kk] for kk in v if kk != 'ops'} for k, v in res['decay'].items()}), flush=True)
    print('tf32 check', res['tf32_check'], flush=True)
    for i in range(a.runs):
        print(f'=== run {i + 1}/{a.runs}', flush=True)
        res['runs'].append(one_run(a, calibrate_clock()))
    res['agg'] = aggregate(res)
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f'{tag}_roofline.json').write_text(json.dumps(res, indent=1))
    (outdir / f'{tag}_roofline.md').write_text(to_md(res))
    print(to_md(res), flush=True)
    print(f'saved {outdir / tag}_roofline.json and .md', flush=True)
    if res['agg']['over_spec']:
        print('STOP: a peak exceeds the derived vendor spec by > 10%:', res['agg']['over_spec'], flush=True)
        sys.exit(5)


if __name__ == '__main__':
    main()
