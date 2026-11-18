# Week 4 perf log

## T0: CPU CI with parity tests on a synthetic network (2026-11-18)

Brief: [tasks/T0_brief.md](tasks/T0_brief.md). Branch `perf/w4-t0-ci`.

**Built**
- `tests/synth.py`: seeded 300-neuron synthetic network built in code (5% density, 20% inhibitory, synapse counts 1-10 x w_syn x 4; sensory 0-19, KCs 20-49 with rate-APL, gap block 50-69 with symmetric random coupling, presyn via `sim.presyn.build_presyn`). No files.
- `tests/test_parity_cpu.py`: `lif_torch.run_trial(device='cpu', dtype=float64, rng='numpy')` vs `lif_numpy.run_trial`, exact equality of spike counts, 3 cases: options off; gap+apl+presyn+t_stim; overlapping blocks with gap+apl+presyn. Each case also asserts a minimum amount of non-sensory activity so equality is not trivial.
- `.github/workflows/ci.yml` (job `cpu-tests`, ubuntu-latest, Python 3.12, CPU torch 2.11.0 from the PyTorch CPU index) and `requirements-ci.txt` (numpy/scipy/pandas/pytest pinned to the WSL bench venv versions).

**Findings**
- Both reference simulators are on main. `lif_torch.run_trial` has no `state=`/`return_state=`, so the split-run test was not added.
  - Correction (2026-11-18, T1): that was wrong; main's `lif_torch.run_trial` has had `state=`/`return_state=` since PR #2 (commit 3472c84). The split-run parity test is briefed separately as T2.
- Local (WSL, ~/flybench/.venv): 3 passed in ~3.3 s; each test < 0.4 s.
- Failure demo: in a scratch copy, `rfc = int(round(t_rfc / dt)) - 1` in `lif_torch.py` makes all 3 tests fail (92 / 31 / 27 neurons differ). A first, weaker blocks case (150-200 Hz) did not catch that break, so the blocks and all-options cases were strengthened (higher drive) until they did.
- No file under `sim/` changed.

## T1: where the GPU step goes, overhead budget (2026-11-18)

Brief: [tasks/T1_brief.md](tasks/T1_brief.md). Branch `perf/w4-t1-overhead`. Report: [perf/week04_day3.md](../../perf/week04_day3.md).

**Built**
- `bench/w4_overhead.py`: items 1-5 (launch, sync, sync-forcing ops at n = 165,122, small H2D, CUDA graph) on cuda and cpu; item 6 (per-step aten ops via `TorchDispatchMode`, syncs via `torch.cuda.set_sync_debug_mode('warn')` attributed to op and source line, H2D/D2H copies) on a warm 200-step window resumed from a 1000-step state; item 7 (predicted vs measured ms/step). Steps are timed by wrapping `lif_torch._decay` from the bench, so `sim/` is untouched and setup is excluded.
- `bench/w4_overhead_plot.py` (stacked bar PNG; needs matplotlib, which the bench venv does not have, so it was run from a separate venv) and `bench/w4_overhead_compare.py` (run-to-run table).
- Results under `perf/results/w4/` (`results/` is gitignored): the reported run, 4 repeats, `repeatability.md`.

**Findings**
- Host-bound, confirmed. Per step: ~117 aten ops, ~106 GPU kernels, 10 syncs, 0-1 H2D. Budget (reported run): dispatch 0.17 + launch 1.08-1.10 + D2H syncs 0.62-0.71 + scalar `x[idx] = c` 0.15 + H2D 0.02 = 2.04 (numpy) / 2.12 (torch) ms vs measured 1.86 / 2.04 ms.
- Sync sites: `int(lens.sum())` x2, `nonzero`, APL `float(...)`, numpy-rng upload as suspected; `idx.numel()`/`pre.numel()` do not sync; not suspected: 4 scalar index assignments (hidden H2D + sync) and bool-mask indexing (1-2/step).
- CUDA micro-benchmarks on WSL vary by >2x between runs for several items (bimodal); steady-state step time is stable (1.65-1.98 ms) but ~11 ms stalls hit 0.6-9.4% of steps. Predicted/steady within +/-20% in 4 of 5 runs.
- main's `sim/` with the bench-pack method: 1.91-2.49 ms/step (rng='torch') vs 3.33 ms on 2026-11-16.
- No file under `sim/` changed; `~/flybench/flybrain/results/w4/bench/*` untouched.
