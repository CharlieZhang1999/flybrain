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

## T1: where the GPU step goes, overhead budget (2026-11-18, revised once)

Brief: [tasks/T1_brief.md](tasks/T1_brief.md); revision: [tasks/T1_revision.md](tasks/T1_revision.md). Branch `perf/w4-t1-overhead`. Report: [perf/week04_day3.md](../../perf/week04_day3.md).

**Built**
- `bench/w4_overhead.py`: items 1-5 (launch, sync, sync-forcing ops at n = 165,122, small H2D, CUDA graph) on cuda and cpu; item 6 (per-step aten ops via `TorchDispatchMode`, syncs via `torch.cuda.set_sync_debug_mode('warn')` attributed to op and source line, H2D/D2H copies) on a warm 200-step window resumed from a 1000-step state; item 7 (predicted vs measured ms/step). `--runs 5` (default) repeats the whole suite in one invocation and reports the median and min-max across runs for every value, a Variability list (everything > 2x across runs), and the budget with units = medians across runs (and sizes) plus per-run predictions. Extra costs are directly measured totals minus already-charged dispatch/launch, clamped at 0. Steps are timed by wrapping `lif_torch._decay` from the bench, so `sim/` is untouched and setup is excluded.
- `bench/w4_overhead_plot.py`: stacked-bar PNG and the HTML budget table that the site embeds verbatim (needs matplotlib; run from a separate venv because `~/flybench/.venv` has none).
- Results: `perf/results/w4/DESKTOP-4C13T5V_20261118-1101_overhead.*` (`results/` is gitignored). The first version's single-run results were removed.

**Findings**
- Host-bound, confirmed. Per step: ~117 aten ops, ~106 GPU kernels, 10 syncs, 0-1 H2D. Budget with median units: dispatch 0.16 + launch 1.00 + D2H syncs 0.38-0.47 + scalar `x[idx] = c` 0.13 + H2D 0.02 = 1.70 (numpy) / 1.77 (torch) ms vs steady-state 1.68 / 1.69 ms. Per-run predicted/steady: median 1.10 (numpy) / 1.24 (torch); in an earlier 5-run invocation whose output was lost: 0.86 / 0.87.
- Sync sites: `int(lens.sum())` x2, `nonzero`, APL `float(...)`, numpy-rng upload as suspected; `idx.numel()`/`pre.numel()` do not sync; not suspected: 4 scalar index assignments (hidden H2D + sync) and bool-mask indexing (1-2/step).
- Regimes: many CUDA micro-benchmarks switch between a fast and a slow value from run to run (up to 16x); the steady step is stable (1.56-1.78 ms), but ~11 ms stalls hit 0.1-25% of steps. The 3.3 ms of 2026-11-16 is an all-steps number with setup, warm-up and stalls; main's `sim/` with the same method: 2.50 [1.95-2.74] ms/step (rng='torch'). Week 5 reports steady ms/step and stall fraction separately.
- No file under `sim/` changed; `~/flybench/flybrain/results/w4/bench/*` untouched.

## T2: save/resume parity test (2026-11-18)

Brief: [tasks/T2_brief.md](tasks/T2_brief.md). Branch `perf/w4-t2-resume-test`.

**Built**
- `tests/test_resume_cpu.py`, 3 cases on the `tests/synth.py` network with blocks + gap + APL + presyn (seed 3, two overlapping blocks [0, 0.12) s and [0.06, 0.2) s), T = 0.2 s split at 0.1 s (step 1000, inside both blocks): NumPy split == NumPy single; Torch (CPU, float64, rng='numpy') split == Torch single; Torch split == NumPy single. Each asserts: split step inside a block and a non-empty delay ring in the saved state; the T0 activity guard (>= 150 non-sensory spikes); final cumulative counts equal; counts at split + new spikes = final counts; concatenated (t, i) records equal the single run's, in order.
- `python tests/test_resume_cpu.py` prints the saved state at the split point for both simulators (block windows, ring slots with delivery step).

**Findings**
- Hypothesis confirmed: exact parity for both simulators and across them. Both docstrings agree on the contract: `counts` are cumulative over the chained trial, `t`/`i` hold only the new segment, `t` is the absolute step.
- At the split, the ring holds 20 spikes in flight (identical in both simulators), in 14 of 18 slots.
- Local (WSL, ~/flybench/.venv): 6 passed in 4.2 s; new tests 0.20 / 0.38 / 0.31 s.
- Failure demo (scratch copy): dropping the ring restore makes the affected tests fail with 33 neurons differing (NumPy: 1 test; Torch: 2 tests).
- Gaps, not fixed (no `sim/` changes): the state dicts are not interchangeable (NumPy stores the RNG as `rng`, Torch as `np_rng` and silently skips a missing key, from reading the code); `lif_numpy` returns the state only with `record=True`.
- No file under `sim/` changed.

## T3: step anatomy on the 5060 Ti (2026-11-18)

Brief: [tasks/T3_brief.md](tasks/T3_brief.md). Branch `perf/w4-t3-anatomy`. Report: [perf/week04_day4.md](../../perf/week04_day4.md).

**Built**
- `bench/w4_anatomy.py`: instrumented copy of the `run_trial` step loop (bundle configuration; same helpers and statement order as `sim/lif_torch.py` 201-256) split into 11 phases. Passes per run: uninstrumented `run_trial`; host `perf_counter` only (phase wall table); `perf_counter` + CUDA events + NVTX (GPU span); per-op gated GPU exec (`torch.cuda._sleep` before each aten op, host-enqueue checked against the gate); per-phase GPU ops (`TorchDispatchMode`) and syncs (sync-debug warnings); nvidia-smi utilisation and clocks. Drift guard: exact spike counts vs `run_trial` over 300 steps from t = 0 (exit code 3 on failure), plus every repeat of the warm window. `--runs 5` aggregates median [min-max].
- `bench/w4_anatomy_plot.py` (stacked bar PNG + site table), `bench/w4_nsys_timeline.py` (one step from an nsys SQLite export).
- `bench/w4_overhead_plot.py`: T1 PNG redrawn for a 390 px screen (840 px wide, 9-11 pt); the T1 budget table HTML is byte-identical.
- `perf/week04_day3.md`: the unmeasured "~20% utilisation" line now cites T3's measurement.

**Findings**
- Drift guard passed (3,710 spikes, 0 neurons differ; warm window 3,438 spikes in every repeat).
- Sum of phase wall medians / steady step (same pass): 0.973 [0.964-0.979].
- Top 3 phases by wall: presyn 0.324 ms (1 sync), synaptic 0.308 ms (1 sync), APL 0.161 ms (1 sync). Gather/scatter + spike phase = 63% of the step with 7 of 10 syncs; gather/scatter is kernel-bound (54 GPU ops, 0.30 ms GPU exec), the spike phase is sync-bound (5 syncs, 0.025 ms GPU exec).
- GPU utilisation 19% [3-22%]; GPU exec of non-sync ops 0.56 ms = 0.34 [0.30-0.36] of the uninstrumented steady step (1.61 ms). Event span / wall is 1.00 by construction.
- nsys 2024.6.2 (CUDA 12.8 toolkit) records NVTX but no CUDA activity under WSL with driver 13.3 (CUPTI_ERROR_INVALID_DEVICE); host-side timeline captured; install command for a newer nsys in the report, not run.
- 13 event records per step cost ~0.25 ms; the events pass is 1.86 [1.72-2.05] ms (within 25% of T1's 1.65-1.73), so the wall table uses the host-only pass.
- No file under `sim/` changed.
