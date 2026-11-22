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

## T5a: sync-free step, CPU parity (Day 6 part 1, 2026-11-22)

Brief: [tasks/T5a_brief.md](tasks/T5a_brief.md). Branch `perf/w4-t5a-syncfree`. Cloud container, CPU only (torch 2.15 locally; CI torch 2.11.0).

**Built**
- `sim/lif_torch_fast.py`: `run_trial` with the reference's signature and return contract plus `spike_cap` / `syn_cap`. The loop body is one method `_Sim.step(t, hits)` over preallocated tensors and a device step counter `t_dev`; Python branches only on static config (options on/off, block windows by step number). Defaults: `spike_cap = min(n, 4096)`, `syn_cap` = sum of the `spike_cap` largest out-degrees of W (cannot overflow unless spike_cap does); T5b sets both from the looming burst.
- Sync sites removed: `nonzero` + `has` -> `torch.nonzero_static(spk, size=spike_cap, fill_value=n)` and mask updates (`where`); `int(lens.sum())` + `repeat_interleave` in the CSC gather -> fixed `syn_cap` gather (`searchsorted` over the column-length cumsum, sentinel column n of length 0, padding reads a dummy entry that adds 0.0 into a dropped row); `pre.numel()` -> ring as a padded `(dly, spike_cap)` tensor, presyn/synaptic updates always run; spikelet `gs.numel()` -> sentinel column of the gap matrix; APL `float(is_kc[idx].sum())` -> 0-d tensors `a_apl`, `kc_rate`; Poisson `ix[rand < p]` -> mask in `where`. `rng='numpy'` still draws on the host and uploads each step (parity mode only). Recording: device buffer `(steps, spike_cap)` compacted after the loop; binning on device.
- Overflow: a device flag, read once after the loop; on overflow a `RuntimeWarning`, `lif_torch_fast.FALLBACKS += 1`, rerun with `lif_torch.run_trial`, `out['fallback'] = True` (dict results always carry `fallback`).
- Save/resume: same state keys and formats as the reference (ring stored unpadded, APL scalars as floats), so states are interchangeable.
- `tests/test_fast_cpu.py` (11 tests): parity x {off, all options, blocks} x {rng numpy, torch} (counts + t/i records exact, T0 activity guards); zero-sync counter; overflow via `spike_cap=1` and `syn_cap=5`; no fallback at default caps; resume (fast split == fast single, and reference state -> fast resume), split at step 1000 inside both blocks with spikes in flight.

**Findings**
- Hypothesis (b) confirmed: fast == reference spike for spike (counts and t/i) on CPU float64 with `rng='numpy'` in all 3 cases. With `rng='torch'` they are also identical in all 3 cases (same `torch.rand` calls on the same generator), so the test asserts exact equality there too; no correlation fallback was needed.
- Zero syncs: over 200 steady steps (`rng='torch'`, gap + APL + presyn, drive on, binning + recording) the dispatch counter sees 0 of `aten::_local_scalar_dense`, `aten::nonzero`, `aten::item`, `aten::is_nonzero`, `aten::masked_select`, and 0 bool-mask `aten::index`/`index_put`; every step issues the same op sequence with the same output shapes. Positive control: the same counter sees `nonzero` and `_local_scalar_dense` in the reference.
- Failure demo (uncommitted scratch edits): padding with neuron n-1 instead of the sentinel n -> all 6 parity tests fail (267 / 267 / 196 / 193 / 201 / 195 neurons differ); writing the ring one slot ahead -> the 3 numpy parity tests fail (116 / 68 / 67 neurons differ).
- CPU timing (not a speed claim; Xeon 2.1 GHz, 4 threads, 2000 steps, median of 3): reference 0.11-0.24 ms/step, fast 0.37-0.69 ms/step at default caps, 0.24-0.51 ms/step at spike_cap 32 / syn_cap 512. The fast path is 2-4x slower on CPU: it does fixed-size work every step and CPU has no syncs to save. GPU timing and graph capture: T5b.
- Local: `pytest -q tests/` 41 passed in ~22 s; new tests ~17 s.
- Gaps for T5b (unverified here): block windows are Python branches on the step number, so a CUDA graph is per set of active drives; no GPU sync check was possible (the dispatch counter cannot see syncs inside C++/CUDA kernels; use `torch.cuda.set_sync_debug_mode` there); hidden H2D from Python scalars in `where` / index assignments on CUDA unverified; graph-safe RNG for `t_gen` not set up; record buffer is steps x spike_cap x 8 bytes.
- No other file under `sim/` changed.

## T5b: CUDA graph + GPU timing of the sync-free step (Day 6 part 2, 2026-11-22)

Brief: [tasks/T5b_brief.md](tasks/T5b_brief.md). Branch `perf/w4-t5b-graph`. Report: [perf/week04_day6.md](../../perf/week04_day6.md). PC (WSL, RTX 5060 Ti, torch 2.11.0+cu128).

**Built**
- `bench/w4_caps.py`: reference on the bundle (looming, rng='torch', float64, 10,000 steps, seeds 0-2); max spikes/step and max gathered synapses/step for W and Wp. Caps = ceil(1.5 x max): `spike_cap = 60`, `syn_cap = 44,483` (Wp needs 3,677).
- `bench/w4_graph.py`: `graph_trial(sim)` (3 eager warm-up steps of the trial on a side stream, one `torch.cuda.graph` capture of `_Sim.step` per active-drive combination, the sim's generator registered with `CUDAGraph.register_generator_state`, one replay per step); timing rows (reference / fast eager / fast graph on cuda float64, float32 and deterministic variants, fast at default caps, fast and reference on cpu), 5 runs, per-step CUDA events rescaled to the host clock plus a second trial without events, peak GPU memory; sync check under `torch.cuda.set_sync_debug_mode('error')` with a positive control and ops per step; full-trial parity (seeds 0-2).
- `bench/w4_graph_plot.py`: bar chart (840 px) and the HTML table the site embeds.
- `tests/test_fast_cuda.py` (skipped without CUDA): sync-debug steady loop + positive control; graph == eager (counts, records, bins, v, g) for all options (2 graphs) and overlapping blocks (3 graphs).
- No change under `sim/` was needed.

**Findings**
- GPU single trial beats the CPU: graph 0.49 [0.43-0.60] ms steady, 0.45 [0.42-0.63] ms all steps, vs CPU 0.64 ms (T1) and 0.55 [0.52-0.59] ms measured in the same session. Reference GPU 1.73 [1.57-1.85], fast eager GPU 1.75 [1.48-1.93], fast eager CPU 1.77 [1.64-1.82] ms.
- Sync check passed (200 steady steps, nothing raised; the reference raises). Capture itself forbids syncs and succeeded. Fast step: 173 aten ops, 150 GPU ops per step (reference ~117/106), which is why eager is not faster than the reference.
- Overflow 0 at the caps in every trial. Default caps: 7.7 ms/step, +166 MB.
- Parity, float64: graph == eager (0 neurons differ, 3 full trials + 5 timed trials), fast == reference on GPU (correlation 1.0, 0 neurons differ, 3 seeds). States differ bitwise (<= 1.7e-16) run to run, eager too (float atomics in `index_add_`), so count identity is observed, not guaranteed. Float32: counts not reproducible run to run; graph vs eager differ in 2 of 3 full trials; identical under `torch.use_deterministic_algorithms(True)`, which costs ~9 ms/step.
- CUDA event times on this WSL box read 4-6% high against the host clock; spans are rescaled per trial.
- Week 5 targets: fusion of the gather/scatter and decay/threshold chain (<= 0.25 ms/step), batching B trials in one graph (8 trials < 1 ms/step).

## T4: compute roofline on the 5060 Ti (Day 5a, 2026-11-22)

Brief: [tasks/T4_brief.md](tasks/T4_brief.md). Branch `perf/w4-t4-roofline`. Report: [perf/week04_day5.md](../../perf/week04_day5.md). PC (WSL, RTX 5060 Ti, torch 2.11.0+cu128, driver 610.88).

**Built**
- `bench/w4_roofline.py`: `torch.matmul` peaks for fp64 / fp32 (TF32 off) / tf32 / bf16 / fp16 at n = 1,024-8,192, 100 timed calls per size with one CUDA event pair each, median + p90 per size, peak = best size's median, `--runs 5`; same-session copy bandwidth (Day 2 method); TF32 flag check; LIF decay FLOP and byte count of `sim/lif_torch._decay` via `TorchDispatchMode` (eager) plus the fused lower bound; derived vendor spec with sources; nvidia-smi clocks/power after each size; host-clock diagnostics. Exit code 5 if a peak is > 10% over spec.
- `bench/w4_roofline_plot.py`: log-log roofline (`perf/results/w4/roofline.png`, 840 px) and the HTML peak table the site embeds.
- No change under `sim/`; no Triton kernel.

**Findings**
- Peaks, median [min-max] of 5 runs, TFLOP/s: fp16 47.67 [47.67-48.08], bf16 48.29 [48.24-48.64], tf32 24.05 [24.05-24.24], fp32 16.88 [16.87-17.27], fp64 0.347 [0.347-0.347]. % of the derived spec: 100.6 / 102.0 / 101.5 / 71.3 / 93.8. No OOM, no errors, nothing > 10% over spec.
- NVIDIA publishes no per-dtype TFLOP/s for the 5060 Ti. Spec derived from the RTX Blackwell whitepaper (RTX 5070 per-SM rates, FP64 = FP32/64) x 36 SMs x 2.57 GHz; consistent with the published 759 AI TOPS. The card runs at 2.62-2.73 GHz under load, hence ~101% on the tensor dtypes; fp32 SIMT is power-limited at 180 W.
- fp32 / fp64 = 48.6 (1/64 by spec). Ridge points at 382 GB/s: fp64 0.91, fp32 44, tf32 63, fp16 125, bf16 126 FLOP/byte.
- LIF decay: 8 FLOP per neuron; fused 49 B (float64, 0.163 FLOP/B) / 33 B (float32, 0.242); eager 244 B / 140 B (14 aten ops). 5.6x below even the fp64 ridge: bandwidth/launch bound. Week 5 decay kernels: judge against 382 GB/s with fused byte counts.
- Copy bandwidth re-measured: 385.9 [385.7-386.1] GB/s (Day 2: 382; NVIDIA: 448).
- Clock: the raw CUDA event times repeat within 1%, while host wall / event sum moves between 0.90 and 1.09 across sessions and the idle-sleep ratio between 1.00 and 1.11. TFLOP/s use events unscaled (unlike T5b's host rescale). A first full run with the idle-sleep correction was discarded.
- > 2x across runs: bf16 and fp16 at n = 1,024 only (bimodal, 15 vs 38-40 TFLOP/s); peaks vary <= 1.02x.
