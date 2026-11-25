# Week 5 perf/GPU log

Team rules: [team/ROLES_perf.md](team/ROLES_perf.md). Briefs: `perf/briefs/w5_Pn_<name>.md`, copied to
[tasks/](tasks/).

## P1: B trials in one CUDA graph, float64 reference (2026-11-25)

Brief: [tasks/P1_brief.md](tasks/P1_brief.md) (Scout one-pager: `perf/briefs/w5_P1_scout.md`). Branch
`perf/w5-p1-batched`. Report: [perf/week05_p1.md](../../perf/week05_p1.md). PC (WSL, RTX 5060 Ti,
torch 2.11.0+cu128, driver 610.88).

**Built**
- `sim/lif_torch_batch.py`: `run_batch(W, params, seeds, blocks_list, ...)` -> list of per-trial results in the
  `lif_torch.run_trial` format (counts, t/i, binned, fallback). State (B, n); one `nonzero_static` over the
  flattened (B x n) spike mask placed into per-trial padded lists; one batched padded CSC gather and one
  `index_add_` into a flattened (B, n + 1) output; drive windows as device masks over a flat (trial, drive,
  neuron) entry list; per-trial caps and (B,) overflow flag, overflowing trials rerun alone with the
  reference. rng='numpy' (host draws per trial, parity) and rng='torch' (one generator per trial, the
  reference's draws; graph keyed by the set of drawing pairs; `draws='always'` = one graph, own stream).
  `graph_loop` (CUDA graph, one replay per step, shared pool), `eager_loop`. Helpers `_PadCSC` / `_deg_bound`
  imported from `sim/lif_torch_fast.py` (unchanged); `sim/lif_torch.py` untouched.
- `bench/w5_batch.py` (scaling B = 1..32 f64 + f32, fast-graph and CPU reference rows, ops/step, gated phases,
  A2/A3/A4 checks, sim-busy check per row, resumable), `bench/w5_batch_plot.py` (PNG + site table),
  `bench/w5_realizations.py` (A3 feasibility of an own RNG stream).
- `tests/test_batch_cpu.py` (16 tests), `tests/test_batch_cuda.py` (5, skipped without CUDA).

**Findings**
- B = 8: **2.04 [1.97-2.09] ms/step steady**, float64: not under 1 ms, and above the brief's 2 ms stop line
  (median). Stop condition hit; not tuned after the first working version.
- Trials/s (1 s trials, all-steps): 0.216 / 0.297 / 0.383 / 0.395 / 0.400 / 0.386 at B = 1 / 2 / 4 / 8 / 16 / 32;
  flat from B = 8. Same session: fast single-trial graph 0.231 (0.44 ms/step), PC CPU single trial 0.168
  (0.60 ms/step). Float32 (info): 0.531 at B = 8, 0.621 at B = 16.
- Why: GPU ops/step 173 -> 180 -> 204 (B = 1 / 8 / 32, +1 per trial for the RNG draw); per-phase GPU time grows
  instead: padded gathers (98,044 entries per trial per step whatever the spike count; the spikelet cap is the
  full gap nnz) linear in B, decay superlinear from B = 8 (0.33 -> 1.22 -> 3.04 ms), consistent with the state
  leaving the 32 MB L2.
- Parity: A1 exact (CPU); A2 0 neurons differ (B = 1 vs B = 8, slots 0/7, 3/4, 7/0); A3 corr 1.0, 0 differ vs GPU
  reference (8 seeds). An own RNG stream could not pass A3: two realizations correlate 0.986-0.996.
- A4: 11 graphs (draws='active') / 1 graph (draws='always') for 8 different blocks lists, one replay per step;
  sync debug 'error' clean over 200 steady steps (eager and graph), reference raises; overflow 0 everywhere.
- Hypotheses: (a) does not hold, (b) does not hold, (c) holds (observed), (d) partly holds.
- Tests on the PC (WSL venv): `pytest -q tests/` 86 passed with CUDA; with `CUDA_VISIBLE_DEVICES=""` 78 passed,
  8 skipped (3 `test_fast_cuda.py` + 5 `test_batch_cuda.py`), which is what CI cpu-tests should report.
- Estimate: one hidden-state chunk (480 bio-s) ~15 min batched f32 B = 8 vs 3.26 h logged (estimate).
