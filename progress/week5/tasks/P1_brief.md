# Brief P1 (Week 5 perf): B trials in one CUDA graph, float64 reference

Lead → Dev. Dev sees only this brief. Hypotheses are UNCONFIRMED: check them before building on them.
Central-plan row: GPU/perf, Week 5, "batched trials". Team rules: `progress/week5/team/ROLES_perf.md`.

## Outcome
A batched simulator that runs B independent trials over one shared network in one CUDA graph on the
RTX 5060 Ti, in float64, with each trial spike-for-spike equal to running it alone. Answers: how many
trials per wall-second does the GPU give at B = 1, 2, 4, 8, 16, 32, and does B = 8 run under 1 ms per step?

## Why (anchor)
The science track's datasets are many independent trials on one network: `scripts/w5_hidden_state.py`
(120 episodes x calib/shuffle) and `scripts/w5_behaviour_panel.py` (seeds x stimuli) each call
`lif_torch.run_trial(..., device='cpu', rng='torch', dtype=torch.float32, record=True)` once per trial,
**each with its own `blocks`** (stimulus and state drives). These feed the decoders that get checked on
Willett's data in Week 9 and the dog path in Week 10. P1 is useful to them only if per-trial blocks and
recording work in the batch.

## Context
- First step: `git checkout main && git pull --ff-only`. Read `sim/lif_torch_fast.py` (`_Sim`, `_PadCSC`,
  caps, overflow flag + fallback), `bench/w4_graph.py` (`graph_trial`, timing method, sync check),
  `tests/test_fast_cpu.py`, `tests/test_fast_cuda.py`, and `perf/week04.md` "Week 5 targets" item 1.
- Week 4 numbers (float64, looming bundle, `rng='torch'`): fast graph 0.49 [0.43-0.60] ms/step steady,
  ~150 GPU ops/step, launch-bound; caps `spike_cap = 60`, `syn_cap = 44,483` (Wp 3,677); one trial's decay
  state 5.3 MB fits the 32 MB L2, 8 trials (42 MB) do not.
- WSL venv `~/flybench/.venv` (torch 2.11.0+cu128); bundle `~/flybench/flybrain/results/w4/cns_bundle_calib.npz`.
- **Shared PC.** The science track runs one CPU sim at a time on this machine. Before every timed run, check
  for running sims (WSL `ps aux | grep python` and Windows `Get-Process python`), record the result in the
  output JSON, and do not time while one runs. Tests and parity may run alongside.
- Prior art (Scout, `perf/briefs/w5_P1_scout.md`): GeNN batch mode on the same fly model reports
  0.054 / 0.103 / 1.57 ms/step at B = 1 / 8 / 32 (float32, other GPU, unverified); a dense-spike SpMM in
  PyTorch scales linearly with B (4.5 ms/step at B = 8). So: keep the gather event-driven, keep the kernel
  count independent of B (one gather over a flattened (trial, spike) list, no per-trial loops in the graph),
  and give each trial an RNG stream that depends only on its seed (counter-based Philox keyed on
  (seed, step, neuron), or per-trial generators if torch 2.11's graph RNG supports B of them; check). If atomic
  `index_add_` order breaks A2, use one fixed-order accumulation for both B = 1 and B > 1. These are guidance,
  not requirements; the acceptance criteria decide.
- Precision: float64 is the reference for every acceptance criterion (Charlie's rule; precision trade-offs
  are a later week). Science runs float32, so add float32 timing rows as information only.

## Hypotheses (UNCONFIRMED; mark each holds / partly / does not hold)
(a) Per-step time grows far slower than B because the step is launch-bound: B = 8 < 1 ms/step steady.
(b) Trials/s keeps growing to at least B = 16 before memory bandwidth or atomics flatten it.
(c) A batched gather over per-trial padded spike lists (B x spike_cap columns, B x syn_cap entries,
    index_add_ into a flattened (B, n+1) output) keeps each trial's float64 counts identical to its solo run.
(d) Per-trial drive windows can live in device masks indexed by the step counter, so one graph serves B
    trials with different `blocks`.

## Build (new files; `sim/lif_torch_fast.py` only for shared helpers; never touch `sim/lif_torch.py`)
1. `sim/lif_torch_batch.py`: `run_batch(W, params, seeds, blocks_list, ..., spike_cap, syn_cap, rng,
   dtype, device, record)` returning a list of per-trial results in the `lif_torch.run_trial` format
   (counts, t/i records when `record=True`, bins, `fallback`). State as (B, n). Per-trial caps and a
   per-trial overflow flag read once after the loop; overflowing trials are rerun alone with the reference
   (as in T5a). Same options as the fast path (gap, APL, presyn, blocks, Poisson drive).
2. RNG: `rng='numpy'` parity mode (per-trial host draws, uploaded) and `rng='torch'` graph mode. In graph
   mode each trial's stream must depend only on its seed, not on B or its slot. Cite the PyTorch docs for
   whatever generator mechanism is used inside the graph.
3. `bench/w5_batch.py`: ms/step steady + stall share + all-steps, 5 runs, median [min-max], for B = 1, 2, 4,
   8, 16, 32 on the looming bundle (2,000 steps after 1,000 warm-up, Day 6 method), float64; trials per
   wall-second and bio-seconds per wall-second per B; peak GPU memory per B; GPU ops per step; float32
   rows as information. Reference rows in the same session: fast single-trial graph and PC CPU single trial.
   State the clock used (raw events vs host-rescaled) and why.
4. `tests/test_batch_cpu.py` and GPU cases in `tests/test_batch_cuda.py` (skipped without CUDA).
5. `perf/week05_p1.md`: table, scaling chart legible at 390 px, caps, where trials/s stops growing and why
   (evidence, not guesses), hypotheses marked, and a time estimate for one `w5_hidden_state.py` chunk
   if run batched vs today's CPU loop (stated as an estimate).

## Acceptance criteria
- **A1 Parity (CPU, float64, rng='numpy')**: on `tests/synth.py` networks, for B = 3 with three different
  seeds and three different `blocks` lists (options off; all options; overlapping blocks), every batched
  trial's counts and t/i records equal `lif_torch.run_trial` run alone with that seed and blocks.
- **A2 Batch invariance (GPU, float64, rng='torch')**: on the bundle, trial with seed s gives identical
  counts in B = 1 and in B = 8 at any slot (test at least 2 slots), over a full looming trial.
- **A3 Statistical parity (GPU, float64)**: batched B = 8 vs the reference on GPU, same seeds: rate
  correlation >= 0.999 on active neurons per trial; share of differing neurons reported.
- **A4 Graph and syncs**: one graph replay per step for the whole batch with different `blocks` per trial;
  `torch.cuda.set_sync_debug_mode('error')` raises nothing over 200 steady steps (positive control included);
  overflow 0 at the stated caps in every timed trial.
- **A5 Numbers**: scaling table B = 1..32 with medians and min-max, the sim-busy check recorded per run, and
  a plain statement of whether B = 8 is under 1 ms/step steady and where trials/s stops growing.
- **A6 Hygiene**: `pytest -q tests/` green on the PC with CUDA (count stated) and CI `cpu-tests` green;
  `git diff --stat main...HEAD -- sim/` shows only `sim/lif_torch_batch.py` and at most helper moves in
  `sim/lif_torch_fast.py` with `tests/test_fast_*.py` still green.

## Budget / stop
About two working days of Dev time. Stop and report if: per-trial parity (A1/A2) fails and the cause is not
an ordering bug; graph capture fails with per-trial blocks; or B = 8 is above 2 ms/step after the first
working version (report the op count and phase that grew, do not tune further).

## Deliverable
Branch `perf/w5-p1-batched` from main; PR to flybrain (attribution header, assigned to Charlie) with code,
results under `perf/results/w5/`, `perf/week05_p1.md`, a P1 entry in `progress/week5/perf_log.md`, this
brief copied to `progress/week5/tasks/P1_brief.md`. Site PR to `CharlieZhang1999.github.io` (branch
`master`): a Week 5 GPU part in `flybrain/` with the table and the same PNG. Report: built / assumed /
unverified, hypotheses marked.

## QA (fresh subagent; this brief + PR links + head SHAs only)
Re-run A1-A6 from the PR head on the PC: tests, parity, sync check, overflow; re-run `bench/w5_batch.py`
for B = 1, 8, 32 (5 runs) when no science sim is running; each median within Dev's min-max widened by 25%.
Recompute the table from the raw JSON. Findings as severity / what fails / repro / expected vs actual. Set
`gpu-qa` and squash-merge only when `cpu-tests` and `gpu-qa` are green on the exact head SHA.
