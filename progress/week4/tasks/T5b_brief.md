# Brief T5b (Week 4 perf, Day 6 part 2): CUDA graph + GPU timing of the sync-free step

Orchestrator → Dev. Dev sees only this brief. Hypotheses are UNCONFIRMED: check them before building on them.

## Outcome
ms per step of sim/lif_torch_fast.py on the RTX 5060 Ti, eager and replayed as a CUDA graph, against the reference (~1.6-1.7 ms steady, WSL float64) and the PC CPU (0.64 ms). Answers: does a single trial on the GPU beat the CPU?

## Context
- First step: in C:\Users\Charlie\Desktop\Workspace\flybrain (or the WSL clone), git checkout main && git pull --ff-only. Main includes T5a (flybrain#12, merge dbd74da): sim/lif_torch_fast.py (_Sim, _prepare(), advance(t), step(t, hits), caps spike_cap/syn_cap, overflow flag read once per trial, fallback to the reference) and tests/test_fast_cpu.py. Read both first.
- WSL venv ~/flybench/.venv (torch 2.11.0+cu128). Bundle ~/flybench/flybrain/results/w4/cns_bundle_calib.npz, looming stimulus as in bench/w4_anatomy.py. WSL timing is bimodal at times: N=5 runs, median + min-max, steady steps (<= 5 ms) with stall fraction separate, same aggregation as T1/T3.
- Open items T5a left for this task:
  1. Zero-sync was shown only at the Python-dispatch level on CPU. Syncs inside CUDA ops (e.g. hidden H2D copies of Python scalars, nonzero_static on CUDA) are unchecked.
  2. Block windows are Python branches on the step number, so one graph covers one set of active drives.
  3. Graph-safe RNG for the torch generator is not set up.
  4. Default caps (spike_cap = min(n, 4096), syn_cap from the largest out-degrees) and the (steps, spike_cap) record buffer may be heavy on 165k neurons.
  5. Site wording fix: the "Day 6 · part 1" paragraph on the site says each of the 10 waits is replaced; add that rng='numpy' (parity mode) still keeps the NumPy draw as a host round trip.

## Hypothesis (UNCONFIRMED)
(a) With rng='torch', torch.cuda.set_sync_debug_mode('error') raises nothing inside the steady loop of the fast path. (b) Eager fast path on CUDA is already faster than the reference (fewer syncs) but still launch-bound. (c) CUDA-graph replay of step gets to ~0.5-0.6 ms/step (T3's projection), below the CPU's 0.64 ms. (d) Caps from the measured looming burst (max spikes/step and max gathered synapses/step, plus a margin) cut memory and time without overflow.

## Build (new files only, plus sim/lif_torch_fast.py changes only if needed for capture; never touch sim/lif_torch.py)
1. Measure the caps: run the reference on the bundle (looming, rng='torch', 3 seeds) and log max spikes/step and max gathered synapses/step (W and presyn Wp). Set caps = max x 1.5 (round up), document, and use them for all GPU runs; overflow count must be 0.
2. Sync check on CUDA: run the fast path's steady loop under torch.cuda.set_sync_debug_mode('error') (rng='torch'); fix any sync it finds in lif_torch_fast.py (keeping T5a's CPU parity tests green) or report it.
3. Graph: capture step with torch.cuda.graph (static inputs; the step counter as a device tensor; one graph per active-drive combination, or move drive windows into device masks, whichever is simpler; register the generator for graph-safe RNG as the PyTorch CUDA Graphs docs describe, and cite the doc section). Replay for the trial.
4. bench/w4_graph.py: ms/step for reference eager, fast eager, fast graph, all float64 rng='torch' on cuda, plus fast eager on cpu; N=5 runs; also float32 rows if cheap. Peak GPU memory per row.
5. Parity on GPU: fast graph vs fast eager with the same seed: identical counts expected (same ops, same RNG); fast vs reference on GPU: rate correlation >= 0.999 on active neurons and share of differing neurons (float64 atomics may reorder sums).
6. perf/week04_day6.md: the table, a bar chart legible at 390 px, caps used, the sync-check result, and 3-5 conclusions with Week 5 targets (batching, fusion).

## Acceptance criteria
- Sync check passes (or every remaining sync is named with its cause).
- Graph replay works for a full looming trial with overflow 0; graph vs eager counts identical; fast vs reference parity row reported.
- ms/step table with medians and min-max; the report states which hypothesis parts held, and plainly whether the GPU single trial beats 0.64 ms.
- pytest -q tests/ green locally and CI cpu-tests green; git diff --stat main...HEAD -- sim/ shows at most sim/lif_torch_fast.py.

## Deliverable
Branch perf/w4-t5b-graph, PR to flybrain with code, results under perf/results/w4/, perf/week04_day6.md, a T5b entry in progress/week4/perf_log.md, this brief saved as progress/week4/tasks/T5b_brief.md. Site PR to CharlieZhang1999.github.io (default branch master): a "Day 6 · part 2: CUDA graph" part in the GPU track section of flybrain/week4/index.html (table + PNG byte-identical to flybrain's), plus the wording fix in item 5 of the context. Summary: built / assumed / unverified.

## Stop condition
Stop and report if graph capture fails in WSL (give the error), if a sync can't be removed without changing the model, or if graph and eager counts differ.

## QA (fresh subagent; brief + PR links + head SHAs only)
Re-run bench/w4_graph.py (5 runs) on the head; each row's median falls inside Dev's min-max widened by 25%; re-run the sync check; confirm graph vs eager identical counts and the parity row; confirm overflow 0 at the stated caps; pytest -q tests/; only allowed sim/ changes; site page at 390 px: no horizontal scroll, PNG text readable, table/PNG byte-identical to flybrain's. Findings as severity / what fails / repro / expected vs actual. Merge each PR only when required checks are green on the exact head SHA (flybrain: cpu-tests + gpu-qa; site: gpu-qa).

## Role rules (paste into each subagent's prompt; there are no agent files)
- Dev: validate the hypothesis first; simplest correct solution; never invent APIs or claim untested behaviour; flag gaps in the criteria; report built / assumed / unverified; update progress/week4/perf_log.md in flybrain and the site's flybrain/week4 page; do not sign off or merge.
- QA (fresh subagent, gets this brief + PR links only): run every check yourself; findings as severity / what fails / repro / expected vs actual; say once if a criterion is untestable; do not edit code or commit; merge only when CI cpu-tests and gpu-qa are green on the exact head SHA.
- The PC session only launches Dev, then QA, and relays QA's report to the Orchestrator (this thread). It does not plan, build or judge.
