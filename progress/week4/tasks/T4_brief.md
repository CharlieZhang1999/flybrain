# Brief T4 (Week 4 perf, Day 5a): compute roofline on the 5060 Ti

Orchestrator → Dev. Dev sees only this brief. Hypotheses are UNCONFIRMED: check them before building on them.

## Outcome
The measured compute ceiling of the RTX 5060 Ti (fp64, fp32, tf32, bf16, fp16 matmul), joined with the measured 382 GB/s copy bandwidth into one roofline plot. Week 5's kernels (LIF decay in Triton and JAX, then the fused step) are reported as "% of roofline" against it.

## Context
- First step: git checkout main && git pull --ff-only in the flybrain clone (after T5b has merged).
- WSL venv ~/flybench/.venv: torch 2.11.0+cu128, triton 3.6.0, sm_120, 36 SMs, 15.9 GB, 32 MB L2.
- Copy bandwidth 382 GB/s measured on Day 2 by bench/w4_bench.py (results under results/w4/bench/ on the PC, perf/results/w4/ in git).
- WSL timing is bimodal at times (see perf/week04_day3.md Variability): use N=5 runs and report median + min-max, as in T1/T3.
- This task does NOT write any Triton kernel. Charlie writes kernels himself; leave that alone.

## Hypothesis (UNCONFIRMED)
(a) Large square matmuls (n = 4096-8192) reach >= 70% of the vendor's spec for each dtype; look up the spec from NVIDIA's published numbers for the 5060 Ti and cite the source, or say plainly that you could not find it. (b) fp64 is roughly 1/64 of fp32 on this consumer part, so the simulator's float64 path has a very low compute roof. (c) The roofline ridge point (FLOP/byte where compute meets 382 GB/s) is far above the LIF step's arithmetic intensity, so the step is memory/overhead bound, not compute bound.

## Build (new files only; do not modify sim/)
1. bench/w4_roofline.py: torch.matmul on square sizes 1024, 2048, 4096, 8192 (skip any that OOM, say so) for fp64, fp32 (torch.backends.cuda.matmul.allow_tf32=False), tf32 (=True), bf16, fp16. Warm up, then time with CUDA events, >=100 repeats per size; TFLOP/s = 2n^3/t. Report median and p90 per size; the peak is the best size's median.
2. Roofline plot perf/results/w4/roofline.png: log-log, bandwidth line at 382 GB/s, one flat roof per dtype, ridge points labelled. Mark the LIF decay op's arithmetic intensity (count its FLOPs and bytes per neuron from sim/lif_torch.py _decay, float64 and float32) as vertical lines. Legible at 390 px (fonts >= 9 px rendered).
3. JSON + MD results in perf/results/w4/ (roofline_*.json, roofline_*.md), with versions logged (torch, CUDA, driver, WSL).

## Acceptance criteria
- All 5 dtypes measured on cuda, N=5 runs, median + min-max of the per-run peak; any run-to-run variation > 2x reported, not averaged away.
- The report states which hypothesis parts held, with numbers (% of spec per dtype; fp64:fp32 ratio; ridge points vs LIF decay intensity).
- perf/week04_day5.md: the peak table, the roofline PNG, and 3-5 conclusions on what roof each Week 5 kernel should be judged against.
- No sim/ changes; cpu-tests green (the new script must not be imported by tests).

## Deliverable
Branch perf/w4-t4-roofline, PR to flybrain with code, results, perf/week04_day5.md, a T4 entry in progress/week4/perf_log.md, and this brief saved as progress/week4/tasks/T4_brief.md. Site PR to CharlieZhang1999.github.io (default branch master): add a "Day 5 · compute roofline" part to the GPU track section of flybrain/week4/index.html (peak table + PNG, byte-identical to flybrain's). Summary: built / assumed / unverified.

## Stop condition
Stop and report if a dtype's matmul errors on sm_120 (report the error, skip that dtype, continue), or if measured peaks exceed the vendor spec by > 10% (the timing is wrong).

## QA (fresh subagent; brief + PR links + head SHAs only)
Re-run bench/w4_roofline.py (5 runs) on the head; each dtype's peak falls inside Dev's min-max widened by 25%; recompute TFLOP/s and ridge points from the JSON; recompute the LIF decay intensity from _decay's source; no sim/ changes; site page at 390 px: no horizontal scroll, PNG text readable, table/PNG byte-identical to flybrain's. Findings as severity / what fails / repro / expected vs actual. Merge each PR only when required checks are green on the exact head SHA (flybrain: cpu-tests + gpu-qa; site: gpu-qa).

## Role rules (paste into each subagent's prompt; there are no agent files)
- Dev: validate the hypothesis first; simplest correct solution; never invent APIs or claim untested behaviour; flag gaps in the criteria; report built / assumed / unverified; update progress/week4/perf_log.md in flybrain and the site's flybrain/week4 page; do not sign off or merge.
- QA (fresh subagent, gets this brief + PR links only): run every check yourself; findings as severity / what fails / repro / expected vs actual; say once if a criterion is untestable; do not edit code or commit; merge only when CI cpu-tests and gpu-qa are green on the exact head SHA.
- The PC session only launches Dev, then QA, and relays QA's report to the Orchestrator (this thread). It does not plan, build or judge.
