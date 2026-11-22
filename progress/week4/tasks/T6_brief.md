# Brief T6 (Week 4 perf, Day 7): Week 4 perf write-up

Orchestrator → Dev. Dev sees only this brief. Hypotheses are UNCONFIRMED: check them before building on them.

## Outcome
One page, perf/week04.md, that a beginner can read top to bottom: what one simulator step costs on each device, why, what was fixed this week, and the Week 5 targets set from the numbers. No new experiments except the small fixes listed below.

## Context
- First step: git checkout main && git pull --ff-only in the flybrain clone. Main has T0-T5b and T4 merged: perf/week04_day3.md (overhead budget), perf/week04_day4.md (step anatomy), perf/week04_day5.md (roofline), perf/week04_day6.md (CUDA graph), perf/results/w4/*, progress/week4/perf_log.md, perf/week04_plan.md, docs/plan.md §3b. Read them all first; every number in the write-up must come from one of these files (cite the file).
- Key numbers to verify against the files, not copy from here: reference GPU step ~1.6-1.7 ms steady (WSL, float64); PC CPU 0.55-0.64 ms; CUDA-graph fast step 0.44-0.49 ms; matmul peaks fp64 0.347, fp32 16.9, tf32 24.0, bf16 48.3, fp16 47.7 TFLOP/s; copy bandwidth 382-386 GB/s; LIF decay intensity 0.16 (f64) / 0.24 (f32) FLOP/B fused.
- Not measured this week: the Mac (CPU and MPS) rows; GPU kernel timelines under WSL (nsys sees no CUDA activity with driver 610.88). Say so in the table, do not fill them.

## Build
1. perf/week04.md:
   - A 5-line plain-language summary at the top.
   - Device table: PC CPU, 5060 Ti reference eager, eager + torch.compile (Day 2), sync-free eager, sync-free CUDA graph (f64, f32); Mac CPU / MPS rows marked "not run". ms/step median [min-max], bio-seconds per wall-second, source file per row.
   - Short sections linking the Day 3-6 reports: overhead budget, step anatomy, roofline, CUDA graph, each with its one key chart (existing PNGs, no re-plotting) and 2-3 sentences.
   - "What held / what didn't" for the week's hypotheses (from each day's verdicts).
   - Week 5 targets, each with a number and the evidence behind it: batched trials (SpMV -> SpMM) with a batch-size scaling curve and trials/s target; fusion targets for the graph step (150 GPU ops/step now); the Triton and JAX LIF decay kernels (Charlie writes these himself; agents only check parity and run the harness) judged against the roofline; Mac MPS run.
   - Open issues: float32 non-reproducibility (atomics), WSL stalls, nsys under WSL (sudo apt-get install -y nsight-systems-2026.1.3 untested), the 44 vs 47 test count between WSL and CI (find the cause and state it; fix only if trivial and test-only).
2. Small fixes from QA findings:
   - Commit the Day 2 bench result (~/flybench/flybrain/results/w4/bench/DESKTOP-4C13T5V_20261116-2308.{md,json} if present) under perf/results/w4/ so the 382 GB/s number is in git.
   - Re-render roofline.png so log-axis tick exponents render >= 9 px at 390 px (e.g., larger tick font or plain decimal tick labels); update the site copy so it stays byte-identical.
3. Site: on CharlieZhang1999.github.io (default branch master), add a "Week 4 GPU summary" block at the top of the GPU track section of flybrain/week4/index.html with the device table and the Week 5 targets; replace the roofline PNG with the re-rendered one. Keep _data/flybrain_weeks.yml week 4 status as is unless the science track has already marked the week done.

## Acceptance criteria
- Every number in perf/week04.md traces to a cited file in the repo; no new uncited numbers.
- Device table complete for all PC rows; Mac rows marked not run.
- Week 5 targets each have a number and a cited reason.
- Roofline PNG exponents >= 9 px rendered at 390 px; site PNG and table byte-identical to flybrain's.
- No sim/ changes; cpu-tests green.

## Deliverable
Branch perf/w4-t6-writeup, PR to flybrain (perf/week04.md, the Day 2 result files, the re-rendered PNG, a T6 entry in progress/week4/perf_log.md, this brief as progress/week4/tasks/T6_brief.md). Site PR as above. Summary: built / assumed / unverified.

## Stop condition
Stop and report if two report files disagree on a number by more than their stated ranges (name both).

## QA (fresh subagent; brief + PR links + head SHAs only)
Trace every number in perf/week04.md to its cited file (list any that don't trace); check the device table rows against the source JSON/MD; check the 44 vs 47 explanation; re-run bench/w4_roofline_plot.py on the committed JSON and confirm the PNG matches the committed one; site at 390 px: no horizontal scroll, PNG text (including exponents) >= 9 px, table/PNG byte-identical; no sim/ changes. Findings as severity / what fails / repro / expected vs actual. Merge each PR only when required checks are green on the exact head SHA (flybrain: cpu-tests + gpu-qa; site: gpu-qa).

## Role rules (paste into each subagent's prompt; there are no agent files)
- Dev: validate the hypothesis first; simplest correct solution; never invent APIs or claim untested behaviour; flag gaps in the criteria; report built / assumed / unverified; update progress/week4/perf_log.md in flybrain and the site's flybrain/week4 page; do not sign off or merge.
- QA (fresh subagent, gets this brief + PR links only): run every check yourself; findings as severity / what fails / repro / expected vs actual; say once if a criterion is untestable; do not edit code or commit; merge only when CI cpu-tests and gpu-qa are green on the exact head SHA.
- The PC session only launches Dev, then QA, and relays QA's report to the Orchestrator (this thread). It does not plan, build or judge.
