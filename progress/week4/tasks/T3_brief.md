=== BRIEF T3 (Week 4 perf, Day 4): step anatomy on the 5060 Ti ===
Orchestrator -> Dev. Hypotheses are UNCONFIRMED: check them before building on them.

Outcome: per-phase wall and GPU time of one steady simulator step, plus a real GPU timeline, so Week 5 knows which phases to make sync-free and graph-capture first.

Context (from T1, merged in flybrain#4; see perf/week04_day3.md):
- Steady step ~1.65-1.73 ms on CUDA (WSL, float64, looming); ~106 kernels, 10 syncs, ~117 aten ops per step; ~11 ms stalls on 3-11% of steps.
- Sync source lines in sim/lif_torch.py on main: 43, 208, 211, 212, 229, 231, 234, 242, 243, 256 (re-check; lines may have moved).
- torch.profiler can't see GPU kernels under WSL (CUPTI_ERROR_INVALID_DEVICE).
- First step: git checkout main && git pull --ff-only in the Windows clone.

Root-cause hypothesis (UNCONFIRMED): (a) the synaptic gather/scatter phase (_CSC.gather + index_add_) and the spike phase (nonzero + the 4 scalar index assignments) hold most of the step's wall time, because they contain 6 of the 10 syncs. (b) Actual GPU busy time per step is small (well under 50% of wall time); T1's "~20% utilisation" was never measured.

Build (do not modify sim/):
1. bench/w4_anatomy.py: an instrumented copy of the lif_torch.run_trial step loop with phases: decay, gap, threshold+nonzero, spike bookkeeping (last/free), presyn, synaptic gather+scatter, Poisson drive, spikelets, APL, reset, ring. Time each phase with torch.cuda.Events and wrap each in torch.cuda.nvtx.range. Drift guard: over 300 steps from the same seed and bundle (rng='numpy', float64), the instrumented loop's spike counts must equal lif_torch.run_trial's exactly; fail loudly otherwise.
2. Per phase: median and p90 wall ms (host-side perf_counter around the phase, no extra sync) and GPU ms (events); syncs in the phase (torch.cuda.set_sync_debug_mode('warn')); kernels in the phase (TorchDispatchMode count). Steady steps only (<= 5 ms), stall fraction separately. 5 runs, median + min-max, same aggregation style as T1.
3. GPU utilisation: sample nvidia-smi --query-gpu=utilization.gpu --format=csv -lms 100 during a 2 s run; report median %. Also report sum of GPU-event time / wall time per step.
4. Timeline: if nsys is available (WSL or native Windows), capture 200 steady steps with NVTX ranges and save the .nsys-rep plus a screenshot of one step. If nsys is not installed, do NOT install it: report the exact install command for Charlie and continue without it.
5. T1 follow-ups: make the T1 PNG legible at 390 px (larger fonts or horizontal layout) and remove or replace the unmeasured "~20% utilisation" sentence in perf/week04_day3.md with item 3's number.

Acceptance criteria:
- Drift guard passes (exact counts over 300 steps).
- Sum of per-phase wall medians within +/-15% of the steady step median measured in the same run.
- Report says which hypothesis parts held, naming the top 3 phases by wall time and their sync counts.
- GPU utilisation and GPU/wall ratio reported with their range across runs.
- perf/week04_day4.md: phase table, a stacked bar legible at 390 px, the timeline screenshot or the reason it's missing, and 3-5 conclusions with Week 5 targets per phase.
- No sim/ changes; cpu-tests green.

Deliverable: branch perf/w4-t3-anatomy, PR to flybrain with code, results under perf/results/w4/, perf/week04_day4.md, T3 entry in progress/week4/perf_log.md, brief saved as progress/week4/tasks/T3_brief.md. Site PR: add a "Day 4 - step anatomy" part to the GPU track section of flybrain/week4/index.html (table + PNG, byte-identical to flybrain's) and replace the Day 3 PNG with the legible one.

Stop condition: stop and report if the drift guard fails after one honest attempt to match the reference loop, or if CUDA events inside the loop change the steady step time by more than 25% versus T1's 1.65-1.73 ms.

QA (fresh subagent; brief + PR links + head SHAs only): re-run bench/w4_anatomy.py (5 runs) on the head; drift guard passes; each phase median falls inside Dev's min-max widened by 25% (values Dev flags as varying >2x are exempt but must be listed); sum of phases within +/-15% of steady; utilisation reproduces within +/-15 percentage points; no sim/ changes; site page at 390 px: no horizontal scroll, PNG text readable (>= 9 px rendered), table/PNG byte-identical to flybrain's. Merge each PR only when required checks are green on the exact head SHA (flybrain: cpu-tests + gpu-qa; site: gpu-qa).
=== END BRIEF ===
