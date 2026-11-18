=== BRIEF T1 (Week 4 perf, Day 3): where the GPU step's 3.3 ms goes (overhead budget) ===
Orchestrator -> Dev. Hypotheses are UNCONFIRMED: check them before building on them.

Outcome: a table that explains the measured 3.3 ms per 0.1-ms step of sim/lif_torch.py on the RTX 5060 Ti (WSL) from first-principles costs: launch, sync, small copies, Python dispatch. It sets the Week 5 targets.

Known facts (measured 2026-11-16 on this PC, results/w4/bench/DESKTOP-4C13T5V_20261116-23*):
- GPU: 3.3 ms/step eager float64, 2.1-2.8 ms with torch.compile of the decay; CPU (6 threads): 0.64 ms/step.
- Copy bandwidth 382 GB/s; one step moves only a few MB.
- torch.profiler can't see GPU kernels under WSL (CUPTI_ERROR_INVALID_DEVICE); CPU-side op counts do work.

Repo facts from T0 (merged 2026-11-18, PR #3, merge 4eab970):
- Pull main first in C:\Users\Charlie\Desktop\Workspace\flybrain (the Windows clone is still at the pre-merge commit). Work from a WSL clone if faster.
- sim/lif_torch.py on main HAS save/resume (state=/return_state=, commit 3472c84 via PR #2). Leave it untouched. The bench-pack copy that produced 3.3 ms differs only by that code; measure with main's version (put the repo's sim/ on the path, the pack's bundle as data) and, if main's ms/step differs from 3.3 ms by more than 10%, report both. (Correction 2026-11-18: an earlier version of this brief said save/resume was absent; that was wrong.)
- The model bundle is not in git: use ~/flybench/flybrain/results/w4/cns_bundle_calib.npz from the bench pack. If results/ is gitignored, commit the JSON/MD/PNG under perf/results/w4/ instead and say so.
- tests/ + CI cpu-tests exist now; keep them green.

Root-cause hypothesis (UNCONFIRMED, validate first): the step is host-bound. Cost ~ N_ops x (Python dispatch + launch) + N_sync x sync round trip + N_h2d x small copy. Suspected sync sites in sim/lif_torch.py: int(lens.sum()) in _CSC.gather; torch.nonzero and idx.numel(); pre.numel(); float(is_kc[idx].sum()) for APL; and with rng='numpy', a NumPy->GPU upload of Poisson hits each step. WSL may make each sync more expensive than native. If the numbers say otherwise (e.g., a single GPU kernel dominates), say so plainly.

Build (new files only; do not modify sim/): bench/w4_overhead.py, run in WSL ~/flybench/.venv on cuda and on cpu, writing JSON + Markdown:
1. Launch cost: 10,000 tiny ops (x.add_(1) on a 1-element tensor), timed host-side without sync (enqueue cost per op) and with a final sync (throughput).
2. Sync round trip: torch.cuda.synchronize() on an idle GPU; .item() on a scalar after one tiny op.
3. Sync-forcing ops at simulator size (n = 165,122): torch.nonzero on a bool vector with 0 / 100 / 2,000 true entries; int(t.sum()).
4. Small host->device copy: torch.as_tensor(np.int64 array of 10 / 100 / 1,000, device='cuda'), pageable and pinned.
5. CUDA graph: 50 tiny kernels launched one by one vs captured once with torch.cuda.graph and replayed; time per replay.
6. Per-step counts in the real simulator: run 200 steps of lif_torch.run_trial on the bundle (looming, rng='torch' and rng='numpy'); count per step: aten ops dispatched, ops that sync (aten::_local_scalar_dense, aten::nonzero, aten::item, D2H copies), H2D copies. Use torch.profiler CPU activity or a TorchDispatchMode counter.
7. Budget: predicted ms/step = sum of count x unit cost, next to measured ms/step for the same run, for rng='numpy' and rng='torch'.
Every timing: median and p90 of >= 1,000 repeats for us-scale items, >= 100 otherwise; log torch/CUDA/driver/WSL versions.

Acceptance criteria:
- All 7 items reported on cuda; items 1-4 and 6 also on cpu.
- Predicted vs measured ms/step agree within +/-20% for at least one rng mode, or the report names what the gap is (with evidence).
- The report states which hypothesis parts held and which did not.
- Nothing under sim/ changes; existing bench results untouched; cpu-tests stays green.
- perf/week04_day3.md: the budget table, a stacked bar (PNG) of predicted cost by category, and 3-5 bullet conclusions with the concrete Week 5 targets they imply (e.g., "remove k syncs/step -> expect x ms").

Deliverable: branch perf/w4-t1-overhead, PR to main in flybrain with code, results, PNG, perf/week04_day3.md, a T1 entry in progress/week4/perf_log.md and progress/week4/tasks/T1_brief.md. Then a second PR to CharlieZhang1999/CharlieZhang1999.github.io (local clone C:\Users\Charlie\Desktop\Workspace\CharlieZhang1999.github.io, default branch master): a "GPU track" section on the Week 4 page (flybrain/week4/index.html, created if missing, following flybrain/README.md in that repo, plus the _data/flybrain_weeks.yml entry for week 4 with status: in progress) showing the budget table and the stacked bar. Report built / assumed / unverified.

Stop condition: stop and report if the bundle or sim/lif_torch.py on main differs from these assumptions; if CUDA graph capture fails in WSL, report the error, skip item 5 and continue; if a measurement varies by >2x between repeats, report it, don't average it away.

QA (fresh subagent; this brief + both PR links + head SHAs only): re-run python bench/w4_overhead.py on the head commit on the 5060 Ti; confirm each reported number reproduces within +/-25% (median); recompute the budget from the JSON; confirm the +/-20% claim or the stated gap; confirm no sim/ changes; open the website page locally (static file) and check it renders at phone width (e.g. headless Chromium/Playwright at 390 px if available, otherwise say it's untested). Findings as severity / what fails / repro / expected vs actual. Merge each PR only if all required checks are green on the exact head SHA: for flybrain, CI cpu-tests + QA's gpu-qa status; for the site repo (no CI), QA's gpu-qa status alone, meaning the page checks passed.
=== END BRIEF ===
