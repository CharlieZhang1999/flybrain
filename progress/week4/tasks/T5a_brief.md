# Brief T5a (Week 4 perf, Day 6 part 1): sync-free step, built and checked on CPU

Orchestrator → Dev. Dev sees only this brief. Hypotheses are UNCONFIRMED: check them before building on them.

## Outcome
`sim/lif_torch_fast.py`: a version of the reference step with no host syncs and static shapes, so Part 2 (T5b, on the 5060 Ti) can capture it as a CUDA graph and time it. This part runs entirely on CPU in the cloud: the GPU PC is offline.

## Context
- Repo `CharlieZhang1999/flybrain`, clone `/home/user/flybrain` (shallow; main at c49cbb7 or later: `git pull` first). Python env: system python3 with torch 2.15 (CPU use only here); install `pytest` and anything else from `requirements-ci.txt` you need. CI runs torch **2.11.0** CPU on Python 3.12: any API you use must exist in 2.11 (check the 2.11 docs; CI is the judge).
- Reference: `sim/lif_torch.py` `run_trial` (do not modify anything under `sim/` except adding the new file). Read it fully first. Its rules (`perf/week04_plan.md`): the reference stays untouched; speed work goes in `sim/lif_torch_fast.py`; fixed caps (spikes per step, gathered synapses per step) with an overflow flag that falls back to the slow path; the fast path must keep save/resume.
- Measured on the GPU (T1/T3, `perf/week04_day3.md`, `perf/week04_day4.md`): 10 syncs/step. Sync sites in the reference: `int(lens.sum())` in `_CSC.gather`; `torch.nonzero` + `idx.numel()`/`has`; `pre.numel()`; the spikelet `gs.numel()`; `float(is_kc[idx].sum())` for APL; with `rng='numpy'`, the NumPy Poisson draw + upload each step; Python `if` on tensor sizes.
- Tests: `tests/synth.py` (`make_network`, `params`), `tests/test_parity_cpu.py`, `tests/test_resume_cpu.py` show how parity and resume are tested. CI job `cpu-tests` = `pytest -q tests/`.

## Hypothesis (UNCONFIRMED)
(a) Every sync can be removed with static-shape equivalents: e.g. `torch.nonzero_static(mask, size=cap, fill_value=n)` (verify it exists and behaves this way in 2.11), a padded column list with a sentinel column of length 0, a fixed `max_syn` gather with padding that adds 0.0 into a dummy row n, APL `kc_rate` kept as a 0-d tensor, the spikelet set and Poisson hits handled as masks. (b) If the padded gather keeps the reference's order (spiking neurons ascending, entries in CSC order), then on CPU float64 with `rng='numpy'` the fast path is **spike-for-spike identical** to the reference, because `index_add_` on CPU sums in index order and padding adds exact zeros. If (b) fails, find out why before relaxing anything; report it.
(c) With `rng='numpy'` the NumPy draw is inherently a host round trip; that mode is for parity only. `rng='torch'` is the sync-free mode.

## Build
1. `sim/lif_torch_fast.py` with `run_trial(...)`: same signature and return contract as the reference (incl. `record`, `bin_steps`, `bin_groups`, `state`, `return_state`, `blocks`, `gap`, `apl`, `presyn`, `t_stim`), plus `spike_cap=None, syn_cap=None` (defaults: derive simply, e.g. from n and max out-degree, and document; T5b sets them from the measured looming burst). Structure the loop body as one function `step(...)` over preallocated tensors and a device step counter, with no `.item()`, `int()/float()/bool()` on tensors, no `nonzero` without static size, and no Python branches on tensor values, so it is capturable later. Python branches on static config (which options are on, block windows by step number) are fine.
2. Overflow: a device flag set when a cap is exceeded; checked once after the loop (one sync per trial is fine). On overflow, rerun with the reference `lif_torch.run_trial` and return its result, with a warning; expose that it happened (e.g. `out['fallback']=True` when a dict is returned, and a module-level counter).
3. Recording on device without per-step syncs (e.g. a preallocated (nsteps, spike_cap) buffer compacted at the end, or binned only); `t`/`i` must match the reference format and order.
4. Save/resume with the same contract as the reference (state dict may hold fast-path-specific tensors; document the keys).
5. `tests/test_fast_cpu.py` on the synthetic network:
   - fast vs reference, CPU float64 `rng='numpy'`: identical counts AND identical `t`/`i` records, for all-off, all-on (gap+APL+presyn+blocks), and blocks-only; same minimum-activity guard as `test_parity_cpu.py`.
   - fast `rng='torch'` vs reference `rng='torch'`: both run; report whether they are identical; if not identical, require rate correlation ≥ 0.999 on active neurons over several seeds (state which).
   - zero syncs per step: a `TorchDispatchMode` counter over the steady loop with `rng='torch'` sees no `aten::_local_scalar_dense`, `aten::nonzero`, `aten::item`, `aten::is_nonzero` (list exactly what you count); assert 0, outside the one end-of-trial check.
   - overflow: a tiny `spike_cap` triggers the fallback and still returns reference-identical results.
   - resume: fast split run == fast single run (counts and records), split inside a block with spikes in flight, as in `test_resume_cpu.py`.
   - all new tests together < 30 s on CI.
6. A short CPU timing note only (ms/step fast vs reference on CPU, synthetic network); not a speed claim. GPU timing and graph capture are T5b.

## Acceptance criteria
- `pytest -q tests/` passes locally and in CI `cpu-tests` on the PR head.
- Exact parity (counts + records) for the three `rng='numpy'` cases; zero-sync assertion passes; overflow and resume tests pass.
- Demonstrate once locally that the parity test fails when the fast path is broken (e.g., in an uncommitted scratch edit, drop the padding sentinel or shift the ring by one) and report the output.
- `git diff --stat origin/main...HEAD -- sim/` shows only the new `sim/lif_torch_fast.py`.

## Deliverable
Branch `perf/w4-t5a-syncfree`, PR to flybrain titled "Sync-free step (CPU parity), Day 6 part 1", body = built / assumed / unverified, plus the list of every sync site removed and how. Progress updates (required): a T5a entry in `progress/week4/perf_log.md`; this brief saved as `progress/week4/tasks/T5a_brief.md`; and on the site repo `CharlieZhang1999/CharlieZhang1999.github.io` (clone at `/home/user/charliezhang1999.github.io`, branch **`claude/project-thread-5rbuq6`**), a short "Day 6 · part 1: sync-free step" paragraph in the GPU track section of `flybrain/week4/index.html` (what was built, parity result, "GPU timing pending"), as a separate PR. Do not merge either PR.

## Stop condition
Stop and report if exact parity (b) fails and you can't explain it after one honest attempt; if a needed static-shape API is missing in torch 2.11; or if a sync can't be removed without changing the model (name it, keep that one, continue).

## Role rules
- Dev: validate the hypothesis first; simplest correct solution; never invent APIs or claim untested behaviour; flag gaps in the criteria; report built / assumed / unverified; update the progress log and site page; do not sign off or merge.
- GitHub: use the GitHub MCP tools (`mcp__github__*`, load via ToolSearch) for PRs; git push from the clone with `git -c push.negotiate=true push -u origin BRANCH`. No `gh` CLI. End PR bodies with:
  🤖 Generated with [Claude Code](https://claude.com/claude-code)

  https://claude.ai/code/session_019wmqTWeAyku7UsXSpiUkDF
  and commit messages with:
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_019wmqTWeAyku7UsXSpiUkDF
- `/mnt/project-files` is a network mount: put a trailing slash on its path, never search the whole folder.

## QA (fresh subagent; brief + PR links + head SHAs only)
Check out the head; run `pytest -q tests/`; confirm CI `cpu-tests` green on the exact head SHA; reproduce the failure demo; read `step()` and confirm no host sync or data-dependent shape remains (list any); confirm the zero-sync counter counts the right ops; confirm no other `sim/` change; site page at 390 px has no horizontal scroll. Findings as severity / what fails / repro / expected vs actual. Note: the cloud can't set the `gpu-qa` commit status; QA records its verdict in a PR comment (ending with the footer `---` / `_Generated by [Claude Code](https://claude.ai/code)_`) and merges only when `cpu-tests` is green on the exact head SHA and QA found no blocking issue.
