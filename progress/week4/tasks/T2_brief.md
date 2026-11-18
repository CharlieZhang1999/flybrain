=== BRIEF T2 (Week 4 perf): parity test for save/resume ===
Orchestrator -> Dev. Hypotheses are UNCONFIRMED: check them before building on them.

Outcome: CI cpu-tests also guards save/resume, so the batched and graph versions in Weeks 5-6 can be checked against a resumable reference.

Context:
- First step: in C:\Users\Charlie\Desktop\Workspace\flybrain, git checkout main && git pull --ff-only (the clone is still on the merged T1 branch). Leave the untracked progress/week4/w4_bench_pack.zip alone.
- sim/lif_torch.py on main has state= / return_state= (commit 3472c84, PR #2). sim/lif_numpy.py has the same (Week 3).
- Hypothesis (unconfirmed): a run to T split at T/2 (first call with return_state=True, record=True; second call with state= and params['t_run']=T) returns, in total, exactly the same per-neuron spike counts and spike (t, i) records as one run to T, for both simulators, on CPU float64 with rng='numpy'. Check the docstrings for exact semantics (e.g. the resumed call returns only new spikes) before writing assertions.

Build:
- tests/test_resume_cpu.py using the synthetic network from tests/synth.py:
  1. NumPy: split run == single run (counts and spike records).
  2. Torch (CPU, float64, rng='numpy'): split run == single run.
  3. Torch split run == NumPy single run.
  Each case with gap + APL + presyn + blocks on, and a split point inside a block while spikes are in flight (non-empty delay ring). Keep T0's minimum-activity guard.
- Do not modify sim/. If the hypothesis fails, that is a real bug: stop and report with a minimal repro; do not loosen the test.

Acceptance criteria:
- pytest -q tests/ passes locally (WSL) and in CI on the PR head; the new tests run < 30 s on CI.
- Demonstrate once locally that the test fails if resume is broken (e.g. in an uncommitted scratch copy, drop the ring restore) and report the output.
- No sim/ changes.

Deliverable: branch perf/w4-t2-resume-test, PR "Tests: save/resume parity", T2 entry in progress/week4/perf_log.md, brief saved as progress/week4/tasks/T2_brief.md. No site update.

Stop condition: stop and report if resume semantics differ between the two simulators, or exact parity fails.

QA (fresh subagent; brief + PR link + head SHA only): check out the head; run pytest -q tests/; confirm CI green on the exact head SHA; reproduce the failure demo; confirm the split point falls inside a block with a non-empty ring (inspect the saved state); no sim/ changes. Set gpu-qa (here: the local pytest result) on the head SHA; merge only if cpu-tests and gpu-qa are green.
=== END BRIEF ===
