# BRIEF T0 (Week 4 perf): CPU CI for the flybrain repo

Orchestrator -> Dev. Hypotheses are UNCONFIRMED: check them before building on them.

**Outcome:** every PR in CharlieZhang1999/flybrain gets an automatic CPU check, so QA has a required check besides its own GPU checklist.

## Context
- Repo CharlieZhang1999/flybrain, local clone (main, in sync). No .github/workflows/ today.
- The real model needs MaleCNS data (gitignored) or results/w4/cns_bundle_calib.npz (82 MB, not in git). CI must not need either.
- Reference simulators: sim/lif_numpy.py (run_trial, Brian2-faithful) and sim/lif_torch.py (run_trial, PyTorch port). On CPU float64 with rng='numpy' the torch one was spike-for-spike identical to NumPy on the full CNS (2026-11-15/16).
- Hypothesis (unconfirmed): both files are on main. If sim/lif_torch.py is missing from main, stop and report; do not copy files in from elsewhere.

## Build
1. A test helper that builds a tiny synthetic network in code (e.g. 300 neurons, ~5% random signed weights scaled so some neurons fire, a 20-neuron gap-junction block, a 30-neuron KC set for APL, 20 sensory neurons with a presyn matrix), seeded, no files.
2. tests/test_parity_cpu.py: lif_torch.run_trial(..., device='cpu', dtype=torch.float64, rng='numpy') vs lif_numpy.run_trial(...) with the same options (gap, apl, presyn, blocks, t_stim): spike counts identical. One case with all options off, one with all on, one with blocks. If lif_torch.run_trial has state=/return_state=, add: a run split in two equals one run.
3. .github/workflows/ci.yml: on pull_request and push to main; ubuntu-latest; Python 3.12; CPU torch (--index-url https://download.pytorch.org/whl/cpu) + numpy scipy pandas pytest; run pytest -q tests/. Job name cpu-tests.
4. Pinned requirements-ci.txt if needed. No GPU, no data.

## Acceptance criteria
- pytest -q tests/ passes locally in WSL (~/flybench/.venv) and in the Actions run on the PR head commit.
- >= 3 parity cases with exact equality; each test < 30 s on CI.
- No file under sim/ changes.
- Demonstrate once locally that the tests fail if parity breaks (flip one constant in an uncommitted scratch copy) and report the failure output.

## Deliverable
Branch perf/w4-t0-ci, PR to main titled "CI: CPU parity tests on a synthetic network", with a summary of built / assumed / unverified. Append a "T0" entry to progress/week4/perf_log.md (create it) and save this brief as progress/week4/tasks/T0_brief.md. No website update.

## Stop condition
Stop and report if the reference files differ from these assumptions, CPU torch can't be installed on the runner, or exact parity fails on the synthetic network (a real bug: do not loosen the test).
