# Week 4 perf log

## T0: CPU CI with parity tests on a synthetic network (2026-11-18)

Brief: [tasks/T0_brief.md](tasks/T0_brief.md). Branch `perf/w4-t0-ci`.

**Built**
- `tests/synth.py`: seeded 300-neuron synthetic network built in code (5% density, 20% inhibitory, synapse counts 1-10 x w_syn x 4; sensory 0-19, KCs 20-49 with rate-APL, gap block 50-69 with symmetric random coupling, presyn via `sim.presyn.build_presyn`). No files.
- `tests/test_parity_cpu.py`: `lif_torch.run_trial(device='cpu', dtype=float64, rng='numpy')` vs `lif_numpy.run_trial`, exact equality of spike counts, 3 cases: options off; gap+apl+presyn+t_stim; overlapping blocks with gap+apl+presyn. Each case also asserts a minimum amount of non-sensory activity so equality is not trivial.
- `.github/workflows/ci.yml` (job `cpu-tests`, ubuntu-latest, Python 3.12, CPU torch 2.11.0 from the PyTorch CPU index) and `requirements-ci.txt` (numpy/scipy/pandas/pytest pinned to the WSL bench venv versions).

**Findings**
- Both reference simulators are on main. `lif_torch.run_trial` has no `state=`/`return_state=`, so the split-run test was not added.
- Local (WSL, ~/flybench/.venv): 3 passed in ~3.3 s; each test < 0.4 s.
- Failure demo: in a scratch copy, `rfc = int(round(t_rfc / dt)) - 1` in `lif_torch.py` makes all 3 tests fail (92 / 31 / 27 neurons differ). A first, weaker blocks case (150-200 Hz) did not catch that break, so the blocks and all-options cases were strengthened (higher drive) until they did.
- No file under `sim/` changed.
