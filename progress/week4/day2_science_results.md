# Week 4 Day 2 (science track): save/resume in the PyTorch simulator

The perf-harness half of day 2 is in `day2_results.md`. This file covers the other day-2 item: save/resume (`state=`) in `sim/lif_torch.py`, so episodes can be chained like Week 3.

## Built
- `sim/lif_torch.run_trial(..., state=None, return_state=False)`. Same contract as `sim/lif_numpy` (Week 3): `return_state=True` adds `'state'` (step, v, g, last spike, delay ring, counts, NumPy and torch RNG states, APL rate, presynaptic traces); passing it back as `state=` continues the same trial up to the new `params['t_run']` (absolute time). Blocks keep absolute times, so a resume can sit inside a block.
  - The state keeps tensors on the device, so chained episodes never copy back to the host. It also accepts NumPy arrays.
  - `counts` are cumulative over the whole chain (as in NumPy); `t`/`i` and `binned` hold only the new segment, and bins restart at the resume step.
  - A resume may use a different `W` (same shape). Day 3 uses this to cut ascending synapses half-way through a trial.
- `scripts/w4d2_resume_check.py`.

## Resume check (full CNS, CPU)
Looming 0–0.3 s, then MDN 0.3–0.5 s, run to 0.7 s, seed 3, motor-pool binning on (10 ms bins, 349 pools). One uninterrupted run vs three chained segments (0–0.2, 0.2–0.45, 0.45–0.7 s), so both splits fall inside a block.

| RNG | spikes (one run) | spikes (chained) | spikes, pool bins and counts identical |
|---|---|---|---|
| `numpy` (parity) | 61,429 | 61,429 | yes |
| `torch` (device) | 65,014 | 65,014 | yes |

Chaining costs nothing measurable (12 s vs 12 s wall for 0.7 bio-seconds). Log: `results/w4/sci/w4d2_resume_check.txt`.
