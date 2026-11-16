# Week 4 Day 1: full-CNS model, PyTorch simulator, first trials

## Built
- `sim/cns.py`: `cns_model(scope, config)` builds the calibrated model (monoamines removed, ×0.581, rate APL, eLN–PN gap junctions, presynaptic sensory modulation K = 0.05, right-JO mirror) for `scope='cns'` (no VNC mask) or `'brain'` (Week 3). `motor_pools(meta)` → 349 pools; `brain_mask(meta)` → regression inputs.
  - Check: `scope='brain'` matrix, gap junctions and presyn indices are identical to `scripts/w3d1_common.presyn_model(0.05, mirror=True)` (max |ΔW| = 0).
  - Full CNS: n = 165,122, nnz = 24,519,583 (brain-only 20,511,419).
- `sim/lif_torch.py`: PyTorch port of the Brian2-faithful step order with blocks, gap junctions, APL, presyn; event-driven synaptic gather (CSC columns of spiking neurons → `index_add_`); `rng='numpy'` for exact parity, `rng='torch'` for speed; optional on-device binning (`bin_steps`, `bin_groups`). No save/resume yet.
- `scripts/w4d1_torch_parity.py`, `scripts/w4d1_first_trials.py`. Logs in `results/w4/`.

## Parity (CPU, float64, rng='numpy')
| scope | stimulus | length | spikes | neurons differing |
|---|---|---|---|---|
| brain | vinegar | 0.05 s | 1,137 | 0 |
| cns | vinegar | 0.3 s | 6,848 | 0 |
| cns | sugar | 0.3 s | 2,486 | 0 |

Spike-for-spike identical to the NumPy simulator.

## Speed (cloud container, 4 CPU cores, no GPU)
| simulator | wall per bio-second |
|---|---|
| NumPy (full CNS) | 27–31 s |
| PyTorch CPU float64 | 14–17 s |
| PyTorch CPU float32, device RNG (looming, 1 s) | 17 s |

CPU time is per-step overhead (~1.6 ms per 0.1 ms step), not synapses. The Mac MPS number is still to measure: `python scripts/w4d1_torch_parity.py bench cns 1.0 mps`.

## First full-CNS trials (drive 0–0.4 s at 150 Hz, run to 0.6 s, seed 0)
| condition | real: active brain / VNC / MN | real: top motor groups (Hz) | shuffled: active brain / VNC / MN |
|---|---|---|---|
| none | 0 / 0 / 0 | – | 0 / 0 / 0 |
| sugar | 205 / 44 / 7 | mid-leg coxa/trochanter < 1 | 77 / 7 / 1 |
| vinegar | 300 / 0 / 0 | – | 351 / 33 / 2 |
| looming | 2,382 / 1,314 / 189 | wing power DLM 109, DVM 56, TTM jump 36, haltere 30 | 1,654 / 233 / 13 |
| DNp01 (giant fiber) | 3 / 4 / 1 | TTM 4.4 | 4 / 0 / 0 |
| MDN | 171 / 345 / 41 | hind leg trochanter-femur 16, thorax-coxa 6 | 120 / 19 / 2 |
| DNg100 | 213 / 666 / 81 | hind leg tr-fe 19, wing ps 16, hg 14 | 238 / 42 / 2 |
| DNa02 | 106 / 139 / 30 | neck 4.7, hind/mid coxa 2–4 | 132 / 34 / 2 |

Reading:
- The VNC is silent at rest and every condition returns to silence after the drive, except looming (570 brain / 70 VNC neurons still active 0.1–0.2 s after offset; the known Week 3 looming tail).
- Looming drives a takeoff-like pattern (flight power muscles, TTM jump muscle, haltere). MDN reaches hind-leg MNs first, consistent with MDN acting mainly on the hind legs. These are plausible, not yet validated.
- The giant fiber barely drives TTMn (4 Hz). In the fly GF→TTMn is an electrical synapse, which EM does not count. Day 3 calibration item.
- **The shuffled connectome almost never reaches motor neurons** (≤ 13 MNs vs up to 189 real). Unlike the word readout, brain→VNC motor output depends strongly on the real wiring. This is the first Phase 2 result where the control clearly separates.
- Sugar drives no feeding MN pattern in the VNC; MN9 is a brain (SEZ) motor neuron and is not in the VNC pools.
