# Week 4 plan: full MaleCNS CNS simulation (Phase 2A, weeks 4–6)

Date: 2026-11-15. Source: `docs/plan.md` §3, 2A.

## Goal for Phase 2A
Simulate the whole male CNS (brain + ventral nerve cord) with the same calibrated LIF as Weeks 1–3, and replace the 16/17-word classifier with a regression: from brain activity, predict VNC motor-neuron pool rates 100 ms ahead (R² per pool), with the shuffled-connectome control in every table.

## Facts checked before planning
- **Neuron count.** The traced MaleCNS set we already load has **165,122** neurons, not 166,700. The 166.7k figure counts every annotated body (incl. untraced fragments); other projects use it for a looser filter. We report 165,122 and say why.
- **Phase 1 already had every neuron in the matrix.** It zeroed all synapses touching the VNC (`scripts/ckpt3_sweep_ablate.regions` + `mask_weights`). Going full-CNS means dropping that mask: nnz 20.5M → 24.5M (+3.5M VNC↔VNC, +0.4M brain→VNC, +0.4M VNC→brain).
- **Motor targets.** 708 `vnc_motor` neurons with functional types (e.g. `Ti flexor MN`, `DLMn c-f`, `b1 MN`), split by subclass: fl/ml/hl legs (381), wm wing (67), hm haltere (16), nm neck (24), ad abdomen (214), xm (6). Grouped by side × subclass × type → **349 pools**; legs also tagged by joint (thorax-coxa … tibia-tarsus), wing by muscle group (power DLM/DVM, steering b/i/iii/hg, tension tp, TTM jump).
- **Prior work (awesome-fly, checked 2026-11-15).** The "first full CNS" claim is no longer safe as written:
  - `seanphan/flyt3`: LIF on 166.7k-neuron MaleCNS, 25.6M edges, readout over top 512 DNs + 512 VNC MNs trained by REINFORCE for tic-tac-toe. Homeostatic thresholds, no Shiu parameters, no motor-pool regression, no shuffled control on the main task.
  - `seohyunjun/mps-malecns-model`: PyTorch/MPS LIF on 166.7k neurons, dt 1 ms, no refractory, ~22 s per bio-second, no motor readout.
  - Several body-coupled demos (DesktopFly, closed-loop-fly, flyverse, webgpu-fly) use DN readouts or MaleCNS extracts, not the full graph with motor-pool regression.
  - **Defensible claim:** "open, Shiu-faithful (dt 0.1 ms, validated against Brian2) simulation of the complete male CNS, with a brain→VNC motor-pool regression reported as R² per pool against a degree-preserving shuffled connectome." Re-check the list before any public post.

## Week 4 (this week)
| Day | Work | Done when |
|---|---|---|
| 1 | `sim/cns.py` (full-CNS model, motor pools); `sim/lif_torch.py` (event-driven PyTorch port); parity + first trials | brain scope = Week 3 model exactly; torch = NumPy spike-for-spike; rest silent ✔ (see day1_results.md) |
| 2 | Speed on the Mac: `w4d1_torch_parity.py bench cns 1.0 mps` and `cpu`; profile; if >10 s/bio-s, batch trials (B seeds in one state tensor) and move Poisson to the device | number for MPS in a table; target < 10 s per bio-second |
| 2 | Save/resume (`state=`) in the torch sim, so episodes can be chained like Week 3 | resume check = one long run |
| 3 | VNC sanity + calibration: (a) DN validation panel (GF→TTMn, MDN→hind legs, DNg100→walking, DNa02→turning) real vs shuffled; (b) GF–TTMn and GF–PSI are electrical in the fly → add as gap junctions (same `sim/gap.py` route as eLN–PN) and re-test; (c) audit VNC transmitter signs (`unclear` share, glutamatergic MN→? edges); (d) looming tail: check it is brain-side, not a new VNC loop | table of DN→pool rates; a short decision note on each calibration |
| 4 | Dataset design for the regression: episodes of random stimulus mixtures (2–4 modalities, random intensities and onsets, 1–5 Hz background sensory drive), 10 ms bins; inputs = brain non-sensory non-motor neurons, targets = 349 pool rates (also coarse: legs × joint, wing groups) | `stimuli/mixtures.py`, a 10-episode pilot with spike files |
| 5 | Generate the training set (real + shuffled): ~200 episodes × 2 s each, binned on device (`bin_groups`) | data in `results/w4/regression/`, run log |
| 6–7 | Week 4 write-up; README status; start ridge baseline if time | `progress/week4/week4_summary.md` |

## Weeks 5–6 (for orientation)
- Week 5: decoders. Ridge on lagged brain bins (lags 0–100 ms) → pool rates at t+100 ms; R² per pool, held-out episodes. Inputs: all brain / DNs only / random 1% subset. Controls: shuffled connectome, shuffled time (circular shift). Then a small nonlinear model only if ridge leaves structure on the table.
- Week 6: benchmark table (speed, R²), update the Connectome Explorer with VNC views and a motor-pool panel, write-up.

## Risks
- **Brain→VNC drive is sparse.** Only strong stimuli (looming) or direct DN drive reach many MNs in the first trials; many pools may have near-zero variance, which makes R² meaningless. Mitigation: report R² only for pools with variance above a floor, and add background drive.
- **Electrical synapses missing.** GF→TTMn is weak (4 Hz) without the known gap junction.
- **Speed.** CPU is overhead-bound (~1.6 ms per step regardless of activity); MPS kernel launches may be similar. Batching trials is the main lever.
