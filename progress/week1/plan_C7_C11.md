# Plan C7–C11, based on Stage_after_C5_C6.md

Base for every run: Brian2-faithful simulator, brain only, monoamines removed, w × 0.581, rate APL k = 0.1.
Trial: odor 0–1.0 s at 150 Hz, then silence to 1.6 s. Odors: hot and vinegar.
A run counts as **persistent** if more than 20 brain neurons fire in the last 100 ms.
All changes come from config-driven copies of W, so returning to γ = 1.0 gives back the normal weight matrix.
There is a checkpoint after each step: report the result, then ask before going on.

## Corrected wording from C6

C6 shows **sufficiency under rescue**: lLN1_bc alone and the other 51 cholinergic LNs alone can each sustain persistence, and the 13 unclear neurons sustain a small residual.

It does **not** show three graph-distinct loops. It also does not show that PNs never take part in recurrence when the local-neuron network is intact. C7c tests the graph question directly.

## C7a: refractory unit test (already done in C0; regression only)

The report's main concern was whether input arriving during the refractory period is discarded. C0 already settled it:

- The original NumPy simulator **did** accumulate input during refractoriness. We fixed it, made it match Brian2's step order and exact integrator, and committed the fix as `03979ff`.
- `scripts/c0_refractory_toy.py` matched Brian2 in 7 of 7 cases.
- On FlyWire with sugar, per-neuron rates agree with Brian2 at r = 0.9994–0.9997.
- All C1–C6 results already use the fixed simulator.

**Action:** re-run `c0_refractory_toy.py` once as a regression check. It takes seconds.

## C7b: transmitter sign audit of the 13 unclear core neurons

What we already know without downloading anything:

| MaleCNS type | Number in core | MaleCNS neuron-level confidence | FlyWire top_nt (confidence) |
|---|---|---|---|
| lLN2T_d | 3 | 0.46–0.48 | GABA, 4 of 4 (0.60) |
| l2LN22 | 3 | 0.47–0.49 | GABA, 4 of 4 (0.40) |
| lLN2F_a | 2 | 0.46–0.47 | GABA, 4 of 4 (0.35) |
| v2LN30 | 2 | 0.43–0.47 | 1 glutamate, 1 serotonin (0.28) |
| lLN10 | 1 | 0.45 | GABA, 2 of 2 (0.52) |
| untyped | 2 | 0.46–0.49 | n/a |

The prior from FlyWire is that these are mostly **inhibitory**. The model currently treats them as excitatory.

Steps:

1. **Download the 2.7 GB per-presynapse neurotransmitter table.** This needs the user's OK. Stream it and keep only the rows for the 13 bodies, so only a small file is stored.
2. **Compute each neuron's outgoing-synapse probability mass** for ACh, GABA, Glu and modulatory transmitters.
3. **Run three variants:**
   - current (unclear treated as +1);
   - zero output for neurons that stay unresolved;
   - best-supported sign, set per neuron, or per synapse if the per-synapse table supports it.

   Each variant runs hot and vinegar on seeds 0 and 1, applied to the whole trial.
4. **Report the gain shift, not only whether persistence survives.** Repeat the C6 rescue test with lLN1_bc restored at β = 0.5 and with the other LNs restored at β = 1.0 under the corrected signs. This shows whether the corrected signs move either module's threshold.

Prediction: C3 already found that inhibitory signs shrink the tail by about 33% without abolishing it. So the sign fix probably removes the ~200-neuron residual but leaves the lLN1_bc module intact.

Cost: download time plus about 10 minutes of simulation.

## C7c: are the modules graph-distinct? (no simulation)

On the weights we actually simulate, find the strongly connected components (SCCs) inside lLN1_bc, inside the other 51 LNs, and inside the 13 unclear neurons. For each group, also measure how much of its recurrent drive passes through neurons outside the group.

This turns "three modules" into a tested claim instead of an assumption.

## C7d (optional): FlyWire control

Run Shiu's own FlyWire v783 model with vinegar ORNs stimulated for 1 s, then silence, and check whether lLN1_bc persists.

The two connectomes have the same lLN1_bc motif (C6b). If FlyWire also persists, that supports describing the eventual fix as a **model calibration**, not a MaleCNS lesion. Cost: about 2 minutes.

## C8: gains inside the recurrent modules (whole-trial, downward sweep)

Only edges *inside* or *between* the LN modules are scaled. Edges from LNs to PNs, to GABA LNs, and from sensory neurons stay at 1.0.

| Parameter | Edges scaled |
|---|---|
| γ1 | lLN1_bc → lLN1_bc |
| γ2 | other cholinergic LN → other cholinergic LN |
| γ12 | lLN1_bc ↔ other cholinergic LN (both directions) |

Steps:

1. **One-dimensional sweeps.** Run γ1 first, then γ12, then γ2, each at 1.0, 0.9, 0.8, 0.7, 0.6 and 0.5, with hot and vinegar on seeds 0 and 1. That is 72 trials, about 27 minutes.
2. **Refine any transition.** Bracket each transition in steps of 0.05, then 0.025, and use 4 seeds near the boundary.
3. **Two-dimensional map.** Run the report's 8-point (γ1, γ12) grid, which takes about 12 minutes.

Metrics for every run:

- silent-tail population size;
- lLN1_bc rate;
- antennal-lobe population during the odor, compared with γ = 1.0;
- MN9 rate;
- decay time.

What counts as a good point: the persistent state has just disappeared (we are just past the bifurcation), and the odor-period response is close to baseline.

Expectation from C6b: lLN1_bc self-input reaches threshold at a group rate of only about 3 Hz, and 64% of each lLN1_bc neuron's excitation comes from outside the group. So γ1 alone probably fails, and the 2D map (γ1 together with γ12) is the decisive test.

**Stopping rule:** if nothing in the 2D map down to 0.5 abolishes persistence, stop. Report that a static chemical-weight correction of the module cannot fix it. Then propose calibrating the model with adaptation or depression, since the report itself notes that the model omits stabilizing physiology such as electrical coupling.

## C9: ignition versus maintenance at the winning gains

Present the odor at γ = 1.0, then switch only the module-internal gains to their winning values, at 1.0 s or at 1.3 s.

Pass criterion:

- the established state collapses after the switch;
- the odor-period response is identical to baseline, because it is untouched before the switch;
- for comparison, whole-trial correction (C8) changes the odor-period response by at most about 10–15%.

## C10: add-back hysteresis

Start from the corrected stable model and step the module gains back up toward 1.0. Persistence should return near the same threshold found in the downward sweep. If it returns at a different level, report the size of the hysteresis.

## C11: full-network validation

Run the full brain + VNC network with the correction (C7b signs plus C8 gains):

- all 18 inputs (17 words plus sugar+bitter);
- no stimulus;
- 3 seeds.

Pass criteria:

- no smell attractor;
- odor responses preserved;
- word-to-word overlap no worse than before;
- MN9 responds to sugar and not to smells;
- no-stimulus runs stay silent;
- no new attractor appears anywhere else.

When C11 passes, go back to the **Week 2 deliverable**: the MaleCNS word-to-activation-flow frontend.
