# Brain → sensory synapses, modelled as presynaptic modulation (9 Nov 2026)

Follow-up to `day1_results.md` and `sidekick_cold_loop_plan.md`. Code: `sim/presyn.py`, `presyn=` in `sim/lif_numpy.run_trial`. Scripts: `scripts/w3d1d_presyn_calibrate.py`, `w3d1e_presyn_words.py`, `w3d1f_presyn_flywire.py`, `w3d1g_word_trials.py`, `w3d1_episodes.py … presyn0.05`, `w3d1_decode.py presyn0.05`. Results: `results/loop/w3d1d_* w3d1e_* w3d1f_* w3d1g_*`, `w3d1_carryover_presyn0.05.csv`, `w3d1_log_presyn0.05.txt`.

## What the fly does (evidence)
- Sensory neurons fire their spikes in the periphery (sensilla on the antenna, labellum and legs). Their axon terminals in the brain receive synapses from central neurons; the connectome shows these clearly (MaleCNS: about 65k inhibitory and a similar weight of excitatory edges onto 15.9k sensory neurons).
- **What those synapses do is presynaptic inhibition, not spike generation.** ORN terminals carry GABA-A and GABA-B receptors. LN GABA gives an early GABA-A phase and a long GABA-B phase (GABA-evoked current lasting about 1–1.5 s), and the inhibition scales with total ORN activity (r² = 0.73 over 14 odours) ([Olsen & Wilson 2008](https://wilson.hms.harvard.edu/sites/projects.iq.harvard.edu/files/wilson-lab/files/olsenwilson2008.pdf)). Sweet GRNs (not bitter GRNs) express GABA-B. GABA inhibits their Ca²⁺ responses presynaptically, and that is one route by which bitter suppresses sweet ([Chu et al. 2014](https://pesquisa.bvsalud.org/brasil/resource/pt/mdl-25131672)). Both GABA components on ORN terminals: [Root et al. / PMC4994068](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4994068/).
- No study reports central input making a sensory neuron spike. Excitatory (cholinergic) axo-axonic synapses onto sensory terminals exist, but their effect has not been characterised.

## How it is modelled now
1. **Central input never reaches a sensory neuron's membrane.** Sensory neurons spike only from their stimulus (the Poisson drive), as in the real fly.
2. **Inhibitory input onto a sensory terminal turns down that neuron's transmitter release.** Each inhibitory spike adds to a fast GABA-A trace (τ 5 ms) and a slow GABA-B trace (τ 300 ms), with an equal steady-state share. The terminal's release gain is `1 / (1 + (GABA_A + GABA_B) / K)`, and it multiplies every outgoing synapse of that sensory neuron.
3. **Excitatory input onto terminals has no effect** (not characterised in the literature; stated as an assumption).
4. **K = 0.05 V** is the one fitted constant. It is chosen so that FlyWire sugar → MN9 (Shiu's validated number) stays within 20%:

| K | FlyWire sugar → MN9 | sugar + bitter | vinegar: mean gain of stimulated ORNs (MaleCNS) |
|---|---|---|---|
| no presynaptic inhibition (cut only) | 78.5 Hz | 2.5 | 1.00 |
| 0.1 | 73.5 | 2.5 | 0.93 |
| **0.05 (chosen)** | **68.0** | **2.0** | ~0.87 (interpolated) |
| 0.03 | 56.5 | 1.0 | 0.81 |
| 0.01 | 31.0 | 1.5 | 0.66 |
| Shiu model as published | 77.0 | 6.0 | n/a |

Stronger K (0.003) also makes vinegar and fruit spread into about 600–900 neurons, so it was rejected. At K = 0.05, bitter still suppresses sugar → MN9 (2 Hz), as in Shiu et al. and in line with Chu et al.

## Results (MaleCNS calibrated + presynaptic modulation, K = 0.05)
**All 17 words + none, seed 0, tail = neurons firing 1.5–2.0 s after a 0.5 s stimulus**

| | Week 2 calibrated | + presynaptic model |
|---|---|---|
| cold tail | 245 | **0** |
| humid tail | 202 | **0** |
| wind tail (non-AL loop, not targeted) | 437 | 464 |
| every other word, and none | 0–2 | 0–2 |
| vinegar target / other PN (Hz) | 117 / 0.5 | 106 / 0.4 |
| hot target / other PN (Hz) | 44 / 0.7 | 36 / 0.6 |

Active neurons during the stimulus change by −30% to +5% (cold 360 → 232, humid 330 → 144, sugar 214 → 171). MaleCNS brain-only sugar → MN9 was already only about 4 Hz and is now 0. MN9 in MaleCNS needs the full-CNS port anyway (Phase 2).

**Episodes (the same 40 × 5-word sequences, 0 ms gaps)**

| Readout | Week 2 calibrated | + presynaptic model |
|---|---|---|
| Isolated readout, all non-sensory neurons | 95.5% (9 errors) | **100% (0 errors)** |
| Isolated readout, DNs + MNs | 82.5% | 90.0% |
| Episode-trained (5-fold), all | 100% | 100% |
| Episode-trained (5-fold), DNs + MNs | 91.5% | 91.5% |

Neurons still firing at the end of an episode (median): after cold/humid, **270 → 49**. Episodes with neither stay about 20. After wind/looming it stays about 450 (the non-AL loop, unchanged).

## Assumptions to state in the README
- Gain is set by inhibitory input only. Excitatory axo-axonic input is ignored, because its effect is unknown.
- The time constants are round numbers consistent with Olsen & Wilson (GABA-A fast, GABA-B lasting about 1 s). K is fitted to keep Shiu's sugar → MN9, not measured.
- Sensory neurons have no spontaneous firing (real ORNs fire at about 3–10 Hz). This is unchanged from Shiu's model.

## Recommendation
Adopt "calibrated + presynaptic model, K = 0.05" as the Week 3 substrate (config name `presyn0.05`; `scripts/w3d1_common.presyn_model(0.05)`). It removes the cold/humid lasting state by a biological rule, not a cold-specific cut. The isolated word readout then needs no episode retraining. The shuffled control, the Explorer views and the Week 2 readout table were built on the old calibration and should be regenerated on this model when Day 6 updates the Explorer.
