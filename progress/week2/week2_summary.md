# Week 2 summary (3 Nov 2026)

**Deliverable:** the Connectome Graph Explorer, https://claude.ai/artifact/Rwffk7YiziRARz86WMhQEE. The source is `artifact/connectome_explorer.html`; views are exported by `scripts/w2d6_export_graph.py` into `results/graph/`.

## 1. The persistent-attractor problem, solved as a model calibration
- The attractor comes from the Shiu model itself. FlyWire v783, run exactly as published, also keeps about 7,900 neurons firing after vinegar or hot (`day1_results.md`).
- It is sustained by the cholinergic eLN–PN chemical loop. Yaksi & Wilson 2010 showed that this connection is electrical, so the calibration does three things. It sets chemical LN→PN synapses to 0. It adds eLN–PN gap junctions (`sim/gap.py`; c0 = 0.2; spikelets matched to the charge a spike passes, 2 mV). It leaves all other chemical edges as they are (`day2_3_gap_junctions.md`).
- **Result:** no persistence for any odor, temperature or humidity word on either connectome. Odor responses become glomerulus-specific. FlyWire taste behavior is unchanged and matches Shiu et al.: sugar→MN9 84 Hz, bitter 0, sugar+bitter suppressed to 0, water weak (about 5 Hz over 5 seeds in both models), Ir94e 0.
- **Known limits:** eLNs are shunted nearly silent. Capping each neuron's total coupling keeps them firing at 64–120 Hz but brings back a tail of about 300–400 neurons. Wind still leaves about 390 neurons firing outside the antennal lobe on MaleCNS. The gap-junction locations are inferred from chemical contacts.

## 2. Word readout: 17 words × 10 seeds, MaleCNS (`results/readout/readout_summary.json`)
Accuracy on the 0–0.5 s stimulus window; chance is 5.9%.

| Model | All non-sensory (test / leave-one-seed-out) | DNs + MNs | Post-stimulus 0.5–1.0 s, all |
|---|---|---|---|
| base | 100% / 100% | 94% / 98% | 69% |
| calibrated | 100% / 100% | 94% / 94% | 94% |
| shuffled (calibrated, degree- and sign-preserving) | 100% / 100% | 86% / 85% | 70% |

The shuffled brain also decodes at 100% from all neurons, as plan.md predicted. That is the main lesson of Phase 1: a linear readout of a driven reservoir works regardless of the wiring. The real wiring shows up only in the motor-only readout (94% vs 86%) and in the post-stimulus window, where the shuffled network falls to 21% motor-only versus 68% for the calibrated one.

## 3. Explorer contents
- MaleCNS: base, calibrated and shuffled models × 17 words. FlyWire: base and calibrated × 9 stimuli (5 taste, vinegar, hot, sound, looming).
- Layout: sensory input on the left, all brain motor neurons in a column on the right (silent ones included), everything else placed by synaptic hops from the stimulus. A force-directed layout is also available.
- Coloring: a heat map of participation during the stimulus, or of spikes in the current 20 ms bin, with Play; or by cell class.
- Also on the page: the readout table, confusion matrices for each model, and notes on how to read it.

## Ready for Week 3
The calibrated MaleCNS model and the stimulus-window readout are the substrate for the GPT loop.
