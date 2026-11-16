# Week 2 Day 1 results (2 Nov 2026)

Scripts: `scripts/w2d1_flywire_control.py`, `scripts/w2d2_ln_pn_edges.py`.
Logs and CSVs: `results/flywire/w2d1_log.txt`, `w2d1_flywire_control.csv`; `results/malecns/w2d2_log.txt`, `w2d2_runs.csv`.
Trial as in C8: stimulus 0–1.0 s at 150 Hz, silence to 1.6 s. A run is persistent if more than 20 neurons fire in the last 100 ms.

## D1: FlyWire v783, Shiu model exactly as published

Shiu's own signs and w_syn, monoamines excitatory, spiking APL, no 0.581 scaling. Seeds 0 and 1.

| Stimulus | Active during stimulus | MN9 | Still active after stimulus | lLN1_bc after stimulus | Persistent |
|---|---|---|---|---|---|
| vinegar (ORN_DM1+DM4) | 8,410 | 0 Hz | 7,850 | 290 Hz | 2/2 |
| hot (TRN_VP2) | 8,260 | 0 Hz | 7,880 | 292 Hz | 2/2 |
| sound (JO-A/B) | 583 | 0 Hz | 0 | 0 | 0/2 |
| looming (LPLC2+LC4) | 1,318 | 0 Hz | 0 | 0 | 0/2 |
| sugar (Shiu's 20 GRNs) | 346 | 83 Hz | 0 | 0 | 0/2 |
| none | 0 | 0 | 0 | 0 | 0/2 |

**The olfactory attractor is a property of the Shiu model, not of MaleCNS.** FlyWire has it too, and it is larger there (about 7,850 neurons versus 3,340), probably because MaleCNS was already scaled by 0.581 with monoamines removed. Sugar→MN9 (83 Hz) matches the Week 1 reference, so the run is set up correctly.

Two things differ from MaleCNS:
- In FlyWire, odor does **not** drive MN9 (0 Hz). In MaleCNS it does (about 50 Hz). That spread is MaleCNS-specific.
- Sound and looming are clean on FlyWire. On MaleCNS with the C8 base config, sound is clean but looming leaves a small tail of 502 neurons, outside the antennal lobe.

## D2: MaleCNS, cholinergic LN ↔ PN chemical edges

LN→PN: 144 cholinergic ALLNs onto all 686 ALPNs. PN→LN: 469 cholinergic ALPNs onto the cholinergic ALLNs. That is 36,897 edges in total. "eLN off" sets those edges and cholinergic LN→LN to 0 (43,900 edges).

| Condition | Odor: AL active | Odor: mean PN rate | MN9 | After odor: brain active | lLN1_bc after | Persistent |
|---|---|---|---|---|---|---|
| base | 932 | 82 Hz | 50 Hz | 3,342 | 302 Hz | 4/4 |
| LN↔PN 0.5 | 730 | 40 Hz | 44 Hz | 2,094 | 282 Hz | 4/4 |
| LN↔PN 0 | 81–357 | 4–5 Hz | 0 | 0 or ~440 | 0 or ~270 Hz | 2/4 (bistable) |
| eLN off | 80–86 | 3–4 Hz | 0 | 0–2 | 0 | **0/4** |

Sound and looming did not change (sound clean; looming 502 either way).

**What this means**
1. The attractor is the cholinergic LN–PN chemical loop together with the LN–LN edges. Cutting either part alone (C8: LN–LN; D2: LN↔PN) is not enough. Cutting both abolishes it in 4 of 4 runs, and odor-driven MN9 activity goes away too.
2. The baseline "odor response" was mostly the attractor spreading. At baseline, vinegar from 2 glomeruli drives all 686 PNs at about 82 Hz on average. In the fly, vinegar activates a handful of glomeruli. So the plan's check "odor-period antennal-lobe activity within 15% of baseline" is the wrong check. The right one is that PNs of the stimulated glomeruli (DM1, DM4 for vinegar) respond strongly and the others weakly.
3. This matches the biology from the research: eLN→PN excitation is through gap junctions, not nicotinic synapses. "eLN off" is therefore a defensible calibration, and it can be applied the same way to FlyWire.

## Revised Days 2–3

1. Run "eLN off" on FlyWire too (vinegar, hot, sugar, none). If it is clean there as well, the same rule works on both connectomes.
2. Check glomerular specificity under eLN off: rates of the DM1/DM4 PNs versus all other PNs for vinegar; the VP2 PNs for hot.
3. As an alternative that keeps the edges, add spike-frequency adaptation (τ_a ≈ 250 ms, sweep Δa). This tests whether adaptation alone can stop the loop, and gives a "brakes" version to compare with the "rewiring" version.
4. Run all 17 words plus none on whichever version passes. Then build the readout as planned on Days 4–5.
