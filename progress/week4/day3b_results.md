# Week 4 Day 3b: fixing the looming persistent state

Follow-up to `day3_results.md` (d). Question from review: is a looming-evoked brain state that persists through ascending feedback biological, and if not, fix it.

## Is it biological?
- **Expected in the fly:** repeated looming shadows put flies into a persistent defensive state that lasts seconds to minutes and fades gradually (Gibson et al. 2015, Curr Biol, PMID 25981791). Ascending neurons send walking and flight state to the brain (Chen et al. 2023, Nat Neurosci; Cheong et al. 2024). The central-complex compass keeps a single activity bump in darkness.
- **What the model did:** in the tail, **all 46 EPG neurons in all 16 glomeruli** were active (median 44 Hz), along with **every PEN** (106–122 Hz), every Delta7, and 15 of 18 PEG. It never decayed. That is the whole ring locked on, not a bump, and it is the same kind of runaway as the Week 1 attractors. **Not biological. Fixed below.**

## What sustains it (`scripts/w4d3b_cx_maintenance.py`)
We ignited the state (looming, W4_CALIB, to 0.6 s), resumed with one excitatory edge class scaled, and counted active neurons 0.8–1.0 s (seed 1; ring = EPG/PEN/PEG/EPGt).

| edge class scaled | g | brain | ring | VNC |
|---|---|---|---|---|
| none | 1 | 642 | 105 | 92 |
| EPG → PEN/PEG (bridge) | 0 | 591 | 88 | 94 |
| PEN/PEG → EPG | 0 | 520 | 60 | 80 |
| all ring ↔ ring | 0 | 420 | 53 | 75 |
| ring → others | 0 | 112 | 110 | 47 |
| **others → ring** | **0.5** | **2** | **0** | 47 |

The ring's own recurrence is not what sustains it. What does is excitatory input from outside the ring. Ranked by synapses × tail rate, that input is almost all from **GLNO** (4 neurons, about 4,100 synapses each onto the ring, firing 140–157 Hz, 83% of the drive) and **LPsP** (2 neurons, about 4,900 synapses each, 17%).

## Both drivers have the wrong sign in the model

| type | MaleCNS (used) | FlyWire | model sign → fix |
|---|---|---|---|
| GLNO | consensus *unclear*, prediction *unclear* (conf 0.44–0.50), so +1 by default | top_nt GABA (3/4), glutamate (1/4); hemilineage LALv1 | +1 → **−1** |
| LPsP | ACh (conf 0.88) | known_nt **dopamine + glutamate, ACh-negative** (Alekseyenko 2013 immuno; Wolff 2024 EASI-FISH) | +1 → **−1** (glutamate; the dopamine part stays removed, as for all monoamines) |

`sim/cns.SIGN_FIX = {'GLNO': -1, 'LPsP': -1}` (`cns_model(sign_fix=...)`).

## Two small loops left: DNg33 pair and IN19B040 quartet
After the sign fix, the brain tail was 2 neurons (DNg33 L/R) and the VNC tail was 47. Both loops are acetylcholine in MaleCNS and FlyWire, so their signs are not in doubt. DNg33 L↔R has 723/773 synapses each way. IN19B040 is all-to-all, with 54–101 synapses per pair. In a LIF model with no adaptation or synaptic depression, any spike re-ignites the partner, so the loop runs forever. We scaled only the within-clique synapses (`scripts/w4d3b_clique_gain.py`, 3 seeds):

| g | brain tail | VNC tail | MNs active during looming |
|---|---|---|---|
| 1.0 | 2 / 2 / 2 | 47 / 47 / 47 | 184 / 184 / 176 |
| 0.5 | 2 / 2 / 2 | 26 / 26 / 26 | 180 / 180 / 175 |
| **0.3** | **0 / 0 / 0** | **0 / 0 / 0** | 177 / 178 / 178 |
| 0.1 | 0 / 0 / 0 | 0 / 0 / 0 | 176 / 182 / 183 |

`sim/cns.CLIQUE_GAIN = ({'DNg33', 'IN19B040'}, 0.3)`. This is a correction for a model limitation (no adaptation), not a claim about the connectome.

## Result (`W4_CALIB` now = GF junctions 8 mV + MN outputs glutamatergic + SIGN_FIX + CLIQUE_GAIN)
Looming 0–0.4 s, mean active neurons per 0.1 s bin, 3 seeds (`results/w4/sci/w4d3_looming_tail_w4calib.txt`):

| t (s) | full CNS brain | full CNS VNC |
|---|---|---|
| 0.0–0.4 | 1,740–1,761 | 1,080–1,157 |
| 0.4–0.5 | 597 | 474 |
| 0.5–1.0 | **0** | **0** |

Before the fix, 0.5–1.0 s had about 540 brain and 70 VNC neurons. The network now returns to silence within 100 ms of offset, in all variants (VNC → brain cut, brain → VNC cut, brain only).

DN panel with the full W4_CALIB (`results/w4/sci/w4d3_panel_summary.md`, `calib+w4` vs `shuffle+w4`): **unchanged** from day 3. GF→TTMn 120 Hz with 88% following, MDN hind/front 1.82/0.03 Hz, DNg100 legs 1.7/1.8/5.6 Hz, looming TTMn/DLMn 277/108 Hz, active MNs real vs shuffled 175 vs 11.

Week 3 words, brain scope, old vs new model (`scripts/w4d3b_word_check.py`, seed 0): spike-count correlation 1.0000 for sugar, bitter, vinegar and hot; 0.9965 for sound; 0.9991 for looming.

## Open item: a rare residual loop in the brain-only model
In the brain-only model (`scope='brain'`), looming ignites a different **11-neuron excitatory loop** (MeVCMe1 ×3, CL213 L/R, DNpe020 ×2, DNg86 L/R, GNG503 L/R; about 30 neurons active in total) in **2 of about 20 runs** (0.5 s looming, seed 0). Weakening DNg33 unmasks it: with the sign fix alone the tail is 2, with the clique gain it is 31. **In the full CNS (the Week 4 model) it did not appear in any of 19 runs.**

Patching loops one at a time will keep finding the next one. The lasting fix is a generic stabilising mechanism (spike-frequency adaptation or short-term synaptic depression) in the LIF, re-validated against Weeks 1–3. For the day-4 pilot, we flag any episode with activity more than 100 ms after its last drive.
