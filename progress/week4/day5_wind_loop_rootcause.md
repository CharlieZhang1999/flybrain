# Week 4 Day 5 (T2d): root cause of the wind loop (LOOP5)

**This is a diagnosis. No fix was implemented.** Nothing under `sim/` or `stimuli/` was changed, and `git status` shows only new untracked files. Every test edits a copy of the final calib W inside a script. One variant instead calls `cns_model` with `mirror=False`.

**Verdict, short:** no annotation or build error explains the loop. The loop is most likely genuine strong reciprocal excitation among cholinergic GNG descending neurons (provisional; see H4 for what remains unchecked). Two findings support this:
- All five LOOP5 types are ACh in MaleCNS (body, type and consensus) and in FlyWire.
- The wiring is bilaterally symmetric, the bodies are unremarkable, and the build removed no inhibition from the loop (Traced inputs after load()).

No sign-flip test that has any annotation support removes the persistence. Option A (LOOP5 edges × 0.3), or adaptation/depression, remains the fix. Details are in the root-cause section at the end.

**Scripts** (each has a docstring with purpose, usage and output):
- `scripts/w4d5_loopid_static.py`: Q1–Q3, no simulation.
- `scripts/w4d5_loopid_runs.py`: Q4 simulations, variants `mono5`, `DNg21_inh`, `GNG423_inh`, `DNg22_inh` and `nomirror`.
- `scripts/w4d5_loopid_analyze.py`: compares the runs with the saved Day 5 runs.

**Outputs:** `results/w4/regression/loop_id/`
- `static.txt`, `nt_table.csv`, `body_table.csv`, `recip_pairs.csv`, `inhib_inputs.csv`
- `runs/*.npz`, `runs_log.txt`, `analyze_runs.{txt,csv}`
- `scripts_backup/`

**Simulation used:** 23 runs of 2.5 s, about 19 min of wall time plus 3 model builds, one at a time. Settings: calib, `**W4_CALIB`, exact Day 5 specs and seeds, CPU float32, `rng='torch'`.

**Thresholds:** a "partner" is a type with a type-pair sum of ≥ 100 synapses into or out of LOOP5 (132 types). Inhibitory inputs are listed from ≥ 5 synapse-equivalents in the final W (170 types).

---

## Q1: transmitter labels and signs

### Data limitation
`body-neurotransmitters-male-cns-v1.0.feather` holds only the top class, its confidence, the type-level top class and confidence, `consensus_nt` and `ground_truth`. **Per-class probabilities are not in it.**
- They exist only in the 2.65 GB `tbar-neurotransmitters-male-cns-v1.0.feather`. That file is not on disk; only the AL subset `tbar_nt_AL.parquet` is, and it contains none of the 12 LOOP5 bodies.
- Downloading it was **blocked**: the proxy answered 403 to CONNECT for `male-cns.janelia.org`, a policy denial. I did not try other routes.
- **So "the GABA or Glu probability of an ACh-labelled neuron" cannot be reported.** I report the top-class confidence and its percentile instead.

### Reference distributions (`static.txt`)

| set | n | p5 | p10 | p25 | median |
|---|---|---|---|---|---|
| MaleCNS `predicted_nt_confidence`, ACh-predicted DNs | 928 | 0.839 | 0.902 | 0.942 | 0.956 |
| FlyWire `top_nt_conf`, ACh DNs | 873 | 0.416 | 0.477 | 0.614 | 0.756 |

### LOOP5 (`nt_table.csv`)
Model nt is `acetylcholine` and the effective sign is +1 for every cell. MaleCNS `ground_truth` is empty for all of them. "pct" is the percentile among ACh DNs in the same dataset.

| type (n) | MaleCNS body pred., conf (pct) | type conf | consensus | FlyWire type: top_nt, conf (pct) | FlyWire known_nt | hemilineage (ito-lee / truman) |
|---|---|---|---|---|---|---|
| DNge027 (2) | ACh 0.872 / 0.916 (10th) | 0.893 | ACh | DNge027: ACh 0.60 / 0.59 (22nd) | – | putative_primary / – |
| GNG423 (4) | ACh 0.869–0.947 (16th) | 0.911 | ACh | CB1582: ACh 0.46 (**8th**) | – | LB7__prim / 08B_put1 |
| DNg21 (2) | ACh 0.820 / 0.797 (**4th**) | 0.808 | ACh | DNg21: ACh 0.68 / 0.59 (28th) | – | MX12__prim / – |
| DNg62 (2) | ACh 0.938 / 0.906 (16th) | 0.921 | ACh | DNg62: ACh 0.56 / 0.56 (17th) | – | MX12__prim / – |
| DNge044 (2) | ACh 0.968 / 0.966 (84th) | 0.967 | ACh | DNge044: ACh 0.70 / 0.68 (34–39th) | – | putative_primary / – |

**Flags:**
- **DNg21:** the lowest MaleCNS confidence (4th percentile).
- **GNG423:** FlyWire confidence 0.46, below 0.5 (8th percentile).
- **No disagreement** for any LOOP5 type: ACh in both datasets, with body, type and consensus all ACh.
- **Hemilineage context:**
  - MX12__prim (DNg21, DNg62) is 24 ACh / 16 Glu / 4 GABA / 3 unclear (n 47), so a glutamatergic MX12 cell would not be unusual.
  - LB7__prim (GNG423) is 37 / 4 / 4 / 1.
  - truman 08B, all versions, is 869 of 877 ACh.
  - These are priors only, not evidence of a mislabel.

### Main partners (selected; all 132 + 170 types are in `nt_table.csv`)

| type | syn into / from LOOP5 | model nt (sign) | MaleCNS body conf / consensus | FlyWire top_nt (conf), known_nt | flag |
|---|---|---|---|---|---|
| SAD093 (entry) | 897 / 29 | ACh (+) | 0.95 / ACh | ACh (0.73) | truman hemilineage `05B_put1`; 05B overall is 301 of 379 GABA, but the 12 `05B_put1` bodies are all ACh-predicted |
| AN01A014 | 804 / 153 | ACh (+) | 0.96 / ACh | ACh (0.86) | – |
| DNge039, DNge078, DNge011, DNge022 | 391 / 1352, 582 / 70, 678 / 288, 315 / 540 | ACh (+) | 0.84–0.96 / ACh | ACh (0.59–0.87) | – |
| GNG282 (OL read-out) | 3 / 582 | ACh (+) | 0.93 / ACh | ACh (0.78) | – |
| **DNg22** | 91 / 178 | ACh (+) | **0.59 / 0.62 (0.4th pct)** / ACh | **serotonin (0.66–0.78); known_nt FMRFa (Song et al. 2023, immuno)** | **disagreement** |
| GNG671 | 102 / 64 | unclear (+) | 0.48 / unclear | dopamine (0.70) | unclear → +1 |
| AN05B096, AN05B105, LN-DN1 | 132 / 0, 92 / 1, 113 / 2 | ACh (+) | 0.56–0.65 | serotonin (0.79–0.87) | disagreement; silent in the wind runs (0 Hz) |
| DNg98 | 2153 / 85 | GABA (−) | 0.87 / GABA | GABA (0.68) | – |
| GNG281 | 391 / 98 | GABA (−) | 0.70 / GABA | glutamate (0.61) | both inhibitory |
| GNG150 | 277 / 620 | GABA (−) | 0.81 / GABA | glutamate (0.47) | both inhibitory |
| DNge029, GNG404 | 442 / 551, 637 / 122 | Glu (−) | 0.78, 0.80 | Glu (0.63, 0.72) | – |
| DNge142 | 825 / 16 | GABA (−) | 0.87 | GABA (0.49) | – |
| DNge033 | 0 / 641 | GABA (−) | 0.89 | ACh (0.47) | disagreement, but **output only** (no input to LOOP5) |

**No partner is labelled excitatory in the model and inhibitory in MaleCNS** (flag count 0). The only partners labelled excitatory in the model and inhibitory in FlyWire are:
- CB0591, whose FlyWire `known_nt` is "gaba-negative", so this is not a real conflict;
- JO-FV, with a mixed per-cell FlyWire top_nt.

Inputs labelled inhibitory in the model but FlyWire-monoamine (AN05B004/040/046, DNg80, AN12B011) are small, at ≤ 48 synapses each. They are listed in `static.txt`.

**Literature (each source opened unless marked):**
- **Stürner et al. 2025** DN atlas, PDF opened at https://faculty.washington.edu/tuthill/docs/sturner_2025.pdf. DNg62 appears only in Fig. 2, in the section "Matching neurons across three datasets". DNge027, DNg21, DNge044, DNg22, GNG423 and SAD093 are not named.
  - Section "Sensory input onto DNs": DN→DN connections make up 413,458 output synapses, 42% of DN brain output, many of them axo-axonic. Strong DN–DN coupling is therefore a documented feature of the wiring.
  - DN transmitters come from EM prediction (Eckstein et al. 2024), with no peptide data. This is broadly consistent with the Stürner 2025 Discussion, but the methods sentence itself is not confirmed.
- **Virtual Fly Brain, DNg62** (https://www.virtualflybrain.org/term/dng62-vfb_fw039180/): classed as "antennal grooming descending neuron 1"; cholinergic, 62% confidence, citing FBrf0259490. MaleCNS lists the synonym "Hampel 2015: aDN1".
- **Hampel et al. 2015, eLife 4:e08758** (aDN): **could not be opened.** elifesciences.org needs JavaScript and PMC returned a reCAPTCHA. Anything about aDN transmitter or persistence is unverified.
- **Virtual Fly Brain, DNg22** (https://www.virtualflybrain.org/term/dng22-fbbt_00047614/): the per-instance tags disagree. The FlyWire instance is "Serotonergic", the MaleCNS instances "Cholinergic" and the MANC instances "GABAergic". The page does not mention FMRFamide; that claim comes only from FlyWire `known_nt_source` "Song et al., 2023 (immuno)", which I did not open.
- **Eckstein et al.** NT classifier: I opened the bioRxiv v1 preprint (https://www.biorxiv.org/content/10.1101/2020.06.12.148775v1.full), section "Classifier accuracy" and Table 1. Accuracy is 87% at synapse level and 94% at neuron level (neuron split). The hemilineage-split test set had no 5-HT or OA neurons (this statement is in the Fig. 3 caption, not Table 1). The published Cell 2024 version (PubMed 38729112) **could not be opened.**
- **Cheong et al. 2024** (eLife 96084; MANC DN organization) and **Marin et al. 2024**: **not opened.** The eLife page needs JavaScript. Any MANC statement about these types is unverified.

## Q2: identity and annotation (`body_table.csv`, `static.txt`)

**Cell counts match FlyWire exactly** (`annot.tsv`, via `flywireType`):
- DNge027, DNg21, DNg62 and DNge044: 2 each (1 L + 1 R).
- GNG423 = CB1582: 4 (2 + 2).
- SAD093: 2. GNG282 = CB0058: 2.
- There are no extra non-traced bodies of these types in MaleCNS.
- `mancType` exists for DNge027 (DNfl046), DNge044 (DNfl030), DNg21 and DNg62. **MANC and BANC counts were not checked** (no local data).

**Labels are consistent:**
- Instances are `<type>_L/R`, matching `somaSide`.
- Each type has a single `group` id.
- The ito-lee hemilineage is the same within each type.
- Soma positions are mirror-symmetric in x.
- The only synonym is DNg62 = "Hampel 2015: aDN1".

**No merge or split signature:**
- Synapse totals have z-scores of +0.36 to +1.04, on log10(in + out) against all 1,314 traced DNs.
- Within each type, cells are 0.89–1.16× the type mean.
- Left/right max/min ratios are 1.01–1.09, at the 12th–66th percentile of the 386 two-cell DN types (median 1.06, p95 1.34).
- Status is "Prelim Roughly traced" for the DNs, as for 1,184 of 1,314 DNs, and "Roughly traced" or "Prelim Roughly traced" for GNG423. None is Orphan, Assign or otherwise low-confidence.
- FlyWire flags the SAD093_R root as `outlier_seg`. This is FlyWire only; the MaleCNS SAD093 pair is symmetric (ratio 0.97 / 1.03).

**Left/right symmetry of the loop is high.** Type-pair synapses, written LL / RR | LR / RL:

| pair | LL / RR \| LR / RL |
|---|---|
| DNge027 → DNge044 | 2 / 1 \| 247 / 248 |
| DNg62 → DNge027 | 6 / 13 \| 198 / 187 |
| GNG423 → DNge027 | 59 / 35 \| 432 / 382 |
| DNg21 → DNge027 | 189 / 273 \| 151 / 212 |
| DNge027 L ↔ R | 200 / 162 |

Mirror-pair edges agree within about 1.5× for the main edges (247/248, 198/187, 432/382); some smaller type-pair edges differ up to 3.2× (DNge027→DNg62 contra 32 vs 10; DNge044→GNG423 ipsi 120 vs 43, contra 20 vs 8; DNge044→DNge027 ipsi 68 vs 141). The full 12 × 12 body matrix is in `static.txt`. A reconstruction error would not normally reproduce itself on both sides.

## Q3: connectivity anomalies

**Reciprocal edges are strong, but in the tail of a real distribution, not off it** (`recip_pairs.csv`). Measure: min(A→B, B→A) per cell pair.
- **Reference:** 11,976 reciprocal DN–DN pairs: median 1, p99 38, max 723 (DNg33 L↔R, the Week 3b clique).
  - GNG–GNG reference (GNG* types, LB/MX/MD/GNG neuromeres and DNg*): 54,979 reciprocal pairs, p99 40, max 914.
- **LOOP5 pairs:**
  - DNge027 L↔R (162) ranks **5th** of all DN–DN pairs; DNg21_L↔DNge027_L (148) 6th; DNg21_R↔DNge027_R (127) 9th.
  - 12 LOOP5 pairs are above the DN–DN 98.8th percentile. The GNG423 pairs are ranked against this DN–DN reference although GNG423 is not a DN.
- **Comparable DN pairs exist outside the loop:** DNge037 L↔R 287, DNg16 L↔R 180, DNge097 L↔R 170 and DNg100 L↔R 119.
- **Fraction of input from other DNs:** 0.35–0.50 for the LOOP5 DNs, at the 89th–98th percentile of all DNs (median 0.12).
- **LOOP5 is an outlier in degree, not in kind.**

**Autapses:** none in LOOP5, raw or final; the diagonal is 0 for all 12 bodies. Across the whole CNS the raw W has 97 diagonal entries and the final W 60.

**Inhibition onto LOOP5 is present at a normal level** (final W, synapse-equivalents = W / (0.275e-3 × 0.581)):
- **Totals:** excitatory 22,448 and inhibitory 13,764 (fraction 0.38).
- **Per cell, raw:** inhibitory fraction 0.277–0.462, at the 10th–67th percentile of all traced DNs (median 0.42), within the normal range. (Range corrected by QA; the earlier per-cell figures, 0.30–0.45 at the 14th–62nd percentile, were wrong.)
- **Largest inhibitory sources:**
  - DNg98 GABA 2,153;
  - DNge142 GABA 825;
  - GNG404 Glu 637;
  - GNG361 Glu 475;
  - GNG231 Glu 457;
  - DNge029 Glu 442;
  - DNge136 GABA 396;
  - GNG281 GABA 391;
  - DNg70 GABA 349;
  - DNge076 GABA 345.
- **Full list:** `inhib_inputs.csv`, with the targets per type.
- **Activity in the tail (saved ep009):**
  - DNg98 18 Hz, GNG281 51 Hz, GNG166 51 Hz, GNG150 102 Hz, DNge029 61 Hz.
  - DNge142, GNG404, GNG361, GNG231, DNge136 and DNge076 are silent.
  - The recruited inhibition is outweighed in mean field. Into DNge027 in the tail: +98 mV from LOOP5 against −7.6 mV from GNG281 and −3.7 mV from DNg98 (an estimate, not a membrane potential).

**What the build changed (H2):**
- **Rows (inputs) of LOOP5:** model W / VSYN vs raw. The only entries changed by more than 0.5 synapse are **JO-FV, −34** (right-JO mirror). Every inhibitory entry is identical in final and raw (raw_inh = model_inh for all 170 types).
- **Monoamine removal:** it deleted 290 synapses onto LOOP5, for example DNge149 (OA) 102, GNG002 (5-HT) 49 and DNge151 (OA) 41. In this model those would have been **excitatory** (`monoamine='excitatory'` gives +1), so removing them removed excitation, not inhibition.
- **Steps that cannot touch LOOP5 rows:**
  - presynaptic row removal acts only on sensory rows;
  - the LN–PN cut and APL act on AL / KC entries;
  - `mn_glu` acts on MN columns;
  - `sign_fix` covers GLNO / LPsP only;
  - `clique_gain` covers DNg33 / IN19B040 only.
  - Columns of LOOP5 changed only onto BM / JO sensory rows, through the presyn step.
- **Not checkable:** inputs from non-Traced bodies (Orphan 15,925, Assign 1,832, …). The model keeps only `status == 'Traced'`, and the full connectome-weights feather is not on disk.
- **Right-JO mirror (entry only):** JO-R → SAD093_R goes from 326 raw to 716 synapse-equivalents. It does not change the loop; it is tested in Q4 (`nomirror`).

**Inhibitory partners with the wrong polarity:**
- None of the main inhibitory inputs is excitatory in either dataset.
- Where FlyWire and MaleCNS disagree, both labels are inhibitory: GNG281 and GNG150 are GABA in MaleCNS and Glu in FlyWire.
- Glutamate is −1 throughout, as is the model convention. Stürner et al. note that glutamate can also excite; this was not tested.

## Q4: tests of the strongest suspects (`analyze_runs.txt`)

**Variants.** All are on a copy of W except `nomirror`. Wind episodes are ep009, ep002 and ep016; non-wind episodes are ep000 and ep019.
- **`mono5`:** outputs of DNg22, GNG671, AN05B096, AN05B105 and LN-DN1 deleted (6,015 entries). These are the excitatory loop inputs that FlyWire calls 5-HT or DA, or that MaleCNS calls "unclear". Deletion is what `monoamine='removed'` would do after relabelling.
- **`DNg21_inh`:** DNg21 outputs set to −|W|. DNg21 is the lowest-confidence LOOP5 member, and MX12 is 34% Glu.
- **`GNG423_inh`:** GNG423 outputs set to −|W|. GNG423 has the lowest FlyWire confidence in the loop.
- **`DNg22_inh`:** DNg22 outputs set to −|W| (1,094 entries), testing the MANC "GABAergic" reading.
- **`nomirror`:** `cns_model(..., mirror=False)`.

**Results.** "Persistence" is the last DNge027 spike minus wind offset, checked against the time course in 100 ms bins. "Brain 2.4–2.5" is distinct brain_mask neurons active in the last bin; background is 23–100. "Saved" is the Day 5 run.

| variant | ep009 persistence | ep002 persistence | ep016 persistence | DNge027 Hz in word (09/02/16) | MN spikes in word vs saved | brain 2.4–2.5 (09/02/16) |
|---|---|---|---|---|---|---|
| saved | to end (≥ 0.73 s) | 1.05 s | to end (≥ 0.54 s) | 129 / 126 / 125 | – | 464 / 396 / 422 |
| mono5 | to end | 1.10 s | to end | 140 / 134 / 132 | +6 / +2 / +4% | 462 / 436 / 431 |
| DNg22_inh | to end | 1.10 s | to end | 143 / 134 / 143 | +9 / +3 / +9% | 452 / 426 / 432 |
| DNg21_inh | **0.27 s** | **1.10 s (to end)** | **0.28 s** | 96 / 97 / 98 | −11 / −5 / −5% | 90 / 369 / 82 |
| GNG423_inh | none (no ignition) | none | 0.02 s | **6.6 / 1.5 / 5.4** | **−39 / −29 / −35%** | 94 / 27 / 65 |
| nomirror | to end | 0.25 s | 0.35 s | 131 / 128 / 125 | −8 / −7 / −4% | 474 / 31 / 81 |
| ref.: LOOP5 × 0.3 (T2c) | 0.01 s | 0.05 s | 0.09 s | 36 / 30 / 32 | −19 / −14 / −15% | background |

**Non-wind episodes (ep000, ep019):**
- DNg21_inh, GNG423_inh and DNg22_inh are **spike-for-spike identical** to the saved runs (170,706 and 185,681 spikes).
- mono5 differs slightly (170,865 and 185,402 spikes), because GNG671 and the others fire during other words.

**What the runs show:**
- **DNg22, the largest annotation conflict among the loop partners:**
  - neither deleting it (as 5-HT) nor making it inhibitory (as GABA) shortens persistence in any episode;
  - its mean-field drive to DNge027 in the tail is only about 1.8 mV against about 98 mV from LOOP5.
- **DNg21 → inhibitory:**
  - shortens the tail in 2 of 3 episodes, from at least 0.5–0.7 s to 0.27–0.28 s, but leaves ep002 persisting to the end;
  - in the tail DNge027 still fires at about 80–100 Hz (15–20 spikes per 100 ms over 2 cells).
  - This flip has no annotation support (ACh in both datasets) and still does not remove persistence.
- **GNG423 → inhibitory:**
  - removes the state by removing **ignition**: DNge027 is at 1.5–6.6 Hz during the word and wind-word MN output falls by 29–39%;
  - GNG423 is the SAD093 relay, so this cuts the wind pathway itself;
  - it is not a targeted fix, and both datasets call GNG423 ACh (MaleCNS 0.87–0.95).
- **No right-JO mirror:** ignition is unchanged (DNge027 125–131 Hz). The tail is shorter in ep002 and ep016 but lasts to the end in ep009.
  - The mirror adds entry drive and so may lengthen the tail on average (n = 1 seed per episode; suggestive only).
  - It is not the cause: the loop still ignites and can persist without it.
  - With one seed per episode and stochastic termination, this effect on duration is suggestive only.

## Hypotheses (C2)

| hypothesis | verdict | deciding numbers |
|---|---|---|
| **H1:** mislabelled transmitter | **Does not hold** | All 12 LOOP5 cells are ACh in MaleCNS (0.80–0.97; consensus ACh) and in FlyWire (0.451–0.70 per cell; 0.46 is the GNG423 type mean); no dataset disagrees. The one strongly disagreeing partner, DNg22 (ACh 0.59–0.62 / 5-HT 0.66–0.78 / FMRFa / MANC GABA), does not matter: `mono5` and `DNg22_inh` persist to the end in 2 of 3 and to 1.10 s in ep002. Flipping the weakest-confidence member (DNg21) leaves ep002 persisting; flipping GNG423 kills the wind response (MN −29 to −39%), not just the tail. |
| **H2:** removed inhibition | **Does not hold** | Every inhibitory entry onto LOOP5 is identical in final and raw W. Monoamine removal deleted 290 synapses that the model would have made excitatory. The inhibitory fraction is 0.277–0.462 (10th–67th percentile of DNs). The only build edit on LOOP5 rows is JO-FV −34. **Caveat:** inputs from non-Traced fragments could not be checked. |
| **H3:** reconstruction error | **Does not hold** | Counts match FlyWire exactly (2 / 4 / 2 / 2 / 2). Synapse-total z is +0.36 to +1.04, L/R max/min 1.01–1.09 (≤ 66th percentile), mirror-symmetric loop edges (for example 247 / 248), no autapses and no low-confidence status. |
| **H4:** genuine strong reciprocal excitation, plus missing adaptation or depression | **Holds provisionally / most likely**, by elimination, plus positive evidence | Reciprocal pairs rank 5th, 6th and 9th of 11,976 DN–DN pairs but are below the DNg33 clique (723), with comparable non-loop pairs (DNge037 287). DN-to-DN input is 35–50% (89th–98th percentile). The five largest recruited inhibitors give about −15 mV into DNge027 against about +98 mV of loop drive in mean field. Scaling only the loop edges × 0.3 removes persistence in 3 of 3 (T2c). **Unchecked:** inputs from non-Traced fragments (Orphan 15,925, Assign 1,832 bodies; the weights feather is not local), per-class NT probabilities, and FlyWire/MANC/BANC connectivity. The × 0.3 result and the mean-field estimate cannot tell genuine wiring from a single-dataset artifact, and missing adaptation was not tested. |

## Root-cause verdict (C5)

**No annotation or build error found. The loop is most likely genuine strong reciprocal excitation (provisional; H4 lists what remains unchecked).**
- The five LOOP5 types are cholinergic in MaleCNS and FlyWire, correctly typed, bilaterally symmetric, and wired with reciprocal edges in the top 1–2% of DN–DN pairs.
- In a LIF network without spike-frequency adaptation, synaptic depression or neuromodulation, this produces a metastable high-rate state, by the same mechanism as the DNg33 / IN19B040 cliques in Week 3b.
- The looming precedent (GLNO / LPsP sign errors) does **not** repeat here.
- **Recommended:** option A as Charlie prefers, i.e. LOOP5 internal edges × 0.3. It is already scanned in T2c: persistence ≤ 0.1 s in 3 of 3, non-wind episodes spike-identical, wind-word MN −14 to −19%, DNge027 in the word about 30–36 Hz. The lasting alternative is generic adaptation or depression.
- **Side effects:**
  - wind-word MN and brain recruitment fall (above);
  - n = 3 with 1 seed each;
  - other latent loops may surface;
  - GNG423 → inhibitory is **not** recommended, because it abolishes the wind response.

## Assumptions
- **Sign convention:** as in `sim/malecns.py` (GABA / Glu / His −1, ACh / unclear +1, monoamines removed). Glutamate is treated as inhibitory throughout.
- **Partner and inhibitory thresholds:** 100 and 5 synapses, chosen by hand.
- **"GNG" reference set:** GNG* types, GNG / LB / MX / MD soma neuromeres and DNg* DNs. The DN–DN set is all `descending_neuron`.
- **Mean-field mV:** τ_syn 5 ms × W × rate (ep009 rates 2.1–2.5 s). This is an estimate, not a membrane potential.
- **Persistence:** the last DNge027 spike after offset, checked against the time course in 100 ms bins (`analyze_runs.txt`).

## Unverified / open
- **MaleCNS per-class NT probabilities:** the tbar file is absent and its download is blocked by the proxy (403 for male-cns.janelia.org).
- **Synapses from non-Traced bodies onto LOOP5:** unknown; the full weights feather is not local.
- **FlyWire, MANC and BANC connectivity for the same pairs:** not compared, because no connectivity data is local. Whether the reciprocal counts replicate across datasets is unverified. Only cell counts and NT predictions were compared with FlyWire.
- **Literature not opened:** Hampel et al. 2015 (aDN), Cheong et al. 2024, Marin et al. 2024, Song et al. 2023 (DNg22 FMRFa), and the published Eckstein et al. 2024. No paper was found or opened that reports the transmitter or persistent activity of DNge027, DNg21, DNge044 or GNG423.
- **Seed coverage:** 1 seed per episode; termination of the state is stochastic. The DNg21_inh and nomirror shortenings are therefore suggestive, not established.
- **Glutamate polarity:** the excitatory reading of glutamate (Stürner et al., Discussion) was not tested.

## Fix chosen (T2e)
Charlie chose option A: `sim/cns.py` `LOOP_GAIN` scales the 101 W entries among the five LOOP5 types by 0.3 and is part of `W4_CALIB`. Verification over 30 Day 5 wind specs:
- the tail is ≤ 0.15 s in 22 of 30 and > 0.3 s in 1 of 30 (ep023, 0.314 s). The Orchestrator accepted this residual;
- non-wind, bg-only and looming runs are spike-identical.

Details are in `progress/week4/day5_loop_gain.md`. This is a targeted patch; generic adaptation or depression remains the lasting alternative.
