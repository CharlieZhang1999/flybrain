# Right-side Johnston's organ (JO) fix: a mirror calibration (10 Nov 2026, side task)

**Short answer.** In MaleCNS v1.0 the right JO hearing neurons are poorly connected because the right side was reconstructed less completely. This is not biology. An optional "mirror" calibration gives each right JO neuron the outputs of its left counterpart, sent to the mirror-image partner neurons. With it, right JO codes drive as many descending neurons (DNs) as the left codes do, and those DNs are the mirror images of the left ones. A DN-only readout then separates A-L, A-R, B-L and B-R perfectly. The mirror is off by default and nothing existing was edited.

## 1. Diagnosis: a reconstruction artifact

The JO body counts are almost the same on both sides. What differs is how the right bodies were typed and how many synapses they make.

| MaleCNS v1.0 (traced network, minconf 0.5) | Left | Right |
|---|---|---|
| JO bodies (all 672 are `Traced`; there are no non-Traced JO bodies) | 348 | 324 |
| Mechanosensory bodies entering through the antennal nerve | 387 | 357 |
| JO bodies typed "unclear" (JO-A-unclear, JO-B-unclear, JO-unclear) | 9 | 130 |
| JO bodies with 0 output synapses | 2 | 84 |
| Total JO output synapses | 197,441 | 67,557 (34%) |
| JO-A output synapses (n bodies) | 16,869 (18) | 880 (32; 24 unclear, median 0) |
| JO-B output synapses (n bodies) | 21,094 (59) | 1,090 (29; median 16) |
| JO-C / D / E / F output synapses | 34–91% of the left value | |

FlyWire 783 is a different animal, and its connectivity table only keeps connections of at least 5 synapses. In FlyWire the two sides are balanced:

| FlyWire 783 | Left | Right |
|---|---|---|
| JO bodies | 586 | 518 |
| JO-A bodies / output synapses | 48 / 1,232 | 51 / 3,948 |
| JO-B bodies / output synapses | 157 / 5,004 | 103 / 4,611 |
| All JO output synapses | 42,492 | 46,270 |

Why this points to an artifact:

- **The bodies are there but cut off.** The right JO bodies exist in normal numbers, but many have no synapses and could not be given a subtype. This is the pattern you get from axons that were fragmented and not fully proofread. Real neurons that were simply missing would show up as low body counts, and they don't.
- **The deficit follows a gradient.** It is worst in JO-A and JO-B (about 5% of left synapses on the right) and mild in JO-E and JO-F (40–130%). A whole missing hemisphere of hearing would not look like this. A partly proofread nerve would.
- **FlyWire is roughly symmetric.** There, right JO-A even has more output synapses than left.
- **The biology is symmetric.** Each JO subgroup (A–E) sends its axons into its own zone of the AMMC on the same side of the brain, and this zone map is the same on both sides ([Matsuo et al. 2014, Front Physiol](https://www.frontiersin.org/articles/10.3389/fphys.2014.00179/full), which summarises Kamikouchi et al. 2006, J Comp Neurol).
- **The MaleCNS paper does not mention this.** It says nothing about how complete the antennal nerve is. It does say that sensory classes such as photoreceptors are "missing in the male CNS" ([Berg et al. 2025, bioRxiv / Cell 2026](https://pubmed.ncbi.nlm.nih.gov/41279223/)). So the paper accepts that sensory afferents are incomplete in places.

Sources in this repo: `scripts/w3d3b_right_jo_diag.py`, which writes `results/loop/w3d3b_diag_malecns.csv`, `w3d3b_diag_flywire.csv` and `w3d3b_diag.json`.

## 2. The fix: `sim/jo_mirror.py`, `mirror_jo(W, meta, mode='replace')`

The function works on the loop's model weight matrix, `presyn_model(0.05)`. It changes only the columns (the outputs) of the 324 right JO neurons. It does not touch their rows, because sensory rows were already removed.

1. **Build a left template for each subtype.** For each left JO subtype (for example JO-A1 or JO-B1_a), it takes the mean output of one neuron onto each partner type on each side.
2. **Mirror the template.** That output is moved to the partner type on the opposite side and split evenly across its neurons. When a type has one neuron per side, this is an exact one-to-one match. Midline partners map to themselves. Partners with no opposite-side homolog are dropped, which loses only 1.3% of the weight.
3. **Give each right neuron a template:**
   - **Same subtype (194 neurons):** the right neuron gets its own subtype's template.
   - **"Unclear" subtype with some outputs (35 neurons):** it gets the template that best matches its existing outputs (cosine similarity), chosen within its group.
   - **JO-A-unclear or JO-B-unclear with no outputs (28 neurons):** it gets the mean of all left JO-A or JO-B neurons.
   - **JO-unclear with no outputs (67 neurons):** left unchanged, because their zone is unknown. None of the action codes use them.
4. **Replace, not add.** In the default `mode='replace'`, the right neurons' fragment outputs are replaced. `mode='max'` keeps any right-only contacts as well.
5. **Neuron counts stay as in the data.** For example, there are still 32 right JO-A neurons. The size-matched code groups (18 / 29 per side) are unchanged.

Assumptions:

- Left and right JO afferents of the same subtype connect the same way onto mirror-image partners.
- MaleCNS `type` names identify true homolog pairs.
- The left side is complete enough to act as the template.

Results of the checks (`results/loop/w3d3b_mirror_check.json`):

- **Only the right JO columns changed.** 257 columns changed, all of them right JO. In every other column the largest difference is 0.0.
- **The right side now matches the left (summed output weight of the size-matched groups):**

  | Group | Before | After |
  |---|---|---|
  | A-R vs A-L | 0.073 vs 2.503 | 2.307 vs 2.503 (92%) |
  | B-R vs B-L | 0.170 vs 1.633 | 1.499 vs 1.633 (92%) |

- **Right neurons reach more partners.** Because the output is split evenly across homologs, the right groups contact more partners with weaker edges: 528 vs 408 for A, 580 vs 398 for B.

## 3. Test (loop model + mirror; 150 Hz for 500 ms; 1.0 s run; seeds 0–4; one process)

Scripts: `scripts/w3d3b_right_jo_sim.py` and `scripts/w3d3b_right_jo_analyze.py`. The baseline is seeds 0–4 of the Day 3 screen.

| Code | DNs > 1 Hz, baseline (left / right) | DNs > 1 Hz, with mirror (left / right) | Top DNs with mirror (Hz) |
|---|---|---|---|
| A-L | 28 (23 / 5) | 28 (23 / 5), identical | DNg84_L 78, DNp55_L 64, DNg33 L/R 42 |
| A-R | **0** | **19 (5 / 14)** | DNg33 R/L 121, DNp55_R 74, DNg84_R 56, DNge130_R 28 |
| B-L | 22 (19 / 3) | 22 (19 / 3), identical | DNp01 (giant fiber)_L 90, DNp12_L 47, DNge145_L 45, DNp11_L 44 |
| B-R | **0** | **23 (2 / 21)** | DNp01 (giant fiber)_R 93, DNge145_R 39, DNg29_R 39, DNp11_R 38, DNp12_R 37 |

- **The right codes are mirror images of the left codes.** Comparing DN type-and-side rates for a left code with the mirrored right code gives r = 0.71 for JO-A and r = 0.83 for JO-B (`w3d3b_symmetry.json`).
- **The steering DNs stay silent.** DNa01 and DNa02 are at most 0.8 Hz for every code, with or without the mirror. So "left does not mean left" still holds.
- **The DN-only readout (leave-one-seed-out, 4 classes) is now perfect.** It is 75% at baseline, where B-R and A-R are confused, and **100% (20/20)** with the mirror (`w3d3b_readout.json`).
- **The mirror creates no new tails.** In the last 0.3 s the only neurons firing are the DNg33 pair:

  | Code | Trials where DNg33 keeps firing (~154 Hz) | Neurons firing in the last 0.3 s, per trial |
  |---|---|---|
  | A-L | 2 of 5 | 0.8 |
  | A-R | 5 of 5 | 2.0 |
  | B-L | 0 | 0 |
  | B-R | 0 | 0 |

  This DNg33 lock-in was already in the model. At baseline, A+B and A-pulse lock DNg33 in 10 of 10 trials, and A-L in 6 of 10. A-R drives DNg33 harder, so it behaves like those codes.

## 4. Recommendation

**Adopt the mirror for the action codes.** Use `W, _ = mirror_jo(W, R.meta)` after `presyn_model(0.05)`. It is grounded in biology (bilateral symmetry, and FlyWire balance), it changes nothing except the right JO outputs, and the left codes give exactly the same results as before. It lets A-R and B-R be used, so "turn left" and "turn right" can be given mirror-image codes, for example A-L and A-R. A real steering signal would still need a different sensory channel, because DNa01 and DNa02 stay silent.

Two things to do before adopting it:

- **Re-run the Day 4 verification with the mirror, including the episode-history setting.**
- **Remember that A-R locks DNg33 into firing in every trial.** If that lock-in matters in the loop, B-R is the cleaner right-side code: it leaves no tail.

## Files

| Kind | Path |
|---|---|
| New module | `sim/jo_mirror.py` |
| Scripts | `scripts/w3d3b_right_jo_diag.py`, `scripts/w3d3b_right_jo_sim.py`, `scripts/w3d3b_right_jo_analyze.py` |
| Results | `results/loop/w3d3b_*`: diag csv/json, `mirror_check.json`, `mirror_assign.csv`, `mirror_templates.csv`, `mirror_<code>.npz`, `dn_summary.csv`, `dn_top.csv`, `steering.csv`, `readout.json`, `symmetry.json` |
| Logs | `results/loop/w3d3b_diag_log.txt`, `w3d3b_sim_log.txt`, `w3d3b_analyze_log.txt` |
