# Week 3 Day 3 results: action codes on the hearing neurons (10 Nov 2026)

Brain model: calibrated MaleCNS + presynaptic modulation (K = 0.05).
Code: `loop/actions.py` (codes, JO groups, chosen mapping), `scripts/w3d3_code_screen.py`, `scripts/w3d3_code_select.py`.
Results: `results/loop/w3d3_screen_<code>.npz`, `w3d3_select.json` (every 5-of-8 subset ranked), `w3d3_dn_steering.csv`, `w3d3_log.txt`.

## Protocol
8 candidate codes on Johnston's organ (JO) hearing neurons, each 500 ms of drive from rest, then 500 ms of silence, seeds 0–9. Left and right groups are matched in size, using a fixed random subsample of the larger side (JO-A 18 per side, JO-B 29 per side). The side comes from the `_L`/`_R` instance suffix.

| Code | Neurons | Pattern | Descending neurons > 1 Hz (left / right side) |
|---|---|---|---|
| A-L | JO-A left (18) | 150 Hz | 28 (23 / 5) |
| A-R | JO-A right (18) | 150 Hz | **0** |
| B-L | JO-B left (29) | 150 Hz | 22 (19 / 3) |
| B-R | JO-B right (29) | 150 Hz | **0** |
| A+B | all JO-A and JO-B (138) | 150 Hz | 45 (36 / 9) |
| A-pulse | all JO-A | 20 ms on / 20 ms off at 300 Hz | 27 (22 / 5) |
| B-pulse | all JO-B | same pulse train | 32 (27 / 5) |
| A-low | all JO-A | 50 Hz | 9 (9 / 0) |

Readout: only the 1,314 descending neurons (DNs), log spike counts in the 500 ms window, L2 logistic regression, leave-one-seed-out. 60 of the 1,314 DNs fire at all for these codes.

## Results
- **All 8 codes: 87.5%.** Every error is B-R read as A-R. Both drive nothing.
- **The right-side JO neurons are nearly disconnected in MaleCNS.** Matched groups of 18 JO-A neurons make 16,869 output synapses on the left and 571 on the right. For JO-B (29 per side) it is 10,890 vs 1,090. Right-side stimulation reaches no DN above 1 Hz. This looks like an incomplete reconstruction of the right antennal nerve (24 of the right JO-A neurons are typed `JO-A-unclear`), not biology. It also explains why even bilateral codes mostly drive left-side DNs.
- **Every 5-code subset without the right-side codes is 100% (leave-one-seed-out).** Chosen: A-L, B-L, A+B, A-pulse, B-pulse (100%, mean top-two probability margin 0.94). A-low was left out because it drives only 9 DNs, the least robust once earlier episodes leave activity behind.
- **"Does left mean left?" No.** The steering DNs DNa01, DNa02, DNa03 and DNg13 are essentially silent for every code (at most 3 Hz in DNa01_R). The codes are carried mainly by DNp01 (the giant fiber, up to 99 Hz), DNg33 (both sides, ~110 Hz for A-pulse), DNp12, DNp11, DNp55, DNg84, DNg29 and DNge145. So a code is an arbitrary label that the DN population carries, not a steering command.

## Chosen mapping (`loop/actions.py` ACTION_CODES; arbitrary)
| Action | Code |
|---|---|
| turn left | A-L |
| turn right | B-L |
| go forward | A+B |
| stop | B-pulse |
| eat | A-pulse |

## Next: Day 4
Verify on 37 fresh trials (seeds 10–46) in the loop setting: each action code plays right after a 5-word episode in one continuous run, so the looming/wind tails are present. Train with that history too, and add a `none` (no code) class. Run the same on the shuffled brain. Target 37/37.
