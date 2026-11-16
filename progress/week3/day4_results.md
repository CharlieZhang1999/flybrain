# Week 3 Day 4 results: 37/37 action verification (10 Nov 2026)

Brain: calibrated MaleCNS + presynaptic modulation (K = 0.05) + right-JO mirror (`right_jo_fix.md`). Shuffled control: the same, with the chemical matrix shuffled while keeping each neuron's degree and output sign (as in Week 2).
Code: `scripts/w3d4_actions.py`, `scripts/w3d4_score.py`; codes in `loop/actions.py` `ACTION_CODES_MIRROR`.
Results: `results/loop/w3d4_{real,shuffle}_{train,test}_*.npz`, `w3d4_score.json`, `w3d4_log.txt`.

## Protocol (the loop setting, not from rest)
Each trial is one continuous run: 5 different random words back to back (500 ms each), then the action code (500 ms), then 500 ms of silence. The readout sees only the 500 ms action window. A median of 213 neurons (real) or 154 (shuffled) were still firing from the episode when the code started, up to ~1,900 after wind/looming.

| Action | Code (all on JO hearing neurons) |
|---|---|
| turn left | JO-B left, 150 Hz |
| turn right | JO-B right (mirrored), 150 Hz |
| go forward | all JO-A + JO-B, 150 Hz |
| stop | all JO-B, 20 ms on/off pulses at 300 Hz |
| eat | all JO-A, same pulses |

- **Training:** 48 trials, 8 for each of the 5 actions + `none` (episode followed by silence), seeds 500–547.
- **Test:** 37 new trials, seeds 10–46, with actions and words in random order (eat 8, go forward 5, stop 9, turn left 8, turn right 7).
- **Readout:** L2 logistic regression on log spike counts.

## Results
| Brain | Readout | Held-out correct | Neurons that fire at all |
|---|---|---|---|
| real | descending neurons only (1,314) | **37 / 37** | 445 |
| real | all non-sensory | 37 / 37 | 3,361 |
| shuffled | descending neurons only | 36 / 37 (one turn right read as `none`) | **24** |
| shuffled | all non-sensory | 37 / 37 | 1,049 |

## What it means
- The original demo's 37/37 is reproduced, from descending neurons only, after a word episode has already left activity in the brain.
- **The shuffled brain does almost as well (36/37).** As with the words in Week 2, reading back a stimulus you injected yourself works with almost any wiring. The difference the real wiring makes is how the code reaches the brain's output: it drives 445 DNs in the real brain against 24 in the shuffled one, and the shuffled brain's one error is a turn being read as nothing.
- The codes remain arbitrary labels. The steering DNs (DNa01/DNa02) stay at or below about 1 Hz for every code, so "turn left" is not a steering command (see `day3_results.md`, `right_jo_fix.md`).

## Follow-ups before Day 5
- The mirror also changes the `sound` word (JO-A/B on both sides). Retrain and check the word readout on the mirrored brain before running the full loop.
