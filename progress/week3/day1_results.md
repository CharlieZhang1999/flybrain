# Week 3 Day 1 results: episodes and carry-over (9 Nov 2026)

Scripts: `scripts/w3d1_common.py` (calibrated model setup), `scripts/w3d1_blocks_check.py`, `scripts/w3d1_episodes.py`, `scripts/w3d1_decode.py`.
Results: `results/loop/w3d1_carryover.csv`, `w3d1_by_prev.csv`, `w3d1_summary.json` (every error), `w3d1_log.txt`; raw window counts in `w3d1_ep_*_X.npz` / `_info.npz`.
Model: Week 2 calibrated MaleCNS (brain only, monoamines removed, w × 0.581, rate APL, chemical cholinergic LN→PN = 0, eLN–PN gap junctions c0 = 0.2, 2 mV spikelets).

## 1a. Simulator: time-varying drive
`sim/lif_numpy.py` `run_trial` takes a new `blocks=[(t0, t1, idx, rate_Hz), ...]` argument: several stimuli inside one continuous run, so the network state carries over. I kept the name `schedule` for the existing step-order argument (renaming it would break the older scripts) and called the new one `blocks`. With `blocks=None` nothing changes.
**Regression:** one block (0–0.5 s, vinegar, seed 3) gives exactly the same 12,001 spikes, at the same times, as the normal call. Speed: about 22 s per biological second in one process, about 27 s with 4 running in parallel.

## 1b. Carry-over test
40 episodes of 5 different words drawn at random from the 17 (the same 40 sequences at every spacing), 500 ms of drive at 150 Hz per word, then 500 ms of silence at the end. Poisson seeds 1000–1039, so no episode reuses a Week 2 trial seed. Spacing between words: 0, 250 or 500 ms. That is 200 word windows per spacing.
The readout uses Week 2's settings (log(1 + count) per neuron in the 500 ms window, stimulated sensory neurons excluded, L2 logistic regression). It has 18 classes: the 17 words plus `none`.

| Readout | Features | 0 ms | 250 ms | 500 ms |
|---|---|---|---|---|
| Isolated (trained on Week 2's 170 single-word trials + 10 `none`) | all non-sensory | **95.5%** | 96.0% | 95.5% |
| Same | DNs + MNs | 82.5% | 82.0% | 81.5% |
| Episode-trained, 5-fold over episodes (isolated trials + the other 32 episodes) | all non-sensory | **100%** | 100% | 100% |
| Same | DNs + MNs | 91.5% | 90.0% | 91.5% |

The first word of every episode decodes at 100% with the isolated readout (all neurons), so the errors come from what happened earlier in the episode.

**The errors are not about the previous word. They come from a lasting cold/humid state.** All 9 errors of the isolated all-neuron readout at 0 ms are words decoded as `cold`, mostly `hot` → `cold` (7 of them, plus umami and water once each). Every one comes after `cold` or `humid` appeared *earlier in the episode*, often 2–3 words back. Of the 15 `hot` windows, the 7 with an earlier cold or humid were all wrong, and the 8 without were all right. Gaps don't help, because the state doesn't decay:

| Silent window (500 ms gap or end of episode) | Neurons still firing (median, range) |
|---|---|
| no cold/humid/wind/looming earlier | 23–41 (1–220) |
| cold or humid earlier (no wind/looming) | 267–281 (241–927) |
| wind or looming earlier (no cold/humid) | 500–609 (6–1,213) |
| both | 800–870 (218–1,427) |

So the residual cold/humid tails from Week 2 (about 200 neurons, about 50–60 in the antennal lobe) are **self-sustaining for at least 2–3 s**. They overlap hot's thermosensory pathway, which pushes `hot` towards `cold`. The non-AL looming/wind loop leaves an even bigger state, but it doesn't confuse any word. Only 66% of the 500 ms gap windows read as `none`; the rest are the windows that come after these states.

## 1c. `none` class
10 no-drive trials on the calibrated model: 0 spikes. The 18-class readout never predicted `none` for a word window with all neurons. With DNs + MNs it did, for weak words (mold, vinegar; top probability about 0.32). The "unsure" rate (top probability < 0.5) is 0% with all neurons and 10–12% with DNs + MNs.

## Decision (plan rule)
- The isolated readout reaches 95.5% back-to-back, which is above the 95% bar. So the rule says: **0 ms gaps**.
- I also recommend the **episode-trained readout** for the loop. It is 100% in 5-fold cross-validation over episodes, including 7/7 of the hot-after-cold/humid cases. It costs nothing extra (the episodes are already simulated). It shows that the lasting state shifts the pattern but doesn't erase the word.
- For the loop, these persistent states build up from round to round. The fly → LLM readout copes, but Day 3's action readout must be trained with episode history before the action window too, not from rest.

## Limitations to carry into the README
- After cold or humid (and after wind or looming), the calibrated model keeps a few hundred neurons firing for seconds. Week 2 saw these as small tails in 1.6 s trials. Day 1 shows they don't die out. This is a property of the model, not of the readout.
- The DN + MN readout is weaker (82% isolated, 91% episode-trained). Most of its errors are mold/vinegar read as `none` or confused with each other, as already seen in Week 2.

## Update (9 Nov): fixed by modelling brain → sensory synapses as presynaptic modulation
See `presynaptic_sensory_model.md`. With that model (K = 0.05), the cold/humid lasting state is gone. The isolated readout gets the same 40 episodes 100% right (0 errors, was 9), and DN + MN accuracy rises from 82.5% to 90%.
