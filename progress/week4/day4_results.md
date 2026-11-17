# Week 4 Day 4: regression dataset design and 10-episode pilot

Question: is the Day 5 training set (about 200 episodes × 2 s, real + shuffled) feasible on this CPU, and informative (motor pools vary, and the shuffled control separates)?

## Built
- `stimuli/mixtures.py`: `sample_episode(rng, words=WORDS, T=2.0)` returns a JSON-able spec (Poisson seed, spec seed, background rate, and per word: rate, onset, offset). 2–4 words with distinct modalities (modalities drawn uniformly, then one word per modality), rate U(30, 150) Hz, onset U(0, T − 0.3) s, duration U(0.2, 1.0) s clipped to T, background U(1, 5) Hz on all 15,912 sensory neurons for [0, T). Times are rounded to the 0.1 ms step. `to_blocks(spec, meta, pos)` gives the `run_trial` blocks.
- `scripts/w4d4_pilot.py`: 10 mixture specs (seeds 4040–4049) + 2 background-only (1 Hz and 5 Hz, the ends of the range), each run with `calib` and `shuffle` (W4_CALIB, same spec and Poisson seed, CPU float32 rng=torch). Modes `check <k> <cfg>` (re-run and compare spikes) and `bgoff` (background switched off at 1.5 s).
- `scripts/w4d4_pilot_summary.py` → `results/w4/regression/pilot/pilot_summary.md` (+ `pool_variance.csv`, `group_variance.csv`). Raw log: `pilot_log.txt`.

Uniform-modality sampling puts looming in about 42% of episodes (2,083 of 5,000 sampled specs); in the pilot, 5 of 10.

## Motor-pool variance (10 ms bins, 2,000 bins pooled over 10 episodes)
Floor 1 Hz² (SD 1 Hz). One spike of a 1-neuron pool in one bin of the 2,000 already gives 5 Hz², so the floor asks for more than a stray spike in small pools.

| level | calib | shuffle | calib, no looming eps | shuffle, no looming eps |
|---|---|---|---|---|
| pools > 1 Hz² (of 349) | 202 | 25 | 144 | 15 |
| pools > 10 Hz² | 182 | 9 | 123 | 5 |
| group × side > 1 Hz² (of 42) | 36 | 7 | 33 | 4 |
| group × side > 10 Hz² | 28 | 0 | 25 | 0 |

Summed pool variance: calib 79,126 Hz², shuffle 463 Hz². In calib, 71% of it comes from the looming episodes, and the top 10 pools (TTMn, b2, hg1, MNwm35/36, MNad28) hold 32%. TTMn variance is 93–94% looming.

Looming saturates the giant fiber at every intensity tried:

| looming rate Hz | GF Hz | TTMn Hz | DLMn Hz |
|---|---|---|---|
| 34 | 183 / 185 | 180 / 184 | 38 / 63 |
| 45 | 192 | 189 | 65 |
| 71 | 232 | 229 | 91 |
| 135 | 270 | 269 | 105 |

## Brain inputs and rest level
Brain-mask neurons (133,258) with any spike per episode: calib 314–2,428 (0.24–1.82%), shuffle 342–1,839. LPLC2/LC4 (311 neurons, the looming word) are themselves in `brain_mask`. Excluding directly driven neurons, calib 314–2,117.

Background only, active non-sensory neurons per 100 ms (bins 0.1–2.0 s):

| cfg | bg | mean | max | brain | VNC | MNs active (0–2 s) |
|---|---|---|---|---|---|---|
| calib | 1 Hz | 100.7 | 198 | 41.6 | 59.1 | 17 |
| calib | 5 Hz | 730.1 | 831 | 226.8 | 503.2 | 164 |
| shuffle | 1 Hz | 8.3 | 17 | 7.2 | 1.1 | 0 |
| shuffle | 5 Hz | 79.6 | 111 | 70.1 | 9.5 | 3 |

At 5 Hz the activity is flat from the first bin (no build-up). The most active types are VNC ascending and intrinsic neurons (AN01B002, ANXXX027, AN08B034, AN05B009, INXXX044). With background switched off at 1.5 s (`bgoff`), it drops from 665 to 73 in the next 100 ms and to 0 from 1.6 s on. Persistence flag: 2 of 10 mixture episodes have a full window more than 100 ms after their last word. Neither is flagged (calib tail means 272.5 and 550.5, against a background-only max of 831).

## Cost
Wall seconds per bio-second (CPU, 4 threads, one process): calib mixtures median 20.9 (18.9–25.2), calib background 20.0, shuffle mixtures 16.8 (13.7–19.6), shuffle background 14.4. Over all 24 runs: median 19.1, max 25.2. Model build 14–15 s per cfg.

## Checks
- Same spec and seed reproduce identical spikes: ep001 calib, 208,370 spikes (`pilot_log.txt`).
- NumPy vs torch parity (CPU float64, rng numpy, cns, vinegar 0.1 s): 2,232 spikes, identical (`results/w4/w4d1_parity_cns.txt`).

## Reading
- **H1 holds in part.** Background does not ignite a persistent state: activity follows the drive and is gone within 100 ms of switching it off. But the rest level is not low at 5 Hz. In calib, 730 non-sensory neurons are active per 100 ms and 164 MNs fire. That is 7× the 1 Hz level for 5× the drive, and about 9× the shuffled control. Background is therefore a motor driver in its own right, so the per-episode background rate must be a regression covariate, or the range narrowed.
- **H2 does not hold.** 202 of 349 pools pass the floor with the real wiring, against 25 shuffled. The real/shuffle gap is about 8× at pool level and 5× at group level, and 170× in summed variance. Fine pools are usable targets, not only the coarse groups.
- **Looming dominates.** It drives 71% of the variance, and lowering the intensity cap does not help: the GF sits at 183 Hz even at 34 Hz drive. This is the GF calibration issue already noted on day 3 (no adaptation). Without the looming episodes, 144 pools still pass, so the signal does not depend on looming.
- **H3 holds.** 19.1 s per bio-second median, 25.2 max. Background on all sensory neurons is not costly: background-only runs are no slower than mixtures.
- **The persistence check covers too few episodes.** 8 of 10 episodes have a word running past 1.8 s, which leaves no full 100 ms window after the 100 ms gap.

## Recommendation for Day 5
- **200 episodes**, real + shuffled, with the same specs and seeds.
- **Length 2.5 s, with words confined to the first 2.0 s** (onset U(0, 1.7) s, offset clipped to 2.0 s), so every episode has a 0.4 s tail for the persistence flag.
- **Intensity 30–150 Hz unchanged.** A lower looming cap would not tame the GF (see above). Instead, make looming a word-weighted draw (about 1/17 of word picks, not 1/7), or analyse the TTM/DLM pools separately. Fixing the GF saturation needs an adaptation mechanism, which is a calibration task and not a dataset choice.
- **Background:** keep 1–5 Hz and store it as a covariate. Or narrow it to 1–3 Hz if the brain → motor regression should not be dominated by background-driven VNC activity. That choice is for the Orchestrator; the pilot only measured 1 and 5 Hz.
- **Compute:** 200 × 2.5 s × (20.9 + 16.8) s per bio-second ≈ 18,850 s ≈ 5.2 h (≈ 7.0 h at the slowest pilot rate, 25.2 s per bio-second for both cfgs). 2.0 s episodes would take ≈ 4.2 h. One process, run sequentially.
