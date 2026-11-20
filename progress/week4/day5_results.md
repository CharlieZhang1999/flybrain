# Week 4 Day 5: regression training set (200 mixtures + 10 background-only, real + shuffled)

**Question:** is the Day 6 brain → motor regression set complete, free of self-sustaining states, and informative? Informative means the motor pools vary, and the shuffled control separates from the real model.

**Model:** main eb24389 (PR #8), so `W4_CALIB` includes `LOOP_GAIN`. All numbers come from `results/w4/regression/day5/summary.md` (written by `scripts/w4d5_dataset_summary.py`) and from `run_log.txt` there.

## Built
- **`scripts/w4d5_dataset.py`** (PR #8, unchanged). Run as `0 210 calib` and `0 210 shuffle`, in resumable chunks, one process at a time.
  - **Calib:** k 0–99, then k 100–209.
  - **Shuffle:** k 0–119. A container restart killed this chunk after ep110. The 111 shuffle files on disk all passed `load_valid`, and no temporary files were left. The rest ran as `0 210 shuffle`, which skipped those 111 and ran 99.
  - **Not reused:** the 40 runs made under the pre-LOOP_GAIN model. They stay in `pre_loopgain/`.
- **`scripts/w4d5_dataset_summary.py`**, extended for this run:
  - The pathology check now uses the T2c Q4 metric (`w4d5_burst_metric.score`). It works on word-free bins that start ≥ 0.1 s after the last word offset, excludes word-driven neurons, and uses θ = 200. It flags `persist` and `burst`, and counts AL events separately.
  - It attributes persist runs to wind tails, and reports per-word tails (DNge027, LOOP5, brain).
  - It splits pool variance into the share from looming episodes and the share from wind episodes (H2).
  - It checks the T2f stop conditions. `check` mode runs them between chunks.
  - New outputs: `word_tails.csv`, and Q4 columns in `episode_metrics.csv`.
- **Doc fix:** `day5_loop_gain.md`, wind alone (old model): DNge027 last at offset + 0.59–1.00 s, was 0.60 (`analyze_windalone.txt`: 0.595, 0.678, 0.999).

## Runs and cost
All 420 runs are present and valid (210 calib, 210 shuffle). Specs pass every A1 check:
- looming in 28 of 200 episodes (14.0%);
- 2 / 3 / 4 words in 60 / 70 / 70 episodes;
- wind in 102 of 200 episodes;
- background rates spread evenly over 1–3 Hz (mean 2.01 Hz).

| cfg | kind | n | median s per bio-s | min | max | wall h |
|---|---|---|---|---|---|---|
| calib | mix | 200 | 23.2 | 18.3 | 29.0 | 3.22 |
| calib | bg-only | 10 | 20.4 | 19.6 | 22.2 | 0.14 |
| shuffle | mix | 200 | 17.0 | 13.6 | 27.6 | 2.46 |
| shuffle | bg-only | 10 | 14.8 | 14.1 | 15.5 | 0.10 |

- **All 420 runs:** median 20.8 s per bio-second, max 29.0. Total simulation wall time was 5.92 h for 1,050 bio-s.
- **Reproducibility:** re-running ep023 calib (a wind episode) gave 144,538 spikes, identical to the saved run.
- **Day 1 parity:** `w4d1_torch_parity.py parity cns 0.1` gives identical=True. `pytest tests/`: 30 passed. Nothing under `sim/` changed.

## Motor-pool variance (10 ms bins, 50,000 bins per cfg)

| level | calib | shuffle | calib, no looming | shuffle, no looming | calib bg-only | shuffle bg-only |
|---|---|---|---|---|---|---|
| pools > 1 Hz² (of 349) | 205 | 30 | 144 | 24 | 21 | 1 |
| pools > 10 Hz² | 162 | 11 | 99 | 9 | 8 | 0 |
| group × side > 1 Hz² (of 42) | 34 | 6 | 29 | 3 | 7 | 0 |

- **Summed pool variance:** calib 29,336 Hz² and shuffle 372 Hz². Background-only episodes give 298 Hz² (calib) and 5.8 Hz² (shuffle).
- **Shares** of the summed variance:

| cfg | pools | looming episodes | wind episodes | both | neither |
|---|---|---|---|---|---|
| calib | all | **55%** | **58%** | 24% | 11% |
| calib | TTMn + DLMn | **88%** | 45% | 37% | 4% |
| shuffle | all | 27% | 54% | 13% | 32% |

  These episodes hold 14% (looming), 51% (wind) and 6% (both) of the bins.
- **Calib top pools:** TTMn L/R (91% / 97% looming), then b2 and hg1 steering MNs (72–82% wind).

## Rest level, pathology and tails
- **Background only, calib.**
  - **MN spikes:** 0.0 per s at 1 Hz and 14.0 per s at 3 Hz (35 spikes, 19 MNs).
  - **Active non-sensory neurons per 100 ms:** 32 at 1 Hz and 175 at 3 Hz.
  - **No drift:** late/early is 0.81–1.18. Shuffle is 0–0.8 MN spikes per s.
- **Q4 metric:**

| cfg | kind | n | persist | burst | burst bins AL / other | AL events, eligible | long non-wind persist | MN all-or-none |
|---|---|---|---|---|---|---|---|---|
| calib | mix | 200 | 8 | 44 | 43 / 12 | 52 | 0 | 0 |
| calib | bg-only | 10 | 0 | 4 | 10 / 1 | 11 | 0 | 0 |
| shuffle | mix | 200 | 0 | 0 | 0 / 0 | 0 | 0 | 0 |
| shuffle | bg-only | 10 | 0 | 0 | 0 / 0 | 0 | 0 | 0 |

  - **Calib persist runs:** all 8 last 0.2–0.3 s.
    - 5 are wind tails: ep011, 023, 109, 155 and 172.
    - 3 hold an AL event: ep086, 093 and 145.
  - **Calib bg-only bursts:** the 4 episodes are 2.11–2.78 Hz (ep205–208), with 2–3 burst bins each.
  - **Old metric** (> 300 in any bin, words included): 182 of 200 calib mixtures. This is reference only.
- **Wind tails, calib** (102 wind words). The tail is the last DNge027 spike after offset.
  - **Distribution:** median 0.058 s; 8 above 0.2 s, 2 above 0.3 s.
  - **ep023:** 0.314 s, the same value as in T2e.
  - **ep155:** 0.508 s (wind 0.24–1.21 s). Mold is on until 1.35 s, inside that tail.
  - **Shuffle:** DNge027 never fires.
- **Tails after each word type** (counts where no other word is on within 0.3 s of the offset):
  - **LOOP5 tail > 0.2 s:** only after wind, 2 of 21.
  - **Brain tail > 0.2 s:** wind 7 of 21, pheromone 1 of 6, every other word 0.
  - **Without the confound filter**, many non-wind words show LOOP5 tails, nearly all with a wind word on nearby.
- **Stop conditions:** none fired.
  - (a) No background-only episode is flagged persist, in either cfg.
  - (b) No mixture has a non-wind-tail persist run longer than 0.4 s (0 of 200 per cfg).
  - (c) The maximum cost is 29.0 s per bio-second.

## Reading
- **H1 holds.**
  - **MN output at rest:** at 3 Hz, background-only calib gives 14.0 MN spikes per s (≤ about 15). At 1 Hz it gives 0.
  - **Background variance is small:** 298 Hz² of pool variance, against 29,336 Hz² in mixtures (1%). Pool variance therefore comes from the words.
  - **But the rest level is not inert:** 21 pools pass the 1 Hz² floor with background alone in calib.
- **H2 holds.** Looming episodes hold 14% of the bins but 55% of the summed pool variance, and 88% of the TTMn + DLMn variance.
  - **Wind is the other main driver:** wind episodes hold 58% of the variance in 51% of the bins, mainly in the steering MNs (b2, hg1, hg3).
  - **Without looming episodes, 144 pools still pass** the floor, against 24 shuffled.
- **H3 holds.** The median is 20.8 s per bio-second, the max 29.0, and the total 5.9 h.
- **Real vs shuffled separates:**
  - 205 vs 30 pools pass the floor;
  - 79× in summed variance;
  - 1,382 vs 546 brain neurons active per episode (median).
- **LOOP_GAIN residue:**
  - **ep155:** the wind tail reaches 0.51 s, above the T2e bound of ≤ 0.31 s measured over 30 specs. A mold word overlaps that tail.
  - **Persistence flag (rate-matched rule):** it flags 1 episode, ep033 calib (dry+bitter+wind, wind until 2.0 s).
  - **No persist run lasts more than 0.3 s.**
- **AL bursts remain:** 52 eligible AL events across 42 calib mixtures, and 11 across 4 bg-only episodes (2.1–2.8 Hz). They are accepted, not fixed. In the Day 6 regression they will show up as brain activity with no motor consequence.

## Assumptions and what is unverified
- **Operational choices (not given in the criteria):**
  - **"Persist lasting > 0.4 s":** a run of ≥ 5 bins.
  - **"Not a wind tail":** a run is a wind tail if a wind word ended at or before the run start, and either the run starts ≤ 0.2 s after that offset or the word's LOOP5 tail reaches the run start.
  - **Stop condition (b):** evaluated per cfg.
  - **LOOP5 chain gap:** 50 ms.
  - **Brain tail:** taken from the first bin that starts at or after the offset.
  - **Confound window:** ±0.3 s.
  - **MN all-or-none burst:** > 50% of the 708 MNs in one 10 ms bin.
- **"Background-only runs are spike-identical" (PR #8) holds for 9 of 10 calib bg-only runs at 2.5 s.**
  - **The exception, ep206 (2.33 Hz):** it diverges at 2.33 s (82,454 vs 82,447 spikes). A spontaneous DNg62 spike at 2.322 s recruits DNge044 and DNge027 under the old gain, but not under ×0.3.
  - **Shuffle:** 10 of 10 bg-only runs are identical to the old ones.
  - **Non-wind mixtures:** ep000, ep001 and ep019 calib are identical to the old ones.
- **Only one episode was re-run** for reproducibility (ep023 calib). A2 asks for one.
- **Tail termination is stochastic,** and there is 1 seed per spec. The tail statistics describe this dataset, not the model's distribution.
- **No search for latent loops** other than LOOP5 was made. Any such loop would only show as Q4 flags, and none lasted more than 0.3 s.
