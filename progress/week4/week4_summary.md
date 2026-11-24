# Week 4 summary (20 Nov 2026)

**Goal (Phase 2A, [week4_plan.md](week4_plan.md)):** simulate the whole male CNS (brain + ventral nerve cord) with the calibrated LIF of Weeks 1–3, and replace the word classifier with a regression: from brain activity, predict VNC motor-pool rates 100 ms ahead, R² per pool, with the shuffled connectome in every table.

**Where it stands:** the full-CNS model runs, is calibrated against the known VNC circuits we could test, and has produced a 420-run regression dataset. A first ridge baseline is in: the real connectome is predictable from brain activity well above the shuffled one, but a large part of that comes from looming episodes. The decoder work proper is Week 5.

> **Shuffle-control fix (2026-11-24, [week5/day2_shuffle_fix.md](../week5/day2_shuffle_fix.md)).** The shuffle control here gave the right Johnston's organ (JO) about 22M extra weak outputs (nnz 46.8M vs 24.5M calib), because the JO mirror ran after the shuffle. It is fixed (mirror, then shuffle; PR #14). The shuffle half of the dataset and the shuffle ridge fit were re-run. Calib is unchanged. Old → new, shuffle: pools > 1 Hz² 30 → 28 (no looming 24 → 24); summed variance 372 → 534 Hz²; ridge median R² 0.017 → 0.024 (R² > 0.1: 5 → 3). Paired with calib on the calib floor pools: 39 → 49 pools, 0.119 vs 0.003 → 0.110 vs 0.002, shuffle ≥ calib in 1 → 8 pools (all with calib R² ≤ 0.09). The real-vs-shuffle gap holds. The numbers below are the pre-fix ones.

Every number below comes from the day report linked next to it, or from the results file it names.

## 1. What was built

| item | what | report |
|---|---|---|
| Full-CNS model | `sim/cns.py`: `cns_model('cns', cfg)`, 165,122 neurons, nnz 24,519,583 (brain-only 20,511,419); `motor_pools` → 349 pools; `brain_mask` → 133,258 regression inputs. | [day1_results.md](day1_results.md) |
| PyTorch simulator | `sim/lif_torch.py`: event-driven port of the Brian2-faithful step; spike-for-spike identical to the NumPy simulator (CPU float64, 0 neurons differ in 3 checks). 14–17 s per bio-second on 4 CPU cores. | [day1_results.md](day1_results.md) |
| Save/resume | `run_trial(state=…, return_state=True)`; three chained segments = one run (61,429 / 65,014 spikes identical, `results/w4/sci/w4d2_resume_check.txt`). | [day2_science_results.md](day2_science_results.md) |
| VNC calibration, GF gap | Rectifying GF → TTMn / PSI gap junctions, 8 mV: TTMn 120.6 Hz with 88% one-to-one following (chemical synapse alone: 7%, one side). | [day3_results.md](day3_results.md) (b) |
| VNC calibration, MN glutamate | MN outputs set glutamatergic (−1); their per-body prediction was 34% unclear and 33% ACh. | [day3_results.md](day3_results.md) (c) |
| Sign fix | GLNO and LPsP set inhibitory (`SIGN_FIX`); they sustained a whole-ring central-complex state after looming. | [day3b_results.md](day3b_results.md) |
| Clique gain | Within-type synapses of DNg33 and IN19B040 × 0.3 (`CLIQUE_GAIN`): looming tail 0 brain / 0 VNC neurons at 0.5–1.0 s (was about 540 / 70). | [day3b_results.md](day3b_results.md) |
| Loop gain | Synapses among the five LOOP5 GNG descending types × 0.3 (`LOOP_GAIN`), the post-wind loop fix chosen by Charlie (option A). | [day5_burst_diagnosis.md](day5_burst_diagnosis.md), [day5_wind_loop_rootcause.md](day5_wind_loop_rootcause.md), [day5_loop_gain.md](day5_loop_gain.md) |
| Class-specific background | `class_v2`: bristles 0 Hz, abdominal O2 sensors 0 Hz, other abdominal sensors a fixed 1 Hz, all other sensory neurons the episode rate (1–3 Hz). | [day5_bg_fix.md](day5_bg_fix.md) |
| Abdominal identification | Of the 1,040 `unknown_sensory | abdomen` neurons, 682 carry a MANC type; mapped through BANC, 494 are abdominal bristles and 18 are O2 sensors (`results/w4/abd_id/external_id.md`, on the shared mount). | [day5_bg_fix.md](day5_bg_fix.md) |
| Mixture episodes | `stimuli/mixtures.py`: 2–4 words of distinct modalities, random rates, onsets and durations, plus background. | [day4_results.md](day4_results.md) |
| Ridge baseline | `scripts/w4d6_ridge_baseline.py` (section 3). | `results/w4/regression/ridge/summary.md` |

The DN validation panel (`results/w4/sci/w4d3_panel_summary.md`) passes for GF, MDN and looming and partly for DNg100; DNa02 cannot be tested without ongoing walking ([day3_results.md](day3_results.md) (a)). Looming activates 175 MNs with the real wiring against 11 shuffled.

## 2. The dataset ([day5_results.md](day5_results.md), `results/w4/regression/day5/summary.md`)
- **Runs.** 200 mixture episodes + 10 background-only, 2.5 s each, for calib and for shuffle: 420 runs, all valid, 5.92 h of simulation for 1,050 bio-s (median 20.8 s per bio-second).
- **Composition.** Looming in 28 of 200 mixtures, wind in 102.
- **Motor pools that vary** (> 1 Hz² over 10 ms bins): 205 of 349 calib against 30 shuffled; 144 against 24 without looming episodes. Summed pool variance is 29,336 Hz² calib against 372 Hz² shuffled.
- **Who drives the variance.** Looming episodes hold 14% of the bins and 55% of the calib pool variance (88% for TTMn + DLMn). Wind episodes hold 58%, mainly in the b2 / hg steering MNs.
- **Known artifacts.** AL relaxation bursts (52 eligible events in calib mixtures), and post-wind loop tails up to 0.51 s (ep155). Neither tripped a stop condition.

## 3. Ridge baseline (`results/w4/regression/ridge/summary.md` and `log.txt`, on the shared mount)

> **Shuffle-control fix (2026-11-24):** the shuffle rows of this table were re-fitted on the fixed shuffle. all brain, all 200: 30 pools, 0.017 → 28 pools, 0.024, R² > 0.1 5 → 3. No looming: 24 pools, 0.009 → 24, 0.014. DNs only: 0.013 → 0.014. Paired: shuffle ≥ calib in 1/39 → 8/49 pools. Calib rows are unchanged (identical predictions). The old outputs are in `ridge/pre_shufflefix/`. The figure copy `results/figures/w4_ridge_r2_calib_vs_shuffle.png` now shows the fixed shuffle. Details: [week5/day2_shuffle_fix.md](../week5/day2_shuffle_fix.md).
**Setup.** Brain inputs (brain_mask neurons with ≥ 20 spikes in the training episodes, z-scored, 200 principal components fitted on the training episodes only) at lags 0–100 ms in 10 ms bins (11 lags, 2,200 features) → the 349 pool rates and 42 group × side rates at t + 100 ms. First 100 ms of each episode discarded. 5-fold cross-validation by episode, stratified by looming and wind; ridge α per target from an inner 4-fold CV on the training episodes. R² on held-out bins pooled over folds, reported for pools above the Day 5 floor of each cfg. Controls: time-shuffled targets (circular shift ≥ 300 ms per episode) and the training-fold mean. Reproducible: re-running fold 0 (`--check`, `log_check.txt`) gives predictions identical to the saved ones, and two full runs gave a byte-identical `r2_per_pool.csv`.

| cfg | inputs | episodes | pools above floor | median R² | IQR | R² > 0.1 | coarse: groups, median R² |
|---|---|---|---|---|---|---|---|
| calib | all brain | all 200 | 205 | **0.138** | 0.062–0.292 | 120 | 34, 0.345 |
| calib | all brain | no looming | 144 | **0.065** | 0.013–0.120 | 45 | 29, 0.156 |
| calib | DNs only | all 200 | 205 | 0.127 | 0.060–0.271 | 118 | 34, 0.332 |
| calib | DNs only | no looming | 144 | 0.055 | 0.005–0.104 | 38 | 29, 0.107 |
| calib | time shuffle | all 200 | 205 | −0.000 | −0.001 to −0.000 | 0 | 34, −0.001 |
| calib | mean only | all 200 | 205 | −0.000 | −0.001 to −0.000 | 0 | 34, −0.001 |
| shuffle | all brain | all 200 | 30 | 0.017 | 0.007–0.057 | 5 | 6, 0.031 |
| shuffle | all brain | no looming | 24 | 0.009 | 0.004–0.066 | 4 | 3, 0.064 |
| shuffle | DNs only | all 200 | 30 | 0.013 | 0.002–0.044 | 4 | 6, 0.020 |
| shuffle | time shuffle / mean only | all 200 | 30 | −0.000 | −0.000 to −0.000 | 0 | 6, −0.000 |

The full table (both levels, every control with and without looming) is in the ridge `summary.md`. The figure `results/w4/regression/ridge/r2_calib_vs_shuffle.png` (copy in [results/figures/w4_ridge_r2_calib_vs_shuffle.png](../../results/figures/w4_ridge_r2_calib_vs_shuffle.png)) plots R² per pool, calib against shuffle.

**Reading.**
- **Real beats shuffled.** Of the 205 calib floor pools, 166 never spike in the shuffled held-out target bins, so their shuffle R² is undefined. On the 39 where both are defined, the median is 0.119 calib against 0.003 shuffled, and calib is higher in 38. On the 23 pools above the floor in both cfgs, the medians are 0.113 and 0.010.
- **Looming carries much of it.** On the same 144 pools, the median falls from 0.100 (all episodes) to 0.065 (non-looming held-out episodes only). The top pools are TTMn, DLMn, DVMn and MNad28 (R² 0.50–0.65), which drop to 0.11 or below on the non-looming episodes (or are undefined there: DVMn never fires without looming). The front-leg promotor and femur reductor pools keep R² 0.51–0.60 without looming.
- **No leak through episode structure.** Time-shuffled targets give R² equal to the mean-only baseline (median −0.000, no pool above 0.1).
- **DNs carry most of it at 200 components.** DN-only inputs (1,318 brain_mask neurons whose superclass contains "descending"; 541–548 pass the spike threshold per fold) reach 94% of the all-brain summed positive R² (35.44 vs 37.52); the median per-pool ratio is 0.94 on the 120 pools with all-brain R² > 0.1. **Caveat: this depends on the PCA size.** With 500 PCs (`sens_k500.txt`, calib only), all-brain inputs give a median of 0.133 and DN-only 0.103 (77%); without looming, 0.055 and 0.040. More components did not help the all-brain fit (0.138 → 0.133), so 200 PCs is not the bottleneck there, but the DN-only share is not stable across K.
- **Hypotheses:** H1 holds (calib ≫ shuffle; on the same 144 pools, median 0.100 → 0.065 without looming). H2 holds (time-shuffle ≈ 0). H3 holds in direction: with 200 PCs DN-only reaches 94% of the all-brain summed R² (median 0.127 vs 0.138); with 500 PCs its median is 77% of the all-brain one (0.103 vs 0.133).

## 4. Caveats
- **No adaptation or synaptic depression in the model.** Strong reciprocal excitation therefore runs forever once lit. The GF fires at 183–270 Hz under looming at every intensity tried, where the fly fires a few spikes ([day4_results.md](day4_results.md)). This inflates looming's share of the motor variance and of R².
- **Metastable loops are patched by targeted gains** (CLIQUE_GAIN, LOOP_GAIN, both × 0.3), not by a mechanism. Patching loops one by one keeps uncovering the next one ([day3b_results.md](day3b_results.md)); one 11-neuron loop remains in the brain-only model. Residual post-wind tails reach 0.51 s ([day5_results.md](day5_results.md)).
- **AL relaxation bursts** remain, accepted, not fixed ([day5_burst_diagnosis.md](day5_burst_diagnosis.md)). In the regression they are brain activity with no motor consequence.
- **Background calibration rests on limited physiology.** No source supports a tonic rate for the abdominal group; class_v2 is a decision, not a measurement ([day5_bg_fix.md](day5_bg_fix.md)).
- **Single seeds.** One Poisson seed per spec. Tail and R² statistics describe this dataset, not the model's distribution.
- **Ridge baseline limits.** 200 PCs keep only about half of the z-scored input variance (50.2–50.5% per fold for all-brain inputs); the no-looming rows evaluate models trained on all episodes; the floor uses the Day 5 variance over the full episodes, not just the sampled bins. See the ridge `summary.md` and the PR report for the full list.

## 5. Week 5 plan
Superseded: Week 5 onward follows the central plan ([Refined 12-week plan](https://claude.ai/artifact/RUjnZ67sWYM1AQnTVJ2Jeh), "Current schedule"): Week 5 = hidden-state dataset plus a 2-day behaviour sanity panel.
From [week4_plan.md](week4_plan.md) ("Weeks 5–6"), with what the baseline adds:
- **Decoders.** Ridge on lagged brain bins, held-out episodes, inputs all brain / DNs only / a random 1% subset; controls shuffled connectome and time shift (done here as a baseline). Then a small nonlinear model only if ridge leaves structure on the table.
- **From the baseline:** refit without looming episodes (not only evaluate) and per word modality, to separate the looming / GF saturation effect from the rest; test the PCA bottleneck (more components or the neuron set directly); add seeds per spec.
- **Model:** GF adaptation and the DNa02-on-DNg100 retest stay open from Day 3.
- **Week 6:** benchmark table (speed, R²), Connectome Explorer VNC views and motor-pool panel, write-up.
