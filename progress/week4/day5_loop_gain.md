# Week 4 Day 5 (T2e): LOOP_GAIN, the wind-loop fix (option A)

**Decision:** Charlie chose option A (`progress/week4/day5_wind_loop_rootcause.md`): scale the synapses among the five LOOP5 types by 0.3. **H1 as written did not hold:**
- 8 of 30 wind specs keep a post-wind tail above 0.15 s;
- 1 of 30 (ep023, 0.314 s) is above the 0.3 s stop threshold.

The Orchestrator accepted this residual tail without further tuning. The ep023 tail decays within 0.4 s, and before the fix 9 of 12 baseline specs had tails above 0.3 s.

## Built

**`sim/cns.py`**
- `LOOP_GAIN = ({'DNge027', 'GNG423', 'DNg21', 'DNg62', 'DNge044'}, 0.3)` is now part of `W4_CALIB`.
- New `cns_model(..., loop_gain=None)`. It scales every W entry whose pre- **and** postsynaptic neurons both belong to the set, cross-type and same-type alike, using the helper `scale_set_edges`.
  - It runs right after `clique_gain` and before the shuffle, the same convention as CLIQUE_GAIN.
  - With `loop_gain=None`, the W is the old one.
- **The shuffle control changes too.** The 101 scaled values are permuted into the shuffle W, which therefore differs from the old one in 99 entries, none at LOOP5 positions (`wcheck.txt`).
- **The pyarrow import is now lazy.** `sim.cns` now imports `sim.malecns.load` (pyarrow) inside `cns_model`, so CI can import `sim.cns` without pyarrow. Nothing imported `load` from `sim.cns`.

**Tests:** `tests/test_loop_gain.py` is synthetic, with no data and no pyarrow. It checks that:
- in-set entries, cross-type included, are scaled by exactly g;
- entries with one endpoint in the set, and all other entries, are bitwise unchanged;
- nnz is kept, g = 1 gives the identity, and a missing type raises;
- the constant is in `W4_CALIB`.

**Scripts:**
- `scripts/w4d5_loopgain_wcheck.py`: static W check;
- `scripts/w4d5_loopgain_runs.py`: modes wind / mix / bg / loom / windalone;
- `scripts/w4d5_loopgain_analyze.py`: analysis.

**Outputs:** `results/w4/regression/loop_gain/`
- `wcheck.txt`, `runs/`, `runs_log.txt`
- `analyze_{wind,mix,bg,loom,windalone}.txt`, `analyze_wind.csv`
- `ep052_lastbin.txt`

**Simulation used:** 48 runs, one at a time, CPU float32, `rng='torch'`, 15–59 s wall each (`runs_log.txt`).

**Day 5 data:**
- These were made under the old W4_CALIB and moved, not deleted, to `results/w4/regression/day5/pre_loopgain/`:
  - the 20 calib mixtures (ep000–019);
  - the 10 calib bg-only runs (ep200–209);
  - the 10 shuffle bg-only runs (ep200–209), moved because the shuffle W changed.
- The old index is kept as `pre_loopgain/index_pre_loopgain.csv`.
- `index.csv` was rebuilt and now has 0 rows; the full generation is a separate step.

## W check (`wcheck.txt`)

| comparison | result |
|---|---|
| old W4_CALIB (git HEAD `sim/cns.py`) vs new `loop_gain=None` | same pattern, 0 entries differ |
| old vs new W4_CALIB, calib | 101 entries differ, all with pre and post in LOOP5 (12 same-type, 89 cross-type); LOOP5 block sum ratio 0.300000 |
| new W4_CALIB vs T2c `loop_gain_W(0.3)` (`w4d5_burst_runs.py`) | 0 entries differ (T2c scaled 101) |
| old vs new W4_CALIB, shuffle | 99 entries differ, 0 at LOOP5 positions |
| gap junctions old vs new | identical (calib and shuffle) |

## Verification table

**Wind specs (`analyze_wind.txt` / `.csv`).**
- **Settings:** calib, new W4_CALIB, exact Day 5 specs and Poisson seeds, class_v2 background, T 2.5 s.
- **Which specs:** the 12 wind specs with k < 20, which have saved Day 5 baselines (now in `pre_loopgain/`), and the first 18 wind specs with k ≥ 20.
- **Tail:** the last DNge027 spike minus wind offset, as in T2c/T2d. It is checked against DNge027 spikes per 100 ms bin from the offset; the first 6 bins are shown. A negative tail means DNge027 stopped before the offset.
- **"Brain":** brain_mask neurons, word-driven neurons excluded.
- **Spike identity with T2c:** ep002, ep009 and ep016 are spike-identical to the T2c g = 0.3 scan runs.

| ep | words | wind (s) | tail (s) | DNge027 /100 ms from offset | DNge027 Hz in word | MN spikes in word | brain active in word | brain 2.4–2.5 | Day 5 tail (s) | Day 5 DNge027 Hz | MN vs Day 5 | brain-in-word vs Day 5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 002 | wind+water+sound+vinegar | 0.75-1.39 | 0.050 | 5 0 0 0 0 0 … | 30.1 | 1280 | 1252 | 27 | 1.051 | 126 | -13.6% | -7.7% |
| 003 | wind+salt+dry+sound | 1.04-1.70 | 0.013 | 2 0 0 0 0 0 … | 32.6 | 1252 | 1843 | 22 | 0.479 | 126.5 | -12.3% | +17.9% |
| 005 | sound+wind+water | 1.05-2.00 | 0.201 | 9 8 1 0 0 | 30.6 | 1994 | 1296 | 33 | 0.500 | 126.3 | -11.9% | -12.1% |
| 007 | sound+wind+pheromone | 1.06-1.97 | 0.112 | 9 2 0 0 0 0 | 37.1 | 1614 | 1531 | 75 | 0.025 | 129.2 | -19.1% | -16.7% |
| 008 | humid+looming+sound+wind | 1.65-1.96 | 0.065 | 5 0 0 0 0 0 | 33.9 | 504 | 1074 | 31 | 0.218 | 114.7 | -16.0% | -10.8% |
| 009 | wind+humid | 0.86-1.77 | 0.008 | 2 0 0 0 0 0 … | 36.1 | 1561 | 1588 | 92 | 0.727 | 128.6 | -19.3% | -5.1% |
| 010 | fruit+wind+cold | 1.54-1.79 | -0.016 | 0 0 0 0 0 0 … | 28.0 | 323 | 1365 | 62 | 0.707 | 124.2 | -26.6% | -24.2% |
| 011 | dry+co2+wind | 1.66-2.00 | 0.244 | 8 5 5 0 0 | 29.7 | 384 | 769 | 51 | 0.494 | 114.2 | -23.8% | -18.2% |
| 013 | wind+umami+vinegar+cold | 0.38-0.81 | 0.060 | 4 0 0 0 0 0 … | 35.0 | 890 | 1220 | 32 | 0.135 | 134.3 | -14.3% | -8.0% |
| 015 | wind+sound+dry | 0.74-1.71 | 0.001 | 1 0 0 0 0 0 … | 33.7 | 2287 | 1705 | 27 | 0.688 | 128.5 | -10.0% | -8.3% |
| 016 | wind+hot+co2+salt | 1.21-1.96 | 0.093 | 9 0 0 0 0 0 | 31.5 | 1339 | 1317 | 68 | 0.538 | 125.1 | -14.6% | -10.2% |
| 017 | wind+vinegar+salt | 0.21-0.63 | 0.176 | 9 6 0 0 0 0 … | 28.6 | 353 | 889 | 12 | 0.540 | 103.7 | -29.3% | -28.0% |
| 022 | wind+cold | 0.18-1.03 | 0.004 | 1 0 0 0 0 0 … | 35.4 | 1595 | 1279 | 3 | – | – | – | – |
| 023 | wind+co2+hot | 0.05-0.61 | **0.314** | 7 8 8 2 0 0 … | 38.0 | 1035 | 1278 | 59 | – | – | – | – |
| 024 | wind+sugar+sound | 0.42-0.74 | 0.279 | 9 6 8 0 0 0 … | 40.2 | 329 | 792 | 75 | – | – | – | – |
| 026 | wind+dry | 0.60-1.11 | 0.081 | 7 0 0 0 0 0 … | 35.5 | 1033 | 1292 | 87 | – | – | – | – |
| 027 | wind+mold+cold | 0.66-1.66 | 0.180 | 6 7 0 0 0 0 … | 38.7 | 1135 | 1258 | 21 | – | – | – | – |
| 032 | sugar+wind+dry | 1.15-1.45 | 0.005 | 2 0 0 0 0 0 … | 33.2 | 637 | 1260 | 93 | – | – | – | – |
| 033 | dry+bitter+wind | 1.56-2.00 | 0.171 | 8 7 0 0 0 | 5.7 | 203 | 716 | 21 | – | – | – | – |
| 034 | wind+mold+hot+dry | 0.90-1.42 | 0.058 | 4 0 0 0 0 0 … | 37.0 | 693 | 1002 | 132 | – | – | – | – |
| 038 | co2+wind | 1.54-1.78 | 0.080 | 4 0 0 0 0 0 … | 33.1 | 448 | 1067 | 64 | – | – | – | – |
| 040 | mold+wind+salt+sound | 0.55-0.90 | -0.001 | 0 0 0 0 0 0 … | 36.3 | 705 | 1384 | 54 | – | – | – | – |
| 043 | wind+sound+water+cold | 0.20-0.55 | -0.004 | 0 0 0 0 0 0 … | 38.1 | 842 | 1224 | 8 | – | – | – | – |
| 045 | wind+pheromone+sound+looming | 0.15-1.07 | 0.171 | 9 5 0 0 0 0 … | 35.2 | 2099 | 1655 | 10 | – | – | – | – |
| 046 | wind+hot+co2 | 0.52-1.43 | 0.097 | 8 0 0 0 0 0 … | 37.5 | 1405 | 1209 | 63 | – | – | – | – |
| 049 | dry+sound+wind+sugar | 1.37-2.00 | 0.013 | 2 0 0 0 0 | 34.3 | 1421 | 1365 | 37 | – | – | – | – |
| 052 | sound+cold+wind | 1.48-1.70 | 0.081 | 7 0 0 0 0 0 … | 35.2 | 394 | 1052 | 389 | – | – | – | – |
| 055 | wind+hot+looming+sugar | 0.47-1.26 | 0.099 | 7 0 0 0 0 0 … | 36.6 | 1943 | 1302 | 27 | – | – | – | – |
| 060 | wind+humid | 0.10-0.46 | 0.013 | 2 0 0 0 0 0 … | 38.1 | 738 | 1173 | 53 | – | – | – | – |
| 061 | humid+wind | 0.95-1.20 | 0.008 | 2 0 0 0 0 0 … | 38.4 | 370 | 946 | 61 | – | – | – | – |

**Summary (`analyze_wind.txt`):**
- **Tail, all 30 specs:**
  - max 0.314 s (ep023);
  - 8 of 30 above 0.15 s, 1 of 30 above 0.3 s;
  - the longest run of consecutive non-empty DNge027 bins is 0.4 s (ep023).
- **The 12 baselines:**
  - the saved Day 5 tail was above 0.3 s in 9 of 12;
  - wind-word MN spikes change by −10.0 to −29.3% (median −15.3%);
  - brain neurons active in the word change by −28.0 to +17.9%.

**ep052, last bin 389 (`ep052_lastbin.txt`):** this is a spontaneous **AL burst**, not the wind loop.
- An AL event (≥ 100 ALPN/ALLN neurons in 10 ms) starts at 2.392 s with 233 neurons.
- AL neurons active per bin jump from 38 to 235 in 2.4–2.5 s. Of the 389, 134 are ALPN and 101 ALLN; most spikes fall in 2.400–2.405 s.
- DNge027 fires 0 spikes after 1.8 s.

**Other checks:**

| check | settings | result | source |
|---|---|---|---|
| non-wind Day 5 ep000 (looming+dry), ep001 (vinegar+hot+sound), ep019 (sound+bitter+looming+humid) | exact Day 5 specs | **spike-identical** to the saved Day 5 runs (170,706 / 100,645 / 185,681 spikes); DNge027 0 spikes | `analyze_mix.txt` |
| bg-only class_v2 3 Hz | seeds 0–2, 1.0 s | **spike-identical** to `results/w4/bg_fix/runs/calib_v2_3Hz_s*` (42,564 / 42,171 / 41,897); metrics equal the verify.csv class_v2 rows (e.g. MN active 10 / 8 / 8) | `analyze_bg.txt` |
| looming alone | 0–0.4 s at 150 Hz, no bg, T 1.0 s, seeds 0–2, old vs new | **spike-identical**; CNS silent from 0.5 s (last spike 0.4235–0.4278 s) | `analyze_loom.txt` |
| wind alone | 0–0.5 s at 150 Hz, no bg, T 1.5 s, seeds 0–2, old vs new | old: about 400–460 brain neurons per bin persist to 1.12 s, 1.20 s and the end (1.5 s), with DNge027 last at offset + 0.59–1.00 s. **New:** silent from 0.6–0.7 s (last spike 0.555 / 0.597 / 0.606 s; DNge027 last at offset + 0.012–0.084 s) | `analyze_windalone.txt` |
| Day 1 parity | `w4d1_torch_parity.py parity cns 0.1` | identical=True. It uses the `cns_model` defaults without W4_CALIB, so it is unaffected by construction | terminal |
| pytest | `pytest tests/` | 30 passed | terminal |

## Reading

- **H1** (tail ≤ 0.15 s in all wind specs): **does not hold.**
  - 22 of 30 specs are at ≤ 0.15 s, 8 are at 0.171–0.314 s, and 1 is above 0.3 s (ep023, 0.314 s).
  - These residual tails decay within 0.2–0.4 s, at about 2–9 DNge027 spikes per 100 ms over 2 cells. The old state ran at about 20–30 spikes per 100 ms until it stopped or the run ended.
  - **The Orchestrator accepted the residual tail** as Charlie's option A, without further tuning.
- **H2** (non-wind spike-identical): **holds**, 3 of 3.
- **H3** (bg-only and looming unchanged): **holds, as tested.** Both are spike-identical. This was tested for bg-only at class_v2 3 Hz, 1.0 s, seeds 0–2 only.
  - **Not identical in the full Day 5 run (T2f):** calib bg-only ep206 (2.33 Hz, 2.5 s) diverges from 2.33 s, because a spontaneous DNg62 spike at 2.322 s no longer recruits DNge044 and DNge027.
  - **Still identical:** the other 9 calib and all 10 shuffle bg-only runs (`day5_results.md`).
  - Wind alone, which was never in the Week 3b word check, now returns to silence within about 0.1 s of offset.
- **Cost:**
  - wind-word MN output falls by 10–29% (median 15%);
  - DNge027 in the word runs at 28–40 Hz instead of 104–134 Hz. In ep033 it is only 5.7 Hz, so the wind loop barely ignites there;
  - brain recruitment in the wind word mostly falls (to −28%), but ep003 rises by 17.9% (not examined).
- **Limits:**
  - 1 seed per spec, and termination of the state is stochastic;
  - the tail statistic depends on the episode;
  - the k ≥ 20 specs have no baseline;
  - other latent loops were not searched for.

## Assumptions
- **Spec selection** (set by the Orchestrator): the 12 wind specs with k < 20, plus the first 18 with k ≥ 20.
- **Wind-alone and looming:** `rng='torch'` float32, as in the Day 5 data; `w4d3b_word_check.py` used numpy float64. No background, compared old vs new in one script.
- **Shuffle:** follows the clique_gain convention (gain before the shuffle). A shuffle that uses an unscaled W would need a different choice.
