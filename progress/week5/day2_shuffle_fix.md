# Week 5 Day 2: shuffle-control fix (JO mirror before the shuffle) and the re-run shuffle controls

**Question:** the shuffled control had about twice as many connections as the real network (T5a, PR #13). After fixing the order of the JO mirror and the shuffle, do the Week 4 and Week 5 shuffle results change? Brief: T5b (Orchestrator). Code: PR #14 (branch `claude/project-thread-jkduez`, from main f1356ae).

## The bug
`cns_model(config='shuffle')` ran shuffle → `build_presyn` → `mirror_jo`. `mirror_jo` gives each right-JO neuron the mean output of its left-JO subtype, mapped onto contralateral homologs. After the shuffle, the left-JO columns point at random targets, so that mean is spread over most cell types: the right JO got about 22M weak extra outputs. This affected every `cns_model` shuffle control from Week 4 day 4 on (Day 5 dataset, ridge baseline, Week 5 behaviour panel). It mostly touches wind and sound, the JO words.

## Built
- **`sim/cns.py`**: new `presyn_mirror_shuffle` (the last three pipeline steps) and the kwarg `cns_model(..., shuffle_order='fixed' | 'legacy')`. The `SHUFFLE_ORDER` comment cites the bug and the numbers.
  - **fixed (default):** mirror the real W first, then shuffle. The mirror runs on `build_presyn(W)[0]`, exactly the matrix calib mirrors. The original sensory rows are put back, the full matrix is shuffled, and `build_presyn` then runs on the shuffled matrix as before.
  - **Why `build_presyn` stays after the shuffle:** this keeps its semantics unchanged. The presynaptic matrix `ps` still comes from the shuffled W's central → sensory rows, as in every earlier shuffle. `shuffle_W` keeps each entry's row, so each sensory row keeps its entry count (481,672 sensory-row entries in all).
  - **legacy:** shuffle → presyn → mirror, the pre-fix order.
  - **calib:** `build_presyn` → `mirror_jo`, unchanged.
- **`tests/test_shuffle_order.py`** (CI, synthetic, no data or pyarrow). On a toy network with 40 left and 40 right JO afferents, the legacy order inflates nnz (13,910 vs 5,606 calib), while the fixed order stays within 5% (5,442). The test also checks:
  - the calib path equals `build_presyn` then `mirror_jo`;
  - an identity "shuffle" in the fixed order gives calib exactly;
  - the mirror leaves no entries in sensory rows.
- **Checks** (`results/w5/shuffle_fix/fix_check.log`; scripts `dump_model.py` and `cmp.py` are stored next to it):
  - **Calib is bit-identical** to f1356ae: W, gap, presyn and apl, 17 arrays, dtype + shape + values.
  - **`shuffle_order='legacy'` is bit-identical** to the old shuffle (nnz 46,833,061).
  - The fixed shuffle's input, restricted to non-sensory rows, equals the calib W exactly. The mirrored right-JO output sum is the same as in calib (23.94 V; legacy 25.13).
  - `pytest -q tests/`: 48 passed. `w4d1_torch_parity.py parity cns 0.1`: identical=True.

| network | nnz (W after presyn + mirror) |
|---|---|
| calib | 24,519,583 |
| shuffle seed 12345, legacy | 46,833,061 |
| shuffle seed 12345, fixed | **24,444,089** (−0.31% vs calib) |
| panel shuffles 1–10, legacy | 46,674,664 – 47,364,598 |
| panel shuffles 1–10, fixed | 24,443,647 – 24,444,466 |

The remaining −0.31% is the duplicate (row, column) pairs that `shuffle_W` merges (about 75k), as before.

## Re-runs
Old outputs were moved, not deleted:
- `results/w5/panel/pre_shufflefix/` (calib files copied there too);
- `results/w4/regression/day5/pre_shufflefix/` (210 shuffle npz + index/summary/variance CSVs);
- `results/w4/regression/ridge/pre_shufflefix/`.

Calib was not re-simulated, because it is unchanged (bit-identical model).

### 1. Behaviour panel (10 shuffles × 13 conditions × 3 seeds)
`results/w5/panel/` (`run.log`, `verify.log`, `shuffle_verify.json`, `scorecard.csv`, `summary.json`, `score.log`, `shufflefix_scorecard_compare.csv`, `shufflefix_trials_compare.csv`). `verify`: seed 12345 through the panel route reproduces `cns_model`'s shuffle (nnz 24,444,089), and the gap junctions are not shuffled.

| pair | real | shuffles, old (AUROC range / passes) | shuffles, new | p, old → new |
|---|---|---|---|---|
| P1 sugar → MN9 | 0.728 | 0.497–0.500 | 0.497–0.500 | 0.091 → 0.091 |
| P2 water → MN9 | 0.500 | 0.499–0.500 | 0.499–0.500 | 0.818 → 0.818 |
| P5 looming → TTMn | 1.000 | 0.479–0.486 | **0.471–0.729** | 0.091 → 0.091 |
| P7 MDN → hl | 0.574 | 0.499–0.516 | 0.495–0.508 | 0.091 → 0.091 |
| P8 DNg100 → legs | 0.576 | 0.495–0.509 | 0.497–0.512 | 0.091 → 0.091 |
| X1 wind → fl (not scored) | 0.531 | 0.492–0.520 | 0.489–0.550 | – |
| P3 bitter ¬MN9 | pass | 10/10 | 10/10 | – |
| P4 sugar+bitter < sugar | pass | 0/10 | 0/10 | – |
| P6 GF off < looming | pass | 0/10 | **1/10** (shuf7) | – |
| P9 dose monotonic | fail | 0/10 | 0/10 | – |
| **hits** | **4/9** | **1/9 in each shuffle** | **1/9 in 9 shuffles, 2/9 in shuf7** | |

- **shuf7:** in 2 of 3 seeds, looming makes the GF fire a single spike (mean GF rate 1.25 Hz in 0–0.4 s). TTMn fires a single spike too (1.25 Hz; mean over seeds 0.83 Hz), so P5 AUROC is 0.729 (best rank 11). With the GF silenced, TTMn falls to 0, so P6 "passes" (0 < 0.83). No other shuffle drives the GF. This is a weak chance path, not the real looming → GF response (real TTMn 277 Hz).
- **Wind (X1):** the shuffled networks drive more motor neurons than before (mean active MNs 3.7 → 7.0; real 103). Other conditions change by ≤ 1 MN on average. No late activity (0.45–0.5 s) in any shuffle, before or after.
- The real network's numbers did not change, since it was not re-run. All real-vs-shuffle p-values are unchanged.

### 2. Day 5 dataset, shuffle (200 mixtures + 10 background-only)
`results/w4/regression/day5/`: `summary.md`, `pool_variance.csv`, `index.csv` (420 rows: 210 calib unchanged, 210 shuffle new), `run_log.txt`, `shufflefix_run_chunk{1,2}.log`, `shufflefix_spikes_compare.csv`, `shufflefix_ttmn.txt`. Same specs and seeds, run as `0 210 shuffle`. A 2 h job limit stopped chunk 1 after ep132; it resumed and skipped the 133 valid files. `check 0 shuffle` re-runs ep000: identical=True. The calib rows of `index.csv` are identical to before (spike counts, 210/210). Stop conditions: none hit. Persist and burst flags are 0 in every shuffle episode, before and after.

| shuffle, Day 5 | old | new | calib (unchanged) |
|---|---|---|---|
| pools > 1 Hz² (of 349) | 30 | **28** | 205 |
| same, without looming episodes | 24 | 24 | 144 |
| same, background-only episodes | 1 | 5 | 21 |
| group × side > 1 Hz² (of 42) | 6 | 4 | 34 |
| summed pool variance (Hz²) | 372 | 534 | 29,336 |
| summed pool variance, background-only (Hz²) | 5.8 | 305.4 | 298.0 |
| share of variance in wind / looming episodes | 54% / 27% | 60% / 28% | 58% / 55% |
| wind words where DNge027 fires (of 102) | 0 | 10 | – |

- **Total spikes per episode barely change.** Wind or sound episodes change by −2.5% to +1.5% (median +0.01%). Other mixtures change by −0.4% to +0.6%, background-only episodes by under 0.1%.
- **But the pools that vary are a different set.** Only 1 of the 28 new floor pools was among the 30 old ones. The cause is not the JO alone. The fixed shuffle permutes a different entry list: 25,001,255 entries, the mirrored network, against 24,891,947 before. So with the same seed it is a new random draw of the whole network, not the old one with a corrected JO.
- **The left TTMn fires tonically in the new shuffle** at about 2 Hz, in mixtures (1.99 Hz) and in background-only episodes (2.16 Hz) alike. Before the fix it was silent. It is the top variance pool (196 Hz²), and it explains most of the background-only variance jump (5.8 → 305 Hz²).

### 3. Ridge baseline, shuffle
`results/w4/regression/ridge/` (`summary.md`, `r2_per_pool.csv`, `log.txt`, `shufflefix_compare.txt`, `shufflefix_run_stdout.log`). Same settings and seed. Both cfgs were re-fitted. **The calib rows are unchanged:** all 8 calib R² columns and all 4 `pred_calib_*.npz` files are identical to before.

| ridge, all-brain inputs, fine pools | old | new | calib (unchanged) |
|---|---|---|---|
| shuffle floor pools, median R² (all episodes) | 30, 0.017 | **28, 0.024** | 205, 0.138 |
| shuffle floor pools, median R² (no looming) | 24, 0.009 | 24, 0.014 | 144, 0.065 |
| shuffle pools with R² > 0.1 (all / no looming) | 5 / 4 | 3 / 2 | 120 / 45 |
| DN-only inputs, shuffle median R² | 0.013 | 0.014 | 0.127 |
| paired, calib floor pools with shuffle R² defined (all episodes) | 39 pools; 0.119 vs 0.003; calib higher in 38 | **49 pools; 0.110 vs 0.002; calib higher in 41 (shuffle ≥ calib in 8)** | |
| paired, no-looming episodes | 17 pools; 0.110 vs 0.005; calib higher in 16 | 30 pools; 0.067 vs −0.000; calib higher in 23 (shuffle ≥ calib in 7) | |
| pools above the floor in both cfgs, median R² calib / shuffle | 23; 0.113 / 0.010 | 22; 0.091 / 0.020 | |
| time-shuffle / mean controls, shuffle | −0.000 | −0.000 (time 0.000) | |

**Flagged (H3 trigger):** the number of paired pools where the shuffle matches or beats calib rose from 1 to 8 (all episodes) and from 1 to 7 (no looming). That is more than 5. Details:
- All 8 are pools that calib predicts poorly: calib R² ≤ 0.09, mostly abdominal MNad pools. Their shuffle R² is 0.003–0.111.
- Only 5 pools are paired in both the old and the new run. Of those, 1 changed winner (L|ad|MNad63). The others are newly paired, because the new shuffle drives different pools.
- The median gap on the paired pools is unchanged (0.110 vs 0.002).

## Hypotheses
- **H1 (shuffle nnz ≈ calib, calib bit-identical): holds.**
  - Calib W, gap, presyn and apl are bit-identical.
  - Fixed shuffle nnz is −0.31% vs calib (all 11 seeds within −0.31% ± 0.01%).
  - Legacy reproduces the old W exactly.
- **H2 (panel conclusion unchanged; shuffle hits stay 1/9): partly holds.**
  - The conclusion is unchanged: real 4/9, all real-vs-shuffle p-values the same, no shuffle reaches AUROC 0.9.
  - 9 of 10 shuffles stay at 1/9. **shuf7 scores 2/9:** P6 passes on a single-spike GF → TTMn chance path, and its P5 AUROC is 0.729.
- **H3 (Day 5 changes mainly in wind/sound; pools far below 205; ridge gap survives): partly holds.**
  - **The gap survives:** 28 vs 205 variance-carrying pools, and shuffle median R² 0.024 vs 0.138 (below 0.05).
  - **The changes are not confined to wind/sound.** Total spikes change slightly more in JO episodes, but the set of varying pools is almost entirely new. The fixed shuffle is a different random realisation of the whole network, and in it the left TTMn fires tonically.
  - **The paired comparison flips for more than 5 pools** (shuffle ≥ calib in 8 of 49, was 1 of 39), all of them pools with calib R² ≤ 0.09. See the flag above.

## Reading
- The real-vs-shuffle separation of Week 4 and Week 5 does not depend on the inflated right-JO mirror: every headline gap survives the fix.
- The old and new shuffles differ as two random draws do, not only in the JO. Seed-to-seed variation of a single shuffle is therefore as large as the fix's effect on the Day 5 and ridge numbers. The panel has 10 shuffles; the Day 5 dataset and the ridge baseline still have one (seed 12345).
- One shuffle (seed 12345, fixed) has a tonically firing left TTMn. Tonic MN activity in a control inflates its pool variance without carrying stimulus information. Its ridge R² stays near 0.

## Not changed, flagged
- **`scripts/w3d1_common.presyn_model(config='shuffle', mirror=True)`**, used by the Week 3 loop (`loop/run_loop.py`), has the same order: shuffle → presyn → mirror. The README row "Full loop … 1 shuffled" therefore comes from an inflated shuffle. It was not fixed or re-run here, because the brief limits the code change to `sim/cns.py`.
- The docstring of `scripts/w5_behaviour_panel.py` still describes the shuffle as "before presynaptic modulation and the JO mirror". It is out of scope for this PR (only `sim/cns.py` may change). The seed route itself is unchanged, and `verify` confirms it.

## Figures
`results/figures/w5_behaviour_panel.png` and `results/figures/w4_ridge_r2_calib_vs_shuffle.png` were regenerated for the fixed shuffle. The old versions are in the `pre_shufflefix/` folders (panel figure: `results/w5/panel/pre_shufflefix/`; ridge: `results/w4/regression/ridge/pre_shufflefix/r2_calib_vs_shuffle.png`).

## Assumptions and unverified
- "Shuffle nnz within 1% of calib" is read as the final W (after presyn and mirror): −0.31%.
- Not checked: which neurons drive the tonic left TTMn in the seed-12345 fixed shuffle, and whether other seeds show it. The panel shuffles 1–10 have no `none`-condition activity (0 spikes), but that protocol has no background drive.
- The 2/9 for shuf7 rests on single spikes; with 3 seeds it is not robust either way.
- The Day 5 dataset and the ridge baseline use one shuffle draw. A second seed would separate the effect of the fix from draw-to-draw variation. Not done, because it is outside the brief.
