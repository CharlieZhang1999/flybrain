# Week 5 Day 1: behaviour sanity panel, real wiring vs 10 shuffles

**Question:** for each known stimulus → behaviour pair, does the full-CNS model (`cns_model('cns', 'calib', **W4_CALIB)`) put the expected output neurons near the top of all motor outputs, and more than shuffled wiring does? Nothing is trained or tuned. Brief: T5a (Orchestrator). Model: main c49cbb7.

All numbers come from `results/w5/panel/` (`scorecard.csv`, `summary.json`, `trials_<net>.csv`, `rates_<net>.csv`, `shuffle_verify.json`, `run.log`, `score.log`), written by `scripts/w5_behaviour_panel.py`.

## Built
- **`scripts/w5_behaviour_panel.py`** with the subcommands `verify` (shuffle checks), `run <net|all>` (resumable per network × condition) and `score` (scorecard, summary, figure). The pair table and scoring rule are copied unchanged into its docstring.
- **`tests/test_w5_panel.py`**: AUROC with ties, best rank, GF silencing of W and the gap junctions, and the direction tests, all on synthetic data.
- **Output set: 436 pools.** These are the 349 VNC pools of `motor_pools`, plus 87 brain (`cb_motor`) pools grouped by side|type and named `cb|<side>|<type>`. The 87 include one untyped right neuron, `cb|R|?`. Hit sets: MN9 = `cb|L|MN9` and `cb|R|MN9` (1 neuron each); TTMn = `L|wm|TTMn` and `R|wm|TTMn`; hl = 54 pools; legs (fl + ml + hl) = 141 pools; fl = 39 pools (X1).
- **Protocol (as in the brief):** 0.5 s trials, Poisson drive 0–0.4 s on the named neurons (`run_trial(W, [], blocks=[(0, 0.4, idx, rate)], rng='torch', float32, cpu)`), no background, seeds 0/1/2. Rates are per pool (mean over its neurons) over 0–0.4 s, minus the same network's `none` at the same seed. `none` was silent in every network and seed (0 spikes in all motor neurons). That makes Δ equal to the absolute rate.
- **Runs:** 11 networks × 13 conditions × 3 seeds = 429 trials, one process. Each network took about 21 s to build and about 4–4.5 min to simulate. The whole run took about 55 min.

### Shuffle construction (A2)
- `shufK` = `cns_model('cns', 'shuffle', **W4_CALIB)`, but with `scripts.w2d4_word_trials.shuffle_W` called with `seed=K` (K = 1..10) instead of 12345. The script swaps the module attribute only for the duration of the call, so the pipeline order is unchanged: scaling, MN-glu, sign fix, clique and loop gain, APL, LN–PN cut, **shuffle**, presynaptic modulation, JO mirror.
- **Check:** seed 12345 through this route gives `W` identical to `cns_model('cns','shuffle',**W4_CALIB)['W']` (indptr, indices and data all equal, nnz 46,833,061). `shuffle_verify.json`: `seed12345_reproduces_cns_model_shuffle: true`.
- **What the shuffle preserves:** each W entry keeps its row (postsynaptic neuron), and the pairs (presynaptic column, weight) are permuted across all 24.9M entries.
  - Before `sum_duplicates`, both in-degree and out-degree are preserved exactly (checked, seeds 1 and 2).
  - Each presynaptic neuron keeps its outgoing weight multiset, so it keeps its sign (Dale) and its total output. The max |Δ column sum| is 4e-15.
  - Each postsynaptic neuron's total input is **not** preserved (correlation 0.51 with the real total).
  - `sum_duplicates` merges about 75k collisions (0.3% of entries). After the merge, in- and out-degree differ for about 29k neurons. The changes are small in total (0.3% of in-degree), but up to 806 for single hub neurons.
- **Not shuffled:** the gap junctions. eLN–PN and GF → TTMn/PSI are identical in the real network and in shuf1 (checked). The GF → TTMn chemical synapses (real: 0.011 and 0.003 V entries, one per side) are absent in shuf1.
- **Finding (pre-existing, not changed):** the shuffled W has 46.7–47.4M nonzeros against 24.5M for real. With `mirror=False` the two are 24.3M and 24.4M. The extra entries come from `sim/jo_mirror.mirror_jo`. In the shuffled W, left-JO columns reach many cell types, and the mirror spreads each type's mean weight across all contralateral homologs. That gives the right JO dense, weak outputs in every shuffle control (Week 2 onward). It mainly affects wind and sound words (X1 here).

## Scorecard (P1–P9 + X1)
Δ = mean over seeds (Hz). Rank: 1 = highest Δ among 436 pools; rank = 1 + #pools strictly higher. p = (1 + #shuffles with AUROC ≥ real) / 11; the smallest possible value is 0.091.

| # | Condition → expected | Real | Shuffles (10) | p | Hit |
|---|---|---|---|---|---|
| P1 | sugar → MN9 | AUROC **0.728**, best rank 6 (MN9 L 10.8 Hz, R 0) | AUROC 0.497–0.500, MN9 Δ = 0 in all | 0.091 | **no** (AUROC < 0.9) |
| P2 | water → MN9 | AUROC 0.500. Water drives **no** output pool (0 of 436) | 0.499–0.500 | 0.818 | no |
| P3 | bitter → MN9 not up | pass (MN9 Δ = 0, rank 32 tied at 0) | pass 10/10 | – | yes (trivial, see reading) |
| P4 | sugar+bitter < sugar (MN9) | pass: 0.0 vs 5.42 Hz (mean of L, R), lower in 3/3 seeds | pass 0/10 (0 vs 0) | – | **yes** |
| P5 | looming → TTMn | AUROC **1.000**, rank 1 (L 287, R 268 Hz) | 0.479–0.486, TTMn 0 Hz | 0.091 | **yes** |
| P6 | looming, GF silenced < P5 (TTMn) | pass: 42.5 vs 277.5 Hz (−85%), 3/3 seeds | pass 0/10 (0 vs 0) | – | **yes** |
| P7 | MDN → hind-leg pools | AUROC 0.574, best rank 8. hl mean 1.79 vs fl+ml 0.55 Hz (hl > fl/ml: yes) | 0.499–0.516; hl ≤ 0.29 Hz | 0.091 | no (AUROC < 0.9) |
| P8 | DNg100 → leg pools | AUROC 0.576, best rank 2. Legs 3.37 vs other 1.75 Hz | 0.495–0.509 | 0.091 | no (AUROC < 0.9) |
| P9 | sugar 50/100/150/200 Hz → MN9 monotonic | **fail**: 0, 0, 5.42, 12.08 Hz (non-decreasing; not strictly increasing below 150 Hz) | 0/10 (all 0) | – | no |
| X1 | wind → front-leg pools (not scored) | AUROC 0.531, best rank 10. fl 6.6 vs other 4.1 Hz; wing steering and GNG668 are on top | 0.492–0.520 | – | – |

**Hits: real 4/9 (P3, P4, P5, P6).** Without the p-value condition the count is also 4. **Shuffles: 1/9 in each of the 10** (P3 only). No shuffle passes any AUROC pair or P4, P6 or P9.

**MN9 absolute rates (Hz, mean over 3 seeds, L / R):**

| Condition | MN9 L / R (Hz) |
|---|---|
| none | 0 / 0 |
| P1 sugar (150 Hz) | 10.8 / 0.0 |
| P2 water | 0 / 0 |
| P3 bitter | 0 / 0 |
| P4 sugar+bitter | 0 / 0 |
| sugar 200 Hz | 24.2 / 0.0 |

For comparison, the repo docs give these earlier values (not re-run here): FlyWire Shiu-style sugar → MN9 about 82 Hz; MaleCNS brain-only calibrated about 14 Hz, and 0 after the Week 3 presynaptic model (`progress/week3/week3_plan.md`, `presynaptic_sensory_model.md`).

Figure: `results/figures/w5_behaviour_panel.png` (AUROC per pair, real vs 10 shuffles; direction pairs real vs number of shuffles passing).

## Persistent activity after the stimulus (0.45–0.5 s, non-driven neurons)
- Real `wind`, seed 2: **302 neurons, 33 of them motor**, are still active at 0.45–0.5 s. The other two wind seeds are at 0.
- Real `sugar_200` seeds 0 and 2: 8 and 11 neurons (no motor). Real `sugar` seed 2: 3 neurons.
- All other conditions, and all shuffles: 0.
- Noted only; the model was not tuned. The wind tail is consistent with the known wind-driven GNG reverberation (`progress/week4/day5_wind_loop_rootcause.md`). I did not check which neurons these are.

## Hypotheses
- **H1 (sugar and water drive MN9 near the top in real, not in shuffles): partly holds.**
  - Sugar drives MN9 in the real network only (10.8 Hz, left side only), and never in a shuffle. But it ranks 6th of 436 and fails the AUROC threshold (0.73) because the right MN9 is silent.
  - Water drives no motor output at all in the real network.
  - The response is far weaker than FlyWire (about 82 Hz).
- **H2 (bitter alone does not drive MN9; bitter suppresses sugar): holds, with caveats.**
  - Bitter gives MN9 Δ = 0. It drives abdominal MNs instead (MNad57 / MNad55, about 55–62 Hz).
  - Sugar+bitter removes the MN9-L response in 3/3 seeds (10.8 → 0 Hz).
  - The P3 pass is uninformative: MN9 is silent under bitter in every shuffle too. The P4 effect sits on a single neuron firing at about 11 Hz.
- **H3 (P7/P8 pass in real but some shuffles also pass; looming → TTMn strongest because of the unshuffled GF junctions): partly holds.**
  - Looming → TTMn is the strongest result (AUROC 1.0, rank 1; shuffles 0.48).
  - P7 and P8 **fail** the 0.9 AUROC in the real network (0.57, 0.58). Both are above every shuffle (p = 0.091), so no shuffle "also passes." MDN lights 14 of 54 hl pools, and also neck and GNG head pools. DNg100 lights 44 of 141 leg pools, plus wing hg/ps.
  - Shuffles keep only traces of direct DN → MN drive: hl Δ ≤ 0.29 Hz for MDN, legs ≤ 0.17 Hz for DNg100.
  - The GF junctions are indeed not shuffled (verified). But in every shuffle, looming never makes the GF fire (0 Hz), so the junctions never engage, and TTMn stays at 0. The real advantage comes from the chemical looming → GF path, which the shuffle destroys. The junction alone does not explain it.
- **H4 (silencing GF removes most of the TTMn response to looming): holds.** TTMn falls from 277.5 to 42.5 Hz (−85%) in 3/3 seeds; GF is at 0 Hz. The remaining 42 Hz comes through other chemical inputs to TTMn. The set of active motor neurons is unchanged (about 196).

## Reading
- The real CNS beats all 10 shuffles on every scored AUROC pair except water, where there is no response at all. But only looming → TTMn reaches the pre-registered AUROC ≥ 0.9.
- The model's motor responses are sparse: in most conditions, most pools sit at exactly 0 Hz. AUROC over a multi-pool hit set (hl 54, legs 141) is then capped near (fraction of hit pools active) + ties × 0.5. MN9 L/R is split by side, so a one-sided response caps P1 near 0.75. This is a property of the pre-registered metric on this model. I did not change it.
- The taste-to-feeding pathway is the weak point. Sugar → MN9 is one-sided, weak and threshold-like (nothing at 50 and 100 Hz, so P9 fails), and water does nothing. The descending-command and looming pathways behave as expected in direction (hl > fl/ml for MDN; legs > other for DNg100), but not in specificity.

## Deviations and choices (A1)
The pair table and scoring rule are unchanged. These operational choices were made where the brief leaves room:
1. MN9 and TTMn Δ = mean over the L and R pools and the 3 seeds.
2. P3 passes if that Δ ≤ 0 **or** the best MN9 pool ranks in the bottom half (rank > 218).
3. P4 and P6 pass if the mean Δ is strictly lower than P1 / P5. A 0 vs 0 result fails.
4. P9 needs a strict increase 50 < 100 < 150 < 200 Hz. The 150 Hz point is the P1 run.
5. P6 (GF-silenced copy) subtracts the intact network's `none`. Both are silent, so this makes no difference.
6. GF silencing zeroes DNp01 rows and columns in W and in the gap matrix. The presynaptic-modulation matrix is untouched, but the GF never spikes (0 Hz checked), so it has no effect.
7. A shuffle has no p-value, so a shuffle "hit" is AUROC ≥ 0.9 or a direction pass.
8. P7 "more than front/mid legs" is reported, but the hit uses the AUROC rule as written in the Scoring section.
9. Best rank uses ties-to-the-top (1 + #pools strictly higher). Zero-Δ ties therefore get optimistic ranks, e.g. P2 rank 1 at Δ = 0.
10. Bug fixed after the run: `trials_<net>.csv` got two extra columns on the `looming_GFsil` rows, with no header. `score` reads both layouts (`read_trials`), and the writer now uses fixed columns. Rates are unaffected.

## Unverified
- The identity of the 302 neurons in the wind tail.
- Why water and the right MN9 are silent: connectivity, or calibration on the MaleCNS taste path. I only ran the panel; no circuit tracing was done.
- The Week 1/3 MN9 reference rates are quoted from earlier docs, not re-run.
- K = 10 limits p to ≥ 0.091.
