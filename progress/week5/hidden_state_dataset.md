# Week 5 hidden-state dataset (T7): dataset card

Ground truth for the Week 6 label-free decoders: the simulated MaleCNS brain carries a
known internal state that switches once per episode, on top of the usual stimulus-mixture
"words". Generator `scripts/w5_hidden_state.py`, baselines `scripts/w5_hidden_state_baseline.py`,
state drive `stimuli/states.py` (Phase A), CI tests `tests/test_states.py`,
`tests/test_w5_hidden_state.py`. Data live on the project mount under
`results/w5/hidden_state/` (not in git).

## What was simulated

- 120 episodes, T = 4 s, two state segments [0, 2) and [2, 4) s.
- Words: `stimuli.mixtures.sample_episode(72000 + k, T=4.0)` (class_v2 defaults: 2-4 words of
  distinct modalities, 30-150 Hz, onset U(0, 3.7) s, duration 0.2-1 s, looming in ~1/6
  (19 of 120), bg_hz U(1, 3)).
- States: each segment's state is drawn uniformly from {baseline, hunger, thirst, arousal}
  by its own generator `np.random.default_rng(73000 + k)`, independent of the words.
  Level = U(0.7, 1.3) x the Phase A level (baseline 0 Hz). Segment counts over the 240
  segments: hunger 70, arousal 59, baseline 57, thirst 54.
- Drive: tonic Poisson input at the segment level to every neuron of the state's set
  (`stimuli.states.state_blocks`), added to `stimuli.mixtures.to_blocks(spec)`.
- Model: `sim.cns.cns_model('cns', cfg, **W4_CALIB)`, `sim.lif_torch.run_trial`, CPU float32,
  rng='torch'. Each spec runs on both configs with the same spec and the same Poisson seed:
  - `calib` (real wiring): sim/ of this branch (sim/ tree dcee740, identical to main;
    episodes 0-51 recorded at branch head 777a4e7, 52-119 at be24bb2, same sim/ tree).
    W nnz 24,519,583.
  - `shuffle` (degree-preserving control): sim/cns.py from **PR #14 head
    7be0fd3e0ef7d889b2db23f5eadede167f9e7ccf** (fixed shuffle / JO-mirror order), run from a
    separate git worktree (`--sim-root`), not committed here; #14 was still unmerged when
    the shuffle half started. W nnz 24,444,089 (-0.31 % vs calib, within the 1 % check).
- Runaway check: per episode and segment, fraction of brain neurons (sim.cns.brain_mask,
  mask excluded) above 200 Hz. Maximum over all 480 segments: 0.000 % (limit 1 %).
- Wall time: calib 3.26 h, shuffle 2.89 h simulation (24.5 / 21.7 s per bio-s), one process
  at a time, resumable chunks (two container restarts; the shuffle tail ep108-119 was re-run).

## State sets and levels

| state | cell type (set) | n | NT | level (Phase A) | per-segment level |
|---|---|---|---|---|---|
| baseline | none | 0 | - | 0 Hz | 0 Hz |
| hunger | ISN | 4 | ACh | 20 Hz | U(14, 26) Hz |
| thirst | BiT | 2 | ACh | 100 Hz | U(70, 130) Hz |
| arousal | pC1/P1, `dimorphism == 'male-specific'` | 126 | ACh | 20 Hz | U(14, 26) Hz |
| (sleep, dropped) | ER5 | 21 | GABA | none (no downstream effect at any level) | never driven |

Body IDs and selection rules: `results/w5/hidden_state/state_sets.json` (unchanged since
Phase A). Phase A levels follow the >= 3 SD rule with the accepted deviation: SD floor
0.0133 Hz (one spike over the 50 targets in the 1.5 s window) and a delta > 0 requirement
(`results/w5/hidden_state/pilot_dose.md`).

## File layout (`results/w5/hidden_state/dataset/`)

| path | content |
|---|---|
| `activity/ep{k:03d}_{cfg}.npz` | `t` (step, dt 0.1 ms, int32), `i` (neuron index = row of `data/malecns/malecns_traced_meta.parquet`, int32), `spec` (JSON, words only), `cfg`, `sim_rev`, `n_spikes`, `frac200` (per segment), `wall_s`. Decoder-mask neurons removed. |
| `specs.json` | the 120 words specs (no state) |
| `index.csv` | one row per activity file (240): seeds, words, bg_hz, spikes, frac200_max, wall, sim_rev; no state columns |
| `labels/ep{k:03d}.npz` | `state_id` (int16) and `level_hz` (float32) per 10 ms bin (400), `segments` (JSON), `bin_s`, `states` (id order: baseline 0, hunger 1, thirst 2, arousal 3, sleep 4) |
| `labels/segments.json`, `labels/index.csv` | the segments of every episode, design constants |
| `labels/driven/ep{k:03d}_{cfg}.npz` | spikes of the 153 decoder-mask neurons (they carry the drive, so they count as labels) |
| `run_log.txt` | per-episode wall time, model builds, nnz check |

## Label hiding

- Activity files hold no state, level or segment field, and no spike of any decoder-mask
  neuron: the mask is all neurons of the four Phase A sets (ISN 4 + BiT 2 + pC1/P1 126 +
  ER5 21 = 153), stored in `state_sets.json` under `decoder_mask` (neuron indices and body IDs).
  ER5 is masked too although it is never driven.
- A decoder therefore has to infer the state from downstream effects. The words spec is
  observable; it is generated independently of the state.
- Labels (and the masked spikes) are only under `labels/`. Week 6 decoders must read only
  `activity/`, `specs.json` and `index.csv`.

## Baselines

Recompute: `python scripts/w5_hidden_state_baseline.py` (results in
`results/w5/hidden_state/baseline/`: summary.md, results.json, confusion_*.csv, pred_*.npz, log.txt).

Design: 50 ms bins (80 per episode), target = state id of the bin; inputs = brain_mask neurons
minus the 153 decoder-mask neurons (133,105 candidates; 7,552 calib / 7,324 shuffle spike at all;
~6,200 kept per fold with >= 5 training spikes), sqrt counts, z-scored, 100 PCs fitted on the
training fold only; multinomial logistic regression, C from {0.001, 0.01, 0.1, 1} by inner 4-fold
CV by episode; outer 5-fold CV by episode (same folds for every row); balanced accuracy on the
pooled held-out bins. Words-only: per-bin rate of each word if on, episode word one-hot, bg_hz
(no activity). Permuted: calib activity with the label sequences permuted across episodes.

| row | inputs | balanced acc. (pooled) | fold mean +- SD | recall baseline / hunger / thirst / arousal |
|---|---|---|---|---|
| (a) supervised, real wiring (calib) | brain, masked | **0.713** | 0.716 +- 0.015 | 0.71 / 0.65 / 0.50 / 1.00 |
| (a) supervised, shuffle control (PR #14) | brain, masked | **0.481** | 0.494 +- 0.027 | 0.33 / 0.55 / 0.26 / 0.79 |
| (b) chance, analytic (4 classes) | - | 0.250 | - | - |
| (b) chance, empirical (calib, labels permuted across episodes) | brain, masked | 0.222 | 0.240 +- 0.022 | 0.18 / 0.41 / 0.12 / 0.17 |
| confound: words only | stimulus spec | 0.237 | 0.246 +- 0.053 | 0.25 / 0.29 / 0.21 / 0.20 |

Arousal (pC1/P1, 126 neurons) is decoded almost perfectly; thirst (BiT, 2 neurons) is the hardest.
The shuffle is above chance: shuffled wiring keeps every row's in-degree and every neuron's
out-degree, so driving 126 (or 4, 2) neurons still moves some downstream activity, but less
specifically.

## Hypotheses

| | hypothesis | verdict | evidence |
|---|---|---|---|
| H1 | each state set has a dose that moves downstream activity without runaway | **partly holds** | ISN 20 Hz, BiT 100 Hz, pC1/P1 20 Hz move their top-50 targets >= 3 SD with 0 % runaway; ER5 moves nothing up to 200 Hz (dropped). In the mixed episodes no segment exceeded 0.000 % of brain neurons > 200 Hz. |
| H2 | pC1/P1 needs a much lower level than ISN/BiT | **partly holds** | pC1/P1 has by far the largest effect per Hz (+30 Hz on its targets at 20 Hz vs +1.5 Hz for ISN, +0.2 Hz for BiT at 100 Hz) and is 5x below BiT, but its chosen level equals ISN's because 20 Hz was the lowest level tested. |
| H3 | supervised decoding on real wiring well above chance and clearly above shuffle | **holds** | 0.713 real vs 0.481 shuffle vs 0.25 chance (0.222 empirical); fold SDs 0.015-0.027. |
| H4 | state not predictable from the words alone | **holds** | words-only 0.237 (fold mean 0.246), at chance; states are drawn by a separate generator. |

## Known gaps

- ISN -> BiT has 0 synapses in MaleCNS (the FlyWire-described inhibition is absent), so hunger
  and thirst are two independent drives here, not one signed axis; the thirst mapping to BiT is
  unverified.
- 327 neurons labelled serotonin in the meta but "unclear" in the NT table (e.g. NPFL1-I,
  CRZ02, l-LNv, DSKMP3) are silenced with the monoamines, so those state candidates are out.
- Monoamines are removed in the model: states are tonic drive to central cell types, not
  neuromodulation.
- Thirst (BiT, 2 neurons) has a weak downstream footprint (Phase A: +0.21 Hz at 100 Hz on its
  top-50 targets); expect it to be the hardest state.
- Behaviour is not a check (sugar->MN9 weak, water drives nothing, PR #13 panel).
- The shuffle half used the unmerged PR #14 sim/cns.py (head 7be0fd3); if #14 changes before
  merge, the shuffle half should be re-checked against the merged version.
- One state switch per episode at a fixed time (2 s); segment lengths do not vary.
