# Week 4 Day 5 (T2b): class-specific background, bristles at 0 Hz

Follows `day5_background_diagnosis.md` and Charlie's decisions of 2026-11-18: bristle sensory neurons get 0 Hz background; the 1,040 `unknown_sensory | abdomen` neurons stay configurable until they are identified; the Day 5 background range is 1–3 Hz; looming is in about 1 of 6 episodes.

Nothing under `sim/` changed.

## Code
**`stimuli/mixtures.py`**
- **Spec records its own background scheme.** New specs carry `bg_scheme = {name: 'class_v1', bristle_hz, abdominal_hz}`. A spec without `bg_scheme` (every T1 pilot spec) is a legacy spec: `bg_hz` on all 15,912 sensory neurons, exactly as on Day 4.
- **`sample_episode(rng, words, T, legacy=False, looming_p=1/6, bg_range=(1, 3), bristle_bg=0.0, abdominal_bg='same')`.**
  - `legacy=True` runs the Day 4 sampler draw for draw: uniform modalities, background 1–5 Hz, no `bg_scheme`.
  - `abdominal_bg` takes `'same'` (= the episode's `bg_hz`, the default for now), `0`, or a rate in Hz.
- **Looming draw.** The looming modality (`vision`, which has only the word `looming`) is included with probability `looming_p`. The other 1–4 modalities are drawn uniformly without replacement from the remaining six. This sets the looming share exactly in expectation, without a weighted draw without replacement.
- **`background_episode(bg_hz, seed, T, legacy=False, bristle_bg, abdominal_bg)`** follows the same rules.
- **`bg_class_masks(meta)`** defines three masks:
  - `sensory` = `sim.presyn.sensory_mask_malecns`;
  - `bristle` = sensory & VNC & subclass `mechanosensory bristle`. VNC uses the `sim.cns.regions` rule, copied into `mixtures._vnc` so that the module does not import `sim.cns` (its loader needs pyarrow, which CI does not install). The checks script asserts equality on the real meta;
  - `abdominal` = sensory & class `unknown_sensory` & subclass `abdomen`.
- **`to_blocks` / `bg_blocks`.**
  - One background block at `bg_hz` on every sensory neuron except bristles. The abdominal group is also left out when its rate differs from `bg_hz`.
  - Then a bristle block if `bristle_hz > 0`, and an abdominal block if its rate is separate and > 0. A neuron at 0 Hz is in no block.
  - Word blocks follow, unchanged.

**Other changes**
- **`scripts/w4d4_pilot.py`** and **`w4d4_pilot_summary.py`** now call the sampler with `legacy=True`, so re-running them regenerates the T1 pilot specs and its looming-share line.
- **`tests/test_mixtures.py`** (CI, synthetic meta, no data needed) checks masks, block structure, legacy blocks, legacy sampler against a frozen Day 4 copy, looming share and `bg_hz` range, and argument validation.
- **`scripts/w4d5_bg_fix_checks.py`** runs the same checks on the real MaleCNS meta and the saved pilot `specs.json`. Output: `results/w4/bg_fix/checks.txt`.
- **`scripts/w4d5_bg_fix_runs.py`** runs the verification simulations. Output: `results/w4/bg_fix/runs/`, `runs_log.txt`, `verify.csv`.

## Class masks (real meta)
- **Masks:** bristle 2,206 and abdominal 1,040, the same counts as the diagnosis. Overlap is 0, and 12,666 other sensory neurons remain.
- **Bristle mask contents.** It includes 2,130 `mechanosensory_tactile | mechanosensory bristle` and 76 `unknown_sensory | mechanosensory bristle`.
- **Other sensory subclasses named "bristle" stay at the background rate.** All of these are gustatory:
  - `labellar bristle` (cb_sensory, 163);
  - `leg bristle` (688 vnc_sensory + 80 sensory_ascending);
  - `wing bristle` (385);
  - `taste bristle` (70, class missing).
- **Head mechanosensory bristles are NOT in the mask (decision needed).** 932 head neurons have type `BM*`, all class `mechanosensory` with no subclass: BM_InOm 745, BM 69, BM_Taste 40, BM_vOcci_vPoOr 26, BM_Vib 22, BM_MaPa 15, BM_Vt_PoOc 8, BM_Hau 7. They are the same modality as the VNC tactile bristles. The brief defined the mask as VNC only, so they still get the episode's background. In the diagnosis, head-only background at 3 Hz drove 2–9 MNs, so they are not the MN driver; whether they should also get 0 Hz is Charlie's call.

## Unit checks (`results/w4/bg_fix/checks.txt`; all 13 pass; `pytest tests/`: 14 pass, also with pyarrow blocked as in CI)
- **Block structure.** With `abdominal_bg='same'`, there is one block of 13,706 neurons at 3 Hz and no bristle in it. With `abdominal_bg=0`, there is one block of 12,666 neurons and no abdominal neuron in any block. With a rate of 2 Hz, the abdominal group gets its own 1,040-neuron block. Every other sensory neuron is covered exactly once, and no non-sensory neuron is covered.
- **Looming share** (new sampler, spec seeds 0–4999): **803 / 5000 = 0.1606**. The target is 1/6 = 0.1667 ± 0.02, and the binomial SE is 0.005.
  - The legacy sampler gives 2,083 / 5000 = 0.417, against 3/7 expected.
  - Each other modality is now in 0.470–0.485 of episodes, up from about 0.43: the probability taken from looming moves to them.
  - Words per episode (2/3/4): 1,619 / 1,653 / 1,728.
- **`bg_hz`** stays in [1, 3] (mean 2.00).
- **T1 pilot:**
  - `w4d4_pilot.specs()` equals the saved `results/w4/regression/pilot/specs.json` (12 specs).
  - `to_blocks` of every saved spec equals the Day 4 `to_blocks` (git e3bbd01) for 12 of 12 specs.
  - Legacy `sample_episode` equals the Day 4 one for seeds 0–999.

## Verification runs
**Setup.** Background only at 3 Hz, 1.0 s, seeds 0–2. `cns_model('cns', cfg, **W4_CALIB)`, CPU float32, `rng='torch'`. The runs were done one at a time in one process. Blocks come from `to_blocks(background_episode(...))`.
- "legacy" = all sensory at 3 Hz. These runs are spike-identical to the diagnosis runs `dose_calib_all_3Hz_s0/s1`, `dose_shuffle_all_3Hz_s0/s1` and QA's `all_3Hz_s2`.
- **Metric definitions:**
  - "Active" means at least 1 spike in 1 s.
  - "Non-sens per 100 ms" is the mean number of active non-sensory neurons over the 100 ms bins 0.1–1.0 s, as in the diagnosis. The VNC and brain columns use the diagnosis groups.

| cfg | background | seed | MNs active | MN spikes | MN spikes ÷ legacy | non-sens active (1 s) | non-sens / VNC / brain active per 100 ms |
|---|---|---|---|---|---|---|---|
| calib | legacy (all sensory) | 0 | 87 | 606 | — | 998 | 492 / 290 / 158 |
| calib | legacy (all sensory) | 1 | 114 | 745 | — | 1,171 | 540 / 317 / 170 |
| calib | legacy (all sensory) | 2 | 77 | 544 | — | 983 | 478 / 284 / 155 |
| calib | bristle 0, abdominal same | 0 | 50 | 127 | 21.0% | 689 | 245 / 129 / 102 |
| calib | bristle 0, abdominal same | 1 | 41 | 100 | 13.4% | 616 | 248 / 125 / 108 |
| calib | bristle 0, abdominal same | 2 | 28 | 50 | 9.2% | 998 | 280 / 126 / 146 |
| calib | bristle 0, abdominal 0 | 0 | 12 | 14 | 2.3% | 799 | 213 / 78 / 131 |
| calib | bristle 0, abdominal 0 | 1 | 12 | 14 | 1.9% | 491 | 157 / 76 / 80 |
| calib | bristle 0, abdominal 0 | 2 | 9 | 16 | 2.9% | 491 | 179 / 82 / 93 |
| shuffle | legacy | 0 / 1 / 2 | 0 / 2 / 1 | 0 / 2 / 1 | — | 185 / 188 / 193 | 37 / 37 / 35 |
| shuffle | bristle 0, abdominal same | 0 / 1 / 2 | 0 / 0 / 1 | 0 / 0 / 1 | — | 193 / 173 / 202 | 33 / 34 / 36 |
| shuffle | bristle 0, abdominal 0 | 0 / 1 / 2 | 1 / 1 / 1 | 1 / 1 / 1 | — | 189 / 179 / 176 | 37 / 29 / 32 |

**Means over seeds 0–2 (calib):**

| background | MNs active | MN spikes | share of legacy MN spikes | non-sens active per 100 ms |
|---|---|---|---|---|
| legacy | 92.7 | 632 | 100% | 503 |
| bristle 0, abdominal same | 39.7 | 92 | 14.6% (9–21% per seed) | 258 |
| bristle 0, abdominal 0 | 11.0 | 14.7 | 2.3% (1.9–2.9% per seed) | 183 |

**Measured against the brief's expectations (from the diagnosis):**
- **Bristle 0 with abdominal 'same' should leave about half of the MN drive. It does not hold:** 9–21% of MN spikes remain per seed (mean 14.6%). MNs active drop from 77–114 to 28–50. The diagnosis's per-class runs explain why "half" overestimates. The abdominal class alone gave 0.25 of the all-VNC-sensory MN spikes. The classes add sub-linearly, and with bristles silent the shared L1 → AN19B001 → IN19B008 chain gets less summed input, so it is recruited less. This was not checked further.
- **Both at 0 should leave about 2–3% (≤ about 20 MN spikes). This holds:** 1.9–2.9% per seed, 14–16 MN spikes and 9–12 MNs active.
  - That compares with the diagnosis estimate of about 13 + 3 spikes (complement run + head-only).
  - Shuffle is at 0–1 MN spikes for every scheme, so the remaining calib MN drive depends on the real wiring.
- **Brain bursts (not investigated).** Non-sensory activity falls less than MN activity because brain activity is mostly head-driven.
  - Two of the six class_v1 calib runs (abdominal same s2, abdominal 0 s0) contain one 100 ms bin with a burst of about 400 brain neurons. Those bins have 424 and 386 active brain neurons, against about 100 in other bins.
  - The bursts are mostly antennal-lobe LNs (lLN*), lateral horn (LHA/LHP), GNG and DNg types. They raise "brain non-sens active over 1 s" to 584–624 in those runs.
  - The three legacy calib runs do not show such a bin. Head background is the same in all schemes, so this looks like a seed-dependent brain event, not an effect of the VNC change. It was not investigated.

## Day 1 parity
`python scripts/w4d1_torch_parity.py parity cns 0.1` (vinegar) gives 2,232 spikes in both torch and NumPy: 0 neurons differ, identical=True (`results/w4/w4d1_parity_cns.txt`). No `sim/` file changed.

## Assumptions and what is unverified
- **Bristle mask follows the brief: VNC only.** The 932 head BM* bristle neurons and all gustatory "bristle" subclasses still get the episode background.
- **The abdominal default stays `'same'`** until Charlie chooses. The verification shows that with 'same', 28–50 MNs are still active at 3 Hz (50–127 MN spikes), against 9–12 MNs (14–16 spikes) with 0.
- **Run coverage.** The verification is at 3 Hz only, the top of the new 1–3 Hz range. Lower rates and mixtures with words were not run with the new scheme. No full episode with the new sampler was simulated.
- **Looming share** is the share of episodes containing looming. The share of time or of word slots was not set.
- **Different random draws.** New specs use a different RNG draw order from Day 4 (one extra `rng.random()` for the looming decision). The same spec seed therefore gives different episodes under the new and legacy samplers. That is intended, and legacy specs are reproduced only with `legacy=True`.
