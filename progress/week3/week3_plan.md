# Week 3 plan: close the fly ↔ LLM loop

*5 Nov 2026. Based on `docs/plan.md` (Week 3), `progress/week2/` (summary, Day 1, Days 2–3, next steps after C8), `results/readout/`, the `sim/`, `stimuli/` and `readout/` code, and the MaleCNS metadata.*

## The short version

Week 2 left a calibrated MaleCNS brain (eLN–PN gap junctions, no odor attractor), a 17-word readout that decodes the 0–0.5 s window at 100%, and the published Connectome Graph Explorer. Week 3 reproduces the rest of chetaslua's demo on that substrate, honestly:

1. **Fly → LLM.** Play a multi-word episode into one continuous simulation, decode each 500 ms window to a word, turn the words into a prompt, and log the LLM's reply.
2. **LLM → fly.** Map a 5-action vocabulary to stimulation codes on Johnston's organ (JO) neurons, and train a descending-neuron (DN) readout that recovers the action. Target: 37/37 held-out trials, as in the original.
3. **Show it.** A 60 s screen capture, a "Loop" view in the Explorer, and a README that says what is deterministic, what is learned, and what the shuffled brain also does.

The one new scientific risk is **carry-over between windows**. Week 2 trained and tested on isolated trials that start from rest. In an episode, each window starts with the previous word's tail still running (calibrated tails: sugar ~50 neurons, bitter ~110, wind ~400, looming 470–1,230). Day 1 measures this before anything is built on it.

## Where Week 2 left us (numbers this plan depends on)

| Item | Value | Source |
|---|---|---|
| Stimulus-window readout, calibrated, all non-sensory | 100% split / 100% leave-one-seed-out | `results/readout/readout_summary.json` |
| Same, DNs + MNs only | 94% / 94% (mold → vinegar is the only error) | same |
| Shuffled brain, all / motor-only | 100% / 85% | same |
| Post-stimulus window (0.5–1.0 s), calibrated, all / motor | 94% / 68% | same |
| Calibrated tails after offset (brain neurons still firing) | 0 for most words; cold ~220, humid ~170, wind ~395, looming ~500 (the second loop, outside the AL) | `day2_3_gap_junctions.md`, `trials_calib_log.txt` |
| MaleCNS sugar → MN9, brain-only calibrated | ~14 Hz (FlyWire 82 Hz) | `day2_3_gap_junctions.md` |
| JO neurons in MaleCNS brain set | JO-A 50, JO-B 88, JO-C 68, JO-E 267. Side is in `instance` (`_L`/`_R`), not `somaSide`. JO-A is lopsided (24 of 50 are `JO-A-unclear_R`) | `malecns_traced_meta.parquet` |
| Steering DNs present | DNa01, DNa02, DNa03, DNg13, one per side each | same |

What is **not** built yet: `loop/` is empty, `notebooks/` doesn't exist, there is no `.env` with an LLM key, and the README status list is still from Week 1.

## Day 1: episodes in one continuous simulation

**1a. Simulator: time-varying stimulus.** `run_trial` takes one `exc_idx` and one `t_stim`. Add a `schedule=` argument (rename the current `schedule` arg to `step_order` to avoid the clash): a list of `(t0, t1, neuron_idx, rate_Hz)` blocks inside one run, so the network state carries over. Regression check: a one-block schedule must give spikes bit-identical to the current call on FlyWire sugar and MaleCNS vinegar (calibrated).

**1b. Carry-over test (the go/no-go for the rest of the week).** 5-word episodes, 500 ms per word, words drawn at random from the 17 (no repeats inside an episode), 40 episodes = 200 windows, calibrated model. Decode each window with the Week 2 readout *as trained on isolated trials* (`readout/linear.py`, all non-sensory and DN+MN). Three spacings:

| Spacing | What it tests |
|---|---|
| back-to-back (0 ms gap) | the honest worst case |
| 250 ms silence between words | whether a short gap lets the tails die |
| 500 ms silence | upper bound on what a gap buys |

Report accuracy by spacing, and accuracy by **previous** word, so we can see whether looming and wind are the offenders.

**Decision rule (no new series):**
- ≥ 95% back-to-back: use 0 ms gaps.
- Otherwise, retrain the readout on episode windows (seeds 0–6 for training episodes, 7–9 for test), so it learns "word X after anything". If that reaches ≥ 95%, use it.
- Otherwise, use the smallest gap that reaches ≥ 95%, and write the gap down as a limitation. Do not go back to fixing the second loop this week.

**1c. A "nothing" class.** Add `none` (no drive, 10 seeds) as an 18th class, so a silent window or gap decodes as nothing instead of a forced word. Also record the readout's top probability per window; anything under 0.5 is logged as "unsure" and left out of the prompt.

Output: `scripts/w3d1_episodes.py`, `results/loop/w3d1_carryover.csv`, `results/loop/w3d1_log.txt`.

## Day 2: fly → LLM

**2a. Prompt template** (`loop/prompt.py`). Each decoded word becomes a phrase by modality, from `stimuli/malecns_words.py`:

| Modality | Phrase |
|---|---|
| taste | "I taste {sugar}." |
| smell | "I smell {vinegar}." (fruit → "ripe fruit", co2 → "CO2", mold → "mould") |
| temperature / humidity | "It feels {hot / cold / dry / humid}." |
| hearing / wind | "I hear a sound." / "I feel wind." |
| vision | "Something is looming toward me." |

The prompt ends with: "You are a fruit fly. Choose one action: turn left, turn right, go forward, stop, eat." The reply is constrained to that list (structured output / tool call with an enum), temperature 0, and the raw reply is logged. If the reply isn't one of the five, it is logged and mapped to `stop`.

**2b. LLM client** (`loop/llm.py`). Provider-agnostic, key read from `.env`. Default to OpenAI to match "fly brain to ChatGPT"; Claude is a one-line switch. Also an **offline stub policy** (rule-based: sugar → eat, looming → turn away, bitter → stop, odors → go forward) so every test and the video can run without a key or network.

**2c. Demo episode from plan.md.** "vinegar + yeast + sugar + warm + bright" maps onto the vocabulary as vinegar, fruit, sugar, hot, looming (looming stands in for "bright"; photoreceptors are inhibitory in this model, see `malecns_words.py`). Put looming last, because its tail is the longest. Run it on 10 seeds and log decoded words, prompt, reply.

Output: `loop/episode.py`, `loop/prompt.py`, `loop/llm.py`, `results/loop/w3d2_fly_to_llm.jsonl`.

## Days 3–4: LLM → fly

**3a. Screen candidate codes on JO** (Day 3). The 5 actions need 5 stimulation patterns that stay inside "hearing" neurons. Screen 8 candidates, 500 ms each, 10 seeds, calibrated model:

| Code | Neurons | Pattern |
|---|---|---|
| A-L, A-R | JO-A, left or right (`instance` suffix) | 150 Hz tonic |
| B-L, B-R | JO-B, left or right | 150 Hz tonic |
| A+B | JO-A and JO-B, both sides | 150 Hz tonic |
| A-pulse | JO-A, both sides | 20 ms on / 20 ms off at 300 Hz (courtship pulse-song-like) |
| B-pulse | JO-B, both sides | same pulse train |
| A-low | JO-A, both sides | 50 Hz tonic |

Fit a DN-only readout (the 1,314 descending neurons only; no MNs, no JO, no other sensory) with leave-one-seed-out. Keep the 5 codes with the best pairwise separation (lowest confusion), and assign them: left/right codes to turn left/right, the others to forward, stop and eat.

Note the left/right lopsidedness: JO-A has more `_R` neurons than `_L`. If A-L and A-R aren't symmetric, try matching the number of neurons stimulated per side (subsample the larger side) before dropping them.

**3b. Verify, chetaslua style** (Day 4). Train the DN readout on seeds 0–6 for the 5 chosen codes. Test on 37 fresh trials (seeds 10–46, actions in a fixed random order). Target 37/37. Also run:
- the **shuffled brain** with the same codes and the same readout procedure;
- a **"does left mean left?" check:** do DNa01/DNa02 fire asymmetrically for the turn-left vs turn-right codes? This isn't needed for 37/37, but it is the one place the loop could say something about the real wiring. Report it either way.

**3c. Separate the two directions in time.** JO-A/B is also the "sound" word, and JO-C/E is "wind". The loop alternates phases (fly → LLM episode, then LLM → fly action), with the Day 1 gap between them, so the action code is never decoded as a word. Log what the word readout says during the action window anyway; it should say "sound".

Output: `loop/actions.py` (codes and DN readout), `results/loop/w3d3_code_screen.csv`, `results/loop/w3d4_verify.csv`, logs.

## Day 5: one closed loop, end to end

`loop/run_loop.py`: three rounds of episode → decode → prompt → LLM → action → JO code → simulate → DN readout → decoded action, all in one continuous simulation. Each round logs the true words, decoded words, prompt, raw reply, intended action and decoded action (`results/loop/w3d5_loop.jsonl`). Run it on the calibrated brain and on the shuffled brain with the offline stub, then once with the real LLM.

**Acceptance for the week:**
1. Fly → LLM: ≥ 95% of words correct inside episodes, at the spacing chosen on Day 1.
2. LLM → fly: 37/37 (or the honest number) on held-out trials.
3. A full 3-round loop runs on a Mac in under about 10 minutes of compute.
4. The shuffled-brain numbers for 1 and 2 are in the same table as the real ones.

## Day 6: Loop view and video

- **Explorer "Loop" tab.** Export one logged loop run (`scripts/w3d6_export_loop.py` → `results/graph/loop_*.json`): population activity in 10 ms bins, the decoded word per window, the prompt, the reply, and the decoded action. The tab plays it on the existing calibrated graph with a caption strip underneath. Keep it within the 16 MB page budget, and republish the same artifact URL.
- **60 s capture.** Record the Loop tab playing (Playwright can make the video automatically; a manual screen capture also works). Save to `results/figures/loop_demo.mp4`.

## Day 7: write-up and repo tidy

- **README rewrite.** Update the status list (Weeks 1–3 done), and add a section with three plain lists:
  - *Deterministic:* the connectome, the LIF equations, the calibration, the stimulus sets and rates.
  - *Learned:* only the two linear readouts (word and action). The connectome weights are never trained.
  - *What the shuffled brain also does:* decodes all 17 words at 100% from all neurons and (fill in from Day 4) the action codes. The real wiring shows only in the motor-only and post-stimulus numbers.
  - Use the wording from plan.md's pitfalls: "connectome-constrained LIF simulation with a trained readout", never "the fly talks".
- **Notebooks** (plan.md deliverable, not written yet): `01_shiu_validation.ipynb` (Week 1 MN9 figure), `02_translator.ipynb` (Week 2 readout and confusion matrices), `03_gpt_loop.ipynb` (this week). Each one loads saved results; none re-runs the long simulations.
- **Week 3 report** in the style of `docs/week1_report.html`.

## Time limits and stopping rules

- Day 1 is the only place where the plan may change shape. Its decision rule picks the spacing; don't open a carry-over sweep series.
- If no 5 JO codes reach 37/37 by the end of Day 4, report the best number, use the best 5, and go on. Rate/pulse codes on other sensory channels (e.g. the unused JO-C/E or JO-F) are an option for later, not this week.
- If the LLM key isn't available by Day 5, ship with the offline stub and add the real LLM call afterwards. Nothing else depends on it.

## Not worth doing this week

- Fixing the second, non-AL loop behind the looming and wind tails. Note it, design around it (Day 1), and leave the SCC search for later.
- Fixing eLN shunting (gap-junction cap). It brings the attractor tail back, and the loop doesn't need eLNs to fire.
- Raising MaleCNS sugar → MN9 from 14 Hz towards chetaslua's 78 Hz. That belongs to the Phase 2 full-CNS port.
- Simultaneous mixtures, intensities, noise, or a 1% electrode subset. These are Phase 2B (weeks 6–8).

## Open questions

- Do the JO codes reach the steering DNs at all in a brain-only model, or does the DN readout pick up something else? Day 4's DNa01/DNa02 check answers this.
- Does a readout trained on episodes still need the word to start from rest, or does it work at any phase of the previous tail? Day 1's accuracy by previous word answers this.

## Needed from Charlie

An LLM API key in `flybrain/.env` (`OPENAI_API_KEY`, or `ANTHROPIC_API_KEY` for Claude) by Day 5. Everything before that runs offline.
