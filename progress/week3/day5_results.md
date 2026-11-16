# Week 3 Day 5 results: the closed loop, end to end (11 Nov 2026)

Brain: calibrated MaleCNS + presynaptic modulation (K = 0.05) + right-JO mirror. Shuffled control: the same, with the chemical matrix shuffled while keeping each neuron's degree and output sign.
Code: `loop/run_loop.py` (runner), `loop/words_readout.py`, `loop/action_readout.py`, `loop/prompt.py`, `loop/llm.py`, `loop/actions.py`; driver `scripts/w3d5_loop.py`.
Results: `results/loop/w3d5_loop_<cfg>_<seed>.json` (every round, plus population spikes in 10 ms bins for playback), `w3d5_loop_*.log`.

## What one run does
Three rounds in **one continuous simulation** (10.5 s of brain time):
1. 5 words back to back (500 ms each). Round 1 is plan.md's demo (vinegar, ripe fruit, sugar, hot, looming); rounds 2–3 are random.
2. The word readout decodes each 500 ms window and builds the prompt. The LLM picks one of five actions.
3. The action plays as its JO code (500 ms), followed by 500 ms of silence. The descending-neuron readout decodes it.
4. The next round starts in the same brain, with the previous rounds' activity still present.

## Readouts retrained for the fixed brain
- **Words:** 17 words × 10 seeds + `none`, on the mirrored brain (`words_readout_presyn0.05m.pkl`, 4,915 features) and on the shuffled mirrored brain (`…ms.pkl`, 6,463 features). The mirror changes the `sound` word, which uses JO-A/B on both sides.
- **Actions:** descending neurons only, trained on the Day 4 training trials (`action_readout_real.pkl` 445 features, `_shuffle.pkl` 24 features). Day 4 held-out: 37/37 real, 36/37 shuffled.

## Results (offline rule-based stand-in for the LLM)
| Run | Words correct | Actions read back correctly | Action confidence |
|---|---|---|---|
| real, seed 4000 | 15 / 15 | 3 / 3 (turn left, go forward, eat) | 0.98–1.00 |
| real, seed 4001 | 15 / 15 | 3 / 3 (turn left, turn right, turn right) | 0.84–0.97 |
| real, seed 4002 | 15 / 15 | 3 / 3 (turn left, stop, eat) | 0.97–0.99 |
| shuffled, seed 4000 | 15 / 15 | 3 / 3 (turn left, go forward, eat) | 0.81–0.93 |
| **Total** | **60 / 60** | **12 / 12** | |

Example round (real, seed 4002, round 2): sound, dry, hot, mold, bitter → all five decoded → "I hear a sound. The air feels dry. It feels hot. I smell mould. I taste something bitter. What should I do?" → **stop** → JO-B pulse code → descending neurons read back **stop** (p = 0.975).

- **Replay check:** the runner re-runs from t = 0 with the extended stimulus list at every step. In all 8 checks, the earlier spikes were reproduced exactly (`prefix_identical = True`).
- **The shuffled brain also closes the loop**, but with lower action confidence (0.81–0.93 vs mostly ≥ 0.97). This is the same lesson as Days 1–4: the loop reads back what we put in, whatever the wiring.

## Acceptance (from the Week 3 plan)
1. Fly → LLM words ≥ 95% in episodes: **met** (60/60 here; 200/200 on Day 1).
2. LLM → fly 37/37: **met** (Day 4), and 12/12 inside the loop.
3. 3-round loop in under ~10 min of compute: **not met.** It took 18–23 min per run with 4 runs in parallel, because the simulator can't save and resume its state, so each step replays from t = 0 (39 s of brain time instead of 10.5 s). Adding save/resume to `sim/lif_numpy.py` would bring it to about 5 min. That's worth doing before the video.
4. Shuffled numbers alongside: **met.**

## Notes
- The stand-in LLM's replies are placeholders. For example, "turn right" for sugar with mould comes from its rule order, not from any reasoning. To use a real LLM, run `python scripts/w3d5_loop.py real 4000 claude-cli` on a computer where Claude Code is logged in, or set `OPENAI_API_KEY` and use `openai`.
- The actions remain arbitrary labels, since the steering DNs are silent (Day 3/4).
