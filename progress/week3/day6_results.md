# Week 3 Day 6: split-screen loop video

## What was built
- **Save/resume** in `sim/lif_numpy.py` (`state=` / `return_state=`): a loop round continues the same simulated brain instead of replaying from t=0. Resumed runs are spike-for-spike identical to a single long run (`scripts/w3d6_resume_check.py`). Loop wall time dropped from ~20 min to 5–7.5 min per 3-round run (acceptance #3, ≤10 min, now met).
- **Loop runs with spikes saved** (`w3d5_loop.py <cfg> <seed> stub w3d6`): real 4000/4001/4002 and shuffled 4000.
- **Export** `scripts/w3d6_export_loop.py` → `artifact/loop/data/loop_<key>.json`. Raster rows: sensory band = up to 8 most active neurons per stimulus set (each word and both JO code groups get a stripe), antennal lobe ≤120, central brain ≤260, descending ≤160, each sorted by first spike.
- **Player** `artifact/loop/index.html`, published at https://claude.ai/artifact/6XL82fiqLqjcc1ZLsjZXpZ . Left: spike raster for the current round, with word windows, the action-code window and the quiet window marked, and population rate below. Right: chat of decoded words (with confidence), the sentence sent to the LLM, the LLM reply, the spike code sent into the hearing neurons, and the descending-neuron readback. Run picker, 0.1–1× speed, scrubber, light/dark.
- **Video** `results/figures/loop_demo.mp4`: 61 s, 1920×1080, 30 fps, dark theme, real connectome seed 4000 at 0.2× brain speed with title and end cards. Rendered frame by frame (deterministic), not screen-captured.

## Results (w3d6 runs)
| run | words | actions | action confidence per round |
|---|---|---|---|
| real 4000 | 15/15 | 3/3 | 0.978, 0.996, 0.975 |
| real 4001 | 15/15 | 3/3 | 0.972, 0.869, 0.842 |
| real 4002 | 15/15 | 3/3 | 0.974, 0.975, 0.988 |
| shuffled 4000 | 15/15 | 3/3 | 0.811, 0.878, 0.931 |

Same as Day 5 (resume does not change results). The shuffled control also passes, with lower action confidence, so the loop shows the readout pipeline works; it does not by itself show the real wiring is needed (see Day 4: real 37/37 vs shuffled 36/37).

## Caveats shown in the video and player
- LLM replies come from the offline rule-based stand-in (`stub`, rules-v1) and are labeled as such. Re-render with real replies by running `w3d5_loop.py real 4000 claude-cli w3d6` on a machine logged into Claude, then `w3d6_export_loop.py`.
- Action codes are arbitrary spike patterns on JO hearing neurons, learned by a DN readout; they are not natural steering commands.

## Update (Nov 12): re-filmed with a real LLM
Charlie asked for the video to use a real LLM. All 4 loops were re-run with `CLAUDE_CLI_MODEL=claude-sonnet-5-5 python3 scripts/w3d5_loop.py <cfg> <seed> claude-cli w3d6c` (Claude through the Claude Code CLI, about $0.14 per reply, 12 replies), exported with `python scripts/w3d6_export_loop.py w3d6c`, and the video re-rendered. The stub-reply video is kept as `results/figures/loop_demo_stub.mp4`; `loop_demo.mp4` and the published player now show Claude's replies.

| run | words | actions | Claude's choices | action confidence |
|---|---|---|---|---|
| real 4000 | 15/15 | 3/3 | turn left, go forward, eat | 0.978, 0.996, 0.975 |
| real 4001 | 15/15 | 3/3 | stop, eat, eat | 0.961, 0.994, 0.993 |
| real 4002 | 15/15 | 3/3 | turn left, turn left, eat | 0.974, 0.991, 0.992 |
| shuffled 4000 | 15/15 | 3/3 | turn left, turn left, eat | 0.811, 0.873, 0.910 |

Notes: Claude is not deterministic here. Round 1 has the same sentence in every run (vinegar, fruit, sugar, hot, looming) and Claude answered "turn left" three times and "stop" once; round 2 of real 4000 and shuffled 4000 is also the same sentence, answered "go forward" and "turn left". The word lists per seed are the same as in the stub runs; only the replies differ. Claude disagreed with the stub on 4 of 12 rounds (real 4001: stop/eat/eat vs turn left/turn right/turn right; real 4002 round 2: turn left vs stop; shuffled round 2: turn left vs go forward). On real 4000 Claude made the same three choices as the stub. Since the action code is played into the brain, a different reply also changes the brain state going into the next round, and every run still decoded all 15 words. The shuffled control still passes with lower action confidence.
