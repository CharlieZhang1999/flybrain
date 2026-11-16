# Week 3 Day 2 results: fly → LLM (10 Nov 2026)

Brain model: calibrated MaleCNS + presynaptic modulation of sensory terminals, K = 0.05 (`presynaptic_sensory_model.md`).
Code: `loop/words_readout.py`, `loop/prompt.py`, `loop/llm.py`, `loop/episode.py`; driver `scripts/w3d2_fly_to_llm.py`.
Log: `results/loop/w3d2_fly_to_llm.jsonl`. One JSON line per episode: true words, decoded words, top probabilities, prompt, system message, LLM provider/model/raw reply/action, neurons still firing at the end, and population spikes in 10 ms bins (for the Explorer's Loop view on Day 6).

## What was built
- **Word readout** (`loop/words_readout.py`): trained once on 170 isolated word trials + 10 no-drive trials of this brain model, using all 4,704 non-sensory neurons that fire, and saved to `results/loop/words_readout_presyn0.05.pkl`. It is 100% correct on the 200 Day 1 episode windows. Windows decoded as `none`, or with top probability < 0.5, are left out of the prompt.
- **Prompt** (`loop/prompt.py`): one first-person phrase per decoded word, by modality ("I smell vinegar.", "It feels hot.", "Something is looming toward me."), then "What should I do?". The system message says the model is a fruit fly and must pick one of: turn left, turn right, go forward, stop, eat.
- **LLM client** (`loop/llm.py`): OpenAI (`OPENAI_API_KEY`, `OPENAI_MODEL`) or Claude (`ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, default claude-sonnet-5-5). The answer is forced through a tool call whose only argument is an enum of the five actions, at temperature 0. The raw reply is logged, and anything outside the list becomes `stop`. With no key it falls back to an offline rule-based stub (danger → turn away or stop, food → eat, attractive smells → go forward), so everything runs without network access.
- **Episode runner** (`loop/episode.py`): plays the words back to back (500 ms each at 150 Hz, no gaps) in one continuous simulation using the Day 1 `blocks=` drive, decodes each window, builds the prompt and asks the LLM.

## Results (offline stub; no API key set yet)
**Demo episode from plan.md:** vinegar, ripe fruit (yeast), sugar, hot (warm) and looming (standing in for "bright"), seeds 2000–2009.

| | Result |
|---|---|
| Words decoded correctly | **50 / 50** (10 of 10 episodes perfect) |
| Lowest top probability | 0.998 |
| Prompt | "I smell vinegar. I smell ripe fruit. I taste sugar. It feels hot. Something is looming toward me. What should I do?" |
| Stub reply | turn left (looming takes priority) |
| Neurons still firing at the end | 406–1,165 (the looming tail outside the antennal lobe) |
| Compute | 80–97 s per 3 s episode with 4 running in parallel |

**Six short scenarios** (seed 3000; all words decoded correctly, top probability ≥ 0.992):

| Prompt | Stub reply |
|---|---|
| I taste sugar. I taste water. | eat |
| I taste something bitter. I smell mould. | stop |
| I smell vinegar. I smell another fly. | go forward |
| I smell CO2. It feels hot. | stop |
| I hear a sound. I feel wind. | stop |
| I smell ripe fruit. Something is looming toward me. | turn left |

## Still to do
- **Real LLM call:** either run `python scripts/w3d2_fly_to_llm.py demo 2000 2010 claude-cli` on a machine where Claude Code is logged in (no API key needed), or add `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` to `flybrain/.env` and use `openai` / `anthropic`. Nothing else changes. The `openai` package is in requirements.txt; `anthropic` needs `pip install anthropic`.
- The stub is a placeholder for testing. Its replies say nothing about the fly. Only the decoding is a result.
- The looming tail (400–1,200 neurons after the episode) will still be running when the action code starts on Day 3. The action readout must be trained with that history present.
