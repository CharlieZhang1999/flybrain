# flybrain

A **connectome-constrained leaky integrate-and-fire (LIF) simulation of the *Drosophila* brain with a trained readout**, wired into a closed loop with a language model. It reproduces chetaslua's "fly brain to ChatGPT" demo on FlyWire and on the MaleCNS v1.0 male connectome, with the controls that demo did not ship. The full plan is in [docs/plan.md](docs/plan.md).

This is not a fly that talks, and nothing here decodes intentions. Every neuron is the same two-variable LIF unit and all biology lives in the wiring. Words are stimuli we inject into annotated sensory neurons; "what the fly says" is the output of a linear classifier we trained on its spikes. The shuffled-connectome control below shows how much of the result comes from the wiring, and how much does not.

## Status

- [x] Setup: venv, dependencies, Shiu et al. reference repo, FlyWire v630/v783 data, MaleCNS v1.0 data
- [x] **Week 1:** NumPy LIF matches the Shiu et al. Brian2 model on FlyWire (sugar → MN9 feeding, bitter suppression). [Report](docs/week1_report.html), [notebook 01](notebooks/01_shiu_validation.ipynb)
- [x] **Week 2:** MaleCNS port; the persistent attractor fixed by modelling eLN–PN coupling as gap junctions; 17-word readout; shuffled-connectome control; Connectome Explorer. [Summary](progress/week2/week2_summary.md), [notebook 02](notebooks/02_translator.ipynb)
- [x] **Week 3:** closed loop between the fly brain and an LLM: words in, a sentence to the LLM, the chosen action played back as a hearing-neuron spike code and read back from descending neurons. [Report](docs/week3_report.html), [notebook 03](notebooks/03_gpt_loop.ipynb), [video](results/figures/loop_demo.mp4)

## Headline numbers (Week 3, MaleCNS)

| Test | Real connectome | Shuffled connectome |
|---|---|---|
| Words decoded inside 5-word episodes (all non-sensory neurons) | 100% | 100% |
| Action codes read back from descending neurons only, 37 held-out trials after a word episode | **37 / 37** (445 DNs fire) | 36 / 37 (24 DNs fire) |
| Full loop with Claude choosing the actions (3 runs real, 1 shuffled; 3 rounds each) | 45/45 words, 9/9 actions, confidence 0.96–1.00 | 15/15 words, 3/3 actions, confidence 0.81–0.91 |

## What is deterministic, what is learned, and what the shuffled brain also does

**Deterministic (fixed, never fitted to the task):**
- The connectome: MaleCNS v1.0 brain-only (or FlyWire v630/v783), synapse counts × neurotransmitter sign, global weight scale 0.581, monoaminergic neurons removed.
- The LIF equations and parameters of Shiu et al. 2024, with Brian2's per-step order.
- The Week 2 calibration: chemical cholinergic LN→PN synapses set to 0 and replaced by eLN–PN gap junctions (Yaksi & Wilson 2010).
- Presynaptic modulation of sensory terminals: GABA-A/B input onto sensory axon terminals scales their release; it never drives the sensory neuron itself.
- The right Johnston's-organ mirror, which copies the left-side JO wiring onto the right side to repair an incomplete reconstruction.
- Stimulus sets (annotated sensory neuron IDs per word), rates (150 Hz Poisson, 500 ms), and the five action spike codes.

**Learned (the only trained parts):**
- The **word readout**: L2 logistic regression on log spike counts of all non-sensory neurons, 17 words + `none`.
- The **action readout**: L2 logistic regression on log spike counts of descending neurons only, 5 actions + `none`.
- The connectome weights are never trained.

**Not deterministic:** the LLM. Called live, Claude answered the identical opening sentence with "turn left" three times and "stop" once.

**What the shuffled brain also does** (chemical matrix shuffled, keeping each neuron's in/out degree and output sign):
- Decodes all 17 words at 100% from all neurons (Week 2), and 15/15 words inside the loop.
- Reads the action codes back at 36/37 from descending neurons, and 3/3 in the loop.

So a linear readout of a driven network works with almost any wiring. The real wiring shows up in narrower places: the motor-only word readout (94% real vs 85% shuffled), words decoded *after* the stimulus ends (68% vs 21% from motor neurons), how many descending neurons a hearing code reaches (445 vs 24), and the action-readout confidence in the loop.

## Known limits

- MaleCNS here is brain only, with no ventral nerve cord, so descending neurons are the motor readout and sugar → MN9 is near zero.
- Wind and looming still leave a tail of activity outside the antennal lobe after the stimulus.
- The action codes are arbitrary spike patterns on hearing neurons, learned by the readout. They are not the fly's natural steering commands. Right-side JO codes only work with the mirror fix.

## Setup

```bash
bash scripts/setup.sh          # venv (Python 3.12), deps, Shiu repo, data links, sanity check
source .venv/bin/activate
# optional .env: NEUPRINT_TOKEN, OPENAI_API_KEY or ANTHROPIC_API_KEY
```

Run one loop (about 5–7 min on one CPU, using save/resume):

```bash
python scripts/w3d5_loop.py real 4000 stub w3d6          # offline rule-based stand-in for the LLM
CLAUDE_CLI_MODEL=claude-sonnet-5-5 \
python scripts/w3d5_loop.py real 4000 claude-cli w3d6c   # Claude through the Claude Code CLI (your Claude login)
python scripts/w3d6_export_loop.py w3d6c                 # data for the split-screen player
```

`loop/llm.py` also supports `openai` and `anthropic` providers with an API key in `.env`.

## Layout

```
sim/          NumPy LIF (lif_numpy.py), gap junctions, presynaptic modulation, JO mirror, MaleCNS loader
stimuli/      word → sensory neuron mappings (FlyWire, MaleCNS)
readout/      linear decoders (Week 2)
loop/         fly ↔ LLM loop: word and action readouts, prompt, LLM client, action codes, runner
notebooks/    01_shiu_validation, 02_translator, 03_gpt_loop (load saved results; no long simulations)
artifact/     Connectome Explorer (Week 2) and split-screen loop player (Week 3)
docs/         plan, Week 1 and Week 3 reports
progress/     day-by-day results for each week
results/      figures/ committed (incl. loop_demo.mp4), raw output ignored
scripts/      setup and one script per experiment (w2d*, w3d*); w3d7_build_notebooks.py rebuilds the notebooks
data/         FlyWire annotations, malecns/  [gitignored data]
third_party/  Drosophila_brain_model (Shiu et al. 2024 reference)  [gitignored]
```

## Data

- FlyWire v630 / v783: connectivity and completeness tables from the Shiu et al. repo.
- MaleCNS v1.0: neuPrint dataset `male-cns:v1.0`, or prepared matrices from YijieYin/connectome_data_prep.

## References

- Shiu et al. 2024, *A Drosophila computational brain model reveals sensorimotor processing*, Nature. Code: https://github.com/philshiu/Drosophila_brain_model
- Yaksi & Wilson 2010, *Electrical coupling between olfactory glomeruli*, Neuron.
- Patrick Mineault, "Are flies playing Beat Saber?" (neuroai.science): the critique this README is written to survive.
