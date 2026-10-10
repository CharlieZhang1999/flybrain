# Fly brain to ChatGPT: reproduction and MaleCNS upgrade plan

Date: 2026-10-09. Companion to `BCI hands on learning roadmap.md`.

## 0. What chetaslua actually built (from the X post of 11 Sep 2026 and the 36kr write-up)

No code was released. Everything below is reconstructed from the public descriptions.

| Component | What they did | Source |
|---|---|---|
| Connectome | MaleCNS v1.0 (Janelia/Cambridge/Google, Cell, 3 Sep 2026). 138k neurons, 15M connections after thresholding weak synapses. | 36kr |
| Simulator | Leaky integrate-and-fire (LIF), NumPy only, one timestep at a time, on a MacBook. Same model family as Shiu et al. 2024. | X post, 36kr |
| Validation | Stimulate sugar gustatory receptor neurons: MN9 (proboscis motor neuron) fires ~78 Hz. Add bitter GRNs: drops to ~3 Hz. "Same direction as the paper" (Shiu et al. 2024). | X post |
| Stimuli vocabulary | 16 sensory conditions (sugar, bitter, vinegar, yeast, warm, bright, wall ahead, ...). Each is a set of sensory neurons driven at a fixed rate. | X post, 36kr |
| Translator | Linear readout from firing-rate vectors of projection + motor neurons (sensory neurons excluded) to 16 word classes. 100% held-out on 16 words. Motor-only readout: 85%. | 36kr |
| Fly to GPT | Decoded words concatenated into a prompt ("i smell vinegar and yeast. i taste sugar. ... what should i do?") and sent to ChatGPT. | X post |
| GPT to fly | ChatGPT's answer ("Turn left") is encoded as stimulation of auditory (Johnston's organ) neurons; a readout verified 37/37 correct. | 36kr |

Honest read: this is a 16-class classifier on top of a deterministic simulation whose inputs are 16 distinct stimulus sets. Linearly separable inputs produce linearly separable rate vectors, so 100% is expected. The scientifically meaningful part is the Shiu validation. The ChatGPT part is presentation. Your Phase 1 reproduces all of it; your Phase 2 makes the decoding genuinely hard and the loop genuinely closed.

Corrections to note in your own write-up:
- FlyWire 139,255 neurons = Dorkenwald et al. 2024 (Nature). Shiu et al. 2024 (Nature) = the LIF simulation on FlyWire v630/v783.
- MaleCNS is not Google's dataset. Google Research contributed segmentation; the connectome is HHMI Janelia FlyEM + Cambridge. CC-BY.

## 1. Setup (day 1)

Hardware: your Mac is enough for Phase 1. A 24 GB GPU helps for Phase 2 sweeps but is not required.

```bash
mkdir -p ~/Desktop/Workspace/Neuro/flybrain && cd ~/Desktop/Workspace/Neuro/flybrain
python3.12 -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas pyarrow scikit-learn matplotlib brian2 jupyter
pip install navis neuprint-python openai   # neuprint for MaleCNS, openai for the GPT loop
```

Data to download:
- FlyWire v783 for the Shiu reference: `2025_Connectivity_783.parquet` (97 MB) and `2025_Completeness_783.csv` from Zenodo record 10676866 or codex.flywire.ai.
- Shiu reference code: `git clone https://github.com/philshiu/Drosophila_brain_model`
- MaleCNS v1.0: neuPrint server at https://neuprint.janelia.org (dataset `male-cns:v1.0`, needs a free Google login for a token), or the prepared sparse matrices in https://github.com/YijieYin/connectome_data_prep. Tutorial: https://github.com/sjcabs/fly_connectome_data_tutorial.
- Cell-type and neurotransmitter annotations ship with both datasets. You need: GRN sugar/bitter IDs, MN9, JO auditory neurons, olfactory receptor neurons by glomerulus (DM1 for vinegar-like, DM2/VA2 yeast-like), thermosensory and photoreceptor IDs, descending neurons (DNs), and for Phase 2 the VNC leg and wing motor neurons.

## 2. Phase 1: reproduce chetaslua (weeks 1 to 3)

### Week 1. Shiu validation on FlyWire (ground truth first)
1. Run `example.ipynb` in philshiu's repo as-is (Brian2, CPU, under an hour). Activate sugar GRNs at 200 Hz, confirm MN9 spikes. Add bitter GRNs, confirm suppression.
2. Reimplement the same LIF in pure NumPy (this is the core of what chetaslua did):
   - Load parquet, threshold synapses (Shiu uses count >= 5), build a CSR sparse weight matrix with signed weights from neurotransmitter prediction (ACh +, GABA/Glu -).
   - LIF per Shiu: dt 0.1 ms, tau_m 20 ms, V_rest -52 mV, V_th -45 mV, refractory 2.2 ms, synaptic weight scale 0.275 mV, exponential synapse tau 5 ms. Check `Drosophila_brain_model/model.py` for the exact constants and copy them.
   - Poisson-drive the stimulated neurons. Record spike counts per neuron per 50 ms bin.
3. Milestone: your NumPy sim and Brian2 agree on MN9 rate within 10% for sugar, sugar+bitter, and no stimulus. Save the comparison figure. This is the one result in the whole project that a neuroscientist will respect.
4. Speed: a 139k-neuron sparse matvec at dt 0.1 ms is ~10k steps per biological second. Expect 1 to 3 minutes per biological second in NumPy. Fine for 16 stimuli. Later swap `scipy.sparse` for MLX or PyTorch MPS (see Kisame76/drosophila-brain-mlx, 0.29 s per bio-second on M4 Pro).

### Week 2. Port to MaleCNS and build the 16-word translator
1. Pull MaleCNS v1.0 connectivity via neuPrint (`fetch_adjacencies` for all neurons with a synapse threshold) or load the prepared matrix from connectome_data_prep. Start with the brain-only subset (~138k neurons) to match chetaslua. Keep the full CNS for Phase 2.
2. Repeat the Shiu test on MaleCNS. MN9 and the GRNs are annotated in MaleCNS. Rates will differ from FlyWire (male brain, different synapse counts). Record both numbers. This is already a small novel data point: sugar/bitter MN9 behaviour across the two connectomes.
3. Define 16 stimuli as sets of sensory neurons plus drive rates. Suggested: sugar, bitter, water, salt-low, salt-high, vinegar (DM1 ORNs), yeast (DM2 and VA2 ORNs), warm (hot cells), cold (cold cells), bright (R1-R6 photoreceptors), dark (none), wind-left, wind-right (JO-CE), loud (JO-AB), wall-ahead (looming via lobula columnar LC4/LPLC2), touch (leg mechanosensors if using CNS). Each stimulus: 500 ms of drive, record 10 trials with different Poisson seeds.
4. Features: mean firing rate per non-sensory neuron over the last 300 ms. Train a linear readout (logistic regression, L2) on 7 trials per class, test on 3 held-out trials. Also train motor-only (DNs + MNs). Expect close to 100% and 85%, like chetaslua.
5. Milestone: confusion matrix for all-neuron and motor-only readouts, plus a shuffled-connectome control (degree-preserving shuffle of the weight matrix). If the shuffled brain also gives 100%, say so. It probably will, and that is the most important lesson of Phase 1.

### Week 3. Close the ChatGPT loop
1. Fly to GPT: run a multi-stimulus episode (vinegar + yeast + sugar + warm + bright), decode each 500 ms window to a word, template into a prompt, send to the OpenAI API (or Claude). Log the reply.
2. GPT to fly: map a small action vocabulary ("turn left", "turn right", "go forward", "stop", "eat") to stimulation patterns on JO auditory neurons (e.g., distinct frequency codes on JO-A vs JO-B populations). Train a second readout from DN activity back to the action. Verify 37/37 style accuracy on held-out trials.
3. Record a 60-second screen capture like the original. Write a README that states plainly what is deterministic, what is learned, and what the shuffled control showed.

Deliverable: `flybrain/` repo with `sim/` (NumPy LIF, both connectomes), `stimuli/`, `readout/`, `loop/`, `notebooks/01_shiu_validation.ipynb`, `02_translator.ipynb`, `03_gpt_loop.ipynb`, results figures, README.

## 3. Phase 2: make it more impressive and more honest (weeks 4 to 12)

Each item below beats the original on a dimension that matters, not just neuron count.

### 2A. Full CNS: read real motor output, not a word lookup (weeks 4 to 6)
- Use all 166,700 MaleCNS neurons including the ventral nerve cord. Readout targets become actual leg motor neuron pools (per leg: coxa, femur, tibia, tarsus MN groups) and wing MNs instead of 16 arbitrary words.
- New decoding task: from brain-only activity, predict VNC motor neuron pool rates 100 ms ahead. Metric: R2 per pool. This is a regression on a 140k to 25k mapping and is not trivially separable.
- Headline claim you can make: "first open NumPy simulation of the complete male fly CNS with brain-to-VNC motor readout." Check cobanov/awesome-fly first to confirm nobody has shipped this with code; as of the October 2026 notes only Eon (unreleased body coupling) and Lulzx/fly-brain (165k LIF in WASM) are close.
- Compute: 166k neurons and ~25M edges, dt 0.1 ms. Use PyTorch sparse on MPS or CUDA. Target: under 10 s per biological second.

### 2B. Make decoding hard, then win it (weeks 6 to 8)
Chetaslua's 100% comes from clean, separable inputs. Make the problem realistic:
1. Mixtures and intensities: stimulate 2 to 4 modalities at once at random intensities; decode each modality's intensity (multi-label regression), not a single word.
2. Noise: background Poisson drive to all sensory neurons at 1 to 5 Hz, trial-to-trial jitter in stimulus onset.
3. Temporal: decode from 50 ms windows instead of 300 ms, so dynamics matter.
4. Readout from a random 1% subset of neurons, like a real electrode would see.
Then compare decoders on the same splits: linear, 1D CNN over time, small transformer (POYO-style tokenization of spikes via torch_brain), and an LLM used only as a hypothesis merger over top-k decoder outputs (the approach that won Brain-to-Text 2024). Report where deep models beat linear and where they don't. Include shuffled-connectome and shuffled-label controls in every table.

### 2C. Real closed loop with a body (weeks 8 to 10)
- Couple the CNS sim to FlyGym 2.x (NeuroMechFly, MuJoCo, `pip install flygym[rl]`). VNC leg MN pool rates drive the 42 leg joint actuators via a learned linear map (or the CPG controller's phase/amplitude).
- Task: vinegar plume at a location in the arena. Olfactory input comes from the body's position in the plume, not from a script. Success = fly reaches source.
- Where RL enters: the connectome weights stay fixed. Train only (a) per-neuron gains on ~2k descending neurons and (b) the MN-to-actuator map, with PPO (Stable-Baselines3 ships with flygym[rl]). This is the "train-your-fly" paradigm with a full CNS instead of a subcircuit. Compare against a shuffled-connectome substrate with the same trainable parameter count. If the real wiring learns faster, that is a result worth a short arXiv note; Jin et al. 2026 (arXiv 2602.17997) claimed this with a graph controller but released no code.

### 2D. Fly agent that talks to an LLM for a reason (weeks 10 to 12)
Replace the "send words to ChatGPT" gimmick with an LLM that does something useful:
- The LLM is an experimenter agent. Give it tools: `stimulate(neuron_ids, rate, duration)`, `record(neuron_ids)`, `shuffle_control()`, `fit_readout()`. Task: "find the minimal set of sensory neurons whose stimulation drives MN9 above 50 Hz" or "which descending neurons are necessary for the vinegar-approach behaviour (ablation)." Score it against brute-force answers.
- This connects to the Phase 2 flagship in the main roadmap (an MCP-tooled neural-data agent) and to the documented gap that current LLM agents fail at end-to-end neural-data analysis.

## 4. Milestones and time

| Week | Milestone | Proof |
|---|---|---|
| 1 | NumPy LIF matches Brian2 on FlyWire sugar/bitter | MN9 rate comparison figure |
| 2 | MaleCNS brain sim, 16-word linear readout, shuffled control | Confusion matrices x3 |
| 3 | Fly to GPT to fly loop reproduced | 60 s video, README |
| 6 | Full 166.7k CNS sim, brain to VNC motor-pool regression | R2 table, speed benchmark |
| 8 | Hard-decoding benchmark, linear vs CNN vs transformer vs LLM-merge | Results table with controls |
| 10 | Closed-loop plume navigation in FlyGym with RL-tuned DN gains | Success-rate curve, real vs shuffled |
| 12 | LLM experimenter agent on the fly sim | Agent vs brute-force table, arXiv note draft |

Roughly 10 to 15 hours per week. Weeks 1 to 3 are the reproduction; stop there if you only want the demo. Weeks 4 to 12 produce something no one has published with code.

## 5. Pitfalls

- Synapse threshold and neurotransmitter sign conventions dominate results. Log them in every run. Shiu used count >= 5 on FlyWire; MaleCNS synapse counts are on a different scale, so re-tune so that unstimulated baseline activity stays near zero and sugar still drives MN9.
- MaleCNS neuPrint IDs differ from FlyWire root IDs. Map cell types by name (hemibrain-style type labels), never by ID.
- Do not claim "the fly brain talks" or "intention decoding." Say "connectome-constrained LIF simulation with a trained readout." Patrick Mineault's critique of the viral fly demos (neuroai.science, "Are flies playing Beat Saber") is the standard your README should survive.
- Always ship the shuffled-connectome control. It is what separates your repo from the dozens in awesome-fly.

## 6. Key links

- Shiu et al. 2024 code: https://github.com/philshiu/Drosophila_brain_model
- FlyWire v783 data: https://zenodo.org/records/10676866 and https://codex.flywire.ai
- MaleCNS portal and neuPrint: https://male-cns.janelia.org , https://neuprint.janelia.org
- Prepared connectome matrices: https://github.com/YijieYin/connectome_data_prep
- Data access tutorial: https://github.com/sjcabs/fly_connectome_data_tutorial
- Apple Silicon LIF with shuffled control: https://github.com/Kisame76/drosophila-brain-mlx
- Browser full-CNS LIF + body: https://github.com/Lulzx/fly-brain
- Trainable gains paradigm: https://github.com/eudald-seeslab/train-your-fly
- FlyGym 2.x: https://neuromechfly.org , https://github.com/NeLy-EPFL/flygym
- torch_brain / POYO: https://github.com/neuro-galaxy/torch_brain
- Index of fly projects: https://github.com/cobanov/awesome-fly
- Original post: https://x.com/chetaslua/status/2098387725972123825 ; write-up: https://eu.36kr.com/en/p/3982957231995910
