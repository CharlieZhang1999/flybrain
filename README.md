# flybrain

A connectome-constrained leaky integrate-and-fire (LIF) simulation of the *Drosophila* brain with a trained readout. The project reproduces chetaslua's "fly brain to ChatGPT" demo on FlyWire v783 and MaleCNS v1.0, then extends it. The full plan is in [docs/plan.md](docs/plan.md).

## Status

- [x] Setup (plan section 1): venv, dependencies, Shiu reference repo, FlyWire v783 data
- [ ] MaleCNS v1.0 data (needs a neuPrint token, see below)
- [ ] Week 1: NumPy LIF vs Brian2 on FlyWire sugar/bitter (MN9)
- [ ] Week 2: MaleCNS port, 16-word readout, shuffled-connectome control
- [ ] Week 3: GPT loop

## Setup

```bash
bash scripts/setup.sh          # venv (Python 3.12), deps, Shiu repo, data links, sanity check
source .venv/bin/activate
cp .env.example .env           # add your neuPrint token (and later the OpenAI key)
```

## Layout

```
sim/          NumPy LIF (FlyWire + MaleCNS)
stimuli/      sensory stimulus definitions
readout/      linear and other decoders
loop/         GPT <-> fly loop
notebooks/    01_shiu_validation, 02_translator, 03_gpt_loop
data/         flywire_v783/ (symlinks into third_party/), malecns/  [gitignored data]
third_party/  Drosophila_brain_model (Shiu et al. 2024 reference)  [gitignored]
results/      figures/ committed, raw output ignored
docs/         plan
scripts/      setup.sh, check_setup.py
```

## Data

- FlyWire v783: `Connectivity_783.parquet` (15.09M edges) and `Completeness_783.csv` (138,639 neurons). Both come from the Shiu repo.
- MaleCNS v1.0: neuPrint dataset `male-cns:v1.0`, or prepared matrices from YijieYin/connectome_data_prep (about 3.5 GB repo).
