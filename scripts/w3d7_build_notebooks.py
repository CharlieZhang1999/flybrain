#!/usr/bin/env python
"""Week 3 Day 7: build and execute notebooks/01_shiu_validation, 02_translator, 03_gpt_loop.
Every notebook only loads saved results (results/...); none re-runs a simulation.
Run: python scripts/w3d7_build_notebooks.py   (needs nbformat, nbconvert, ipykernel, matplotlib)"""
from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / 'notebooks'; NB.mkdir(exist_ok=True)
SETUP = """from pathlib import Path
import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
R = ROOT / 'results'
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})"""


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


NB01 = [
    md("""# 01 · Week 1: validating the NumPy LIF against Shiu et al. (Brian2)
A connectome-constrained leaky integrate-and-fire (LIF) simulation of the whole FlyWire v630 brain (127,400 neurons), reimplemented in NumPy (`sim/lif_numpy.py`) and compared with the authors' Brian2 code. Stimulus: the 21 sugar gustatory receptor neurons at 150 Hz Poisson for 1 s. Readout: MN9, the proboscis-extension motor neuron.

This notebook only loads saved per-neuron rates; it does not re-run either simulator.
- `results/flywire_sugar_brian2_fresh.npy`: Brian2 (Shiu code), 3 trials × 127,400 neurons, Hz
- `results/flywire_sugar_numpy_b2sched.npy`: NumPy with Brian2's step order, mean of 3 trials, Hz"""),
    code(SETUP),
    code("""ids = pd.read_csv(ROOT/'third_party/Drosophila_brain_model/2023_03_23_completeness_630_final.csv', index_col=0).index.to_numpy()
b2 = np.load(R/'flywire_sugar_brian2_fresh.npy'); npy = np.load(R/'flywire_sugar_numpy_b2sched.npy')
MN9 = 720575940660219265  # left MN9, Shiu et al. readout
i = int(np.flatnonzero(ids == MN9)[0])
print(f'neurons: {len(ids):,}')
print(f'MN9  Brian2: {b2[:, i].mean():.1f} ± {b2[:, i].std():.1f} Hz (3 trials)   NumPy: {npy[i]:.1f} Hz')
print(f'active neurons  Brian2: {(b2.mean(0) > 0).sum()}   NumPy: {(npy > 0).sum()}')"""),
    code("""m = b2.mean(0); act = (m > 0) | (npy > 0)
r = np.corrcoef(m[act], npy[act])[0, 1]; slope = np.polyfit(m[act], npy[act], 1)[0]
fig, ax = plt.subplots(figsize=(4.6, 4.4))
ax.scatter(m[act], npy[act], s=8, alpha=.5, color='#4a7a3d')
ax.scatter([m[i]], [npy[i]], s=50, color='#b0483a', label='MN9', zorder=3)
lim = max(m.max(), npy.max()) * 1.05; ax.plot([0, lim], [0, lim], 'k--', lw=.8)
ax.set(xlabel='Brian2 rate (Hz)', ylabel='NumPy rate (Hz)', title=f'sugar 150 Hz: {act.sum()} active neurons\\nr = {r:.4f}, slope = {slope:.2f}')
ax.legend(frameon=False); plt.tight_layout()"""),
    md("""**Reading it:** the active neurons lie on the identity line (correlation and slope in the title). Week 1's open item was NumPy sugar→MN9 running 13% hot (105.7 vs 93.3 Hz in the Week 1 report). After Week 1 the simulator was changed to Brian2's per-step order and integrator and to block input during refractoriness (`schedule='brian2'`, the default in `run_trial`; commit `03979ff`). This saved pair is the reference used from Week 2 on: MN9 83.4 Hz (NumPy) vs 81.3 ± 0.9 Hz (Brian2, a fresh 3-trial run), so the gap is now about 3%. The Brian2 rate differs from the Week 1 report's 93.3 Hz because it is a separate reference run; compare the two simulators within each pair, not across them.

Week 1 behavioral checks (from `docs/week1_report.html`, before the step-order fix): sugar→MN9 105.7 Hz (Brian2 93.3 ± 3.2, chetaslua's post 78), sugar+bitter→MN9 2.7 Hz (suppressed, as in the paper), water 13 Hz."""),
    code("""c = pd.read_csv(R/'flywire/w2d1_flywire_control.csv')
print('Week 2 starting point: FlyWire v783 run exactly as published (no calibration)')
c.groupby('word')[['stim_active', 'mn9_stim_Hz', 'tail_brain', 'persistent']].mean(numeric_only=False).round(1)"""),
    md("""The last column is why Week 2 existed: in the published model, vinegar and hot leave ~7,900 neurons firing after the stimulus ends (a persistent attractor). Week 2 traced it to the cholinergic eLN→PN chemical loop and replaced it with eLN–PN gap junctions (Yaksi & Wilson 2010). See `02_translator.ipynb`."""),
]

NB02 = [
    md("""# 02 · Week 2: the word translator on MaleCNS
Brain: MaleCNS v1.0 brain-only LIF (monoamines removed, w × 0.581, rate APL). Three versions:
- **base**: as ported.
- **calib**: chemical cholinergic LN→PN = 0 plus eLN–PN gap junctions (c0 = 0.2, 2 mV spikelets). Removes the persistent attractor.
- **shuffle**: calib with the chemical matrix shuffled, keeping each neuron's in/out degree and output sign.

17 words × 10 Poisson seeds, 500 ms at 150 Hz. Readout: L2 logistic regression on log(1 + spike count) per neuron. *Learned:* only this readout. The connectome is never trained.

Loads `results/readout/readout_summary.json` and `trials_*.npz`; does not re-run the 510 trials."""),
    code(SETUP),
    code("""s = json.load(open(R/'readout/readout_summary.json')); W = s['words']
rows = []
for k, v in s['results'].items():
    model, win, feat = k.split('/')
    rows.append(dict(model=model, window={'stim': '0-0.5 s (stimulus)', 'post': '0.5-1.0 s (after)'}[win],
                     features={'all': 'all non-sensory', 'motor': 'DNs + MNs'}[feat],
                     test_split=v['acc_split'], leave_one_seed_out=v['acc_loso'], n_features=v.get('n_features')))
t = pd.DataFrame(rows); print(f"chance = {s['chance']:.3f}")
t.pivot_table(index=['window', 'features'], columns='model', values='leave_one_seed_out').round(3)"""),
    md("""**The main lesson of Phase 1:** the shuffled brain also decodes all 17 words at 100% from all neurons. A linear readout of a driven network works with almost any wiring. The real wiring shows up only in the motor-only readout and after the stimulus ends."""),
    code("""fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, k in zip(axs, ['calib/stim/motor', 'shuffle/stim/motor', 'shuffle/post/motor']):
    C = np.array(s['results'][k]['confusion']); ax.imshow(C, cmap='Greens')
    ax.set_xticks(range(17), W, rotation=90, fontsize=7); ax.set_yticks(range(17), W, fontsize=7)
    ax.set_title(f"{k}  (LOSO {s['results'][k]['acc_loso']:.0%})", fontsize=9); ax.set_xlabel('decoded'); ax.set_ylabel('true')
plt.tight_layout()"""),
    code("""fig, ax = plt.subplots(figsize=(7, 3.2))
for m, col in [('base', '#999'), ('calib', '#4a7a3d'), ('shuffle', '#b0483a')]:
    d = np.load(R/f'readout/trials_{m}.npz', allow_pickle=True)
    pop = d['pop']; t = np.arange(pop.shape[1]) * 10
    ax.plot(t, pop.mean(0), color=col, label=m)
ax.axvspan(0, 500, color='#f2b13c', alpha=.12); ax.set(xlabel='ms', ylabel='spikes / 10 ms (mean of 170 trials)', yscale='log')
ax.legend(frameon=False); ax.set_title('Population activity: stimulus 0-500 ms, then silence'); plt.tight_layout()"""),
    md("""Interactive version with the brain graph, heat maps and these confusion matrices: the Week 2 Connectome Explorer (`artifact/connectome_explorer.html`)."""),
]

NB03 = [
    md("""# 03 · Week 3: the closed loop between the fly brain and an LLM
A connectome-constrained LIF simulation with trained readouts, wired to a language model:
1. **Words in:** 5 stimuli back to back (500 ms each) drive sensory neurons of one continuous simulation.
2. **Fly → LLM:** the word readout decodes each 500 ms window; decoded words become a first-person sentence ("I smell vinegar. ... What should I do?").
3. **LLM → fly:** the LLM picks one of five actions; the action is played as a spike code on Johnston's organ (JO) hearing neurons.
4. **Read back:** a readout of descending neurons (DNs) only decodes the action.

Brain: Week 2 calibrated MaleCNS + presynaptic modulation of sensory terminals (K = 0.05) + right-JO mirror. Loads saved results only."""),
    code(SETUP + "\nL = R/'loop'"),
    md("""## Day 1: carry-over between words
40 episodes of 5 words, spacing 0, 250 or 500 ms. The original calibrated model (left) vs the presynaptic-modulation model (right): with presynaptic inhibition of sensory terminals, a readout trained on isolated words reads episodes perfectly."""),
    code("""a = pd.read_csv(L/'w3d1_carryover.csv'); b = pd.read_csv(L/'w3d1_carryover_presyn0.05.csv')
show = lambda d: d[d.readout.isin(['isolated', 'episode-trained 5-fold'])].pivot_table(index=['readout', 'features'], columns='gap_ms', values='acc')
print('calibrated (Week 2)'); display(show(a).round(3)); print('+ presynaptic modulation K=0.05')
display(b[['readout', 'features', 'acc_test_eps']].round(3))"""),
    md("""## Day 3-4: action codes on the hearing neurons
Five codes on JO-A/JO-B neurons (left, right, both, pulsed). Day 4 test: each code played after a 5-word episode, 37 held-out trials."""),
    code("""sc = json.load(open(L/'w3d4_score.json'))
pd.DataFrame([dict(readout=k, correct=f"{v['correct']}/{v['n']}", neurons_that_fire=v['features_active'], errors='; '.join(v['errors']))
              for k, v in sc.items() if isinstance(v, dict)])"""),
    md("""The shuffled brain reads the codes back almost as well (36/37), but through 24 DNs instead of 445: the real wiring routes the hearing input to many more descending neurons. Reading back a stimulus you injected yourself works with almost any wiring."""),
    md("""## Day 5-6: the full loop
Three rounds in one 10.5 s simulation. `w3d6` runs use an offline rule-based stand-in for the LLM; `w3d6c` runs call Claude (claude-sonnet-5-5) live through the Claude Code CLI."""),
    code("""rows = []
for tag, llm in [('w3d6', 'offline stand-in'), ('w3d6c', 'Claude')]:
    for f in sorted(L.glob(f'{tag}_loop_*[0-9].json')):
        d = json.load(open(f))
        rows.append(dict(llm=llm, run=f"{d['cfg']} {d['seed']}", words=f"{sum(r['words_correct'] for r in d['rounds'])}/15",
                         actions=' / '.join(r['action'] for r in d['rounds']),
                         read_back=f"{sum(r['action_ok'] for r in d['rounds'])}/3",
                         confidence=' '.join(f"{r['action_p']:.2f}" for r in d['rounds'])))
pd.DataFrame(rows)"""),
    code("""d = json.load(open(L/'w3d6c_loop_real_4000.json'))
pop = np.array(d['pop_10ms']); t = np.arange(len(pop)) / 100
fig, ax = plt.subplots(figsize=(11, 3.2)); ax.plot(t, pop, color='#6a5fb8', lw=.8)
for r in d['rounds']:
    for k, w in enumerate(r['decoded']):
        ax.text(r['t_start'] + k*.5 + .25, pop.max()*1.02, w, ha='center', fontsize=7, rotation=90, va='bottom')
    ax.axvspan(r['t_words'], r['t_words'] + .5, color='#d04a3a', alpha=.15)
    ax.text(r['t_words'] + .25, pop.max()*1.02, f"{r['action']}\\n({r['code']})", ha='center', fontsize=7, color='#d04a3a', va='bottom')
ax.set(xlabel='brain time (s)', ylabel='spikes / 10 ms', xlim=(0, d['t_total'])); plt.tight_layout()
for r in d['rounds']: print(f"round {r['round']}: {r['prompt']!r} -> Claude: {r['llm_raw']!r} -> DNs read {r['action_decoded']} ({r['action_p']:.2f})")"""),
    md("""## What is deterministic, what is learned
- **Deterministic:** connectome weights, LIF equations, the Week 2 calibration, presynaptic modulation, the right-JO mirror, stimulus sets and rates, the action codes.
- **Learned:** only the two linear readouts (words: all non-sensory neurons; actions: DNs only). The connectome is never trained.
- **Not deterministic:** the LLM. The identical round-1 sentence got "turn left" three times and "stop" once from Claude.
- **What the shuffled brain also does:** 15/15 words and 3/3 actions in the loop, with lower action confidence (0.81-0.93).

Playback: `artifact/loop/index.html` (split-screen player) and `results/figures/loop_demo.mp4`."""),
]


def build(name, cells):
    nb = nbf.v4.new_notebook(); nb.cells = cells
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    ExecutePreprocessor(timeout=300, kernel_name='python3').preprocess(nb, {'metadata': {'path': str(NB)}})
    nbf.write(nb, NB / name); print('wrote', NB / name)


build('01_shiu_validation.ipynb', NB01)
build('02_translator.ipynb', NB02)
build('03_gpt_loop.ipynb', NB03)
