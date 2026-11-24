#!/usr/bin/env python
"""W5 T7 Phase A: dose pilot for the hidden-state drive (calib only).

Every episode is background only (stimuli.mixtures.background_episode, class_v2
defaults, bg_hz = BG_HZ, T = 2 s) plus tonic Poisson drive on one state set over
the whole episode (stimuli.states.state_blocks, segment [0, T)).
  baseline  0 Hz, Poisson seeds SEED0 + j, j = 0..N_BASE-1
  driven    hunger (ISN), thirst (BiT), arousal (pC1/P1 male-specific), sleep (ER5)
            at LEVELS = 20 / 50 / 100 / 200 Hz, seeds SEED0 + 0 and SEED0 + 1
            (the same Poisson seeds as baseline j = 0, 1).
The brief asks for 2 seeds per level; the 0 Hz level gets N_BASE = 6 seeds so that
the seed-to-seed SD used by the 3 SD rule is less noisy (the SD of seeds 0-1 alone
is reported too). The 0 Hz episodes are shared by all four sets (no drive).
cfg 'calib' only, sim.cns.W4_CALIB, CPU float32 rng='torch'. One process.

Resumable: an episode whose npz exists, loads and has the same spec is skipped.
Files are written to a temporary name and renamed.

Usage:
  python scripts/w5_state_pilot.py run         # simulate the missing episodes
  python scripts/w5_state_pilot.py summary     # dose table + plot from the npz files
Output (results/w5/hidden_state/):
  state_sets.json            body IDs per state set (selection rules; fixed after Phase A)
  pilot/targets.json         top-50 direct downstream targets per set (neuron index, bodyId, |w|)
  pilot/ep_{name}.npz        t (step, dt 0.1 ms, int32), i (neuron, int32), spec (JSON), wall_s
  pilot/run_log.txt          per-episode wall time (appended)
  pilot_dose.csv / .md       dose table (one row per state x level x seed, plus summary rows)
  pilot_dose.png             dose plot
Metrics (window WIN = [0.5, 2.0) s, after the onset transient):
  set_hz      mean rate of the driven set
  down_hz     mean rate of its top-50 direct downstream targets (ranked by summed |W|
              from the set in the calib model W; set members excluded)
  brain_hz    mean rate over sim.cns.brain_mask neurons (central brain, non-sensory,
              non-motor), excluding the driven set
  frac_200    fraction of those brain neurons with rate > 200 Hz (runaway if > 1 %)
Choice rule (brief task 3): the lowest level with |mean down_hz(level) - mean
down_hz(0 Hz)| >= 3 * sd0 (and > 0), and frac_200 <= 1 % in every seed of that level.
sd0 = max(SD of the 0 Hz down_hz across seeds, SD_FLOOR_HZ). The floor is the rate
resolution of the measure, one spike over the 50 targets in the window
(1 / (50 * 1.5 s) = 0.0133 Hz): the targets are often silent at 0 Hz in every seed,
so the raw SD is 0 and the rule would otherwise accept any change, even 0.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stimuli.mixtures import background_episode  # noqa: E402
from stimuli import states as st  # noqa: E402

HS = ROOT / 'results' / 'w5' / 'hidden_state'
OUT = HS / 'pilot'
SETS_JSON = HS / 'state_sets.json'
CFG = 'calib'
T = 2.0
BG_HZ = 2.0
SEED0 = 71000
N_BASE = 6
N_SEED = 2
LEVELS = (20.0, 50.0, 100.0, 200.0)
DRIVEN = ('hunger', 'thirst', 'arousal', 'sleep')
N_TOP = 50
WIN = (0.5, 2.0)
RUNAWAY_HZ, RUNAWAY_FRAC, N_SD = 200.0, 0.01, 3.0
SD_FLOOR_HZ = 1.0 / (N_TOP * (WIN[1] - WIN[0]))
ANNOT = ROOT / 'data' / 'malecns' / 'body-annotations-male-cns-v1.0-minconf-0.5.feather'


def episodes():
    """[(name, state, level, seed_j)] in run order: baseline first, then by level."""
    eps = [(f'baseline_000_s{j}', 'baseline', 0.0, j) for j in range(N_BASE)]
    for lv in LEVELS:
        for s in DRIVEN:
            for j in range(N_SEED):
                eps.append((f'{s}_{int(lv):03d}_s{j}', s, lv, j))
    return eps


def spec_for(state, level, j):
    spec = background_episode(BG_HZ, SEED0 + j, T=T)
    spec['state_segments'] = st.check_segments([(0.0, T, state, level)], T)
    return spec


def path(name):
    return OUT / f'ep_{name}.npz'


def log(msg):
    print(msg, flush=True)
    with open(OUT / 'run_log.txt', 'a') as f:
        f.write(msg + '\n')


def load_valid(name, spec):
    f = path(name)
    if not f.exists():
        return None
    try:
        d = np.load(f)
        ok = json.loads(str(d['spec'])) == spec and len(d['t']) == len(d['i'])
        return dict(t=d['t'], i=d['i'], wall_s=float(d['wall_s'])) if ok else None
    except Exception:
        return None


def write_sets(meta):
    annot = pd.read_feather(ANNOT, columns=['bodyId', 'type', 'status', 'dimorphism'])
    sets = st.state_sets(annot, DRIVEN)
    ids = set(meta.bodyId.astype(np.int64))
    for s, b in sets.items():
        missing = [x for x in b if x not in ids]
        if missing:
            raise ValueError(f'{s}: {len(missing)} body IDs not in the simulated meta')
    doc = dict(source=ANNOT.name, rules={s: st.STATE_RULES[s] for s in DRIVEN},
               cell_type={s: st.CELL_TYPE[s] for s in DRIVEN},
               n={s: len(b) for s, b in sets.items()}, body_ids=sets)
    if SETS_JSON.exists():
        old = json.loads(SETS_JSON.read_text())
        if old['body_ids'] != {s: list(b) for s, b in sets.items()}:
            raise RuntimeError(f'{SETS_JSON} exists with different body IDs; fixed after Phase A')
        return sets
    SETS_JSON.write_text(json.dumps(doc, indent=1))
    return sets


def write_targets(m, sets, pos):
    f = OUT / 'targets.json'
    tg = {}
    for s, b in sets.items():
        idx = np.array(sorted(pos[x] for x in b))
        ti, tw = st.downstream_targets(m['W'], idx, N_TOP)
        tg[s] = dict(idx=ti.tolist(), bodyId=m['meta'].bodyId.to_numpy()[ti].astype(np.int64).tolist(),
                     abs_w=np.round(tw, 6).tolist(), set_idx=idx.tolist())
    f.write_text(json.dumps(tg))
    return tg


def run():
    import torch
    from sim.cns import cns_model, W4_CALIB
    from sim.lif_numpy import P
    from sim import lif_torch
    from stimuli.mixtures import to_blocks
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [e for e in episodes() if load_valid(e[0], spec_for(*e[1:])) is None]
    log(f'[run] {len(todo)} of {len(episodes())} episodes to simulate')
    if not todo:
        return
    t0 = time.time()
    m = cns_model('cns', CFG, **W4_CALIB)
    pos = {int(b): k for k, b in enumerate(m['meta'].bodyId)}
    log(f'[{CFG}] model built in {time.time() - t0:.1f} s, torch threads {torch.get_num_threads()}')
    sets = write_sets(m['meta'])
    write_targets(m, sets, pos)
    for name, s, lv, j in todo:
        spec = spec_for(s, lv, j)
        blocks = to_blocks(spec, m['meta'], pos) + st.state_blocks(spec['state_segments'], sets, pos)
        p = dict(P); p['t_run'] = T
        t1 = time.time()
        r = lif_torch.run_trial(m['W'], [], params=p, seed=spec['seed'], apl=m['apl'], gap=m['gap'],
                                presyn=m['presyn'], blocks=blocks, record=True, device='cpu',
                                rng='torch', dtype=torch.float32)
        wall = time.time() - t1
        tmp = OUT / f'.tmp_{name}.npz'
        np.savez_compressed(tmp, t=r['t'].astype(np.int32), i=r['i'].astype(np.int32),
                            spec=json.dumps(spec), cfg=CFG, wall_s=wall)
        os.replace(tmp, path(name))
        log(f'[{CFG}] {name}: {len(r["t"]):,} spikes, {wall:.1f} s wall')


def metrics(d, n, set_idx, down_idx, brain_idx):
    k = (d['t'] >= WIN[0] / 1e-4) & (d['t'] < WIN[1] / 1e-4)
    rate = np.bincount(d['i'][k], minlength=n) / (WIN[1] - WIN[0])
    rb = rate[brain_idx]
    return dict(set_hz=rate[set_idx].mean(), down_hz=rate[down_idx].mean(), brain_hz=rb.mean(),
                frac_200=float((rb > RUNAWAY_HZ).mean()), all_hz=rate.mean())


def summary():
    from sim.cns import brain_mask
    meta = pd.read_parquet(ROOT / 'data' / 'malecns' / 'malecns_traced_meta.parquet')
    n = len(meta)
    bm = brain_mask(meta)
    tg = json.loads((OUT / 'targets.json').read_text())
    rows = []
    for name, s, lv, j in episodes():
        d = load_valid(name, spec_for(s, lv, j))
        if d is None:
            continue
        for ss in (DRIVEN if s == 'baseline' else (s,)):
            sidx, didx = np.array(tg[ss]['set_idx']), np.array(tg[ss]['idx'])
            b = bm.copy(); b[sidx] = False
            rows.append(dict(state=ss, level_hz=lv, seed_j=j, file=path(name).name,
                             **metrics(d, n, sidx, didx, np.flatnonzero(b)), wall_s=d['wall_s']))
    df = pd.DataFrame(rows)
    df.to_csv(HS / 'pilot_dose_episodes.csv', index=False)
    tab, chosen = [], {}
    for s in DRIVEN:
        b = df[(df.state == s) & (df.level_hz == 0)]
        mu0, sd0_raw = b.down_hz.mean(), b.down_hz.std(ddof=1)
        sd0 = max(sd0_raw, SD_FLOOR_HZ)
        sd0_2 = max(b[b.seed_j < N_SEED].down_hz.std(ddof=1), SD_FLOOR_HZ)
        for lv, g in df[df.state == s].groupby('level_hz'):
            dlt = g.down_hz.mean() - mu0
            run_away = bool((g.frac_200 > RUNAWAY_FRAC).any())
            moves = lv > 0 and abs(dlt) > 0 and abs(dlt) >= N_SD * sd0
            tab.append(dict(state=s, cell_type=st.CELL_TYPE[s], level_hz=lv, n_seeds=len(g),
                            set_hz=g.set_hz.mean(), down_hz=g.down_hz.mean(), d_down_hz=dlt,
                            sd0_raw=sd0_raw, sd0=sd0, z_down=dlt / sd0, z_down_2seed_sd=dlt / sd0_2,
                            brain_hz=g.brain_hz.mean(), frac_200_max=g.frac_200.max(),
                            moves_3sd=moves, runaway=run_away))
            if moves and not run_away and s not in chosen:
                chosen[s] = lv
    tab = pd.DataFrame(tab)
    tab['chosen'] = [chosen.get(s) == lv for s, lv in zip(tab.state, tab.level_hz)]
    tab.to_csv(HS / 'pilot_dose.csv', index=False)
    fmt = tab.copy()
    for c in ('set_hz', 'down_hz', 'd_down_hz', 'brain_hz'):
        fmt[c] = fmt[c].map(lambda x: f'{x:.2f}')
    for c in ('sd0_raw', 'sd0'):
        fmt[c] = fmt[c].map(lambda x: f'{x:.4f}')
    for c in ():
        fmt[c] = fmt[c].map(lambda x: f'{x:.2f}')
    for c in ('z_down', 'z_down_2seed_sd'):
        fmt[c] = fmt[c].map(lambda x: f'{x:.1f}')
    fmt['frac_200_max'] = fmt['frac_200_max'].map(lambda x: f'{100 * x:.3f} %')
    (HS / 'pilot_dose.md').write_text(fmt.to_markdown(index=False) + '\n\nchosen: ' + json.dumps(chosen) + '\n')
    print(fmt.to_string(index=False))
    print('chosen level per state:', chosen)
    print(f'wall: {df.drop_duplicates("file").wall_s.sum():.0f} s simulation over {df.file.nunique()} episodes')
    plot(df, tab, chosen)
    return tab, chosen


def plot(df, tab, chosen):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    col = dict(hunger='#1f77b4', thirst='#2ca02c', arousal='#d62728', sleep='#9467bd')
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    xs = [0] + list(LEVELS)
    xp = {lv: k for k, lv in enumerate(xs)}
    for s in DRIVEN:
        g = df[df.state == s]
        t = tab[tab.state == s]
        for a, c, lab in ((ax[0], 'down_hz', 'top-50 downstream rate (Hz)'),
                          (ax[1], 'brain_hz', 'brain mean rate (Hz)'),
                          (ax[2], 'frac_200', 'brain fraction > 200 Hz')):
            a.scatter([xp[v] for v in g.level_hz], g[c], s=12, color=col[s], alpha=0.5)
            mean = g.groupby('level_hz')[c].mean()
            a.plot([xp[v] for v in mean.index], mean.values, '-o', ms=4, color=col[s],
                   label=f'{s} ({st.CELL_TYPE[s]})')
            a.set_ylabel(lab)
        if s in chosen:
            y = t[t.level_hz == chosen[s]].down_hz.iloc[0]
            ax[0].scatter([xp[chosen[s]]], [y], s=160, facecolor='none', edgecolor=col[s], lw=2)
    ax[0].set_yscale('symlog', linthresh=1)
    ax[2].axhline(RUNAWAY_FRAC, color='k', ls='--', lw=1)
    ax[2].text(0, RUNAWAY_FRAC, ' runaway (1 %)', va='bottom', fontsize=8)
    for a in ax:
        a.set_xticks(range(len(xs)), [f'{v:g}' for v in xs])
        a.set_xlabel('state drive (Hz, Poisson per neuron)')
    ax[0].legend(fontsize=8)
    ax[0].set_title('circle = chosen level (>= 3 SD, no runaway)', fontsize=9)
    fig.suptitle(f'W5 T7 Phase A dose pilot (calib, bg {BG_HZ:g} Hz class_v2, T {T:g} s, '
                 f'window {WIN[0]}-{WIN[1]} s)', fontsize=10)
    fig.tight_layout()
    fig.savefig(HS / 'pilot_dose.png', dpi=130)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'run'
    {'run': run, 'summary': summary}[cmd]()
