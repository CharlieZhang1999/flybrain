#!/usr/bin/env python
"""Checkpoint 5, Stage A: compact antennal-lobe lesion set.

Base (same as ckpt4 attractor localisation): brain-only network, monoamines
removed, w_syn = 0.275 mV x 0.581, rate-unit APL k = 0.1, seed 0.
Trial: 1.0 s stimulus at 150 Hz, 0.5 s silence; tail window = 1.2-1.5 s.

Conditions (each a sim.lesion config, applied to a copy of the base matrix):
  A baseline                      no change
  B lLN1_bc_out0                  lLN1_bc outputs -> 0
  C v2LN30_out0                   v2LN30 outputs -> 0
  D v2LN30_inhib                  v2LN30 outputs forced inhibitory
  E achLN_to_LN0                  cholinergic AL LN -> AL LN edges -> 0
  F achLN_to_PN0                  cholinergic AL LN -> AL PN edges -> 0
  G gabaLN_out0 (neg. control)    GABAergic AL LN outputs -> 0
Outputs: results/malecns/ckpt5A_scorecard.csv, ckpt5A_runs.csv, ckpt5A_configs.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.malecns import load  # noqa: E402
from sim.lif_numpy import run_trial, P  # noqa: E402
from sim.lesion import apply, select  # noqa: E402
from stimuli.malecns_words import WORDS  # noqa: E402
from scripts.ckpt2_ab import make_apl, stim_idx  # noqa: E402
from scripts.ckpt3_sweep_ablate import regions, mask_weights  # noqa: E402

OUT = ROOT / 'results' / 'malecns'
SCALE, K_APL, DT = 0.581, 0.1, P['dt']
INPUTS = list(WORDS) + ['sugar+bitter']
ACH_LN = "`class` == 'ALLN' and nt == 'acetylcholine'"
GABA_LN = "`class` == 'ALLN' and nt == 'gaba'"
LN = "`class` == 'ALLN'"
PN = "`class` == 'ALPN'"

CONDITIONS = {
    'A_baseline':      [],
    'B_lLN1_bc_out0':  [dict(op='silence_out', group="type == 'lLN1_bc'")],
    'C_v2LN30_out0':   [dict(op='silence_out', group="type == 'v2LN30'")],
    'D_v2LN30_inhib':  [dict(op='set_sign', group="type == 'v2LN30'", sign=-1)],
    'E_achLN_to_LN0':  [dict(op='scale_edges', pre=ACH_LN, post=LN, alpha=0.0)],
    'F_achLN_to_PN0':  [dict(op='scale_edges', pre=ACH_LN, post=PN, alpha=0.0)],
    'G_gabaLN_out0':   [dict(op='silence_out', group=GABA_LN)],
}


def main(only=None):
    meta, W1 = load(w_syn=1.0, monoamine='removed')
    vnc, _ = regions(meta)
    W_base = mask_weights(W1 * (P['w_syn'] * SCALE), lambda r, c: ~vnc[r] & ~vnc[c])
    pos = {b: i for i, b in enumerate(meta.bodyId)}
    mn9 = np.flatnonzero(meta.type.values == 'MN9')
    al = select(meta, f'{LN} or {PN}')
    groups = {'lLN1_bc': select(meta, "type == 'lLN1_bc'"),
              'achLN': select(meta, ACH_LN), 'gabaLN': select(meta, GABA_LN)}
    p = dict(P); p['t_run'] = 1.5
    stim_end, tail0 = int(1.0 / DT), int(1.2 / DT)

    runs, cards, logs = [], [], {}
    for name, cfg in CONDITIONS.items():
        if only and name not in only:
            continue
        W, log = apply(W_base, meta, cfg)
        logs[name] = dict(config=cfg, log=log)
        W_rate, kc_idx, w_kc, _ = make_apl(meta, W)
        apl = dict(kc_idx=kc_idx, w_kc=w_kc, k=K_APL, tau=0.01)
        c0, t0_ = run_trial(W_rate, np.array([], int), params=p, seed=0, apl=apl,
                            t_stim=1.0, tail_from=1.2)
        no_stim_active = int((c0 > 0).sum())
        stim_sets = {}
        for w in INPUTS:
            t0 = time.time()
            r = run_trial(W_rate, stim_idx(meta, pos, w), params=p, seed=0, apl=apl,
                          t_stim=1.0, tail_from=1.2, record=True)
            st, si = r['t'], r['i']
            in_stim = st < stim_end
            stim_cnt = np.bincount(si[in_stim], minlength=len(meta))
            tail_cnt = r['tail']
            stim_sets[w] = stim_cnt > 0
            row = dict(condition=name, word=w,
                       stim_active=int((stim_cnt > 0).sum()),
                       stim_active_AL=int((stim_cnt[al] > 0).sum()),
                       tail_brain=int((tail_cnt > 0).sum()),
                       tail_AL=int((tail_cnt[al] > 0).sum()),
                       persistent=bool((tail_cnt > 0).any()),
                       mn9_L=int(stim_cnt[mn9[0]]), mn9_R=int(stim_cnt[mn9[1]]))
            for gname, gm in groups.items():
                row[f'{gname}_stim_Hz'] = float(np.median(stim_cnt[gm]) / 1.0)
                row[f'{gname}_tail_Hz'] = float(np.median(tail_cnt[gm]) / 0.3)
            runs.append(row)
            print(f'[{name}] {w:13s} stim={row["stim_active"]:6d} AL={row["stim_active_AL"]:4d} '
                  f'tail={row["tail_brain"]:5d} (AL {row["tail_AL"]:4d}) MN9={row["mn9_L"]:3d} '
                  f'({time.time()-t0:.0f}s)', flush=True)
        ws = list(WORDS)
        J = [(stim_sets[x] & stim_sets[y]).sum() / max(1, (stim_sets[x] | stim_sets[y]).sum())
             for i, x in enumerate(ws) for y in ws[i + 1:]]
        d = pd.DataFrame([r for r in runs if r['condition'] == name]).set_index('word')
        card = dict(condition=name,
                    persistent=f"{int(d.persistent.sum())}/{len(d)}",
                    tail_brain_median=int(d.tail_brain.median()),
                    tail_AL_median=int(d.tail_AL.median()),
                    stim_overlap=round(float(np.mean(J)), 3),
                    smell_AL_stim_median=int(d.loc[['vinegar', 'fruit', 'co2', 'mold', 'pheromone'], 'stim_active_AL'].median()),
                    mn9_sugar=int(d.loc['sugar', 'mn9_L']), mn9_bitter=int(d.loc['bitter', 'mn9_L']),
                    mn9_hot=int(d.loc['hot', 'mn9_L']), mn9_salt=int(d.loc['salt', 'mn9_L']),
                    mn9_sugar_bitter=int(d.loc['sugar+bitter', 'mn9_L']),
                    no_stim_active=no_stim_active,
                    lLN1_bc_tail_Hz=round(float(d.lLN1_bc_tail_Hz.median()), 1),
                    achLN_tail_Hz=round(float(d.achLN_tail_Hz.median()), 1),
                    gabaLN_tail_Hz=round(float(d.gabaLN_tail_Hz.median()), 1))
        cards.append(card)
        print('== CARD', json.dumps(card), flush=True)
        pd.DataFrame(runs).to_csv(OUT / 'ckpt5A_runs.csv', index=False)
        pd.DataFrame(cards).to_csv(OUT / 'ckpt5A_scorecard.csv', index=False)
        json.dump(logs, open(OUT / 'ckpt5A_configs.json', 'w'), indent=1)
    print('DONE', flush=True)


if __name__ == '__main__':
    main(set(sys.argv[1:]) or None)
