#!/usr/bin/env python
"""C2: maintenance lesions of the recurrent components found in C1.

Simulator: Brian2-faithful schedule (default since 2026-10-30, verified in C0).
Base: brain-only, monoamines removed, w x 0.581, rate APL k = 0.1.
Trial: odor 0-1.0 s, silence 1.0-1.6 s; lesion (outputs of a group set to 0,
inputs kept) switched on at t = 1.3 s, when the persistent state is established.

Groups (from results/malecns/c1_nodes_*.csv):
  base_SCC_exc   largest excitatory SCC of the baseline tail population (558)
  base_SCC_sgn   largest signed SCC of the baseline tail (912)
  core_SCC_exc   largest excitatory SCC of the LN core seen with LN->PN cut (172)
  core_FVS       greedy cycle-breaking set of the core (104)
  base_FVS       greedy cycle-breaking set of the baseline tail (288)
  rand_172       control: 172 random baseline-tail AL neurons NOT in base_SCC_exc
Readout: active neurons per 5-ms bin (brain, AL), time to collapse (first bin
after the lesion with brain activity < 10% of the 100 ms before it), and
rates of lLN1_bc / cholinergic LNs / GABA LNs / PNs before vs after.
Outputs: results/malecns/c2_bins.csv, c2_summary.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.malecns import load  # noqa: E402
from sim.lif_numpy import run_trial, P  # noqa: E402
from sim.lesion import select  # noqa: E402
from scripts.ckpt2_ab import make_apl, stim_idx  # noqa: E402
from scripts.ckpt3_sweep_ablate import regions, mask_weights  # noqa: E402
from scripts.ckpt5_stageA import ACH_LN, LN, PN, SCALE, K_APL  # noqa: E402

OUT = ROOT / 'results' / 'malecns'
DT, T_LES, T_END, BIN = P['dt'], 1.3, 1.6, 0.005


def group_masks(meta):
    pos = {b: i for i, b in enumerate(meta.bodyId)}
    def mask(ids):
        m = np.zeros(len(meta), bool); m[[pos[b] for b in ids]] = True; return m
    base = pd.read_csv(OUT / 'c1_nodes_base_tail.csv')
    core = pd.read_csv(OUT / 'c1_nodes_lpcut_core.csv')
    g = {'base_SCC_exc': mask(base.bodyId[base.in_big_exc_scc]),
         'base_SCC_sgn': mask(base.bodyId),          # signed SCC = 912 of 914; use all tail AL
         'core_SCC_exc': mask(core.bodyId[core.in_big_exc_scc]),
         'core_FVS': mask(core.bodyId[core.fvs_order >= 0]),
         'base_FVS': mask(base.bodyId[base.fvs_order >= 0])}
    rest = base.bodyId[~base.in_big_exc_scc].to_numpy()
    rng = np.random.default_rng(0)
    g['rand_172'] = mask(rng.choice(rest, size=min(172, len(rest)), replace=False))
    return g


def main():
    meta, W1 = load(w_syn=1.0, monoamine='removed')
    vnc, _ = regions(meta)
    W = mask_weights(W1 * (P['w_syn'] * SCALE), lambda r, c: ~vnc[r] & ~vnc[c])
    Wr, kc, wkc, _ = make_apl(meta, W)
    apl = dict(kc_idx=kc, w_kc=wkc, k=K_APL, tau=0.01)
    pos = {b: i for i, b in enumerate(meta.bodyId)}
    al = select(meta, f'{LN} or {PN}')
    rate_groups = {'lLN1_bc': select(meta, "type == 'lLN1_bc'"), 'eLN': select(meta, ACH_LN),
                   'GABA_LN': select(meta, "`class` == 'ALLN' and nt == 'gaba'"), 'PN': select(meta, PN)}
    groups = group_masks(meta)
    p = dict(P); p['t_run'] = T_END
    nb = int(round(T_END / BIN)); bstep = int(round(BIN / DT)); les_bin = int(round(T_LES / BIN))
    bins, summ = [], []
    conds = {'no_lesion': None, **groups}
    for name, gm in conds.items():
        sw = None if gm is None else (T_LES, mask_weights(Wr, lambda r, c, m=gm: ~m[c]))
        for odor in ['hot', 'vinegar']:
            r = run_trial(Wr, stim_idx(meta, pos, odor), params=p, seed=0, apl=apl,
                          t_stim=1.0, w_switch=sw, record=True)
            b = r['t'] // bstep
            act_brain = np.array([len(np.unique(r['i'][b == k])) for k in range(nb)])
            act_al = np.array([al[np.unique(r['i'][b == k])].sum() for k in range(nb)])
            for k in range(nb):
                bins.append(dict(condition=name, odor=odor, t_ms=k * 5,
                                 brain=int(act_brain[k]), AL=int(act_al[k])))
            pre = act_brain[les_bin - 20:les_bin].mean()
            post = act_brain[les_bin:]
            below = np.flatnonzero(post < 0.1 * pre)
            t_col = float(below[0] * 5) if len(below) else np.nan
            row = dict(condition=name, odor=odor,
                       group_size=int(0 if gm is None else gm.sum()),
                       brain_pre=round(pre, 0), brain_last100=round(act_brain[-20:].mean(), 0),
                       AL_last100=round(act_al[-20:].mean(), 0),
                       collapse_ms=t_col)
            t_s = r['t'] * DT
            for gname, gmask in rate_groups.items():
                for lab, a, bnd in [('pre', T_LES - 0.1, T_LES), ('post', T_END - 0.1, T_END)]:
                    m = (t_s >= a) & (t_s < bnd)
                    row[f'{gname}_{lab}_Hz'] = round(float(np.bincount(r['i'][m], minlength=len(meta))[gmask].mean() / 0.1), 1)
            summ.append(row)
            print(f'{name:13s} {odor:8s} n={row["group_size"]:4d} brain {row["brain_pre"]:.0f} -> '
                  f'{row["brain_last100"]:.0f}  AL {row["AL_last100"]:.0f}  collapse={t_col} ms  '
                  f'eLN {row["eLN_pre_Hz"]}->{row["eLN_post_Hz"]} Hz', flush=True)
        pd.DataFrame(bins).to_csv(OUT / 'c2_bins.csv', index=False)
        pd.DataFrame(summ).to_csv(OUT / 'c2_summary.csv', index=False)
    print('DONE')


if __name__ == '__main__':
    main()
