#!/usr/bin/env python
"""Checkpoint 5, Stage B1 + B2.

Base = Stage A base (brain-only, monoamines removed, w x 0.581, rate APL k=0.1).

B1 (identify the self-sustaining LN core): run 'hot' and 'vinegar' with
cholinergic ALLN -> ALPN cut (condition F); list AL neurons still firing in the
last 300 ms of silence by type / transmitter, and the signed weight among them.

B2 (split LN -> LN), each on the base network:
  E1  eLN -> eLN        = 0   (eLN = cholinergic ALLN)
  E2  eLN -> GABA LN    = 0
  E2g eLN -> Glu LN     = 0
  E3  eLN -> unclear LN = 0
  U   unclear-LN outputs = 0
Readout per 100-ms tail window: AL / brain active, eLN and PN rates, plus the
odor-period AL response. Odors hot, vinegar; seeds 0, 1.
Outputs: results/malecns/ckpt5B1_core.csv, ckpt5B2_windows.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.malecns import load  # noqa: E402
from sim.lif_numpy import run_trial, P  # noqa: E402
from sim.lesion import apply, select  # noqa: E402
from scripts.ckpt2_ab import make_apl, stim_idx  # noqa: E402
from scripts.ckpt3_sweep_ablate import regions, mask_weights  # noqa: E402
from scripts.ckpt5_stageA import ACH_LN, LN, PN, SCALE, K_APL  # noqa: E402
from scripts.ckpt5_stageAhalf import WINDOWS  # noqa: E402

OUT = ROOT / 'results' / 'malecns'
DT = P['dt']
GABA_LN = "`class` == 'ALLN' and nt == 'gaba'"
GLU_LN = "`class` == 'ALLN' and nt == 'glutamate'"
UNC_LN = "`class` == 'ALLN' and nt == 'unclear'"
B2 = {
    'base': [],
    'E1_eLN_to_eLN0': [dict(op='scale_edges', pre=ACH_LN, post=ACH_LN, alpha=0.0)],
    'E2_eLN_to_GABA0': [dict(op='scale_edges', pre=ACH_LN, post=GABA_LN, alpha=0.0)],
    'E2g_eLN_to_Glu0': [dict(op='scale_edges', pre=ACH_LN, post=GLU_LN, alpha=0.0)],
    'E3_eLN_to_unc0': [dict(op='scale_edges', pre=ACH_LN, post=UNC_LN, alpha=0.0)],
    'U_uncLN_out0': [dict(op='silence_out', group=UNC_LN)],
}


def prep(W_base, meta, cfg):
    W, log = apply(W_base, meta, cfg)
    Wr, kc, wkc, _ = make_apl(meta, W)
    return Wr, dict(kc_idx=kc, w_kc=wkc, k=K_APL, tau=0.01), log


def main():
    meta, W1 = load(w_syn=1.0, monoamine='removed')
    vnc, _ = regions(meta)
    W_base = mask_weights(W1 * (P['w_syn'] * SCALE), lambda r, c: ~vnc[r] & ~vnc[c])
    pos = {b: i for i, b in enumerate(meta.bodyId)}
    al = select(meta, f'{LN} or {PN}')
    eln, pn = select(meta, ACH_LN), select(meta, PN)
    p = dict(P); p['t_run'] = 1.5

    # ---------------- B1 ----------------
    Wf, aplf, _ = prep(W_base, meta, [dict(op='scale_edges', pre=ACH_LN, post=PN, alpha=0.0)])
    tails = []
    for odor in ['hot', 'vinegar']:
        r = run_trial(Wf, stim_idx(meta, pos, odor), params=p, seed=0, apl=aplf,
                      t_stim=1.0, tail_from=1.2)
        tails.append(r[1])
    tail = np.minimum(*tails)                       # active in BOTH odors' tails
    core = (tail > 0) & al
    hz = tail[core] / 0.3
    df = meta[core][['bodyId', 'type', 'class', 'nt', 'somaSide']].assign(tail_Hz=hz)
    df.to_csv(OUT / 'ckpt5B1_core.csv', index=False)
    print(f'=== B1: AL neurons persisting in both odors with LP cut: {core.sum()} ===')
    print(df.groupby(['class', 'nt']).agg(n=('bodyId', 'size'), median_Hz=('tail_Hz', 'median')).round(0).to_string())
    print(df.groupby(['type', 'nt']).agg(n=('bodyId', 'size'), Hz=('tail_Hz', 'median'))
          .sort_values('n', ascending=False).head(25).round(0).to_string())
    idx = np.flatnonzero(core)
    S = Wf[idx][:, idx]                             # within-core, post x pre
    print(f'within-core: E = {S.data[S.data > 0].sum()*1e3:.0f} mV, I = {-S.data[S.data < 0].sum()*1e3:.0f} mV, '
          f'edges = {S.nnz}')
    pre_nt = meta.nt.values[idx][S.tocoo().col]
    print('within-core |w| by presynaptic NT (mV):',
          (pd.Series(np.abs(S.tocoo().data)).groupby(pre_nt).sum() * 1e3).round(0).to_dict())
    # incoming excitation to core neurons from the core vs everything else
    ext_in = np.asarray(Wf[idx].multiply(Wf[idx] > 0).sum(axis=1)).ravel()
    int_in = np.asarray(S.multiply(S > 0).sum(axis=1)).ravel()
    print(f'median share of core-neuron excitatory input that comes from the core: '
          f'{np.median(int_in / np.maximum(ext_in, 1e-12)):.2f}', flush=True)

    # ---------------- B2 ----------------
    rows = []
    for name, cfg in B2.items():
        Wr, apl, log = prep(W_base, meta, cfg)
        print(f'[B2] {name}: {log}', flush=True)
        for odor in ['hot', 'vinegar']:
            for seed in [0, 1]:
                r = run_trial(Wr, stim_idx(meta, pos, odor), params=p, seed=seed, apl=apl,
                              t_stim=1.0, record=True)
                st, si = r['t'] * DT, r['i']
                for a, b, lab in WINDOWS:
                    m = (st >= a) & (st < b)
                    cnt = np.bincount(si[m], minlength=len(meta))
                    rows.append(dict(condition=name, odor=odor, seed=seed, window=lab,
                                     AL_active=int((cnt[al] > 0).sum()),
                                     brain_active=int((cnt > 0).sum()),
                                     eLN_Hz=float(cnt[eln].mean() / (b - a)),
                                     PN_Hz=float(cnt[pn].mean() / (b - a))))
                print(f'   {odor:8s} s{seed} last-100ms brain={rows[-1]["brain_active"]:5d} '
                      f'eLN={rows[-1]["eLN_Hz"]:.0f}Hz PN={rows[-1]["PN_Hz"]:.0f}Hz', flush=True)
        pd.DataFrame(rows).to_csv(OUT / 'ckpt5B2_windows.csv', index=False)
    print('DONE')


if __name__ == '__main__':
    main()
