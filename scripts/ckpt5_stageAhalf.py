#!/usr/bin/env python
"""Checkpoint 5, Stage A-half: is LN->PN needed for ignition or maintenance?

Base = Stage A base (brain-only, monoamines removed, w x 0.581, rate APL k=0.1).
LP = cholinergic ALLN -> ALPN edges (Stage A condition F).
Timeline: odor 0-1.0 s, silence 1.0-1.5 s.

  base          LP intact throughout
  F_full        LP = 0 throughout
  maint_1.0     LP intact during odor, cut at offset (1.0 s)
  maint_1.1     LP cut 100 ms into the silent tail (attractor already formed)
  ignite_only   LP = 0 during odor, restored at offset

Readout per 100-ms window of the tail: AL neurons active, whole-brain neurons
active, total AL spikes, cholinergic-LN and PN rates. Odors: hot, vinegar;
seeds 0, 1. Output: results/malecns/ckpt5Ahalf_windows.csv
Also writes the light NT audit of 'unclear' ALLNs: ckpt5_unclear_ALLN_audit.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.malecns import load, DATA  # noqa: E402
from sim.lif_numpy import run_trial, P  # noqa: E402
from sim.lesion import apply, select  # noqa: E402
from scripts.ckpt2_ab import make_apl, stim_idx  # noqa: E402
from scripts.ckpt3_sweep_ablate import regions, mask_weights  # noqa: E402
from scripts.ckpt5_stageA import ACH_LN, LN, PN, SCALE, K_APL  # noqa: E402

OUT = ROOT / 'results' / 'malecns'
DT = P['dt']
LP_CUT = [dict(op='scale_edges', pre=ACH_LN, post=PN, alpha=0.0)]
WINDOWS = [(0.5, 1.0, 'odor_late'), (1.0, 1.1, 'tail_0-100'), (1.1, 1.2, 'tail_100-200'),
           (1.2, 1.3, 'tail_200-300'), (1.3, 1.4, 'tail_300-400'), (1.4, 1.5, 'tail_400-500')]


def audit(meta):
    nt = pd.read_feather(DATA / 'body-neurotransmitters-male-cns-v1.0.feather')
    unc = meta[(meta['class'] == 'ALLN') & (meta.nt == 'unclear')][['bodyId', 'type', 'somaSide']]
    a = unc.merge(nt, left_on='bodyId', right_on='body', how='left')
    cols = ['bodyId', 'type', 'somaSide', 'predicted_nt', 'predicted_nt_confidence',
            'total_nt_predictions', 'celltype_predicted_nt', 'celltype_predicted_nt_confidence',
            'consensus_nt', 'ground_truth']
    a = a[cols].sort_values('predicted_nt_confidence', ascending=False)
    a.to_csv(OUT / 'ckpt5_unclear_ALLN_audit.csv', index=False)
    print('=== unclear ALLN audit (neuron-level file) ===')
    print(a.predicted_nt.value_counts().to_string())
    print(a.groupby('type', dropna=False).agg(n=('bodyId', 'size'),
          pred=('predicted_nt', lambda s: ','.join(sorted(set(map(str, s))))),
          conf=('predicted_nt_confidence', 'mean'),
          ct_pred=('celltype_predicted_nt', 'first')).round(2).to_string())


def main():
    meta, W1 = load(w_syn=1.0, monoamine='removed')
    audit(meta)
    vnc, _ = regions(meta)
    W_base = mask_weights(W1 * (P['w_syn'] * SCALE), lambda r, c: ~vnc[r] & ~vnc[c])
    W_cut, _ = apply(W_base, meta, LP_CUT)
    Wb, kc, wkc, _ = make_apl(meta, W_base)
    Wc, _, _, _ = make_apl(meta, W_cut)
    apl = dict(kc_idx=kc, w_kc=wkc, k=K_APL, tau=0.01)
    pos = {b: i for i, b in enumerate(meta.bodyId)}
    al = select(meta, f'{LN} or {PN}')
    eln, pn = select(meta, ACH_LN), select(meta, PN)
    conds = {
        'base':        (Wb, None),
        'F_full':      (Wc, None),
        'maint_1.0':   (Wb, (1.0, Wc)),
        'maint_1.1':   (Wb, (1.1, Wc)),
        'ignite_only': (Wc, (1.0, Wb)),
    }
    p = dict(P); p['t_run'] = 1.5
    rows = []
    for odor in ['hot', 'vinegar']:
        for seed in [0, 1]:
            for name, (W0, sw) in conds.items():
                r = run_trial(W0, stim_idx(meta, pos, odor), params=p, seed=seed, apl=apl,
                              t_stim=1.0, w_switch=sw, record=True)
                st, si = r['t'] * DT, r['i']
                for a, b, lab in WINDOWS:
                    m = (st >= a) & (st < b)
                    cnt = np.bincount(si[m], minlength=len(meta))
                    dur = b - a
                    rows.append(dict(odor=odor, seed=seed, condition=name, window=lab,
                                     AL_active=int((cnt[al] > 0).sum()),
                                     brain_active=int((cnt > 0).sum()),
                                     AL_spikes=int(cnt[al].sum()),
                                     eLN_Hz=float(cnt[eln].mean() / dur),
                                     PN_Hz=float(cnt[pn].mean() / dur)))
                last = rows[-1]
                print(f'{odor:8s} s{seed} {name:12s} last-100ms AL={last["AL_active"]:4d} '
                      f'brain={last["brain_active"]:5d}', flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'ckpt5Ahalf_windows.csv', index=False)
    print('DONE')


if __name__ == '__main__':
    main()
