#!/usr/bin/env python
"""C3: sign sensitivity of the unclear-transmitter members of the causal core.

The 104-neuron cycle-breaking set (C1, lpcut_core) contains 13 ALLNs whose
MaleCNS transmitter is 'unclear' (confidence 0.43-0.49), which the model
treats as excitatory. Confirmed cholinergic members are left unchanged.

Treatments of those 13 neurons' outputs: current (+1), zero, inhibitory (-1).
Modes: 'whole'  - applied from t = 0 (does the attractor still form? does the
                  odor response survive?)
       'maint'  - applied at t = 1.3 s, after the state is established (C2 timing)
Trial: odor 0-1.0 s, silence 1.0-1.6 s. Odors hot, vinegar; seeds 0, 1.
Base: Brian2-faithful simulator, brain-only, monoamines removed, w x 0.581,
rate APL k = 0.1.
Outputs: results/malecns/c3_runs.csv
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
DT, T_SW, T_END = P['dt'], 1.3, 1.6


def with_sign(W, mask, sign):
    """Return a copy of W with the outputs (columns) of `mask` given `sign`."""
    C = W.tocoo(copy=True)
    hit = mask[C.col]
    data = C.data.copy()
    data[hit] = np.abs(data[hit]) * sign
    from scipy import sparse
    out = sparse.csc_matrix((data, (C.row, C.col)), shape=W.shape)
    out.eliminate_zeros()
    return out


def main():
    meta, W1 = load(w_syn=1.0, monoamine='removed')
    vnc, _ = regions(meta)
    W = mask_weights(W1 * (P['w_syn'] * SCALE), lambda r, c: ~vnc[r] & ~vnc[c])
    Wr, kc, wkc, _ = make_apl(meta, W)
    apl = dict(kc_idx=kc, w_kc=wkc, k=K_APL, tau=0.01)
    pos = {b: i for i, b in enumerate(meta.bodyId)}

    core = pd.read_csv(OUT / 'c1_nodes_lpcut_core.csv')
    fvs = core[core.fvs_order >= 0]
    unc_ids = fvs.bodyId[fvs.nt == 'unclear'].to_numpy()
    unc = np.zeros(len(meta), bool); unc[[pos[b] for b in unc_ids]] = True
    set104 = np.zeros(len(meta), bool); set104[[pos[b] for b in fvs.bodyId]] = True
    print(f'unclear core neurons: {unc.sum()} ->',
          pd.Series(meta.type.values[unc]).value_counts().to_dict(), flush=True)

    al = select(meta, f'{LN} or {PN}')
    groups = {'set104': set104, 'lLN1_bc': select(meta, "type == 'lLN1_bc'"),
              'eLN': select(meta, ACH_LN), 'PN': select(meta, PN)}
    mn9 = np.flatnonzero(meta.type.values == 'MN9')
    variants = {'current': Wr, 'zero': with_sign(Wr, unc, 0.0), 'inhib': with_sign(Wr, unc, -1.0)}
    p = dict(P); p['t_run'] = T_END
    rows = []
    for treat, Wt in variants.items():
        for mode in ['whole', 'maint']:
            if treat == 'current' and mode == 'maint':
                continue                               # identical to 'whole'
            W0, sw = (Wt, None) if mode == 'whole' else (Wr, (T_SW, Wt))
            for odor in ['hot', 'vinegar']:
                for seed in [0, 1]:
                    r = run_trial(W0, stim_idx(meta, pos, odor), params=p, seed=seed,
                                  apl=apl, t_stim=1.0, w_switch=sw, record=True)
                    ts, si = r['t'] * DT, r['i']
                    def cnt(a, b):
                        m = (ts >= a) & (ts < b)
                        return np.bincount(si[m], minlength=len(meta))
                    odor_c, last = cnt(0.5, 1.0), cnt(T_END - 0.1, T_END)
                    row = dict(treatment=treat, mode=mode, odor=odor, seed=seed,
                               odor_AL_active=int((odor_c[al] > 0).sum()),
                               odor_brain_active=int((odor_c > 0).sum()),
                               mn9_odor_Hz=float(odor_c[mn9[0]] / 0.5),
                               tail_last100_brain=int((last > 0).sum()),
                               tail_last100_AL=int((last[al] > 0).sum()),
                               persistent=bool((last > 0).sum() > 20))
                    for gname, gm in groups.items():
                        row[f'{gname}_tail_Hz'] = round(float(last[gm].mean() / 0.1), 1)
                    rows.append(row)
                    print(f'{treat:7s} {mode:5s} {odor:8s} s{seed}: odor AL={row["odor_AL_active"]:4d} '
                          f'MN9={row["mn9_odor_Hz"]:5.1f} | last100 brain={row["tail_last100_brain"]:5d} '
                          f'set104={row["set104_tail_Hz"]:6.1f}Hz PN={row["PN_tail_Hz"]:5.1f}Hz', flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'c3_runs.csv', index=False)
    print('DONE')


if __name__ == '__main__':
    main()
