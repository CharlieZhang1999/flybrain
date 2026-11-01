#!/usr/bin/env python
"""C7b step 2: sign-resolved core.

Per-T-bar transmitter probabilities (data/malecns/tbar_nt_AL.parquet, T-bars
with conf >= 0.5) are averaged per neuron. Calibration: on the 638 AL neurons
with MaleCNS ground truth, the argmax of the mean recovers it for 98.3%.
The 13 unclear core neurons are near 50/50 ACh vs GABA (vote share 0.45-0.52).

Sign variants for the 13 unclear neurons' outputs:
  current   +1 (model default for 'unclear')
  zero      0 (unresolved)
  expected  p(ACh) - p(GABA+Glu+His) of the mean T-bar profile (monoamines -> 0)
  argmax    sign of the most probable transmitter (9 inhibitory, 4 excitatory)
Part A, whole trial: hot + vinegar x seeds 0, 1 (same metrics as C4).
Part B, threshold shift: C6-style rescue where the other 91 core neurons are
  silenced at 1.3 s except the restored group, with the 13 unclear neurons
  kept active at the variant's sign: lLN1_bc beta 0.5, otherLN beta 0.5 and 1.0.
Outputs: results/malecns/c7b_signs.csv, c7b_whole.csv, c7b_rescue.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.lif_numpy import run_trial  # noqa: E402
from scripts.ckpt2_ab import stim_idx  # noqa: E402
from scripts.c4_core_gain import Runner, DT, T_END, OUT  # noqa: E402
from scripts.c5_gain_decomp import core_groups, T_SW  # noqa: E402

NTS = ['acetylcholine', 'dopamine', 'gaba', 'glutamate', 'histamine', 'octopamine', 'serotonin']
INH = ['gaba', 'glutamate', 'histamine']


def tbar_profile():
    t = pd.read_parquet(ROOT / 'data' / 'malecns' / 'tbar_nt_AL.parquet')
    t = t[t.conf >= 0.5]
    P = t.groupby('body')[[f'nt_{n}_prob' for n in NTS]].mean()
    P.columns = NTS
    P['n_tbar'] = t.groupby('body').size()
    P['argmax'] = P[NTS].idxmax(axis=1)
    P['expected'] = P['acetylcholine'] - P[INH].sum(axis=1)
    P['argmax_sign'] = np.select([P.argmax == 'acetylcholine', P.argmax.isin(INH)], [1.0, -1.0], 0.0)
    return P


def with_signs(W, s):
    """Copy of W where column j with finite s[j] gets |w| * s[j]."""
    C = W.tocoo(copy=True)
    data = C.data.copy()
    hit = np.isfinite(s[C.col])
    data[hit] = np.abs(data[hit]) * s[C.col][hit]
    out = sparse.csc_matrix((data, (C.row, C.col)), shape=W.shape)
    out.eliminate_zeros()
    return out


def metrics(R, r, extra):
    ts, si = r['t'] * DT, r['i']
    def cnt(a, b):
        m = (ts >= a) & (ts < b)
        return np.bincount(si[m], minlength=len(R.meta))
    odor_c, last = cnt(0.5, 1.0), cnt(T_END - 0.1, T_END)
    return dict(**extra, odor_AL=int((odor_c[R.al] > 0).sum()),
                mn9_odor_Hz=float(odor_c[R.mn9[0]] / 0.5),
                tail_brain=int((last > 0).sum()), tail_AL=int((last[R.al] > 0).sum()),
                lLN1_bc_tail_Hz=float(last[R.lln1].mean() / 0.1),
                persistent=bool((last > 0).sum() > 20))


def main():
    R = Runner()
    G = core_groups(R)
    unc = G['unclear']
    P = tbar_profile()
    ids = R.meta.bodyId.values[unc]
    sig = P.loc[ids, ['n_tbar', 'acetylcholine', 'gaba', 'glutamate', 'argmax', 'expected', 'argmax_sign']]
    sig.insert(0, 'type', R.meta.type.values[unc])
    sig.to_csv(OUT / 'c7b_signs.csv')
    print(sig.round(3).to_string(), flush=True)

    variants = {}
    for name, vals in [('current', np.ones(unc.sum())), ('zero', np.zeros(unc.sum())),
                       ('expected', sig.expected.to_numpy()), ('argmax', sig.argmax_sign.to_numpy())]:
        s = np.full(len(R.meta), np.nan); s[unc] = vals
        variants[name] = (s, R.Wr if name == 'current' else with_signs(R.Wr, s))

    rows = []
    for name, (_, W) in variants.items():
        for w in ['hot', 'vinegar']:
            for seed in [0, 1]:
                r = run_trial(W, stim_idx(R.meta, R.pos, w), params=R.p, seed=seed, apl=R.apl,
                              t_stim=1.0, record=True)
                x = metrics(R, r, dict(variant=name, word=w, seed=seed)); rows.append(x)
                print(f'[whole] {name:8s} {w:8s} s{seed}: odor AL={x["odor_AL"]:4d} MN9={x["mn9_odor_Hz"]:5.1f} '
                      f'| tail brain={x["tail_brain"]:5d} AL={x["tail_AL"]:4d} lLN1_bc={x["lLN1_bc_tail_Hz"]:5.1f} Hz', flush=True)
        pd.DataFrame(rows).to_csv(OUT / 'c7b_whole.csv', index=False)

    resc = []
    for name, (s, W) in variants.items():
        for grp, beta in [('lLN1_bc', 0.5), ('otherLN', 0.5), ('otherLN', 1.0)]:
            sc = np.ones(len(R.meta)); sc[G['ach91']] = 0.0; sc[G[grp]] = beta
            C = W.tocoo()
            W2 = sparse.csc_matrix((C.data * sc[C.col], (C.row, C.col)), shape=W.shape)
            for w in ['hot', 'vinegar']:
                r = run_trial(W, stim_idx(R.meta, R.pos, w), params=R.p, seed=0, apl=R.apl,
                              t_stim=1.0, w_switch=(T_SW, W2), record=True)
                x = metrics(R, r, dict(variant=name, restored=grp, beta=beta, word=w)); resc.append(x)
                print(f'[rescue] {name:8s} {grp:8s} b={beta}: {w:8s} tail brain={x["tail_brain"]:5d} '
                      f'AL={x["tail_AL"]:4d} persistent={x["persistent"]}', flush=True)
        pd.DataFrame(resc).to_csv(OUT / 'c7b_rescue.csv', index=False)
    print('DONE')


if __name__ == '__main__':
    main()
