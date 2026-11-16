#!/usr/bin/env python
"""Week 3 Day 1 follow-up: locate and test the cold/humid loop (calibrated MaleCNS).
1. Excitatory edges among the core antennal-lobe tail types.
2. Quick lesions, cold and humid, 0-0.5 s drive, silence to 2.0 s, seed 0:
   A  no chemical input onto any sensory neuron (central -> sensory-terminal edges = 0)
   B  no chemical input onto thermo/hygrosensory neurons only
   C  l2LN22 (sign 'unclear', modelled excitatory; FlyWire calls it GABA) set inhibitory
Output: results/loop/w3d1c_log.txt (printed), w3d1c_lesions.csv"""
import numpy as np
import pandas as pd
from scipy import sparse
from w3d1_common import calib_model, ROOT, DT
from sim.lif_numpy import run_trial
from scripts.ckpt2_ab import stim_idx

R, W, gap = calib_model()
m = R.meta
core_types = ['TRN_VP3a', 'TRN_VP3b', 'VP3+_l2PN', 'v2LN4', 'l2LN22', 'l2LN20', 'M_l2PN10t19', 'HRN_VP5']
core = np.flatnonzero(m.type.isin(core_types).to_numpy())
C = W[core][:, core].tocoo()
e = pd.DataFrame(dict(pre=m.type.values[core[C.col]], post=m.type.values[core[C.row]], w=C.data / (0.275e-3 * 0.581)))
print('chemical edges among core types (w in synapse-count units, signed):')
print(e.groupby(['pre', 'post']).w.sum().round(0).sort_values().to_string())
sens = m.superclass.astype(str).str.contains('sensory').to_numpy()
th = m['class'].isin(['thermosensory', 'hygrosensory']).to_numpy()
into = W.multiply(sens[:, None]).tocsc()
print(f'\nchemical edges onto sensory neurons: {into.nnz}, sum excit={into.data[into.data>0].sum()/(0.275e-3*0.581):.0f}, '
      f'inhib={into.data[into.data<0].sum()/(0.275e-3*0.581):.0f} synapses')
tin = W.multiply(th[:, None]).tocoo()
d = pd.DataFrame(dict(pre=m.type.values[tin.col], post=m.type.values[tin.row], w=tin.data / (0.275e-3 * 0.581)))
print('top inputs onto thermo/hygro sensory neurons:')
print(d.groupby(['pre']).w.sum().round(0).sort_values(key=abs, ascending=False).head(15).to_string())


def zero_rows(W, rows):
    D = sparse.diags((~rows).astype(float))
    return (D @ W).tocsc()


def flip(W, cols):
    D = sparse.diags(np.where(cols, -1.0, 1.0))
    return (W @ D).tocsc()


l2ln22 = (m.type == 'l2LN22').to_numpy()
conds = {'calib': W, 'A_no_input_to_sensory': zero_rows(W, sens),
         'B_no_input_to_thermohygro': zero_rows(W, th), 'C_l2LN22_inhib': flip(W, l2ln22)}
rows = []
for name, Wc in conds.items():
    for word in ['cold', 'humid']:
        p = dict(R.p); p['t_run'] = 2.0
        r = run_trial(Wc, stim_idx(m, R.pos, word), params=p, seed=0, apl=R.apl, t_stim=0.5, record=True, gap=gap)
        ts, si = r['t'] * DT, r['i']
        stim_act = len(np.unique(si[ts < 0.5]))
        tail = len(np.unique(si[ts >= 1.5]))
        vp3 = np.flatnonzero(m.type.isin(['TRN_VP3a']).to_numpy())
        vp3_hz = np.isin(si[ts >= 1.5], vp3).sum() / len(vp3) / 0.5
        rows.append(dict(cond=name, word=word, active_stim=stim_act, tail_1p5_2s=tail, TRN_VP3a_tail_Hz=round(vp3_hz, 1)))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(ROOT / 'results' / 'loop' / 'w3d1c_lesions.csv', index=False)
