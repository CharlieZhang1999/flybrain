#!/usr/bin/env python
"""W4D3 (science track): is the looming tail brain-side or a new VNC loop?

Looming 0-0.4 s at 150 Hz, run to 1.0 s; active neurons per 0.1 s bin after
offset, brain vs VNC. Variants of the calibrated model:
  cns         full CNS
  cns-noAN    full CNS with every VNC -> brain synapse removed (no ascending
              feedback). If the brain tail is unchanged, it is brain-side.
  cns-noDN    full CNS with every brain -> VNC synapse removed after 0.4 s is
              not possible in one run, so instead: brain -> VNC removed entirely
              (VNC sees only its own sensory input; looming is visual, so the
              VNC should then stay silent).
  brain       Week 3 brain-only model (reference).
Also lists the cell types active in the VNC tail and whether they receive
descending input from neurons active in the brain tail.
Usage: python scripts/w4d3_looming_tail.py [seeds=3] [w4calib]
  w4calib: use sim/cns.W4_CALIB (GF gap junctions 8 mV, MN outputs glutamatergic)
Output: results/w4/sci/w4d3_looming_tail.txt / .csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.cns import cns_model, _mask, W4_CALIB  # noqa: E402
from sim.lif_numpy import P  # noqa: E402
from sim import lif_torch  # noqa: E402
from scripts.ckpt2_ab import stim_idx  # noqa: E402

OUT = ROOT / 'results' / 'w4' / 'sci'
T_STIM, T_RUN, BIN = 0.4, 1.0, 0.1


def main(seeds=3, calib=''):
    kw4 = W4_CALIB if calib == 'w4calib' else {}
    sfx = '_w4calib' if kw4 else ''
    m = cns_model('cns', 'calib', **kw4)
    meta, vnc = m['meta'], m['vnc']
    t = meta.type.astype(str).values
    pos = {b: i for i, b in enumerate(meta.bodyId)}
    exc = stim_idx(meta, pos, 'looming')
    Wb = cns_model('brain', 'calib', **kw4)['W']
    variants = {'cns': m['W'],
                'cns-noAN': _mask(m['W'], lambda r, c: ~(vnc[c] & ~vnc[r])),
                'cns-noDN': _mask(m['W'], lambda r, c: ~(~vnc[c] & vnc[r])),
                'brain': Wb}
    p = dict(P); p['t_run'] = T_RUN
    nb = int(round(T_RUN / BIN))
    rows, lines, tails = [], [], {}
    for name, W in variants.items():
        for seed in range(int(seeds)):
            r = lif_torch.run_trial(W, exc, params=p, seed=seed, apl=m['apl'], gap=m['gap'],
                                    presyn=m['presyn'], t_stim=T_STIM, record=True,
                                    device='cpu', rng='torch', dtype=torch.float32)
            b = np.minimum((r['t'] * P['dt'] / BIN).astype(int), nb - 1)
            for k in range(nb):
                ix = np.unique(r['i'][b == k])
                rows.append(dict(variant=name, seed=seed, t0=round(k * BIN, 1),
                                 brain=int((~vnc[ix]).sum()), vnc=int(vnc[ix].sum()),
                                 spikes=int((b == k).sum())))
            late = r['i'][r['t'] * P['dt'] >= T_STIM + 0.1]
            tails[(name, seed)] = np.bincount(late, minlength=len(meta))
            print(name, seed, [(x['brain'], x['vnc']) for x in rows[-nb:]], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f'w4d3_looming_tail{sfx}.csv', index=False)
    piv = df.groupby(['variant', 't0'])[['brain', 'vnc']].mean().round(0).astype(int)
    tab = piv.unstack('variant')
    lines += ['# W4D3 looming tail (looming 0-0.4 s; mean active neurons per 0.1 s bin, %d seeds)' % int(seeds), '',
              tab.to_string(), '']
    # who is active in the full-CNS VNC tail, and do they get DN input from brain-tail neurons?
    tail = sum(tails[('cns', s)] for s in range(int(seeds)))
    tv = np.flatnonzero((tail > 0) & vnc)
    tb = np.flatnonzero((tail > 0) & ~vnc)
    vc = pd.Series(t[tv]).value_counts()
    lines.append(f'VNC tail neurons (0.5-1.0 s, any seed): {len(tv)}; top types: ' +
                 ', '.join(f'{a} {b}' for a, b in vc.head(12).items()))
    bc = pd.Series(t[tb]).value_counts()
    lines.append(f'Brain tail neurons: {len(tb)}; top types: ' + ', '.join(f'{a} {b}' for a, b in bc.head(12).items()))
    sc = meta.superclass.astype(str).values
    dn_tail = tb[sc[tb] == 'descending_neuron']
    lines.append(f'Descending neurons active in the tail: {len(dn_tail)} ' +
                 str(pd.Series(t[dn_tail]).value_counts().head(10).to_dict()))
    W = m['W'].tocsr()
    fromdn = np.asarray(W[tv][:, dn_tail].sum(axis=1)).ravel() if len(dn_tail) else np.zeros(len(tv))
    lines.append(f'VNC tail neurons with direct input from tail DNs: {(fromdn > 0).sum()} / {len(tv)}')
    (OUT / f'w4d3_looming_tail{sfx}.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main(*sys.argv[1:])
