#!/usr/bin/env python
"""W4D3b: what maintains the looming-ignited central-complex (CX) state?

Ignite with the full W4_CALIB model (looming 0-0.4 s, run to 0.6 s, save
state), then resume to 1.0 s with one edge class scaled by g, and count active
neurons in 0.8-1.0 s. Edge classes (excitatory chemical synapses only):
  pb      EPG -> PEN/PEG/EPGt          (protocerebral bridge)
  eb      PEN/PEG/EPGt -> EPG          (ellipsoid body / gall)
  cx      all among EPG, PEN, PEG, EPGt
  cx_out  CX ring types -> anything else
  in_cx   anything else -> CX ring types
Usage: python scripts/w4d3b_cx_maintenance.py [seeds=1,2]
Output: results/w4/sci/w4d3b_cx_maintenance.txt
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.cns import cns_model, W4_CALIB  # noqa: E402
from sim.lif_numpy import P  # noqa: E402
from sim import lif_torch  # noqa: E402
from scripts.ckpt2_ab import stim_idx  # noqa: E402

RING = ['EPG', 'PEN_a(PEN1)', 'PEN_b(PEN2)', 'PEG', 'EPGt']


def scale_edges(W, keep, g):
    """Scale excitatory entries where keep(row, col) is True by g."""
    C = W.tocoo()
    k = keep(C.row, C.col) & (C.data > 0)
    d = C.data.copy(); d[k] *= g
    return sparse.csc_matrix((d, (C.row, C.col)), shape=W.shape)


def classes(t):
    epg = t == 'EPG'
    pen = np.isin(t, ['PEN_a(PEN1)', 'PEN_b(PEN2)', 'PEG', 'EPGt'])
    ring = np.isin(t, RING)
    return {
        'pb': lambda r, c: epg[c] & pen[r],
        'eb': lambda r, c: pen[c] & epg[r],
        'cx': lambda r, c: ring[c] & ring[r],
        'cx_out': lambda r, c: ring[c] & ~ring[r],
        'in_cx': lambda r, c: ~ring[c] & ring[r],
    }, ring


def main(seeds='1,2'):
    m = cns_model('cns', 'calib', **W4_CALIB)
    meta, vnc = m['meta'], m['vnc']
    t = meta.type.astype(str).values
    pos = {b: i for i, b in enumerate(meta.bodyId)}
    exc = stim_idx(meta, pos, 'looming')
    cls, ring = classes(t)
    kw = dict(apl=m['apl'], gap=m['gap'], presyn=m['presyn'], t_stim=0.4, record=True,
              device='cpu', rng='torch', dtype=torch.float32)
    lines = ['# W4D3b: maintenance of the looming CX state (resume at 0.6 s with edge class x g; active in 0.8-1.0 s)', '',
             '| edge class | g | seed | brain | CX ring | VNC |', '|---|---|---|---|---|---|']
    for s in [int(x) for x in seeds.split(',')]:
        p = dict(P); p['t_run'] = 0.6
        a = lif_torch.run_trial(m['W'], exc, params=p, seed=s, return_state=True, **kw)
        p2 = dict(P); p2['t_run'] = 1.0
        for name, keep in [('none', None)] + list(cls.items()):
            for g in ([1.0] if keep is None else [0.0, 0.5]):
                W = m['W'] if keep is None else scale_edges(m['W'], keep, g)
                b = lif_torch.run_trial(W, exc, params=p2, seed=s, state=a['state'], **kw)
                late = np.unique(b['i'][b['t'] * P['dt'] >= 0.8])
                row = f'| {name} | {g} | {s} | {int((~vnc[late]).sum())} | {int(ring[late].sum())} | {int(vnc[late].sum())} |'
                print(row, flush=True); lines.append(row)
    out = ROOT / 'results' / 'w4' / 'sci'
    (out / 'w4d3b_cx_maintenance.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:])
