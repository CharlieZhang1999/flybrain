#!/usr/bin/env python
"""W4D3b: smallest local gain that stops the two self-sustaining excitatory
cliques left after the GLNO/LPsP sign fix: DNg33 L<->R (723/773 synapses,
ACh in MaleCNS and FlyWire) and the IN19B040 quartet (all-to-all, 54-101
synapses per pair, ACh). Only the within-clique edges are scaled by g.
Looming 0-0.4 s, run to 1.0 s, W4_CALIB; reports active neurons 0.6-1.0 s and
the MN count during the drive.
Usage: python scripts/w4d3b_clique_gain.py [seeds=3]
Output: results/w4/sci/w4d3b_clique_gain.txt
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.cns import cns_model, W4_CALIB  # noqa: E402
from sim.lif_numpy import P  # noqa: E402
from sim import lif_torch  # noqa: E402
from scripts.ckpt2_ab import stim_idx  # noqa: E402
from scripts.w4d3b_cx_maintenance import scale_edges  # noqa: E402


def main(seeds=3):
    m = cns_model('cns', 'calib', **W4_CALIB)
    meta, vnc = m['meta'], m['vnc']
    t = meta.type.astype(str).values
    mn = meta.superclass.astype(str).values == 'vnc_motor'
    pos = {b: i for i, b in enumerate(meta.bodyId)}
    exc = stim_idx(meta, pos, 'looming')
    cl = {k: t == k for k in ['DNg33', 'IN19B040']}
    inside = lambda r, c: (cl['DNg33'][r] & cl['DNg33'][c]) | (cl['IN19B040'][r] & cl['IN19B040'][c])
    p = dict(P); p['t_run'] = 1.0
    lines = ['# W4D3b clique gain (looming 0-0.4 s; active 0.6-1.0 s per seed; MNs active during drive)', '',
             '| g (within DNg33 pair and IN19B040 quartet) | brain tail | VNC tail | MNs during drive |', '|---|---|---|---|']
    for g in [1.0, 0.5, 0.3, 0.2, 0.1]:
        W = scale_edges(m['W'], inside, g)
        bt, vt, md = [], [], []
        for s in range(int(seeds)):
            r = lif_torch.run_trial(W, exc, params=p, seed=s, apl=m['apl'], gap=m['gap'], presyn=m['presyn'],
                                    t_stim=0.4, record=True, device='cpu', rng='torch', dtype=torch.float32)
            ts = r['t'] * P['dt']
            late = np.unique(r['i'][ts >= 0.6]); dur = np.unique(r['i'][ts < 0.4])
            bt.append(int((~vnc[late]).sum())); vt.append(int(vnc[late].sum())); md.append(int(mn[dur].sum()))
        row = f'| {g} | {bt} | {vt} | {md} |'
        print(row, flush=True); lines.append(row)
    (ROOT / 'results' / 'w4' / 'sci' / 'w4d3b_clique_gain.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:])
