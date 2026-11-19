#!/usr/bin/env python
"""W4D5 T2d: simulations testing the strongest annotation suspects for the wind loop (Q4).
Diagnosis only: every edit acts on a copy of the final calib W inside this script; nothing under
sim/ or stimuli/ is changed. Run ONE simulation process at a time.

Model: cns_model('cns', 'calib', **W4_CALIB); CPU float32, rng='torch'; exact Day 5 specs
(results/w4/regression/day5/specs.json: class_v2 background, same Poisson seed, T = 2.5 s).

Variants (chosen from scripts/w4d5_loopid_static.py, see progress/week4/day5_wind_loop_rootcause.md)
  mono5       outputs of DNg22, GNG671, AN05B096, AN05B105, LN-DN1 deleted: the excitatory loop inputs
              that FlyWire predicts as serotonin/dopamine (or MaleCNS 'unclear'); deleting them is what
              monoamine='removed' would do if they were relabelled.
  DNg21_inh   every chemical output of DNg21 made inhibitory (-|W|): lowest-confidence LOOP5 member
              (MaleCNS ACh 0.81, 4th percentile of ACh DNs; MX12 hemilineage 34% glutamate).
  GNG423_inh  every chemical output of GNG423 made inhibitory: lowest FlyWire confidence in LOOP5
              (ACh 0.46, 8th percentile of FlyWire ACh DNs).
  DNg22_inh   every chemical output of DNg22 made inhibitory: the partner with the most disagreeing
              labels (MaleCNS ACh 0.59-0.62, FlyWire serotonin 0.66-0.78, FlyWire known_nt FMRFa,
              MANC instances tagged GABAergic on VFB); tests the GABA reading.
  nomirror    no edit of W; model built with cns_model(..., mirror=False), i.e. without the right-JO
              mirror (which raises JO-R -> SAD093_R from 326 to 716 syn-eq). Wind episodes only.
Episodes: wind ep009, ep002, ep016; non-wind ep000, ep019.
Usage:  python scripts/w4d5_loopid_runs.py [variant ...]   (default: all four)
Output: results/w4/regression/loop_id/runs/<variant>_ep<k>.npz (t: step of 0.1 ms, i: neuron,
        spec: JSON incl. variant and number of W entries edited); runs_log.txt (wall time).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'results' / 'w4' / 'regression' / 'loop_id'
RUNS = OUT / 'runs'
DAY5 = ROOT / 'results' / 'w4' / 'regression' / 'day5'
EPISODES = [9, 2, 16, 0, 19]
MONO5 = ['DNg22', 'GNG671', 'AN05B096', 'AN05B105', 'LN-DN1']
VARIANTS = {'mono5': ('delete', MONO5), 'DNg21_inh': ('inhib', ['DNg21']),
            'GNG423_inh': ('inhib', ['GNG423']), 'DNg22_inh': ('inhib', ['DNg22'])}


def log(msg):
    print(msg, flush=True)
    with open(OUT / 'runs_log.txt', 'a') as f:
        f.write(msg + '\n')


def edit_W(W, ty, how, types):
    """Copy of W with the columns (presynaptic) of `types` deleted or made inhibitory."""
    from scipy import sparse
    C = W.tocoo()
    k = np.isin(ty[C.col], types)
    d = C.data.copy()
    if how == 'delete':
        d[k] = 0.0
    elif how == 'inhib':
        d[k] = -np.abs(d[k])
    else:
        raise ValueError(how)
    W2 = sparse.csc_matrix((d, (C.row, C.col)), shape=C.shape)
    W2.eliminate_zeros()
    return W2, int(k.sum()), int((C.data[k] > 0).sum())


def main():
    import torch
    from sim.cns import cns_model, W4_CALIB
    from sim.lif_numpy import P
    from sim import lif_torch
    from stimuli.mixtures import to_blocks
    torch.set_num_threads(4)
    want = sys.argv[1:] or list(VARIANTS)
    want = [v for v in want if v != 'nomirror'] + [v for v in want if v == 'nomirror']   # rebuild last
    RUNS.mkdir(parents=True, exist_ok=True)
    log(f'[start] {want} {time.strftime("%Y-%m-%d %H:%M:%S")}')
    t0 = time.time()
    m = cns_model('cns', 'calib', verbose=False, **W4_CALIB)
    meta = m['meta']
    ty = meta.type.fillna('?').astype(str).values
    pos = {b: i for i, b in enumerate(meta.bodyId)}
    specs = json.loads((DAY5 / 'specs.json').read_text())
    log(f'[model] built in {time.time() - t0:.1f} s')
    for v in want:
        if v == 'nomirror':
            m = cns_model('cns', 'calib', verbose=False, mirror=False, **W4_CALIB)
            W, how, types, ne = m['W'], 'mirror=False', [], 0
            log(f'[{v}] model rebuilt with mirror=False')
            eps = EPISODES[:3]
        else:
            how, types = VARIANTS[v]
            W, ne, npos = edit_W(m['W'], ty, how, types)
            log(f'[{v}] {how} {types}: {ne} W entries edited ({npos} were positive)')
            eps = EPISODES
        for k in eps:
            f = RUNS / f'{v}_ep{k:03d}.npz'
            if f.exists():
                continue
            spec = specs[k]
            p = dict(P); p['t_run'] = spec['T']
            t1 = time.time()
            r = lif_torch.run_trial(W, [], params=p, seed=spec['seed'], apl=m['apl'], gap=m['gap'],
                                    presyn=m['presyn'], blocks=to_blocks(spec, meta, pos), record=True,
                                    device='cpu', rng='torch', dtype=torch.float32)
            tmp = RUNS / f'.{v}_ep{k:03d}.tmp.npz'
            np.savez_compressed(tmp, t=np.asarray(r['t'], np.int32), i=np.asarray(r['i'], np.int32),
                                spec=json.dumps(dict(spec, variant=v, edit=how, types=types, n_entries=ne)))
            tmp.replace(f)
            log(f'[{v}] ep{k:03d}: {len(r["t"]):,} spikes, {time.time() - t1:.1f} s wall')
    log(f'[end] {time.strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()
