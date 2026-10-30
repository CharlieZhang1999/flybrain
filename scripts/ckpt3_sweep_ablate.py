#!/usr/bin/env python
"""Checkpoint 3.1: weight sweep, then VNC ablations if persistence survives.

Fixed: monoamines removed, rate-unit APL with k = 0.1 (calibrated in ckpt2/2b),
1.0 s stimulus at 150 Hz, then 0.5 s silence; tail window = last 0.3 s.

Step A (sweep): scale of Shiu's 0.275 mV in {0.699, 0.581, 0.55, 0.50, 0.45}.
  0.699 = 1/1.43 (flymsg density ratio); 0.581 = MaleCNS paper's
  male-to-FlyWire weight alignment factor.
Step B (ablations, only if any word persists at scale 0.581):
  full        : brain + VNC (reused from step A)
  brain_only  : VNC neurons removed (all their rows and columns zeroed)
  no_vnc_vnc  : VNC -> VNC weights zeroed, everything else kept
  no_abd_out  : outgoing weights of abdominal VNC neurons zeroed (inputs kept)

A word "persists" if any neuron fires in the tail window. Tail activity is
reported for brain, VNC and abdominal VNC separately.
Outputs: results/malecns/ckpt3_runs.csv (one row per condition x word)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.malecns import load  # noqa: E402
from sim.lif_numpy import run_trial, P  # noqa: E402
from stimuli.malecns_words import WORDS  # noqa: E402
from scripts.ckpt2_ab import make_apl, stim_idx  # noqa: E402

OUT = ROOT / 'results' / 'malecns' / 'ckpt3_runs.csv'
SCALES = [0.699, 0.581, 0.55, 0.50, 0.45]
ABLATION_SCALE = 0.581
K_APL = 0.1
INPUTS = list(WORDS) + ['sugar+bitter']


def regions(meta):
    sc = meta.superclass.fillna('')
    nm = meta.somaNeuromere.fillna('')
    vnc = (sc.str.startswith('vnc_')
           | sc.isin(['ascending_neuron', 'sensory_ascending',
                      'sensory_ascending_tbc', 'efferent_ascending'])
           | nm.str.match(r'^(T[123]|A\d+)'))
    abd = nm.str.match(r'^A\d+')
    return vnc.to_numpy(), abd.to_numpy()


def mask_weights(W, keep_entry):
    """Zero matrix entries where keep_entry(rows, cols) is False."""
    C = W.tocoo()
    k = keep_entry(C.row, C.col)
    return sparse.csc_matrix((C.data[k], (C.row[k], C.col[k])), shape=W.shape)


def run_condition(label, scale, meta, W, vnc, abd, rows):
    pos = {b: i for i, b in enumerate(meta.bodyId)}
    mn9 = np.flatnonzero(meta.type.values == 'MN9')
    W_rate, kc_idx, w_kc, is_kc = make_apl(meta, W)
    apl = dict(kc_idx=kc_idx, w_kc=w_kc, k=K_APL, tau=0.01)
    p = dict(P); p['t_run'] = 1.5
    acts = {}
    for w in INPUTS:
        t0 = time.time()
        c, tail = run_trial(W_rate, stim_idx(meta, pos, w), params=p, seed=0,
                            apl=apl, t_stim=1.0, tail_from=1.2)
        a, ta = c > 0, tail > 0
        acts[w] = a
        r = dict(condition=label, scale=scale, word=w,
                 active=int(a.sum()), kc_frac=float(a[is_kc].mean()),
                 mn9_L=int(c[mn9[0]]), mn9_R=int(c[mn9[1]]),
                 tail_total=int(ta.sum()), tail_brain=int(ta[~vnc].sum()),
                 tail_vnc=int(ta[vnc].sum()), tail_abd=int(ta[abd].sum()),
                 persistent=bool(ta.any()))
        rows.append(r)
        print(f'[{label} s={scale}] {w:13s} active={r["active"]:6d} MN9={r["mn9_L"]:3d} '
              f'tail brain/vnc/abd={r["tail_brain"]}/{r["tail_vnc"]}/{r["tail_abd"]} '
              f'({time.time()-t0:.0f}s)', flush=True)
    ws = list(WORDS)
    J = [(acts[x] & acts[y]).sum() / max(1, (acts[x] | acts[y]).sum())
         for i, x in enumerate(ws) for y in ws[i + 1:]]
    rows.append(dict(condition=label, scale=scale, word='_mean_jaccard',
                     kc_frac=float(np.mean(J))))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    n_pers = sum(r['persistent'] for r in rows
                 if r.get('condition') == label and r.get('scale') == scale
                 and r['word'] != '_mean_jaccard')
    print(f'== {label} scale {scale}: persistent {n_pers}/{len(INPUTS)}, '
          f'overlap {np.mean(J):.2f}', flush=True)
    return n_pers


def main():
    rows = []
    meta, W1 = load(w_syn=1.0, monoamine='removed')     # raw signed counts
    vnc, abd = regions(meta)
    print(f'regions: brain={int((~vnc).sum())} vnc={int(vnc.sum())} abd={int(abd.sum())}')

    persist = {}
    for s in SCALES:
        persist[s] = run_condition('full', s, meta, W1 * (P['w_syn'] * s), vnc, abd, rows)

    if persist[ABLATION_SCALE] == 0:
        print(f'No persistence at scale {ABLATION_SCALE}; ablations skipped.')
        return
    W = W1 * (P['w_syn'] * ABLATION_SCALE)
    ablations = {
        'brain_only': lambda r, c: ~vnc[r] & ~vnc[c],
        'no_vnc_vnc': lambda r, c: ~(vnc[r] & vnc[c]),
        'no_abd_out': lambda r, c: ~abd[c],          # c = presynaptic column
    }
    for label, keep in ablations.items():
        run_condition(label, ABLATION_SCALE, meta, mask_weights(W, keep), vnc, abd, rows)
    print('DONE')


if __name__ == '__main__':
    main()
