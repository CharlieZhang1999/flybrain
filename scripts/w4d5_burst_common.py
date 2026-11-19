"""W4D5 T2c: shared helpers for the burst / wind-persistence diagnosis (no simulation here).

Purpose: masks (brain_mask, AL gap-junction set, AL classes, wind word), burst-event
detection and per-bin activity counts used by scripts/w4d5_burst_*.py.
Usage: import scripts.w4d5_burst_common as C  (from the repo root, sys.path has the root).
Output: none (library).

Definitions
  brain bin count  number of distinct brain_mask neurons with >= 1 spike in a 100 ms bin
                   (same as results/w4/regression/day5/stop_check.py, which also removes
                   word-driven neurons; brain_mask contains no sensory neuron, so for the
                   bg-only runs this is identical).
  AL event         a 10 ms window (sliding in 1 ms steps) in which >= AL_EVENT_MIN neurons
                   of the AL set (all ALPN + all ALLN) fire; events closer than 50 ms are merged.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'results' / 'w4' / 'regression' / 'burst_diag'
DAY5 = ROOT / 'results' / 'w4' / 'regression' / 'day5'
SCRATCH = Path('/tmp/claude-0/t2c')
STEP = 1e-4                  # simulator step (s)
AL_EVENT_MIN = 100


def masks(meta, vnc):
    from sim.cns import brain_mask
    cl = meta['class'].fillna('?').astype(str).values
    nt = meta.nt.fillna('?').astype(str).values
    bm = brain_mask(meta, vnc)
    alpn = cl == 'ALPN'
    alln = cl == 'ALLN'
    return dict(brain=bm, alpn=alpn, alln=alln, al=alpn | alln, ach_ln=alln & (nt == 'acetylcholine'),
                gaba_ln=alln & (nt == 'gaba'))


def load_run(f):
    z = np.load(f, allow_pickle=True)
    spec = json.loads(str(z['spec'])) if 'spec' in z.files else {}
    return z['t'].astype(np.int64), z['i'].astype(np.int64), spec


def bin_counts(t, i, sel, T, width=0.1, t0=0.0):
    """Distinct neurons of mask sel active per bin of `width` s, from t0 to T."""
    nb = int(round((T - t0) / width))
    k = sel[i] & (t >= round(t0 / STEP))
    b = ((t[k] - round(t0 / STEP)) // round(width / STEP))
    out = np.zeros(nb, int)
    for j in range(nb):
        out[j] = len(np.unique(i[k][b == j]))
    return out


def al_events(t, i, al, T, t0=0.0, win_ms=10, nmin=AL_EVENT_MIN):
    """Onset times (s) and sizes of AL events (see module docstring)."""
    k = al[i] & (t >= round(t0 / STEP))
    ms = (t[k] // 10).astype(int)
    ii = i[k]
    n_ms = int(round(T * 1000))
    if not len(ms):
        return np.array([]), np.array([])
    df = pd.DataFrame(dict(ms=ms, i=ii)).drop_duplicates()
    # distinct neurons in [m, m+win) via per-ms neuron sets
    per = df.groupby('ms').i.apply(set).reindex(range(n_ms + win_ms), fill_value=set())
    sizes = np.array([len(set().union(*per.iloc[m:m + win_ms])) if any(len(s) for s in per.iloc[m:m + win_ms]) else 0
                      for m in range(n_ms)])
    on, sz, last = [], [], -10 ** 9
    for m in np.flatnonzero(sizes >= nmin):
        if m - last > 50:
            on.append(m / 1000); sz.append(sizes[m:m + 20].max())
        last = m
    return np.array(on), np.array(sz)


def word_free(spec, nb, width=0.1):
    return np.array([all(not (j * width < w['offset'] and (j + 1) * width > w['onset']) for w in spec['words'])
                     for j in range(nb)])


def load_meta():
    import pickle
    f = SCRATCH / 'meta.pkl'
    if f.exists():
        d = pickle.load(open(f, 'rb'))
        return d['meta'], d['vnc']
    from sim.cns import cns_model, W4_CALIB
    m = cns_model('cns', 'calib', verbose=False, **W4_CALIB)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    pickle.dump(dict(meta=m['meta'], vnc=m['vnc']), open(f, 'wb'))
    return m['meta'], m['vnc']
