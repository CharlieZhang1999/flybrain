#!/usr/bin/env python
"""W4D5 T2c (Q4): word-free pathology / persistence metric, applied to the saved Day 5
episodes (20 calib mixtures k 0-19, 10 calib + 10 shuffle bg-only k 200-209). No simulation.

Metric (per episode):
  eligible bin   100 ms bin [0.1 j, 0.1 j + 0.1) that overlaps no word window and starts
                 >= GRACE (0.1 s) after the most recent word offset (Week 3b: driven activity
                 returns to silence within 100 ms of offset).
  brain_wf(j)    distinct brain_mask neurons active in the bin, word-driven neurons excluded
                 (as stop_check.py).
  theta          THETA = 200 neurons: above every non-event bg-only bin seen (max 124 at
                 3 Hz class_v2 calib; max 175-282 only under legacy 5 Hz all-sensory bg) and
                 below the pathological levels (AL events 366-440, post-wind state 400-800).
  flags          'persist'  = >= 2 consecutive eligible bins > THETA
                 'burst'    = an eligible bin > THETA that is not part of a 'persist' run
                 AL events (scripts/w4d5_burst_common.al_events) in eligible bins are counted
                 separately, so a burst can be attributed to the antennal lobe.
  score          P_wf = fraction of eligible bins > THETA; also the number of such bins.
Old metric for comparison: any bin > 300 (stop_check.txt).

Usage:  python scripts/w4d5_burst_metric.py [run_dir_glob ...]
        (default: the Day 5 episodes; extra globs, e.g. burst_diag runs, are scored too)
Output: results/w4/regression/burst_diag/metric.csv, metric.txt
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.w4d5_burst_common as C  # noqa: E402

GRACE = 0.1
THETA = 200


def score(t, i, spec, M, meta, pos, label):
    from stimuli.mixtures import word_idx
    T = float(spec['T'])
    nb = int(round(T / 0.1))
    drv = np.zeros(len(meta), bool)
    for w in spec.get('words', []):
        drv[word_idx(meta, pos, w['word'])] = True
    b = C.bin_counts(t, i, M['brain'] & ~drv, T)
    elig = np.ones(nb, bool)
    for j in range(nb):
        a, z = j * 0.1, (j + 1) * 0.1
        for w in spec.get('words', []):
            if a < w['offset'] + GRACE - 1e-9 and z > w['onset']:
                elig[j] = False
    hi = elig & (b > THETA)
    persist = np.zeros(nb, bool)
    j = 0
    while j < nb:
        if hi[j]:
            k = j
            while k + 1 < nb and hi[k + 1]:
                k += 1
            if k > j:
                persist[j:k + 1] = True
            j = k + 1
        else:
            j += 1
    on, _ = C.al_events(t, i, M['al'], T)
    ev_bins = sorted({int(o // 0.1) for o in on})
    ev_elig = [e for e in ev_bins if e < nb and elig[e]]
    words = '+'.join(w['word'] for w in spec.get('words', [])) or '(background)'
    return dict(run=label, words=words, bg_hz=spec.get('bg_hz'), n_bins=nb, n_eligible=int(elig.sum()),
                n_hi=int(hi.sum()), P_wf=round(hi.sum() / max(elig.sum(), 1), 3),
                persist_bins=int(persist.sum()), burst_bins=int((hi & ~persist).sum()),
                al_events=len(on), al_events_eligible=len(ev_elig),
                old_any_gt300=int((b > 300).any()), max_eligible=int(b[elig].max()) if elig.any() else 0,
                flag=('persist' if persist.any() else 'burst' if hi.any() else 'ok'))


def main(extra):
    meta, vnc = C.load_meta()
    M = C.masks(meta, vnc)
    pos = {b_: k for k, b_ in enumerate(meta.bodyId)}
    rows = []
    idx = pd.read_csv(C.DAY5 / 'index.csv')
    for r in idx.itertuples():
        t, i, spec = C.load_run(C.DAY5 / r.file)
        rows.append(dict(score(t, i, spec, M, meta, pos, f'day5 ep{r.episode:03d}'), cfg=r.cfg, kind=r.kind))
    for g in extra:
        for f in sorted(glob.glob(g)):
            t, i, spec = C.load_run(f)
            if 'T' not in spec:
                continue
            rows.append(dict(score(t, i, spec, M, meta, pos, Path(f).stem),
                             cfg=spec.get('cfg', 'calib'), kind='diag'))
    df = pd.DataFrame(rows)
    df.to_csv(C.OUT / 'metric.csv', index=False)
    lines = [f'# Word-free pathology metric (GRACE {GRACE} s, THETA {THETA}); see script docstring', '',
             df.to_string(index=False), '']
    for (cfg, kind), g in df[df.kind != 'diag'].groupby(['cfg', 'kind']):
        lines.append(f'SUMMARY {cfg} {kind}: n={len(g)}; flag persist {int((g.flag == "persist").sum())}, '
                     f'burst {int((g.flag == "burst").sum())}, ok {int((g.flag == "ok").sum())}; '
                     f'old metric (any bin > 300) {int(g.old_any_gt300.sum())}; '
                     f'mean P_wf {g.P_wf.mean():.3f}; eligible bins {int(g.n_eligible.sum())}')
    (C.OUT / 'metric.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main(sys.argv[1:])
