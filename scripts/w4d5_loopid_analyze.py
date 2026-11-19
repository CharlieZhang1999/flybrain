#!/usr/bin/env python
"""W4D5 T2d: analysis of the suspect-test runs (scripts/w4d5_loopid_runs.py) against the saved Day 5 runs.

Purpose: per variant x episode, report
  - persistence after wind offset: last DNge027 spike minus wind offset (s), and the brain bin
    count (brain_mask neurons active per 100 ms) in the bin 2.4-2.5 s;
  - DNge027 rate (Hz, per cell) and LOOP5 mean rate during the wind word;
  - MN spikes (vnc_motor) and active brain neurons during the wind word, with % change vs saved;
  - for non-wind episodes: total spikes and whether the run is spike-for-spike identical to saved.
Baseline = results/w4/regression/day5/ep<k>_calib.npz (same model, spec and seed).
Usage:  python scripts/w4d5_loopid_analyze.py
Output: results/w4/regression/loop_id/analyze_runs.txt and analyze_runs.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'results' / 'w4' / 'regression' / 'loop_id'
DAY5 = ROOT / 'results' / 'w4' / 'regression' / 'day5'
LOOP5 = ['DNge027', 'GNG423', 'DNg21', 'DNg62', 'DNge044']
STEP = 1e-4


def main():
    from sim.cns import regions, brain_mask
    meta = pd.read_parquet(ROOT / 'data' / 'malecns' / 'malecns_traced_meta.parquet')
    vnc, _ = regions(meta)
    bm = brain_mask(meta, vnc)
    ty = meta.type.fillna('?').astype(str).values
    d27 = ty == 'DNge027'
    l5 = np.isin(ty, LOOP5)
    mn = meta.superclass.astype(str).values == 'vnc_motor'
    specs = json.loads((DAY5 / 'specs.json').read_text())
    rows = []

    def stats(f, k):
        z = np.load(f, allow_pickle=True)
        t, i = z['t'].astype(np.int64), z['i'].astype(np.int64)
        sp = specs[k]
        T = sp['T']
        w = [x for x in sp['words'] if x['word'] == 'wind']
        r = dict(spikes=len(t))
        lb = (t >= round(2.4 / STEP)) & bm[i]
        r['brain_2.4_2.5'] = len(np.unique(i[lb]))
        if w:
            on, off = w[0]['onset'], w[0]['offset']
            a, b = round(on / STEP), round(off / STEP)
            inw = (t >= a) & (t < b)
            r['DNge027_Hz_word'] = (inw & d27[i]).sum() / d27.sum() / (off - on)
            r['LOOP5_Hz_word'] = (inw & l5[i]).sum() / l5.sum() / (off - on)
            r['MN_spikes_word'] = int((inw & mn[i]).sum())
            r['brain_active_word'] = len(np.unique(i[inw & bm[i]]))
            td = t[d27[i]]
            r['last_DNge027_after_off_s'] = (td.max() * STEP - off) if len(td) and td.max() >= b else np.nan
            r['wind_offset'] = off
            r['DNge027_spikes_after_off'] = int(((t >= b) & d27[i]).sum())
        else:
            r['DNge027_spikes_total'] = int(d27[i].sum())
        return r, t, i

    base = {}
    for k in sorted({int(f.stem.split('_ep')[1]) for f in (OUT / 'runs').glob('*_ep*.npz')}):
        r, t, i = stats(DAY5 / f'ep{k:03d}_calib.npz', k)
        base[k] = (r, t, i)
        rows.append(dict(variant='saved', ep=k, **r))
    for f in sorted((OUT / 'runs').glob('*_ep*.npz')):
        v, k = f.stem.split('_ep')[0], int(f.stem.split('_ep')[1])
        r, t, i = stats(f, k)
        b = base[k]
        r['identical_to_saved'] = bool(len(t) == len(b[1]) and np.array_equal(t, b[1]) and np.array_equal(i, b[2]))
        if 'MN_spikes_word' in r:
            r['MN_word_pct'] = 100 * (r['MN_spikes_word'] / b[0]['MN_spikes_word'] - 1)
            r['brain_word_pct'] = 100 * (r['brain_active_word'] / b[0]['brain_active_word'] - 1)
        rows.append(dict(variant=v, ep=k, **r))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'analyze_runs.csv', index=False)
    with pd.option_context('display.width', 300, 'display.max_columns', 30, 'display.float_format', '{:.3f}'.format):
        txt = df.to_string(index=False)
    # post-offset time course (100 ms bins from wind offset): brain_mask count and DNge027 spikes
    tc = ['', 'post-offset time course, 100 ms bins from wind offset: brain neurons active | DNge027 spikes']
    for k in sorted(base):
        w = [x for x in specs[k]['words'] if x['word'] == 'wind']
        if not w:
            continue
        off = w[0]['offset']
        for v, f in [('saved', DAY5 / f'ep{k:03d}_calib.npz')] + \
                [(g.stem.split('_ep')[0], g) for g in sorted((OUT / 'runs').glob(f'*_ep{k:03d}.npz'))]:
            z = np.load(f, allow_pickle=True)
            t, i = z['t'] * STEP, z['i']
            ed = np.arange(off, specs[k]['T'] - 1e-9, 0.1)
            b = [len(np.unique(i[(t >= a) & (t < a + 0.1) & bm[i]])) for a in ed]
            d = [int(((t >= a) & (t < a + 0.1) & d27[i]).sum()) for a in ed]
            tc.append(f'ep{k:03d} {v:11s} brain {b} | DNge027 {d}')
    txt += '\n' + '\n'.join(tc)
    (OUT / 'analyze_runs.txt').write_text(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
