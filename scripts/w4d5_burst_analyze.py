#!/usr/bin/env python
"""W4D5 T2c: analysis of the burst-diagnosis runs (no simulation).

Parts
  census   AL events and max brain bin in every existing background run (bg_diag, bg_fix,
           QA runs if present, Day 5 bg-only, burst_diag al_* runs).
  al       AL burst rate x seed table (al_rate), head/olfactory table (al_head), ablations
           (al_ablate) against the saved Day 5 baseline of the same spec.
  wind     wind test matrix: brain (word-driven excluded) per 100 ms bin, tail activity,
           GNG core / optic-lobe intrinsic / VNC non-sensory / MN active after wind offset.
  gain     loop-gain scan (revision): per run persistence after wind offset (last DNge027 /
           LOOP5 spike - wind offset), DNge027 and LOOP5 rates during the wind word, MN spikes and
           brain neurons active (word-driven excluded) during the wind window, vs the saved Day 5
           run (gain 1.0); for non-wind runs the same counts over all word windows plus the
           per-neuron spike-count correlation with the saved run.
  lesion   resume 2.0-2.5 s variants: active counts 2.2-2.5 s by group; identity check of the
           intact resume against the saved Day 5 ep009.
Usage:  python scripts/w4d5_burst_analyze.py <part> [<part> ...]
Output: results/w4/regression/burst_diag/analyze_<part>.txt
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
from scripts.w4d5_burst_runs import GNG_CORE  # noqa: E402

RUNS = C.OUT / 'runs'


def ctx():
    meta, vnc = C.load_meta()
    M = C.masks(meta, vnc)
    ty = meta.type.fillna('?').astype(str).values
    sc = meta.superclass.fillna('?').astype(str).values
    sens = pd.Series(sc).str.contains('sensory').values
    M.update(core=np.isin(ty, GNG_CORE), ol=sc == 'ol_intrinsic', vnc_ns=vnc & ~sens,
             mn=sc == 'vnc_motor')
    pos = {b: k for k, b in enumerate(meta.bodyId)}
    return meta, vnc, M, pos


def n_active(t, i, mask, a, b):
    k = (t >= round(a / C.STEP)) & (t < round(b / C.STEP)) & mask[i]
    return len(np.unique(i[k]))


def part_census(meta, vnc, M, pos):
    fs = []
    for g in ['results/w4/regression/bg_diag/runs/*.npz', 'results/w4/bg_fix/runs/*.npz',
              'results/w4/regression/day5/ep2*_*.npz']:
        fs += sorted(glob.glob(str(ROOT / g)))
    fs += sorted(glob.glob('/tmp/claude-0/qa2/runs/*.npz'))
    fs += sorted(glob.glob(str(RUNS / 'al_*.npz')))
    out = []
    for f in fs:
        t, i, s = C.load_run(f)
        T = float(s.get('T', 1.0)) if s else 1.0
        on, sz = C.al_events(t, i, M['al'], T)
        b = C.bin_counts(t, i, M['brain'], T)
        out.append(f"{'/'.join(Path(f).parts[-3:]):62s} T {T:.1f} AL events {len(on)} "
                   f"at {np.round(on, 2).tolist()} | brain bin max {b.max()} median {int(np.median(b))}")
    return out


def part_al(meta, vnc, M, pos):
    out = ['## al_rate (class_v2 bg only, T 2.0 s): AL events per run', '']
    rows = []
    for f in sorted(RUNS.glob('al_rate__*.npz')):
        t, i, s = C.load_run(f)
        on, _ = C.al_events(t, i, M['al'], 2.0)
        b = C.bin_counts(t, i, M['brain'], 2.0)
        rows.append(dict(hz=s['bg_hz'], seed=s['seed'], events=len(on), onsets=np.round(on, 2).tolist(),
                         brain_max=b.max(), brain_median=int(np.median(b))))
    if rows:
        df = pd.DataFrame(rows)
        out += [df.to_string(index=False), '',
                df.pivot_table(index='hz', columns='seed', values='events').to_string(), '']
    out += ['## al_head (2.5 Hz, T 2.0 s)', '']
    rows = []
    for f in sorted(RUNS.glob('al_head__*.npz')):
        t, i, s = C.load_run(f)
        on, _ = C.al_events(t, i, M['al'], 2.0)
        b = C.bin_counts(t, i, M['brain'], 2.0)
        rows.append(dict(variant=s['variant'], seed=s['seed'], n_driven=s['n_driven'], events=len(on),
                         onsets=np.round(on, 2).tolist(), brain_max=b.max(), brain_median=int(np.median(b))))
    if rows:
        out += [pd.DataFrame(rows).to_string(index=False), '']
    out += ['## al_ablate (Day 5 bg-only spec, same Poisson seed; baseline = saved Day 5 run)', '']
    rows = []
    for k in (205, 207):
        t, i, s = C.load_run(C.DAY5 / f'ep{k}_calib.npz')
        on, _ = C.al_events(t, i, M['al'], 2.5)
        b = C.bin_counts(t, i, M['brain'], 2.5)
        rows.append(dict(spec=f'ep{k}', ablation='none (saved)', events=len(on), onsets=np.round(on, 2).tolist(),
                         brain_max=b.max(), brain_median=int(np.median(b))))
        for f in sorted(RUNS.glob(f'al_ablate__ep{k}_*.npz')):
            t, i, s = C.load_run(f)
            on, _ = C.al_events(t, i, M['al'], 2.5)
            b = C.bin_counts(t, i, M['brain'], 2.5)
            rows.append(dict(spec=f'ep{k}', ablation=s['ablation'], events=len(on), onsets=np.round(on, 2).tolist(),
                             brain_max=b.max(), brain_median=int(np.median(b))))
    out += [pd.DataFrame(rows).to_string(index=False)]
    return out


def part_wind(meta, vnc, M, pos):
    from stimuli.mixtures import word_idx
    out = ['brain = brain_mask active per 100 ms bin, word-driven neurons excluded. '
           'tail = from wind offset + 0.1 s (or last offset + 0.1 s for "last") to T. '
           'core = GNG core types (GNG_CORE), OL = ol_intrinsic, VNC = VNC non-sensory, MN = motor; '
           'counts are distinct neurons active in the last 0.3 s of the run.', '']
    rows = []
    for f in sorted(RUNS.glob('wind__*.npz')):
        t, i, s = C.load_run(f)
        T = s['T']
        drv = np.zeros(len(meta), bool)
        for w in s['words']:
            drv[word_idx(meta, pos, w['word'])] = True
        b = C.bin_counts(t, i, M['brain'] & ~drv, T)
        toff = s['switch_s']
        lo = max(w['offset'] for w in s['words'])
        tail_bins = [j for j in range(len(b)) if j * 0.1 >= lo + 0.1 - 1e-9]
        rows.append(dict(run=f.stem.split('__')[1], cond=s['condition'], wind_off=toff, last_off=lo,
                         tail_brain=b[tail_bins].tolist(),
                         core=n_active(t, i, M['core'], T - 0.3, T), OL=n_active(t, i, M['ol'], T - 0.3, T),
                         VNC=n_active(t, i, M['vnc_ns'], T - 0.3, T), MN=n_active(t, i, M['mn'], T - 0.3, T),
                         brain_bins=b.tolist()))
    df = pd.DataFrame(rows)
    out += [df.drop(columns='brain_bins').to_string(index=False), '']
    for r in rows:
        out.append(f"{r['run']}: brain per bin {r['brain_bins']}")
    return out


def part_lesion(meta, vnc, M, pos):
    out = ['Resume at 2.0 s from Day 5 ep009 (bg on), variants applied 2.0-2.5 s. '
           'Counts = distinct neurons active 2.2-2.5 s.', '']
    t0, i0, _ = C.load_run(C.DAY5 / 'ep009_calib.npz')
    rows = []
    for f in sorted(RUNS.glob('lesion__*.npz')):
        t, i, s = C.load_run(f)
        rows.append(dict(variant=s['lesion'], brain=n_active(t, i, M['brain'], 2.2, 2.5),
                         core=n_active(t, i, M['core'], 2.2, 2.5), OL=n_active(t, i, M['ol'], 2.2, 2.5),
                         VNC=n_active(t, i, M['vnc_ns'], 2.2, 2.5), MN=n_active(t, i, M['mn'], 2.2, 2.5),
                         brain_bins_2to25=C.bin_counts(t, i, M['brain'], 2.5)[20:].tolist()))
        if s['lesion'] == 'intact':
            same = len(t) == len(t0) and np.array_equal(t, t0) and np.array_equal(i, i0)
            out.append(f'intact resume identical to saved ep009: {same} ({len(t):,} vs {len(t0):,} spikes)')
    rows.append(dict(variant='saved ep009', brain=n_active(t0, i0, M['brain'], 2.2, 2.5),
                     core=n_active(t0, i0, M['core'], 2.2, 2.5), OL=n_active(t0, i0, M['ol'], 2.2, 2.5),
                     VNC=n_active(t0, i0, M['vnc_ns'], 2.2, 2.5), MN=n_active(t0, i0, M['mn'], 2.2, 2.5),
                     brain_bins_2to25=C.bin_counts(t0, i0, M['brain'], 2.5)[20:].tolist()))
    out += ['', pd.DataFrame(rows).to_string(index=False)]
    return out


def part_gain(meta, vnc, M, pos):
    from stimuli.mixtures import word_idx
    from scripts.w4d5_burst_runs import LOOP5
    ty = meta.type.fillna('?').astype(str).values
    d27 = ty == 'DNge027'; L5 = np.isin(ty, LOOP5)
    rows = []
    files = sorted(RUNS.glob('gain__*.npz'))
    eps = sorted({int(f.stem.split('_ep')[1]) for f in files})
    for k in eps:
        runs = [('1.0 (saved Day 5)', C.DAY5 / f'ep{k:03d}_calib.npz')] + \
               [(f.stem.split('__')[1].split('_')[0][1:], f) for f in files if f.stem.endswith(f'ep{k:03d}')]
        base = None
        for g, f in runs:
            t, i, s = C.load_run(f)
            ws = [w for w in s['words'] if w['word'] == 'wind']
            drv = np.zeros(len(meta), bool)
            for w in s['words']:
                drv[word_idx(meta, pos, w['word'])] = True
            win = [(w['onset'], w['offset']) for w in (ws or s['words'])]
            inw = np.zeros(len(t), bool)
            for a, b in win:
                inw |= (t >= a / C.STEP) & (t < b / C.STEP)
            dur = sum(b - a for a, b in win)
            cnt = np.bincount(i, minlength=len(meta))
            row = dict(ep=k, gain=g, words='+'.join(w['word'] for w in s['words']))
            if ws:
                off = ws[0]['offset']
                ld = t[d27[i]].max() * C.STEP if d27[i].any() else np.nan
                ll = t[L5[i]].max() * C.STEP if L5[i].any() else np.nan
                row.update(wind=f"{ws[0]['onset']:.2f}-{off:.2f}", persist_DNge027_s=round(ld - off, 3),
                           persist_loop5_s=round(ll - off, 3))
            row.update(DNge027_Hz=round((inw & d27[i]).sum() / dur / 2, 1),
                       loop5_Hz=round((inw & L5[i]).sum() / dur / L5.sum(), 1),
                       MN_spikes_win=int((inw & M['mn'][i]).sum()),
                       brain_active_win=len(np.unique(i[inw & M['brain'][i] & ~drv[i]])),
                       brain_last_bin=C.bin_counts(t, i, M['brain'] & ~drv, s['T'])[-1])
            if base is None:
                base = cnt
            else:
                nz = (base > 0) | (cnt > 0)
                row['count_corr_vs_g1'] = round(float(np.corrcoef(base[nz], cnt[nz])[0, 1]), 4)
            rows.append(row)
    return ['Windows: the wind word (wind episodes) or all word windows (non-wind). '
            'persist = last spike - wind offset (s; T 2.5 s caps it). brain_last_bin = brain (word-driven '
            'excluded) active 2.4-2.5 s. count_corr = per-neuron spike-count correlation with gain 1.0 (whole run).',
            '', pd.DataFrame(rows).to_string(index=False)]


if __name__ == '__main__':
    parts = dict(gain=part_gain, census=part_census, al=part_al, wind=part_wind, lesion=part_lesion)
    args = sys.argv[1:]
    if not args or any(a not in parts for a in args):
        sys.exit(__doc__)
    c = ctx()
    for a in args:
        lines = parts[a](*c)
        (C.OUT / f'analyze_{a}.txt').write_text('\n'.join(lines) + '\n')
        print('\n'.join(lines))
