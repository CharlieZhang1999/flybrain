#!/usr/bin/env python
"""W4D5 T2e: analysis of the LOOP_GAIN verification runs (no simulation).

Parts (inputs: results/w4/regression/loop_gain/runs/, written by w4d5_loopgain_runs.py)
  wind   per Day 5 wind episode, new W4_CALIB vs the saved Day 5 run (old W4_CALIB, k < 20;
         day5/pre_loopgain/ once moved):
           persist_s     last DNge027 spike - wind offset (s; T 2.5 s caps it), as T2c/T2d
           d27_bins      DNge027 spikes per 100 ms bin from wind offset to T (the check)
           tail_bins_s   0.1 x number of consecutive non-empty bins from wind offset
           DNge027_Hz    DNge027 rate in the wind word (per cell)
           MN_win        motor-neuron spikes in the wind window
           brain_win     brain_mask neurons active in the wind window, word-driven excluded
           brain_last    brain_mask neurons active 2.4-2.5 s, word-driven excluded
           same_as_T2c_g0.3  spike-identical to the T2c gain-scan run (burst_diag), where one exists
  mix    non-wind Day 5 episodes: spike-identical to the saved Day 5 run?
  bg     class_v2 3 Hz bg-only: identical to results/w4/bg_fix/runs/calib_v2_3Hz_s*.npz? plus
         the bg_fix metrics (scripts/w4d5_bg_fix_runs.metrics) vs verify.csv
  loom   looming 0-0.4 s: brain / VNC active per 100 ms bin, old vs new; identical?
  windalone  wind 0-0.5 s: same, plus last DNge027 spike - 0.5 s
Usage:  python scripts/w4d5_loopgain_analyze.py <part> [<part> ...]
Output: results/w4/regression/loop_gain/analyze_<part>.txt (+ .csv for wind)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'results' / 'w4' / 'regression' / 'loop_gain'
RUNS = OUT / 'runs'
DAY5 = ROOT / 'results' / 'w4' / 'regression' / 'day5'
BGFIX = ROOT / 'results' / 'w4' / 'bg_fix'
STEP = 1e-4


def ctx():
    import pickle
    f = Path('/tmp/claude-0/t2e_meta.pkl')
    if f.exists():
        meta, vnc = pickle.load(open(f, 'rb'))
    else:
        from sim.malecns import load
        from sim.cns import regions
        meta, _ = load(w_syn=1.0, monoamine='removed')
        vnc, _ = regions(meta)
        f.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump((meta, vnc), open(f, 'wb'))
    from sim.cns import brain_mask
    ty = meta.type.fillna('?').astype(str).values
    sc = meta.superclass.fillna('?').astype(str).values
    return dict(meta=meta, vnc=vnc, pos={b: k for k, b in enumerate(meta.bodyId)},
                d27=ty == 'DNge027', mn=sc == 'vnc_motor', brain=brain_mask(meta, vnc),
                sens=pd.Series(sc).str.contains('sensory').values)


def load(f):
    z = np.load(f, allow_pickle=True)
    return z['t'].astype(np.int64), z['i'].astype(np.int64), json.loads(str(z['spec']))


def baseline(k):
    """Saved Day 5 calib run of episode k (old W4_CALIB): moved to day5/pre_loopgain/ by T2e."""
    for d in (DAY5 / 'pre_loopgain', DAY5):
        if (d / f'ep{k:03d}_calib.npz').exists():
            return d / f'ep{k:03d}_calib.npz'
    return None


def identical(a, b):
    return len(a[0]) == len(b[0]) and np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def active_bins(t, i, mask, T, width=0.1):
    nb = int(round(T / width)); b = t // round(width / STEP)
    return [len(np.unique(i[mask[i] & (b == j)])) for j in range(nb)]


def wind_metrics(C, t, i, s):
    from stimuli.mixtures import word_idx
    w = [x for x in s['words'] if x['word'] == 'wind'][0]
    on, off, T = w['onset'], w['offset'], s['T']
    drv = np.zeros(len(C['meta']), bool)
    for x in s['words']:
        drv[word_idx(C['meta'], C['pos'], x['word'])] = True
    inw = (t >= on / STEP) & (t < off / STEP)
    d = C['d27'][i]
    last = t[d].max() * STEP if d.any() else np.nan
    nb = int(np.ceil((T - off) / 0.1 - 1e-9))
    bins = [int(((t[d] >= round((off + 0.1 * j) / STEP)) & (t[d] < round(min(off + 0.1 * (j + 1), T) / STEP))).sum())
            for j in range(nb)]
    run = next((j for j, c in enumerate(bins) if c == 0), nb)
    return dict(wind=f'{on:.2f}-{off:.2f}', persist_s=round(last - off, 3), tail_bins_s=round(0.1 * run, 1),
                d27_bins=' '.join(map(str, bins)),
                DNge027_Hz=round((inw & d).sum() / (off - on) / C['d27'].sum(), 1),
                MN_win=int((inw & C['mn'][i]).sum()),
                brain_win=len(np.unique(i[inw & C['brain'][i] & ~drv[i]])),
                brain_last=active_bins(t, i, C['brain'] & ~drv, T)[-1],
                spikes=len(t))


def part_wind(C):
    rows = []
    for f in sorted(RUNS.glob('wind__ep*.npz')):
        k = int(f.stem.split('ep')[1])
        t, i, s = load(f)
        words = '+'.join(x['word'] for x in s['words'])
        row = dict(ep=k, words=words, **wind_metrics(C, t, i, s))
        b = baseline(k)
        if b is not None:
            tb, ib, sb = load(b)
            assert {kk: v for kk, v in s.items() if kk != 'episode'} == sb, f'spec mismatch ep{k}'
            mb = wind_metrics(C, tb, ib, sb)
            row.update({f'{kk}_d5': mb[kk] for kk in ('persist_s', 'd27_bins', 'DNge027_Hz', 'MN_win',
                                                     'brain_win', 'brain_last')})
            row['MN_win_change_pct'] = round(100 * (row['MN_win'] / mb['MN_win'] - 1), 1)
            row['brain_win_change_pct'] = round(100 * (row['brain_win'] / mb['brain_win'] - 1), 1)
        g = ROOT / 'results' / 'w4' / 'regression' / 'burst_diag' / 'runs' / f'gain__g0.3_ep{k:03d}.npz'
        if g.exists():                    # T2c scan run on a copy of W (LOOP5 x 0.3)
            row['same_as_T2c_g0.3'] = identical((t, i), load(g))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'analyze_wind.csv', index=False)
    out = [__doc__.split('Parts')[1].split('mix ')[0].strip(), '',
           '_d5 = saved Day 5 run (old W4_CALIB). Columns without suffix = new W4_CALIB.', '',
           df.to_string(index=False), '',
           f'n episodes {len(df)}; persist_s max {df.persist_s.max():.3f} (ep{int(df.ep[df.persist_s.idxmax()]):03d}); '
           f'n > 0.15 s: {int((df.persist_s > 0.15).sum())}; n > 0.3 s: {int((df.persist_s > 0.3).sum())}; '
           f'tail_bins_s max {df.tail_bins_s.max():.1f}']
    if 'persist_s_d5' in df:
        d = df.dropna(subset=['persist_s_d5'])
        out.append(f'baselines (k < 20): n {len(d)}; saved persist_s > 0.3 s in {int((d.persist_s_d5 > 0.3).sum())}; '
                   f'MN_win change {d.MN_win_change_pct.min():.1f} to {d.MN_win_change_pct.max():.1f}% '
                   f'(median {d.MN_win_change_pct.median():.1f}); brain_win change '
                   f'{d.brain_win_change_pct.min():.1f} to {d.brain_win_change_pct.max():.1f}%')
    return out


def part_mix(C):
    out = ['Non-wind Day 5 episodes, new W4_CALIB vs saved Day 5 (old W4_CALIB).', '']
    for f in sorted(RUNS.glob('mix__ep*.npz')):
        k = int(f.stem.split('ep')[1])
        a = load(f); b = load(baseline(k))
        words = '+'.join(x['word'] for x in a[2]['words'])
        d27 = int(C['d27'][a[1]].sum()), int(C['d27'][b[1]].sum())
        out.append(f'ep{k:03d} {words}: spikes new {len(a[0]):,} / saved {len(b[0]):,}; identical={identical(a, b)}; '
                   f'DNge027 spikes new {d27[0]} / saved {d27[1]}')
    return out


def part_bg(C):
    from scripts.w4d5_bg_fix_runs import groups, metrics
    from stimuli.mixtures import to_blocks
    g = groups(C['meta'])
    ver = pd.read_csv(BGFIX / 'verify.csv')
    ver = ver[(ver.cfg == 'calib') & (ver.scheme == 'v2')]
    out = ['class_v2 3 Hz bg-only, T 1.0 s, new W4_CALIB vs results/w4/bg_fix (old W4_CALIB).', '']
    rows = []
    for s in (0, 1, 2):
        a = load(RUNS / f'bg__v2_3Hz_s{s}.npz')
        b = load(BGFIX / 'runs' / f'calib_v2_3Hz_s{s}.npz')
        drv = np.zeros(len(C['meta']), bool)
        for bl in to_blocks(a[2], C['meta'], C['pos']):
            drv[bl[2]] = True
        m = metrics(g, a[0].astype(np.int32), a[1].astype(np.int32), drv)
        out.append(f's{s}: spikes new {len(a[0]):,} / bg_fix {len(b[0]):,}; identical={identical(a, b)}; '
                   f'DNge027 spikes new {int(C["d27"][a[1]].sum())} / bg_fix {int(C["d27"][b[1]].sum())}')
        rows.append(dict(src='new', seed=s, **m))
    cols = ['mn_active', 'mn_spikes', 'ns_active', 'vnc_ns_active', 'brain_ns_active', 'ns_per100',
            'vnc_ns_per100', 'brain_ns_per100', 'driven_rate_hz']
    new = pd.DataFrame(rows)[['src', 'seed'] + cols]
    old = ver.assign(src='verify.csv')[['src', 'seed'] + cols]
    out += ['', pd.concat([old, new]).to_string(index=False)]
    return out


def part_single(C, mode, t_off):
    out = [f'{mode}: no background, word 0-{t_off} s at 150 Hz. Active neurons per 100 ms bin '
           '(brain = ~VNC incl. sensory, as w4d3_looming_tail; brain_mask = non-sensory brain). old = loop_gain=None.', '']
    rows = []
    for s in (0, 1, 2):
        r = {v: load(RUNS / f'{mode}__{v}_s{s}.npz') for v in ('old', 'new')}
        out.append(f's{s}: spikes old {len(r["old"][0]):,} / new {len(r["new"][0]):,}; identical={identical(r["old"], r["new"])}')
        for v, (t, i, sp) in r.items():
            d = C['d27'][i]
            last_any = t.max() * STEP if len(t) else np.nan
            rows.append(dict(model=v, seed=s, brain=active_bins(t, i, ~C['vnc'], sp['T']),
                             vnc=active_bins(t, i, C['vnc'], sp['T']),
                             brain_mask=active_bins(t, i, C['brain'], sp['T']),
                             d27_last_minus_off=round(t[d].max() * STEP - t_off, 3) if d.any() else None,
                             last_spike_s=round(last_any, 4)))
    out += ['', pd.DataFrame(rows).to_string(index=False)]
    return out


if __name__ == '__main__':
    parts = dict(wind=part_wind, mix=part_mix, bg=part_bg, loom=lambda C: part_single(C, 'loom', 0.4),
                 windalone=lambda C: part_single(C, 'windalone', 0.5))
    args = sys.argv[1:]
    if not args or any(a not in parts for a in args):
        sys.exit(__doc__)
    C = ctx()
    for a in args:
        lines = parts[a](C)
        txt = '\n'.join(map(str, lines)) + '\n'
        (OUT / f'analyze_{a}.txt').write_text(txt)
        print(txt)
