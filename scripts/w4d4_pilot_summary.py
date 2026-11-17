#!/usr/bin/env python
"""W4D4: summarise the regression pilot (scripts/w4d4_pilot.py), real vs shuffled.

Bins every episode at 10 ms and reports:
  1. motor-pool variance: per pool (sim.cns.motor_pools, 349 pools) the
     across-bin variance of the pool rate (Hz per neuron), pooled over the 10
     mixture episodes (2,000 bins); pools above a variance floor, real vs
     shuffled; the same without the looming episodes; the share of the summed
     variance that comes from looming episodes; the top pools; GF / TTMn / DLMn
     rates inside each looming window against the looming drive rate.
     Floor = 1 Hz^2 (SD 1 Hz). Reason: in a 10 ms bin one spike of one neuron
     is 100 Hz, so a 1-neuron pool that spikes in a single bin of the 2,000
     already has variance 5 Hz^2. A floor of 1 Hz^2 asks for more than a
     stray spike in small pools, while a pool of up to ~20 neurons can pass with
     a handful of spikes. 0 Hz^2 (any spike) and 10 Hz^2 are shown as sensitivity.
  2. the same at coarse level: pool group x side (42 groups);
  3. brain-input activity: brain_mask neurons with any spike, per episode
     (with and without the neurons that a word drives directly: LPLC2/LC4 for
     looming are in brain_mask);
  4. rest level, background-only episodes: active non-sensory neurons per
     100 ms bin (bins from 0.1 s on), brain / VNC split, top cell types;
     plus, if present, the bgoff run (background switched off at 1.5 s);
  5. persistence flag: active non-sensory neurons per 100 ms window, in
     windows that start >100 ms after the episode's last word ends, against the
     background-only baseline of the same cfg; flagged if the tail mean is
     above 2x the largest background-only bin;
  6. wall seconds per bio-second (index_{cfg}.csv).
  Also (section 0): the number of sensory neurons that get background, and the
  share of sampled specs (seeds 0-4999) that contain looming.
Usage: python scripts/w4d4_pilot_summary.py
Output: results/w4/regression/pilot/pilot_summary.md (+ pool_variance.csv, group_variance.csv)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.cns import cns_model, motor_pools, brain_mask, W4_CALIB  # noqa: E402
from sim.presyn import sensory_mask_malecns  # noqa: E402
from stimuli.mixtures import word_idx, last_offset, sample_episode  # noqa: E402

PILOT = ROOT / 'results' / 'w4' / 'regression' / 'pilot'
CFGS = ('calib', 'shuffle')
BIN = 100            # steps = 10 ms
WIN = 1000           # steps = 100 ms
DT = 1e-4
FLOOR = 1.0
FLOORS = (0.0, 1.0, 10.0)


def load_eps(cfg):
    idx = pd.read_csv(PILOT / f'index_{cfg}.csv')
    eps = []
    for r in idx.itertuples():
        d = np.load(PILOT / r.file)
        eps.append(dict(ep=r.ep, kind=r.kind, t=d['t'].astype(np.int64), i=d['i'].astype(np.int64),
                        spec=json.loads(str(d['spec']))))
    return idx, eps


def binned(t, i, nsteps, gid, ngroups, width):
    """(n_bins, ngroups) spike counts; gid maps neuron -> group (-1 = ignore)."""
    nb = -(-nsteps // width)
    g = gid[i]; k = g >= 0
    out = np.zeros((nb, ngroups))
    np.add.at(out, (t[k] // width, g[k]), 1)
    return out


def var_table(eps, gid, sizes, names):
    """Per group: variance across 10 ms bins of rate (Hz/neuron), pooled over eps."""
    rates, loom = [], []
    for e in eps:
        ns = int(round(e['spec']['T'] / DT))
        c = binned(e['t'], e['i'], ns, gid, len(sizes), BIN)
        rates.append(c / (sizes * BIN * DT))
        loom.append(np.full(len(c), any(w['word'] == 'looming' for w in e['spec']['words'])))
    R, L = np.vstack(rates), np.concatenate(loom)
    dev2 = (R - R.mean(0)) ** 2
    df = pd.DataFrame(dict(name=names, n=sizes, mean_hz=R.mean(0), var=R.var(0),
                           var_noloom=R[~L].var(0), loom_share=dev2[L].sum(0) / np.maximum(dev2.sum(0), 1e-30)))
    return df, L


def active_per_window(e, mask, width=WIN, t_from=0.0, t_to=None):
    """Distinct active neurons (in mask) per window of `width` steps in [t_from, t_to)."""
    T = e['spec']['T']
    t_to = T if t_to is None else t_to
    s0, s1 = int(round(t_from / DT)), int(round(t_to / DT))
    out = []
    for a in range(s0, s1 - width + 1, width):
        k = (e['t'] >= a) & (e['t'] < a + width) & mask[e['i']]
        out.append(len(np.unique(e['i'][k])))
    return np.array(out)


def main():
    m = cns_model('cns', 'calib', verbose=False, **W4_CALIB)
    meta, vnc = m['meta'], m['vnc']
    del m
    pos = {b: i for i, b in enumerate(meta.bodyId)}
    n = len(meta)
    sens = sensory_mask_malecns(meta)
    bm = brain_mask(meta, vnc)
    nonsens = ~sens
    is_mn = meta.superclass.astype(str).values == 'vnc_motor'
    types = meta.type.fillna('?').astype(str).values
    pools, pdf = motor_pools(meta)
    names = list(pools)
    gid = np.full(n, -1)
    for k, nm in enumerate(names):
        gid[pools[nm]] = k
    sizes = np.array([len(pools[nm]) for nm in names], float)
    pdf = pdf.set_index('pool').loc[names].reset_index()
    pdf['coarse'] = pdf.group + ' | ' + pdf.side
    cnames = sorted(pdf.coarse.unique())
    cid = {c: k for k, c in enumerate(cnames)}
    gid_c = np.full(n, -1)
    for nm, c in zip(names, pdf.coarse):
        gid_c[pools[nm]] = cid[c]
    csizes = np.bincount(gid_c[gid_c >= 0], minlength=len(cnames)).astype(float)

    L = []                               # markdown lines
    L += ['# W4D4 regression pilot: summary', '',
          'Generated by `scripts/w4d4_pilot_summary.py` from `results/w4/regression/pilot/`. '
          '10 mixture episodes + 2 background-only, 2 s each, cfg calib (real) and shuffle, '
          'W4_CALIB, CPU float32 rng=torch. 10 ms bins (2,000 bins over the 10 mixture episodes).', '']
    n_loom = sum(any(w['word'] == 'looming' for w in sample_episode(s)['words']) for s in range(5000))
    L += ['## 0. Design facts', '',
          f'- Sensory neurons receiving background (sensory_mask_malecns): {int(sens.sum()):,}.',
          f'- Episodes containing looming, spec seeds 0-4999: {n_loom:,} of 5,000 ({n_loom / 5000:.1%}).', '']
    res, idxs, allv = {}, {}, {}
    for cfg in CFGS:
        idx, eps = load_eps(cfg)
        idxs[cfg] = idx
        mix = [e for e in eps if e['kind'] == 'mix']
        bg = [e for e in eps if e['kind'] == 'bg_only']
        pv, loomL = var_table(mix, gid, sizes, names)
        pv = pv.merge(pdf[['pool', 'group', 'side']].rename(columns={'pool': 'name'}), on='name')
        cv, _ = var_table(mix, gid_c, csizes, cnames)
        res[cfg] = dict(eps=eps, mix=mix, bg=bg, pv=pv, cv=cv, n_loom_bins=int(loomL.sum()))
        allv[cfg] = (pv.assign(cfg=cfg), cv.assign(cfg=cfg))
    pd.concat([allv[c][0] for c in CFGS]).to_csv(PILOT / 'pool_variance.csv', index=False)
    pd.concat([allv[c][1] for c in CFGS]).to_csv(PILOT / 'group_variance.csv', index=False)

    # 1-2. variance tables
    nloom = sum(any(w['word'] == 'looming' for w in e['spec']['words']) for e in res['calib']['mix'])
    L += ['## 1. Motor-pool variance (349 pools)', '',
          f'Floor {FLOOR:g} Hz² (SD 1 Hz) is the headline; see the script docstring for why. '
          f'{nloom} of the 10 mixture episodes contain looming.', '',
          '| level | floor (Hz²) | calib | shuffle | calib, no looming eps | shuffle, no looming eps |',
          '|---|---|---|---|---|---|']
    for lvl, key, tot in [('pool', 'pv', len(names)), ('group × side', 'cv', len(cnames))]:
        for f in FLOORS:
            c = [int((res[cfg][key]['var'] > f).sum()) for cfg in CFGS]
            d = [int((res[cfg][key]['var_noloom'] > f).sum()) for cfg in CFGS]
            L.append(f'| {lvl} (of {tot}) | {f:g} | {c[0]} | {c[1]} | {d[0]} | {d[1]} |')
    L.append('')
    for cfg in CFGS:
        pv = res[cfg]['pv']
        tv = pv['var'].sum()
        lsh = (pv['var'] * pv['loom_share']).sum() / tv if tv > 0 else float('nan')
        top = pv.sort_values('var', ascending=False).head(10)
        L += [f'**{cfg}**: summed pool variance {tv:,.0f} Hz²; share from looming-episode bins '
              f'{lsh:.0%} ({res[cfg]["n_loom_bins"]} of 2,000 bins); top-10 pools hold '
              f'{top["var"].sum() / tv if tv > 0 else float("nan"):.0%}.', '',
              '| pool | n | group | mean Hz | var Hz² | var without looming eps | looming share |',
              '|---|---|---|---|---|---|---|']
        for r in top.itertuples():
            L.append(f'| {r.name} | {r.n:.0f} | {r.group} | {r.mean_hz:.2f} | {r.var:,.1f} | '
                     f'{r.var_noloom:,.1f} | {r.loom_share:.0%} |')
        L.append('')
        cv = res[cfg]['cv'].sort_values('var', ascending=False)
        L += [f'Coarse groups above floor ({cfg}): ' +
              ', '.join(f'{r.name} ({r.var:,.1f})' for r in cv.itertuples() if r.var > FLOOR) or 'none', '']

    # 1b. looming drive vs GF / TTMn / DLMn (does a lower intensity cap help?)
    named = {'GF': types == 'DNp01', 'TTMn': types == 'TTMn',
             'DLMn': pd.Series(types).str.startswith('DLMn').to_numpy()}
    L += ['### Looming episodes: GF / TTMn / DLMn rate inside the looming window (calib)', '',
          '| ep | looming rate Hz | window s | GF Hz | TTMn Hz | DLMn Hz |', '|---|---|---|---|---|---|']
    for e in res['calib']['mix']:
        for w in e['spec']['words']:
            if w['word'] != 'looming':
                continue
            k = (e['t'] * DT >= w['onset']) & (e['t'] * DT < w['offset'])
            dur = w['offset'] - w['onset']
            r = {nm: named[nm][e['i'][k]].sum() / named[nm].sum() / dur for nm in named}
            L.append(f'| {e["ep"]} | {w["rate_hz"]:.0f} | {dur:.2f} | {r["GF"]:.0f} | '
                     f'{r["TTMn"]:.0f} | {r["DLMn"]:.0f} |')
    L.append('')

    # 3. brain-input activity
    L += ['## 3. Brain-input activity (brain_mask = brain, non-sensory, non-motor: '
          f'{int(bm.sum()):,} neurons)', '',
          '| ep | words | bg Hz | calib active | calib frac | calib active excl. driven | '
          'shuffle active | shuffle frac | shuffle active excl. driven |', '|---|---|---|---|---|---|---|---|---|']
    for k in range(len(res['calib']['eps'])):
        row = []
        spec = res['calib']['eps'][k]['spec']
        drv = np.zeros(n, bool)
        for w in spec['words']:
            drv[word_idx(meta, pos, w['word'])] = True
        for cfg in CFGS:
            e = res[cfg]['eps'][k]
            act = np.zeros(n, bool); act[np.unique(e['i'])] = True
            row += [int((act & bm).sum()), (act & bm).sum() / bm.sum(), int((act & bm & ~drv).sum())]
        words = '+'.join(w['word'] for w in spec['words']) or '(background)'
        L.append(f'| {k} | {words} | {spec["bg_hz"]:.2f} | {row[0]:,} | {row[1]:.2%} | {row[2]:,} | '
                 f'{row[3]:,} | {row[4]:.2%} | {row[5]:,} |')
    L.append('')

    # 4. rest level
    L += ['## 4. Rest level, background only', '',
          'Active non-sensory neurons per 100 ms bin, bins 0.1-2.0 s (19 bins).', '',
          '| cfg | bg Hz | mean | max | last bin | of which brain (mean) | VNC (mean) | MNs active (any, 0-2 s) | top non-sensory types (spikes) |',
          '|---|---|---|---|---|---|---|---|---|']
    base = {}
    for cfg in CFGS:
        for e in res[cfg]['bg']:
            a = active_per_window(e, nonsens, t_from=0.1)
            ab = active_per_window(e, nonsens & ~vnc, t_from=0.1)
            av = active_per_window(e, nonsens & vnc, t_from=0.1)
            base.setdefault(cfg, []).append(a)
            mn_act = len(np.unique(e['i'][is_mn[e['i']]]))
            ii = e['i'][nonsens[e['i']]]
            tc = pd.Series(types[ii]).value_counts().head(5)
            tops = ', '.join(f'{t} ({c})' for t, c in tc.items()) or '-'
            L.append(f'| {cfg} | {e["spec"]["bg_hz"]:g} | {a.mean():.1f} | {a.max()} | {a[-1]} | '
                     f'{ab.mean():.1f} | {av.mean():.1f} | {mn_act} | {tops} |')
    L.append('')
    for cfg in CFGS:                     # optional H1 test (w4d4_pilot.py bgoff)
        f = PILOT / f'bgoff_{cfg}.npz'
        if f.exists():
            d = np.load(f); sp = json.loads(str(d['spec']))
            e = dict(t=d['t'].astype(np.int64), i=d['i'].astype(np.int64), spec=sp)
            a = active_per_window(e, nonsens)
            L += [f'Background switched off ({cfg}, `w4d4_pilot.py bgoff`): {sp["bg_hz"]:g} Hz on '
                  f'[0, {sp["bg_off"]}) s. Active non-sensory per 100 ms: ' + ', '.join(map(str, a)) + '.', '']

    # 5. persistence
    L += ['## 5. Persistence flag', '',
          'Tail = consecutive 100 ms windows starting 100 ms after the last word ends (only episodes '
          'with at least one full window). Flag if the tail mean exceeds 2x the largest '
          'background-only bin of the same cfg.', '',
          '| cfg | ep | words | last offset (s) | tail windows | tail active mean | tail max | bg-only max | flag |',
          '|---|---|---|---|---|---|---|---|---|']
    nflag, nelig = {c: 0 for c in CFGS}, 0
    for cfg in CFGS:
        bmax = max(int(b.max()) for b in base[cfg])
        for e in res[cfg]['mix']:
            lo = last_offset(e['spec'])
            a = active_per_window(e, nonsens, t_from=lo + 0.1)
            words = '+'.join(w['word'] for w in e['spec']['words'])
            if not len(a):
                L.append(f'| {cfg} | {e["ep"]} | {words} | {lo:.3f} | 0 | - | - | {bmax} | n/a |')
                continue
            nelig += cfg == 'calib'
            flag = a.mean() > 2 * bmax
            nflag[cfg] += int(flag)
            L.append(f'| {cfg} | {e["ep"]} | {words} | {lo:.3f} | {len(a)} | {a.mean():.1f} | '
                     f'{a.max()} | {bmax} | {"YES" if flag else "no"} |')
    L += ['', f'Eligible mixture episodes: {nelig} of 10 per cfg. Flagged: '
          + ', '.join(f'{c} {nflag[c]}' for c in CFGS) + '.', '']

    # 6. cost
    L += ['## 6. Cost (wall seconds per bio-second, CPU, 4 threads, one process)', '',
          '| cfg | kind | episodes | median | min | max |', '|---|---|---|---|---|---|']
    for cfg in CFGS:
        for kind, g in idxs[cfg].groupby('kind'):
            w = g.wall_per_bio_s
            L.append(f'| {cfg} | {kind} | {len(g)} | {w.median():.1f} | {w.min():.1f} | {w.max():.1f} |')
    allw = pd.concat(idxs.values()).wall_per_bio_s
    L += ['', f'All 24 runs: median {allw.median():.1f} s per bio-second, '
          f'mean {allw.mean():.1f}, max {allw.max():.1f}.', '']

    (PILOT / 'pilot_summary.md').write_text('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
