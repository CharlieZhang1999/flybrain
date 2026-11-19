#!/usr/bin/env python
"""W4D5: summarise the regression training set (scripts/w4d5_dataset.py), real vs shuffled.

Reads results/w4/regression/day5/ (ep{k:03d}_{cfg}.npz, index.csv, run_log.txt) and reports:
  0. composition and spec checks: looming share, modality counts, words per episode,
     background-rate histogram, word rate / time ranges, scheme; specs.json against the
     generator; files present per cfg.
  1. motor-pool variance (sim.cns.motor_pools, fine pools and coarse group x side), 10 ms
     bins, pooled over the 200 mixture episodes (50,000 bins): pools above the Day 4 floor
     1 Hz^2 (0 and 10 Hz^2 as sensitivity), with and without looming episodes; share of the
     summed variance from looming-episode bins (all pools, and TTMn / DLMn pools);
     background-only episodes, pooled the same way, for comparison (H1).
  2. brain_mask neurons with any spike per episode (fraction), also excluding the
     neurons a word drives directly.
  3. rest level, background-only episodes: active non-sensory neurons per 100 ms (bins
     0.1-2.5 s), brain / VNC split, MN spikes (per s) and MNs active. Self-sustaining
     check: median of bins 1.5-2.5 s / median of bins 0.1-1.1 s; flagged if > 1.5.
  4. persistence flag: active non-sensory neurons per 100 ms in windows from 100 ms after
     the episode's last word ends to 2.5 s, against the background-only episode of the same
     cfg with the nearest bg_hz (rate-matched); flagged if the tail mean > 2x that
     episode's largest bin (bins 0.1-2.5 s). The Day 4 rule (2x the largest bin of any
     background-only episode) is also counted.
  5. pathology: (a) brain burst = > 300 brain_mask neurons active in one 100 ms bin
     (0.1 s grid), counted excluding word-driven neurons, in any bin and in word-free bins
     (no word on during the bin); (b) all-or-none MN burst = a 10 ms bin in which more than
     half of all motor neurons (superclass vnc_motor) spike. Both thresholds and the MN
     definition are operational choices (the brief does not define (b)).
  6. cost: wall seconds per bio-second (index.csv) and reproducibility checks (run_log.txt).
Usage: python scripts/w4d5_dataset_summary.py
Output: results/w4/regression/day5/summary.md (+ pool_variance.csv, group_variance.csv,
        episode_metrics.csv)
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from sim.cns import cns_model, motor_pools, brain_mask, W4_CALIB  # noqa: E402
from sim.presyn import sensory_mask_malecns  # noqa: E402
from stimuli.mixtures import word_idx, last_offset  # noqa: E402
import w4d5_dataset as ds  # noqa: E402

OUT = ds.OUT
CFGS = ds.CFGS
DT = 1e-4
BIN = 100            # steps = 10 ms
WIN = 1000           # steps = 100 ms
FLOOR = 1.0
FLOORS = (0.0, 1.0, 10.0)
BRAIN_BURST = 300
MN_BURST_FRAC = 0.5
DRIFT = 1.5


def has_loom(spec):
    return any(w['word'] == 'looming' for w in spec['words'])


def load_eps(cfg):
    eps = []
    for k in range(ds.N_EP):
        f = ds.path(k, cfg)
        if not f.exists():
            continue
        d = np.load(f)
        sp = json.loads(str(d['spec']))
        eps.append(dict(ep=k, kind='mix' if sp['words'] else 'bg_only', t=d['t'].astype(np.int64),
                        i=d['i'].astype(np.int64), spec=sp))
    return eps


def binned(t, i, nsteps, gid, ngroups, width):
    nb = -(-nsteps // width)
    g = gid[i]; k = g >= 0
    out = np.zeros((nb, ngroups))
    np.add.at(out, (t[k] // width, g[k]), 1)
    return out


def var_table(eps, gid, sizes, names):
    """Per group: variance across 10 ms bins of the rate (Hz per neuron), pooled over eps."""
    rates, loom = [], []
    for e in eps:
        ns = int(round(e['spec']['T'] / DT))
        c = binned(e['t'], e['i'], ns, gid, len(sizes), BIN)
        rates.append(c / (sizes * BIN * DT))
        loom.append(np.full(len(c), has_loom(e['spec'])))
    R, L = np.vstack(rates), np.concatenate(loom)
    dev2 = (R - R.mean(0)) ** 2
    vnl = R[~L].var(0) if (~L).any() else np.full(len(sizes), np.nan)
    df = pd.DataFrame(dict(name=names, n=sizes, mean_hz=R.mean(0), var=R.var(0), var_noloom=vnl,
                           loom_share=dev2[L].sum(0) / np.maximum(dev2.sum(0), 1e-30)))
    return df, L


def active_per_bin(e, mask, width=WIN, t_from=0.0, t_to=None):
    """Distinct active neurons (in mask) per window of `width` steps in [t_from, t_to)."""
    T = e['spec']['T']
    t_to = T if t_to is None else t_to
    s0, s1 = int(round(t_from / DT)), int(round(t_to / DT))
    k = mask[e['i']] & (e['t'] >= s0) & (e['t'] < s1)
    t, i = e['t'][k], e['i'][k]
    nb = (s1 - s0) // width
    if nb <= 0:
        return np.zeros(0, int)
    b = (t - s0) // width
    keep = b < nb
    pairs = np.unique(np.stack([b[keep], i[keep]]), axis=1)
    return np.bincount(pairs[0], minlength=nb)


def fmt_hist(vals, edges):
    h, _ = np.histogram(vals, bins=edges)
    return ' | '.join(f'{a:.2f}-{b:.2f}: {c}' for a, b, c in zip(edges[:-1], edges[1:], h))


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
    n_mn = int(is_mn.sum())
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
    ttm_dlm = pdf.type.eq('TTMn') | pdf.type.str.startswith('DLMn')

    specs = ds.specs()
    mix_specs = specs[:ds.N_MIX]
    L = ['# W4D5 regression training set: summary', '',
         'Generated by `scripts/w4d5_dataset_summary.py` from `results/w4/regression/day5/`. '
         f'{ds.N_MIX} mixture episodes (k 0-{ds.N_MIX - 1}) + {len(ds.BG_HZ)} background-only '
         f'(k {ds.N_MIX}-{ds.N_EP - 1}), T = {ds.T} s, words in [0, {ds.T_WORDS}] s, class_v2 background; '
         'cfg calib (real) and shuffle, W4_CALIB, CPU float32 rng=torch, same spec and Poisson seed.', '']

    # 0. composition
    loom = [has_loom(s) for s in mix_specs]
    nl = int(sum(loom))
    mods = Counter(w['modality'] for s in mix_specs for w in s['words'])
    words = Counter(w['word'] for s in mix_specs for w in s['words'])
    nw = Counter(len(s['words']) for s in mix_specs)
    rates = [w['rate_hz'] for s in mix_specs for w in s['words']]
    ons = [w['onset'] for s in mix_specs for w in s['words']]
    offs = [w['offset'] for s in mix_specs for w in s['words']]
    bgs = [s['bg_hz'] for s in mix_specs]
    saved = json.loads((OUT / 'specs.json').read_text()) if (OUT / 'specs.json').exists() else None
    checks = {
        '2-4 words, distinct modalities': all(2 <= len(s['words']) <= 4
                                              and len({w['modality'] for w in s['words']}) == len(s['words'])
                                              for s in mix_specs),
        'word rates in [30, 150] Hz': all(30 <= r <= 150 for r in rates),
        'words within [0, 2.0] s': all(0 <= a < b <= 2.0 for a, b in zip(ons, offs)),
        'T = 2.5 s (all 210)': all(s['T'] == 2.5 for s in specs),
        'bg_hz in [1, 3] (all 210)': all(1 <= s['bg_hz'] <= 3 for s in specs),
        'class_v2 default scheme (all 210)': all(s['bg_scheme'] == dict(name='class_v2', bristle_hz=0.0, o2_hz=0.0,
                                                                         abdominal_hz=1.0) for s in specs),
        'looming share within 1/6 ± 0.06': abs(nl / ds.N_MIX - 1 / 6) <= 0.06,
        'specs deterministic (generated twice, equal)': ds.specs() == specs,
        'specs.json equals the generator': saved == specs,
        'spec seeds disjoint from the T1 pilot (4040-4049)': not ({s['spec_seed'] for s in mix_specs}
                                                                  & set(range(4040, 4050))),
    }
    L += ['## 0. Dataset composition', '',
          f'- Looming episodes: **{nl} of {ds.N_MIX} ({nl / ds.N_MIX:.1%})**; target 1/6 = 16.7% ± 6 points.',
          '- Words per episode (2 / 3 / 4): ' + ' / '.join(str(nw.get(k, 0)) for k in (2, 3, 4)) + '.',
          '- Episodes per modality: ' + ', '.join(f'{k} {v}' for k, v in sorted(mods.items())) + '.',
          '- Word picks: ' + ', '.join(f'{k} {v}' for k, v in sorted(words.items())) + '.',
          f'- Word rate {min(rates):.1f}-{max(rates):.1f} Hz; onset {min(ons):.4f}-{max(ons):.4f} s; '
          f'offset max {max(offs):.4f} s; last word offset per episode: median '
          f'{np.median([last_offset(s) for s in mix_specs]):.3f} s.',
          f'- Background rate (mixtures), 0.25 Hz bins: {fmt_hist(bgs, np.arange(1.0, 3.01, 0.25))}. '
          f'Mean {np.mean(bgs):.2f} Hz.',
          '- Background-only episodes: bg_hz ' + ', '.join(f'{b:g}' for b in ds.BG_HZ) + '.', '',
          '| spec check | result |', '|---|---|']
    L += [f'| {k} | {"PASS" if v else "FAIL"} |' for k, v in checks.items()]
    L.append('')

    res, allv, files = {}, {}, {}
    for cfg in CFGS:
        eps = load_eps(cfg)
        files[cfg] = len(eps)
        mix = [e for e in eps if e['kind'] == 'mix']
        bg = [e for e in eps if e['kind'] == 'bg_only']
        pv, loomL = var_table(mix, gid, sizes, names)
        pv = pv.merge(pdf[['pool', 'group', 'side', 'type']].rename(columns={'pool': 'name'}), on='name')
        cv, _ = var_table(mix, gid_c, csizes, cnames)
        bpv, _ = var_table(bg, gid, sizes, names)
        bcv, _ = var_table(bg, gid_c, csizes, cnames)
        res[cfg] = dict(eps=eps, mix=mix, bg=bg, pv=pv, cv=cv, bpv=bpv, bcv=bcv, n_loom_bins=int(loomL.sum()),
                        n_bins=len(loomL))
        allv[cfg] = (pv.assign(cfg=cfg), cv.assign(cfg=cfg))
    pd.concat([allv[c][0] for c in CFGS]).to_csv(OUT / 'pool_variance.csv', index=False)
    pd.concat([allv[c][1] for c in CFGS]).to_csv(OUT / 'group_variance.csv', index=False)
    L += ['Files present: ' + ', '.join(f'{c} {files[c]} of {ds.N_EP}' for c in CFGS) + '.', '']

    # 1. pool variance
    nb = res['calib']['n_bins']
    L += [f'## 1. Motor-pool variance ({len(names)} pools, {len(cnames)} group × side), 10 ms bins', '',
          f'Pooled over the mixture episodes ({nb:,} bins per cfg). Floor {FLOOR:g} Hz² as on Day 4. '
          'Note: with 25× more bins than the pilot, one stray spike adds 25× less variance, so the same '
          'floor is stricter for sparse pools than on Day 4.', '',
          '| level | floor (Hz²) | calib | shuffle | calib, no looming eps | shuffle, no looming eps | '
          'calib, background-only eps | shuffle, background-only eps |', '|---|---|---|---|---|---|---|---|']
    for lvl, key, bkey, tot in [('pool', 'pv', 'bpv', len(names)), ('group × side', 'cv', 'bcv', len(cnames))]:
        for f in FLOORS:
            c = [int((res[cfg][key]['var'] > f).sum()) for cfg in CFGS]
            d = [int((res[cfg][key]['var_noloom'] > f).sum()) for cfg in CFGS]
            b = [int((res[cfg][bkey]['var'] > f).sum()) for cfg in CFGS]
            L.append(f'| {lvl} (of {tot}) | {f:g} | {c[0]} | {c[1]} | {d[0]} | {d[1]} | {b[0]} | {b[1]} |')
    L.append('')
    for cfg in CFGS:
        pv = res[cfg]['pv']
        tv = pv['var'].sum()
        lsh = (pv['var'] * pv['loom_share']).sum() / tv if tv > 0 else float('nan')
        sub = pv[ttm_dlm.to_numpy()]
        tsh = (sub['var'] * sub['loom_share']).sum() / max(sub['var'].sum(), 1e-30)
        btv = res[cfg]['bpv']['var'].sum()
        top = pv.sort_values('var', ascending=False).head(10)
        L += [f'**{cfg}**: summed pool variance {tv:,.0f} Hz² (background-only episodes: {btv:,.1f} Hz²). '
              f'Share from looming-episode bins: **{lsh:.0%}** ({res[cfg]["n_loom_bins"]:,} of {nb:,} bins); '
              f'TTMn + DLMn pools ({len(sub)} pools, {sub["var"].sum():,.0f} Hz²): **{tsh:.0%}** from looming. '
              f'Top-10 pools hold {top["var"].sum() / tv if tv > 0 else float("nan"):.0%}.', '',
              '| pool | n | group | mean Hz | var Hz² | var without looming eps | looming share |',
              '|---|---|---|---|---|---|---|']
        for r in top.itertuples():
            L.append(f'| {r.name} | {r.n:.0f} | {r.group} | {r.mean_hz:.2f} | {r.var:,.1f} | '
                     f'{r.var_noloom:,.1f} | {r.loom_share:.0%} |')
        L.append('')

    # per-episode metrics (sections 2-5)
    rows = []
    bgrows = {c: [] for c in CFGS}
    for cfg in CFGS:
        for e in res[cfg]['eps']:
            sp = e['spec']
            drv = np.zeros(n, bool)
            for w in sp['words']:
                drv[word_idx(meta, pos, w['word'])] = True
            act = np.zeros(n, bool); act[np.unique(e['i'])] = True
            brain_b = active_per_bin(e, bm & ~drv)
            word_free = np.ones(len(brain_b), bool)
            for w in sp['words']:
                for b in range(len(brain_b)):
                    if b * 0.1 < w['offset'] and (b + 1) * 0.1 > w['onset']:
                        word_free[b] = False
            mn_k = is_mn[e['i']]
            ns = int(round(sp['T'] / DT))
            mn_bins = np.zeros(-(-ns // BIN), int)
            if mn_k.any():
                pr = np.unique(np.stack([e['t'][mn_k] // BIN, e['i'][mn_k]]), axis=1)
                mn_bins = np.bincount(pr[0], minlength=len(mn_bins))
            r = dict(cfg=cfg, ep=e['ep'], kind=e['kind'], bg_hz=sp['bg_hz'], looming=has_loom(sp),
                     words='+'.join(w['word'] for w in sp['words']) or '(background)',
                     brain_active=int((act & bm).sum()), brain_frac=(act & bm).sum() / bm.sum(),
                     brain_active_nodrv=int((act & bm & ~drv).sum()),
                     brain_bin_max=int(brain_b.max()), brain_bin_max_wordfree=int(brain_b[word_free].max())
                     if word_free.any() else 0,
                     mn_spikes=int(mn_k.sum()), mn_active=int(len(np.unique(e['i'][mn_k]))),
                     mn_bin_max_frac=mn_bins.max() / n_mn)
            if e['kind'] == 'mix':
                lo = last_offset(sp)
                a = active_per_bin(e, nonsens, t_from=lo + 0.1)
                r.update(last_offset=lo, tail_windows=len(a), tail_mean=float(a.mean()) if len(a) else np.nan,
                         tail_max=int(a.max()) if len(a) else np.nan)
            else:
                a = active_per_bin(e, nonsens, t_from=0.1)
                r.update(ns_mean=a.mean(), ns_max=int(a.max()), ns_last=int(a[-1]),
                         brain_mean=active_per_bin(e, nonsens & ~vnc, t_from=0.1).mean(),
                         vnc_mean=active_per_bin(e, nonsens & vnc, t_from=0.1).mean(),
                         late_early=np.median(a[14:]) / max(np.median(a[:10]), 1e-9),
                         n_burst_bins=int((brain_b > BRAIN_BURST).sum()), bins=a)
                bgrows[cfg].append(r)
            rows.append(r)
    em = pd.DataFrame(rows)
    em.drop(columns=['bins'], errors='ignore').to_csv(OUT / 'episode_metrics.csv', index=False)
    mixm = em[em.kind == 'mix']

    # 2. brain inputs
    L += [f'## 2. Brain-input activity (brain_mask: {int(bm.sum()):,} neurons)', '',
          'Per mixture episode: brain_mask neurons with any spike in 2.5 s.', '',
          '| cfg | median active | min | max | median fraction | min | max | '
          'median active, word-driven neurons excluded |', '|---|---|---|---|---|---|---|---|']
    for cfg in CFGS:
        g = mixm[mixm.cfg == cfg]
        L.append(f'| {cfg} | {g.brain_active.median():,.0f} | {g.brain_active.min():,} | {g.brain_active.max():,} | '
                 f'{g.brain_frac.median():.2%} | {g.brain_frac.min():.2%} | {g.brain_frac.max():.2%} | '
                 f'{g.brain_active_nodrv.median():,.0f} |')
    L.append('')

    # 3. rest level
    L += ['## 3. Rest level, background-only episodes (2.5 s)', '',
          'Active non-sensory neurons per 100 ms bin, bins 0.1-2.5 s (24 bins). late/early = median of bins '
          f'1.5-2.5 s / median of bins 0.1-1.1 s (medians, so that single burst bins do not move it); a self-sustaining build-up would show as late/early > {DRIFT:g}.', '',
          '| cfg | bg Hz | mean | max | last bin | brain (mean) | VNC (mean) | MN spikes | MN spikes per s | '
          'MNs active | late/early | brain burst bins |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for cfg in CFGS:
        for r in bgrows[cfg]:
            L.append(f'| {cfg} | {r["bg_hz"]:g} | {r["ns_mean"]:.1f} | {r["ns_max"]} | {r["ns_last"]} | '
                     f'{r["brain_mean"]:.1f} | {r["vnc_mean"]:.1f} | {r["mn_spikes"]} | {r["mn_spikes"] / 2.5:.1f} | '
                     f'{r["mn_active"]} | {r["late_early"]:.2f} | {r["n_burst_bins"]} |')
    drift = {c: [r['ep'] for r in bgrows[c] if r['late_early'] > DRIFT] for c in CFGS}
    L += ['', 'Self-sustaining check (late/early > ' + f'{DRIFT:g}): '
          + ', '.join(f'{c} {len(drift[c])} ({", ".join(map(str, drift[c])) or "none"})' for c in CFGS) + '.', '']
    for cfg in CFGS:
        L.append(f'Per-bin trace, {cfg} (active non-sensory per 100 ms, 0.1-2.5 s):')
        L.append('')
        for r in bgrows[cfg]:
            L.append(f'- {r["bg_hz"]:g} Hz: ' + ' '.join(map(str, r['bins'])))
        L.append('')

    # 4. persistence
    L += ['## 4. Persistence flag', '',
          'Tail = consecutive 100 ms windows from 100 ms after the last word ends to 2.5 s. Baseline = the '
          'background-only episode of the same cfg with the nearest bg_hz (bins 0.1-2.5 s). '
          'Flag: tail mean > 2 × baseline max. Day 4 rule: tail mean > 2 × the largest bin of any '
          'background-only episode of the cfg.', '',
          '| cfg | eligible eps | tail windows (median) | tail mean / baseline mean (median) | max ratio | '
          'flagged (rate-matched) | flagged (Day 4 rule) | flagged episodes |', '|---|---|---|---|---|---|---|---|']
    pers = {}
    for cfg in CFGS:
        bg = bgrows[cfg]
        bmax_all = max(r['ns_max'] for r in bg)
        g = mixm[mixm.cfg == cfg].copy()
        near = [min(bg, key=lambda r: abs(r['bg_hz'] - b)) for b in g.bg_hz]
        g['base_mean'] = [r['ns_mean'] for r in near]
        g['base_max'] = [r['ns_max'] for r in near]
        g['ratio'] = g.tail_mean / g.base_mean.clip(lower=1e-9)
        f1 = g[g.tail_mean > 2 * g.base_max]
        f2 = g[g.tail_mean > 2 * bmax_all]
        pers[cfg] = f1
        L.append(f'| {cfg} | {int((g.tail_windows > 0).sum())} | {g.tail_windows.median():.0f} | '
                 f'{g.ratio.median():.2f} | {g.ratio.max():.2f} | {len(f1)} | {len(f2)} | '
                 f'{", ".join(map(str, f1.ep)) or "none"} |')
    L.append('')
    for cfg in CFGS:
        f1 = pers[cfg]
        if len(f1):
            L += [f'Flagged ({cfg}):', '', '| ep | words | bg Hz | last offset | tail mean | baseline max |',
                  '|---|---|---|---|---|---|']
            L += [f'| {r.ep} | {r.words} | {r.bg_hz:.2f} | {r.last_offset:.3f} | {r.tail_mean:.1f} | {r.base_max} |'
                  for r in f1.itertuples()]
            L.append('')

    # 5. pathology
    L += ['## 5. Pathology check', '',
          f'(a) brain burst: > {BRAIN_BURST} brain_mask neurons active in one 100 ms bin (0.1 s grid), word-driven '
          'neurons excluded; counted in any bin and in word-free bins only. '
          f'(b) all-or-none MN burst: a 10 ms bin in which > {MN_BURST_FRAC:.0%} of the {n_mn:,} motor neurons spike.', '',
          '| cfg | kind | episodes | (a) any bin | (a) word-free bins | (b) MN burst | any (a, any bin) or (b) | '
          'share | max brain bin | max MN fraction in a 10 ms bin |', '|---|---|---|---|---|---|---|---|---|---|']
    patho = {}
    for cfg in CFGS:
        for kind in ('mix', 'bg_only'):
            g = em[(em.cfg == cfg) & (em.kind == kind)]
            a = g.brain_bin_max > BRAIN_BURST
            aw = g.brain_bin_max_wordfree > BRAIN_BURST
            b = g.mn_bin_max_frac > MN_BURST_FRAC
            anyp = a | b
            patho[(cfg, kind)] = g[anyp | aw]
            L.append(f'| {cfg} | {kind} | {len(g)} | {int(a.sum())} | {int(aw.sum())} | {int(b.sum())} | '
                     f'{int(anyp.sum())} | {anyp.mean():.1%} | {g.brain_bin_max.max()} | {g.mn_bin_max_frac.max():.1%} |')
    L.append('')
    for (cfg, kind), g in patho.items():
        if len(g):
            L += [f'Episodes with a burst ({cfg}, {kind}): '
                  + ', '.join(f'{r.ep} ({r.words}; brain max {r.brain_bin_max}, word-free {r.brain_bin_max_wordfree}, '
                              f'MN {r.mn_bin_max_frac:.0%})' for r in g.itertuples()) + '.', '']

    # 6. cost
    idx = pd.read_csv(OUT / 'index.csv')
    L += ['## 6. Cost (wall seconds per bio-second, CPU, one process)', '',
          '| cfg | kind | episodes | median | min | max | total wall h |', '|---|---|---|---|---|---|---|']
    for (cfg, kind), g in idx.groupby(['cfg', 'kind']):
        w = g.wall_per_bio_s
        L.append(f'| {cfg} | {kind} | {len(g)} | {w.median():.1f} | {w.min():.1f} | {w.max():.1f} | '
                 f'{g.wall_s.sum() / 3600:.2f} |')
    L += ['', f'All {len(idx)} runs: median {idx.wall_per_bio_s.median():.1f} s per bio-second, mean '
          f'{idx.wall_per_bio_s.mean():.1f}, max {idx.wall_per_bio_s.max():.1f}; simulation wall total '
          f'{idx.wall_s.sum() / 3600:.2f} h for {idx["T"].sum():.0f} bio-s.', '']
    log = (OUT / 'run_log.txt').read_text().splitlines() if (OUT / 'run_log.txt').exists() else []
    chk = [x for x in log if x.startswith('[check]')]
    L += ['Reproducibility (`w4d5_dataset.py check`, run_log.txt): ' + ('; '.join(chk) if chk else 'not run') + '.', '']

    (OUT / 'summary.md').write_text('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
