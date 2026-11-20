#!/usr/bin/env python
"""W4D5: summarise the regression training set (scripts/w4d5_dataset.py), real vs shuffled.

Reads results/w4/regression/day5/ (ep{k:03d}_{cfg}.npz, index.csv, run_log.txt) and reports:
  0. composition and spec checks: looming share, modality counts, words per episode,
     background-rate histogram, word rate / time ranges, scheme; specs.json against the
     generator; files present per cfg.
  1. motor-pool variance (sim.cns.motor_pools, fine pools and coarse group x side), 10 ms
     bins, pooled over the 200 mixture episodes (50,000 bins): pools above the Day 4 floor
     1 Hz^2 (0 and 10 Hz^2 as sensitivity), with and without looming episodes; share of the
     summed variance from looming-episode bins and from wind-episode bins (all pools, and
     TTMn / DLMn pools; H2); background-only episodes, pooled the same way, for comparison (H1).
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
  5. pathology: (a) the T2c Q4 word-free metric (scripts/w4d5_burst_metric.score): eligible
     100 ms bins overlap no word and start >= 0.1 s after the most recent word offset;
     brain_mask neurons active per bin, word-driven neurons excluded; THETA 200; flag
     'persist' (>= 2 consecutive eligible bins > THETA), 'burst' (a single bin); AL events
     (scripts/w4d5_burst_common.al_events) counted separately, and a 'burst' bin that holds
     an AL-event onset is counted as an AL burst. A persist run is attributed to a wind
     tail if a wind word ended at or before the run start and either the run starts
     <= 0.2 s after that offset or the wind word's LOOP5 tail reaches the run start.
     (b) all-or-none MN burst = a 10 ms bin in which more than half of all motor neurons
     (superclass vnc_motor) spike (operational; the brief does not define it).
     The old metric (> 300 brain neurons in any 100 ms bin) is shown for reference only.
  5b. word tails: per word occurrence, from its offset o:
     d27_last   last DNge027 spike in the episode - o (wind words; the T2c/T2d/T2e tail)
     loop5      LOOP5 (DNge027, GNG423, DNg21, DNg62, DNge044) chain tail: the last spike of
                the chain of LOOP5 spikes that starts within 50 ms of o with gaps <= 50 ms
     brain      first 100 ms bin starting at or after o; (end of the run of consecutive bins
                > THETA from there) - o, word-driven neurons of all words excluded; 0 if none
     confounded another word is on at some time in (o - 0.3 s, o + 0.3 s) (a later word, or an
                earlier one whose own tail may still run)
     Per word type: occurrences, and how many have a tail > 0.2 s (loop5, brain).
  6. cost: wall seconds per bio-second (index.csv) and reproducibility checks (run_log.txt).
  7. stop conditions (T2f): (a) any background-only episode flagged 'persist'; (b) > 10% of
     the mixture episodes of a cfg with a non-wind-tail persist run longer than 0.4 s
     (>= 5 bins); (c) cost > 40 s per bio-second (any run).
Usage: python scripts/w4d5_dataset_summary.py          # full summary
       python scripts/w4d5_dataset_summary.py check    # stop conditions only, episodes so far
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
import w4d5_burst_metric as BM  # noqa: E402
import w4d5_burst_common as C  # noqa: E402
from sim.cns import LOOP_GAIN  # noqa: E402

OUT = ds.OUT
CFGS = ds.CFGS
DT = 1e-4
BIN = 100            # steps = 10 ms
WIN = 1000           # steps = 100 ms
FLOOR = 1.0
FLOORS = (0.0, 1.0, 10.0)
BRAIN_BURST = 300     # old metric, reference only
THETA = BM.THETA
GRACE = BM.GRACE
CHAIN_GAP = 0.05
TAIL_LONG = 0.2
PERSIST_LONG = 0.4
COST_MAX = 40.0
STOP_MIX_FRAC = 0.10
MN_BURST_FRAC = 0.5
DRIFT = 1.5


def has_loom(spec):
    return any(w['word'] == 'looming' for w in spec['words'])


def has_wind(spec):
    return any(w['word'] == 'wind' for w in spec['words'])


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
    rates, loom, wind = [], [], []
    for e in eps:
        ns = int(round(e['spec']['T'] / DT))
        c = binned(e['t'], e['i'], ns, gid, len(sizes), BIN)
        rates.append(c / (sizes * BIN * DT))
        loom.append(np.full(len(c), has_loom(e['spec'])))
        wind.append(np.full(len(c), has_wind(e['spec'])))
    R, L, Wd = np.vstack(rates), np.concatenate(loom), np.concatenate(wind)
    dev2 = (R - R.mean(0)) ** 2
    tot = np.maximum(dev2.sum(0), 1e-30)
    vnl = R[~L].var(0) if (~L).any() else np.full(len(sizes), np.nan)
    df = pd.DataFrame(dict(name=names, n=sizes, mean_hz=R.mean(0), var=R.var(0), var_noloom=vnl,
                           loom_share=dev2[L].sum(0) / tot, wind_share=dev2[Wd].sum(0) / tot,
                           both_share=dev2[L & Wd].sum(0) / tot))
    return df, (L, Wd)


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


def chain_tail(times, o, gap=CHAIN_GAP):
    """End of the chain of spikes (sorted times, s) that starts within `gap` of o, minus o."""
    cur = o
    for x in times[np.searchsorted(times, o):]:
        if x - cur > gap:
            break
        cur = x
    return cur - o


def runs_of(mask):
    """(start, length) of each run of True in a boolean array."""
    out, j = [], 0
    while j < len(mask):
        if mask[j]:
            k = j
            while k + 1 < len(mask) and mask[k + 1]:
                k += 1
            out.append((j, k - j + 1))
            j = k + 1
        else:
            j += 1
    return out


def q4(e, X):
    """T2c Q4 metric (BM.score) plus persist runs, AL bursts, word tails for one episode."""
    sp, t, i = e['spec'], e['t'], e['i']
    T = float(sp['T'])
    nb = int(round(T / 0.1))
    sc = BM.score(t, i, sp, X['M'], X['meta'], X['pos'], f"ep{e['ep']:03d}")
    drv = np.zeros(X['n'], bool)
    for w in sp['words']:
        drv[word_idx(X['meta'], X['pos'], w['word'])] = True
    b = C.bin_counts(t, i, X['M']['brain'] & ~drv, T)
    elig = np.ones(nb, bool)                       # same rule as BM.score
    for j in range(nb):
        for w in sp['words']:
            if j * 0.1 < w['offset'] + GRACE - 1e-9 and (j + 1) * 0.1 > w['onset']:
                elig[j] = False
    hi = elig & (b > THETA)
    runs = runs_of(hi)
    prs = [(a, n_) for a, n_ in runs if n_ >= 2]
    assert sum(n_ for _, n_ in prs) == sc['persist_bins'] and sum(n_ for _, n_ in runs if n_ == 1) == sc['burst_bins']
    on, _ = C.al_events(t, i, X['M']['al'], T)
    ev_bins = {int(x // 0.1) for x in on}
    tl = np.sort(t[X['loop5'][i]] * DT)
    td = np.sort(t[X['d27'][i]] * DT)
    tails = []
    for w in sp['words']:
        o = w['offset']
        j0 = int(np.ceil(o / 0.1 - 1e-9))
        k = j0
        while k < nb and b[k] > THETA:
            k += 1
        tails.append(dict(word=w['word'], onset=w['onset'], offset=o,
                          d27_last=(td[-1] - o) if len(td) else np.nan,
                          loop5=chain_tail(tl, o), brain=(k * 0.1 - o) if k > j0 else 0.0,
                          room=T - o,
                          confounded=any(v is not w and v['onset'] < o + 0.3 and v['offset'] > o - 0.3
                                         for v in sp['words'])))
    winds = [x for x in tails if x['word'] == 'wind']
    pr_info = []
    for a, n_ in prs:
        start = a * 0.1
        wt = any(x['offset'] <= start + 1e-9 and (start - x['offset'] <= 0.2 + 1e-9
                                                    or x['offset'] + x['loop5'] >= start - 1e-9) for x in winds)
        pr_info.append(dict(start=round(start, 1), dur=round(n_ * 0.1, 1), wind_tail=wt,
                            al=any(j in ev_bins for j in range(a, a + n_)),
                            peak=int(b[a:a + n_].max())))
    bursts = [a for a, n_ in runs if n_ == 1]
    return dict(flag=sc['flag'], persist_bins=sc['persist_bins'], burst_bins=sc['burst_bins'],
                n_eligible=sc['n_eligible'], max_eligible=sc['max_eligible'], P_wf=sc['P_wf'],
                al_events=sc['al_events'], al_events_eligible=sc['al_events_eligible'],
                al_bursts=sum(a in ev_bins for a in bursts), other_bursts=sum(a not in ev_bins for a in bursts),
                burst_at=[round(a * 0.1, 1) for a in bursts], persist_runs=pr_info,
                long_nonwind=any(p['dur'] > PERSIST_LONG + 1e-9 and not p['wind_tail'] for p in pr_info),
                old_any_gt300=int((b > BRAIN_BURST).any()), old_wordfree_gt300=int(((b > BRAIN_BURST) & elig).any()),
                tails=tails)


def context():
    m = cns_model('cns', 'calib', verbose=False, **W4_CALIB)
    meta, vnc = m['meta'], m['vnc']
    del m
    types = meta.type.fillna('?').astype(str).values
    return dict(meta=meta, vnc=vnc, n=len(meta), pos={b: i for i, b in enumerate(meta.bodyId)},
                M=C.masks(meta, vnc), types=types, d27=types == 'DNge027',
                loop5=np.isin(types, sorted(LOOP_GAIN[0])))


def stop_lines(qrows, idx):
    """Stop conditions (T2f) from per-episode Q4 rows and index.csv; returns (lines, stop?)."""
    q = pd.DataFrame(qrows)
    L, stop = [], False
    for cfg in CFGS:
        g = q[(q.cfg == cfg) & (q.kind == 'bg_only')]
        bad = g[g.flag == 'persist']
        stop |= len(bad) > 0
        L.append(f'- (a) {cfg} background-only: {len(g)} episodes, {len(bad)} flagged persist'
                 + (f' ({", ".join(map(str, bad.ep))})' if len(bad) else '') + '.')
    for cfg in CFGS:
        g = q[(q.cfg == cfg) & (q.kind == 'mix')]
        bad = g[g.long_nonwind]
        frac = len(bad) / len(g) if len(g) else 0.0
        stop |= frac > STOP_MIX_FRAC
        L.append(f'- (b) {cfg} mixtures: {len(bad)} of {len(g)} ({frac:.1%}) with a non-wind-tail persist run '
                 f'> {PERSIST_LONG:g} s' + (f' ({", ".join(map(str, bad.ep))})' if len(bad) else '')
                 + f'; limit {STOP_MIX_FRAC:.0%}.')
    if len(idx):
        mx = idx.wall_per_bio_s.max()
        stop |= mx > COST_MAX
        L.append(f'- (c) cost: {len(idx)} runs, median {idx.wall_per_bio_s.median():.1f}, max {mx:.1f} s per '
                 f'bio-second; limit {COST_MAX:g}.')
    L.append(f'- **Stop: {"YES" if stop else "no"}.**')
    return L, stop


def check():
    """Stop conditions on the episodes saved so far (between chunks)."""
    X = context()
    rows = []
    for cfg in CFGS:
        for e in load_eps(cfg):
            r = q4(e, X)
            rows.append(dict(cfg=cfg, ep=e['ep'], kind=e['kind'], **{k: v for k, v in r.items() if k != 'tails'}))
            if r['flag'] != 'ok' or r['al_events_eligible']:
                print(cfg, e['ep'], e['kind'], r['flag'], 'persist', r['persist_runs'], 'bursts', r['burst_at'],
                      'AL elig', r['al_events_eligible'], flush=True)
    idx = pd.read_csv(OUT / 'index.csv')
    L, stop = stop_lines(rows, idx)
    print('\n'.join(L))
    return stop


def main():
    X = context()
    meta, vnc, pos = X['meta'], X['vnc'], X['pos']
    n = len(meta)
    sens = sensory_mask_malecns(meta)
    bm = brain_mask(meta, vnc)
    nonsens = ~sens
    is_mn = meta.superclass.astype(str).values == 'vnc_motor'
    n_mn = int(is_mn.sum())
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
        pv, (loomL, windL) = var_table(mix, gid, sizes, names)
        pv = pv.merge(pdf[['pool', 'group', 'side', 'type']].rename(columns={'pool': 'name'}), on='name')
        cv, _ = var_table(mix, gid_c, csizes, cnames)
        bpv, _ = var_table(bg, gid, sizes, names)
        bcv, _ = var_table(bg, gid_c, csizes, cnames)
        res[cfg] = dict(eps=eps, mix=mix, bg=bg, pv=pv, cv=cv, bpv=bpv, bcv=bcv, n_loom_bins=int(loomL.sum()),
                        n_wind_bins=int(windL.sum()), n_both_bins=int((loomL & windL).sum()), n_bins=len(loomL))
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
    def share(df, col):
        v = df['var'].sum()
        return (df['var'] * df[col]).sum() / v if v > 0 else float('nan')

    def pct(x):
        return '–' if np.isnan(x) else f'{x:.0%}'

    L += ['Share of the summed pool variance (sum over pools of squared deviations from the pooled mean) that '
          'falls in bins of looming episodes and of wind episodes (H2). Episodes with both words count in both '
          'columns; "both" is their overlap. Bin share = the share expected if variance were spread evenly.', '',
          '| cfg | pools | summed var (Hz²) | looming eps | wind eps | both | neither | bins: looming / wind / both |',
          '|---|---|---|---|---|---|---|---|']
    for cfg in CFGS:
        pv = res[cfg]['pv']
        for lab, sub in (('all', pv), ('TTMn + DLMn', pv[ttm_dlm.to_numpy()])):
            ls, ws, bs = share(sub, 'loom_share'), share(sub, 'wind_share'), share(sub, 'both_share')
            L.append(f'| {cfg} | {lab} ({len(sub)}) | {sub["var"].sum():,.0f} | **{pct(ls)}** | **{pct(ws)}** | {pct(bs)} | '
                     f'{pct(1 - ls - ws + bs)} | {res[cfg]["n_loom_bins"] / nb:.0%} / {res[cfg]["n_wind_bins"] / nb:.0%} / '
                     f'{res[cfg]["n_both_bins"] / nb:.0%} |')
    L.append('')
    for cfg in CFGS:
        pv = res[cfg]['pv']
        tv = pv['var'].sum()
        btv = res[cfg]['bpv']['var'].sum()
        top = pv.sort_values('var', ascending=False).head(10)
        L += [f'**{cfg}**: summed pool variance {tv:,.0f} Hz² (background-only episodes: {btv:,.1f} Hz²). '
              f'Top-10 pools hold {top["var"].sum() / tv if tv > 0 else float("nan"):.0%}.', '',
              '| pool | n | group | mean Hz | var Hz² | var without looming eps | looming share | wind share |',
              '|---|---|---|---|---|---|---|---|']
        for r in top.itertuples():
            L.append(f'| {r.name} | {r.n:.0f} | {r.group} | {r.mean_hz:.2f} | {r.var:,.1f} | '
                     f'{r.var_noloom:,.1f} | {r.loom_share:.0%} | {r.wind_share:.0%} |')
        L.append('')

    # per-episode metrics (sections 2-5)
    rows, tailrows = [], []
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
            qq = q4(e, X)
            for tw in qq.pop('tails'):
                tailrows.append(dict(cfg=cfg, ep=e['ep'], **tw))
            r = dict(cfg=cfg, ep=e['ep'], kind=e['kind'], bg_hz=sp['bg_hz'], looming=has_loom(sp),
                     words='+'.join(w['word'] for w in sp['words']) or '(background)',
                     brain_active=int((act & bm).sum()), brain_frac=(act & bm).sum() / bm.sum(),
                     brain_active_nodrv=int((act & bm & ~drv).sum()),
                     brain_bin_max=int(brain_b.max()), brain_bin_max_wordfree=int(brain_b[word_free].max())
                     if word_free.any() else 0,
                     mn_spikes=int(mn_k.sum()), mn_active=int(len(np.unique(e['i'][mn_k]))),
                     mn_bin_max_frac=mn_bins.max() / n_mn, **qq)
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
                         late_early=np.median(a[14:]) / max(np.median(a[:10]), 1e-9), bins=a)
                bgrows[cfg].append(r)
            rows.append(r)
    em = pd.DataFrame(rows)
    em.drop(columns=['bins'], errors='ignore').to_csv(OUT / 'episode_metrics.csv', index=False)
    tw = pd.DataFrame(tailrows)
    tw.to_csv(OUT / 'word_tails.csv', index=False)
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
          'MNs active | late/early | Q4 flag | AL events (eligible) |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for cfg in CFGS:
        for r in bgrows[cfg]:
            L.append(f'| {cfg} | {r["bg_hz"]:g} | {r["ns_mean"]:.1f} | {r["ns_max"]} | {r["ns_last"]} | '
                     f'{r["brain_mean"]:.1f} | {r["vnc_mean"]:.1f} | {r["mn_spikes"]} | {r["mn_spikes"] / 2.5:.1f} | '
                     f'{r["mn_active"]} | {r["late_early"]:.2f} | {r["flag"]} | {r["al_events"]} ({r["al_events_eligible"]}) |')
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
          f'(a) T2c Q4 word-free metric (`scripts/w4d5_burst_metric.py`): eligible 100 ms bins overlap no word and '
          f'start >= {GRACE:g} s after the most recent word offset; brain_mask neurons active per bin, word-driven '
          f'neurons excluded; THETA = {THETA}. persist = >= 2 consecutive eligible bins > THETA; burst = a single '
          'such bin. AL burst = a burst bin that holds an AL-event onset (>= 100 ALPN/ALLN in 10 ms). '
          'Long non-wind persist = a persist run > 0.4 s not attributed to a wind tail (see docstring). '
          f'(b) all-or-none MN burst: a 10 ms bin in which > {MN_BURST_FRAC:.0%} of the {n_mn:,} motor neurons spike. '
          f'Old metric (> {BRAIN_BURST} in any bin / in an eligible bin) for reference only.', '',
          '| cfg | kind | episodes | persist | burst | ok | burst bins: AL / other | episodes with an eligible AL event | '
          'AL events (all / eligible) | long non-wind persist | (b) MN burst | old > 300 any / eligible | '
          'max eligible bin |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for cfg in CFGS:
        for kind in ('mix', 'bg_only'):
            g = em[(em.cfg == cfg) & (em.kind == kind)]
            b = g.mn_bin_max_frac > MN_BURST_FRAC
            L.append(f'| {cfg} | {kind} | {len(g)} | {int((g.flag == "persist").sum())} | '
                     f'{int((g.flag == "burst").sum())} | {int((g.flag == "ok").sum())} | '
                     f'{int(g.al_bursts.sum())} / {int(g.other_bursts.sum())} | {int((g.al_events_eligible > 0).sum())} | '
                     f'{int(g.al_events.sum())} / {int(g.al_events_eligible.sum())} | {int(g.long_nonwind.sum())} | '
                     f'{int(b.sum())} | {int(g.old_any_gt300.sum())} / {int(g.old_wordfree_gt300.sum())} | '
                     f'{int(g.max_eligible.max())} |')
    L.append('')
    for cfg in CFGS:
        g = em[(em.cfg == cfg) & (em.flag != 'ok')]
        if len(g):
            L += [f'Flagged episodes ({cfg}):', '', '| ep | kind | words | flag | persist runs (start s, dur s, '
                  'wind tail?, AL event?, peak) | burst bins at (s) | AL bursts |', '|---|---|---|---|---|---|---|']
            for r in g.itertuples():
                prs = '; '.join(f'{p["start"]:.1f}, {p["dur"]:.1f}, {"wind" if p["wind_tail"] else "NOT wind"}, '
                                f'{"AL" if p["al"] else "-"}, {p["peak"]}' for p in r.persist_runs) or '-'
                L.append(f'| {r.ep} | {r.kind} | {r.words} | {r.flag} | {prs} | '
                         f'{", ".join(f"{x:.1f}" for x in r.burst_at) or "-"} | {r.al_bursts} |')
            L.append('')

    # 5b. word tails
    L += ['## 5b. Tails after words', '',
          'Per word occurrence, from its offset o (definitions in the script docstring): d27_last = last DNge027 '
          'spike in the episode - o (the T2c-T2e wind tail; negative = DNge027 stopped before o; capped by T); '
          f'loop5 = LOOP5 spike chain from o, gaps <= {CHAIN_GAP * 1000:.0f} ms; brain = run of 100 ms bins > {THETA} '
          'from the first bin starting at or after o (word-driven neurons excluded). Confounded = another word on '
          'at some time within 0.3 s before or after o.', '']
    for cfg in CFGS:
        w = tw[(tw.cfg == cfg) & (tw.word == 'wind')]
        if not len(w):
            continue
        d = w.d27_last.dropna()
        L += [f'**Wind, {cfg}** ({len(w)} wind words; DNge027 fired in {len(d)}): d27_last median '
              f'{d.median() if len(d) else float("nan"):.3f} s, max {d.max() if len(d) else float("nan"):.3f} s; '
              f'> 0.15 s: {int((d > 0.15).sum())}, > 0.2 s: {int((d > 0.2).sum())}, > 0.3 s: {int((d > 0.3).sum())}. '
              f'loop5 median {w.loop5.median():.3f} s, max {w.loop5.max():.3f} s, > 0.2 s: {int((w.loop5 > 0.2).sum())}. '
              f'brain tail > 0.2 s: {int((w.brain > 0.2).sum())}.', '',
              '| ep | wind (s) | d27_last (s) | loop5 (s) | brain (s) | confounded |', '|---|---|---|---|---|---|']
        L += [f'| {r.ep} | {r.onset:.2f}-{r.offset:.2f} | {r.d27_last:.3f} | {r.loop5:.3f} | {r.brain:.1f} | '
              f'{"yes" if r.confounded else ""} |' for r in w.itertuples()]
        L.append('')
    L += ['Per word type, mixtures: occurrences with a tail > 0.2 s (all occurrences; in brackets the '
          'unconfounded ones).', '',
          '| word | ' + ' | '.join(f'{c} n (unconf.) | {c} loop5 > 0.2 s | {c} brain > 0.2 s' for c in CFGS) + ' |',
          '|---|' + '---|---|---|' * len(CFGS)]
    for word in sorted(tw.word.unique()):
        cells = []
        for cfg in CFGS:
            g = tw[(tw.cfg == cfg) & (tw.word == word)]
            u = g[~g.confounded]
            cells.append(f'{len(g)} ({len(u)}) | {int((g.loop5 > TAIL_LONG).sum())} ({int((u.loop5 > TAIL_LONG).sum())}) | '
                         f'{int((g.brain > TAIL_LONG).sum())} ({int((u.brain > TAIL_LONG).sum())})')
        L.append(f'| {word} | ' + ' | '.join(cells) + ' |')
    L.append('')

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
    start = max((k for k, x in enumerate(log) if x.startswith('[T2f]')), default=0)
    chk = [x for x in log[start:] if x.startswith('[check]')]
    L += ['Reproducibility (`w4d5_dataset.py check`, run_log.txt, since the T2f re-run): '
          + ('; '.join(chk) if chk else 'not run') + '.', '']

    # 7. stop conditions
    sl, stop = stop_lines(em.to_dict('records'), idx)
    L += ['## 7. Stop conditions (T2f)', ''] + sl + ['']

    (OUT / 'summary.md').write_text('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    if sys.argv[1:] == ['check']:
        sys.exit(1 if check() else 0)
    main()
