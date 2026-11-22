#!/usr/bin/env python
"""W5 day 1: pre-registered behaviour sanity panel, real wiring vs 10 shuffles.

Purpose
  For each known stimulus -> behaviour pair, does the full-CNS model
  (cns_model('cns', 'calib', **W4_CALIB)) put the expected output neurons near
  the top of all motor outputs, and more than shuffled wiring does? Nothing is
  trained or tuned. Brief: progress/week4/team/T5a_behaviour_panel_brief.md
  (on the project mount). Results: progress/week5/day1_behaviour_panel.md.

Pairs (fixed before any run; copied unchanged from the brief)
  | #  | Condition                                   | Expected outputs ("hits")        | Expected direction |
  | P1 | sugar                                       | MN9 (L, R)                       | up |
  | P2 | water                                       | MN9                              | up |
  | P3 | bitter                                      | MN9                              | not up (delta <= 0 or rank in bottom half) |
  | P4 | sugar + bitter together                     | MN9                              | lower than P1 on the same network/seed |
  | P5 | looming                                     | TTMn (L, R)                      | up |
  | P6 | looming with GF (DNp01) silenced (zero its  | TTMn                             | lower than P5 |
  |    |   W rows and columns and its gap junctions) |                                  | |
  | P7 | MDN (all 4)                                 | hind-leg pools (subclass 'hl')   | up, more than front/mid legs |
  | P8 | DNg100 (both)                               | leg pools (fl, ml, hl)           | up |
  | P9 | sugar at 50 / 100 / 150 / 200 Hz            | MN9                              | monotonic increase (dose-response) |
  | X1 | wind (JO-C/E), exploratory, not scored      | front-leg pools                  | report only |

Protocol
  Each condition: 0.5 s trial, Poisson drive 0-0.4 s at 150 Hz (P9: 50-200 Hz)
  on the named neurons (sensory words: the word's neurons, stimuli/malecns_words),
  no background. Seeds 0, 1, 2. Per output pool: mean rate per neuron over
  0-0.4 s, minus the 'none' condition of the same network and seed (P6, the
  GF-silenced copy, uses the intact network's 'none'; with no background
  'none' is silent, checked in the log).

Output pools
  The 349 VNC motor pools of sim.cns.motor_pools ('<side>|<subclass>|<type>')
  plus the brain (cb_motor) motor neurons grouped by side|type, named
  'cb|<side>|<type>' (87 pools; e.g. 'cb|L|MN9'). 436 pools in all.

Networks
  'real'  = cns_model('cns', 'calib', **W4_CALIB).
  'shufK' (K = 1..10) = cns_model('cns', 'shuffle', **W4_CALIB) with
  scripts.w2d4_word_trials.shuffle_W called with seed=K instead of its
  default 12345, i.e. the same permutation step at the same point of the
  pipeline (after calibration scaling, sign fixes, gains, APL and LN-PN cut;
  before presynaptic modulation and the JO mirror). `verify` checks that
  seed 12345 through this route reproduces cns_model's shuffle W exactly.

Scoring (unchanged from the brief)
  P1, P2, P5, P7, P8: AUROC of the expected pools vs all other output pools,
  ranked by delta-rate (mean over seeds; ties get average rank), plus the rank
  of the best expected pool (1 = highest delta; rank = 1 + #pools strictly
  higher). Empirical p = (1 + #shuffles with AUROC >= real) / 11.
  P3, P4, P6, P9: pass/fail on the stated direction, real and every shuffle.
  Hit: real AUROC >= 0.9 AND p <= 0.1 (AUROC pairs), or direction test passes.
  Operational definitions where the brief leaves a choice (also in the doc):
    MN9 / TTMn delta = mean over the L and R pools and the 3 seeds.
    P3 passes if MN9 delta <= 0 or the best MN9 pool ranks in the bottom half
       (rank > 436 / 2).
    P4 / P6 pass if the mean delta is strictly lower than P1 / P5.
    P9 passes if MN9 delta is strictly increasing over 50 < 100 < 150 < 200 Hz.
    A shuffle has no p-value, so a shuffle 'hit' is AUROC >= 0.9 (or direction
    pass). The real network is reported both ways.
    P7 'more than front/mid legs' is reported (hl mean delta vs fl+ml) but the
    hit uses the AUROC rule, as the scoring section says.

Usage
  python scripts/w5_behaviour_panel.py verify          # shuffle checks (A2)
  python scripts/w5_behaviour_panel.py run real        # one network (resumable)
  python scripts/w5_behaviour_panel.py run all         # real + shuf1..shuf10
  python scripts/w5_behaviour_panel.py score           # scorecard + figure
Output (results/w5/panel/)
  shuffle_verify.json; pools.csv; rates_<net>.csv (net, cond, seed, pool,
  rate_hz); trials_<net>.csv (per-trial diagnostics: GF / MN9 / TTMn rates,
  late-window activity 0.45-0.5 s, wall time); scorecard.csv (one row per
  network x pair); summary.json; results/figures/w5_behaviour_panel.png.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / 'results' / 'w5' / 'panel'
FIG = ROOT / 'results' / 'figures' / 'w5_behaviour_panel.png'
SEEDS = (0, 1, 2)
T_STIM, T_RUN, LATE = 0.4, 0.5, 0.45
RATE = 150.0
NETS = ['real'] + [f'shuf{k}' for k in range(1, 11)]

# condition -> list of (word-or-'type:<celltype>', rate Hz); 'gf_silenced' handled in run_net
CONDS = {
    'none': [],
    'sugar': [('sugar', RATE)],                    # P1, P9 at 150 Hz
    'water': [('water', RATE)],                    # P2
    'bitter': [('bitter', RATE)],                  # P3
    'sugar+bitter': [('sugar', RATE), ('bitter', RATE)],   # P4
    'looming': [('looming', RATE)],                # P5
    'looming_GFsil': [('looming', RATE)],          # P6 (GF-silenced copy)
    'MDN': [('type:MDN', RATE)],                   # P7
    'DNg100': [('type:DNg100', RATE)],             # P8
    'sugar_50': [('sugar', 50.0)],                 # P9
    'sugar_100': [('sugar', 100.0)],
    'sugar_200': [('sugar', 200.0)],
    'wind': [('wind', RATE)],                      # X1
}
AUROC_PAIRS = {'P1': ('sugar', 'MN9'), 'P2': ('water', 'MN9'), 'P5': ('looming', 'TTMn'),
               'P7': ('MDN', 'hl'), 'P8': ('DNg100', 'legs')}
DIR_PAIRS = ['P3', 'P4', 'P6', 'P9']


# --------------------------------------------------------------------------- pools
def output_pools(meta):
    """VNC motor pools (sim.cns.motor_pools) + brain motor pools 'cb|side|type'."""
    from sim.cns import motor_pools
    pools, info = motor_pools(meta)
    info = info.assign(kind='vnc')
    cb = meta[meta.superclass.astype(str) == 'cb_motor']
    rows = []
    for (side, typ), g in cb.groupby([cb.somaSide.fillna('?').astype(str),
                                      cb.type.fillna('?').astype(str)]):
        name = f'cb|{side}|{typ}'
        pools[name] = g.index.to_numpy()
        rows.append(dict(pool=name, side=side, subclass='cb', type=typ, group='head', n=len(g), kind='cb'))
    info = pd.concat([info, pd.DataFrame(rows)], ignore_index=True)
    return pools, info


def hit_sets(info):
    t, sub = info.type.astype(str), info.subclass.astype(str)
    return {
        'MN9': info.pool[(info.kind == 'cb') & (t == 'MN9')].tolist(),
        'TTMn': info.pool[(info.kind == 'vnc') & (t == 'TTMn')].tolist(),
        'hl': info.pool[(info.kind == 'vnc') & (sub == 'hl')].tolist(),
        'fl': info.pool[(info.kind == 'vnc') & (sub == 'fl')].tolist(),
        'ml': info.pool[(info.kind == 'vnc') & (sub == 'ml')].tolist(),
        'legs': info.pool[(info.kind == 'vnc') & sub.isin(['fl', 'ml', 'hl'])].tolist(),
    }


# ------------------------------------------------------------------------ networks
def load_net(name, verbose=True):
    """'real' or 'shuf<K>': cns_model with W4_CALIB; shuffles use shuffle_W(seed=K)."""
    from sim.cns import cns_model, W4_CALIB
    if name == 'real':
        return cns_model('cns', 'calib', verbose=verbose, **W4_CALIB)
    seed = int(name.replace('shuf', ''))
    return _shuffled(seed, verbose=verbose)


def _shuffled(seed, verbose=True, capture=None):
    """cns_model('cns', 'shuffle', **W4_CALIB) with the shuffle seed replaced.

    cns_model imports shuffle_W from scripts.w2d4_word_trials inside the call,
    so swapping the module attribute for the duration of the call changes only
    the seed. capture: optional dict that receives the W passed to shuffle_W."""
    import scripts.w2d4_word_trials as wt
    from sim.cns import cns_model, W4_CALIB
    orig = wt.shuffle_W

    def patched(W, seed_=None):
        if capture is not None:
            capture['W_in'] = W.copy()
        return orig(W, seed=seed)

    wt.shuffle_W = patched
    try:
        return cns_model('cns', 'shuffle', verbose=verbose, **W4_CALIB)
    finally:
        wt.shuffle_W = orig


def silence_gf(m):
    """Copy of model m with DNp01 rows / columns of W and its gap junctions zeroed."""
    from scipy import sparse
    meta = m['meta']
    gf = np.flatnonzero(meta.type.astype(str).values == 'DNp01')
    keep = np.ones(m['W'].shape[0]); keep[gf] = 0
    D = sparse.diags(keep)
    W = (D @ m['W'] @ D).tocsc(); W.eliminate_zeros()
    gap = dict(m['gap'])
    loc = np.flatnonzero(np.isin(gap['idx'], gf))
    kg = np.ones(len(gap['idx'])); kg[loc] = 0
    Dg = sparse.diags(kg)
    C = (Dg @ gap['C'] @ Dg).tocsr(); C.eliminate_zeros()
    gap['C'] = C
    out = dict(m); out['W'] = W; out['gap'] = gap
    info = dict(gf_idx=gf.tolist(), W_nnz_removed=int(m['W'].nnz - W.nnz),
                gap_nnz_removed=int(m['gap']['C'].nnz - C.nnz))
    return out, info


# ------------------------------------------------------------------------- verify
def verify():
    """A2: seed 12345 reproduces cns_model shuffle W; degree preservation; gap not shuffled."""
    from scripts.w2d4_word_trials import shuffle_W
    from sim.cns import cns_model, W4_CALIB
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {}
    ref = cns_model('cns', 'shuffle', **W4_CALIB)
    cap = {}
    mine = _shuffled(12345, capture=cap)
    d = (ref['W'] - mine['W'])
    rep['seed12345_reproduces_cns_model_shuffle'] = bool(
        d.nnz == 0 and ref['W'].nnz == mine['W'].nnz
        and np.array_equal(ref['W'].indptr, mine['W'].indptr)
        and np.array_equal(ref['W'].indices, mine['W'].indices)
        and np.array_equal(ref['W'].data, mine['W'].data))
    rep['seed12345_nnz'] = int(mine['W'].nnz)
    del ref, mine
    real = cns_model('cns', 'calib', **W4_CALIB)
    W0 = cap['W_in'].tocsc()   # pre-shuffle W (identical for every seed: deterministic pipeline)
    for s in (1, 2):
        Ws = shuffle_W(W0, seed=s).tocsc()
        C0, Cs = W0.tocoo(), Ws.tocoo()
        in0, ins = np.bincount(C0.row, minlength=W0.shape[0]), np.bincount(Cs.row, minlength=W0.shape[0])
        out0, outs = np.diff(W0.indptr), np.diff(Ws.indptr)
        # per-presynaptic weight sums (each permuted entry keeps its own data + column)
        ow0, ows = np.asarray(W0.sum(axis=0)).ravel(), np.asarray(Ws.sum(axis=0)).ravel()
        iw0, iws = np.asarray(W0.sum(axis=1)).ravel(), np.asarray(Ws.sum(axis=1)).ravel()
        perm = np.random.default_rng(s).permutation(len(C0.data))   # as shuffle_W, before sum_duplicates
        pre_rows, pre_cols = C0.row, C0.col[perm]
        rep[f'seed{s}'] = dict(
            before_merge_in_degree_identical=bool(np.array_equal(
                np.bincount(C0.row, minlength=W0.shape[0]), np.bincount(pre_rows, minlength=W0.shape[0]))),
            before_merge_out_degree_identical=bool(np.array_equal(
                np.bincount(C0.col, minlength=W0.shape[0]), np.bincount(pre_cols, minlength=W0.shape[0]))),
            in_degree_total_abs_change_frac=float(np.abs(in0 - ins).sum() / in0.sum()),
            nnz_before=int(W0.nnz), nnz_after=int(Ws.nnz),
            duplicates_merged=int(W0.nnz - Ws.nnz),
            in_degree_identical=bool(np.array_equal(in0, ins)),
            in_degree_neurons_changed=int((in0 != ins).sum()),
            in_degree_max_abs_change=int(np.abs(in0 - ins).max()),
            out_degree_identical=bool(np.array_equal(out0, outs)),
            out_degree_neurons_changed=int((out0 != outs).sum()),
            out_degree_max_abs_change=int(np.abs(out0 - outs).max()),
            out_weight_sum_max_abs_diff=float(np.abs(ow0 - ows).max()),
            in_weight_sum_max_abs_diff=float(np.abs(iw0 - iws).max()),
            in_weight_sum_corr=float(np.corrcoef(iw0, iws)[0, 1]),
        )
    m1 = _shuffled(1)
    g0, g1 = real['gap'], m1['gap']
    rep['gap_identical_real_vs_shuf1'] = bool(
        np.array_equal(g0['idx'], g1['idx']) and (g0['C'] != g1['C']).nnz == 0)
    rep['W_real_vs_shuf1_differ'] = bool((real['W'] != m1['W']).nnz > 0)
    t = real['meta'].type.astype(str).values
    gf, tt = np.flatnonzero(t == 'DNp01'), np.flatnonzero(t == 'TTMn')
    rep['GF_to_TTMn_chem_real'] = real['W'][tt][:, gf].toarray().tolist()
    rep['GF_to_TTMn_chem_shuf1'] = m1['W'][tt][:, gf].toarray().tolist()
    pools, info = output_pools(real['meta'])
    rep['n_pools'] = dict(total=len(pools), vnc=int((info.kind == 'vnc').sum()),
                          cb=int((info.kind == 'cb').sum()))
    rep['hit_sets'] = hit_sets(info)
    rep['hit_set_sizes'] = {k: len(v) for k, v in rep['hit_sets'].items()}
    info.to_csv(OUT / 'pools.csv', index=False)
    (OUT / 'shuffle_verify.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != 'hit_sets'}, indent=1), flush=True)


# ---------------------------------------------------------------------------- run
def drive_blocks(meta, pos, cond):
    from stimuli.mixtures import word_idx
    t = meta.type.astype(str).values
    blocks = []
    for what, r in CONDS[cond]:
        if what.startswith('type:'):
            ix = np.flatnonzero(t == what[5:])
        else:
            ix = word_idx(meta, pos, what)
        assert len(ix), (cond, what)
        blocks.append((0.0, T_STIM, ix, r))
    return blocks


def run_net(name):
    import torch
    from sim import lif_torch
    from sim.lif_numpy import P
    OUT.mkdir(parents=True, exist_ok=True)
    f_rates, f_trials = OUT / f'rates_{name}.csv', OUT / f'trials_{name}.csv'
    done = set(pd.read_csv(f_trials).cond) if f_trials.exists() else set()
    todo = [c for c in CONDS if c not in done]
    if not todo:
        print(f'{name}: all conditions done', flush=True)
        return
    t0 = time.time()
    m = load_net(name)
    print(f'{name}: model built in {time.time() - t0:.0f}s', flush=True)
    meta = m['meta']
    pos = {b: i for i, b in enumerate(meta.bodyId)}
    pools, _ = output_pools(meta)
    names = list(pools)
    t = meta.type.astype(str).values
    named = {k: np.flatnonzero(t == k) for k in ('DNp01', 'MN9', 'TTMn')}
    p = dict(P); p['t_run'] = T_RUN
    n_stim, n_late0 = int(round(T_STIM / P['dt'])), int(round(LATE / P['dt']))
    m_sil = None
    for c in todo:
        net = m
        extra = dict(gf_W_nnz_removed=np.nan, gf_gap_nnz_removed=np.nan)   # fixed columns
        if c == 'looming_GFsil':
            if m_sil is None:
                m_sil, sinfo = silence_gf(m)
                print(f'{name}: GF silenced {sinfo}', flush=True)
            net, extra = m_sil, dict(gf_W_nnz_removed=sinfo['W_nnz_removed'],
                                     gf_gap_nnz_removed=sinfo['gap_nnz_removed'])
        blocks = drive_blocks(meta, pos, c)
        rrows, trows = [], []
        for seed in SEEDS:
            t1 = time.time()
            r = lif_torch.run_trial(net['W'], [], params=p, seed=seed, apl=net['apl'], gap=net['gap'],
                                    presyn=net['presyn'], blocks=blocks or None, record=True,
                                    device='cpu', rng='torch', dtype=torch.float32)
            ts, si = np.asarray(r['t']), np.asarray(r['i'])
            rate = np.bincount(si[ts < n_stim], minlength=len(meta)) / T_STIM
            late = ts >= n_late0
            stim_set = np.concatenate([b[2] for b in blocks]) if blocks else np.array([], int)
            late_nonstim = late & ~np.isin(si, stim_set)
            for nm_ in names:
                rrows.append((name, c, seed, nm_, float(rate[pools[nm_]].mean())))
            mn = (meta.superclass.astype(str).values == 'vnc_motor') | (meta.superclass.astype(str).values == 'cb_motor')
            trows.append(dict(
                net=name, cond=c, seed=seed,
                n_driven=int(len(stim_set)),
                GF_hz=float(rate[named['DNp01']].mean()),
                MN9_L_hz=float(rate[named['MN9'][meta.somaSide.astype(str).values[named['MN9']] == 'L']].mean()),
                MN9_R_hz=float(rate[named['MN9'][meta.somaSide.astype(str).values[named['MN9']] == 'R']].mean()),
                TTMn_hz=float(rate[named['TTMn']].mean()),
                active_motor_0_04=int((rate[mn] > 0).sum()),
                spikes_total=int(len(ts)),
                late_spikes=int(late.sum()),
                late_active_neurons=int(len(np.unique(si[late]))),
                late_active_motor=int(len(np.intersect1d(np.unique(si[late]), np.flatnonzero(mn)))),
                late_active_nonstim=int(len(np.unique(si[late_nonstim]))),
                wall_s=round(time.time() - t1, 1), **extra))
            tr = trows[-1]
            print(f'{name} {c} seed {seed}: MN9 {tr["MN9_L_hz"]:.1f}/{tr["MN9_R_hz"]:.1f} Hz, '
                  f'TTMn {tr["TTMn_hz"]:.1f}, GF {tr["GF_hz"]:.0f}, active MN {tr["active_motor_0_04"]}, '
                  f'late(0.45-0.5 s) active {tr["late_active_neurons"]} | {tr["wall_s"]:.0f}s', flush=True)
        rr = pd.DataFrame(rrows, columns=['net', 'cond', 'seed', 'pool', 'rate_hz'])
        rr.to_csv(f_rates, mode='a', header=not f_rates.exists(), index=False)
        pd.DataFrame(trows).to_csv(f_trials, mode='a', header=not f_trials.exists(), index=False)
    print(f'{name}: done in {time.time() - t0:.0f}s', flush=True)


# -------------------------------------------------------------------------- score
def auroc(score, hit):
    from scipy.stats import rankdata
    hit = np.asarray(hit, bool)
    r = rankdata(score)
    n1, n0 = hit.sum(), (~hit).sum()
    return float((r[hit].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def best_rank(score, hit):
    s = np.asarray(score)
    return int(1 + (s > s[np.asarray(hit, bool)].max()).sum())


def score_net(R, info, H):
    """R: rates of one network (cond, seed, pool, rate_hz). Returns dict pair -> metrics."""
    piv = R.pivot_table(index=['cond', 'seed'], columns='pool', values='rate_hz')
    piv = piv[info.pool.tolist()]
    none = piv.loc['none']
    delta = {c: (piv.loc[c] - none).mean(axis=0) for c in piv.index.get_level_values(0).unique()}
    delta_seed = {c: (piv.loc[c] - none) for c in piv.index.get_level_values(0).unique()}
    pools = np.array(info.pool)
    N = len(pools)
    out = {}
    for pr, (c, hs) in AUROC_PAIRS.items():
        hit = np.isin(pools, H[hs])
        s = delta[c].values
        out[pr] = dict(cond=c, auroc=auroc(s, hit), best_rank=best_rank(s, hit),
                       hit_mean_delta=float(s[hit].mean()), other_mean_delta=float(s[~hit].mean()))
    hl, flml = np.isin(pools, H['hl']), np.isin(pools, H['fl'] + H['ml'])
    out['P7'].update(hl_mean_delta=float(delta['MDN'].values[hl].mean()),
                     flml_mean_delta=float(delta['MDN'].values[flml].mean()),
                     hl_gt_flml=bool(delta['MDN'].values[hl].mean() > delta['MDN'].values[flml].mean()))
    mn9, ttm = np.isin(pools, H['MN9']), np.isin(pools, H['TTMn'])

    def mean_on(c, mask):
        return float(delta[c].values[mask].mean())

    def per_seed(c, mask):
        return delta_seed[c].values[:, mask].mean(axis=1)

    br3 = best_rank(delta['bitter'].values, mn9)
    d3 = mean_on('bitter', mn9)
    out['P3'] = dict(cond='bitter', mn9_delta=d3, best_rank=br3,
                     passed=bool(d3 <= 0 or br3 > N / 2))
    d1, d4 = mean_on('sugar', mn9), mean_on('sugar+bitter', mn9)
    out['P4'] = dict(cond='sugar+bitter', mn9_delta=d4, p1_mn9_delta=d1, passed=bool(d4 < d1),
                     seeds_lower=int((per_seed('sugar+bitter', mn9) < per_seed('sugar', mn9)).sum()))
    # P6: GF-silenced copy minus the intact network's 'none'
    d5, d6 = mean_on('looming', ttm), mean_on('looming_GFsil', ttm)
    out['P6'] = dict(cond='looming_GFsil', ttmn_delta=d6, p5_ttmn_delta=d5, passed=bool(d6 < d5),
                     frac_removed=float(1 - d6 / d5) if d5 > 0 else float('nan'),
                     seeds_lower=int((per_seed('looming_GFsil', ttm) < per_seed('looming', ttm)).sum()))
    dose = [mean_on(c, mn9) for c in ('sugar_50', 'sugar_100', 'sugar', 'sugar_200')]
    out['P9'] = dict(cond='sugar_50/100/150/200', mn9_delta_50_100_150_200=dose,
                     passed=bool(np.all(np.diff(dose) > 0)))
    fl = np.isin(pools, H['fl'])
    out['X1'] = dict(cond='wind', auroc=auroc(delta['wind'].values, fl),
                     best_rank=best_rank(delta['wind'].values, fl),
                     fl_mean_delta=float(delta['wind'].values[fl].mean()),
                     other_mean_delta=float(delta['wind'].values[~fl].mean()))
    return out


def read_trials(path):
    """trials_<net>.csv. Files written before the fixed-column fix have a 15-field
    header and 17 fields on the looming_GFsil rows (two GF-silencing columns
    appended without a header); both layouts are read here."""
    import csv
    with open(path) as f:
        rows = list(csv.reader(f))
    head, body = rows[0], rows[1:]
    extra = ['gf_W_nnz_removed', 'gf_gap_nnz_removed']
    if 'gf_W_nnz_removed' not in head:
        head = head + extra
        body = [r + [''] * (len(head) - len(r)) for r in body]
    df = pd.DataFrame(body, columns=head)
    for c in df.columns:
        if c not in ('net', 'cond'):
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def score():
    info = pd.read_csv(OUT / 'pools.csv')
    H = hit_sets(info)
    nets = [n for n in NETS if (OUT / f'rates_{n}.csv').exists()]
    S = {}
    for n in nets:
        R = pd.read_csv(OUT / f'rates_{n}.csv')
        missing = set(CONDS) - set(R.cond)
        if missing:
            print(f'{n}: incomplete, missing {sorted(missing)}; skipped')
            continue
        S[n] = score_net(R, info, H)
    shufs = [n for n in S if n != 'real']
    K = len(shufs)
    rows = []
    for n, sc in S.items():
        for pr, d in sc.items():
            row = dict(net=n, pair=pr, **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in d.items()})
            if pr in AUROC_PAIRS:
                row['crit_pass'] = bool(d['auroc'] >= 0.9)
            elif pr in DIR_PAIRS:
                row['crit_pass'] = d['passed']
            rows.append(row)
    df = pd.DataFrame(rows)
    summ = dict(K=K, nets=list(S), n_pools=len(info))
    if 'real' in S:
        for pr in AUROC_PAIRS:
            a = S['real'][pr]['auroc']
            ge = sum(S[s][pr]['auroc'] >= a for s in shufs)
            df.loc[(df.net == 'real') & (df.pair == pr), 'p_emp'] = (1 + ge) / (K + 1)
        real = df[df.net == 'real'].set_index('pair')
        hit = {}
        for pr in list(AUROC_PAIRS) + DIR_PAIRS:
            hit[pr] = bool(real.loc[pr, 'crit_pass'] and (pr not in AUROC_PAIRS or real.loc[pr, 'p_emp'] <= 0.1))
        df.loc[df.net == 'real', 'hit'] = df.loc[df.net == 'real', 'pair'].map(hit)
        summ['real_hits'] = int(sum(hit.values()))
        summ['real_hits_criterion_only'] = int(real.loc[list(AUROC_PAIRS) + DIR_PAIRS, 'crit_pass'].sum())
        summ['real_hit_pairs'] = [k for k, v in hit.items() if v]
    sh = df[df.net.isin(shufs) & df.pair.isin(list(AUROC_PAIRS) + DIR_PAIRS)]
    df.loc[sh.index, 'hit'] = sh.crit_pass
    summ['shuffle_hits'] = {n: int(g.crit_pass.sum()) for n, g in sh.groupby('net')}
    summ['shuffle_pass_count_per_pair'] = {p: int(g.crit_pass.sum()) for p, g in sh.groupby('pair')}
    # MN9 absolute rates (Hz) for P1 / P2 / P4, mean over seeds, L and R
    T = pd.concat([read_trials(OUT / f'trials_{n}.csv') for n in S])
    ab = T[T.cond.isin(['none', 'sugar', 'water', 'sugar+bitter', 'bitter'])]
    summ['mn9_abs_hz'] = {n: {c: dict(L=round(g.MN9_L_hz.mean(), 2), R=round(g.MN9_R_hz.mean(), 2))
                              for c, g in gn.groupby('cond')} for n, gn in ab.groupby('net')}
    late = T.groupby(['net', 'cond']).late_active_nonstim.mean().unstack()
    summ['late_active_nonstim_mean'] = late.round(1).to_dict(orient='index')
    df.to_csv(OUT / 'scorecard.csv', index=False)
    (OUT / 'summary.json').write_text(json.dumps(summ, indent=1, default=float))
    print(df.to_string(max_colwidth=40))
    print(json.dumps(summ, indent=1, default=float))
    if 'real' in S and K:
        figure(df, shufs)


def figure(df, shufs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    prs = list(AUROC_PAIRS) + ['X1']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw=dict(width_ratios=[3, 2]))
    ax = axes[0]
    rng = np.random.default_rng(0)
    for i, pr in enumerate(prs):
        a = df[(df.pair == pr) & df.net.isin(shufs)].auroc.values
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(a)), a, s=14, color='0.6',
                   label='shuffle (10)' if i == 0 else None, zorder=2)
        r = df[(df.pair == pr) & (df.net == 'real')].auroc.values[0]
        ax.scatter([i], [r], s=70, marker='D', color='C3', label='real' if i == 0 else None, zorder=3)
    ax.axhline(0.9, ls='--', color='k', lw=0.8); ax.axhline(0.5, ls=':', color='0.5', lw=0.8)
    lab = {'P1': 'P1 sugar→MN9', 'P2': 'P2 water→MN9', 'P5': 'P5 loom→TTMn', 'P7': 'P7 MDN→hl',
           'P8': 'P8 DNg100→legs', 'X1': 'X1 wind→fl\n(not scored)'}
    ax.set_xticks(range(len(prs))); ax.set_xticklabels([lab[p] for p in prs], fontsize=8)
    ax.set_ylabel('AUROC (expected pools vs other output pools, 436 in all)'); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc='lower left'); ax.set_title('AUROC pairs: real vs 10 shuffles', fontsize=10)
    ax = axes[1]
    dp = DIR_PAIRS
    real = [bool(df[(df.pair == p) & (df.net == 'real')].crit_pass.values[0]) for p in dp]
    npass = [int(df[(df.pair == p) & df.net.isin(shufs)].crit_pass.sum()) for p in dp]
    ax.bar(np.arange(len(dp)) - 0.2, [10 * x for x in real], width=0.4, color='C3', label='real (pass = full bar)')
    ax.bar(np.arange(len(dp)) + 0.2, npass, width=0.4, color='0.6', label='shuffles passing (of 10)')
    ax.set_xticks(range(len(dp))); ax.set_xticklabels(['P3 bitter\n¬MN9', 'P4 sug+bit\n< sugar',
                                                       'P6 GF off\n< loom', 'P9 dose\nmonotonic'], fontsize=8)
    ax.set_ylim(0, 10.5); ax.set_ylabel('count'); ax.legend(fontsize=8)
    ax.set_title('Direction pairs', fontsize=10)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130)
    print('figure', FIG)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'score'
    if cmd == 'verify':
        verify()
    elif cmd == 'run':
        which = sys.argv[2] if len(sys.argv) > 2 else 'real'
        for n in (NETS if which == 'all' else which.split(',')):
            run_net(n)
    elif cmd == 'score':
        score()
    else:
        raise SystemExit(__doc__)
