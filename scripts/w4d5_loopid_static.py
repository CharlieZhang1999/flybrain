#!/usr/bin/env python
"""W4D5 T2d: static (no simulation) audit of the wind loop LOOP5 for annotation / build errors.

Purpose: answers brief Q1-Q3 with data.
  Q1  NT table: model nt/sign (effective sign of the column in the final calib W) vs MaleCNS
      body/type predictions + confidence + consensus + ground_truth vs FlyWire top_nt /
      top_nt_conf / known_nt, for LOOP5 and every partner type above threshold; reference
      percentiles of the confidences; hemilineage NT composition for the loop's hemilineages.
  Q2  identity: per-body annotation fields, cell counts vs FlyWire (annot.tsv) counts, per-body
      synapse totals (in/out, traced partners) vs the type and the DN population, L/R symmetry of
      the loop sub-matrix.
  Q3  connectivity: reciprocal cell-pair counts vs all DN-DN and GNG-GNG (cb_intrinsic + DN,
      somaNeuromere / type GNG*) pairs; autapses; inhibitory inputs to LOOP5 (final W vs raw);
      synapses removed by monoamine removal, presyn row removal, LN-PN cut, APL, mn_glu,
      sign_fix, clique_gain, right-JO mirror.
Usage:  python scripts/w4d5_loopid_static.py
Output: results/w4/regression/loop_id/static.txt, nt_table.csv, inhib_inputs.csv,
        body_table.csv, recip_pairs.csv
Thresholds (stated): partner types = type-pair sum >= 100 synapses into or out of LOOP5;
        inhibitory input types listed from >= 5 synapses.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'results' / 'w4' / 'regression' / 'loop_id'
D = ROOT / 'data' / 'malecns'
LOOP5 = ['DNge027', 'GNG423', 'DNg21', 'DNg62', 'DNge044']
VSYN = 0.275e-3 * 0.581          # volts per synapse in the calib model
PARTNER_MIN = 100
INH_MIN = 5
lines = []


def P(s=''):
    print(s, flush=True)
    lines.append(str(s))


def pct(ref, x):
    ref = np.asarray(ref, float); ref = ref[np.isfinite(ref)]
    return 100.0 * (ref < x).mean() if len(ref) else np.nan


def main():
    from sim.cns import cns_model, W4_CALIB
    from sim.malecns import load
    OUT.mkdir(parents=True, exist_ok=True)
    m = cns_model('cns', 'calib', verbose=False, **W4_CALIB)
    meta, W = m['meta'], m['W'].tocsc()
    _, W1 = load(w_syn=1.0, monoamine='removed')          # signed synapse counts
    _, We = load(w_syn=1.0, monoamine='excitatory')       # monoamines kept as +
    W1 = W1.tocsc(); We = We.tocsc()
    S = abs(W1).tocsr()                                    # unsigned counts, rows post
    ty = meta.type.fillna('?').astype(str).values
    n = len(meta)
    L = np.isin(ty, LOOP5)
    li = np.flatnonzero(L)
    # effective sign of each presynaptic neuron in the final W
    Wc = W.tocsc()
    colsum = np.asarray(Wc.sum(axis=0)).ravel()
    colabs = np.asarray(abs(Wc).sum(axis=0)).ravel()
    eff = np.where(colabs > 0, np.sign(colsum), 0)

    ann = pd.read_feather(D / 'body-annotations-male-cns-v1.0-minconf-0.5.feather')
    ann = ann.set_index('bodyId')
    nt = pd.read_feather(D / 'body-neurotransmitters-male-cns-v1.0.feather').set_index('body')
    fw = pd.read_csv(ROOT / 'data' / 'flywire_annot' / 'annot.tsv', sep='\t', low_memory=False)

    # ---------------- partner selection
    Sl_in = S[li]                                    # LOOP5 rows: inputs
    inp = pd.Series(np.asarray(Sl_in.sum(axis=0)).ravel()).groupby(ty).sum()
    Sl_out = S.T.tocsr()[li]                         # LOOP5 cols as rows: outputs
    outp = pd.Series(np.asarray(Sl_out.sum(axis=0)).ravel()).groupby(ty).sum()
    part = sorted(set(inp[inp >= PARTNER_MIN].index) | set(outp[outp >= PARTNER_MIN].index))
    part = [t for t in part if t not in LOOP5 and t != '?' and t != 'nan']
    # inhibitory inputs (final model sign)
    Wl = W[li]
    inh_v = pd.Series(np.asarray(Wl.minimum(0).sum(axis=0)).ravel() / VSYN)
    exc_v = pd.Series(np.asarray(Wl.maximum(0).sum(axis=0)).ravel() / VSYN)
    inh_t = (-inh_v).groupby(ty).sum().sort_values(ascending=False)
    inh_types = [t for t in inh_t[inh_t >= INH_MIN].index if t not in ('?', 'nan')]
    n_inh_untyped = float(inh_t.get('?', 0))
    types = LOOP5 + [t for t in part if t not in LOOP5] + [t for t in inh_types if t not in part]

    # ---------------- Q1: NT table
    rows = []
    for t in types:
        ix = np.flatnonzero(ty == t)
        bids = meta.bodyId.values[ix]
        for j, b in zip(ix, bids):
            r = nt.loc[b] if b in nt.index else None
            a = ann.loc[b]
            fwt = a['flywireType'] if isinstance(a['flywireType'], str) else t
            f = fw[fw.cell_type == fwt]
            if not len(f):
                f = fw[fw.cell_type == t]
            fside = {'L': 'left', 'R': 'right'}.get(str(a['somaSide']), None)
            fs = f[f.side == fside] if fside is not None and (f.side == fside).any() else f
            rows.append(dict(
                type=t, bodyId=b, side=a['somaSide'], role='LOOP5' if t in LOOP5 else
                ('partner' if t in part else 'inh_input'),
                syn_into_loop=int(S[li][:, j].sum()), syn_from_loop=int(S[j][:, li].sum()),
                model_nt=meta.nt.values[j], model_sign_meta=int(meta.sign.values[j]),
                model_sign_eff=int(eff[j]),
                mcns_pred=None if r is None else r['predicted_nt'],
                mcns_conf=np.nan if r is None else r['predicted_nt_confidence'],
                mcns_npred=np.nan if r is None else r['total_nt_predictions'],
                mcns_type_pred=None if r is None else r['celltype_predicted_nt'],
                mcns_type_conf=np.nan if r is None else r['celltype_predicted_nt_confidence'],
                mcns_consensus=None if r is None else r['consensus_nt'],
                mcns_ground_truth=None if r is None else r['ground_truth'],
                flywireType=a['flywireType'], fw_n=len(f),
                fw_top_nt=';'.join(sorted(set(fs.top_nt.dropna().astype(str)))),
                fw_conf=fs.top_nt_conf.mean() if len(fs) else np.nan,
                fw_known_nt=';'.join(sorted(set(f.known_nt.dropna().astype(str)))),
                fw_known_src=';'.join(sorted(set(f.known_nt_source.dropna().astype(str)))),
                itoleeHl=a['itoleeHl'], trumanHl=a['trumanHl']))
    T = pd.DataFrame(rows)
    # reference distributions
    tr = nt[nt.index.isin(meta.bodyId)]
    ach = tr[tr.predicted_nt == 'acetylcholine'].predicted_nt_confidence
    dnb = meta.bodyId[meta.superclass.astype(str) == 'descending_neuron']
    ach_dn = tr[(tr.predicted_nt == 'acetylcholine') & tr.index.isin(dnb)].predicted_nt_confidence
    fw_ach = fw[fw.top_nt == 'acetylcholine'].top_nt_conf
    fw_ach_dn = fw[(fw.top_nt == 'acetylcholine') & (fw.super_class == 'descending')].top_nt_conf
    T['mcns_conf_pct_achDN'] = [pct(ach_dn, c) for c in T.mcns_conf]
    T['fw_conf_pct_achDN'] = [pct(fw_ach_dn, c) for c in T.fw_conf]
    T['model_inh'] = T.model_sign_eff < 0
    pred_inh = T.mcns_pred.isin(['gaba', 'glutamate', 'histamine'])
    fw_inh = T.fw_top_nt.str.contains('gaba|glutamate|histamine', na=False) | \
        T.fw_known_nt.str.contains('gaba|glutamate|histamine', na=False)
    T['flag'] = ''
    T.loc[(T.mcns_conf < 0.6), 'flag'] += 'mcns_conf<0.6;'
    T.loc[(T.fw_conf < 0.5), 'flag'] += 'fw_conf<0.5;'
    T.loc[(T.model_sign_eff > 0) & fw_inh, 'flag'] += 'model_exc_but_FW_inh;'
    T.loc[(T.model_sign_eff < 0) & ~fw_inh & (T.fw_n > 0), 'flag'] += 'model_inh_but_FW_exc;'
    T.loc[(T.mcns_pred != T.mcns_type_pred), 'flag'] += 'body_vs_type_pred;'
    T.loc[(T.model_sign_eff > 0) & pred_inh, 'flag'] += 'model_exc_but_mcns_inh;'
    T.to_csv(OUT / 'nt_table.csv', index=False)

    P('# T2d static audit of LOOP5 (no simulation)')
    P(f'partner threshold: type-pair sum >= {PARTNER_MIN} syn into or out of LOOP5 -> {len(part)} types; '
      f'inhibitory input types >= {INH_MIN} syn (final W) -> {len(inh_types)} types')
    P('')
    P('## Reference distributions of NT confidence')
    for nm, r in [('MaleCNS predicted_nt_confidence, all traced ACh-predicted', ach),
                  ('MaleCNS, ACh-predicted DNs', ach_dn),
                  ('FlyWire top_nt_conf, all ACh', fw_ach), ('FlyWire top_nt_conf, ACh DNs', fw_ach_dn)]:
        q = np.nanpercentile(r, [5, 10, 25, 50, 75])
        P(f'{nm}: n {len(r):,}; p5 {q[0]:.3f} p10 {q[1]:.3f} p25 {q[2]:.3f} median {q[3]:.3f} p75 {q[4]:.3f}')
    P('')
    P('## Q1 per-type NT summary (mean over cells; pct = percentile among ACh-predicted DNs)')
    g = T.groupby('type', sort=False).agg(
        role=('role', 'first'), n=('bodyId', 'size'), into=('syn_into_loop', 'sum'), out=('syn_from_loop', 'sum'),
        nt=('model_nt', lambda s: ';'.join(sorted(set(s)))), sign=('model_sign_eff', lambda s: ';'.join(map(str, sorted(set(s))))),
        mc=('mcns_pred', lambda s: ';'.join(sorted(set(map(str, s))))), mconf=('mcns_conf', 'mean'),
        mpct=('mcns_conf_pct_achDN', 'mean'),
        tconf=('mcns_type_conf', 'mean'), cons=('mcns_consensus', lambda s: ';'.join(sorted(set(map(str, s))))),
        gt=('mcns_ground_truth', lambda s: ';'.join(sorted(set(map(str, s))))),
        fwt=('flywireType', 'first'), fwn=('fw_n', 'first'), fwnt=('fw_top_nt', lambda s: ';'.join(sorted(set(s)))),
        fwc=('fw_conf', 'mean'), fwpct=('fw_conf_pct_achDN', 'mean'), known=('fw_known_nt', 'first'),
        flags=('flag', lambda s: ';'.join(sorted(set(';'.join(s).split(';')) - {''}))))
    with pd.option_context('display.width', 300, 'display.max_columns', 40, 'display.max_rows', 400,
                           'display.float_format', '{:.3f}'.format):
        P(g.to_string())
    P('')
    P('Note: body-neurotransmitters-male-cns-v1.0.feather holds only the top class and its confidence '
      '(columns: ' + ', '.join(nt.reset_index().columns) + '); per-class probabilities are only in the '
      '2.65 GB tbar-neurotransmitters feather, which is not on disk (only tbar_nt_AL.parquet, AL neurons).')
    al = pd.read_parquet(D / 'tbar_nt_AL.parquet', columns=['body'])
    P(f'LOOP5/partner bodies present in tbar_nt_AL.parquet: {int(T.bodyId.isin(al.body).sum())}')
    # hemilineage composition
    P('')
    P('## Hemilineage NT composition (MaleCNS predicted_nt over all traced bodies of the hemilineage)')
    a2 = ann[ann.status == 'Traced'].join(nt[['predicted_nt']], how='left')
    for col in ['itoleeHl', 'trumanHl']:
        for hl in sorted(set(T[T.role == 'LOOP5'][col].dropna()) | set(T[T.type == 'SAD093'][col].dropna())):
            s = a2[a2[col] == hl].predicted_nt.value_counts()
            P(f'{col}={hl}: n {s.sum()}: ' + ', '.join(f'{k} {v}' for k, v in s.items()))
        if col == 'trumanHl':
            for hl in ['05B', '08B']:
                s = a2[a2[col].astype(str).str.startswith(hl)].predicted_nt.value_counts()
                P(f'{col} startswith {hl}: n {s.sum()}: ' + ', '.join(f'{k} {v}' for k, v in s.items()))

    # ---------------- Q2: identity
    P('')
    P('## Q2 identity: bodies of LOOP5 and main partners')
    keyt = LOOP5 + ['SAD093', 'GNG282']
    cols = ['type', 'instance', 'somaSide', 'group', 'flywireType', 'synonyms', 'statusLabel', 'mancType',
            'mancGroup', 'supertype', 'subclass', 'itoleeHl', 'trumanHl', 'somaLocation', 'birthtime']
    ins_all = np.asarray(S.sum(axis=1)).ravel()
    out_all = np.asarray(S.sum(axis=0)).ravel()
    B = []
    for t in keyt:
        for j in np.flatnonzero(ty == t):
            b = meta.bodyId.values[j]
            a = ann.loc[b]
            B.append(dict(bodyId=b, **{c: a[c] for c in cols}, syn_in=int(ins_all[j]), syn_out=int(out_all[j]),
                          n_pre_partners=int((S[j] > 0).sum()), n_post_partners=int((S[:, j] > 0).sum())))
    Bt = pd.DataFrame(B)
    Bt.to_csv(OUT / 'body_table.csv', index=False)
    with pd.option_context('display.width', 300, 'display.max_columns', 40, 'display.max_colwidth', 28):
        P(Bt.to_string(index=False))
    # cell counts vs FlyWire, MANC-type counts in MaleCNS
    P('')
    P('cell counts: MaleCNS traced (all statuses in annotation) vs FlyWire annot.tsv (flywireType)')
    for t in keyt:
        fwt = ann[ann.type == t].flywireType.dropna().unique()
        nfw = int(fw.cell_type.isin(fwt).sum())
        nall = int((ann.type == t).sum())
        P(f'  {t}: MaleCNS traced {int((ty == t).sum())} (all statuses {nall}), flywireType {list(fwt)}: FlyWire {nfw}; '
          f'FlyWire sides ' + str(fw[fw.cell_type.isin(fwt)].side.value_counts().to_dict()))
    # outliers: within-type ratio and z vs DN population (log total synapses)
    P('')
    P('synapse-total outliers: log10(in+out) z-score vs all traced DNs; ratio to type mean')
    dn = meta.superclass.astype(str).values == 'descending_neuron'
    tot = ins_all + out_all
    lt = np.log10(tot[dn & (tot > 0)])
    for t in keyt:
        ix = np.flatnonzero(ty == t)
        mu = tot[ix].mean()
        P(f'  {t}: ' + ', '.join(f'{meta.bodyId.values[j]} in {ins_all[j]:.0f} out {out_all[j]:.0f} '
                                 f'z {(np.log10(tot[j]) - lt.mean()) / lt.std():+.2f} ratio {tot[j] / mu:.2f}'
                                 for j in ix))
    # within-type L/R asymmetry reference: ratio max/min of total synapses for all 2-cell DN types
    tt2 = pd.Series(ty[dn]).value_counts()
    rr = []
    for t in tt2[tt2 == 2].index:
        ix = np.flatnonzero(ty == t)
        if tot[ix].min() > 0:
            rr.append(tot[ix].max() / tot[ix].min())
    rr = np.array(rr)
    P(f'  reference: max/min total synapses within the {len(rr)} two-cell DN types: median {np.median(rr):.2f}, '
      f'p90 {np.percentile(rr, 90):.2f}, p95 {np.percentile(rr, 95):.2f}')
    for t in ['DNge027', 'DNg21', 'DNg62', 'DNge044']:
        ix = np.flatnonzero(ty == t)
        x = tot[ix].max() / tot[ix].min()
        P(f'  {t}: max/min {x:.2f} (percentile {pct(rr, x):.0f})')
    # loop submatrix per body (raw synapses, rows post)
    P('')
    P('LOOP5 body x body synapse matrix (rows = post, cols = pre; MaleCNS raw counts)')
    lab = [f'{ty[j]}_{meta.somaSide.values[j]}_{meta.bodyId.values[j]}' for j in li]
    order = np.argsort(lab)
    M5 = S[li][:, li].toarray()[np.ix_(order, order)]
    lab = [lab[k] for k in order]
    df5 = pd.DataFrame(M5.astype(int), index=lab, columns=[x.split('_')[0][:7] + x.split('_')[1] for x in lab])
    with pd.option_context('display.width', 300, 'display.max_columns', 40):
        P(df5.to_string())
    # L/R symmetry: ipsi vs contra counts and mirror-pair differences
    side = meta.somaSide.astype(str).values
    P('')
    P('L/R symmetry of loop inputs per type pair (pre -> post): synapses L->L, R->R, L->R, R->L')
    for tpre in LOOP5:
        for tpost in LOOP5:
            pre = np.flatnonzero(ty == tpre); post = np.flatnonzero(ty == tpost)
            c = {}
            for sp in 'LR':
                for so in 'LR':
                    c[sp + so] = int(S[post[side[post] == so]][:, pre[side[pre] == sp]].sum())
            if sum(c.values()) >= 20:
                P(f'  {tpre:8s} -> {tpost:8s}: LL {c["LL"]:4d} RR {c["RR"]:4d} | LR {c["LR"]:4d} RL {c["RL"]:4d}')

    # ---------------- Q3: connectivity anomalies
    P('')
    P('## Q3 connectivity')
    gng = (np.char.startswith(ty.astype("U"), 'GNG') |
           np.isin(meta.somaNeuromere.astype(str).values, ['GNG', 'LB', 'MX', 'MD', 'SEZ']))
    gngdn = gng | (dn & np.char.startswith(ty.astype("U"), 'DNg'))
    U = np.flatnonzero(dn | gngdn | L)
    A = S[U][:, U].toarray()                         # rows post, cols pre
    np.fill_diagonal(A, 0)
    iu = np.triu_indices(len(U), 1)
    pairs = pd.DataFrame(dict(a=U[iu[1]], b=U[iu[0]], ab=A[iu[0], iu[1]], ba=A[iu[1], iu[0]]))
    pairs = pairs[(pairs.ab > 0) | (pairs.ba > 0)].reset_index(drop=True)
    del A
    pairs['recip_min'] = np.minimum(pairs.ab, pairs.ba)
    pairs['sum'] = pairs.ab + pairs.ba
    sets = {'DN-DN': dn, 'GNG-GNG (GNG* types, GNG/LB/MX/MD neuromere, DNg*/DNge*)': gngdn}
    lp = pairs[L[pairs.a] & L[pairs.b]].copy()
    lp['A'] = [f'{ty[x]}_{side[x]}' for x in lp.a]; lp['B'] = [f'{ty[x]}_{side[x]}' for x in lp.b]
    for nm, msk in sets.items():
        ref = pairs[msk[pairs.a] & msk[pairs.b]]
        refr = ref[ref.recip_min > 0]
        P(f'{nm}: {len(ref):,} connected cell pairs, {len(refr):,} reciprocal (both directions >= 1 syn); '
          f'reciprocal min(ab,ba): median {refr.recip_min.median():.0f}, p90 {refr.recip_min.quantile(.9):.0f}, '
          f'p99 {refr.recip_min.quantile(.99):.0f}, max {refr.recip_min.max():.0f}')
        lp[f'pct_{nm[:6]}'] = [pct(refr.recip_min, x) for x in lp.recip_min]
        lp[f'rank_{nm[:6]}'] = [int((refr.recip_min > x).sum()) + 1 for x in lp.recip_min]
    lp = lp.sort_values('recip_min', ascending=False)
    lp[['A', 'B', 'ab', 'ba', 'recip_min', 'sum'] + [c for c in lp.columns if c.startswith(('pct_', 'rank_'))]] \
        .to_csv(OUT / 'recip_pairs.csv', index=False)
    with pd.option_context('display.width', 300, 'display.max_rows', 100, 'display.float_format', '{:.1f}'.format):
        P(lp[['A', 'B', 'ab', 'ba', 'recip_min'] + [c for c in lp.columns if c.startswith(('pct_', 'rank_'))]]
          .head(30).to_string(index=False))
    # top reciprocal DN-DN pairs in the whole CNS, for context
    ref = pairs[dn[pairs.a] & dn[pairs.b]].sort_values('recip_min', ascending=False).head(15)
    P('top 15 reciprocal DN-DN cell pairs in MaleCNS (min of the two directions):')
    for _, r in ref.iterrows():
        P(f'  {ty[int(r.a)]}_{side[int(r.a)]} <-> {ty[int(r.b)]}_{side[int(r.b)]}: {r.ab:.0f} / {r.ba:.0f}')
    # LOOP5 as a whole: within-group synapse fraction vs other 5-type groups? -> fraction of input from loop
    loop_in = S[li][:, li].sum(); all_in = S[li].sum()
    P(f'LOOP5 receives {loop_in:.0f} of {all_in:.0f} input synapses ({100 * loop_in / all_in:.1f}%) from LOOP5 itself')
    dnfr = []
    for j in np.flatnonzero(dn):
        tot_in = ins_all[j]
        if tot_in > 0:
            dnfr.append(S[j][:, dn].sum() / tot_in)
    P(f'reference: fraction of a DN\'s input from other DNs, all traced DNs: median {np.median(dnfr):.3f}, '
      f'p90 {np.percentile(dnfr, 90):.3f}, p99 {np.percentile(dnfr, 99):.3f}; LOOP5 DN members: ' +
      ', '.join(f'{ty[j]}_{side[j]} {S[j][:, dn].sum() / ins_all[j]:.3f} (pct {pct(dnfr, S[j][:, dn].sum() / ins_all[j]):.0f})'
                for j in li if dn[j]))
    # autapses
    dg_raw = W1.diagonal()
    dg_mod = W.diagonal()
    P(f'autapses: raw W diagonal nnz over the whole CNS {int((dg_raw != 0).sum())}; LOOP5 raw diag '
      f'{dg_raw[li].tolist()}; final model diag nnz {int((dg_mod != 0).sum())}, LOOP5 {dg_mod[li].tolist()}')

    # inhibitory inputs: final W vs raw
    P('')
    P('### Inhibitory input onto LOOP5 (final calib W; syn-equivalents = W / VSYN)')
    tot_exc = exc_v.sum(); tot_inh = -inh_v.sum()
    P(f'untyped (type NaN) inhibitory input onto LOOP5: {n_inh_untyped:.0f} syn-eq')
    P(f'total onto LOOP5 (12 cells): excitatory {tot_exc:.0f}, inhibitory {tot_inh:.0f} syn-eq '
      f'(inhibitory fraction {tot_inh / (tot_exc + tot_inh):.3f})')
    rawneg = pd.Series(np.asarray(W1[li].minimum(0).sum(axis=0)).ravel()).groupby(ty).sum()
    rawpos = pd.Series(np.asarray(W1[li].maximum(0).sum(axis=0)).ravel()).groupby(ty).sum()
    tot_ins = pd.Series(ins_all)
    fr_dn = []
    Wr = W1.tocsr()
    for j in np.flatnonzero(dn):
        row = Wr[j]
        neg = -row.data[row.data < 0].sum(); pos_ = row.data[row.data > 0].sum()
        if neg + pos_ > 0:
            fr_dn.append(neg / (neg + pos_))
    fl = [(-Wr[j].data[Wr[j].data < 0].sum()) / abs(Wr[j].data).sum() for j in li]
    P(f'raw inhibitory fraction of input, all traced DNs: median {np.median(fr_dn):.3f}, p10 {np.percentile(fr_dn, 10):.3f}, '
      f'p25 {np.percentile(fr_dn, 25):.3f}; LOOP5 cells: ' +
      ', '.join(f'{ty[j]}_{side[j]} {x:.3f} (pct {pct(fr_dn, x):.0f})' for j, x in zip(li, fl)))
    I = pd.DataFrame(dict(model_inh_syn_eq=inh_t, raw_inh_syn=-rawneg.reindex(inh_t.index),
                          raw_exc_syn=rawpos.reindex(inh_t.index)))
    I = I[I.model_inh_syn_eq >= INH_MIN]
    I['nt'] = [';'.join(sorted(set(meta.nt.values[ty == t]))) for t in I.index]
    I['mcns_conf'] = [nt.reindex(meta.bodyId.values[ty == t]).predicted_nt_confidence.mean() for t in I.index]
    I['targets'] = [', '.join(f'{u}:{int(-W[np.flatnonzero(ty == u)][:, ty == t].sum() / VSYN)}'
                              for u in LOOP5 if W[np.flatnonzero(ty == u)][:, ty == t].sum() < 0) for t in I.index]
    I.to_csv(OUT / 'inhib_inputs.csv')
    with pd.option_context('display.width', 300, 'display.max_rows', 200, 'display.max_colwidth', 80,
                           'display.float_format', '{:.2f}'.format):
        P(I.to_string())
    # build-removed inputs: compare final W (rescaled) vs raw, LOOP5 rows only
    P('')
    P('### What the build changed on LOOP5 rows (inputs) and columns (outputs)')
    Wm = (W / VSYN).tocsr()
    R1 = W1.tocsr()
    dif_in = (Wm[li] - R1[li]).tocoo()
    big = np.abs(dif_in.data) > 0.5
    P(f'LOOP5 rows: entries changed by the build (|final/VSYN - raw| > 0.5 syn): {int(big.sum())}; '
      f'by presynaptic type: ' + str(pd.Series(dif_in.data[big]).groupby(ty[dif_in.col[big]]).sum().round(0).to_dict()))
    dif_out = (Wm[:, li] - R1[:, li]).tocoo()
    big = np.abs(dif_out.data) > 0.5
    P(f'LOOP5 columns: entries changed: {int(big.sum())}; by postsynaptic type (top 10 |sum|): ' +
      str(pd.Series(dif_out.data[big]).groupby(ty[dif_out.row[big]]).sum().sort_values(key=abs, ascending=False).head(10).round(0).to_dict()))
    # monoamine synapses onto LOOP5
    mono = (We.tocsr()[li] - R1[li]).tocoo()
    mk = mono.data != 0
    P(f'monoamine synapses removed onto LOOP5 (mono-excitatory minus mono-removed): {mono.data[mk].sum():.0f} syn; '
      f'by type/nt: ' + str(pd.Series(mono.data[mk]).groupby([ty[mono.col[mk]][q] + '/' + meta.nt.values[mono.col[mk]][q]
                                                                   for q in range(mk.sum())]).sum().to_dict()))
    mono_o = (We.tocsr()[:, li] - R1[:, li])
    P(f'monoamine synapses from LOOP5 (should be 0, LOOP5 is ACh): {abs(mono_o).sum():.0f}')
    # SAD093 / entry: JO -> SAD093 raw vs final (right-JO mirror)
    sad = np.flatnonzero(ty == 'SAD093')
    jo = np.char.startswith(ty.astype("U"), 'JO-')
    from sim.jo_mirror import _side
    jside = _side(meta)          # instance suffix, as the mirror uses (somaSide is empty for JO)
    for j in sad:
        P(f'SAD093_{side[j]}: JO input raw {R1[j][:, jo].sum():.0f} syn, final {Wm[j][:, jo].sum():.0f} syn-eq '
          f'(right-JO mirror); JO-L raw {R1[j][:, jo & (jside == "L")].sum():.0f}, JO-R raw {R1[j][:, jo & (jside == "R")].sum():.0f} '
          f'-> final {Wm[j][:, jo & (jside == "R")].sum():.0f}')
    # inhibitory inputs onto SAD093 (entry) for completeness
    for j in sad:
        r = W[j].toarray().ravel() / VSYN
        s = pd.Series(r[r < 0]).groupby(ty[r < 0]).sum().sort_values().head(5)
        P(f'SAD093_{side[j]} inhibitory inputs (final): ' + ', '.join(f'{k} {-v:.0f}' for k, v in s.items()) +
          f'; total inh {-r[r < 0].sum():.0f} vs exc {r[r > 0].sum():.0f}')
    # glutamatergic partners treated as exc or vice versa: listed in nt table flags
    P('')
    P('### Partners whose model sign disagrees with MaleCNS or FlyWire (from nt_table.csv flags)')
    for flag in ['model_exc_but_FW_inh', 'model_inh_but_FW_exc', 'model_exc_but_mcns_inh', 'mcns_conf<0.6',
                 'fw_conf<0.5', 'body_vs_type_pred']:
        x = T[T.flag.str.contains(flag, regex=False) & ((T.syn_into_loop + T.syn_from_loop) >= 10)]
        P(f'{flag}: ' + (', '.join(f'{r.type}_{r.side}({r.role}, in {r.syn_into_loop}, out {r.syn_from_loop}, '
                                   f'model {r.model_nt}/{r.model_sign_eff:+d}, mcns {r.mcns_pred} {r.mcns_conf:.2f}, '
                                   f'FW {r.fw_top_nt} {r.fw_conf:.2f} known {r.fw_known_nt or "-"})'
                                   for r in x.itertuples()) or 'none'))
    (OUT / 'static.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
