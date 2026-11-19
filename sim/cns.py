"""Week 4: the full MaleCNS v1.0 model (brain + ventral nerve cord).

Phases 1 (weeks 1-3) simulated all 165,122 traced neurons but zeroed every
synapse that touches the VNC (scripts/ckpt3_sweep_ablate.regions + mask_weights),
so the VNC was silent. `cns_model(scope='cns')` builds the same calibrated model
without that mask; `scope='brain'` rebuilds the Week 3 model exactly, as a
reference for parity checks.

Pipeline (identical to scripts/w3d1_common.presyn_model, minus the VNC mask):
  synapse count x sign x w_syn x 0.581, monoamines removed
  -> rate APL replaces the spiking APL (k = 0.1)
  -> cholinergic LN->PN chemical synapses cut, eLN-PN gap junctions (c0 = 0.2)
  -> optional degree-preserving shuffle (control)
  -> presynaptic modulation of all sensory terminals (K = 0.05), which also
     removes every central -> sensory row (brain and VNC sensory neurons alike)
  -> optional right-JO mirror

Motor pools (`motor_pools`): VNC motor neurons grouped by side x nerve-cord
subclass x functional type, e.g. 'L|fl|Ti flexor MN'. Leg subclasses fl/ml/hl
(front/middle/hind leg), wm wing, hm haltere, nm neck, ad abdomen, xm other.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.lif_numpy import P  # noqa: E402
from sim.lesion import select  # noqa: E402
from sim.gap import build_gap  # noqa: E402

SCALE, K_APL, K_PRESYN = 0.581, 0.1, 0.05
# Week 4 day 3 calibration decisions (progress/week4/day3_results.md). Not the
# cns_model defaults, so Week 4 day 1-2 results stay reproducible; pass
# **W4_CALIB from day 4 on.
GF_GAP_MV = 8.0
# Week 4 day 3b: sign corrections for the two cell types that drive the
# looming-ignited central-complex state (progress/week4/day3b_results.md).
#   GLNO: MaleCNS consensus and prediction 'unclear' (-> +1); FlyWire predicts
#         GABA (3/4) / glutamate (1/4), hemilineage LALv1.
#   LPsP: MaleCNS predicts ACh (0.88); FlyWire known_nt = dopamine + glutamate,
#         ACh-negative (Alekseyenko 2013; Wolff 2024 EASI-FISH). Glutamate -> -1.
SIGN_FIX = {'GLNO': -1, 'LPsP': -1}
# Two small all-to-all excitatory cliques (ACh, consistent in MaleCNS and
# FlyWire) reverberate forever in a LIF model without adaptation: DNg33 L<->R
# (~750 synapses each way) and the IN19B040 quartet. Scale only the
# within-clique synapses (smallest gain that stops them, day 3b sweep).
CLIQUE_GAIN = ({'DNg33', 'IN19B040'}, 0.3)
# Week 4 day 5: a five-type GNG loop of strong reciprocal ACh excitation
# (DNge027, GNG423, DNg21, DNg62, DNge044; 101 W entries) keeps firing after a
# wind word ends (0.54 s, 1.05 s and >= 0.73 s to the end of the run in three
# Day 5 episodes). No annotation or build error: it persists because the model
# has no adaptation (progress/week4/day5_wind_loop_rootcause.md).
# Unlike CLIQUE_GAIN it spans several types, so scale every synapse whose pre-
# AND postsynaptic neuron are both in the set. 0.3: post-wind tail <= 0.09 s
# in 3/3 wind episodes, non-wind ep000/ep019 spike-identical (T2c gain scan,
# progress/week4/day5_burst_diagnosis.md; verification day5_loop_gain.md).
LOOP_GAIN = ({'DNge027', 'GNG423', 'DNg21', 'DNg62', 'DNge044'}, 0.3)
W4_CALIB = dict(gf_gap=GF_GAP_MV, mn_glu=True, sign_fix=SIGN_FIX, clique_gain=CLIQUE_GAIN,
                loop_gain=LOOP_GAIN)

LEG_SEGMENT = {   # functional type -> joint it moves (Azevedo et al. 2024 naming)
    'Pleural remotor/abductor MN': 'thorax-coxa', 'Tergopleural/Pleural promotor MN': 'thorax-coxa',
    'Sternal anterior rotator MN': 'thorax-coxa', 'Sternal posterior rotator MN': 'thorax-coxa',
    'Sternal adductor MN': 'thorax-coxa',
    'Tergotr. MN': 'coxa-trochanter', 'Sternotrochanter MN': 'coxa-trochanter',
    'Tr extensor MN': 'coxa-trochanter', 'Tr flexor MN': 'coxa-trochanter',
    'Acc. tr flexor MN': 'coxa-trochanter',
    'Fe reductor MN': 'trochanter-femur',
    'Ti extensor MN': 'femur-tibia', 'Ti flexor MN': 'femur-tibia', 'Acc. ti flexor MN': 'femur-tibia',
    'ltm MN': 'tarsus (long tendon)', 'ltm1-tibia MN': 'tarsus (long tendon)',
    'ltm2-femur MN': 'tarsus (long tendon)',
    'Ta depressor MN': 'tibia-tarsus', 'Ta levator MN': 'tibia-tarsus',
}
WING_GROUP = {    # wing MN type prefix -> muscle group
    'DLMn': 'power (DLM)', 'DVMn': 'power (DVM)',
    'b1': 'steering b', 'b2': 'steering b', 'b3': 'steering b',
    'i1': 'steering i', 'i2': 'steering i',
    'iii1': 'steering iii', 'iii3': 'steering iii',
    'hg1': 'steering hg', 'hg2': 'steering hg', 'hg3': 'steering hg', 'hg4': 'steering hg',
    'tp1': 'tension tp', 'tp2': 'tension tp', 'tpn': 'tension tp',
    'ps1': 'pleurosternal', 'ps2': 'pleurosternal',
    'TTMn': 'jump (TTM)', 'STTMm': 'jump (TTM)',
}


def regions(meta):
    """Boolean masks (vnc, abdominal) - same rule as Phase 1."""
    sc = meta.superclass.fillna('').astype(str)
    nm = meta.somaNeuromere.fillna('').astype(str)
    vnc = (sc.str.startswith('vnc_')
           | sc.isin(['ascending_neuron', 'sensory_ascending',
                      'sensory_ascending_tbc', 'efferent_ascending'])
           | nm.str.match(r'^(T[123]|A\d+)'))
    abd = nm.str.match(r'^A\d+')
    return vnc.to_numpy(), abd.to_numpy()


def _mask(W, keep_entry):
    from scipy import sparse
    C = W.tocoo()
    k = keep_entry(C.row, C.col)
    return sparse.csc_matrix((C.data[k], (C.row[k], C.col[k])), shape=W.shape)


def _apl(meta, W):
    from scipy import sparse
    is_kc = meta.type.fillna('').str.startswith('KC').to_numpy()
    apl = np.flatnonzero(meta.type.values == 'APL')
    kc_idx = np.flatnonzero(is_kc)
    w_kc = -np.asarray(W[kc_idx][:, apl].sum(axis=1)).ravel()
    keep = np.ones(W.shape[0]); keep[apl] = 0
    D = sparse.diags(keep)
    W2 = (D @ W @ D).tocsc(); W2.eliminate_zeros()
    return W2, dict(kc_idx=kc_idx, w_kc=w_kc, k=K_APL, tau=0.01)


def scale_set_edges(W, types_of, types, g):
    """W with every entry whose row (post) and column (pre) neuron types are both
    in `types` multiplied by g; all other entries unchanged. types_of: type per neuron."""
    from scipy import sparse
    L = np.isin(types_of, list(types))
    missing = set(types) - set(types_of[L])
    if missing:
        raise ValueError(f'loop_gain: no neurons of type {sorted(missing)}')
    C = W.tocoo()
    k = L[C.row] & L[C.col]
    d = C.data.copy(); d[k] *= g
    return sparse.csc_matrix((d, (C.row, C.col)), shape=W.shape)


def cns_model(scope='cns', config='calib', K=K_PRESYN, mirror=True, verbose=True, gf_gap=None,
              mn_glu=False, sign_fix=None, clique_gain=None, loop_gain=None):
    """Return dict(meta, W, gap, presyn, apl, vnc, info).

    scope : 'cns' (brain + VNC, Week 4) or 'brain' (Week 3 model, VNC masked).
    config: 'calib' (real wiring) or 'shuffle' (degree-preserving control).
    gf_gap: None (Weeks 1-4 day 2) or a spikelet size in mV: adds rectifying
            GF -> TTMn / PSI gap junctions (sim/gap.build_gf_gap, Week 4 day 3).
            Added in both configs (like the eLN-PN junctions, not shuffled).
    mn_glu: if True, every chemical output of a VNC motor neuron is
            glutamatergic (-1). Fly motor neurons are glutamatergic; the
            per-body prediction gives their few central synapses 'unclear'
            (+1) or acetylcholine for most MNs (Week 4 day 3 sign audit).
    sign_fix: optional {cell type: +1 / -1}; sets the sign of every chemical
            output of neurons of that type (Week 4 day 3b: SIGN_FIX).
    clique_gain: optional (types, g); synapses between two neurons of the same
            listed type are scaled by g (Week 4 day 3b: CLIQUE_GAIN).
    loop_gain: optional (types, g); synapses whose pre- and postsynaptic
            neurons both belong to the listed types (same or different type)
            are scaled by g (Week 4 day 5: LOOP_GAIN). Like clique_gain it acts
            before the shuffle, so the shuffle control permutes the scaled W.
    """
    from scripts.w2d3_gap_malecns import cut_ln_pn
    from scripts.w2d4_word_trials import shuffle_W, GAP_KW
    from scripts.ckpt5_stageA import ACH_LN, PN
    from sim.presyn import build_presyn, sensory_mask_malecns
    from sim.malecns import load        # needs pyarrow: imported here so CI can import sim.cns

    meta, W1 = load(w_syn=1.0, monoamine='removed')
    vnc, _ = regions(meta)
    W = W1 * (P['w_syn'] * SCALE)
    if mn_glu:
        from scipy import sparse
        is_mn = (meta.superclass.astype(str) == 'vnc_motor').to_numpy()
        W = (W @ sparse.diags((~is_mn).astype(float))
             - abs(W) @ sparse.diags(is_mn.astype(float))).tocsc()   # columns = presynaptic
        W.eliminate_zeros()
    if sign_fix:
        from scipy import sparse
        tt = meta.type.astype(str).to_numpy()
        pos_, neg_, keep_ = np.zeros(len(tt)), np.zeros(len(tt)), np.ones(len(tt))
        for typ, sg in sign_fix.items():
            k = tt == typ
            if not k.any():
                raise ValueError(f'sign_fix: no neurons of type {typ}')
            keep_[k] = 0
            (pos_ if sg > 0 else neg_)[k] = 1
        W = (W @ sparse.diags(keep_) + abs(W) @ sparse.diags(pos_)
             - abs(W) @ sparse.diags(neg_)).tocsc()
        W.eliminate_zeros()
    if clique_gain:
        types, gq = clique_gain
        tt = meta.type.astype(str).to_numpy()
        C = W.tocoo()
        k = (tt[C.row] == tt[C.col]) & np.isin(tt[C.col], list(types))
        d = C.data.copy(); d[k] *= gq
        from scipy import sparse
        W = sparse.csc_matrix((d, (C.row, C.col)), shape=W.shape)
    if loop_gain:
        W = scale_set_edges(W, meta.type.astype(str).to_numpy(), *loop_gain)
    if scope == 'brain':
        W = _mask(W, lambda r, c: ~vnc[r] & ~vnc[c])
    elif scope != 'cns':
        raise ValueError(scope)
    W, apl = _apl(meta, W)
    ln, pn = select(meta, ACH_LN), select(meta, PN)
    W = cut_ln_pn(W, ln, pn)
    gap, ginfo = build_gap(W1, ln, pn, v_spike=0.002, **GAP_KW)
    if gf_gap is not None and scope == 'cns':      # TTMn / PSI are VNC neurons
        from sim.gap import build_gf_gap, merge_gap
        gg, gginfo = build_gf_gap(meta, gf_gap, v_spike=0.002)
        gap = merge_gap(gap, gg)
        ginfo = dict(ginfo, gf=gginfo)
    if config == 'shuffle':
        W = shuffle_W(W)
    elif config != 'calib':
        raise ValueError(config)
    W, ps, pinfo = build_presyn(W, sensory_mask_malecns(meta), K)
    info = dict(scope=scope, config=config, nnz=int(W.nnz), presyn=pinfo, gap=ginfo)
    if mirror:
        from sim.jo_mirror import mirror_jo
        W, minfo = mirror_jo(W, meta)
        info['mirror'] = {k: v for k, v in minfo.items() if k != 'templates'}
    if verbose:
        print(f'[cns_model] scope={scope} config={config} n={W.shape[0]:,} nnz={W.nnz:,}', flush=True)
    return dict(meta=meta, W=W.tocsc(), gap=gap, presyn=ps, apl=apl, vnc=vnc, info=info)


def motor_pools(meta, min_size=1):
    """dict pool_name -> int array of neuron indices, for VNC motor neurons.

    Pool name: '<side>|<subclass>|<type>' (fine) - the regression targets.
    Also returns a DataFrame with one row per pool: side, subclass, type,
    joint (legs) or muscle group (wing), n.
    """
    mn = meta[meta.superclass.astype(str) == 'vnc_motor']
    rows, pools = [], {}
    for (side, sub, typ), g in mn.groupby([mn.somaSide.fillna('?').astype(str),
                                           mn.subclass.fillna('?').astype(str),
                                           mn.type.fillna('?').astype(str)]):
        if len(g) < min_size:
            continue
        name = f'{side}|{sub}|{typ}'
        pools[name] = g.index.to_numpy()
        if sub in ('fl', 'ml', 'hl'):
            grp = LEG_SEGMENT.get(typ, 'other leg')
        elif sub == 'wm':
            grp = WING_GROUP.get(typ.split(' ')[0], 'other wing')
        else:
            grp = {'hm': 'haltere', 'nm': 'neck', 'ad': 'abdomen', 'xm': 'other'}.get(sub, sub)
        rows.append(dict(pool=name, side=side, subclass=sub, type=typ, group=grp, n=len(g)))
    return pools, pd.DataFrame(rows)


def brain_mask(meta, vnc=None):
    """Neurons allowed as regression inputs: brain, non-sensory, non-motor."""
    if vnc is None:
        vnc, _ = regions(meta)
    sc = meta.superclass.astype(str)
    return ~vnc & ~sc.str.contains('sensory').to_numpy() & ~sc.str.contains('motor').to_numpy()
