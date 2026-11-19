#!/usr/bin/env python
"""W4D5 T2c: simulations for the burst / wind-persistence diagnosis. Diagnosis only:
nothing under sim/ or stimuli/ is changed; ablations act on copies of W / gap.

Model: cns_model('cns', cfg, **W4_CALIB); CPU float32, rng='torch', fixed seeds.
Run ONE mode at a time (one simulation process at a time). Existing outputs are skipped.

Modes
  al_rate     class_v2 background only, T = 2.0 s, rates 1.5/2.0/2.5/3.0/3.5 Hz x seeds 0-2.
  al_head     T = 2.0 s, 2.5 Hz, seeds 0-2: olfactory ORNs only (class 'olfactory', 2,639);
              class_v2 minus ORNs; all head sensory (sensory & not VNC).
  al_ablate   bg-only Day 5 specs ep205 and ep207 (T = 2.5 s, same Poisson seed) with:
              AL gap junctions (eLN-PN) removed (GF junctions kept), AL gap x0.5,
              apl=None, presyn=None.
  wind        wind test matrix (C2). Specs: A = Day 5 ep009 exactly (wind 0.857-1.771 +
              humid 1.639-2.0, bg 2.76 Hz, T 2.5); B = wind alone 110 Hz 0.3-1.0 s, bg 2.5 Hz,
              T 2.0, Poisson seeds 1 and 2. Conditions: bg_on (as the dataset), bg_during
              (bg switched off at wind offset via state/resume), no_bg, bg_after (no bg until
              wind offset, then bg on), shuffle_bg_on. Word blocks are never changed.
  gain        loop-gain scan (revision, candidate fix B, diagnostic only): edges whose pre AND
              post are both in LOOP5 (DNge027, GNG423, DNg21, DNg62, DNge044) scaled by g on a copy
              of W; g in 0.5/0.3/0.1 x Day 5 wind episodes ep009, ep002, ep016 (calib, exact
              specs incl. class_v2 bg, T 2.5 s); plus non-wind ep000 and ep019 at g_check (argument).
              Usage: python scripts/w4d5_burst_runs.py gain [g_check]
  lesion      spec A, bg on, run to 2.0 s (return_state), then resume 2.0-2.5 s with the
              intact W (must equal saved ep009) and with edge cuts on a copy of W
              (see LESIONS), plus bg off.
Usage:  python scripts/w4d5_burst_runs.py <mode>
Output: results/w4/regression/burst_diag/runs/<mode>__<name>.npz (t: step, dt 0.1 ms;
        i: neuron; spec: JSON), runs_log.txt (wall time per run).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'results' / 'w4' / 'regression' / 'burst_diag'
RUNS = OUT / 'runs'
DAY5 = ROOT / 'results' / 'w4' / 'regression' / 'day5'
GNG_CORE = ['DNge027', 'GNG423', 'DNg21', 'DNg62', 'DNge044', 'DNge022', 'DNge078', 'DNge019',
            'AN01A014', 'SAD093', 'DNge011', 'DNge039', 'DNge012', 'DNg12_a', 'DNg12_b', 'DNg12_c',
            'DNg12_d', 'DNg12_e', 'DNg12_f', 'DNg12_g', 'DNg12_h']


LOOP5 = ['DNge027', 'GNG423', 'DNg21', 'DNg62', 'DNge044']


def log(msg):
    print(msg, flush=True)
    with open(OUT / 'runs_log.txt', 'a') as f:
        f.write(msg + '\n')


class Model:
    def __init__(self, cfg='calib'):
        from sim.cns import cns_model, W4_CALIB
        t0 = time.time()
        self.m = cns_model('cns', cfg, verbose=False, **W4_CALIB)
        self.cfg = cfg
        self.meta, self.vnc = self.m['meta'], self.m['vnc']
        self.pos = {b: i for i, b in enumerate(self.meta.bodyId)}
        log(f'[model] {cfg} built in {time.time() - t0:.1f} s')

    def run(self, T, blocks, seed, W=None, gap='keep', apl='keep', presyn='keep', state=None,
            return_state=False):
        import torch
        from sim.lif_numpy import P
        from sim import lif_torch
        p = dict(P); p['t_run'] = T
        m = self.m
        return lif_torch.run_trial(m['W'] if W is None else W, [], params=p, seed=seed,
                                   apl=m['apl'] if apl == 'keep' else apl,
                                   gap=m['gap'] if gap == 'keep' else gap,
                                   presyn=m['presyn'] if presyn == 'keep' else presyn,
                                   blocks=blocks, record=True, device='cpu', rng='torch',
                                   dtype=torch.float32, state=state, return_state=return_state)


def out_path(mode, name):
    return RUNS / f'{mode}__{name}.npz'


def save(mode, name, t, i, spec, wall):
    RUNS.mkdir(parents=True, exist_ok=True)
    tmp = RUNS / f'.{mode}__{name}.tmp.npz'          # write then rename (no partial files)
    np.savez_compressed(tmp, t=np.asarray(t, np.int32), i=np.asarray(i, np.int32), spec=json.dumps(spec))
    tmp.replace(out_path(mode, name))
    log(f'[{mode}] {name}: {len(t):,} spikes, {wall:.1f} s wall')


def done(mode, name):
    return out_path(mode, name).exists()


def clip_blocks(blocks, a, b):
    """Blocks restricted to [a, b) (absolute times); empty ones dropped."""
    out = []
    for b0, b1, ix, r in blocks:
        lo, hi = max(b0, a), min(b1, b)
        if hi > lo:
            out.append((lo, hi, ix, r))
    return out


# ---------------------------------------------------------------- AL bursts
def al_rate():
    from stimuli.mixtures import background_episode, bg_blocks
    M = Model()
    for hz in (1.5, 2.0, 2.5, 3.0, 3.5):
        for s in (0, 1, 2):
            name = f'v2_{hz:.1f}Hz_s{s}'
            if done('al_rate', name):
                continue
            spec = background_episode(hz, s, T=2.0)
            t0 = time.time(); r = M.run(2.0, bg_blocks(spec, M.meta), s)
            save('al_rate', name, r['t'], r['i'], spec, time.time() - t0)


def al_head():
    from stimuli.mixtures import background_episode, bg_blocks, bg_class_masks
    M = Model()
    mk = bg_class_masks(M.meta)
    olf = (M.meta['class'].astype(str).values == 'olfactory') & mk['sensory']
    head = mk['sensory'] & ~M.vnc
    hz, T = 2.5, 2.0
    for s in (0, 1, 2):
        v2 = bg_blocks(background_episode(hz, s, T=T), M.meta)
        variants = {
            'olf_only': [(0.0, T, np.flatnonzero(olf), hz)],
            'v2_minus_olf': [(b0, b1, ix[~olf[ix]], r) for b0, b1, ix, r in v2],
            'head_only': [(0.0, T, np.flatnonzero(head), hz)],
        }
        for v, bl in variants.items():
            name = f'{v}_{hz:.1f}Hz_s{s}'
            if done('al_head', name):
                continue
            spec = dict(T=T, seed=s, bg_hz=hz, variant=v, n_driven=int(sum(len(b[2]) for b in bl)))
            t0 = time.time(); r = M.run(T, bl, s)
            save('al_head', name, r['t'], r['i'], spec, time.time() - t0)


def _gap_scaled(gap, n_al, g):
    """Copy of the merged gap dict with the AL block (first n_al idx entries) scaled by g."""
    from scipy import sparse
    C = gap['C'].tocoo()
    k = (C.row < n_al) & (C.col < n_al)
    d = C.data.copy(); d[k] *= g
    C2 = sparse.csr_matrix((d, (C.row, C.col)), shape=C.shape); C2.eliminate_zeros()
    return dict(gap, C=C2)


def al_ablate():
    from stimuli.mixtures import to_blocks
    M = Model()
    n_al = len(M.m['gap']['idx']) - 6            # GF block (TTMn, PSI, DNp01 x2) is last 6
    specs = json.loads((DAY5 / 'specs.json').read_text())
    var = {'gapAL0': dict(gap=_gap_scaled(M.m['gap'], n_al, 0.0)),
           'gapALhalf': dict(gap=_gap_scaled(M.m['gap'], n_al, 0.5)),
           'aplNone': dict(apl=None),
           'presynNone': dict(presyn=None)}
    for k in (207, 205):
        spec = specs[k]
        for v, kw in var.items():
            name = f'ep{k}_{v}'
            if done('al_ablate', name):
                continue
            t0 = time.time(); r = M.run(spec['T'], to_blocks(spec, M.meta, M.pos), spec['seed'], **kw)
            save('al_ablate', name, r['t'], r['i'], dict(spec, ablation=v), time.time() - t0)


# ---------------------------------------------------------------- wind
def wind_specs():
    from stimuli.mixtures import background_episode
    specs = json.loads((DAY5 / 'specs.json').read_text())
    A = specs[9]
    out = {'A': A}
    for s in (1, 2):
        B = background_episode(2.5, s, T=2.0)
        B['words'] = [dict(word='wind', modality='mechanosense', rate_hz=110.0, onset=0.3, offset=1.0)]
        out[f'B_s{s}'] = B
    return out


def wind_offset(spec):
    return [w['offset'] for w in spec['words'] if w['word'] == 'wind'][0]


def wind_conditions(M, key, spec, conds):
    from stimuli.mixtures import bg_blocks, to_blocks
    T, seed, toff = spec['T'], spec['seed'], wind_offset(spec)
    bg = bg_blocks(spec, M.meta)
    words = to_blocks(spec, M.meta, M.pos)[len(bg):]
    for c in conds:
        name = f'{key}_{c}'
        if done('wind', name):
            continue
        t0 = time.time()
        if c in ('bg_on', 'shuffle_bg_on'):
            r = M.run(T, bg + words, seed); t, i = r['t'], r['i']
        elif c == 'no_bg':
            r = M.run(T, words, seed); t, i = r['t'], r['i']
        elif c in ('bg_during', 'bg_after'):
            first = (bg if c == 'bg_during' else []) + words
            second = (clip_blocks(bg, toff, T) if c == 'bg_after' else []) + clip_blocks(words, toff, T)
            a = M.run(toff, clip_blocks(first, 0, toff), seed, return_state=True)
            b = M.run(T, second, seed, state=a['state'])
            t, i = np.concatenate([a['t'], b['t']]), np.concatenate([a['i'], b['i']])
        save('wind', name, t, i, dict(spec, condition=c, cfg=M.cfg, switch_s=toff), time.time() - t0)


def wind():
    S = wind_specs()
    M = Model('calib')
    for key, spec in S.items():
        wind_conditions(M, key, spec, ['bg_on', 'bg_during', 'no_bg', 'bg_after'])
    del M
    M = Model('shuffle')
    for key, spec in S.items():
        wind_conditions(M, key, spec, ['shuffle_bg_on'])


def lesion():
    from sim.cns import _mask
    from stimuli.mixtures import to_blocks, bg_blocks
    spec = wind_specs()['A']
    M = Model('calib')
    meta, vnc = M.meta, M.vnc
    ty = meta.type.fillna('?').astype(str).values
    sc = meta.superclass.fillna('?').astype(str).values
    ol = sc == 'ol_intrinsic'
    core = np.isin(ty, GNG_CORE)
    d27 = ty == 'DNge027'
    LESIONS = {
        'intact': None,
        'bg_off': None,
        'cut_DNge027_out': lambda r, c: ~d27[c],
        'cut_GNGcore_internal': lambda r, c: ~(core[r] & core[c]),
        'cut_VNC_to_brain': lambda r, c: ~(vnc[c] & ~vnc[r]),
        'cut_brain_to_VNC': lambda r, c: ~(~vnc[c] & vnc[r]),
        'cut_central_to_OLintrinsic': lambda r, c: ~(ol[r] & ~ol[c]),
        'cut_DNge027_LR_pair': lambda r, c: ~(d27[r] & d27[c]),
        'DNge027_pair_x0.3': 0.3,          # scale (not cut) the L<->R synapses, as CLIQUE_GAIN
        'DNge027_pair_x0.1': 0.1,
    }
    T0, T1 = 2.0, spec['T']
    blocks = to_blocks(spec, meta, M.pos)
    t0 = time.time()
    a = M.run(T0, clip_blocks(blocks, 0, T0), spec['seed'], return_state=True)
    log(f'[lesion] ignition 0-{T0} s: {len(a["t"]):,} spikes, {time.time() - t0:.1f} s wall')
    for name, f in LESIONS.items():
        if done('lesion', name):
            continue
        if isinstance(f, float):
            from scipy import sparse
            Cq = M.m['W'].tocoo(); dq = Cq.data.copy()
            dq[d27[Cq.row] & d27[Cq.col]] *= f
            W = sparse.csc_matrix((dq, (Cq.row, Cq.col)), shape=Cq.shape)
        else:
            W = None if f is None else _mask(M.m['W'], f)
        bl = [] if name == 'bg_off' else clip_blocks(blocks, T0, T1)
        t0 = time.time()
        b = M.run(T1, bl, spec['seed'], W=W, state=a['state'])
        t, i = np.concatenate([a['t'], b['t']]), np.concatenate([a['i'], b['i']])
        nn = '' if W is None else f', W nnz {W.nnz:,} (intact {M.m["W"].nnz:,})'
        save('lesion', name, t, i, dict(spec, lesion=name, resume_s=T0), time.time() - t0)
        if nn:
            log(f'[lesion] {name}{nn}')


def loop_gain_W(M, g):
    from scipy import sparse
    ty = M.meta.type.fillna('?').astype(str).values
    L = np.isin(ty, LOOP5)
    C = M.m['W'].tocoo(); d = C.data.copy()
    k = L[C.row] & L[C.col]
    d[k] *= g
    return sparse.csc_matrix((d, (C.row, C.col)), shape=C.shape), int(k.sum())


def gain():
    from stimuli.mixtures import to_blocks
    g_check = float(sys.argv[2]) if len(sys.argv) > 2 else None
    specs = json.loads((DAY5 / 'specs.json').read_text())
    M = Model('calib')
    jobs = [(g, k) for g in (0.5, 0.3, 0.1) for k in (9, 2, 16)]
    if g_check is not None:
        jobs += [(g_check, 0), (g_check, 19)]
    for g, k in jobs:
        name = f'g{g}_ep{k:03d}'
        if done('gain', name):
            continue
        W, ne = loop_gain_W(M, g)
        spec = specs[k]
        t0 = time.time()
        r = M.run(spec['T'], to_blocks(spec, M.meta, M.pos), spec['seed'], W=W)
        save('gain', name, r['t'], r['i'], dict(spec, loop_gain=g, n_entries_scaled=ne), time.time() - t0)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    modes = dict(gain=gain, al_rate=al_rate, al_head=al_head, al_ablate=al_ablate, wind=wind, lesion=lesion)
    if len(sys.argv) < 2 or sys.argv[1] not in modes:
        sys.exit(__doc__)
    log(f'[start] {sys.argv[1]} {time.strftime("%Y-%m-%d %H:%M:%S")}')
    modes[sys.argv[1]]()
    log(f'[end] {sys.argv[1]} {time.strftime("%Y-%m-%d %H:%M:%S")}')
