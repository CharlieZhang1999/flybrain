#!/usr/bin/env python
"""W5 T7 Phase B: the hidden-state dataset (central plan, Week 5 row).

Episodes k = 0..119, all T = 4 s:
  words     stimuli.mixtures.sample_episode(SPEC_SEED0 + k, T=4.0), class_v2 defaults
            (2-4 words, distinct modalities, 30-150 Hz, looming in ~1/6, bg_hz U(1, 3)).
  states    two segments, [0, 2) and [2, 4) s, drawn from a separate generator
            np.random.default_rng(STATE_SEED0 + k) (independent of the words): each
            segment's state is uniform over STATES_USED = baseline, hunger, thirst,
            arousal; its level is U(0.7, 1.3) x LEVEL_HZ[state] (baseline: 0 Hz).
            LEVEL_HZ are the Phase A choices (results/w5/hidden_state/pilot_dose.md):
            hunger ISN 20 Hz, thirst BiT 100 Hz, arousal pC1/P1 (male-specific) 20 Hz.
            ER5 (sleep) is dropped (no downstream effect at any level in Phase A).
  drive     stimuli.states.state_blocks (tonic Poisson to the state set's neurons),
            appended to stimuli.mixtures.to_blocks(spec). Same Poisson seed (spec['seed'])
            and same spec on both configs.
Configs: 'calib' (real wiring) and 'shuffle' (degree-preserving control), sim.cns.W4_CALIB,
CPU float32 rng='torch'. All calib episodes run first. The shuffle half uses sim/cns.py
from PR #14 (fixed shuffle / JO-mirror order), loaded from a separate git worktree given by
--sim-root (not committed to this branch); the sim/ revision used is stored in every
activity file (sim_rev) and in index.csv.

Label hiding (brief tasks 4 and 6):
  activity/ files hold the spikes of every neuron EXCEPT the decoder mask (all neurons of
  the four Phase A state sets, ISN + BiT + pC1/P1 + ER5, 153 neurons, listed in
  state_sets.json 'decoder_mask'), and the words spec only: no state, no level, no segment.
  labels/ holds the ground truth (state id + level per 10 ms bin, the segments) and the
  masked neurons' spikes (driven/), which carry the drive and so count as labels.

Resumable: an episode whose activity npz exists and is valid (readable, same words spec,
spike count consistent, cfg) and whose driven npz exists is skipped. Files are written to a
temporary name and renamed. Run one simulation process at a time.
Runaway stop: if more than 1 % of brain neurons (sim.cns.brain_mask, mask excluded) fire
> 200 Hz in either segment of an episode, the run stops (exit 3) and logs it.

Usage:
  python scripts/w5_hidden_state.py run <k0> <k1> <cfg> [--sim-root DIR]
  python scripts/w5_hidden_state.py nnz [--sim-root DIR]   # build calib (this tree) and
                                     # shuffle (DIR) and compare W nnz (must be within 1 %)
  python scripts/w5_hidden_state.py labels                  # (re)write labels/ for all k
  python scripts/w5_hidden_state.py index                   # rebuild index.csv files
Output (results/w5/hidden_state/dataset/):
  activity/ep{k:03d}_{cfg}.npz  t (step, dt 0.1 ms, int32), i (neuron index into
                                malecns_traced_meta, int32; mask neurons removed), spec (JSON,
                                words only), cfg, sim_rev, n_spikes, frac200 (per segment
                                runaway check), wall_s
  specs.json                    the 120 words specs (no state)
  index.csv                     one row per activity file (no state columns)
  labels/segments.json          per episode: the two segments (state, level_hz, t_on, t_off)
  labels/ep{k:03d}.npz          state_id (int16), level_hz (float32) per 10 ms bin (400),
                                segments (JSON), bin_s
  labels/index.csv              per episode: state and level of each segment
  labels/driven/ep{k:03d}_{cfg}.npz  spikes of the decoder-mask neurons (t, i)
  run_log.txt                   per-episode wall time (appended)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stimuli.mixtures import sample_episode  # noqa: E402
from stimuli import states as st  # noqa: E402

HS = ROOT / 'results' / 'w5' / 'hidden_state'
OUT = HS / 'dataset'
ACT, LAB = OUT / 'activity', OUT / 'labels'
SETS_JSON = HS / 'state_sets.json'
CFGS = ('calib', 'shuffle')
T, T_SEG = 4.0, 2.0
N_EP = 120
SPEC_SEED0 = 72000
STATE_SEED0 = 73000
LEVEL_HZ = {'baseline': 0.0, 'hunger': 20.0, 'thirst': 100.0, 'arousal': 20.0}
STATES_USED = ('baseline', 'hunger', 'thirst', 'arousal')
LEVEL_SCALE = (0.7, 1.3)
MASK_STATES = ('hunger', 'thirst', 'arousal', 'sleep')
LABEL_BIN_S = 0.01
RUNAWAY_HZ, RUNAWAY_FRAC = 200.0, 0.01


def spec_for(k):
    """Words spec of episode k (no state): depends only on k."""
    if not 0 <= k < N_EP:
        raise ValueError(f'episode {k} out of range 0..{N_EP - 1}')
    return sample_episode(SPEC_SEED0 + k, T=T)


def segments_for(k):
    """State segments of episode k: depends only on k (own generator, not the words')."""
    rng = np.random.default_rng(STATE_SEED0 + k)
    segs = []
    for j in range(int(round(T / T_SEG))):
        s = STATES_USED[int(rng.integers(len(STATES_USED)))]
        lv = float(rng.uniform(*LEVEL_SCALE)) * LEVEL_HZ[s]
        segs.append((j * T_SEG, (j + 1) * T_SEG, s, lv))
    return st.check_segments(segs, T)


def act_path(k, cfg):
    return ACT / f'ep{k:03d}_{cfg}.npz'


def drv_path(k, cfg):
    return LAB / 'driven' / f'ep{k:03d}_{cfg}.npz'


def log(msg):
    print(msg, flush=True)
    with open(OUT / 'run_log.txt', 'a') as f:
        f.write(msg + '\n')


def load_sets():
    d = json.loads(SETS_JSON.read_text())
    return {s: [int(b) for b in d['body_ids'][s]] for s in MASK_STATES}, d


def mask_idx(meta):
    """Sorted neuron indices (rows of malecns_traced_meta) of the decoder mask."""
    sets, _ = load_sets()
    pos = {int(b): k for k, b in enumerate(meta.bodyId)}
    return np.array(sorted({pos[b] for s in MASK_STATES for b in sets[s]}), np.int64)


def write_mask(meta):
    """Add the decoder mask to state_sets.json (body IDs unchanged; re-read first)."""
    sets, d = load_sets()
    idx = mask_idx(meta)
    want = dict(states=list(MASK_STATES), n=int(len(idx)), neuron_idx=idx.tolist(),
                body_ids=meta.bodyId.to_numpy()[idx].astype(np.int64).tolist(),
                index_into='data/malecns/malecns_traced_meta.parquet row order (= model neuron index)',
                rule='all neurons of the four Phase A state sets are removed from every decoder input '
                     'and from the activity files (their spikes are in labels/driven/)')
    if d.get('decoder_mask') == want:
        return idx
    if 'decoder_mask' in d:
        raise RuntimeError('state_sets.json has a different decoder_mask')
    d['decoder_mask'] = want
    SETS_JSON.write_text(json.dumps(d, indent=1))
    time.sleep(10)
    if json.loads(SETS_JSON.read_text()).get('decoder_mask') != want:
        raise RuntimeError('state_sets.json re-read does not match what was written')
    return idx


def load_valid(k, cfg):
    f = act_path(k, cfg)
    if not f.exists() or not drv_path(k, cfg).exists():
        return None
    try:
        with np.load(f) as d:
            n = int(d['n_spikes'])
            sp = json.loads(str(d['spec']))
            if len(d['t']) != n or len(d['i']) != n or str(d['cfg']) != cfg or sp != spec_for(k):
                return None
            r = dict(spec=sp, n_spikes=n, wall_s=float(d['wall_s']), sim_rev=str(d['sim_rev']),
                     frac200=d['frac200'].tolist())
        with np.load(drv_path(k, cfg)) as d:
            r['n_driven_spikes'] = len(d['t'])
        return r
    except Exception:
        return None


def write_specs():
    sp = [spec_for(k) for k in range(N_EP)]
    f = OUT / 'specs.json'
    if f.exists() and json.loads(f.read_text()) == sp:
        return
    f.write_text(json.dumps(sp, indent=1))


def write_labels():
    (LAB / 'driven').mkdir(parents=True, exist_ok=True)
    allseg, rows = {}, []
    for k in range(N_EP):
        segs = segments_for(k)
        sid, lev = st.label_bins(segs, T, LABEL_BIN_S)
        f = LAB / f'ep{k:03d}.npz'
        tmp = LAB / f'.ep{k:03d}.{os.getpid()}.tmp.npz'
        with open(tmp, 'wb') as fh:
            np.savez_compressed(fh, state_id=sid, level_hz=lev, segments=json.dumps(segs),
                                bin_s=np.float64(LABEL_BIN_S), states=json.dumps(list(st.STATES)))
        os.replace(tmp, f)
        allseg[f'{k:03d}'] = segs
        rows.append(dict(episode=k, state_seed=STATE_SEED0 + k,
                         **{f'seg{j}_{c}': s[c] for j, s in enumerate(segs) for c in ('state', 'level_hz')},
                         file=f.name))
    (LAB / 'segments.json').write_text(json.dumps(dict(
        states=list(st.STATES), level_hz=LEVEL_HZ, level_scale=LEVEL_SCALE, state_seed0=STATE_SEED0,
        episodes=allseg), indent=1))
    pd.DataFrame(rows).to_csv(LAB / 'index.csv', index=False)


def write_index():
    rows = []
    for k in range(N_EP):
        for cfg in CFGS:
            r = load_valid(k, cfg)
            if r is None:
                continue
            sp = r['spec']
            rows.append(dict(episode=k, cfg=cfg, sim_rev=r['sim_rev'], seed=sp['seed'],
                             spec_seed=sp['spec_seed'],
                             words='+'.join(w['word'] for w in sp['words']), n_words=len(sp['words']),
                             bg_hz=sp['bg_hz'], has_looming=any(w['word'] == 'looming' for w in sp['words']),
                             T=sp['T'], n_spikes=r['n_spikes'], n_driven_spikes=r['n_driven_spikes'],
                             frac200_max=max(r['frac200']), wall_s=round(r['wall_s'], 2),
                             wall_per_bio_s=round(r['wall_s'] / sp['T'], 2), file=act_path(k, cfg).name))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'index.csv', index=False)
    return df


def git_rev(root):
    try:
        sha = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--', 'sim'],
                                        text=True).strip()
        return sha + ('+dirty' if dirty else '')
    except Exception:
        return 'unknown'


def use_sim_root(sim_root):
    """Import sim/ (and the scripts/ it pulls in) from sim_root instead of this tree.
    stimuli is already imported from this tree. Returns the sim revision string."""
    if sim_root is None:
        return f'{git_rev(ROOT)} (this branch)'
    sim_root = Path(sim_root).resolve()
    for name in [n for n in sys.modules if n == 'sim' or n.startswith(('sim.', 'scripts'))]:
        del sys.modules[name]
    sys.path.insert(0, str(sim_root))
    import sim
    if Path(sim.__file__).resolve().parents[1] != sim_root:
        raise RuntimeError(f'sim imported from {sim.__file__}, not {sim_root}')
    return f'{git_rev(sim_root)} ({sim_root.name})'


def build(cfg):
    import torch
    from sim.cns import cns_model, W4_CALIB
    import sim.cns
    t0 = time.time()
    m = cns_model('cns', cfg, **W4_CALIB)
    log(f'[{cfg}] model built in {time.time() - t0:.1f} s from {sim.cns.__file__}, W nnz {m["W"].nnz:,}, '
        f'shuffle_order {m["info"].get("shuffle_order", "n/a")}, torch threads {torch.get_num_threads()}')
    return m


def run(k0, k1, cfg, sim_root=None):
    for d in (ACT, LAB / 'driven'):
        d.mkdir(parents=True, exist_ok=True)
    write_specs()
    write_labels()
    todo = [k for k in range(k0, k1) if load_valid(k, cfg) is None]
    log(f'[chunk] episodes {k0}..{k1 - 1} {cfg}: {len(todo)} to run ({time.strftime("%Y-%m-%d %H:%M:%S")})')
    if not todo:
        write_index()
        return
    rev = use_sim_root(sim_root)
    log(f'[{cfg}] sim rev {rev}')
    import torch
    from sim.cns import brain_mask
    from sim.lif_numpy import P
    from sim import lif_torch
    from stimuli.mixtures import to_blocks
    m = build(cfg)
    meta = m['meta']
    pos = {int(b): i for i, b in enumerate(meta.bodyId)}
    n = len(meta)
    midx = write_mask(meta)
    is_mask = np.zeros(n, bool); is_mask[midx] = True
    bm = brain_mask(meta) & ~is_mask
    sets, _ = load_sets()
    nseg = int(round(T / T_SEG))
    for k in todo:
        spec, segs = spec_for(k), segments_for(k)
        blocks = to_blocks(spec, meta, pos) + st.state_blocks(segs, sets, pos)
        p = dict(P); p['t_run'] = T
        t0 = time.time()
        r = lif_torch.run_trial(m['W'], [], params=p, seed=spec['seed'], apl=m['apl'], gap=m['gap'],
                                presyn=m['presyn'], blocks=blocks, record=True, device='cpu',
                                rng='torch', dtype=torch.float32)
        wall = time.time() - t0
        ts, si = r['t'].astype(np.int32), r['i'].astype(np.int32)
        seg = np.minimum(ts // int(round(T_SEG / 1e-4)), nseg - 1)
        frac = []
        for j in range(nseg):
            rate = np.bincount(si[seg == j], minlength=n)[bm] / T_SEG
            frac.append(float((rate > RUNAWAY_HZ).mean()))
        km = is_mask[si]
        tmp = LAB / 'driven' / f'.ep{k:03d}_{cfg}.{os.getpid()}.tmp.npz'
        with open(tmp, 'wb') as fh:
            np.savez_compressed(fh, t=ts[km], i=si[km], cfg=cfg)
        os.replace(tmp, drv_path(k, cfg))
        tmp = ACT / f'.ep{k:03d}_{cfg}.{os.getpid()}.tmp.npz'
        with open(tmp, 'wb') as fh:
            np.savez_compressed(fh, t=ts[~km], i=si[~km], spec=json.dumps(spec), cfg=cfg, sim_rev=rev,
                                n_spikes=np.int64((~km).sum()), frac200=np.array(frac),
                                wall_s=np.float64(wall))
        os.replace(tmp, act_path(k, cfg))
        log(f'[{cfg}] ep{k:03d}: {len(ts):,} spikes ({km.sum():,} masked), frac>200Hz '
            f'{" / ".join(f"{100 * f:.3f}%" for f in frac)}, {wall:.1f} s wall ({wall / T:.1f} s per bio-s)')
        if max(frac) > RUNAWAY_FRAC:
            log(f'[STOP] ep{k:03d} {cfg}: runaway (> {100 * RUNAWAY_FRAC:g} % of brain neurons > {RUNAWAY_HZ:g} Hz)')
            write_index()
            sys.exit(3)
    df = write_index()
    log(f'[chunk] done ({time.strftime("%Y-%m-%d %H:%M:%S")}); index.csv has {len(df)} rows')


def nnz(sim_root):
    """W nnz of calib (this tree) vs shuffle (sim_root). Separate processes keep the imports apart."""
    out = {}
    for cfg, root in (('calib', None), ('shuffle', sim_root)):
        code = ('import sys; sys.argv=["x"]; sys.path.insert(0, %r); import scripts.w5_hidden_state as h; '
                'h.use_sim_root(%r); from sim.cns import cns_model, W4_CALIB; '
                'm = cns_model("cns", %r, verbose=False, **W4_CALIB); print("NNZ", m["W"].nnz, '
                'm["info"].get("shuffle_order", "n/a"))') % (str(ROOT), root, cfg)
        res = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
        line = [x for x in res.stdout.splitlines() if x.startswith('NNZ')][-1].split()
        out[cfg] = (int(line[1]), line[2])
    rel = out['shuffle'][0] / out['calib'][0] - 1
    log(f'[nnz] calib {out["calib"][0]:,}, shuffle {out["shuffle"][0]:,} (order {out["shuffle"][1]}, '
        f'sim root {sim_root}): {100 * rel:+.3f} %')
    return abs(rel) <= 0.01


def main(*args):
    OUT.mkdir(parents=True, exist_ok=True)
    args = list(args)
    sim_root = None
    if '--sim-root' in args:
        j = args.index('--sim-root')
        sim_root = args[j + 1]
        del args[j:j + 2]
    if not args:
        sys.exit(__doc__)
    if args[0] == 'labels':
        write_specs(); write_labels(); return
    if args[0] == 'index':
        write_specs(); print(f'index.csv: {len(write_index())} rows'); return
    if args[0] == 'nnz':
        sys.exit(0 if nnz(sim_root) else 4)
    if args[0] == 'run':
        k0, k1, cfg = int(args[1]), int(args[2]), args[3]
        if not 0 <= k0 < k1 <= N_EP or cfg not in CFGS:
            sys.exit(f'need 0 <= k0 < k1 <= {N_EP} and cfg in {CFGS}')
        run(k0, k1, cfg, sim_root)
        return
    sys.exit(__doc__)


if __name__ == '__main__':
    main(*sys.argv[1:])
