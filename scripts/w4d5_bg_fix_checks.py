#!/usr/bin/env python
"""W4D5 (T2b): unit checks for the class-specific background (stimuli/mixtures.py). No simulation.

Checks, on the real MaleCNS meta (sim.malecns.load):
  1. class masks: counts of sensory / bristle / abdominal / O2 (abdominal & type SAxx01|SNxx16)
     / rest-abdominal; SAxx01/SNxx16 neurons outside the abdominal mask (reported); bristle-like
     subclasses and head bristle (BM*) types that the bristle mask does NOT include (reported only);
  2. block structure of a class_v1 spec: no bristle in any background block; abdominal
     in the bg_hz block for 'same', absent for 0, in its own block for a rate; every other
     sensory neuron in the bg_hz block exactly once;
  2b. block structure of the default class_v2 spec at bg 1 and 3 Hz: bristle 0, O2 0,
     rest-abdominal exactly 1 Hz, every other sensory neuron at bg_hz exactly once, no
     non-sensory neuron driven;
  2c. class_v1 specs saved with the verification runs (results/w4/bg_fix/runs/*.npz) give
     to_blocks identical to the class_v1 code (frozen copy, git 9f5ef11, argv[2]);
  3. looming share over spec seeds 0-4999 (new sampler; target 1/6 +- 0.02) and the
     legacy sampler (Day 4: 3/7); bg_hz range;
  4. T1 pilot: the saved results/w4/regression/pilot/specs.json are regenerated
     exactly by scripts/w4d4_pilot.specs() (legacy=True), and to_blocks() of every saved
     spec equals the Day 4 to_blocks (frozen copy of the pre-change stimuli/mixtures.py,
     git e3bbd01, passed as argv[1]).
Usage:  python scripts/w4d5_bg_fix_checks.py <Day 4 mixtures.py (e3bbd01)> <class_v1 mixtures.py (9f5ef11)>
                                            [pilot specs.json]
Output: results/w4/bg_fix/checks.txt (and stdout). Exit code 1 if any check fails.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.malecns import load  # noqa: E402
from sim.cns import regions  # noqa: E402
from stimuli import mixtures as mx  # noqa: E402

OUT = ROOT / 'results' / 'w4' / 'bg_fix'
LINES, FAIL = [], []


def log(s=''):
    print(s, flush=True)
    LINES.append(s)


def check(name, ok, detail=''):
    log(f'[{"PASS" if ok else "FAIL"}] {name}' + (f': {detail}' if detail else ''))
    if not ok:
        FAIL.append(name)


def _load(name, path):
    spec_ = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return mod


def _same_blocks(a, b):
    return len(a) == len(b) and all(x[0] == y[0] and x[1] == y[1] and x[3] == y[3] and np.array_equal(x[2], y[2])
                                    for x, y in zip(a, b))


def main(legacy_path, v1_path, specs_path=None):
    specs_path = Path(specs_path or ROOT / 'results' / 'w4' / 'regression' / 'pilot' / 'specs.json')
    meta, _ = load(w_syn=1.0, monoamine='removed')
    pos = {b: i for i, b in enumerate(meta.bodyId)}
    mk = mx.bg_class_masks(meta)
    sens, bri, abd = mk['sensory'], mk['bristle'], mk['abdominal']
    vnc, _ = regions(meta)

    log('## 1. Class masks')
    log(f'sensory {sens.sum():,}; bristle {bri.sum():,}; abdominal {abd.sum():,}; overlap {(bri & abd).sum()}')
    check('mixtures._vnc == sim.cns.regions VNC mask', np.array_equal(mx._vnc(meta), vnc))
    check('bristle = 2,206 and abdominal = 1,040 (diagnosis counts)', bri.sum() == 2206 and abd.sum() == 1040)
    o2 = mk['o2']
    log(f'O2 (abdominal & type in {mx.O2_TYPES}): {o2.sum()}; '
        + ', '.join(f'{k} {v}' for k, v in meta[o2].type.value_counts().items())
        + f'; rest-abdominal {(abd & ~o2).sum():,}')
    check('O2 = 18 (12 SAxx01 + 6 SNxx16) and rest-abdominal = 1,022', o2.sum() == 18 and (abd & ~o2).sum() == 1022)
    ty = meta.type.astype(str).isin(mx.O2_TYPES).to_numpy()
    out = meta[ty & ~abd]
    log(f'SAxx01/SNxx16 outside the abdominal mask (not O2, stay at bg_hz): {len(out)}')
    for _, r in out.iterrows():
        log(f'  bodyId {r.bodyId}: type {r.type}, instance {r.instance}, superclass {r.superclass}, '
            f'class {r["class"]}, subclass {r.subclass}, sensory {bool(sens[_])}')
    s = meta[sens]
    sub = s.subclass.astype(str)
    g = s[sub.str.contains('bristle', case=False)].groupby(['superclass', 'class', 'subclass'], observed=True,
                                                           dropna=False).size()
    log('sensory subclasses containing "bristle" (superclass, class, subclass: n):')
    for k, v in g.items():
        log(f'  {k}: {v}')
    head = s[~vnc[sens]]
    bm = head[head.type.astype(str).str.match(r'^BM(_|$)')]
    log(f'head sensory types BM* (bristle mechanosensory, NOT in the bristle mask): {len(bm):,} neurons; '
        + ', '.join(f'{k} {v}' for k, v in bm.type.value_counts().items()))
    log('  their (class, subclass): ' + ', '.join(
        f'{k} {v}' for k, v in bm.groupby(['class', 'subclass'], observed=True, dropna=False).size().items()))

    log('\n## 2. Block structure (class_v1)')
    for abd_bg in ('same', 0.0, 2.0):
        sp = mx.background_episode(3.0, 0, T=1.0, abdominal_bg=abd_bg, scheme='class_v1')
        bl = mx.to_blocks(sp, meta, pos)
        cover = np.zeros(len(meta), int)
        rate = np.zeros(len(meta))
        for t0, t1, ix, r in bl:
            cover[ix] += 1
            rate[ix] += r
        a_expect = 3.0 if abd_bg == 'same' else float(abd_bg)
        rest = sens & ~bri & ~abd
        ok = ((cover[bri] == 0).all() and (cover[~sens] == 0).all() and (cover[rest] == 1).all()
              and np.allclose(rate[rest], 3.0) and np.allclose(rate[abd], a_expect)
              and (cover[abd] == (1 if a_expect > 0 else 0)).all())
        check(f"abdominal_bg={abd_bg!r}: blocks {[(len(b[2]), b[3]) for b in bl]}", ok,
              f'bristle covered {int((cover[bri] > 0).sum())}, abdominal rate {sorted(set(rate[abd].tolist()))}, '
              f'driven {int((cover > 0).sum()):,}')
    sp = mx.background_episode(3.0, 0, T=1.0, bristle_bg=0.5, scheme='class_v1', abdominal_bg='same')
    bl = mx.to_blocks(sp, meta, pos)
    check('class_v1 bristle_bg=0.5 gives a separate bristle block', len(bl) == 2 and set(bl[1][2]) == set(np.flatnonzero(bri))
          and bl[1][3] == 0.5)
    sp = mx.sample_episode(7)
    bl = mx.to_blocks(sp, meta, pos)
    check('sampled spec (class_v2): background blocks exclude bristles and O2, words follow',
          not np.isin(np.flatnonzero(bri | o2), np.concatenate([b[2] for b in bl[:2]])).any()
          and len(bl) == 2 + len(sp['words']) and bl[1][3] == 1.0 and np.array_equal(bl[1][2], np.flatnonzero(abd & ~o2)),
          f"spec 7: bg {sp['bg_hz']} Hz, blocks {[(len(b[2]), b[3]) for b in bl]}, words {[w['word'] for w in sp['words']]}")

    log('\n## 2b. Block structure (class_v2, default: bristle 0, O2 0, other abdominal 1 Hz)')
    rest = sens & ~bri & ~abd
    for bg_hz in (1.0, 3.0):
        sp = mx.background_episode(bg_hz, 0, T=1.0)
        bl = mx.to_blocks(sp, meta, pos)
        cover = np.zeros(len(meta), int)
        rate = np.zeros(len(meta))
        for t0, t1, ix, r in bl:
            cover[ix] += 1
            rate[ix] += r
        ok = ((cover[bri] == 0).all() and (cover[o2] == 0).all() and (cover[~sens] == 0).all()
              and (cover[abd & ~o2] == 1).all() and (rate[abd & ~o2] == 1.0).all()
              and (cover[rest] == 1).all() and (rate[rest] == bg_hz).all())
        check(f"bg {bg_hz} Hz, scheme {sp['bg_scheme']}: blocks {[(len(b[2]), b[3]) for b in bl]}", ok,
              f'driven {int((cover > 0).sum()):,}; rates: bristle {sorted(set(rate[bri].tolist()))}, '
              f'O2 {sorted(set(rate[o2].tolist()))}, rest-abdominal {sorted(set(rate[abd & ~o2].tolist()))}, '
              f'other sensory {sorted(set(rate[rest].tolist()))}, non-sensory {sorted(set(rate[~sens].tolist()))}')

    log('\n## 2c. Saved class_v1 specs (results/w4/bg_fix/runs)')
    v1_mx = _load('mixtures_class_v1', v1_path)
    n_v1, same_v1 = 0, 0
    for f in sorted((OUT / 'runs').glob('*.npz')):
        sp = json.loads(str(np.load(f)['spec']))
        if sp.get('bg_scheme', {}).get('name') != 'class_v1':
            continue
        n_v1 += 1
        same_v1 += _same_blocks(mx.to_blocks(sp, meta, pos), v1_mx.to_blocks(sp, meta, pos))
    check('to_blocks(saved class_v1 spec) identical to class_v1 to_blocks', n_v1 > 0 and same_v1 == n_v1,
          f'{same_v1} / {n_v1} specs')
    check('class_v1 background_episode == 9f5ef11 background_episode (same, 0, 2 Hz)',
          all(mx.background_episode(3.0, s, T=1.0, scheme='class_v1', abdominal_bg=a)
              == v1_mx.background_episode(3.0, s, T=1.0, abdominal_bg=a) for s in range(3) for a in ('same', 0.0, 2.0)))
    check('sample_episode(scheme=class_v1) == 9f5ef11 sample_episode, seeds 0-999',
          all(mx.sample_episode(s, scheme='class_v1', abdominal_bg='same') == v1_mx.sample_episode(s)
              for s in range(1000)))

    log('\n## 3. Looming share and background range, spec seeds 0-4999')
    new = [mx.sample_episode(s) for s in range(5000)]
    old = [mx.sample_episode(s, legacy=True) for s in range(5000)]
    nl = sum(any(w['word'] == 'looming' for w in e['words']) for e in new)
    ol = sum(any(w['word'] == 'looming' for w in e['words']) for e in old)
    share = nl / 5000
    check('new sampler looming share within 1/6 +- 0.02', abs(share - 1 / 6) <= 0.02,
          f'{nl} / 5000 = {share:.4f} (1/6 = {1 / 6:.4f}; binomial SE {np.sqrt(share * (1 - share) / 5000):.4f})')
    log(f'legacy sampler looming share: {ol} / 5000 = {ol / 5000:.4f} (3/7 = {3 / 7:.4f})')
    bg = np.array([e['bg_hz'] for e in new])
    check('bg_hz in [1, 3]', bg.min() >= 1 and bg.max() <= 3, f'min {bg.min()}, max {bg.max()}, mean {bg.mean():.3f}')
    nw = np.bincount([len(e['words']) for e in new], minlength=5)[2:]
    log(f'words per episode 2/3/4: {nw.tolist()}')
    mods = {}
    for e in new:
        for w in e['words']:
            mods[w['modality']] = mods.get(w['modality'], 0) + 1
    log('episodes containing each modality: ' + ', '.join(f'{k} {v / 5000:.3f}' for k, v in sorted(mods.items())))
    check('at most one word per modality', all(len({w['modality'] for w in e['words']}) == len(e['words'])
                                               for e in new))

    log('\n## 4. T1 pilot specs')
    saved = json.load(open(specs_path))
    sys.path.insert(0, str(ROOT / 'scripts'))
    import w4d4_pilot
    regen = w4d4_pilot.specs()
    check('w4d4_pilot.specs() == saved specs.json', regen == saved, f'{len(saved)} specs')
    old_mx = _load('mixtures_day4', legacy_path)
    same = [_same_blocks(mx.to_blocks(sp, meta, pos), old_mx.to_blocks(sp, meta, pos)) for sp in saved]
    check('to_blocks(saved spec) identical to Day 4 to_blocks', all(same), f'{sum(same)} / {len(same)} specs')
    check('legacy sample_episode == Day 4 sample_episode, seeds 0-999',
          all(mx.sample_episode(s, legacy=True) == old_mx.sample_episode(s) for s in range(1000)))

    log(f'\n{len(FAIL)} failed' + (f': {FAIL}' if FAIL else ''))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'checks.txt').write_text('\n'.join(LINES) + '\n')
    return not FAIL


if __name__ == '__main__':
    sys.exit(0 if main(*sys.argv[1:]) else 1)
