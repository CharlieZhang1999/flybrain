"""Hidden internal states for the Week 5 dataset (T7).

Monoamines are removed in the model, so an internal state is imposed as tonic
Poisson drive to an identified, non-monoamine central cell-type set (Scout note
progress/week5/state_standins.md). Nothing here touches the taste input path.

State sets (STATE_RULES; selected from the MaleCNS v1.0 annotation table,
body-annotations-male-cns-v1.0-minconf-0.5.feather, rows with status 'Traced'):
  hunger   ISN  (type == 'ISN', 4 neurons, ACh)
  thirst   BiT  (type == 'BiT', 2 neurons, ACh)
  arousal  male-specific pC1 / P1 (type starts with 'pC1' and
           dimorphism == 'male-specific', 126 neurons, ACh)
  sleep    ER5  (type == 'ER5', 21 neurons, GABA; backup state)
'baseline' is the no-drive state (state id 0, level 0 Hz).

A state spec is a list of segments, each a dict(t_on, t_off, state, level_hz):
the state's set receives Poisson drive at level_hz on [t_on, t_off). Segments
must not overlap; times are rounded to the 0.1 ms simulator step.
state_blocks(segments, sets, pos) turns them into run_trial blocks
(t_on, t_off, neuron_idx, rate_hz), to be appended to stimuli.mixtures.to_blocks.
A baseline segment, or any segment at 0 Hz, gives no block (as in mixtures:
neurons at 0 Hz are in no block, so their refractory period is unchanged).

label_bins(segments, T, bin_s) gives the per-bin ground truth (state id and
level) that Phase B stores apart from the activity files.

This module imports no torch and no MaleCNS loader, so CI can test it on a
synthetic annotation table.
"""
import numpy as np

DT = 1e-4
STATES = ('baseline', 'hunger', 'thirst', 'arousal', 'sleep')   # index = state id
STATE_ID = {s: k for k, s in enumerate(STATES)}
STATE_RULES = {
    'hunger': dict(type='ISN'),
    'thirst': dict(type='BiT'),
    'arousal': dict(type_prefix='pC1', dimorphism='male-specific'),
    'sleep': dict(type='ER5'),
}
CELL_TYPE = {'hunger': 'ISN', 'thirst': 'BiT', 'arousal': 'pC1/P1 (male-specific)', 'sleep': 'ER5'}


def _q(x):
    """Round a time to the simulator step (0.1 ms), as stimuli.mixtures._q."""
    return round(round(float(x) / DT) * DT, 4)


def _col(df, name):
    return df[name].astype(object).where(df[name].notna(), '').astype(str).to_numpy()


def select_state_ids(annot, state):
    """Sorted body IDs of a state's set, from an annotation table with columns
    bodyId, type, status and (for 'arousal') dimorphism. Only 'Traced' rows count
    when a status column is present."""
    if state not in STATE_RULES:
        raise ValueError(f'unknown state {state!r}; driven states are {sorted(STATE_RULES)}')
    rule = STATE_RULES[state]
    ty = _col(annot, 'type')
    k = np.ones(len(annot), bool)
    if 'status' in annot:
        k &= _col(annot, 'status') == 'Traced'
    if 'type' in rule:
        k &= ty == rule['type']
    if 'type_prefix' in rule:
        k &= np.char.startswith(ty.astype(str), rule['type_prefix'])
    if 'dimorphism' in rule:
        k &= _col(annot, 'dimorphism') == rule['dimorphism']
    ids = np.sort(annot['bodyId'].to_numpy()[k].astype(np.int64))
    if len(ids) == 0:
        raise ValueError(f'state {state!r}: no neurons match {rule}')
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f'state {state!r}: duplicate body IDs')
    return [int(b) for b in ids]


def state_sets(annot, states=tuple(STATE_RULES)):
    """dict state -> sorted body IDs for each driven state."""
    return {s: select_state_ids(annot, s) for s in states}


def segment(t_on, t_off, state, level_hz=0.0):
    """One validated segment dict (times rounded to the step)."""
    if state not in STATE_ID:
        raise ValueError(f'unknown state {state!r}; states are {STATES}')
    t_on, t_off, lv = _q(t_on), _q(t_off), float(level_hz)
    if not 0.0 <= t_on < t_off:
        raise ValueError(f'segment needs 0 <= t_on < t_off, got [{t_on}, {t_off})')
    if lv < 0:
        raise ValueError(f'level_hz must be >= 0, got {lv}')
    if state == 'baseline' and lv != 0:
        raise ValueError('a baseline segment has level 0 Hz')
    return dict(t_on=t_on, t_off=t_off, state=state, level_hz=round(lv, 3))


def check_segments(segments, T=None):
    """Validated, time-sorted copy of a segment list (dicts or (t_on, t_off, state,
    level_hz) tuples). Raises on overlap or on a segment past T."""
    segs = [segment(**s) if isinstance(s, dict) else segment(*s) for s in segments]
    segs.sort(key=lambda s: s['t_on'])
    for a, b in zip(segs, segs[1:]):
        if b['t_on'] < a['t_off']:
            raise ValueError(f'segments overlap: {a} and {b}')
    if T is not None and segs and segs[-1]['t_off'] > _q(T):
        raise ValueError(f'segment {segs[-1]} ends after T={T}')
    return segs


def state_blocks(segments, sets, pos):
    """run_trial blocks [(t_on, t_off, idx, rate_hz), ...] for the driven segments.

    sets: dict state -> body IDs (state_sets); pos: dict body ID -> neuron index.
    Baseline and 0 Hz segments give no block."""
    out = []
    for s in check_segments(segments):
        if s['state'] == 'baseline' or s['level_hz'] <= 0:
            continue
        missing = [b for b in sets[s['state']] if b not in pos]
        if missing:
            raise KeyError(f'state {s["state"]!r}: body IDs not in the model: {missing[:5]}')
        idx = np.array(sorted(pos[b] for b in sets[s['state']]), dtype=np.int64)
        out.append((s['t_on'], s['t_off'], idx, float(s['level_hz'])))
    return out


def label_bins(segments, T, bin_s=0.01):
    """Per-bin ground truth: (state_id int16, level_hz float32), n_bins = round(T / bin_s).
    A bin takes the segment that covers its centre; uncovered bins are baseline, 0 Hz."""
    nb = int(round(T / bin_s))
    c = (np.arange(nb) + 0.5) * bin_s
    sid = np.zeros(nb, np.int16)
    lev = np.zeros(nb, np.float32)
    for s in check_segments(segments, T):
        k = (c >= s['t_on']) & (c < s['t_off'])
        sid[k] = STATE_ID[s['state']]
        lev[k] = s['level_hz']
    return sid, lev


def downstream_targets(W, idx, n_top=50):
    """Top n_top direct postsynaptic targets of the neuron set idx, ranked by the
    summed absolute weight they receive from it (W[post, pre], csc or dense).
    Members of the set are excluded. Returns (target indices, their weights)."""
    idx = np.asarray(idx, np.int64)
    w = np.asarray(abs(W[:, idx]).sum(axis=1), float).ravel()
    w[idx] = 0.0
    nz = np.flatnonzero(w > 0)
    order = nz[np.lexsort((nz, -w[nz]))][:n_top]
    return order.astype(np.int64), w[order]
