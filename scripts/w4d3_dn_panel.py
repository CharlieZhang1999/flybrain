#!/usr/bin/env python
"""W4D3 (science track): descending-neuron validation panel, real vs shuffled,
with and without the GF -> TTMn / PSI gap junctions.

Conditions (Poisson 150 Hz on the named neurons 0-0.3 s, run to 0.4 s):
  none; GF (DNp01 both sides); MDN (4); DNg100 (2); DNa02_L / DNa02_R (one
  side each, for the turning asymmetry); looming (sensory word, takeoff).
Configs: calib, shuffle, calib+gf<mV> (rectifying GF junctions, spikelet in mV),
  +mnglu (motor-neuron outputs glutamatergic), e.g. calib+gf8+mnglu.
Per trial we record the rate (Hz, 0-0.3 s) of every motor pool and of a few
named neurons (GF, TTMn, PSI, DLMn), and the TTMn and PSI spikes that follow
a GF spike within 3 ms (1:1 following).
Usage: python scripts/w4d3_dn_panel.py <config> [seeds=3] [conds=all]
Output: results/w4/sci/w4d3_panel_<config>.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.cns import cns_model, motor_pools  # noqa: E402
from sim.lif_numpy import P  # noqa: E402
from sim import lif_torch  # noqa: E402
from scripts.ckpt2_ab import stim_idx  # noqa: E402

OUT = ROOT / 'results' / 'w4' / 'sci'
CONDS = ['none', 'GF', 'MDN', 'DNg100', 'DNa02_L', 'DNa02_R', 'looming']
T_STIM, T_RUN = 0.3, 0.4
DN = {'GF': 'DNp01'}


def drive(meta, pos, c):
    t = meta.type.astype(str).values
    side = meta.somaSide.astype(str).values
    if c == 'none':
        return np.array([], int)
    if c == 'looming':
        return stim_idx(meta, pos, c)
    typ, _, sd = c.partition('_')
    ix = t == DN.get(typ, typ)
    if sd:
        ix &= side == sd
    return np.flatnonzero(ix)


def follow(ts, si, src, dst, win=30):
    """Fraction of src spikes followed by a dst spike within win steps."""
    a = np.sort(ts[np.isin(si, src)]); b = np.sort(ts[np.isin(si, dst)])
    if not len(a) or not len(b):
        return 0.0 if len(a) else np.nan
    k = np.searchsorted(b, a)
    ok = (k < len(b)) & (b[np.minimum(k, len(b) - 1)] - a <= win)
    return float(ok.mean())


def main(config='calib', seeds=3, conds='all'):
    OUT.mkdir(parents=True, exist_ok=True)
    parts = config.split('+')            # e.g. calib, shuffle+gf8, calib+gf8+mnglu
    cfg, gf, mnglu = parts[0], None, 'mnglu' in parts
    for x in parts[1:]:
        if x.startswith('gf'):
            gf = float(x[2:])
    m = cns_model('cns', cfg, gf_gap=gf, mn_glu=mnglu)
    meta = m['meta']
    t = meta.type.astype(str)
    pos = {b: i for i, b in enumerate(meta.bodyId)}
    pools, _ = motor_pools(meta)
    named = {'GF': np.flatnonzero(t.values == 'DNp01'), 'TTMn': np.flatnonzero(t.values == 'TTMn'),
             'PSI': np.flatnonzero(t.values == 'PSI'), 'DLMn': np.flatnonzero(t.str.startswith('DLMn').values)}
    p = dict(P); p['t_run'] = T_RUN
    conds = CONDS if conds == 'all' else conds.split(',')
    rows = []
    for c in conds:
        exc = drive(meta, pos, c)
        for seed in range(int(seeds)):
            t0 = time.time()
            r = lif_torch.run_trial(m['W'], exc, params=p, seed=seed, apl=m['apl'], gap=m['gap'],
                                    presyn=m['presyn'], t_stim=T_STIM, record=True,
                                    device='cpu', rng='torch', dtype=__import__('torch').float32)
            ts, si = r['t'], r['i']
            win = ts * P['dt'] < T_STIM
            rate = np.bincount(si[win], minlength=len(meta)) / T_STIM
            base = dict(config=config, cond=c, seed=seed)
            for name, ix in pools.items():
                rows.append(dict(base, pool=name, rate_hz=rate[ix].mean()))
            for name, ix in named.items():
                rows.append(dict(base, pool='@' + name, rate_hz=rate[ix].mean()))
            rows.append(dict(base, pool='@follow_TTMn', rate_hz=follow(ts, si, named['GF'], named['TTMn'])))
            rows.append(dict(base, pool='@follow_PSI', rate_hz=follow(ts, si, named['GF'], named['PSI'])))
            act = rate > 0
            rows.append(dict(base, pool='@active_MN', rate_hz=int(act[meta.superclass.astype(str).values == 'vnc_motor'].sum())))
            rows.append(dict(base, pool='@active_VNC', rate_hz=int(act[m['vnc']].sum())))
            print(f'{config} {c} seed {seed}: GF {rate[named["GF"]].mean():.0f} Hz, '
                  f'TTMn {rate[named["TTMn"]].mean():.1f}, PSI {rate[named["PSI"]].mean():.1f}, '
                  f'DLMn {rate[named["DLMn"]].mean():.1f} | {time.time() - t0:.0f}s', flush=True)
        pd.DataFrame(rows).to_csv(OUT / f'w4d3_panel_{config}.csv', index=False)


if __name__ == '__main__':
    main(*sys.argv[1:])
