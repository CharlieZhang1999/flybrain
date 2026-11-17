#!/usr/bin/env python
"""W4D3: summarise results/w4/sci/w4d3_panel_*.csv into the day-3 tables.
Output: results/w4/sci/w4d3_panel_summary.md"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'results' / 'w4' / 'sci'


def load():
    df = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob('w4d3_panel_*.csv'))])
    parts = df.pool.str.split('|', expand=True)
    df['side'], df['sub'], df['type'] = parts[0], parts[1], parts[2]
    return df


def g(df, cfg, cond, pool=None, sub=None, side=None, typ=None):
    d = df[(df.config == cfg) & (df.cond == cond)]
    if pool is not None:
        d = d[d.pool == pool]
    if sub is not None:
        d = d[d['sub'].isin(sub if isinstance(sub, list) else [sub])]
    if side is not None:
        d = d[d.side == side]
    if typ is not None:
        d = d[d['type'].fillna('').str.startswith(typ)]
    if not len(d):
        return np.nan
    return d.groupby('seed').rate_hz.mean().mean()   # mean over pools, then seeds


def main():
    df = load()
    cfgs = [c for c in ['calib', 'shuffle', 'calib+gf4', 'calib+gf8', 'calib+gf12', 'shuffle+gf8', 'calib+gf8+mnglu', 'shuffle+gf8+mnglu', 'calib+w4', 'shuffle+w4']
            if c in set(df.config)]
    L = ['# W4D3 descending-neuron panel (rates in Hz during 0-0.3 s drive, mean of seeds)', '']
    # 1. expected-target table
    checks = [
        ('GF', 'TTMn (jump)', dict(pool='@TTMn')),
        ('GF', 'PSI', dict(pool='@PSI')),
        ('GF', 'DLMn (power)', dict(pool='@DLMn')),
        ('GF', 'TTMn spikes following a GF spike <3 ms (fraction)', dict(pool='@follow_TTMn')),
        ('MDN', 'hind-leg MN pools', dict(sub='hl')),
        ('MDN', 'front-leg MN pools', dict(sub='fl')),
        ('DNg100', 'front / mid / hind leg pools', None),
        ('looming', 'TTMn (jump)', dict(pool='@TTMn')),
        ('looming', 'DLMn (power)', dict(pool='@DLMn')),
    ]
    L.append('| stimulus | readout | ' + ' | '.join(cfgs) + ' |')
    L.append('|---|---|' + '---|' * len(cfgs))
    for cond, lab, kw in checks:
        if kw is None:
            vals = ['{:.1f} / {:.1f} / {:.1f}'.format(*(g(df, c, cond, sub=s) for s in ['fl', 'ml', 'hl'])) for c in cfgs]
        else:
            vals = [f'{g(df, c, cond, **kw):.2f}' for c in cfgs]
        L.append(f'| {cond} | {lab} | ' + ' | '.join(vals) + ' |')
    # 2. active MNs per condition
    L += ['', '## Active motor neurons (any spike in 0-0.3 s, mean of seeds)', '',
          '| stimulus | ' + ' | '.join(cfgs) + ' |', '|---|' + '---|' * len(cfgs)]
    for cond in ['none', 'GF', 'MDN', 'DNg100', 'DNa02_L', 'DNa02_R', 'looming']:
        L.append(f'| {cond} | ' + ' | '.join(f'{g(df, c, cond, pool="@active_MN"):.0f}' for c in cfgs) + ' |')
    # 3. DNa02 asymmetry: leg + neck pools, ipsi vs contra
    L += ['', '## DNa02 one side: ipsilateral vs contralateral MN pools (Hz)', '',
          '| config | driven | ipsi legs | contra legs | ipsi neck | contra neck | leg asymmetry (i-c)/(i+c) |',
          '|---|---|---|---|---|---|---|']
    for c in cfgs:
        for sd, other in [('L', 'R'), ('R', 'L')]:
            cond = f'DNa02_{sd}'
            il, cl = g(df, c, cond, sub=['fl', 'ml', 'hl'], side=sd), g(df, c, cond, sub=['fl', 'ml', 'hl'], side=other)
            ineck, cneck = g(df, c, cond, sub='nm', side=sd), g(df, c, cond, sub='nm', side=other)
            asym = (il - cl) / (il + cl) if il + cl > 0 else np.nan
            L.append(f'| {c} | {sd} | {il:.2f} | {cl:.2f} | {ineck:.2f} | {cneck:.2f} | {asym:+.2f} |')
    # 4. top pool groups per DN in calib vs shuffle
    from sim.cns import motor_pools
    from sim.malecns import load as mload
    meta, _ = mload(w_syn=1.0, monoamine='removed')
    _, pdf = motor_pools(meta)
    d = df[~df.pool.str.startswith('@')].merge(pdf[['pool', 'group']], on='pool')
    L += ['', '## Top motor groups per stimulus (side x subclass / group, Hz)', '']
    for c in [x for x in ['calib', 'shuffle', 'calib+gf8+mnglu', 'calib+w4'] if x in cfgs]:
        L.append(f'**{c}**')
        L.append('')
        for cond in ['GF', 'MDN', 'DNg100', 'DNa02_L', 'DNa02_R', 'looming']:
            x = d[(d.config == c) & (d.cond == cond)]
            gg = x.groupby(['side', 'sub', 'group']).rate_hz.mean().sort_values(ascending=False)
            top = ', '.join(f'{a}-{b}/{k} {v:.1f}' for (a, b, k), v in gg.head(5).items() if v > 0.05) or 'none'
            L.append(f'- {cond}: {top}')
        L.append('')
    (OUT / 'w4d3_panel_summary.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
