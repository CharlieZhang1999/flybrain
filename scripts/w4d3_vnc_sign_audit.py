#!/usr/bin/env python
"""W4D3 (science track): transmitter-sign audit for the VNC part of the model.

Synapse-weighted shares of presynaptic transmitter (meta.nt = consensus_nt,
else predicted_nt) for edge classes: brain->brain (reference), VNC->VNC,
brain->VNC (DNs), VNC->brain (ANs), all edges onto motor neurons, and edges
made by motor neurons (MN -> X). Also: share of VNC synapses whose
presynaptic prediction confidence is < 0.5, and the main 'unclear' and
glutamatergic presynaptic types onto MNs.
Usage: python scripts/w4d3_vnc_sign_audit.py
Output: results/w4/sci/w4d3_sign_audit.txt
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.malecns import load, DATA  # noqa: E402
from sim.cns import regions  # noqa: E402

OUT = ROOT / 'results' / 'w4' / 'sci'


def main():
    meta, W = load(w_syn=1.0, monoamine='excitatory')    # keep monoamines to count them
    vnc, _ = regions(meta)
    C = W.tocoo()
    pre, post, syn = C.col, C.row, np.abs(C.data)
    nt = meta.nt.astype(str).values
    sc = meta.superclass.astype(str).values
    mn = sc == 'vnc_motor'
    conf = pd.read_feather(DATA / 'body-neurotransmitters-male-cns-v1.0.feather',
                           columns=['body', 'predicted_nt_confidence', 'consensus_nt'])
    cmap = conf.set_index('body').predicted_nt_confidence
    pconf = meta.bodyId.map(cmap).fillna(0).to_numpy()
    cons_unclear = meta.bodyId.map(conf.set_index('body').consensus_nt).fillna('unclear').eq('unclear').to_numpy()
    classes = {
        'brain -> brain': ~vnc[pre] & ~vnc[post],
        'VNC -> VNC': vnc[pre] & vnc[post],
        'brain -> VNC': ~vnc[pre] & vnc[post],
        'VNC -> brain': vnc[pre] & ~vnc[post],
        'any -> MN': mn[post],
        'MN -> any': mn[pre],
    }
    nts = ['acetylcholine', 'gaba', 'glutamate', 'histamine', 'dopamine', 'serotonin', 'octopamine', 'unclear']
    lines = ['# W4D3 VNC transmitter-sign audit (synapse-weighted shares, %)', '',
             '| edge class | synapses | ' + ' | '.join(nts) + ' | consensus unclear (fallback to prediction) | pred. conf < 0.5 |',
             '|' + '---|' * (len(nts) + 4)]
    for name, k in classes.items():
        tot = syn[k].sum()
        sh = [100 * syn[k & (nt[pre] == x)].sum() / tot for x in nts]
        cu = 100 * syn[k & cons_unclear[pre]].sum() / tot
        lc = 100 * syn[k & (pconf[pre] < 0.5)].sum() / tot
        lines.append(f'| {name} | {tot / 1e6:.2f}M | ' + ' | '.join(f'{v:.1f}' for v in sh) + f' | {cu:.1f} | {lc:.1f} |')
    # MN outputs: where do they go?
    k = mn[pre]
    lines += ['', f'MN -> X edges: {k.sum():,} edges, {syn[k].sum():,.0f} synapses; '
              f'MN transmitters: ' + ', '.join(f'{a} {b}' for a, b in pd.Series(nt[mn]).value_counts().items())]
    tgt = pd.Series(syn[k], index=sc[post[k]]).groupby(level=0).sum().sort_values(ascending=False)
    lines.append('MN -> X targets by superclass (synapses): ' + ', '.join(f'{a} {b:,.0f}' for a, b in tgt.head(6).items()))
    # presynaptic types onto MNs, by transmitter
    t = meta.type.astype(str).values
    k = mn[post]
    df = pd.DataFrame(dict(type=t[pre[k]], nt=nt[pre[k]], syn=syn[k]))
    for x in ['glutamate', 'unclear', 'gaba', 'acetylcholine']:
        g = df[df.nt == x].groupby('type').syn.sum().sort_values(ascending=False)
        lines.append(f'Top {x} inputs to MNs ({g.sum():,.0f} syn): ' + ', '.join(f'{a} {b:,.0f}' for a, b in g.head(8).items()))
    hl = df.assign(hl=df.type.str.extract(r'^IN(\d\d[A-B])')[0]).dropna()
    g = hl.groupby(['hl', 'nt']).syn.sum().unstack(fill_value=0)
    g = g.loc[g.sum(axis=1).sort_values(ascending=False).index[:12]]
    lines += ['', 'Hemilineage (IN type prefix) inputs to MNs by transmitter (synapses):', g.to_string()]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'w4d3_sign_audit.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
