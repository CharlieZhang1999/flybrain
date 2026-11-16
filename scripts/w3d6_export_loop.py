#!/usr/bin/env python
"""Week 3 Day 6: export loop runs for the split-screen loop player.
Input: results/loop/w3d6_loop_<cfg>_<seed>.json + _spikes.npz (from w3d5_loop.py ... w3d6).
Raster rows (ordered top to bottom, each group sorted by first spike):
  sensory input   : stimulated word and JO neurons that fired (up to 160)
  antennal lobe   : ALPN / ALLN (up to 120 most active)
  central brain   : everything else that is not DN / motor (up to 260 most active)
  descending      : descending neurons (up to 160 most active)
Spikes are stored as two base64 arrays: row (uint16) and time in 0.5 ms units (uint16).
Output: artifact/loop/data/loop_<cfg>_<seed>.json and artifact/loop/data/index.json"""
import base64
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stimuli.malecns_words import WORDS, select_ids  # noqa: E402
from loop.actions import jo_groups  # noqa: E402

L = ROOT / 'results' / 'loop'; OUT = ROOT / 'artifact' / 'loop' / 'data'; OUT.mkdir(parents=True, exist_ok=True)
CAPS = [('Sensory input', 160), ('Antennal lobe', 120), ('Central brain', 260), ('Descending neurons', 160)]
meta = pd.read_parquet(ROOT / 'data' / 'malecns' / 'malecns_traced_meta.parquet')
n = len(meta); pos = {b: i for i, b in enumerate(meta.bodyId)}
stim = np.zeros(n, bool)
STIM_SETS = {}
for w in WORDS:
    ids = np.array([pos[b] for b in select_ids(meta, w)]); stim[ids] = True; STIM_SETS[w] = ids
g = jo_groups(meta); stim[g['A']] = True; stim[g['B']] = True
STIM_SETS['JO-A'] = np.asarray(g['A']); STIM_SETS['JO-B'] = np.asarray(g['B'])
PER_STIM = 8  # sensory band: up to 8 most active neurons per stimulus set, so every word gets a stripe
cls = meta['class'].astype(str).to_numpy(); sup = meta.superclass.astype(str).to_numpy()
grp = np.full(n, 2)
grp[np.isin(cls, ['ALPN', 'ALLN'])] = 1
grp[sup == 'descending_neuron'] = 3
grp[np.char.find(sup.astype(str), 'motor') >= 0] = -1
grp[np.char.find(sup.astype(str), 'sensory') >= 0] = -1
grp[stim] = 0


def b64(a, dt):
    return base64.b64encode(np.asarray(a, dtype=dt).tobytes()).decode()


TAG = sys.argv[1] if len(sys.argv) > 1 else 'w3d6'  # w3d6 = offline stub replies, w3d6c = Claude replies
index = []
for f in sorted(L.glob(f'{TAG}_loop_*_spikes.npz')):
    key = f.name.replace(f'{TAG}_loop_', '').replace('_spikes.npz', '')
    run = json.load(open(L / f'{TAG}_loop_{key}.json'))
    sp = np.load(f); t, i = sp['t'].astype(float), sp['i'].astype(int)
    tot = np.bincount(i, minlength=n)
    first = np.full(n, 1e9); u, k = np.unique(i, return_index=True); first[u] = t[k]
    rows, groups = [], []
    for gi, (name, cap) in enumerate(CAPS):
        if gi == 0:
            picks = []
            for ids in STIM_SETS.values():
                a = ids[tot[ids] > 0]
                picks += [x for x in a[np.argsort(-tot[a])][:PER_STIM] if x not in picks]
            c = np.array(picks[:cap], dtype=int)
        else:
            c = np.flatnonzero((grp == gi) & (tot > 0))
            c = c[np.argsort(-tot[c])][:cap]
        c = c[np.argsort(first[c])]
        groups.append(dict(name=name, start=len(rows), n=int(len(c))))
        rows += list(c)
    rows = np.array(rows); loc = np.full(n, -1); loc[rows] = np.arange(len(rows))
    m = loc[i] >= 0
    order = np.argsort(t[m], kind='stable')
    R, T = loc[i[m]][order], np.round(t[m][order] * 2000).astype(int)
    view = dict(key=key, cfg=run['cfg'], seed=run['seed'], t_total=run['t_total'],
                rounds=run['rounds'], pop_10ms=run['pop_10ms'], groups=groups, n_rows=int(len(rows)),
                row_types=[str(meta.type.iloc[r]) if isinstance(meta.type.iloc[r], str) else '' for r in rows],
                spikes_row=b64(R, np.uint16), spikes_t=b64(T, np.uint16), n_spikes_shown=int(m.sum()),
                n_spikes_total=int(len(t)), n_active=int((tot > 0).sum()))
    json.dump(view, open(OUT / f'loop_{key}.json', 'w'))
    index.append(dict(key=key, cfg=run['cfg'], seed=run['seed']))
    print(key, 'rows', len(rows), 'spikes shown', int(m.sum()), 'of', len(t), 'size',
          (OUT / f'loop_{key}.json').stat().st_size // 1024, 'KB')
json.dump(index, open(OUT / 'index.json', 'w'))
