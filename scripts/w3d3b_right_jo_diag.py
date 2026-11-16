#!/usr/bin/env python
"""Week 3 Day 3b: is the right-JO deficit in MaleCNS a reconstruction artifact?
Counts JO bodies (all statuses) and output synapses per JO group and side in
MaleCNS v1.0 (traced network, minconf 0.5) and FlyWire 783 (Shiu et al. model
connectivity, >=5 synapses per connection).
Output: results/loop/w3d3b_diag_malecns.csv, w3d3b_diag_flywire.csv, w3d3b_diag.json"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'results' / 'loop'
D = ROOT / 'data'
TP = ROOT / 'third_party' / 'Drosophila_brain_model'


def grp(t):
    t = pd.Series(t).astype(str)
    g = t.str.extract(r'^JO-([A-Z]+)')[0].fillna('unclear/other')
    g[t.isin(['JO-unclear'])] = 'unclear/other'
    return g.to_numpy()


# MaleCNS: all bodies, then output synapses in the traced network
a = pd.read_feather(D / 'malecns' / 'body-annotations-male-cns-v1.0-minconf-0.5.feather')
jo_all = a[a['type'].astype(str).str.startswith('JO')]
an = a[a.entryNerve.astype(str) == 'AN']
status = dict(jo_status=jo_all.status.astype(str).value_counts().to_dict(),
              mechanosensory_AN_by_status_side=pd.crosstab(
                  an[an['class'] == 'mechanosensory'].status.astype(str),
                  an[an['class'] == 'mechanosensory'].rootSide.astype(str)).to_dict())
W = sparse.load_npz(D / 'malecns' / 'malecns_traced_net_mono-removed.npz').tocsc()
m = pd.read_parquet(D / 'malecns' / 'malecns_traced_meta.parquet')
m['out'] = np.asarray(abs(W).sum(0)).ravel()
jo = m[m['type'].astype(str).str.startswith('JO')].copy()
jo['side'] = jo.instance.astype(str).str[-1]
jo['group'] = grp(jo['type'])
jo['unclear'] = jo['type'].str.contains('unclear')
mc = jo.groupby(['group', 'side']).agg(n=('out', 'size'), n_unclear=('unclear', 'sum'),
                                       out_syn=('out', 'sum'), median_out=('out', 'median'),
                                       n_lt10=('out', lambda x: int((x < 10).sum()))).unstack()
mc.to_csv(OUT / 'w3d3b_diag_malecns.csv')

# FlyWire 783
f = pd.read_csv(D / 'flywire_annot' / 'annot.tsv', sep='\t', low_memory=False)
fj = f[f.cell_type.astype(str).str.startswith('JO')].copy()
c = pd.read_parquet(TP / 'Connectivity_783.parquet')
comp = pd.read_csv(TP / 'Completeness_783.csv', index_col=0)
pos = pd.Series(np.arange(len(comp)), index=comp.index)
out = np.bincount(c.Presynaptic_Index, weights=np.abs(c.Connectivity), minlength=len(comp))
ix = pos.reindex(fj.root_id).to_numpy()
fj['out'] = np.where(np.isnan(ix), 0, out[np.nan_to_num(ix).astype(int)])
fj['group'] = grp(fj.cell_type)
fw = fj.groupby(['group', 'side']).agg(n=('out', 'size'), out_syn=('out', 'sum'),
                                       median_out=('out', 'median')).unstack()
fw.to_csv(OUT / 'w3d3b_diag_flywire.csv')

summ = dict(malecns_bodies=jo.side.value_counts().to_dict(),
            malecns_out=jo.groupby('side').out.sum().to_dict(),
            malecns_unclear=jo[jo.unclear].side.value_counts().to_dict(),
            malecns_zero_out=jo[jo.out == 0].side.value_counts().to_dict(),
            flywire_bodies=fj.side.value_counts().to_dict(),
            flywire_out=fj.groupby('side').out.sum().to_dict(), **status)
json.dump(summ, open(OUT / 'w3d3b_diag.json', 'w'), indent=1, default=str)
pd.set_option('display.width', 200)
print(mc.to_string()); print(fw.to_string()); print(json.dumps(summ, indent=1, default=str))
