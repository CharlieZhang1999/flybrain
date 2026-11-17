#!/usr/bin/env python
"""W4D3: neurons active in the VNC looming tail with VNC->brain synapses cut (common to 3 seeds), and their excitatory strongly connected components. Run from the repo root."""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, torch
from scipy.sparse.csgraph import connected_components
from sim.cns import cns_model, _mask
from sim.lif_numpy import P
from sim import lif_torch
from scripts.ckpt2_ab import stim_idx
m=cns_model('cns','calib'); meta,vnc=m['meta'],m['vnc']
t=meta.type.astype(str).values; nt=meta.nt.astype(str).values; sc=meta.superclass.astype(str).values
pos={b:i for i,b in enumerate(meta.bodyId)}; exc=stim_idx(meta,pos,'looming')
W=_mask(m['W'], lambda r,c: ~(vnc[c]&~vnc[r]))
p=dict(P); p['t_run']=1.0
tails=[]
for s in range(3):
    r=lif_torch.run_trial(W,exc,params=p,seed=s,apl=m['apl'],gap=m['gap'],presyn=m['presyn'],t_stim=0.4,record=True,device='cpu',rng='torch',dtype=torch.float32)
    late=r['i'][r['t']*P['dt']>=0.6]; tails.append(set(np.unique(late)))
common=sorted(set.intersection(*tails)); print('tail sizes',[len(x) for x in tails],'common',len(common))
ix=np.array(common)
rates=None
Ws=W.tocsr()[ix][:,ix]
E=Ws.copy(); E.data[E.data<0]=0; E.eliminate_zeros()
nc,lab=connected_components(E,directed=True,connection='strong')
big=np.bincount(lab); print('SCC sizes (excitatory, among tail):',sorted(big[big>1],reverse=True))
df=pd.DataFrame(dict(type=t[ix],sc=sc[ix],nt=nt[ix],side=meta.somaSide.astype(str).values[ix],scc=lab))
df['in_scc']=df.scc.map(lambda k: big[k]>1)
print(df.sort_values(['in_scc','sc'],ascending=False).to_string())
# edges in loop by presyn nt
k=np.isin(lab, np.flatnonzero(big>1))
sub=E[k][:,k].tocoo()
pre_nt=nt[ix[k]][sub.col]; pre_sc=sc[ix[k]][sub.col]
print('loop edges by pre nt:', pd.Series(sub.data/ (0.275e-3*0.581), index=pre_nt).groupby(level=0).sum().round(0).to_dict())
print('loop edges by pre superclass:', pd.Series(sub.data/(0.275e-3*0.581), index=pre_sc).groupby(level=0).sum().round(0).to_dict())
np.save('/tmp/claude-0/vnc_tail_common.npy', ix)
