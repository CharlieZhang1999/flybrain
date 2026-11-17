#!/usr/bin/env python
"""W4D3: looming tail (0.6-1.0 s) after cutting each candidate loop: MN outputs glutamatergic / removed, DNg33 L<->R, IN19B040 quartet, all three. Run from the repo root. Output copied to results/w4/sci/w4d3_loop_lesions.txt."""
import sys; sys.path.insert(0,'.')
import numpy as np, torch
from sim.cns import cns_model, _mask
from sim.lif_numpy import P
from sim import lif_torch
from scripts.ckpt2_ab import stim_idx
m=cns_model('cns','calib'); meta,vnc=m['meta'],m['vnc']
t=meta.type.astype(str).values; sc=meta.superclass.astype(str).values
pos={b:i for i,b in enumerate(meta.bodyId)}; exc=stim_idx(meta,pos,'looming')
mn=sc=='vnc_motor'; dng33=t=='DNg33'; in19=t=='IN19B040'
def mnglu(W):
    W=W.tocsc().copy(); C=W.tocoo(); k=mn[C.col]
    C.data[k]=-np.abs(C.data[k]); 
    from scipy import sparse
    return sparse.csc_matrix((C.data,(C.row,C.col)),shape=W.shape)
V={'cns':m['W'],
   'cns, MN outputs glutamatergic':mnglu(m['W']),
   'cns, MN outputs removed':_mask(m['W'],lambda r,c:~mn[c]),
   'cns, DNg33 L<->R cut':_mask(m['W'],lambda r,c:~(dng33[r]&dng33[c])),
   'cns, IN19B040<->IN19B040 cut':_mask(m['W'],lambda r,c:~(in19[r]&in19[c])),
}
V['cns, all three']=_mask(_mask(_mask(m['W'],lambda r,c:~mn[c]),lambda r,c:~(dng33[r]&dng33[c])),lambda r,c:~(in19[r]&in19[c]))
p=dict(P); p['t_run']=1.0
for name,W in V.items():
    out=[]
    for s in range(3):
        r=lif_torch.run_trial(W,exc,params=p,seed=s,apl=m['apl'],gap=m['gap'],presyn=m['presyn'],t_stim=0.4,record=True,device='cpu',rng='torch',dtype=torch.float32)
        tt=r['t']*P['dt']; dur=r['i'][tt<0.4]; late=np.unique(r['i'][tt>=0.6])
        act=np.unique(dur)
        out.append((int((~vnc[late]).sum()),int(vnc[late].sum()),int(mn[act].sum())))
    print(f'{name:34s} tail 0.6-1.0 s (brain, VNC) per seed: {[o[:2] for o in out]} | MNs active during drive: {[o[2] for o in out]}',flush=True)
