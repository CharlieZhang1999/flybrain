#!/usr/bin/env python
"""W4D3: is the brain tail self-sustaining once ignited? Looming run to 0.6 s, save state, resume to 1.2 s with and without VNC->brain synapses (uses the day-2 save/resume). Run from the repo root. Output: results/w4/sci/w4d3_tail_cut_ascending.txt."""
import sys; sys.path.insert(0,'.')
import numpy as np, torch, pandas as pd
from sim.cns import cns_model, _mask
from sim.lif_numpy import P
from sim import lif_torch
from scripts.ckpt2_ab import stim_idx
m=cns_model('cns','calib'); meta,vnc=m['meta'],m['vnc']
t=meta.type.astype(str).values
pos={b:i for i,b in enumerate(meta.bodyId)}; exc=stim_idx(meta,pos,'looming')
noAN=_mask(m['W'],lambda r,c:~(vnc[c]&~vnc[r]))
cx=pd.Series(t).str.match(r'^(EPG|PEN|PEG|Delta7|EL|ER\d|EPGt)').values
kw=dict(apl=m['apl'],gap=m['gap'],presyn=m['presyn'],t_stim=0.4,record=True,device='cpu',rng='torch',dtype=torch.float32)
for s in [1,2]:
    p=dict(P); p['t_run']=0.6
    a=lif_torch.run_trial(m['W'],exc,params=p,seed=s,return_state=True,**kw)
    for name,W in [('keep AN',m['W']),('cut VNC->brain at 0.6 s',noAN)]:
        p2=dict(P); p2['t_run']=1.2
        b=lif_torch.run_trial(W,exc,params=p2,seed=s,state=a['state'],**kw)
        late=np.unique(b['i'][b['t']*P['dt']>=1.0])
        print(f'seed {s} {name:24s}: active 1.0-1.2 s brain {int((~vnc[late]).sum())} (CX {int(cx[late].sum())}) VNC {int(vnc[late].sum())}',flush=True)
