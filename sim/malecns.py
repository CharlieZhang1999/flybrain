"""Build and cache the MaleCNS v1.0 network for the NumPy LIF.

Inputs (data/malecns/, from male-cns.janelia.org/download flat-connectome):
  body-annotations-male-cns-v1.0-minconf-0.5.feather
  body-neurotransmitters-male-cns-v1.0.feather
  connectome-weights-male-cns-v1.0-minconf-0.5.feather   (152M edges)

Network: all neurons with status == 'Traced' (brain + VNC, ~165k).
Weight: synapse count x sign x w_syn. Sign from consensus_nt (falling back to
predicted_nt): gaba, glutamate, histamine -> -1; acetylcholine and unclear -> +1.
Dopamine, serotonin, octopamine depend on the `monoamine` option:
  'excitatory' -> +1, the Shiu et al. 2024 convention;
  'removed'    -> synapses deleted. Biologically these act slowly (plasticity,
                  gain changes); this model has no slow pathway, so 'removed'
                  means no contribution at all, not a modulatory one.
In MaleCNS, DPM and PPL1 dopamine neurons each make >10k synapses onto Kenyon
cells, so under 'excitatory' they form a fast DPM <-> KC excitatory loop.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.feather as pf
from scipy import sparse

from sim.lif_numpy import P

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'malecns'
META = DATA / 'malecns_traced_meta.parquet'
INHIB = {'gaba', 'glutamate', 'histamine'}
MODUL = {'dopamine', 'serotonin', 'octopamine'}

META_COLS = ['bodyId', 'type', 'instance', 'superclass', 'class', 'subclass',
             'somaSide', 'flywireType', 'exitNerve', 'somaNeuromere']


def cache_path(monoamine):
    return DATA / f'malecns_traced_net_mono-{monoamine}.npz'


def load_meta(monoamine='removed'):
    a = pd.read_feather(DATA / 'body-annotations-male-cns-v1.0-minconf-0.5.feather')
    a = a[a.status == 'Traced'][META_COLS].reset_index(drop=True)
    nt = pd.read_feather(DATA / 'body-neurotransmitters-male-cns-v1.0.feather',
                         columns=['body', 'consensus_nt', 'predicted_nt'])
    nt = nt.set_index('body')
    cons = nt['consensus_nt'].where(nt['consensus_nt'] != 'unclear', nt['predicted_nt'])
    a['nt'] = a.bodyId.map(cons).fillna('unclear').astype(str)
    mono = {'removed': 0, 'excitatory': 1}[monoamine]   # 'excitatory' = Shiu convention
    a['sign'] = np.select([a.nt.isin(INHIB), a.nt.isin(MODUL)], [-1, mono], 1).astype(np.int8)
    return a


def build(monoamine='removed', min_syn=1, force=False):
    CACHE = cache_path(monoamine)
    if CACHE.exists() and META.exists() and not force:
        return
    meta = load_meta(monoamine)
    ids = meta.bodyId.to_numpy()
    print(f'traced neurons: {len(ids):,}')
    t = pf.read_table(DATA / 'connectome-weights-male-cns-v1.0-minconf-0.5.feather')
    keep = pc.and_(pc.is_in(t['body_pre'], value_set=pa_array(ids)),
                   pc.is_in(t['body_post'], value_set=pa_array(ids)))
    if min_syn > 1:
        keep = pc.and_(keep, pc.greater_equal(t['weight'], min_syn))
    t = t.filter(keep)
    print(f'edges among traced neurons: {t.num_rows:,}')
    pos = pd.Series(np.arange(len(ids)), index=ids)
    pre = pos.reindex(t['body_pre'].to_numpy()).to_numpy()
    post = pos.reindex(t['body_post'].to_numpy()).to_numpy()
    syn = t['weight'].to_numpy().astype(np.float32)
    signed = syn * meta.sign.to_numpy()[pre]
    nz = signed != 0
    W = sparse.csc_matrix((signed[nz], (post[nz], pre[nz])),
                          shape=(len(ids), len(ids)), dtype=np.float32)
    sparse.save_npz(CACHE, W)
    meta.to_parquet(META)
    print(f'saved {CACHE.name}: nnz={W.nnz:,}, total synapses={int(syn.sum()):,}')


def pa_array(x):
    import pyarrow as pa
    return pa.array(x)


def load(w_syn=P['w_syn'], monoamine='removed'):
    """Return (meta DataFrame, W csc in volts). Rows/cols follow meta order.

    monoamine: 'removed' (dopamine/serotonin/octopamine synapses deleted) or
               'excitatory' (Shiu et al. convention, counted as +1).
    """
    build(monoamine)
    W = sparse.load_npz(cache_path(monoamine)).tocsc().astype(np.float64) * w_syn
    meta = pd.read_parquet(META)
    return meta, W


if __name__ == '__main__':
    for m in ('removed', 'excitatory'):
        build(m, force=True)
