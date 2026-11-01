#!/usr/bin/env python
"""C6b: transmitter and self-wiring audit of lLN1_bc (no simulation).

Transmitter: MaleCNS prediction / consensus / ground_truth vs FlyWire top_nt
and known_nt. Wiring: within-group connectivity in the weights the simulator
uses (MaleCNS x 0.581, FlyWire v783 Shiu weights), and the group rate at
which the mean recurrent drive alone reaches threshold
(mean drive = sum_w * rate * tau_s; threshold gap = v_th - v_rst).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.malecns import load  # noqa: E402
from sim.lif_numpy import P  # noqa: E402
from scripts.ckpt5_stageA import SCALE  # noqa: E402

D = ROOT / 'data'
VTH = (P['v_th'] - P['v_rst']) * 1e3
TS = P.get('tau_syn', 0.005)


def report(tag, Wmv, g, gaba):
    sub = Wmv[g][:, g]
    self_in = np.asarray(sub.sum(axis=1)).ravel()
    rows = Wmv[g]
    exc = np.asarray(rows.multiply(rows > 0).sum(axis=1)).ravel()
    inh = -np.asarray(rows.multiply(rows < 0).sum(axis=1)).ravel()
    n = g.sum(); nnz = (sub != 0).sum()
    print(f'[{tag}] n={n} within-group edges={nnz} ({nnz / (n * (n - 1)):.0%} of pairs)')
    print(f'  self input per neuron: {self_in.mean():.1f} mV per group volley; '
          f'{self_in.sum() / exc.sum():.0%} of its excitatory input; total exc {exc.mean():.0f} mV, inh {inh.mean():.0f} mV')
    print(f'  group rate for self-drive to reach threshold ({VTH:.0f} mV): '
          f'{VTH / (self_in.mean() * TS):.1f} Hz')


def main():
    nt = pd.read_feather(D / 'malecns' / 'body-neurotransmitters-male-cns-v1.0.feather')
    x = nt[nt.cell_type == 'lLN1_bc']
    print('MaleCNS lLN1_bc:', len(x), 'consensus', x.consensus_nt.value_counts().to_dict(),
          'ground_truth', x.ground_truth.value_counts().to_dict(),
          f'per-neuron conf {x.predicted_nt_confidence.min():.2f}-{x.predicted_nt_confidence.max():.2f}')
    a = pd.read_csv(D / 'flywire_annot' / 'annot.tsv', sep='\t', low_memory=False)
    f = a[a.cell_type == 'lLN1_bc']
    print('FlyWire lLN1_bc:', len(f), 'top_nt', f.top_nt.value_counts().to_dict(),
          f'mean conf {f.top_nt_conf.mean():.2f}', 'known', f.known_nt.dropna().unique().tolist(),
          f.known_nt_source.dropna().unique().tolist())

    meta, W1 = load(w_syn=1.0, monoamine='removed')
    g = (meta.type == 'lLN1_bc').values
    gaba = ((meta['class'] == 'ALLN') & (meta.nt == 'gaba')).values
    report(f'MaleCNS x{SCALE}', (W1 * (P['w_syn'] * SCALE * 1e3)).tocsr(), g, gaba)
    print('  raw within-group synapses:', int(abs(W1.tocsr()[g][:, g]).sum()))

    comp = pd.read_csv(D / 'flywire_v783' / 'Completeness_783.csv', index_col=0)
    pos = {b: i for i, b in enumerate(comp.index.to_numpy(np.int64))}
    con = pd.read_parquet(D / 'flywire_v783' / 'Connectivity_783.parquet')
    Wf = sparse.csr_matrix((con['Excitatory x Connectivity'].to_numpy(float) * P['w_syn'] * 1e3,
                            (con['Postsynaptic_Index'], con['Presynaptic_Index'])), shape=(len(pos),) * 2)
    def fm(sel):
        m = np.zeros(len(pos), bool); m[[pos[r] for r in a.root_id[sel] if r in pos]] = True; return m
    gf = fm(a.cell_type == 'lLN1_bc')
    report('FlyWire (Shiu weights)', Wf, gf, fm((a.cell_class == 'ALLN') & (a.top_nt == 'gaba')))
    print('  raw within-group synapses:', int(abs(Wf[gf][:, gf]).sum() / (P['w_syn'] * 1e3)))


if __name__ == '__main__':
    main()
