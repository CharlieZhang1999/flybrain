#!/usr/bin/env python
"""C0: what happens to synaptic input that lands during refractoriness?

Toy: three Shiu LIF neurons A, B, T (exact Shiu equations / constants).
  A spikes at step 0.
  A -> T  (strong)  : T spikes once A's input arrives, then is refractory 2.2 ms.
  A -> B  (strong)  : B spikes a bit later than T.
  B -> T  (w_test)  : arrives 1.8 ms after B's spike, timed to land while T is
                      still refractory.
If refractory-time input is applied (and g is frozen, not decaying), T fires
again right after its refractory period ends. If it is blocked, T stays silent.
Run in Brian2 (reference) and in our NumPy simulator; compare spike times and,
for Brian2, the g trace of T.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party' / 'Drosophila_brain_model'))
from sim.lif_numpy import run_trial, P  # noqa: E402

W_AT, W_AB = 200e-3, 200e-3       # volts; B fires with T, its input reaches T at step 44 (refractory)
W_TEST = 100e-3                   # B->T


def brian2_run(w_bt):
    from brian2 import (NeuronGroup, Synapses, StateMonitor, SpikeMonitor, Network,
                        mV, ms, defaultclock, prefs, start_scope)
    from model import default_params as dp
    start_scope()
    prefs.codegen.target = 'numpy'
    defaultclock.dt = 0.1 * ms
    neu = NeuronGroup(3, model=dp['eqs'], method='linear', threshold=dp['eq_th'],
                      reset=dp['eq_rst'], refractory='rfc', namespace=dp)
    neu.v = dp['v_0']; neu.g = 0; neu.rfc = dp['t_rfc']
    # A is kicked once at step 0 exactly like a stimulated neuron (v += w_syn*f_poi
    # in the synapses slot), so both simulators see the same input event.
    from brian2 import SpikeGeneratorGroup
    sg = SpikeGeneratorGroup(1, [0], [0 * ms])
    kick = Synapses(sg, neu, on_pre='v += 68.75*mV'); kick.connect(i=0, j=0)
    neu.rfc[0] = 0 * ms
    syn = Synapses(neu, neu, 'w : volt', on_pre='g += w', delay=dp['t_dly'], namespace=dp)
    syn.connect(i=[0, 0, 1], j=[2, 1, 2])
    syn.w = [W_AT * 1e3 * mV, W_AB * 1e3 * mV, w_bt * 1e3 * mV]
    sm = StateMonitor(neu, ['v', 'g'], record=[2])
    sp = SpikeMonitor(neu)
    net = Network(neu, syn, sm, sp, sg, kick); net.run(10 * ms)
    tr = sp.spike_trains()
    steps = {k: np.round(np.asarray(v / ms) * 10).astype(int).tolist() for k, v in tr.items()}
    return steps, np.asarray(sm.g[0] / mV), np.asarray(sm.v[0] / mV)


def numpy_run(w_bt):
    W = sparse.csc_matrix(([W_AT, W_AB, w_bt], ([2, 1, 2], [0, 0, 1])), shape=(3, 3))
    p = dict(P); p['t_run'] = 0.010
    r = run_trial(W, np.array([0]), r_poi=1e5, params=p, seed=0, t_stim=1e-4, record=True)
    return {k: r['t'][r['i'] == k].tolist() for k in range(3)}


def main():
    for w in [0.0, W_TEST]:
        b_steps, g, v = brian2_run(w)
        n_steps = numpy_run(w)
        print(f'--- B->T weight = {w*1e3:.0f} mV')
        print(f'  Brian2 spike steps: A={b_steps.get(0)}, B={b_steps.get(1)}, T={b_steps.get(2)}')
        print(f'  NumPy  spike steps: A={n_steps[0]}, B={n_steps[1]}, T={n_steps[2]}')
        if w:
            t_spk = b_steps[2][0]
            win = range(max(t_spk - 2, 0), min(t_spk + 40, len(g)))
            print('  Brian2 T trace (step: g mV, v mV):')
            print('   ' + '  '.join(f'{s}:{g[s]:.1f}/{v[s]:.1f}' for s in win if s % 2 == 0))


if __name__ == '__main__':
    main()
