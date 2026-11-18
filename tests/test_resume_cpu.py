"""CPU save/resume parity on the synthetic network (Week 4, T2).

A run to T split at T/2 (first call record=True, return_state=True; second call
state=..., params['t_run']=T) must equal one uninterrupted run to T, spike for
spike. Semantics (docstrings of both simulators): the resumed call returns
cumulative counts (whole trial) but only the new spikes in 't'/'i'; 't' is the
absolute step. All options are on (blocks, gap, APL, presyn) and the split falls
inside both blocks while spikes are in flight (non-empty delay ring).

Inspect the saved state at the split point:  python tests/test_resume_cpu.py
"""
import sys
from pathlib import Path

if __name__ == '__main__':          # under pytest, conftest.py sets the path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch

from sim import lif_numpy, lif_torch
from synth import make_network, params

TORCH_KW = dict(device='cpu', dtype=torch.float64, rng='numpy')
T_RUN, T_SPLIT = 0.2, 0.1
MIN_CENTRAL = 150                  # same activity guard as test_parity_cpu


@pytest.fixture(scope='module')
def net():
    return make_network()


def _blocks(net):
    s = net['sens']
    # two overlapping blocks; T_SPLIT lies inside both
    return [(0.0, 0.12, s[:12], 200.0), (0.06, 0.2, s[8:], 250.0)]


def _kw(net):
    return dict(seed=3, blocks=_blocks(net), gap=net['gap'], apl=net['apl'],
                presyn=net['presyn'], record=True)


def _run(sim, net, t_run, **extra):
    kw = TORCH_KW if sim is lif_torch else {}
    return sim.run_trial(net['W_ps'], [], params=params(t_run), **_kw(net), **kw, **extra)


def _split(sim, net):
    a = _run(sim, net, T_SPLIT, return_state=True)
    b = _run(sim, net, T_RUN, state=a['state'])
    return a, b


def _check_split_point(net, state):
    """The split must be inside a block with spikes in flight."""
    dt = params()['dt']
    t = int(state['t'])
    assert t == int(round(T_SPLIT / dt))
    assert any(int(round(b0 / dt)) <= t < int(round(b1 / dt)) for b0, b1, _, _ in _blocks(net))
    in_flight = sum(len(x) for x in state['ring'])
    assert in_flight > 0, 'delay ring empty at the split point'


def _check_same(single, a, b, n_skip):
    c = np.asarray(single['counts'])
    assert int(c[n_skip:].sum()) >= MIN_CENTRAL, c[n_skip:].sum()
    # spikes before the split happened before the split; the resumed call adds only new ones
    assert np.all(np.asarray(a['t']) < np.asarray(b['t']).min())
    cb = np.asarray(b['counts'])
    diff = np.flatnonzero(cb != c)
    assert diff.size == 0, (f'{diff.size} neurons differ; first {diff[:10].tolist()}: '
                            f'single {c[diff[:10]].tolist()} split {cb[diff[:10]].tolist()}')
    # cumulative counts = counts at split + new spikes
    np.testing.assert_array_equal(cb, np.asarray(a['counts'])
                                  + np.bincount(np.asarray(b['i']), minlength=len(c)))
    t_cat = np.concatenate([a['t'], b['t']]).astype(np.int64)
    i_cat = np.concatenate([a['i'], b['i']]).astype(np.int64)
    assert len(t_cat) == len(single['t']), (len(t_cat), len(single['t']))
    np.testing.assert_array_equal(t_cat, np.asarray(single['t'], dtype=np.int64))
    np.testing.assert_array_equal(i_cat, np.asarray(single['i'], dtype=np.int64))


def test_resume_numpy(net):
    single = _run(lif_numpy, net, T_RUN)
    a, b = _split(lif_numpy, net)
    _check_split_point(net, a['state'])
    _check_same(single, a, b, len(net['sens']))


def test_resume_torch(net):
    single = _run(lif_torch, net, T_RUN)
    a, b = _split(lif_torch, net)
    _check_split_point(net, a['state'])
    _check_same(single, a, b, len(net['sens']))


def test_resume_torch_vs_numpy(net):
    single = _run(lif_numpy, net, T_RUN)
    a, b = _split(lif_torch, net)
    _check_split_point(net, a['state'])
    _check_same(single, a, b, len(net['sens']))


def _summary():
    """Print the saved state at the split point for both simulators."""
    net = make_network()
    dt = params()['dt']
    for sim in (lif_numpy, lif_torch):
        st = _run(sim, net, T_SPLIT, return_state=True)['state']
        t = int(st['t']); dly = len(st['ring'])
        print(f'{sim.__name__}: split at step {t} (t = {t * dt:.4f} s), ring of {dly} slots')
        for b0, b1, ix, r in _blocks(net):
            s0, s1 = int(round(b0 / dt)), int(round(b1 / dt))
            print(f'  block steps [{s0}, {s1}) idx {np.asarray(ix).tolist()} {r} Hz: '
                  f'{"inside" if s0 <= t < s1 else "outside"}')
        for k in range(dly):         # slot (t + k) % dly is delivered at step t + k
            x = np.asarray(st['ring'][(t + k) % dly])
            if len(x):
                print(f'  delivered at step {t + k} (spiked at {t + k - dly}): {x.tolist()}')
        print(f'  in flight: {sum(len(x) for x in st["ring"])} spikes')


if __name__ == '__main__':
    _summary()
