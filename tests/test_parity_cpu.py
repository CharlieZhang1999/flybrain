"""CPU parity: sim/lif_torch.run_trial (float64, rng='numpy') must match
sim/lif_numpy.run_trial spike-for-spike on a small synthetic network.

lif_torch has no state=/return_state= (see its docstring), so there is no
split-run test yet.
"""
import numpy as np
import pytest
import torch

from sim import lif_numpy, lif_torch
from synth import make_network, params

TORCH_KW = dict(device='cpu', dtype=torch.float64, rng='numpy')


@pytest.fixture(scope='module')
def net():
    return make_network()


def _both(W, exc, **kw):
    c_np = lif_numpy.run_trial(W, exc, **kw)
    c_t = lif_torch.run_trial(W, exc, **kw, **TORCH_KW)
    return np.asarray(c_np), np.asarray(c_t)


def _check(c_np, c_t, n_skip, min_central):
    # the network must actually do something beyond the driven neurons,
    # otherwise equality would be trivial
    assert int(c_np[n_skip:].sum()) >= min_central, c_np[n_skip:].sum()
    diff = np.flatnonzero(c_np != c_t)
    assert diff.size == 0, (f'{diff.size} neurons differ; first {diff[:10].tolist()}: '
                            f'numpy {c_np[diff[:10]].tolist()} torch {c_t[diff[:10]].tolist()}')


def test_parity_options_off(net):
    c_np, c_t = _both(net['W'], net['sens'], params=params(0.2), seed=0)
    _check(c_np, c_t, len(net['sens']), 200)


def test_parity_all_options(net):
    c_np, c_t = _both(net['W_ps'], net['sens'], r_poi=250.0, params=params(0.2), seed=1,
                      gap=net['gap'], apl=net['apl'], presyn=net['presyn'], t_stim=0.15)
    _check(c_np, c_t, len(net['sens']), 150)


def test_parity_blocks(net):
    s = net['sens']
    # two overlapping blocks; s[8:12] is driven by both during 0.06-0.12 s
    blocks = [(0.0, 0.12, s[:12], 200.0), (0.06, 0.2, s[8:], 250.0)]
    c_np, c_t = _both(net['W_ps'], [], params=params(0.2), seed=3, blocks=blocks,
                      gap=net['gap'], apl=net['apl'], presyn=net['presyn'])
    _check(c_np, c_t, len(s), 150)
