"""CPU checks for sim/lif_torch_fast.py (Week 4, T5a): the sync-free step.

  - parity: fast == reference (sim/lif_torch), CPU float64, rng='numpy':
    identical counts AND identical t/i records, for options off, all options
    (gap + APL + presyn + t_stim) and blocks (+ gap + APL + presyn).
  - rng='torch': fast == reference, same checks (both draw the same
    torch.rand calls from the same generator, so they are expected identical).
  - zero syncs: a TorchDispatchMode over the steady loop (rng='torch', all
    options) counts the ops listed in SYNC_OPS plus aten::index / index_put
    calls with a bool-mask index (data-dependent shape); all must be 0. A
    positive control shows the same counter sees the reference's syncs. Every
    step must also issue the same op sequence with the same output shapes.
  - overflow: a tiny spike_cap / syn_cap falls back to the reference with a
    warning and returns reference-identical results.
  - resume: fast split == fast single (and reference state -> fast resume),
    split inside both blocks with spikes in flight, as in test_resume_cpu.py.
"""
import warnings

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sim import lif_torch, lif_torch_fast
from synth import make_network, params

KW = dict(device='cpu', dtype=torch.float64, record=True)
# Caps for the tests (the default caps are spike_cap = n = 300 and syn_cap ~ nnz,
# which only cost time here). On these runs the busiest step has 4 spikes and
# gathers 67 synapses, so neither cap is reached; test_no_fallback_at_default_caps
# covers the defaults and test_overflow_fallback the caps being exceeded.
CAPS = dict(spike_cap=32, syn_cap=512)
T_RUN, T_SPLIT = 0.2, 0.1
SYNC_OPS = {'aten::_local_scalar_dense', 'aten::nonzero', 'aten::item',
            'aten::is_nonzero', 'aten::masked_select'}


@pytest.fixture(scope='module')
def net():
    return make_network()


_REF = {}


def _ref(net, name, rng):
    """Reference result for a case, computed once per module."""
    if (name, rng) not in _REF:
        W, exc, kw, _ = _case(net, name)
        _REF[name, rng] = lif_torch.run_trial(W, exc, rng=rng, **kw, **KW)
    return _REF[name, rng]


def _blocks(net):
    s = net['sens']
    # two overlapping blocks; s[8:12] driven by both during 0.06-0.12 s; T_SPLIT inside both
    return [(0.0, 0.12, s[:12], 200.0), (0.06, 0.2, s[8:], 250.0)]


def _case(net, name):
    """(W, exc_idx, kwargs, min non-sensory spikes): the test_parity_cpu cases."""
    if name == 'off':
        return net['W'], net['sens'], dict(params=params(T_RUN), seed=0), 200
    if name == 'all':
        return net['W_ps'], net['sens'], dict(
            r_poi=250.0, params=params(T_RUN), seed=1, gap=net['gap'], apl=net['apl'],
            presyn=net['presyn'], t_stim=0.15), 150
    return net['W_ps'], [], dict(params=params(T_RUN), seed=3, blocks=_blocks(net),
                                 gap=net['gap'], apl=net['apl'], presyn=net['presyn']), 150


def _same(ref, fast, n_skip, min_central):
    c, cf = np.asarray(ref['counts']), np.asarray(fast['counts'])
    assert int(c[n_skip:].sum()) >= min_central, c[n_skip:].sum()
    diff = np.flatnonzero(c != cf)
    assert diff.size == 0, (f'{diff.size} neurons differ; first {diff[:10].tolist()}: '
                            f'ref {c[diff[:10]].tolist()} fast {cf[diff[:10]].tolist()}')
    assert len(ref['t']) == len(fast['t']), (len(ref['t']), len(fast['t']))
    assert fast['t'].dtype == ref['t'].dtype and fast['i'].dtype == ref['i'].dtype
    np.testing.assert_array_equal(fast['t'], ref['t'])
    np.testing.assert_array_equal(fast['i'], ref['i'])


@pytest.mark.parametrize('rng', ['numpy', 'torch'])
@pytest.mark.parametrize('name', ['off', 'all', 'blocks'])
def test_parity(net, name, rng):
    W, exc, kw, min_central = _case(net, name)
    ref = _ref(net, name, rng)
    fast = lif_torch_fast.run_trial(W, exc, rng=rng, **kw, **KW, **CAPS)
    assert fast['fallback'] is False
    _same(ref, fast, len(net['sens']), min_central)


class _OpCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.syncs, self.ops = {}, []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = func._schema.name
        bad = name in SYNC_OPS
        if name in ('aten::index', 'aten::index_put', 'aten::index_put_', 'aten::_index_put_impl_'):
            ix = args[1] if len(args) > 1 else ()
            bad |= any(isinstance(x, torch.Tensor) and x.dtype == torch.bool
                       for x in (ix or ()))
        if bad:
            self.syncs[name] = self.syncs.get(name, 0) + 1
        out = func(*args, **kwargs)
        shapes = tuple(tuple(o.shape) for o in (out if isinstance(out, (tuple, list)) else (out,))
                       if isinstance(o, torch.Tensor))
        self.ops.append((str(func), shapes))
        return out


def test_zero_syncs_per_step(net):
    W, exc, kw, _ = _case(net, 'all')
    kw = dict(kw, t_stim=None)            # drive on in every step: one static branch set
    sim = lif_torch_fast._prepare(W, exc, rng='torch', bin_steps=50, **kw, **KW, **CAPS)
    for t in range(200):                  # warm up into activity
        sim.advance(t)
    per_step = []
    cnt = _OpCounter()
    with cnt:
        for t in range(200, 400):
            k = len(cnt.ops)
            sim.advance(t)
            per_step.append(cnt.ops[k:])
    assert cnt.syncs == {}, cnt.syncs
    assert all(p == per_step[0] for p in per_step), 'op sequence / shapes vary between steps'
    assert int(sim.counts[len(net['sens']):].sum()) > 0
    # positive control: the same counter sees the reference's syncs
    ref = _OpCounter()
    with ref:
        lif_torch.run_trial(W, exc, rng='torch', **dict(kw, params=params(0.01)), **KW)
    assert ref.syncs.get('aten::nonzero', 0) > 0 and ref.syncs.get('aten::_local_scalar_dense', 0) > 0, ref.syncs


@pytest.mark.parametrize('cap', [dict(spike_cap=1), dict(syn_cap=5)])
def test_overflow_fallback(net, cap):
    W, exc, kw, min_central = _case(net, 'blocks')
    ref = _ref(net, 'blocks', 'numpy')
    n0 = lif_torch_fast.FALLBACKS
    with pytest.warns(RuntimeWarning, match='exceeded'):
        fast = lif_torch_fast.run_trial(W, exc, rng='numpy', **kw, **KW, **cap)
    assert fast['fallback'] is True
    assert lif_torch_fast.FALLBACKS == n0 + 1
    _same(ref, fast, len(net['sens']), min_central)


def test_no_fallback_at_default_caps(net):
    W, exc, kw, _ = _case(net, 'off')
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        out = lif_torch_fast.run_trial(W, exc, rng='torch', **kw, **KW)
    assert out['fallback'] is False


def _run(sim, net, t_run, **extra):
    W, exc, kw, _ = _case(net, 'blocks')
    kw = dict(kw, params=params(t_run))
    if sim is lif_torch_fast:
        extra = dict(CAPS, **extra)
    return sim.run_trial(W, exc, rng='numpy', **kw, **KW, **extra)


def _check_resume(net, single, a, b):
    dt = params()['dt']
    st = a['state']
    assert int(st['t']) == int(round(T_SPLIT / dt))
    assert any(int(round(b0 / dt)) <= st['t'] < int(round(b1 / dt)) for b0, b1, _, _ in _blocks(net))
    assert sum(len(x) for x in st['ring']) > 0, 'delay ring empty at the split point'
    c = np.asarray(single['counts'])
    assert int(c[len(net['sens']):].sum()) >= 150
    np.testing.assert_array_equal(np.asarray(b['counts']), c)
    np.testing.assert_array_equal(np.asarray(b['counts']), np.asarray(a['counts'])
                                  + np.bincount(np.asarray(b['i']), minlength=len(c)))
    np.testing.assert_array_equal(np.concatenate([a['t'], b['t']]), single['t'])
    np.testing.assert_array_equal(np.concatenate([a['i'], b['i']]), single['i'])


def test_resume_fast(net):
    single = _run(lif_torch_fast, net, T_RUN)
    a = _run(lif_torch_fast, net, T_SPLIT, return_state=True)
    b = _run(lif_torch_fast, net, T_RUN, state=a['state'])
    assert not (a['fallback'] or b['fallback'] or single['fallback'])
    _check_resume(net, single, a, b)
    # the state has the reference's keys and formats: reference -> fast resume
    a_ref = _run(lif_torch, net, T_SPLIT, return_state=True)
    assert set(a_ref['state']) == set(a['state'])
    b2 = _run(lif_torch_fast, net, T_RUN, state=a_ref['state'])
    _check_resume(net, single, a_ref, b2)
