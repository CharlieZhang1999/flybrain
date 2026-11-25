"""CPU checks for sim/lif_torch_batch.py (Week 5, P1): B trials in one batch.

  - parity (A1): B = 3, three seeds and three different blocks lists per batch,
    for options off, all options (gap + APL + presyn) and overlapping blocks
    (+ all options), CPU float64: every batched trial's counts and t/i records
    equal lif_torch.run_trial run alone with that seed and blocks. rng='numpy'
    (A1) and rng='torch' (both draw the same numbers from per-trial
    generators, so they are expected identical too). Plus the exc_idx / t_stim
    drive (blocks None) and binning.
  - slot / batch-size invariance: a trial gives the same records alone (B = 1)
    and in slot 2 of B = 3.
  - draws='always': the stream depends only on the trial's seed and blocks
    (same result alone and in a batch); with one always-on drive it equals the
    reference.
  - zero syncs: TorchDispatchMode over the steady loop (rng='torch', all
    options, recording + binning): no sync ops, no bool-mask indexing, and the
    same op sequence and shapes every step.
  - overflow: one trial (last slot) over spike_cap is rerun alone (warning, fallback=True),
    the other trials are kept; all equal the reference.
"""
import warnings

import numpy as np
import pytest
import torch

from sim import lif_torch, lif_torch_batch
from synth import make_network, params
from test_fast_cpu import SYNC_OPS, _OpCounter, _blocks, _same

KW = dict(device='cpu', dtype=torch.float64, record=True)
CAPS = dict(spike_cap=32, syn_cap=512)       # as tests/test_fast_cpu.py (not reached here)
T_RUN = 0.2


@pytest.fixture(scope='module')
def net():
    return make_network()


def _blocks_lists(net, name):
    s = net['sens']
    if name == 'off':
        return [[(0.0, 0.2, s, 150.0)],
                [(0.0, 0.1, s[:10], 300.0), (0.1, 0.2, s[10:], 200.0)],
                [(0.02, 0.18, s[5:], 250.0)]]
    if name == 'all':
        return [[(0.0, 0.15, s, 250.0)],
                [(0.0, 0.08, s[10:], 300.0), (0.08, 0.2, s[:10], 200.0)],
                [(0.05, 0.2, s[3:17], 280.0)]]
    return [_blocks(net),                                                   # overlapping blocks
            [(0.0, 0.15, s[:15], 220.0), (0.05, 0.2, s[5:], 180.0), (0.1, 0.12, s[:4], 400.0)],
            [(0.03, 0.2, s, 150.0), (0.03, 0.2, s[::2], 100.0)]]


def _case(net, name):
    """(W, shared kwargs, seeds, blocks lists, min non-sensory spikes per trial)."""
    if name == 'off':
        return net['W'], dict(params=params(T_RUN)), [0, 1, 2], _blocks_lists(net, name), 100
    opts = dict(params=params(T_RUN), gap=net['gap'], apl=net['apl'], presyn=net['presyn'])
    if name == 'all':
        return net['W_ps'], opts, [3, 4, 5], _blocks_lists(net, name), 50
    return net['W_ps'], opts, [6, 7, 8], _blocks_lists(net, name), 50


def _ref(W, kw, seed, blocks, rng, exc=(), **extra):
    return lif_torch.run_trial(W, exc, seed=seed, blocks=blocks, rng=rng, **kw, **KW, **extra)


@pytest.mark.parametrize('rng', ['numpy', 'torch'])
@pytest.mark.parametrize('name', ['off', 'all', 'overlap'])
def test_parity_batch(net, name, rng):
    W, kw, seeds, bl, min_central = _case(net, name)
    out = lif_torch_batch.run_batch(W, seeds=seeds, blocks_list=bl, rng=rng, **kw, **KW, **CAPS)
    assert len(out) == 3
    for b in range(3):
        assert out[b]['fallback'] is False
        _same(_ref(W, kw, seeds[b], bl[b], rng), out[b], len(net['sens']), min_central)
    # the three trials really differ
    assert not np.array_equal(out[0]['counts'], out[1]['counts'])


@pytest.mark.parametrize('rng', ['numpy', 'torch'])
def test_parity_exc_drive_and_bins(net, rng):
    """blocks None: exc_idx / r_poi / t_stim drive (one trial with blocks), binning."""
    W = net['W_ps']
    kw = dict(params=params(T_RUN), gap=net['gap'], apl=net['apl'], presyn=net['presyn'],
              r_poi=250.0, t_stim=0.15)
    bl = [None, _blocks(net), None]
    seeds = [1, 2, 9]
    out = lif_torch_batch.run_batch(W, seeds=seeds, blocks_list=bl, exc_idx=net['sens'], rng=rng,
                                    bin_steps=50, **kw, **KW, **CAPS)
    for b in range(3):
        ref = _ref(W, kw, seeds[b], bl[b], rng, exc=net['sens'], bin_steps=50)
        _same(ref, out[b], len(net['sens']), 50)
        np.testing.assert_array_equal(out[b]['binned'], ref['binned'])
    groups = [net['kc'], net['gap_idx']]
    out = lif_torch_batch.run_batch(W, seeds=seeds, blocks_list=bl, exc_idx=net['sens'], rng=rng,
                                    bin_steps=50, bin_groups=groups, **kw, **KW, **CAPS)
    for b in range(3):
        ref = _ref(W, kw, seeds[b], bl[b], rng, exc=net['sens'], bin_steps=50, bin_groups=groups)
        np.testing.assert_array_equal(out[b]['binned'], ref['binned'])


@pytest.mark.parametrize('draws', ['active', 'always'])
def test_batch_size_and_slot_invariance(net, draws):
    W, kw, seeds, bl, _ = _case(net, 'overlap')
    full = lif_torch_batch.run_batch(W, seeds=seeds, blocks_list=bl, rng='torch', draws=draws,
                                     **kw, **KW, **CAPS)
    alone = lif_torch_batch.run_batch(W, seeds=seeds[2:], blocks_list=bl[2:], rng='torch',
                                      draws=draws, **kw, **KW, **CAPS)[0]
    _same(alone, full[2], len(net['sens']), 50)
    # slot 0 of a reordered batch
    rev = lif_torch_batch.run_batch(W, seeds=seeds[::-1], blocks_list=bl[::-1], rng='torch',
                                    draws=draws, **kw, **KW, **CAPS)
    _same(full[2], rev[0], len(net['sens']), 50)


def test_draws_always_equals_reference_with_always_on_drive(net):
    s = net['sens']
    W = net['W']
    bl = [[(0.0, 0.2, s, 200.0)], [(0.0, 0.2, s[::2], 300.0)]]
    out = lif_torch_batch.run_batch(W, seeds=[4, 5], blocks_list=bl, rng='torch', draws='always',
                                    params=params(T_RUN), **KW, **CAPS)
    for b in range(2):
        _same(_ref(W, dict(params=params(T_RUN)), [4, 5][b], bl[b], 'torch'), out[b],
              len(net['sens']), 50)


def test_zero_syncs_per_step(net):
    W, kw, seeds, bl, _ = _case(net, 'overlap')
    bl = [[(0.0, 1.0, x[2], x[3]) for x in b] for b in bl]   # every drive on: one key
    s = lif_torch_batch._prepare(W, seeds, bl, rng='torch', bin_steps=50, **dict(kw, params=params(0.04)),
                                 **KW, **CAPS)
    for t in range(200):
        s.advance(t)
    per_step = []
    cnt = _OpCounter()
    with cnt:
        for t in range(200, 400):
            k = len(cnt.ops)
            s.advance(t)
            per_step.append(cnt.ops[k:])
    assert cnt.syncs == {}, cnt.syncs
    assert set(SYNC_OPS).isdisjoint(cnt.syncs)
    assert all(p == per_step[0] for p in per_step), 'op sequence / shapes vary between steps'
    assert int(s.counts[:, len(net['sens']):].sum()) > 0


def test_overflow_reruns_only_that_trial(net):
    s = net['sens']
    W = net['W']
    kw = dict(params=params(T_RUN))
    # the overflowing trial is in the last slot: a trial after an overflowing one can lose
    # spikes from the flattened list and is then flagged too (see the module docstring)
    bl = [[(0.0, 0.2, s[:2], 50.0)], [(0.0, 0.2, s[:2], 60.0)], [(0.0, 0.2, s, 600.0)]]
    seeds = [0, 1, 2]
    caps = dict(spike_cap=4, syn_cap=512)
    refs = [_ref(W, kw, seeds[b], bl[b], 'numpy') for b in range(3)]
    busiest = [int(np.bincount(r['t']).max()) if len(r['t']) else 0 for r in refs]
    assert busiest[0] <= 4 and busiest[1] <= 4 and busiest[2] > 4, busiest
    n0 = lif_torch_batch.FALLBACKS
    with pytest.warns(RuntimeWarning, match='trial 2'):
        out = lif_torch_batch.run_batch(W, seeds=seeds, blocks_list=bl, rng='numpy', **kw, **KW, **caps)
    assert [o['fallback'] for o in out] == [False, False, True]
    assert lif_torch_batch.FALLBACKS == n0 + 1
    for b in range(3):
        _same(refs[b], out[b], 0, 0)


def test_no_record_returns_counts(net):
    W, kw, seeds, bl, _ = _case(net, 'off')
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        out = lif_torch_batch.run_batch(W, seeds=seeds, blocks_list=bl, rng='numpy', device='cpu',
                                        dtype=torch.float64, **kw)
    ref = lif_torch.run_trial(W, [], seed=seeds[1], blocks=bl[1], rng='numpy', device='cpu',
                              dtype=torch.float64, **kw)
    assert isinstance(out[1], np.ndarray) and out[1].dtype == ref.dtype
    np.testing.assert_array_equal(out[1], ref)


@pytest.mark.parametrize('name', ['off', 'overlap'])
def test_state_bitwise(net, name):
    """Final v and g of every batched trial equal the reference's bitwise (catches
    differences too small to change a spike count over 0.2 s)."""
    W, kw, seeds, bl, _ = _case(net, name)
    s = lif_torch_batch._prepare(W, seeds, bl, rng='numpy', **kw, **KW, **CAPS)
    lif_torch_batch.eager_loop(s)
    for b in range(3):
        st = _ref(W, kw, seeds[b], bl[b], 'numpy', return_state=True)['state']
        assert torch.equal(s.v[b], st['v']) and torch.equal(s.g[b], st['g'])
        assert torch.equal(s.last[b].long(), st['last'])
