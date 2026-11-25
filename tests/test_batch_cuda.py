"""CUDA checks for sim/lif_torch_batch.py (Week 5, P1). Skipped without CUDA
(CI cpu-tests skips them; run locally on the GPU box).

  - sync check: the batched steady loop (rng='torch', gap + APL + presyn,
    different blocks per trial, recording + binning) under
    torch.cuda.set_sync_debug_mode('error') raises nothing; positive control:
    the reference raises under the same mode.
  - graph replay (lif_torch_batch.graph_loop) == eager, same seeds, different
    overlapping blocks per trial: identical counts, records, bins and state.
    draws='active' captures one graph per set of drawing (trial, drive) pairs;
    draws='always' captures exactly one graph for the same blocks.
  - batch-size / slot invariance in graph mode: a trial alone (B = 1) and in
    slot 2 of B = 3 gives identical counts and records.
  - batched graph vs reference on CUDA (rng='torch', float64): same draws, so
    identical counts are expected (observed; float atomics make it not
    guaranteed).
"""
import numpy as np
import pytest
import torch

from sim import lif_torch, lif_torch_batch
from synth import make_network, params
from test_batch_cpu import CAPS, _case

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
KW = dict(device='cuda', dtype=torch.float64, record=True, bin_steps=50, rng='torch')


@pytest.fixture(scope='module')
def net():
    return make_network()


def _prep(net, name, draws='active', sl=slice(None)):
    W, kw, seeds, bl, _ = _case(net, name)
    return lif_torch_batch._prepare(W, seeds[sl], bl[sl], draws=draws, **kw, **KW, **CAPS)


def test_sync_debug_steady_loop(net):
    s = _prep(net, 'overlap')
    lif_torch_batch.eager_loop(s, stop=200)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode('error')
    try:
        lif_torch_batch.eager_loop(s, stop=400)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    assert int(s.counts[:, len(net['sens']):].sum()) > 0
    assert not bool(s.ovf.any())
    W, kw, seeds, bl, _ = _case(net, 'overlap')
    torch.cuda.set_sync_debug_mode('error')
    try:
        with pytest.raises(RuntimeError, match='synchroniz'):
            lif_torch.run_trial(W, [], seed=seeds[0], blocks=bl[0], rng='torch', device='cuda',
                                dtype=torch.float64, **dict(kw, params=params(0.001)))
    finally:
        torch.cuda.set_sync_debug_mode(0)


@pytest.mark.parametrize('draws', ['active', 'always'])
def test_graph_equals_eager(net, draws):
    e = _prep(net, 'overlap', draws)
    lif_torch_batch.eager_loop(e)
    g = _prep(net, 'overlap', draws)
    graphs = lif_torch_batch.graph_loop(g)
    torch.cuda.synchronize()
    if draws == 'always':
        assert len(graphs) == 1
    else:
        assert len(graphs) == len(set(g.seg_keys)) > 3
    assert not bool(e.ovf.any()) and not bool(g.ovf.any())
    assert int(e.counts[:, len(net['sens']):].sum(1).min()) >= 50
    assert torch.equal(e.counts, g.counts)
    assert torch.equal(e.rec, g.rec)
    assert torch.equal(e.binned, g.binned)
    assert torch.equal(e.v, g.v) and torch.equal(e.g, g.g)


def test_graph_batch_size_and_slot_invariance(net):
    full = _prep(net, 'overlap')
    lif_torch_batch.graph_loop(full)
    alone = _prep(net, 'overlap', sl=slice(2, 3))
    lif_torch_batch.graph_loop(alone)
    torch.cuda.synchronize()
    assert torch.equal(full.counts[2], alone.counts[0])
    assert torch.equal(full.rec[:, 2], alone.rec[:, 0])


def test_graph_vs_reference_cuda(net):
    W, kw, seeds, bl, _ = _case(net, 'overlap')
    out = lif_torch_batch.run_batch(W, seeds=seeds, blocks_list=bl, device='cuda',
                                    dtype=torch.float64, rng='torch', record=True, **kw, **CAPS)
    for b in range(3):
        ref = lif_torch.run_trial(W, [], seed=seeds[b], blocks=bl[b], rng='torch', device='cuda',
                                  dtype=torch.float64, record=True, **kw)
        assert out[b]['fallback'] is False
        np.testing.assert_array_equal(out[b]['counts'], ref['counts'])
        np.testing.assert_array_equal(out[b]['t'], ref['t'])
        np.testing.assert_array_equal(out[b]['i'], ref['i'])
