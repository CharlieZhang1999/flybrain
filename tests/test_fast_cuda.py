"""CUDA checks for sim/lif_torch_fast.py (Week 4, T5b). Skipped without CUDA
(CI cpu-tests skips them; run locally on the GPU box).

  - sync check: the steady loop (rng='torch', gap + APL + presyn, drive on,
    recording + binning) under torch.cuda.set_sync_debug_mode('error') raises
    nothing; positive control: the reference raises under the same mode.
  - graph replay (bench/w4_graph.graph_trial) == eager, same seed: identical
    counts and spike records, for all options and for overlapping blocks (one
    graph per active-drive combination: 3 here).
"""
import sys
from pathlib import Path

import pytest
import torch

from sim import lif_torch, lif_torch_fast
from synth import make_network, params
from test_fast_cpu import CAPS, _case

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
KW = dict(device='cuda', dtype=torch.float64, record=True, bin_steps=50, rng='torch')


@pytest.fixture(scope='module')
def net():
    return make_network()


@pytest.fixture(scope='module')
def w4_graph():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
    import w4_graph
    return w4_graph


def _prep(net, name):
    W, exc, kw, _ = _case(net, name)
    return lif_torch_fast._prepare(W, exc, **kw, **KW, **CAPS)


def test_sync_debug_steady_loop(net):
    W, exc, kw, _ = _case(net, 'all')
    kw = dict(kw, t_stim=None)
    s = lif_torch_fast._prepare(W, exc, **kw, **KW, **CAPS)
    for t in range(200):
        s.advance(t)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode('error')
    try:
        for t in range(200, 400):
            s.advance(t)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    assert int(s.counts[len(net['sens']):].sum()) > 0
    assert not bool(s.ovf)
    torch.cuda.set_sync_debug_mode('error')
    try:
        with pytest.raises(RuntimeError, match='synchroniz'):
            lif_torch.run_trial(W, exc, rng='torch', device='cuda', dtype=torch.float64,
                                **dict(kw, params=params(0.001)))
    finally:
        torch.cuda.set_sync_debug_mode(0)


@pytest.mark.parametrize('name, n_graphs', [('all', 2), ('blocks', 3)])
def test_graph_equals_eager(net, w4_graph, name, n_graphs):
    e = _prep(net, name)
    w4_graph.eager_trial(e)
    g = _prep(net, name)
    graphs = w4_graph.graph_trial(g)
    torch.cuda.synchronize()
    assert len(graphs) == n_graphs      # 'all': drive on, then off after t_stim
    assert not bool(e.ovf) and not bool(g.ovf)
    assert int(e.counts[len(net['sens']):].sum()) >= 150
    assert torch.equal(e.counts, g.counts)
    assert torch.equal(e.rec, g.rec)
    assert torch.equal(e.binned, g.binned)
    assert torch.equal(e.v, g.v) and torch.equal(e.g, g.g)
