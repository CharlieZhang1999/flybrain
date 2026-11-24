"""scripts/w5_hidden_state_ceiling.py: data-loading shapes, folds and model I/O on a synthetic
dataset (no data/, no sklearn)."""
import numpy as np
import pytest

from scripts import w5_hidden_state_baseline as bl
from scripts import w5_hidden_state_ceiling as ce

N_EP, N_NEUR = 6, 40


@pytest.fixture
def synth_ds(tmp_path, monkeypatch):
    (tmp_path / 'activity').mkdir()
    (tmp_path / 'labels').mkdir()
    rng = np.random.default_rng(0)
    for k in range(N_EP):
        sid = np.repeat(np.array([k % 4, (k + 1) % 4], np.int16), 200)      # two 2 s segments, 10 ms bins
        np.savez(tmp_path / 'labels' / f'ep{k:03d}.npz', state_id=sid, bin_s=0.01)
        n = 500
        t = np.sort(rng.integers(0, 40000, n)).astype(np.int32)          # 1e-4 s steps, 4 s
        i = rng.integers(0, N_NEUR, n).astype(np.int32)
        i[i == 3] = 4                                                      # neuron 3 = masked, never spikes
        i[i == 7] = 8                                                      # neuron 7 = active but not an input
        np.savez(tmp_path / 'activity' / f'ep{k:03d}_calib.npz', t=t, i=i)
    monkeypatch.setattr(bl, 'DS', tmp_path)
    monkeypatch.setattr(bl, 'N_EP', N_EP)
    return tmp_path


def test_loader_shapes(synth_ds):
    y = bl.labels()
    assert y.shape == (N_EP, bl.NB) == (N_EP, 80)
    assert (y[:, :40] == (np.arange(N_EP) % 4)[:, None]).all()
    inp = np.setdiff1d(np.arange(N_NEUR), [3, 7])
    X = bl.brain_inputs('calib', inp, np.array([3]), N_NEUR)
    assert X.shape[:2] == (N_EP, 80) and X.dtype == np.float32
    assert X.shape[2] == len(inp)                                  # every input neuron spikes once here
    Ftr, Fte = ce.prep(X, np.arange(4), np.arange(4, 6), 'full')
    assert Ftr.shape[:2] == (4, 80) and Fte.shape[:2] == (2, 80) and Ftr.shape[2] == Fte.shape[2] <= X.shape[2]
    assert np.allclose(Ftr.reshape(-1, Ftr.shape[2]).mean(0), 0, atol=1e-4)
    t, v = ce.segment_vote(y.astype(np.int64), y)
    assert (t == v).all() and len(t) == 2 * N_EP
    assert ce.summarize(y.astype(np.int64), y)['seg_acc'] == 1.0


def test_same_folds_as_baseline():
    f = bl.folds(np.arange(120), bl.N_FOLD, bl.SEED)
    assert len(f) == 5 and sorted(np.concatenate(f).tolist()) == list(range(120))
    assert all(len(x) == 24 for x in f)


@pytest.mark.parametrize('model', ['mlp', 'gru_causal', 'gru_bi', 'tf'])
def test_model_output_shapes(model, monkeypatch):
    monkeypatch.setattr(ce, "MAX_EP", 5)
    torch = pytest.importorskip('torch')
    hp = dict(ce.GRID[model][0], d=16, h=16) if model == 'mlp' else dict(ce.GRID[model][0], d=16)
    net = ce.build(model, 30, hp)
    out = net(torch.zeros(3, 80, 30))
    assert out.shape == (3, 80, 4)
    if model == 'gru_causal':                                       # causal: future bins don't change the past
        x = torch.randn(1, 80, 30)
        net.eval()
        a = net(x)
        x[:, 50:] += 5
        assert torch.allclose(a[:, :50], net(x)[:, :50], atol=1e-6)
    rng = np.random.default_rng(1)
    F = rng.standard_normal((4, 80, 30)).astype(np.float32)
    yy = np.repeat(rng.integers(0, 4, (4, 2)), 40, axis=1)
    ba, ep, _ = ce.train(model, dict(hp, in_drop=0.0, drop=0.0), F, yy, F, yy, seed=0)
    assert 0 <= ba <= 1 and ep >= 1
