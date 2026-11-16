"""Week 3 shared setup: the calibrated MaleCNS model from Week 2 (w2d4 'calib')."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.lesion import select  # noqa: E402
from sim.malecns import load  # noqa: E402
from sim.gap import build_gap  # noqa: E402
from scripts.c4_core_gain import Runner, DT  # noqa: E402
from scripts.ckpt5_stageA import ACH_LN, PN  # noqa: E402
from scripts.w2d3_gap_malecns import cut_ln_pn  # noqa: E402
from scripts.w2d4_word_trials import shuffle_W, GAP_KW  # noqa: E402


def calib_model(config='calib'):
    R = Runner()
    ln, pn = select(R.meta, ACH_LN), select(R.meta, PN)
    _, Wsyn = load(w_syn=1.0, monoamine='removed')
    W = cut_ln_pn(R.Wr, ln, pn)
    gap, _ = build_gap(Wsyn, ln, pn, v_spike=0.002, **GAP_KW)
    if config == 'shuffle':
        W = shuffle_W(W)
    return R, W, gap


def presyn_model(K, config='calib', mirror=False):
    """Calibrated model + presynaptic modulation of sensory terminals (sim/presyn.py).
    mirror=True also applies the right-JO mirror calibration (sim/jo_mirror.py)."""
    from sim.presyn import build_presyn, sensory_mask_malecns
    R, W, gap = calib_model(config)
    W2, ps, info = build_presyn(W, sensory_mask_malecns(R.meta), K)
    if mirror:
        from sim.jo_mirror import mirror_jo
        W2, minfo = mirror_jo(W2, R.meta)
        info = dict(info, mirror={k: v for k, v in minfo.items() if k != 'templates'})
    return R, W2, gap, ps, info
