#!/usr/bin/env python
"""Week 2 Days 2-3, part b: gap junctions with charge-matched spikelets.

Part a (w2d3_gap_*.py) gave each partner an instant c * 40 mV jump per spike.
A real spike lasts ~1 ms and the junction plus membrane low-pass filter it
(tau_m = 20 ms), so the charge it moves is about c * 40 mV * (1 ms / 20 ms)
= c * 2 mV. Part b uses v_spike = 2 mV and larger c0 in {0.04, 0.1, 0.2}.
Everything else as w2d3_gap_malecns.py / w2d3_gap_flywire.py.
Usage: w2d3b_gap_charge.py malecns|flywire
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sim.gap as G  # noqa: E402

_orig = G.build_gap
def build_gap_charge(*a, **k):
    gap, info = _orig(*a, **k)
    gap['v_spike'] = 0.002
    info['v_spike_mV'] = 2.0
    return gap, info
G.build_gap = build_gap_charge

if __name__ == '__main__':
    which = sys.argv[1]
    if which == 'malecns':
        import scripts.w2d3_gap_malecns as M
        M.build_gap = build_gap_charge
        M.OUT_NAME = 'w2d3b_gap_runs.csv'
        src = open(M.__file__).read()
        # same main, different c0 list and output file
        M.main.__globals__['build_gap'] = build_gap_charge
        import types
        code = src.replace("for c0 in [0.01, 0.02, 0.04]:", "for c0 in [0.04, 0.1, 0.2]:") \
                  .replace("'w2d3_gap_runs.csv'", "'w2d3b_gap_runs.csv'") \
                  .replace("conds = [('base', R.Wr, None), ('LN->PN chem 0', Wcut, None)]", "conds = []") \
                  .replace("from sim.gap import build_gap, pn_glomeruli", "from sim.gap import pn_glomeruli")
        g = {'__name__': 'w2d3b', '__file__': M.__file__, 'build_gap': build_gap_charge}
        exec(compile(code.replace("if __name__ == '__main__':\n    main()", ""), M.__file__, 'exec'), g)
        g['build_gap'] = build_gap_charge
        g['main']()
    else:
        import scripts.w2d3_gap_flywire as F
        src = open(F.__file__).read()
        code = src.replace("'w2d3_gap_flywire.csv'", "'w2d3b_gap_flywire.csv'") \
                  .replace("conds = [('base', W, None), ('LN->PN chem 0', Wcut, None)]", "conds = []") \
                  .replace("from sim.gap import build_gap, pn_glomeruli", "from sim.gap import pn_glomeruli")
        g = {'__name__': 'w2d3b', '__file__': F.__file__, 'build_gap': build_gap_charge}
        exec(compile(code.replace("if __name__ == '__main__':", "if False:"), F.__file__, 'exec'), g)
        g['build_gap'] = build_gap_charge
        g['main']([0.04, 0.1, 0.2])
