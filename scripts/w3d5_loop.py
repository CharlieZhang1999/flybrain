#!/usr/bin/env python
"""Week 3 Day 5: run the closed loop. Round 1 uses plan.md's demo words
(vinegar, fruit, sugar, hot, looming); rounds 2-3 are random.
Usage: w3d5_loop.py <real|shuffle> <seed> [provider] [tag]
Output: results/loop/w3d5_loop_<cfg>_<seed>.json"""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from loop.run_loop import run  # noqa: E402

cfg, seed = sys.argv[1], int(sys.argv[2])
prov = sys.argv[3] if len(sys.argv) > 3 else None
r, spikes = run(seed, cfg, word_lists=[['vinegar', 'fruit', 'sugar', 'hot', 'looming']], provider=prov)
for x in r['rounds']:
    print(f"[{cfg} {seed}] round {x['round']}: {'+'.join(x['true'])} -> {x['decoded']} | "
          f"'{x['prompt']}' -> {x['action']} ({x['llm_provider']}) -> code {x['code']} -> "
          f"decoded {x['action_decoded']} p={x['action_p']} ok={x['action_ok']}", flush=True)
tag = sys.argv[4] if len(sys.argv) > 4 else 'w3d5'
json.dump(r, open(ROOT / 'results' / 'loop' / f'{tag}_loop_{cfg}_{seed}.json', 'w'))
import numpy as np  # noqa: E402
np.savez_compressed(ROOT / 'results' / 'loop' / f'{tag}_loop_{cfg}_{seed}_spikes.npz', t=spikes['t'].astype(np.float32), i=spikes['i'].astype(np.int32))
print(f'DONE wall {r["wall_s"]} s')
