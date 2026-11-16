#!/usr/bin/env python
"""Week 3 Day 2: fly -> LLM episodes on the loop's brain model (calibrated MaleCNS +
presynaptic modulation, K = 0.05).
  demo     : plan.md's 'vinegar + yeast + sugar + warm + bright' as
             vinegar, fruit, sugar, hot, looming (looming last: longest tail), seeds 2000-2009
  scenarios: six short episodes chosen to exercise different replies, seed 3000
Usage: w3d2_fly_to_llm.py demo <seed_start> <seed_end> | scenarios [provider]
Output: results/loop/w3d2_fly_to_llm_<part>.jsonl (one JSON object per episode)"""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from loop.episode import fly_to_llm  # noqa: E402

DEMO = ['vinegar', 'fruit', 'sugar', 'hot', 'looming']
SCEN = [['sugar', 'water'], ['bitter', 'mold'], ['vinegar', 'pheromone'],
        ['co2', 'hot'], ['sound', 'wind'], ['fruit', 'looming']]

part = sys.argv[1]
provider = None
if part == 'demo':
    a, b = int(sys.argv[2]), int(sys.argv[3]); provider = sys.argv[4] if len(sys.argv) > 4 else None
    jobs = [(DEMO, s) for s in range(a, b)]; tag = f'demo_{a}_{b}'
else:
    provider = sys.argv[2] if len(sys.argv) > 2 else None
    jobs = [(w, 3000) for w in SCEN]; tag = 'scenarios'
out = ROOT / 'results' / 'loop' / f'w3d2_fly_to_llm_{tag}.jsonl'
with open(out, 'w') as f:
    for words, seed in jobs:
        r = fly_to_llm(words, seed, provider=provider)
        f.write(json.dumps(r) + '\n'); f.flush()
        print(f"seed {seed} {'+'.join(words)} -> {r['decoded']} {r['p_top']} | {r['llm_action']} "
              f"({r['llm_provider']}) tail={r['tail_neurons']} {r['sim_s']} s", flush=True)
print('DONE')
