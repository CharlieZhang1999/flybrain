#!/usr/bin/env python
"""Week 3 Day 7: build docs/week3_report.html from docs/week3_report.src.html,
inlining the loop still (results/figures/loop_still_claude.jpg) and the Claude loop runs (results/loop/w3d6c_*)."""
import base64, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; L = ROOT / 'results' / 'loop'
src = (ROOT / 'docs' / 'week3_report.src.html').read_text()
still = 'data:image/jpeg;base64,' + base64.b64encode((ROOT / 'results/figures/loop_still_claude.jpg').read_bytes()).decode()
d = json.load(open(L / 'w3d6c_loop_real_4000.json'))
loop = dict(t_total=d['t_total'], pop=d['pop_10ms'],
            rounds=[{k: r[k] for k in ('round', 't_start', 't_words', 'decoded', 'prompt', 'action', 'code', 'action_decoded', 'action_p', 'action_ok')} for r in d['rounds']])
runs = []
for f in sorted(L.glob('w3d6c_loop_*[0-9].json')):
    x = json.load(open(f)); rr = x['rounds']
    runs.append(dict(run=('real' if x['cfg'] == 'real' else 'shuffled') + f" {x['seed']}",
                     words=f"{sum(r['words_correct'] for r in rr)}/15", actions=' · '.join(r['action'] for r in rr),
                     ok=f"{sum(r['action_ok'] for r in rr)}/3", conf=', '.join(f"{r['action_p']:.2f}" for r in rr)))
out = src.replace('__STILL__', still).replace('__LOOP__', json.dumps(loop)).replace('__RUNS__', json.dumps(runs))
(ROOT / 'docs' / 'week3_report.html').write_text(out); print('wrote docs/week3_report.html', len(out) // 1024, 'KB')
