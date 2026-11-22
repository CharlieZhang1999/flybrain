#!/usr/bin/env python
"""Week 4 device table (perf/week04.md and the site's Week 4 GPU summary) from
the committed result JSONs. No measurement: it only reads and formats.

Sources (perf/results/w4/):
  DESKTOP-4C13T5V_20261122-1053_graph.json   Day 6 part 2 (T5b): CPU, reference,
      sync-free eager and graph rows; 5 runs, median [min-max], steady and all steps.
  DESKTOP-4C13T5V_20261116-2309.json          Day 2 bench: eager vs torch.compile,
      0.2 bio-s, setup included, median of 3 repeats and the worst repeat.

bio-s per wall-s = 0.1 ms (dt, one step) / ms per step.
Usage (any Python 3, stdlib only):
  python bench/w4_summary_table.py
Writes perf/results/w4/week04_device_table.md and week04_device_table.html.
"""
import json
from pathlib import Path

D = Path(__file__).resolve().parent.parent / 'perf' / 'results' / 'w4'
GRAPH = 'DESKTOP-4C13T5V_20261122-1053_graph.json'
DAY2 = 'DESKTOP-4C13T5V_20261116-2309.json'
DT_MS = 0.1

# (label, dtype, graph.json row key, short source)
GRAPH_ROWS = [
    ('PC CPU (6 threads), reference', 'f64', 'ref eager cpu f64', 'Day 6'),
    ('5060 Ti, reference eager', 'f64', 'ref eager cuda f64', 'Day 6'),
    ('5060 Ti, sync-free eager', 'f64', 'fast eager cuda f64', 'Day 6'),
    ('5060 Ti, sync-free CUDA graph', 'f64', 'fast graph cuda f64', 'Day 6'),
    ('5060 Ti, sync-free CUDA graph', 'f32', 'fast graph cuda f32', 'Day 6'),
]


def mmm(a):
    return f"{a['median']:.2f} [{a['min']:.2f}-{a['max']:.2f}]"


def bio(a):
    return f"{DT_MS / a['median']:.3f} [{DT_MS / a['max']:.3f}-{DT_MS / a['min']:.3f}]"


def rows():
    g = json.loads((D / GRAPH).read_text())['agg']['rows']
    out = []
    for label, dt, key, src in GRAPH_ROWS[:2]:
        r = g[key]
        out.append((label, dt, mmm(r['steady_mean_ms']), mmm(r['window_ms_per_step_no_events']),
                    bio(r['steady_mean_ms']), src, GRAPH))
    d2 = json.loads((D / DAY2).read_text())['sim']
    c = next(x for x in d2 if x['device'] == 'cuda' and x['dtype'] == 'float64' and x['compile'])
    out.append(('5060 Ti, reference eager + torch.compile', 'f64', '-',
                f"{c['ms_per_step']:.2f} (worst {c['worst_s_per_bio_s'] / 10:.2f})",
                f"{1 / c['s_per_bio_s']:.3f} (all steps)", 'Day 2', DAY2))
    for label, dt, key, src in GRAPH_ROWS[2:]:
        r = g[key]
        out.append((label, dt, mmm(r['steady_mean_ms']), mmm(r['window_ms_per_step_no_events']),
                    bio(r['steady_mean_ms']), src, GRAPH))
    out.append(('Mac CPU', '-', 'not run', 'not run', 'not run', '-', '-'))
    out.append(('Mac MPS', '-', 'not run', 'not run', 'not run', '-', '-'))
    return out


def md(rs):
    h = ['| device and mode | dtype | steady ms/step | all steps ms/step | bio-s per wall-s | source |',
         '|---|---|---|---|---|---|']
    for label, dt, st, al, b, src, f in rs:
        s = f'{src}, `perf/results/w4/{f}`' if f != '-' else 'not measured in Week 4'
        h.append(f'| {label} | {dt} | {st} | {al} | {b} | {s} |')
    return '\n'.join(h) + '\n'


def _rg(s):
    """'1.73 [1.57-1.85]' -> '1.73 <span class="rg">[1.57–1.85]</span>' (site style)."""
    if ' [' not in s:
        return s
    v, r = s.split(' [', 1)
    return f'{v} <span class="rg">[{r[:-1].replace("-", "–")}]</span>'


def html(rs):
    h = ['<table class="w4sum">',
         '<thead><tr><th>device and mode</th><th>dtype</th><th class="n">steady ms/step</th>'
         '<th class="n">all steps ms/step</th><th class="n">bio-s per wall-s</th><th>source</th></tr></thead>',
         '<tbody>']
    for label, dt, st, al, b, src, f in rs:
        h.append(f'<tr><td>{label}</td><td>{dt}</td><td class="n">{_rg(st)}</td><td class="n">{_rg(al)}</td>'
                 f'<td class="n">{_rg(b)}</td><td>{src}</td></tr>')
    h += ['</tbody>', '</table>',
          '<p class="note">Day 6 rows: one looming trial per run, 2,000 steps after 1,000 warm-up steps, 5 runs, '
          'median [min–max] across runs; steady = mean of the steps ≤ 5 ms, all steps = wall time / steps with '
          'stalls. Day 2 row: 0.2 bio-s from t = 0 with setup and warm-up included, median of 3 repeats and the '
          'worst repeat; it is not comparable with the steady column. bio-s per wall-s = 0.1 ms / ms per step. '
          'The Mac rows were not measured in Week 4. Files: flybrain <code>perf/results/w4/' + GRAPH + '</code> '
          'and <code>perf/results/w4/' + DAY2 + '</code>.</p>']
    return '\n'.join(h) + '\n'


def main():
    rs = rows()
    (D / 'week04_device_table.md').write_text(md(rs), encoding='utf-8', newline='\n')
    (D / 'week04_device_table.html').write_text(html(rs), encoding='utf-8', newline='\n')
    print(md(rs))


if __name__ == '__main__':
    main()
