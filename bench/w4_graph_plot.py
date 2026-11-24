#!/usr/bin/env python
"""Bar chart of ms per step by row (reference / fast eager / fast graph, GPU and
CPU) from a w4_graph.py JSON, plus the HTML table that the site embeds verbatim.

Sized for a 390 px phone: 4.2 in wide at 200 dpi (840 px), 10.5-12 pt text, so
text renders at >= 10 px when the image is scaled to ~300 px.
Needs only matplotlib (run from a venv that has it):
  python bench/w4_graph_plot.py perf/results/w4/<tag>_graph.json
Output: <tag>_graph_bars.png and <tag>_graph_table.html next to the JSON.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = '#1f1f1e', '#52514e', '#ecebe6'
BLUE, ORANGE, GREEN, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#a3a29b'
CPU_MS = 0.64          # PC CPU baseline of the brief (T1: reference, 6 CPU threads)

# (json row, label, colour) in display order; rows missing from the JSON are skipped
ROWS = [('ref eager cuda f64', 'Reference, GPU', GRAY),
        ('fast eager cuda f64', 'Sync-free, GPU eager', ORANGE),
        ('fast graph cuda f64', 'Sync-free, GPU graph', BLUE),
        ('fast graph cuda f32', 'Sync-free, GPU graph, float32', BLUE),
        ('fast eager cuda f32', 'Sync-free, GPU eager, float32', ORANGE),
        ('fast eager cuda f64 default caps', 'Sync-free, GPU eager, default caps', ORANGE),
        ('fast graph cuda f64 deterministic', 'Sync-free, GPU graph, deterministic', BLUE),
        ('ref eager cpu f64', 'Reference, CPU', GREEN),
        ('fast eager cpu f64', 'Sync-free, CPU', GREEN)]
PLOT = ['ref eager cuda f64', 'fast eager cuda f64', 'fast graph cuda f64', 'fast graph cuda f32',
        'ref eager cpu f64', 'fast eager cpu f64']
PLOT_LABEL = {'ref eager cuda f64': 'Reference\nGPU', 'fast eager cuda f64': 'Sync-free\nGPU eager',
              'fast graph cuda f64': 'Sync-free\nGPU graph', 'fast graph cuda f32': 'Graph\nfloat32',
              'ref eager cpu f64': 'Reference\nCPU', 'fast eager cpu f64': 'Sync-free\nCPU'}


def main():
    src = Path(sys.argv[1])
    res = json.loads(src.read_text())
    A, n = res['agg']['rows'], len(res['runs'])
    color = {k: c for k, _, c in ROWS}
    names = [k for k in PLOT if k in A]
    med = [A[k]['steady_mean_ms']['median'] for k in names]
    lo = [m - A[k]['steady_mean_ms']['min'] for k, m in zip(names, med)]
    hi = [A[k]['steady_mean_ms']['max'] - m for k, m in zip(names, med)]
    fig, ax = plt.subplots(figsize=(4.2, 4.4), dpi=200)
    y = list(range(len(names)))
    ax.barh(y, med, height=0.62, color=[color[k] for k in names], edgecolor='white', linewidth=1.2)
    ax.errorbar(med, y, xerr=[lo, hi], fmt='none', ecolor=INK, elinewidth=1.0, capsize=2.5)
    for yy, m, h in zip(y, med, hi):
        x = m + h + 0.04
        if x < CPU_MS + 0.04 < x + 0.25:     # keep the label clear of the CPU line
            x = CPU_MS + 0.05
        ax.text(x, yy, f'{m:.2f}', ha='left', va='center', fontsize=11, color=INK)
    ax.axvline(CPU_MS, color=INK, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(CPU_MS + 0.03, -0.75, f'CPU {CPU_MS:.2f} ms (T1)', ha='left', va='bottom', fontsize=10.5, color=INK)
    ax.set_yticks(y, [PLOT_LABEL[k] for k in names], fontsize=11, color=INK)
    ax.invert_yaxis()
    xmax = max(m + h for m, h in zip(med, hi))
    ax.set_xlim(0, xmax * 1.2)
    ax.set_ylim(len(names) - 0.5, -1.0)
    ax.set_xlabel(f'ms per step, steady (median of {n} runs)', fontsize=10.5, color=MUTED)
    ax.tick_params(axis='x', labelsize=10.5, colors=MUTED, length=0)
    ax.tick_params(axis='y', length=0)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#c9c8c2')
    ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_title('Time per step, one trial', loc='left', fontsize=12, color=INK)
    fig.tight_layout()
    png = src.with_name(src.stem + '_bars.png')
    fig.savefig(png, facecolor='white')
    print('saved', png)
    tab = src.with_name(src.stem + '_table.html')
    tab.write_text(table_html(res), encoding='utf-8', newline='\n')
    print('saved', tab)


def _mmm(a, f='{:.2f}'):
    if a is None:
        return '–'
    return f"{f.format(a['median'])} <span class=\"rg\">[{f.format(a['min'])}–{f.format(a['max'])}]</span>"


def table_html(res):
    """The ms/step table as an HTML fragment; the site embeds this file verbatim."""
    A, n = res['agg']['rows'], len(res['runs'])
    h = ['<table class="graph">',
         '<thead><tr><th>run</th><th class="n">steady ms/step</th><th class="n">all steps ms/step</th>'
         '<th class="n">stall steps</th><th class="n">peak GPU MB</th></tr></thead>', '<tbody>']
    for k, lab, _ in ROWS:
        if k not in A:
            continue
        a = A[k]
        steps = '' if a['steps'] == res['args']['steps'] else f' ({a["steps"]} steps)'
        mem = '–' if a['peak_mem_mb'] is None else f"{a['peak_mem_mb']['median']:.0f}"
        h.append(f'<tr><td>{lab}{steps}</td><td class="n">{_mmm(a["steady_mean_ms"])}</td>'
                 f'<td class="n">{_mmm(a["window_ms_per_step_no_events"])}</td>'
                 f'<td class="n">{_mmm(a["stall_fraction"], "{:.1%}")}</td>'
                 f'<td class="n">{mem}</td></tr>')
    h += ['</tbody>', '</table>',
          f'<p class="note">Medians across {n} runs, [min–max] across runs; one trial per run of '
          f'{res["args"]["steps"]:,} steps after {res["args"]["warm"]:,} warm-up steps, looming, float64 unless '
          f'marked. Steady: mean of the steps ≤ 5 ms (GPU: CUDA events rescaled to the host clock, CPU: '
          f'host clock). All steps: wall time / steps, stalls included, in a second trial without events '
          f'(GPU). Rows marked 200 steps: every step takes over 5 ms, so they have no steady value and '
          f'count as 100% stall steps by that rule. Peak GPU MB: allocated during setup and trial. '
          f'Caps: {res["caps"]["spike_cap"]} spikes '
          f'and {res["caps"]["syn_cap"]:,} synapses per step.</p>']
    return '\n'.join(h) + '\n'


if __name__ == '__main__':
    main()
