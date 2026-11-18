#!/usr/bin/env python
"""Stacked bar of the predicted GPU step cost by category (from a w4_overhead.py
JSON: unit costs = medians across runs), with the measured steady-state and
all-steps ms/step (medians across runs) as markers.

Needs only matplotlib (kept out of w4_overhead.py so the bench runs without it).
  python bench/w4_overhead_plot.py perf/results/w4/<tag>_overhead.json [out.png]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# categorical slots 1-5 of the default validated palette (light mode), fixed order
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, MUTED = '#1f1f1e', '#6b6a64'


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(src.stem + '_budget.png')
    res = json.loads(src.read_text())
    rows = res['agg']['budget']
    n = len(res['runs'])
    cats = list(rows[0]['category_ms_agg_units'].keys())
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    ys = list(range(len(rows)))
    xmax = 0.0
    for y, b in zip(ys, rows):
        left = 0.0
        for i, c in enumerate(cats):
            w = b['category_ms_agg_units'][c]
            if w > 0:
                ax.barh(y, w, left=left, height=0.5, color=COLORS[i % len(COLORS)],
                        edgecolor='white', linewidth=2, label=c if y == 0 else None)
                if w > 0.15:
                    ax.text(left + w / 2, y, f'{w:.2f}', ha='center', va='center', fontsize=8, color='white')
            left += w
        st, al = b['steady_ms']['median'], b['all_steps_ms']['median']
        ax.plot([st, st], [y - 0.35, y + 0.35], color=INK, linewidth=2)
        ax.plot([al, al], [y - 0.35, y + 0.35], color=INK, linewidth=1.2, linestyle=(0, (2, 2)))
        ax.text(st, y + 0.42, f'measured steady {st:.2f} (all steps {al:.2f}, dashed)', ha='center',
                va='bottom', fontsize=8, color=INK)
        ax.text(left + 0.03, y - 0.05, f"predicted {b['predicted_ms_agg_units']:.2f}", ha='left',
                va='center', fontsize=8, color=MUTED, bbox=dict(facecolor='white', edgecolor='none', pad=1))
        xmax = max(xmax, left, st, al)
    ax.set_yticks(ys, [f"rng='{b['rng']}'" for b in rows])
    ax.set_xlabel(f'ms per 0.1-ms step, medians of {n} runs (RTX 5060 Ti, WSL, float64 eager)', color=MUTED)
    ax.set_xlim(0, xmax * 1.3)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#c9c8c2')
    ax.tick_params(colors=MUTED, length=0)
    ax.xaxis.grid(True, color='#ecebe6'); ax.set_axisbelow(True)
    ax.set_title('Predicted GPU step cost by category vs measured', loc='left', fontsize=11, color=INK)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, facecolor='white')
    print('saved', out)
    tab = out.with_name(out.name.replace('_budget.png', '_budget_table.html'))
    tab.write_text(table_html(rows, n), encoding='utf-8', newline='\n')
    print('saved', tab)


def _mmm(a, f='{:.2f}'):
    return f"{f.format(a['median'])} <span class=\"rg\">[{f.format(a['min'])}–{f.format(a['max'])}]</span>"


def table_html(rows, n):
    """The budget table as an HTML fragment; the site embeds this file verbatim."""
    h = ['<table class="budget">',
         '<thead><tr><th>ms per step</th>' + ''.join(f'<th class="n">{b["rng"]} rng</th>' for b in rows)
         + '</tr></thead>', '<tbody>']
    for c in rows[0]['category_ms_agg_units']:
        h.append(f'<tr><td>{c}</td>' + ''.join(f'<td class="n">{b["category_ms_agg_units"][c]:.2f}</td>'
                                              for b in rows) + '</tr>')
    h.append('<tr class="total"><td>Predicted (unit costs = medians of runs)</td>'
             + ''.join(f'<td class="n">{b["predicted_ms_agg_units"]:.2f}</td>' for b in rows) + '</tr>')
    for lab, key, f in (('Predicted per run, median [min–max]', 'predicted_ms_per_run', '{:.2f}'),
                        ('Measured, steady state (steps ≤ 5 ms)', 'steady_ms', '{:.2f}'),
                        ('Measured, all steps', 'all_steps_ms', '{:.2f}'),
                        ('Share of stall steps (~11 ms)', 'stall_fraction', '{:.1%}'),
                        ('Predicted / steady, per run', 'ratio_pred_over_steady_per_run', '{:.2f}')):
        h.append(f'<tr><td>{lab}</td>' + ''.join(f'<td class="n">{_mmm(b[key], f)}</td>' for b in rows)
                 + '</tr>')
    h += ['</tbody>', '</table>',
          f'<p class="note">Medians across {n} runs, [min–max] across runs. Categories use unit costs that '
          'are medians across runs and sizes.</p>']
    return '\n'.join(h) + '\n'


if __name__ == '__main__':
    main()
