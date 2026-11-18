#!/usr/bin/env python
"""Stacked bar of the predicted GPU step cost by category (from w4_overhead.py JSON),
with the measured ms/step of the same steps as a marker.

Needs only numpy + matplotlib (kept out of w4_overhead.py so the bench runs
without matplotlib).
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
    rows = res['budget']
    cats = list(rows[0]['category_ms'].keys())
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=150)
    ys = list(range(len(rows)))
    for y, b in zip(ys, rows):
        left = 0.0
        for i, c in enumerate(cats):
            w = b['category_ms'][c]
            if w <= 0:
                continue
            ax.barh(y, w, left=left, height=0.5, color=COLORS[i % len(COLORS)],
                    edgecolor='white', linewidth=2, label=c if y == 0 else None)
            if w > 0.12:
                ax.text(left + w / 2, y, f'{w:.2f}', ha='center', va='center', fontsize=8, color='white')
            left += w
        m, ms = b['measured_ms_mean'], b['measured_ms_steady_mean']
        ax.plot([m, m], [y - 0.35, y + 0.35], color=INK, linewidth=2)
        ax.plot([ms, ms], [y - 0.35, y + 0.35], color=INK, linewidth=1.2, linestyle=(0, (2, 2)))
        ax.text(m, y + 0.42, f'measured {m:.2f} (steady {ms:.2f}, dashed)', ha='center', va='bottom',
                fontsize=8, color=INK)
        ax.text(left + 0.03, y - 0.05, f'predicted {left:.2f}', ha='left', va='center', fontsize=8,
                color=MUTED)
    ax.set_yticks(ys, [f"rng='{b['rng']}'" for b in rows])
    ax.set_xlabel('ms per 0.1-ms step (RTX 5060 Ti, WSL, float64 eager)', color=MUTED)
    xmax = max(max(sum(b['category_ms'].values()), b['measured_ms_mean']) for b in rows)
    ax.set_xlim(0, xmax * 1.25)
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


if __name__ == '__main__':
    main()
