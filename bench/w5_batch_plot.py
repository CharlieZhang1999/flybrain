#!/usr/bin/env python
"""Scaling chart for Week 5 P1 from a w5_batch.py JSON, plus the HTML table the
site embeds verbatim.

Two stacked panels sharing the B axis (1, 2, 4, 8, 16, 32): (top) all-steps
ms per step, (bottom) one-second trials per wall-second, float64 (blue) and
float32 (orange, information only), median of the runs with min-max bars.
Reference lines: the Week 4 fast single-trial graph and the PC CPU single
trial, measured in the same session; the 1 ms/step target in the top panel.
Sized for a 390 px phone like the Week 4 charts: 4.2 in wide at 200 dpi
(840 px), 10.5-12 pt text.
Needs only matplotlib:
  python bench/w5_batch_plot.py perf/results/w5/<tag>_batch.json
Output: <tag>_batch_scaling.png and <tag>_batch_table.html next to the JSON.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = '#1f1f1e', '#52514e', '#ecebe6'
BLUE, ORANGE, GRAY = '#2a78d6', '#eb6834', '#a3a29b'


def series(A, dt, key):
    names = sorted((k for k in A if k.startswith(f'batch graph cuda {dt} B=')), key=lambda k: A[k]['B'])
    B = [A[k]['B'] for k in names]
    v = [A[k][key] for k in names]
    return B, [x['median'] for x in v], [x['min'] for x in v], [x['max'] for x in v]


def main():
    src = Path(sys.argv[1])
    res = json.loads(src.read_text())
    A, n = res['agg']['rows'], len(res['runs'])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.2, 6.6), dpi=200, sharex=True)
    for ax, key, ylab in ((ax1, 'window_ms_per_step_no_events', 'ms per step (all steps)'),
                          (ax2, 'trials_per_s_all_steps', '1 s trials per wall-second')):
        for dt, col, lab in (('f64', BLUE, 'float64'), ('f32', ORANGE, 'float32 (info)')):
            B, med, lo, hi = series(A, dt, key)
            if not B:
                continue
            ax.errorbar(B, med, yerr=[[m - a for m, a in zip(med, lo)], [b - m for m, b in zip(med, hi)]],
                        color=col, lw=2, marker='o', ms=6, capsize=2.5, label=lab,
                        mec='white', mew=1.2)
        for row, lab, ls in (('fast graph cuda f64', 'single trial, GPU graph', (0, (4, 3))),
                             ('ref eager cpu f64', 'single trial, CPU', (0, (1, 2)))):
            if row in A:
                ax.axhline(A[row][key]['median'], color=GRAY, lw=1.4, ls=ls, label=lab)
        ax.set_ylabel(ylab, fontsize=11, color=INK)
        ax.set_xscale('log', base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32], ['1', '2', '4', '8', '16', '32'])
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_color(MUTED)
        ax.tick_params(colors=INK, labelsize=10.5)
    ax1.set_yscale('log', base=2)
    yt = [0.25, 0.5, 1, 2, 4, 8, 16]
    ax1.set_yticks(yt, [f'{x:g}' for x in yt])
    ax1.axhline(1.0, color=INK, lw=1.0, ls=(0, (6, 3)))
    ax1.text(32, 0.93, 'target for B = 8: 1 ms', fontsize=10, color=INK, va='top', ha='right')
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel('trials in the batch (B)', fontsize=11, color=INK)
    ax1.legend(fontsize=9.5, frameon=False, loc='upper left')
    fig.suptitle(f'B trials in one CUDA graph (RTX 5060 Ti), {n} runs', fontsize=11.5, color=INK, x=0.02,
                 ha='left')
    fig.tight_layout()
    out = src.with_name(src.name.replace('_batch.json', '_batch_scaling.png'))
    fig.savefig(out, facecolor='white')
    print('saved', out)

    rows = []
    for k, a in A.items():
        m = lambda x, f='{:.3f}': '-' if x is None else f"{f.format(x['median'])} [{f.format(x['min'])}-{f.format(x['max'])}]"  # noqa: E731
        rows.append(f"<tr><td>{k}</td><td>{a['B']}</td><td>{m(a['steady_mean_ms'])}</td>"
                    f"<td>{m(a['stall_fraction'], '{:.3f}')}</td><td>{m(a['window_ms_per_step_no_events'])}</td>"
                    f"<td>{m(a['trials_per_s_all_steps'])}</td><td>{m(a['peak_mem_mb'], '{:.0f}')}</td>"
                    f"<td>{a['overflow_total'] if a['overflow_total'] is not None else '-'}</td></tr>")
    html = ('<table class="perf"><thead><tr><th>row</th><th>B</th><th>steady ms/step</th><th>stall share</th>'
            '<th>all-steps ms/step</th><th>trials/s</th><th>peak GPU MB</th><th>overflow</th></tr></thead>'
            '<tbody>' + ''.join(rows) + '</tbody></table>\n')
    out2 = src.with_name(src.name.replace('_batch.json', '_batch_table.html'))
    out2.write_text(html)
    print('saved', out2)


if __name__ == '__main__':
    main()
