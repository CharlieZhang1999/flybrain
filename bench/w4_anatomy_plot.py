#!/usr/bin/env python
"""Stacked bar of one steady GPU step by phase group (host wall time and GPU
execution time of the non-sync ops), from a w4_anatomy.py JSON, plus the HTML
phase table that the site embeds verbatim.

Sized for a 390 px phone: the PNG is 4.2 in wide at 200 dpi (840 px), so 9-10 pt
text renders at >= 10 px when the image is scaled to ~360 px.
Needs only matplotlib (kept out of w4_anatomy.py so the bench runs without it).
  python bench/w4_anatomy_plot.py perf/results/w4/<tag>_anatomy.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# categorical slots 1-5 of the default validated palette (light mode), fixed order; 'other' in gray
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#a3a29b']
INK, MUTED = '#1f1f1e', '#52514e'
GROUPS = [('Gather/scatter (presyn + synaptic)', ['presyn', 'synaptic']),
          ('Spike detect, bookkeeping, reset', ['threshold+nonzero', 'bookkeeping', 'reset']),
          ('APL', ['apl']),
          ('Drive + spikelets', ['drive', 'spikelets']),
          ('Decay + gap', ['decay', 'gap']),
          ('Other (ring, loop)', ['ring', '(loop)'])]
PHASES = ['decay', 'gap', 'threshold+nonzero', 'bookkeeping', 'presyn', 'synaptic', 'drive',
          'spikelets', 'apl', 'reset', 'ring']


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(src.stem + '_phases.png')
    res = json.loads(src.read_text())
    A, n = res['agg'], len(res['runs'])
    ph = A['phases']
    rows = [('Host wall', {p: ph[p]['wall_median_ms']['median'] for p in ph}),
            ('GPU exec\n(non-sync ops)', {p: ph[p].get('gpu_exec_median_ms', {'median': 0.0})['median']
                                          for p in ph})]
    steady = A['step']['host_steady_median_ms']['median']
    fig, ax = plt.subplots(figsize=(4.2, 4.1), dpi=200)
    for y, (lab, vals) in enumerate(rows):
        left = 0.0
        for i, (gname, members) in enumerate(GROUPS):
            w = sum(vals.get(m, 0.0) for m in members)
            if w <= 0:
                continue
            ax.barh(y, w, left=left, height=0.62, color=COLORS[i], edgecolor='white', linewidth=1.5,
                    label=gname if y == 0 else None)
            if w >= 0.2:
                ax.text(left + 0.02, y, f'{w:.2f}', ha='left', va='center', fontsize=9, color='white',
                        fontweight='bold')
            left += w
        ax.text((max(left, steady) if y == 0 else left) + 0.04, y, f'{left:.2f}', ha='left', va='center', fontsize=9, color=INK)
    ax.plot([steady, steady], [-0.45, 0.45], color=INK, linewidth=1.6)
    ax.text(steady, -0.5, f'steady step {steady:.2f}', ha='right', va='bottom', fontsize=9, color=INK)
    ax.set_yticks([0, 1], [r[0] for r in rows], fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, max(steady, 1.0) * 1.22)
    ax.set_ylim(1.5, -0.9)
    ax.set_xlabel(f'ms per 0.1-ms step (median of {n} runs)', fontsize=9, color=MUTED)
    ax.tick_params(axis='x', labelsize=9, colors=MUTED, length=0)
    ax.tick_params(axis='y', length=0)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#c9c8c2')
    ax.xaxis.grid(True, color='#ecebe6'); ax.set_axisbelow(True)
    ax.set_title('One steady GPU step by phase', loc='left', fontsize=11, color=INK)
    ax.legend(loc='upper left', bbox_to_anchor=(-0.32, -0.2), ncol=1, frameon=False, fontsize=9,
              handlelength=1.2, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, facecolor='white')
    print('saved', out)
    tab = out.with_name(out.name.replace('_phases.png', '_phases_table.html'))
    tab.write_text(table_html(res), encoding='utf-8', newline='\n')
    print('saved', tab)


def _mmm(a, f='{:.2f}'):
    return f"{f.format(a['median'])} <span class=\"rg\">[{f.format(a['min'])}–{f.format(a['max'])}]</span>"


def table_html(res):
    """The phase table as an HTML fragment; the site embeds this file verbatim."""
    A, n = res['agg'], len(res['runs'])
    ph, S = A['phases'], A['step']
    h = ['<table class="anatomy">',
         '<thead><tr><th>phase</th><th class="n">wall ms</th><th class="n">GPU exec ms</th>'
         '<th class="n">syncs</th><th class="n">kernels</th></tr></thead>', '<tbody>']
    for p in sorted(PHASES, key=lambda p: -ph[p]['wall_median_ms']['median']):
        r = ph[p]
        h.append(f'<tr><td>{p}</td><td class="n">{_mmm(r["wall_median_ms"])}</td>'
                 f'<td class="n">{r["gpu_exec_median_ms"]["median"]:.2f}</td>'
                 f'<td class="n">{r["syncs"]["median"]:g}</td><td class="n">{r["kernels"]["median"]:.0f}</td></tr>')
    h.append(f'<tr class="total"><td>Sum of phase medians</td><td class="n">{_mmm(S["sum_phase_wall_medians_ms"])}</td>'
             f'<td class="n">{S["gpu_exec_nonsync_step_ms"]["median"]:.2f}</td>'
             f'<td class="n">{S["syncs_total"]["median"]:g}</td><td class="n">{S["kernels_total"]["median"]:.0f}</td></tr>')
    for lab, key, f in (('Steady step (≤ 5 ms), same pass', 'host_steady_median_ms', '{:.2f}'),
                        ('Sum / steady', 'sum_over_steady', '{:.2f}'),
                        ('Steady step, uninstrumented run_trial', 'reference_steady_median_ms', '{:.2f}'),
                        ('Steady step, with CUDA events + NVTX', 'full_steady_median_ms', '{:.2f}'),
                        ('Share of stall steps (> 5 ms)', 'host_stall_fraction', '{:.1%}'),
                        ('GPU utilisation, nvidia-smi (%)', 'util_median_pct', '{:.0f}'),
                        ('GPU exec / wall (non-sync ops)', 'gpu_exec_nonsync_over_wall', '{:.2f}')):
        h.append(f'<tr><td>{lab}</td><td class="n" colspan="4">{_mmm(S[key], f)}</td></tr>')
    h += ['</tbody>', '</table>',
          f'<p class="note">Medians across {n} runs, [min–max] across runs. Wall: host perf_counter per '
          'phase, steady steps only. GPU exec: kernels of the '
          'non-sync ops timed with the host hidden (a GPU sleep before each op); the 10 sync ops are '
          'excluded. Kernels: aten ops that launch GPU work.</p>']
    return '\n'.join(h) + '\n'


if __name__ == '__main__':
    main()
