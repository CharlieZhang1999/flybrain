#!/usr/bin/env python
"""Roofline PNG and the peak table HTML (the site embeds it verbatim) from a
w4_roofline.py JSON.

Log-log: x = arithmetic intensity (FLOP/byte), y = TFLOP/s. Slanted line: the
Day 2 copy bandwidth, 382 GB/s. One flat roof per dtype at its peak (median
across runs of the per-run peak), ridge point (peak / 382 GB/s) marked; peak
and ridge values in the legend below the axes. Vertical lines: LIF decay (sim/lif_torch._decay) intensity, fused
lower bound of the traffic, float64 and float32 (the eager values are in the
table note).

Sized for a 390 px phone: 4.2 in wide at 200 dpi (840 px), text >= 9.5 pt,
i.e. >= 11 px when the image is scaled to ~350 px.
Needs only matplotlib:
  python bench/w4_roofline_plot.py perf/results/w4/<tag>_roofline.json
Output: perf/results/w4/roofline.png and <tag>_roofline_table.html next to the JSON.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

INK, MUTED, GRID = '#1f1f1e', '#52514e', '#ecebe6'
COL = {'fp16': '#2a78d6', 'bf16': '#eb6834', 'tf32': '#1baf7a', 'fp32': '#eda100', 'fp64': '#e87ba4'}
ORDER = ['fp16', 'bf16', 'tf32', 'fp32', 'fp64']
LABEL = {'fp16': 'fp16', 'bf16': 'bf16', 'tf32': 'tf32', 'fp32': 'fp32 (no TF32)', 'fp64': 'fp64'}


def main():
    src = Path(sys.argv[1])
    res = json.loads(src.read_text())
    A, bw = res['agg']['dtypes'], res['bw_day2_GBps']
    tb = bw / 1e3                                      # TFLOP/s per FLOP/byte
    x0, x1 = 0.01, 1000.0
    fig, ax = plt.subplots(figsize=(4.2, 5.6), dpi=200)
    ax.plot([x0, x1], [tb * x0, tb * x1], color=INK, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.text(20, tb * 20 / 2.2, f'{bw:.0f} GB/s', rotation=0, fontsize=9.5, color=INK, ha='left', va='top')
    names = [k for k in ORDER if A.get(k, {}).get('peak_tflops')]
    handles = []
    for k in names:
        p = A[k]['peak_tflops']['median']
        r = p / tb
        ls = (0, (3, 2)) if k == 'fp16' else '-'   # fp16 and bf16 roofs nearly coincide: fp16 dashed on top
        ax.plot([r, x1], [p, p], color=COL[k], linewidth=2.0, linestyle=ls, solid_capstyle='round',
                zorder=5 if k == 'fp16' else 3)
        ax.plot([r], [p], 'o', ms=6, color=COL[k], mec='white', mew=1.5, zorder=4)
        handles.append(Line2D([], [], color=COL[k], linewidth=2.0, linestyle=ls, marker='o', ms=5, mec='white',
                              label=f'{LABEL[k]}: {p:.3g} TFLOP/s, ridge {r:.3g}'))
    leg = ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(-0.17, -0.13), fontsize=9.5, frameon=True, framealpha=1.0,
                    edgecolor='#c9c8c2', handlelength=1.6, borderpad=0.5, labelspacing=0.35)
    for t in leg.get_texts():
        t.set_color(INK)
    dec = res['decay']
    for k, lab, ls in (('float64', 'LIF decay f64', '-'), ('float32', 'LIF decay f32', (0, (1.5, 1.5)))):
        xi = dec[k]['fused_flop_per_byte']
        ax.axvline(xi, color=MUTED, linewidth=1.2, linestyle=ls, zorder=2)
    y_lab = 0.5
    ax.text(dec['float64']['fused_flop_per_byte'] / 1.12, y_lab, f"LIF decay f64 {dec['float64']['fused_flop_per_byte']:.2f}",
            rotation=90, ha='right', va='bottom', fontsize=9.5, color=INK)
    ax.text(dec['float32']['fused_flop_per_byte'] * 1.12, y_lab, f"f32 {dec['float32']['fused_flop_per_byte']:.2f}",
            rotation=90, ha='left', va='bottom', fontsize=9.5, color=INK)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(x0, x1)
    ax.set_ylim(0.003, 200)
    ax.set_xlabel('FLOP per byte (log)', fontsize=10, color=MUTED)
    ax.set_ylabel('TFLOP/s (log)', fontsize=10, color=MUTED)
    ax.tick_params(labelsize=9.5, colors=MUTED, length=0)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#c9c8c2')
    ax.grid(True, which='major', color=GRID); ax.set_axisbelow(True)
    ax.set_title(f'Roofline, RTX 5060 Ti (median of {len(res["runs"])} runs)', loc='left', fontsize=11.5, color=INK)
    fig.tight_layout()
    png = src.parent / 'roofline.png'
    fig.savefig(png, facecolor='white')
    print('saved', png)
    tab = src.with_name(src.stem + '_table.html')
    tab.write_text(table_html(res), encoding='utf-8', newline='\n')
    print('saved', tab)


def _mmm(a, f='{:.2f}'):
    return f"{f.format(a['median'])} <span class=\"rg\">[{f.format(a['min'])}–{f.format(a['max'])}]</span>"


def _f(x):
    return f'{x:.3f}' if x < 1 else f'{x:.1f}'


def table_html(res):
    """The peak table as an HTML fragment; the site embeds this file verbatim."""
    A, n, bw = res['agg']['dtypes'], len(res['runs']), res['bw_day2_GBps']
    dec = res['decay']
    h = ['<table class="roofline">',
         '<thead><tr><th>dtype</th><th class="n">peak TFLOP/s</th><th class="n">spec TFLOP/s</th>'
         '<th class="n">% of spec</th><th class="n">ridge FLOP/byte</th></tr></thead>', '<tbody>']
    for k in ORDER:
        a = A.get(k)
        if not a or not a.get('peak_tflops'):
            h.append(f'<tr><td>{LABEL[k]}</td><td class="n">error</td><td class="n">–</td>'
                     f'<td class="n">–</td><td class="n">–</td></tr>')
            continue
        p = a['peak_tflops']
        f = '{:.3f}' if p['median'] < 1 else '{:.1f}'
        h.append(f'<tr><td>{LABEL[k]}</td><td class="n">{_mmm(p, f)}</td>'
                 f'<td class="n">{_f(a["spec_tflops"])}</td>'
                 f'<td class="n">{a["pct_of_spec"]:.0f}%</td>'
                 f'<td class="n">{a["ridge_flop_per_byte_382"]:.3g}</td></tr>')
    d64, d32 = dec['float64'], dec['float32']
    h += ['</tbody>', '</table>',
          f'<p class="note">torch.matmul on square matrices of n = 1,024 to 8,192, {res["args"]["reps"]} timed repeats per size, '
          f'{n} runs. Peak: the best size’s median in each run; median across {n} runs, [min–max] across runs. '
          f'Spec: NVIDIA publishes no TFLOP/s per type for the 5060 Ti; these are scaled from the RTX 5070 in '
          f'NVIDIA’s RTX Blackwell whitepaper (per SM and clock, 36 SMs, 2.57 GHz boost), fp16/bf16 with 32-bit '
          f'accumulate, fp64 = fp32 / 64. Ridge: peak / {bw:.0f} GB/s copy bandwidth (Day 2). LIF decay per '
          f'neuron: {d64["flops_per_neuron"]:g} FLOPs; {d64["fused_bytes_per_neuron"]:.0f} bytes in float64 and '
          f'{d32["fused_bytes_per_neuron"]:.0f} in float32 if fused (each array read or written once), '
          f'{d64["eager_bytes_per_neuron"]:.0f} and {d32["eager_bytes_per_neuron"]:.0f} as run today, one '
          f'kernel per operation.</p>']
    return '\n'.join(h) + '\n'


if __name__ == '__main__':
    main()
