#!/usr/bin/env python
"""Render one steady step from an nsys capture of `w4_anatomy.py --nsys-window`
(exported to SQLite with `nsys export -t sqlite`) as a PNG timeline: the NVTX
phase ranges of the step and, if the capture has -t osrt, the OS-runtime calls
longer than 20 us below them.

This is a host-side timeline: on this WSL setup nsys 2024.6.2 collects no CUDA
activity (CUPTI_ERROR_INVALID_DEVICE; driver 13.3 is newer than this nsys
supports), so there are no GPU kernel rows.

  python bench/w4_nsys_timeline.py <capture>.sqlite [out.png]
The step drawn is the one whose duration is closest to the median of the window.
"""
import sqlite3
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

PHASES = ['decay', 'gap', 'threshold+nonzero', 'bookkeeping', 'presyn', 'synaptic', 'drive',
          'spikelets', 'apl', 'reset', 'ring', 'apl_rate']
GROUP = {'presyn': 0, 'synaptic': 0, 'threshold+nonzero': 1, 'bookkeeping': 1, 'reset': 1, 'apl': 2,
         'apl_rate': 2, 'drive': 3, 'spikelets': 3, 'decay': 4, 'gap': 4, 'ring': 5}
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#a3a29b']
INK, MUTED = '#1f1f1e', '#52514e'


def load(db):
    c = sqlite3.connect(db)
    has_str = c.execute("select count(*) from sqlite_master where name='StringIds'").fetchone()[0]
    q = ('select n.start, n.end, coalesce(n.text, s.value) from NVTX_EVENTS n '
         'left join StringIds s on n.textId = s.id where n.end is not null') if has_str else \
        'select start, end, text from NVTX_EVENTS where end is not null'
    nv = [(a, b, t) for a, b, t in c.execute(q) if t]
    osrt = []
    if c.execute("select count(*) from sqlite_master where name='OSRT_API'").fetchone()[0]:
        osrt = list(c.execute('select o.start, o.end, s.value from OSRT_API o join StringIds s '
                              'on o.nameId = s.id'))
    return nv, osrt


def main():
    db = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else db.with_name(db.stem + '_step.png')
    nv, osrt = load(db)
    steps = sorted([(a, b, t) for a, b, t in nv if t.startswith('step ')])
    dur = [b - a for a, b, _ in steps[1:-1]]
    med = statistics.median(dur)
    s0, s1, st = min(steps[1:-1], key=lambda x: abs((x[1] - x[0]) - med))
    print(f'{len(steps)} steps, median {med / 1e6:.3f} ms, drawing {st} ({(s1 - s0) / 1e6:.3f} ms)')
    ph = [(a, b, t) for a, b, t in nv if t in PHASES and s0 <= a < s1]
    os_ = [(a, b, t) for a, b, t in osrt if s0 <= a < s1 and b - a > 20_000]
    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=200)
    ms = lambda x: (x - s0) / 1e6  # noqa: E731
    for a, b, t in ph:
        ax.barh(1, ms(b) - ms(a), left=ms(a), height=0.6, color=COLORS[GROUP[t]], edgecolor='white',
                linewidth=0.8)
    for a, b, t in os_:
        ax.barh(0, ms(b) - ms(a), left=ms(a), height=0.6, color='#52514e', edgecolor='white', linewidth=0.8)
    # label the three longest phase ranges
    for a, b, t in sorted(ph, key=lambda x: x[0] - x[1])[:3]:
        ax.text((ms(a) + ms(b)) / 2, 1.42, t, ha='center', va='bottom', fontsize=8, color=INK)
    if os_:
        ax.set_yticks([0, 1], ['OS calls\n> 20 us', 'phases\n(NVTX)'], fontsize=9, color=INK)
        ax.set_ylim(-0.6, 2.0)
    else:                    # capture without -t osrt, or no call > 20 us in this step
        ax.set_yticks([1], ['phases\n(NVTX)'], fontsize=9, color=INK)
        ax.set_ylim(0.4, 2.0)
    ax.set_xlim(0, ms(s1) * 1.02)
    ax.set_xlabel(f'ms within one steady step ({(s1 - s0) / 1e6:.2f} ms; host side)', fontsize=9, color=MUTED)
    ax.tick_params(axis='x', labelsize=9, colors=MUTED, length=0)
    ax.tick_params(axis='y', length=0)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color('#c9c8c2')
    ax.xaxis.grid(True, color='#ecebe6'); ax.set_axisbelow(True)
    ax.set_title('nsys timeline of one step (host)', loc='left', fontsize=11, color=INK)
    names = ['Gather/scatter', 'Spike detect, bookkeeping, reset', 'APL', 'Drive + spikelets', 'Decay + gap',
             'Ring'] + (['OS call (e.g. ioctl)'] if os_ else [])
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c) for c in (COLORS + ['#52514e'])[:len(names)]], labels=names, loc='upper left',
              bbox_to_anchor=(-0.25, -0.28), ncol=2, frameon=False, fontsize=8, handlelength=1.0)
    fig.tight_layout()
    fig.savefig(out, facecolor='white')
    print('saved', out)
    kinds = {}
    for a, b, t in os_:
        kinds.setdefault(t, []).append((b - a) / 1e3)
    print('OS calls > 20 us in that step:', {k: (len(v), round(sum(v), 1)) for k, v in kinds.items()})


if __name__ == '__main__':
    main()
