#!/usr/bin/env python
"""Compare several w4_overhead.py JSON results: per-item medians across runs
(max/min ratio, how many runs flagged >2x within the run) and the budget lines.

  python bench/w4_overhead_compare.py perf/results/w4/*_overhead.json perf/results/w4/repeats/*.json
"""
import json
import sys


def main():
    rs = [json.load(open(f)) for f in sys.argv[1:]]
    print(f'{len(rs)} runs:', ', '.join(r['tag'] for r in rs))
    print('| device | item | measurement | medians (us) per run | max/min | runs flagged 2x |')
    print('|---|---|---|---|---|---|')
    for d in ('cuda', 'cpu'):
        for it, v in rs[0]['micro'].get(d, {}).items():
            for k, s in v.items():
                if not (isinstance(s, dict) and 'median' in s):
                    continue
                xs = [r['micro'][d][it][k]['median'] for r in rs]
                fl = sum(r['micro'][d][it][k]['varies_2x'] for r in rs)
                print(f"| {d} | {it} | {k} | {' / '.join(f'{x:g}' for x in xs)} | "
                      f'{max(xs) / min(xs):.2f} | {fl}/{len(rs)} |')
    print()
    print('| run | rng | predicted ms | measured mean ms | steady mean ms | stall fraction | '
          'pred/mean | pred/steady |')
    print('|---|---|---|---|---|---|---|---|')
    for r in rs:
        for b in r.get('budget', []):
            print(f"| {r['tag']} | {b['rng']} | {b['predicted_ms']} | {b['measured_ms_mean']} | "
                  f"{b['measured_ms_steady_mean']} | {b['stall_fraction']:.1%} | "
                  f"{b['ratio_pred_over_meas']} | {b['ratio_pred_over_steady']} |")


if __name__ == '__main__':
    main()
