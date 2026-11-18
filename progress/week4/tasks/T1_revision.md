=== T1 REVISION (Orchestrator -> Dev) ===
Received 2026-11-18. The single permitted revision of T1; fix on the existing branches (flybrain#4, site#14).

QA findings accepted by the Orchestrator (summary):
- F1: the headline sync row and predicted values didn't reproduce, because the reported run's nonzero_true100 (356 us) came from a slow regime and was used as the sync unit.
- F2: the index_put_ unit (scalar setitem minus tensor setitem) went negative in a QA run (-42 us, which is -0.17 ms/step).
- F3: some variance went undisclosed: graph replay enqueue reached 162 us, tensor setitem 88 us, the numpy stall fraction 16.3%, and the bench-method torch timing 2.68 ms.
- F4: the CPU micro-benchmarks vary more than "within 1.1x for most".
- F5: the site says graph replay is 2-4x faster while the doc says 2-7x, and the yml says "each 2 ms simulation step", which is ambiguous.

Fix on the existing branches (flybrain#4, site#14); do not open new PRs.
1. F1, robust units and repeated runs. Run the whole micro-benchmark suite N=5 times in one invocation (or 5 invocations, aggregated by a script). For every value, report the median across runs plus min-max. Sync unit for nonzero = median across the three sizes and the 5 runs; do the same for any unit that showed a >2x split. The budget table reports, per rng mode, the median predicted ms/step and its min-max across the 5 runs, next to the median steady-state ms/step and its range.
2. F2, no negative costs. Measure the scalar index-assignment cost directly (e.g. time `v[idx] = const` on a 1-100-element idx at sim size), not as a difference of two medians. If any unit is still derived by subtraction, clamp it at 0 and say so.
3. F3/F4, disclosure. A "Variability" section in perf/week04_day3.md: every value that varied >2x across runs (graph replay enqueue, tensor setitem, nonzero, as_tensor, pinned copy, stall fraction, bench-method ms/step, the CPU items), with its range. Remove "within 1.1x for most" unless the 5 runs support it.
4. Name the regime finding plainly in the conclusions: the 3.3 ms from 2026-11-16 includes stalls and warm-up; steady state is ~1.5-2.0 ms; ~11 ms stalls hit a variable share of steps on WSL. Week 5 should report steady ms/step and stall fraction separately.
5. F5, site: regenerate the table and PNG from the new numbers; make graph-replay speedup wording match the doc; reword the yml summary so it says "about 2 ms of wall time per 0.1 ms simulated step".
6. Update the T1 perf_log entry; push both branches.

=== REVISED QA CRITERIA (replace the T1 "+/-25%" check) ===
- QA runs the suite with the same N=5 aggregation once. For each value: QA's median must fall within Dev's reported min-max widened by 25% on each side. Values Dev flagged as varying >2x are exempt from the numeric match but must be listed in the Variability section.
- Budget: QA recomputes from the JSON (exact match). The median predicted/steady ratio across QA's 5 runs is within 0.8-1.2 for at least one rng mode, with no negative category anywhere.
- Site PR gate: page checks (as before) plus its table and PNG are byte-identical to the flybrain PR's.
- All other T1 checks unchanged. This is the last revision: if it fails again, report and stop.
=== END ===
