# Perf/GPU track team, Week 5 onward

Perf-track variant of [ROLES.md](ROLES.md) (science track). Same four bots and the same loop; only the
differences are written here. This is a working charter, not a plan. The only plan is the central plan
("Central plan: decoding animal minds"); the perf/GPU track is a section of it, and every brief traces to a
row of that section. Week 5 starting point: the targets in [perf/week04.md](../../../perf/week04.md#week-5-targets).

Anchors: every perf task must say which run it speeds up or makes trustworthy on the path to Frank Willett's
speech-BCI data (Week 9 real-data check) and the dog path (Week 10). A task that only moves a benchmark
number with no consumer is flagged to Charlie before it starts.

## Bots (4)

| Bot | Who | Does | Never does |
|---|---|---|---|
| **Lead** | the perf project thread's Claude (runs on the PC over Remote Control) | Turns central-plan perf rows into briefs `perf/briefs/w5_Pn_<name>.md` (Week 4 format: Outcome, Context, Tasks, Hypotheses, Acceptance criteria A1..An, Budget/stop, Deliverable). Writes A1..An before work starts. Schedules GPU runs around the science sims. Updates the perf section of the central plan and reports to Charlie; asks him only on picker cards. | Write production code; merge |
| **Dev** | fresh subagent per brief | Builds on branch `perf/w5-pn-<name>` from main, runs benches on the 5060 Ti (WSL `~/flybench/.venv`, torch cu128), opens a PR (attribution header, assigned to Charlie), reports numbers with paths to the result JSON/MD. Marks each hypothesis holds / partly / does not hold; lists built / assumed / unverified. | Change acceptance criteria; merge; write Charlie's Triton/JAX decay kernels (he writes them; Dev only wires the harness and parity) |
| **QA** | fresh subagent per PR, never the Dev that built it; gets the brief, PR link and head SHA only | Re-runs every criterion from the PR head on the PC: tests (47 on GPU, 44 + 3 skipped on CI), parity, and the timing rows recomputed from raw JSON. Sets the `gpu-qa` commit status. Squash-merges only when `cpu-tests` (Actions) and `gpu-qa` are green on the exact head SHA. | Edit code or commit; merge with a criterion failing or waived without Lead's note |
| **Scout** | subagent, research only | Prior art for the brief at hand (batched sparse gather / SpMM for spiking nets, fused LIF kernels, graph-safe RNG, nsys/CUPTI under WSL), with citations, as a one-pager the Lead turns into a brief. | Run benches; edit repo code; contact anyone outside the project |

## Loop per task
1. Lead writes `perf/briefs/w5_Pn_<name>.md` with A1..An. Perf tasks are numbered P1, P2, ... so they never
   collide with the science track's T numbers.
2. Dev builds, opens PR, reports. 3. QA verifies from scratch; merge if all pass, else back to Dev (max 2
   rounds, then Lead escalates to Charlie on a card). 4. Lead logs the task in `progress/week5/perf_log.md`,
   updates the central plan's perf section and posts a short result.

## Rules carried from Week 4 (perf)
- **Shared machine.** The science track runs one CPU sim at a time on this PC. No timed GPU benchmark while a
  science sim is running (check WSL `ps` and Windows `python` processes first and record the check in the
  result JSON); correctness tests may run alongside. If busy, Dev waits or runs only untimed work.
- Report steady ms/step and stall share as two numbers; 5 runs, median [min-max]; say which clock (events
  raw vs host-rescaled) until that open issue is settled.
- Parity before speed: float64 spike counts equal to the reference over full trials (0 neurons differ).
  float32 results carry the reproducibility caveat.
- `sim/lif_torch*.py` is owned by the perf track; science must not depend on an unmerged change to it.
- Old results moved, never deleted; `results/` stays gitignored except the files a report cites.
