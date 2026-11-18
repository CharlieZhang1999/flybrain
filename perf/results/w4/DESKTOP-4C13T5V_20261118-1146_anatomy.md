# Week 4 T3 step anatomy (DESKTOP-4C13T5V_20261118-1146)

Env: NVIDIA GeForce RTX 5060 Ti sm_120, torch 2.11.0+cu128, driver 610.88, WSL=True. rng=numpy, float64, stim=looming, warm 1000 steps, window 200 steps x 10 repeats, 5 runs.

Drift guard: {'steps': 300, 'spikes_ref': 3710, 'spikes_instrumented': 3710, 'neurons_differing': 0, 'passed': True}

Every value: median across runs [min-max across runs]; per run, each value is the median over steady steps (<= 5 ms) of the window. wall = host-only pass (perf_counter); "events pass" = perf_counter + CUDA events + NVTX. Ratios to wall use the uninstrumented run_trial steady median.

| phase | wall ms | wall p90 ms | wall ms, events pass | GPU span ms (events pass) | GPU exec ms, non-sync ops (gated) | sync ops span ms (gated) | syncs | GPU aten ops | aten ops |
|---|---|---|---|---|---|---|---|---|---|
| decay | 0.097 [0.093-0.113] | 0.139 [0.134-0.161] | 0.126 [0.106-0.134] | 0.161 [0.160-0.179] | 0.084 [0.083-0.084] | 0.000 [0.000-0.000] | 0 | 14 | 15 |
| gap | 0.072 [0.070-0.082] | 0.107 [0.100-0.116] | 0.092 [0.080-0.104] | 0.105 [0.098-0.111] | 0.056 [0.055-0.057] | 0.000 [0.000-0.000] | 0 | 9 | 10 |
| threshold+nonzero | 0.151 [0.141-0.211] | 0.254 [0.236-0.280] | 0.192 [0.166-0.215] | 0.140 [0.140-0.150] | 0.010 [0.010-0.010] | 0.151 [0.133-0.169] | 1 | 3 | 4 |
| bookkeeping | 0.073 [0.072-0.077] | 0.106 [0.101-0.114] | 0.109 [0.107-0.112] | 0.107 [0.105-0.109] | 0.000 [0.000-0.000] | 0.158 [0.149-0.223] | 2 | 2 | 4 |
| presyn | 0.324 [0.316-0.341] | 0.422 [0.407-0.465] | 0.345 [0.328-0.364] | 0.385 [0.344-0.460] | 0.158 [0.157-0.159] | 0.120 [0.106-0.130] | 1 | 29.171 | 29.171 |
| synaptic | 0.308 [0.306-0.416] | 0.471 [0.459-0.566] | 0.334 [0.293-0.438] | 0.339 [0.304-0.395] | 0.146 [0.145-0.148] | 0.132 [0.114-0.143] | 1 | 25 | 26 |
| drive | 0.117 [0.113-0.130] | 0.175 [0.165-0.204] | 0.136 [0.123-0.152] | 0.123 [0.114-0.131] | 0.027 [0.026-0.027] | 0.067 [0.058-0.085] | 1 | 6 | 9.065 |
| spikelets | 0.121 [0.121-0.135] | 0.181 [0.177-0.191] | 0.133 [0.128-0.160] | 0.127 [0.125-0.148] | 0.010 [0.010-0.010] | 0.148 [0.135-0.159] | 1 | 3 | 3 |
| apl | 0.161 [0.148-0.174] | 0.209 [0.204-0.226] | 0.183 [0.176-0.217] | 0.216 [0.212-0.230] | 0.047 [0.047-0.047] | 0.108 [0.104-0.125] | 1 | 9 | 10 |
| reset | 0.101 [0.097-0.109] | 0.139 [0.135-0.153] | 0.108 [0.103-0.124] | 0.102 [0.091-0.125] | 0.015 [0.015-0.016] | 0.164 [0.160-0.191] | 2 | 5 | 7 |
| ring | 0.000 [0.000-0.000] | 0.001 [0.001-0.001] | 0.009 [0.008-0.011] | 0.002 [0.002-0.003] | 0.000 [0.000-0.000] | 0.000 [0.000-0.000] | 0 | 0 | 0 |
| (loop) | 0.000 [0.000-0.001] | 0.001 [0.001-0.001] | 0.009 [0.008-0.014] | 0.002 [0.002-0.002] | - | - | - | - | - |

| step | value |
|---|---|
| reference_steady_median_ms | 1.609 [1.552-1.864] |
| reference_steady_mean_ms | 1.671 [1.609-1.905] |
| reference_stall_fraction | 0.050 [0.030-0.181] |
| host_steady_median_ms | 1.568 [1.520-1.806] |
| host_steady_mean_ms | 1.636 [1.611-1.813] |
| host_stall_fraction | 0.078 [0.021-0.149] |
| host_stall_median_ms | 10.799 [10.635-10.807] |
| sum_phase_wall_medians_ms | 1.519 [1.488-1.741] |
| sum_over_steady | 0.973 [0.964-0.979] |
| sum_incl_loop_over_steady | 0.973 [0.964-0.979] |
| host_over_reference | 0.998 [0.816-1.119] |
| full_steady_median_ms | 1.855 [1.716-2.045] |
| full_steady_mean_ms | 1.878 [1.756-2.107] |
| full_stall_fraction | 0.063 [0.047-0.211] |
| full_sum_over_steady | 0.961 [0.958-0.987] |
| full_over_reference | 1.153 [0.921-1.290] |
| gpu_span_over_wall | 1.000 [0.999-1.002] |
| gpu_exec_nonsync_step_ms | 0.558 [0.555-0.560] |
| gpu_sync_ops_span_step_ms | 1.096 [0.963-1.224] |
| gpu_exec_nonsync_over_wall | 0.345 [0.298-0.359] |
| gpu_upper_over_wall | 1.029 [0.856-1.128] |
| util_median_pct | 19.000 [3.000-22.000] |
| util_idle_before_pct | 0.000 [0.000-0.000] |
| util_window_s | 3.198 [2.417-8.441] |
| util_clock_median_mhz | 2767.000 [2760.000-2767.000] |
| gated_clock_median_mhz | 2767.000 [2767.000-2767.000] |
| gated_clock_min_mhz | 2587.000 [1687.000-2767.000] |
| host_pass_clock_min_mhz | 2220.000 [2062.000-2407.000] |
| kernels_total | 105.170 [105.170-105.170] |
| syncs_total | 10.000 [10.000-10.000] |
| aten_ops_total | 117.240 [117.240-117.240] |

Top 3 phases by wall median: presyn, synaptic, apl

## Variability (max/min across runs > 2)

| value | median | min | max | max/min |
|---|---|---|---|---|
| step reference_stall_fraction | 0.0503 | 0.0302 | 0.1809 | 5.99 |
| step host_stall_fraction | 0.0784 | 0.0211 | 0.1492 | 7.07 |
| step full_stall_fraction | 0.0628 | 0.0472 | 0.2111 | 4.47 |
| step util_median_pct | 19.0 | 3.0 | 22.0 | 7.33 |
| step util_idle_before_pct | 0.0 | 0.0 | 0.0 | None |
| step util_window_s | 3.198 | 2.417 | 8.441 | 3.49 |

Syncs by phase / op (run 1, per step):

```
{
 "bookkeeping / index_put_": 2.0,
 "reset / index_put_": 2.0,
 "threshold+nonzero / nonzero": 1.0,
 "presyn / _local_scalar_dense": 1.0,
 "synaptic / _local_scalar_dense": 1.0,
 "drive / _to_copy": 1.0,
 "spikelets / index": 1.0,
 "apl / _local_scalar_dense": 1.0
}
```
