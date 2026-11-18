# Week 4 T1 overhead budget: DESKTOP-4C13T5V_20261118-1101 (5 runs in one invocation)

Every value: median across runs [min-max across runs]. Per run, each timing is itself the median of >= 1,000 repeats (item 1: 100 batches of 10,000 ops; simulator: 1,990 steps).

```
{
 "host": "DESKTOP-4C13T5V",
 "platform": "Linux-6.18.58.1-microsoft-standard-WSL2-x86_64-with-glibc2.39",
 "python": "3.12.3",
 "torch": "2.11.0+cu128",
 "numpy": "2.5.5",
 "cpu_threads": 6,
 "kernel": "6.18.58.1-microsoft-standard-WSL2",
 "gpu": "NVIDIA GeForce RTX 5060 Ti",
 "cc": "sm_120",
 "cuda_runtime": "12.8",
 "cudnn": 91900,
 "driver": "610.88",
 "wsl": true,
 "wsl_version": "WSL version: 3.0.5.0\r\nKernel version: 6.18.58.1-1\r\nWSLg version: 1.0.79\r\nMSRDC version: 1.2.7391\r\nDirect3D version: 1.611.1-81528511\r\nDXCore version: 10.0.26100.1-240331-1435.ge-release\r\nWindows version: 10.0.26200.9974",
 "triton": "3.6.0"
}
```

## Budget (cuda, ms per 0.1-ms step)

| category | rng=numpy | rng=torch |
|---|---|---|
| python+dispatch | 0.16 | 0.16 |
| kernel launch | 1.00 | 1.01 |
| sync round trips (D2H) | 0.38 | 0.47 |
| scalar index_put_ (hidden H2D+sync) | 0.13 | 0.13 |
| as_tensor H2D (pageable) | 0.02 | 0.00 |
| **predicted, units = medians across runs** | **1.70** | **1.77** |
| predicted per run, median [min-max] | 1.89 [1.66-2.40] | 2.04 [1.75-2.62] |
| measured steady state (steps <= 5 ms) | 1.68 [1.56-1.71] | 1.69 [1.60-1.78] |
| measured all steps | 2.70 [1.68-3.90] | 2.31 [2.00-3.73] |
| stall fraction | 0.110 [0.001-0.249] | 0.066 [0.037-0.210] |
| predicted / steady, per run | 1.10 [1.06-1.44] | 1.24 [1.04-1.55] |
| any negative category | False | False |

Unit costs from the medians across runs (us):

```
{
 "c_disp": 1.394,
 "c_enq": 10.913,
 "c_launch": 9.519,
 "launch_clamped": false,
 "total": {
  "nonzero": 106.431,
  "index": 108.34,
  "_local_scalar_dense": 78.825,
  "index_put_": 43.803,
  "_to_copy": 34.671
 },
 "extra": {
  "nonzero": 95.518,
  "index": 86.514,
  "_local_scalar_dense": 66.518,
  "index_put_": 31.496,
  "_to_copy": 22.364
 },
 "clamped": []
}
```

## Variability: values whose min-max across runs differs by more than 2x

| value | median | min | max | max/min |
|---|---|---|---|---|
| cuda 1_launch enqueue_us_per_op (us) | 10.237 | 5.353 | 11.422 | 2.13 |
| cuda 1_launch throughput_us_per_op_with_sync (us) | 10.913 | 5.558 | 12.08 | 2.17 |
| cuda 2_sync add_then_item_us (us) | 78.825 | 73.61 | 163.012 | 2.21 |
| cuda 3_syncops nonzero_n165122_true0_us (us) | 103.435 | 98.048 | 352.444 | 3.59 |
| cuda 3_syncops nonzero_n165122_true100_us (us) | 332.852 | 105.67 | 362.689 | 3.43 |
| cuda 3_syncops nonzero_n165122_true2000_us (us) | 106.431 | 98.457 | 361.103 | 3.67 |
| cuda 3_syncops int_sum_n165122_us (us) | 97.76 | 96.327 | 229.498 | 2.38 |
| cuda 3_syncops mask_index_len20_us (us) | 117.222 | 107.958 | 385.839 | 3.57 |
| cuda 3_syncops setitem_scalar_n165122_k1_us (us) | 44.327 | 36.599 | 160.65 | 4.39 |
| cuda 3_syncops setitem_tensor_n165122_k1_us (us) | 9.297 | 8.586 | 88.258 | 10.28 |
| cuda 3_syncops setitem_scalar_n165122_k20_us (us) | 43.803 | 36.418 | 165.99 | 4.56 |
| cuda 3_syncops setitem_tensor_n165122_k20_us (us) | 8.396 | 5.44 | 83.363 | 15.32 |
| cuda 3_syncops setitem_scalar_n165122_k100_us (us) | 36.669 | 35.977 | 165.549 | 4.6 |
| cuda 3_syncops setitem_tensor_n165122_k100_us (us) | 5.48 | 5.36 | 81.575 | 15.22 |
| cuda 4_h2d pageable_as_tensor_len10_us (us) | 30.566 | 29.971 | 86.545 | 2.89 |
| cuda 4_h2d pageable_as_tensor_len10_with_sync_us (us) | 31.898 | 30.508 | 100.526 | 3.3 |
| cuda 4_h2d pinned_nonblocking_len10_enqueue_us (us) | 5.491 | 5.431 | 86.57 | 15.94 |
| cuda 4_h2d pageable_as_tensor_len1000_us (us) | 35.748 | 29.756 | 91.624 | 3.08 |
| cuda 4_h2d pageable_as_tensor_len1000_with_sync_us (us) | 36.459 | 30.016 | 103.42 | 3.45 |
| cuda 4_h2d pinned_nonblocking_len1000_enqueue_us (us) | 8.378 | 5.49 | 47.934 | 8.73 |
| cuda 4_h2d pinned_nonblocking_len1000_with_sync_us (us) | 52.319 | 50.892 | 103.51 | 2.03 |
| cuda 5_graph graph_replay_with_sync_us (us) | 74.633 | 73.802 | 175.492 | 2.38 |
| cuda 5_graph graph_replay_enqueue_us (us) | 10.621 | 10.098 | 161.231 | 15.97 |
| cuda rng=torch stall fraction | 0.066 | 0.037 | 0.21 | 5.72 |
| cuda rng=numpy stall fraction | 0.11 | 0.001 | 0.249 | 497.4 |
| cuda rng=numpy all-steps mean ms/step | 2.697 | 1.684 | 3.902 | 2.32 |
| cpu rng=torch stall fraction | 0.0 | 0.0 | 0.0 | None |
| cpu rng=numpy stall fraction | 0.0 | 0.0 | 0.004 | None |

## Micro-benchmarks (cuda), us

| item | measurement | median | min | max | max/min | p90 (median) | runs >2x within |
|---|---|---|---|---|---|---|---|
| 1_launch | enqueue_us_per_op | 10.237 | 5.353 | 11.422 | 2.13 | 12.828 | 3/5 |
| 1_launch | throughput_us_per_op_with_sync | 10.913 | 5.558 | 12.08 | 2.17 | 13.41 | 3/5 |
| 1_launch | single_op_interval_us | 4.147 | 4.065 | 4.308 | 1.06 | 5.321 | 0/5 |
| 2_sync | synchronize_idle_us | 2.455 | 1.953 | 2.485 | 1.27 | 2.514 | 0/5 |
| 2_sync | add_then_item_us | 78.825 | 73.61 | 163.012 | 2.21 | 159.922 | 1/5 |
| 2_sync | item_only_us | 44.815 | 42.881 | 83.087 | 1.94 | 45.476 | 0/5 |
| 3_syncops | nonzero_n165122_true0_us | 103.435 | 98.048 | 352.444 | 3.59 | 118.924 | 2/5 |
| 3_syncops | nonzero_n165122_true100_us | 332.852 | 105.67 | 362.689 | 3.43 | 369.372 | 2/5 |
| 3_syncops | nonzero_n165122_true2000_us | 106.431 | 98.457 | 361.103 | 3.67 | 129.795 | 1/5 |
| 3_syncops | int_sum_n165122_us | 97.76 | 96.327 | 229.498 | 2.38 | 108.279 | 1/5 |
| 3_syncops | mask_index_len20_us | 117.222 | 107.958 | 385.839 | 3.57 | 397.705 | 3/5 |
| 3_syncops | mask_index_len311_us | 99.459 | 91.91 | 117.662 | 1.28 | 130.709 | 0/5 |
| 3_syncops | setitem_scalar_n165122_k1_us | 44.327 | 36.599 | 160.65 | 4.39 | 61.574 | 2/5 |
| 3_syncops | setitem_tensor_n165122_k1_us | 9.297 | 8.586 | 88.258 | 10.28 | 11.662 | 0/5 |
| 3_syncops | setitem_scalar_n165122_k20_us | 43.803 | 36.418 | 165.99 | 4.56 | 52.657 | 0/5 |
| 3_syncops | setitem_tensor_n165122_k20_us | 8.396 | 5.44 | 83.363 | 15.32 | 10.619 | 0/5 |
| 3_syncops | setitem_scalar_n165122_k100_us | 36.669 | 35.977 | 165.549 | 4.6 | 46.007 | 0/5 |
| 3_syncops | setitem_tensor_n165122_k100_us | 5.48 | 5.36 | 81.575 | 15.22 | 5.831 | 0/5 |
| 4_h2d | pageable_as_tensor_len10_us | 30.566 | 29.971 | 86.545 | 2.89 | 31.669 | 0/5 |
| 4_h2d | pageable_as_tensor_len10_with_sync_us | 31.898 | 30.508 | 100.526 | 3.3 | 38.251 | 1/5 |
| 4_h2d | pinned_nonblocking_len10_enqueue_us | 5.491 | 5.431 | 86.57 | 15.94 | 6.132 | 0/5 |
| 4_h2d | pinned_nonblocking_len10_with_sync_us | 51.375 | 47.499 | 62.118 | 1.31 | 55.123 | 0/5 |
| 4_h2d | pageable_as_tensor_len100_us | 34.671 | 30.512 | 38.969 | 1.28 | 42.27 | 1/5 |
| 4_h2d | pageable_as_tensor_len100_with_sync_us | 35.545 | 30.697 | 36.539 | 1.19 | 41.148 | 0/5 |
| 4_h2d | pinned_nonblocking_len100_enqueue_us | 8.406 | 5.4 | 8.686 | 1.61 | 10.048 | 0/5 |
| 4_h2d | pinned_nonblocking_len100_with_sync_us | 51.119 | 47.458 | 62.238 | 1.31 | 56.437 | 0/5 |
| 4_h2d | pageable_as_tensor_len1000_us | 35.748 | 29.756 | 91.624 | 3.08 | 43.483 | 1/5 |
| 4_h2d | pageable_as_tensor_len1000_with_sync_us | 36.459 | 30.016 | 103.42 | 3.45 | 43.934 | 1/5 |
| 4_h2d | pinned_nonblocking_len1000_enqueue_us | 8.378 | 5.49 | 47.934 | 8.73 | 12.514 | 0/5 |
| 4_h2d | pinned_nonblocking_len1000_with_sync_us | 52.319 | 50.892 | 103.51 | 2.03 | 61.553 | 0/5 |
| 5_graph | eager_50_with_sync_us | 327.22 | 317.981 | 341.053 | 1.07 | 3878.879 | 3/5 |
| 5_graph | eager_50_enqueue_us | 229.871 | 221.734 | 290.181 | 1.31 | 301.012 | 1/5 |
| 5_graph | graph_replay_with_sync_us | 74.633 | 73.802 | 175.492 | 2.38 | 78.787 | 0/5 |
| 5_graph | graph_replay_enqueue_us | 10.621 | 10.098 | 161.231 | 15.97 | 12.255 | 0/5 |

## Micro-benchmarks (cpu), us

| item | measurement | median | min | max | max/min | p90 (median) | runs >2x within |
|---|---|---|---|---|---|---|---|
| 1_launch | enqueue_us_per_op | 1.394 | 1.379 | 1.464 | 1.06 | 1.42 | 0/5 |
| 1_launch | throughput_us_per_op_with_sync | 1.394 | 1.379 | 1.464 | 1.06 | 1.42 | 0/5 |
| 1_launch | single_op_interval_us | 1.362 | 1.343 | 1.423 | 1.06 | 1.383 | 0/5 |
| 2_sync | add_then_item_us | 1.503 | 1.494 | 1.512 | 1.01 | 1.523 | 0/5 |
| 2_sync | item_only_us | 0.13 | 0.129 | 0.13 | 1.01 | 0.131 | 0/5 |
| 3_syncops | nonzero_n165122_true0_us | 4.889 | 4.85 | 4.899 | 1.01 | 4.969 | 0/5 |
| 3_syncops | nonzero_n165122_true100_us | 16.221 | 16.112 | 16.436 | 1.02 | 26.312 | 0/5 |
| 3_syncops | nonzero_n165122_true2000_us | 30.818 | 18.286 | 31.4 | 1.72 | 38.328 | 0/5 |
| 3_syncops | int_sum_n165122_us | 4.91 | 3.948 | 6.192 | 1.57 | 5.138 | 0/5 |
| 3_syncops | mask_index_len20_us | 3.367 | 3.167 | 5.01 | 1.58 | 3.427 | 2/5 |
| 3_syncops | mask_index_len311_us | 3.637 | 3.627 | 3.685 | 1.02 | 3.697 | 0/5 |
| 3_syncops | setitem_scalar_n165122_k1_us | 1.353 | 1.343 | 2.082 | 1.55 | 1.392 | 0/5 |
| 3_syncops | setitem_tensor_n165122_k1_us | 1.322 | 1.313 | 1.323 | 1.01 | 1.342 | 0/5 |
| 3_syncops | setitem_scalar_n165122_k20_us | 1.383 | 1.365 | 1.392 | 1.02 | 1.413 | 0/5 |
| 3_syncops | setitem_tensor_n165122_k20_us | 1.152 | 1.145 | 1.162 | 1.01 | 1.172 | 0/5 |
| 3_syncops | setitem_scalar_n165122_k100_us | 1.493 | 1.474 | 1.503 | 1.02 | 1.523 | 0/5 |
| 3_syncops | setitem_tensor_n165122_k100_us | 1.253 | 1.245 | 1.273 | 1.02 | 1.273 | 0/5 |
| 4_h2d | pageable_as_tensor_len10_us | 0.836 | 0.812 | 0.842 | 1.04 | 0.847 | 0/5 |
| 4_h2d | pageable_as_tensor_len100_us | 0.832 | 0.812 | 0.842 | 1.04 | 0.852 | 0/5 |
| 4_h2d | pageable_as_tensor_len1000_us | 0.831 | 0.817 | 0.832 | 1.02 | 0.842 | 0/5 |

## Simulator (cuda)

### rng=torch

```
{
 "spikes_per_step": {
  "median": 16.5,
  "min": 16.5,
  "max": 16.5,
  "max_over_min": 1.0,
  "runs": [
   16.5,
   16.5,
   16.5,
   16.5,
   16.5
  ]
 },
 "counts": {
  "aten_ops": 117.51,
  "kernel_ops": 106.45,
  "host_ops": 11.06,
  "syncs": 10.0,
  "h2d_copies": 0.0,
  "d2h_copies": 3.0,
  "nonzero": 1.0,
  "local_scalar_dense": 3.0
 },
 "timing": {
  "steady_mean_ms": {
   "median": 1.693,
   "min": 1.604,
   "max": 1.785
  },
  "stall_fraction": {
   "median": 0.066,
   "min": 0.037,
   "max": 0.21
  },
  "per_step_ms:mean": {
   "median": 2.307,
   "min": 1.999,
   "max": 3.726
  },
  "per_step_ms:median": {
   "median": 1.66,
   "min": 1.565,
   "max": 1.879
  }
 }
}
```

### rng=numpy

```
{
 "spikes_per_step": {
  "median": 17.07,
  "min": 17.07,
  "max": 17.07,
  "max_over_min": 1.0,
  "runs": [
   17.07,
   17.07,
   17.07,
   17.07,
   17.07
  ]
 },
 "counts": {
  "aten_ops": 117.24,
  "kernel_ops": 105.17,
  "host_ops": 12.07,
  "syncs": 10.0,
  "h2d_copies": 1.0,
  "d2h_copies": 3.0,
  "nonzero": 1.0,
  "local_scalar_dense": 3.0
 },
 "timing": {
  "steady_mean_ms": {
   "median": 1.675,
   "min": 1.556,
   "max": 1.711
  },
  "stall_fraction": {
   "median": 0.11,
   "min": 0.001,
   "max": 0.249
  },
  "per_step_ms:mean": {
   "median": 2.697,
   "min": 1.684,
   "max": 3.902
  },
  "per_step_ms:median": {
   "median": 1.666,
   "min": 1.557,
   "max": 1.784
  }
 }
}
```

Syncs by op @ line, rng=torch, run 1 (per step):

```
{
 "_local_scalar_dense @ gather:43": 2.0,
 "nonzero @ run_trial:208": 1.0,
 "index_put_ @ run_trial:211": 1.0,
 "index_put_ @ run_trial:212": 1.0,
 "index @ run_trial:231": 1.0,
 "index @ run_trial:234": 1.0,
 "index_put_ @ run_trial:242": 1.0,
 "index_put_ @ run_trial:243": 1.0,
 "_local_scalar_dense @ run_trial:256": 1.0
}
```

Syncs by op @ line, rng=numpy, run 1 (per step):

```
{
 "_local_scalar_dense @ gather:43": 2.0,
 "nonzero @ run_trial:208": 1.0,
 "index_put_ @ run_trial:211": 1.0,
 "index_put_ @ run_trial:212": 1.0,
 "_to_copy @ run_trial:229": 1.0,
 "index @ run_trial:234": 1.0,
 "index_put_ @ run_trial:242": 1.0,
 "index_put_ @ run_trial:243": 1.0,
 "_local_scalar_dense @ run_trial:256": 1.0
}
```

## Simulator (cpu)

### rng=torch

```
{
 "spikes_per_step": {
  "median": 15.96,
  "min": 15.96,
  "max": 15.96,
  "max_over_min": 1.0,
  "runs": [
   15.96,
   15.96,
   15.96,
   15.96,
   15.96
  ]
 },
 "counts": {
  "aten_ops": 117.16,
  "kernel_ops": 0.0,
  "host_ops": 117.16,
  "syncs": 0.0,
  "h2d_copies": 0.0,
  "d2h_copies": 0.0,
  "nonzero": 1.0,
  "local_scalar_dense": 3.0
 },
 "timing": {
  "steady_mean_ms": {
   "median": 0.596,
   "min": 0.55,
   "max": 0.651
  },
  "stall_fraction": {
   "median": 0.0,
   "min": 0.0,
   "max": 0.0
  },
  "per_step_ms:mean": {
   "median": 0.596,
   "min": 0.55,
   "max": 0.651
  },
  "per_step_ms:median": {
   "median": 0.559,
   "min": 0.512,
   "max": 0.649
  }
 }
}
```

### rng=numpy

```
{
 "spikes_per_step": {
  "median": 17.07,
  "min": 17.07,
  "max": 17.07,
  "max_over_min": 1.0,
  "runs": [
   17.07,
   17.07,
   17.07,
   17.07,
   17.07
  ]
 },
 "counts": {
  "aten_ops": 116.24,
  "kernel_ops": 0.0,
  "host_ops": 116.24,
  "syncs": 0.0,
  "h2d_copies": 0.0,
  "d2h_copies": 0.0,
  "nonzero": 1.0,
  "local_scalar_dense": 3.0
 },
 "timing": {
  "steady_mean_ms": {
   "median": 0.619,
   "min": 0.587,
   "max": 0.649
  },
  "stall_fraction": {
   "median": 0.0,
   "min": 0.0,
   "max": 0.004
  },
  "per_step_ms:mean": {
   "median": 0.62,
   "min": 0.608,
   "max": 0.649
  },
  "per_step_ms:median": {
   "median": 0.592,
   "min": 0.551,
   "max": 0.637
  }
 }
}
```

## Bench-pack method with main's sim/ (0.2 s, float64 eager, setup included, median of 3)

rng='torch': 2.50 [1.95-2.74] ms/step; rng='numpy': 2.50 [1.77-2.77] ms/step; 2026-11-16 bench pack: 3.327 ms/step (rng='torch').
