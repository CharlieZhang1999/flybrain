# Week 4 T1 overhead budget: DESKTOP-4C13T5V_20261118-1022

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

## Micro-benchmarks (cuda)

| item | measurement | n | median | p90 | 2x flag |
|---|---|---|---|---|---|
| 1_launch | n_ops | | 10000 | | |
| 1_launch | batches | | 100 | | |
| 1_launch | enqueue_us_per_op | 100 | 11.025 | 12.867 | **yes** |
| 1_launch | throughput_us_per_op_with_sync | 100 | 11.728 | 13.537 | **yes** |
| 1_launch | single_op_interval_us | 100000 | 4.209 | 7.917 | no |
| 2_sync | synchronize_idle_us | 2000 | 1.949 | 1.98 | no |
| 2_sync | add_then_item_us | 2000 | 72.408 | 77.214 | no |
| 2_sync | item_only_us | 2000 | 40.341 | 42.834 | no |
| 3_syncops | nonzero_n165122_true0_us | 1000 | 103.5 | 123.543 | no |
| 3_syncops | nonzero_n165122_true100_us | 1000 | 355.873 | 367.492 | no |
| 3_syncops | nonzero_n165122_true2000_us | 1000 | 99.547 | 108.449 | no |
| 3_syncops | int_sum_n165122_us | 1000 | 98.144 | 225.739 | **yes** |
| 3_syncops | mask_index_len20_us | 1000 | 108.755 | 139.774 | no |
| 3_syncops | mask_index_len311_us | 1000 | 116.916 | 150.935 | no |
| 3_syncops | setitem_scalar_n165122_k20_us | 1000 | 45.384 | 53.072 | no |
| 3_syncops | setitem_tensor_n165122_k20_us | 1000 | 7.852 | 9.778 | no |
| 3_syncops | setitem_scalar_n165122_k200_us | 1000 | 44.567 | 52.37 | no |
| 3_syncops | setitem_tensor_n165122_k200_us | 1000 | 8.063 | 12.285 | no |
| 4_h2d | pageable_as_tensor_len10_us | 1000 | 33.987 | 43.033 | no |
| 4_h2d | pageable_as_tensor_len10_with_sync_us | 1000 | 30.878 | 31.641 | no |
| 4_h2d | pinned_nonblocking_len10_enqueue_us | 1000 | 5.496 | 5.977 | no |
| 4_h2d | pinned_nonblocking_len10_with_sync_us | 1000 | 49.842 | 52.891 | no |
| 4_h2d | pageable_as_tensor_len100_us | 1000 | 30.477 | 34.207 | no |
| 4_h2d | pageable_as_tensor_len100_with_sync_us | 1000 | 31.069 | 33.516 | no |
| 4_h2d | pinned_nonblocking_len100_enqueue_us | 1000 | 5.496 | 5.967 | no |
| 4_h2d | pinned_nonblocking_len100_with_sync_us | 1000 | 62.142 | 102.082 | **yes** |
| 4_h2d | pageable_as_tensor_len1000_us | 1000 | 86.598 | 92.91 | no |
| 4_h2d | pageable_as_tensor_len1000_with_sync_us | 1000 | 100.241 | 108.394 | no |
| 4_h2d | pinned_nonblocking_len1000_enqueue_us | 1000 | 48.331 | 58.256 | no |
| 4_h2d | pinned_nonblocking_len1000_with_sync_us | 1000 | 102.92 | 110.833 | no |
| 5_graph | kernels | | 50 | | |
| 5_graph | eager_50_with_sync_us | 1000 | 349.07 | 3976.624 | **yes** |
| 5_graph | eager_50_enqueue_us | 1000 | 266.982 | 4004.043 | **yes** |
| 5_graph | graph_replay_with_sync_us | 1000 | 176.763 | 192.853 | no |
| 5_graph | graph_replay_enqueue_us | 1000 | 10.108 | 161.702 | **yes** |
| 5_graph | capture_ok | | True | | |

## Micro-benchmarks (cpu)

| item | measurement | n | median | p90 | 2x flag |
|---|---|---|---|---|---|
| 1_launch | n_ops | | 10000 | | |
| 1_launch | batches | | 100 | | |
| 1_launch | enqueue_us_per_op | 100 | 1.415 | 1.45 | no |
| 1_launch | throughput_us_per_op_with_sync | 100 | 1.415 | 1.45 | no |
| 1_launch | single_op_interval_us | 100000 | 1.388 | 1.409 | no |
| 2_sync | add_then_item_us | 2000 | 1.539 | 1.56 | no |
| 2_sync | item_only_us | 2000 | 0.13 | 0.13 | no |
| 3_syncops | nonzero_n165122_true0_us | 1000 | 6.597 | 6.847 | no |
| 3_syncops | nonzero_n165122_true100_us | 1000 | 28.867 | 32.845 | no |
| 3_syncops | nonzero_n165122_true2000_us | 1000 | 31.115 | 41.8 | no |
| 3_syncops | int_sum_n165122_us | 1000 | 3.958 | 4.968 | no |
| 3_syncops | mask_index_len20_us | 1000 | 3.109 | 3.148 | no |
| 3_syncops | mask_index_len311_us | 1000 | 3.619 | 3.678 | no |
| 3_syncops | setitem_scalar_n165122_k20_us | 1000 | 1.409 | 2.179 | no |
| 3_syncops | setitem_tensor_n165122_k20_us | 1000 | 1.799 | 1.969 | no |
| 3_syncops | setitem_scalar_n165122_k200_us | 1000 | 2.498 | 3.008 | no |
| 3_syncops | setitem_tensor_n165122_k200_us | 1000 | 2.099 | 2.329 | no |
| 4_h2d | pageable_as_tensor_len10_us | 1000 | 0.84 | 0.85 | no |
| 4_h2d | pageable_as_tensor_len100_us | 1000 | 0.83 | 0.84 | no |
| 4_h2d | pageable_as_tensor_len1000_us | 1000 | 0.83 | 0.85 | no |

## Simulator per-step counts and timing (cuda)

### rng=torch

```
{
 "spikes_per_step": 16.5,
 "counts_per_step": {
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
  "per_step_ms": {
   "n": 1990,
   "median": 1.952,
   "p90": 2.581,
   "mean": 2.039,
   "min": 1.189,
   "chunk_median_ratio": 1.32,
   "p90_over_median": 1.32,
   "varies_2x": false
  },
  "run_mean_ms": {
   "n": 10,
   "median": 2.016,
   "p90": 2.342,
   "mean": 2.039,
   "min": 1.693,
   "chunk_median_ratio": 1.33,
   "p90_over_median": 1.16,
   "varies_2x": false
  },
  "stall_threshold_ms": 5.0,
  "stall_steps": 12,
  "stall_fraction": 0.006,
  "stall_median_ms": 11.213,
  "steady_mean_ms": 1.984,
  "stall_ms_per_step": 0.067
 }
}
```

Syncs by op @ line (per step):

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

H2D by op @ line (per step):

```
{}
```

### rng=numpy

```
{
 "spikes_per_step": 17.07,
 "counts_per_step": {
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
  "per_step_ms": {
   "n": 1990,
   "median": 1.728,
   "p90": 2.239,
   "mean": 1.859,
   "min": 1.172,
   "chunk_median_ratio": 1.28,
   "p90_over_median": 1.3,
   "varies_2x": false
  },
  "run_mean_ms": {
   "n": 10,
   "median": 1.738,
   "p90": 2.047,
   "mean": 1.859,
   "min": 1.582,
   "chunk_median_ratio": 1.37,
   "p90_over_median": 1.18,
   "varies_2x": false
  },
  "stall_threshold_ms": 5.0,
  "stall_steps": 27,
  "stall_fraction": 0.0136,
  "stall_median_ms": 11.014,
  "steady_mean_ms": 1.735,
  "stall_ms_per_step": 0.147
 }
}
```

Syncs by op @ line (per step):

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

H2D by op @ line (per step):

```
{
 "_to_copy @ run_trial:229": 1.0
}
```

## Simulator per-step counts and timing (cpu)

### rng=torch

```
{
 "spikes_per_step": 15.96,
 "counts_per_step": {
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
  "per_step_ms": {
   "n": 1990,
   "median": 0.697,
   "p90": 0.85,
   "mean": 0.69,
   "min": 0.429,
   "chunk_median_ratio": 1.26,
   "p90_over_median": 1.22,
   "varies_2x": false
  },
  "run_mean_ms": {
   "n": 10,
   "median": 0.703,
   "p90": 0.762,
   "mean": 0.69,
   "min": 0.58,
   "chunk_median_ratio": 1.22,
   "p90_over_median": 1.08,
   "varies_2x": false
  },
  "stall_threshold_ms": 5.0,
  "stall_steps": 0,
  "stall_fraction": 0.0,
  "stall_median_ms": null,
  "steady_mean_ms": 0.69,
  "stall_ms_per_step": 0.0
 }
}
```

Syncs by op @ line (per step):

```
{}
```

H2D by op @ line (per step):

```
{}
```

### rng=numpy

```
{
 "spikes_per_step": 17.07,
 "counts_per_step": {
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
  "per_step_ms": {
   "n": 1990,
   "median": 0.654,
   "p90": 0.842,
   "mean": 0.657,
   "min": 0.348,
   "chunk_median_ratio": 1.34,
   "p90_over_median": 1.29,
   "varies_2x": false
  },
  "run_mean_ms": {
   "n": 10,
   "median": 0.656,
   "p90": 0.723,
   "mean": 0.657,
   "min": 0.556,
   "chunk_median_ratio": 1.22,
   "p90_over_median": 1.1,
   "varies_2x": false
  },
  "stall_threshold_ms": 5.0,
  "stall_steps": 0,
  "stall_fraction": 0.0,
  "stall_median_ms": null,
  "steady_mean_ms": 0.657,
  "stall_ms_per_step": 0.0
 }
}
```

Syncs by op @ line (per step):

```
{}
```

H2D by op @ line (per step):

```
{}
```

## Bench-pack method with main's sim/ (0.2 s, setup included)

```
{
 "torch": {
  "bio_s": 0.2,
  "repeats": 3,
  "ms_per_step_incl_setup": 1.908,
  "walls_s": [
   3.816,
   3.446,
   3.835
  ],
  "per_step_interval_ms": {
   "n": 5997,
   "median": 1.576,
   "p90": 2.119,
   "mean": 1.791,
   "min": 0.842,
   "chunk_median_ratio": 1.23,
   "p90_over_median": 1.34,
   "varies_2x": false
  }
 },
 "numpy": {
  "bio_s": 0.2,
  "repeats": 3,
  "ms_per_step_incl_setup": 1.886,
  "walls_s": [
   3.773,
   3.573,
   4.348
  ],
  "per_step_interval_ms": {
   "n": 5997,
   "median": 1.583,
   "p90": 2.19,
   "mean": 1.891,
   "min": 0.684,
   "chunk_median_ratio": 1.16,
   "p90_over_median": 1.38,
   "varies_2x": false
  }
 },
 "reference_ms_per_step_bench_pack": 3.327
}
```

## Budget (cuda)

### rng=numpy: predicted 2.037 ms vs measured mean 1.859 ms (ratio 1.096), steady-state mean 1.735 ms (ratio 1.174); stalls 0.147 ms/step (1.4% of steps)

| category | ms/step |
|---|---|
| python+dispatch | 0.1659 |
| kernel launch | 1.0846 |
| sync round trips (D2H) | 0.6154 |
| scalar index_put_ (hidden H2D+sync) | 0.1501 |
| as_tensor H2D (pageable) | 0.0208 |

```
{
 "unit_costs_us": {
  "c_disp": 1.415,
  "c_enq": 11.728,
  "c_launch": 10.313,
  "extra_nonzero": 344.145,
  "extra_index": 93.46,
  "extra__local_scalar_dense": 59.265,
  "extra_index_put_": 37.532,
  "extra__to_copy": 20.844
 },
 "counts_per_step": {
  "aten_ops": 117.24,
  "kernel_ops": 105.17,
  "syncs_by_op": {
   "_local_scalar_dense": 3.0,
   "nonzero": 1.0,
   "index_put_": 4.0,
   "_to_copy": 1.0,
   "index": 1.0
  },
  "h2d": 1.0
 }
}
```

### rng=torch: predicted 2.123 ms vs measured mean 2.039 ms (ratio 1.041), steady-state mean 1.984 ms (ratio 1.07); stalls 0.067 ms/step (0.6% of steps)

| category | ms/step |
|---|---|
| python+dispatch | 0.1663 |
| kernel launch | 1.0978 |
| sync round trips (D2H) | 0.7089 |
| scalar index_put_ (hidden H2D+sync) | 0.1501 |
| as_tensor H2D (pageable) | 0.0 |

```
{
 "unit_costs_us": {
  "c_disp": 1.415,
  "c_enq": 11.728,
  "c_launch": 10.313,
  "extra_nonzero": 344.145,
  "extra_index": 93.46,
  "extra__local_scalar_dense": 59.265,
  "extra_index_put_": 37.532,
  "extra__to_copy": 20.844
 },
 "counts_per_step": {
  "aten_ops": 117.51,
  "kernel_ops": 106.45,
  "syncs_by_op": {
   "_local_scalar_dense": 3.0,
   "nonzero": 1.0,
   "index_put_": 4.0,
   "index": 2.0
  },
  "h2d": 0
 }
}
```
