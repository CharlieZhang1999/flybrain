# Week 4 T1 overhead budget: DESKTOP-4C13T5V_20261118-1034

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
| 1_launch | enqueue_us_per_op | 100 | 8.225 | 12.728 | no |
| 1_launch | throughput_us_per_op_with_sync | 100 | 8.52 | 13.445 | **yes** |
| 1_launch | single_op_interval_us | 100000 | 4.228 | 4.899 | no |
| 2_sync | synchronize_idle_us | 2000 | 1.953 | 2.004 | no |
| 2_sync | add_then_item_us | 2000 | 72.648 | 78.539 | no |
| 2_sync | item_only_us | 2000 | 44.355 | 45.647 | no |
| 3_syncops | nonzero_n165122_true0_us | 1000 | 98.277 | 110.229 | no |
| 3_syncops | nonzero_n165122_true100_us | 1000 | 102.179 | 119.497 | no |
| 3_syncops | nonzero_n165122_true2000_us | 1000 | 98.376 | 114.908 | no |
| 3_syncops | int_sum_n165122_us | 1000 | 94.76 | 102.675 | no |
| 3_syncops | mask_index_len20_us | 1000 | 99.86 | 126.991 | no |
| 3_syncops | mask_index_len311_us | 1000 | 91.639 | 119.908 | no |
| 3_syncops | setitem_scalar_n165122_k20_us | 1000 | 37.456 | 44.625 | no |
| 3_syncops | setitem_tensor_n165122_k20_us | 1000 | 5.601 | 6.162 | no |
| 3_syncops | setitem_scalar_n165122_k200_us | 1000 | 36.525 | 43.803 | no |
| 3_syncops | setitem_tensor_n165122_k200_us | 1000 | 5.44 | 5.761 | no |
| 4_h2d | pageable_as_tensor_len10_us | 1000 | 31.37 | 37.451 | no |
| 4_h2d | pageable_as_tensor_len10_with_sync_us | 1000 | 30.077 | 37.601 | no |
| 4_h2d | pinned_nonblocking_len10_enqueue_us | 1000 | 5.39 | 5.861 | no |
| 4_h2d | pinned_nonblocking_len10_with_sync_us | 1000 | 45.406 | 49.093 | no |
| 4_h2d | pageable_as_tensor_len100_us | 1000 | 30.938 | 37.381 | no |
| 4_h2d | pageable_as_tensor_len100_with_sync_us | 1000 | 30.167 | 35.086 | no |
| 4_h2d | pinned_nonblocking_len100_enqueue_us | 1000 | 5.37 | 5.871 | no |
| 4_h2d | pinned_nonblocking_len100_with_sync_us | 1000 | 46.709 | 47.901 | no |
| 4_h2d | pageable_as_tensor_len1000_us | 1000 | 33.649 | 42.962 | no |
| 4_h2d | pageable_as_tensor_len1000_with_sync_us | 1000 | 36.314 | 42.14 | no |
| 4_h2d | pinned_nonblocking_len1000_enqueue_us | 1000 | 8.256 | 11.501 | no |
| 4_h2d | pinned_nonblocking_len1000_with_sync_us | 1000 | 50.827 | 59.303 | no |
| 5_graph | kernels | | 50 | | |
| 5_graph | eager_50_with_sync_us | 1000 | 339.265 | 583.579 | no |
| 5_graph | eager_50_enqueue_us | 1000 | 284.976 | 325.538 | no |
| 5_graph | graph_replay_with_sync_us | 1000 | 81.495 | 85.643 | no |
| 5_graph | graph_replay_enqueue_us | 1000 | 13.035 | 15.7 | no |
| 5_graph | capture_ok | | True | | |

## Micro-benchmarks (cpu)

| item | measurement | n | median | p90 | 2x flag |
|---|---|---|---|---|---|
| 1_launch | n_ops | | 10000 | | |
| 1_launch | batches | | 100 | | |
| 1_launch | enqueue_us_per_op | 100 | 1.414 | 1.452 | no |
| 1_launch | throughput_us_per_op_with_sync | 100 | 1.414 | 1.452 | no |
| 1_launch | single_op_interval_us | 100000 | 1.383 | 1.403 | no |
| 2_sync | add_then_item_us | 2000 | 1.573 | 1.603 | no |
| 2_sync | item_only_us | 2000 | 0.151 | 0.161 | no |
| 3_syncops | nonzero_n165122_true0_us | 1000 | 7.825 | 8.136 | no |
| 3_syncops | nonzero_n165122_true100_us | 1000 | 29.081 | 33.634 | no |
| 3_syncops | nonzero_n165122_true2000_us | 1000 | 31.149 | 33.875 | no |
| 3_syncops | int_sum_n165122_us | 1000 | 3.918 | 4.018 | no |
| 3_syncops | mask_index_len20_us | 1000 | 3.126 | 3.166 | no |
| 3_syncops | mask_index_len311_us | 1000 | 3.616 | 3.647 | no |
| 3_syncops | setitem_scalar_n165122_k20_us | 1000 | 1.402 | 1.423 | no |
| 3_syncops | setitem_tensor_n165122_k20_us | 1000 | 1.172 | 1.192 | no |
| 3_syncops | setitem_scalar_n165122_k200_us | 1000 | 1.674 | 1.703 | no |
| 3_syncops | setitem_tensor_n165122_k200_us | 1000 | 1.422 | 1.442 | no |
| 4_h2d | pageable_as_tensor_len10_us | 1000 | 0.822 | 0.842 | no |
| 4_h2d | pageable_as_tensor_len100_us | 1000 | 0.831 | 0.842 | no |
| 4_h2d | pageable_as_tensor_len1000_us | 1000 | 0.821 | 0.841 | no |

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
   "median": 1.618,
   "p90": 2.254,
   "mean": 2.008,
   "min": 1.173,
   "chunk_median_ratio": 1.2,
   "p90_over_median": 1.39,
   "varies_2x": false
  },
  "run_mean_ms": {
   "n": 10,
   "median": 1.773,
   "p90": 2.744,
   "mean": 2.008,
   "min": 1.494,
   "chunk_median_ratio": 1.59,
   "p90_over_median": 1.55,
   "varies_2x": false
  },
  "stall_threshold_ms": 5.0,
  "stall_steps": 76,
  "stall_fraction": 0.0382,
  "stall_median_ms": 11.179,
  "steady_mean_ms": 1.676,
  "stall_ms_per_step": 0.396
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
   "median": 1.629,
   "p90": 2.409,
   "mean": 2.201,
   "min": 1.115,
   "chunk_median_ratio": 1.17,
   "p90_over_median": 1.48,
   "varies_2x": false
  },
  "run_mean_ms": {
   "n": 10,
   "median": 1.797,
   "p90": 3.242,
   "mean": 2.201,
   "min": 1.555,
   "chunk_median_ratio": 1.53,
   "p90_over_median": 1.8,
   "varies_2x": false
  },
  "stall_threshold_ms": 5.0,
  "stall_steps": 123,
  "stall_fraction": 0.0618,
  "stall_median_ms": 10.71,
  "steady_mean_ms": 1.654,
  "stall_ms_per_step": 0.649
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
   "median": 0.528,
   "p90": 0.747,
   "mean": 0.56,
   "min": 0.345,
   "chunk_median_ratio": 1.15,
   "p90_over_median": 1.42,
   "varies_2x": false
  },
  "run_mean_ms": {
   "n": 10,
   "median": 0.555,
   "p90": 0.582,
   "mean": 0.56,
   "min": 0.504,
   "chunk_median_ratio": 1.17,
   "p90_over_median": 1.05,
   "varies_2x": false
  },
  "stall_threshold_ms": 5.0,
  "stall_steps": 0,
  "stall_fraction": 0.0,
  "stall_median_ms": null,
  "steady_mean_ms": 0.56,
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
   "median": 0.528,
   "p90": 0.795,
   "mean": 0.572,
   "min": 0.351,
   "chunk_median_ratio": 1.14,
   "p90_over_median": 1.5,
   "varies_2x": false
  },
  "run_mean_ms": {
   "n": 10,
   "median": 0.576,
   "p90": 0.593,
   "mean": 0.572,
   "min": 0.499,
   "chunk_median_ratio": 1.12,
   "p90_over_median": 1.03,
   "varies_2x": false
  },
  "stall_threshold_ms": 5.0,
  "stall_steps": 0,
  "stall_fraction": 0.0,
  "stall_median_ms": null,
  "steady_mean_ms": 0.572,
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
  "ms_per_step_incl_setup": 2.433,
  "walls_s": [
   5.74,
   4.867,
   4.813
  ],
  "per_step_interval_ms": {
   "n": 5997,
   "median": 1.78,
   "p90": 2.694,
   "mean": 2.49,
   "min": 0.82,
   "chunk_median_ratio": 1.14,
   "p90_over_median": 1.51,
   "varies_2x": false
  }
 },
 "numpy": {
  "bio_s": 0.2,
  "repeats": 3,
  "ms_per_step_incl_setup": 1.899,
  "walls_s": [
   4.794,
   3.798,
   3.798
  ],
  "per_step_interval_ms": {
   "n": 5997,
   "median": 1.511,
   "p90": 2.04,
   "mean": 2.006,
   "min": 0.827,
   "chunk_median_ratio": 1.14,
   "p90_over_median": 1.35,
   "varies_2x": false
  }
 },
 "reference_ms_per_step_bench_pack": 3.327
}
```

## Budget (cuda)

### rng=numpy: predicted 1.418 ms vs measured mean 2.201 ms (ratio 0.644), steady-state mean 1.654 ms (ratio 0.858); stalls 0.649 ms/step (6.2% of steps)

| category | ms/step |
|---|---|
| python+dispatch | 0.1658 |
| kernel launch | 0.7473 |
| sync round trips (D2H) | 0.3564 |
| scalar index_put_ (hidden H2D+sync) | 0.1274 |
| as_tensor H2D (pageable) | 0.0214 |

```
{
 "unit_costs_us": {
  "c_disp": 1.414,
  "c_enq": 8.52,
  "c_launch": 7.106,
  "extra_nonzero": 93.659,
  "extra_index": 74.599,
  "extra__local_scalar_dense": 62.714,
  "extra_index_put_": 31.855,
  "extra__to_copy": 21.436
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

### rng=torch: predicted 1.481 ms vs measured mean 2.008 ms (ratio 0.738), steady-state mean 1.676 ms (ratio 0.884); stalls 0.396 ms/step (3.8% of steps)

| category | ms/step |
|---|---|
| python+dispatch | 0.1662 |
| kernel launch | 0.7564 |
| sync round trips (D2H) | 0.431 |
| scalar index_put_ (hidden H2D+sync) | 0.1274 |
| as_tensor H2D (pageable) | 0.0 |

```
{
 "unit_costs_us": {
  "c_disp": 1.414,
  "c_enq": 8.52,
  "c_launch": 7.106,
  "extra_nonzero": 93.659,
  "extra_index": 74.599,
  "extra__local_scalar_dense": 62.714,
  "extra_index_put_": 31.855,
  "extra__to_copy": 21.436
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
