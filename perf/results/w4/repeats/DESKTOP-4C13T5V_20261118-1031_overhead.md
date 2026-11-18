# Week 4 T1 overhead budget: DESKTOP-4C13T5V_20261118-1031

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
| 1_launch | enqueue_us_per_op | 100 | 11.094 | 13.186 | no |
| 1_launch | throughput_us_per_op_with_sync | 100 | 11.708 | 13.803 | no |
| 1_launch | single_op_interval_us | 100000 | 4.512 | 6.34 | no |
| 2_sync | synchronize_idle_us | 2000 | 1.92 | 1.96 | no |
| 2_sync | add_then_item_us | 2000 | 78.902 | 85.724 | no |
| 2_sync | item_only_us | 2000 | 45.732 | 48.008 | no |
| 3_syncops | nonzero_n165122_true0_us | 1000 | 99.298 | 113.293 | no |
| 3_syncops | nonzero_n165122_true100_us | 1000 | 98.727 | 110.872 | no |
| 3_syncops | nonzero_n165122_true2000_us | 1000 | 105.29 | 363.777 | **yes** |
| 3_syncops | int_sum_n165122_us | 1000 | 105.145 | 235.459 | **yes** |
| 3_syncops | mask_index_len20_us | 1000 | 111.602 | 134.846 | no |
| 3_syncops | mask_index_len311_us | 1000 | 107.171 | 125.601 | no |
| 3_syncops | setitem_scalar_n165122_k20_us | 1000 | 42.274 | 49.719 | no |
| 3_syncops | setitem_tensor_n165122_k20_us | 1000 | 8.154 | 9.655 | no |
| 3_syncops | setitem_scalar_n165122_k200_us | 1000 | 43.345 | 50.829 | no |
| 3_syncops | setitem_tensor_n165122_k200_us | 1000 | 7.755 | 10.146 | no |
| 4_h2d | pageable_as_tensor_len10_us | 1000 | 31.908 | 39.102 | no |
| 4_h2d | pageable_as_tensor_len10_with_sync_us | 1000 | 33.029 | 38.652 | no |
| 4_h2d | pinned_nonblocking_len10_enqueue_us | 1000 | 5.413 | 5.914 | no |
| 4_h2d | pinned_nonblocking_len10_with_sync_us | 1000 | 50.254 | 55.722 | no |
| 4_h2d | pageable_as_tensor_len100_us | 1000 | 33.865 | 42.894 | no |
| 4_h2d | pageable_as_tensor_len100_with_sync_us | 1000 | 35.23 | 44.285 | no |
| 4_h2d | pinned_nonblocking_len100_enqueue_us | 1000 | 8.024 | 12.177 | no |
| 4_h2d | pinned_nonblocking_len100_with_sync_us | 1000 | 54.326 | 60.615 | no |
| 4_h2d | pageable_as_tensor_len1000_us | 1000 | 34.69 | 43.285 | no |
| 4_h2d | pageable_as_tensor_len1000_with_sync_us | 1000 | 35.105 | 41.724 | no |
| 4_h2d | pinned_nonblocking_len1000_enqueue_us | 1000 | 8.225 | 11.207 | no |
| 4_h2d | pinned_nonblocking_len1000_with_sync_us | 1000 | 53.45 | 58.903 | no |
| 5_graph | kernels | | 50 | | |
| 5_graph | eager_50_with_sync_us | 1000 | 394.604 | 1147.042 | **yes** |
| 5_graph | eager_50_enqueue_us | 1000 | 225.126 | 256.385 | no |
| 5_graph | graph_replay_with_sync_us | 1000 | 76.073 | 179.445 | **yes** |
| 5_graph | graph_replay_enqueue_us | 1000 | 10.35 | 11.343 | no |
| 5_graph | capture_ok | | True | | |

## Micro-benchmarks (cpu)

| item | measurement | n | median | p90 | 2x flag |
|---|---|---|---|---|---|
| 1_launch | n_ops | | 10000 | | |
| 1_launch | batches | | 100 | | |
| 1_launch | enqueue_us_per_op | 100 | 1.525 | 1.538 | no |
| 1_launch | throughput_us_per_op_with_sync | 100 | 1.525 | 1.538 | no |
| 1_launch | single_op_interval_us | 100000 | 1.48 | 1.501 | no |
| 2_sync | add_then_item_us | 2000 | 1.592 | 1.612 | no |
| 2_sync | item_only_us | 2000 | 0.152 | 0.162 | no |
| 3_syncops | nonzero_n165122_true0_us | 1000 | 4.927 | 5.019 | no |
| 3_syncops | nonzero_n165122_true100_us | 1000 | 16.323 | 26.278 | no |
| 3_syncops | nonzero_n165122_true2000_us | 1000 | 23.536 | 36.164 | no |
| 3_syncops | int_sum_n165122_us | 1000 | 3.995 | 4.999 | no |
| 3_syncops | mask_index_len20_us | 1000 | 3.376 | 3.417 | no |
| 3_syncops | mask_index_len311_us | 1000 | 3.67 | 3.731 | no |
| 3_syncops | setitem_scalar_n165122_k20_us | 1000 | 1.419 | 1.44 | no |
| 3_syncops | setitem_tensor_n165122_k20_us | 1000 | 1.156 | 1.176 | no |
| 3_syncops | setitem_scalar_n165122_k200_us | 1000 | 1.663 | 1.693 | no |
| 3_syncops | setitem_tensor_n165122_k200_us | 1000 | 1.409 | 1.439 | no |
| 4_h2d | pageable_as_tensor_len10_us | 1000 | 0.841 | 0.862 | no |
| 4_h2d | pageable_as_tensor_len100_us | 1000 | 0.831 | 0.842 | no |
| 4_h2d | pageable_as_tensor_len1000_us | 1000 | 0.831 | 0.852 | no |

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
   "median": 1.7,
   "p90": 2.376,
   "mean": 2.224,
   "min": 1.209,
   "chunk_median_ratio": 1.31,
   "p90_over_median": 1.4,
   "varies_2x": false
  },
  "run_mean_ms": {
   "n": 10,
   "median": 1.937,
   "p90": 3.662,
   "mean": 2.224,
   "min": 1.521,
   "chunk_median_ratio": 1.81,
   "p90_over_median": 1.89,
   "varies_2x": false
  },
  "stall_threshold_ms": 5.0,
  "stall_steps": 108,
  "stall_fraction": 0.0543,
  "stall_median_ms": 11.279,
  "steady_mean_ms": 1.724,
  "stall_ms_per_step": 0.593
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
   "median": 1.682,
   "p90": 2.486,
   "mean": 2.317,
   "min": 1.131,
   "chunk_median_ratio": 1.16,
   "p90_over_median": 1.48,
   "varies_2x": false
  },
  "run_mean_ms": {
   "n": 10,
   "median": 1.905,
   "p90": 3.198,
   "mean": 2.317,
   "min": 1.569,
   "chunk_median_ratio": 1.77,
   "p90_over_median": 1.68,
   "varies_2x": false
  },
  "stall_threshold_ms": 5.0,
  "stall_steps": 146,
  "stall_fraction": 0.0734,
  "stall_median_ms": 10.705,
  "steady_mean_ms": 1.68,
  "stall_ms_per_step": 0.761
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
   "median": 0.561,
   "p90": 0.809,
   "mean": 0.595,
   "min": 0.339,
   "chunk_median_ratio": 1.4,
   "p90_over_median": 1.44,
   "varies_2x": false
  },
  "run_mean_ms": {
   "n": 10,
   "median": 0.591,
   "p90": 0.679,
   "mean": 0.595,
   "min": 0.464,
   "chunk_median_ratio": 1.27,
   "p90_over_median": 1.15,
   "varies_2x": false
  },
  "stall_threshold_ms": 5.0,
  "stall_steps": 0,
  "stall_fraction": 0.0,
  "stall_median_ms": null,
  "steady_mean_ms": 0.595,
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
   "median": 0.518,
   "p90": 0.766,
   "mean": 0.556,
   "min": 0.34,
   "chunk_median_ratio": 1.26,
   "p90_over_median": 1.48,
   "varies_2x": false
  },
  "run_mean_ms": {
   "n": 10,
   "median": 0.556,
   "p90": 0.581,
   "mean": 0.556,
   "min": 0.451,
   "chunk_median_ratio": 1.21,
   "p90_over_median": 1.05,
   "varies_2x": false
  },
  "stall_threshold_ms": 5.0,
  "stall_steps": 0,
  "stall_fraction": 0.0,
  "stall_median_ms": null,
  "steady_mean_ms": 0.556,
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
  "ms_per_step_incl_setup": 2.491,
  "walls_s": [
   4.663,
   6.122,
   4.982
  ],
  "per_step_interval_ms": {
   "n": 5997,
   "median": 1.513,
   "p90": 6.703,
   "mean": 2.572,
   "min": 0.812,
   "chunk_median_ratio": 1.22,
   "p90_over_median": 4.43,
   "varies_2x": true
  }
 },
 "numpy": {
  "bio_s": 0.2,
  "repeats": 3,
  "ms_per_step_incl_setup": 2.806,
  "walls_s": [
   6.722,
   5.612,
   4.092
  ],
  "per_step_interval_ms": {
   "n": 5997,
   "median": 1.415,
   "p90": 6.345,
   "mean": 2.681,
   "min": 0.664,
   "chunk_median_ratio": 1.23,
   "p90_over_median": 4.48,
   "varies_2x": true
  }
 },
 "reference_ms_per_step_bench_pack": 3.327
}
```

## Budget (cuda)

### rng=numpy: predicted 1.773 ms vs measured mean 2.317 ms (ratio 0.765), steady-state mean 1.68 ms (ratio 1.055); stalls 0.761 ms/step (7.3% of steps)

| category | ms/step |
|---|---|
| python+dispatch | 0.1788 |
| kernel launch | 1.0709 |
| sync round trips (D2H) | 0.3678 |
| scalar index_put_ (hidden H2D+sync) | 0.1365 |
| as_tensor H2D (pageable) | 0.0187 |

```
{
 "unit_costs_us": {
  "c_disp": 1.525,
  "c_enq": 11.708,
  "c_launch": 10.183,
  "extra_nonzero": 87.019,
  "extra_index": 83.755,
  "extra__local_scalar_dense": 65.669,
  "extra_index_put_": 34.12,
  "extra__to_copy": 18.675
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

### rng=torch: predicted 1.851 ms vs measured mean 2.224 ms (ratio 0.832), steady-state mean 1.724 ms (ratio 1.074); stalls 0.593 ms/step (5.4% of steps)

| category | ms/step |
|---|---|
| python+dispatch | 0.1792 |
| kernel launch | 1.084 |
| sync round trips (D2H) | 0.4515 |
| scalar index_put_ (hidden H2D+sync) | 0.1365 |
| as_tensor H2D (pageable) | 0.0 |

```
{
 "unit_costs_us": {
  "c_disp": 1.525,
  "c_enq": 11.708,
  "c_launch": 10.183,
  "extra_nonzero": 87.019,
  "extra_index": 83.755,
  "extra__local_scalar_dense": 65.669,
  "extra_index_put_": 34.12,
  "extra__to_copy": 18.675
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
