# Week 4 T1 overhead budget: DESKTOP-4C13T5V_20261118-1029

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
| 1_launch | enqueue_us_per_op | 100 | 5.641 | 13.19 | **yes** |
| 1_launch | throughput_us_per_op_with_sync | 100 | 5.744 | 13.574 | **yes** |
| 1_launch | single_op_interval_us | 100000 | 4.242 | 5.933 | no |
| 2_sync | synchronize_idle_us | 2000 | 1.933 | 1.973 | no |
| 2_sync | add_then_item_us | 2000 | 74.401 | 79.658 | no |
| 2_sync | item_only_us | 2000 | 42.229 | 43.1 | no |
| 3_syncops | nonzero_n165122_true0_us | 1000 | 360.012 | 372.442 | no |
| 3_syncops | nonzero_n165122_true100_us | 1000 | 359.881 | 370.188 | **yes** |
| 3_syncops | nonzero_n165122_true2000_us | 1000 | 99.26 | 353.739 | **yes** |
| 3_syncops | int_sum_n165122_us | 1000 | 97.057 | 100.948 | no |
| 3_syncops | mask_index_len20_us | 1000 | 92.704 | 123.115 | no |
| 3_syncops | mask_index_len311_us | 1000 | 91.206 | 119.619 | no |
| 3_syncops | setitem_scalar_n165122_k20_us | 1000 | 36.821 | 43.362 | no |
| 3_syncops | setitem_tensor_n165122_k20_us | 1000 | 5.46 | 5.84 | no |
| 3_syncops | setitem_scalar_n165122_k200_us | 1000 | 36.24 | 41.198 | no |
| 3_syncops | setitem_tensor_n165122_k200_us | 1000 | 5.479 | 5.98 | no |
| 4_h2d | pageable_as_tensor_len10_us | 1000 | 30.811 | 35.179 | no |
| 4_h2d | pageable_as_tensor_len10_with_sync_us | 1000 | 31.002 | 42.621 | no |
| 4_h2d | pinned_nonblocking_len10_enqueue_us | 1000 | 5.55 | 6.631 | no |
| 4_h2d | pinned_nonblocking_len10_with_sync_us | 1000 | 47.992 | 50.556 | no |
| 4_h2d | pageable_as_tensor_len100_us | 1000 | 30.943 | 32.395 | no |
| 4_h2d | pageable_as_tensor_len100_with_sync_us | 1000 | 31.103 | 32.876 | no |
| 4_h2d | pinned_nonblocking_len100_enqueue_us | 1000 | 5.51 | 6.471 | no |
| 4_h2d | pinned_nonblocking_len100_with_sync_us | 1000 | 48.362 | 50.998 | no |
| 4_h2d | pageable_as_tensor_len1000_us | 1000 | 30.737 | 34.128 | no |
| 4_h2d | pageable_as_tensor_len1000_with_sync_us | 1000 | 30.908 | 34.008 | no |
| 4_h2d | pinned_nonblocking_len1000_enqueue_us | 1000 | 5.61 | 6.652 | no |
| 4_h2d | pinned_nonblocking_len1000_with_sync_us | 1000 | 50.231 | 52.089 | no |
| 5_graph | kernels | | 50 | | |
| 5_graph | eager_50_with_sync_us | 1000 | 556.511 | 600.816 | no |
| 5_graph | eager_50_enqueue_us | 1000 | 238.947 | 316.472 | no |
| 5_graph | graph_replay_with_sync_us | 1000 | 81.262 | 87.734 | no |
| 5_graph | graph_replay_enqueue_us | 1000 | 10.609 | 13.935 | no |
| 5_graph | capture_ok | | True | | |

## Micro-benchmarks (cpu)

| item | measurement | n | median | p90 | 2x flag |
|---|---|---|---|---|---|
| 1_launch | n_ops | | 10000 | | |
| 1_launch | batches | | 100 | | |
| 1_launch | enqueue_us_per_op | 100 | 1.433 | 1.47 | no |
| 1_launch | throughput_us_per_op_with_sync | 100 | 1.433 | 1.47 | no |
| 1_launch | single_op_interval_us | 100000 | 1.402 | 1.422 | no |
| 2_sync | add_then_item_us | 2000 | 1.573 | 1.593 | no |
| 2_sync | item_only_us | 2000 | 0.151 | 0.161 | no |
| 3_syncops | nonzero_n165122_true0_us | 1000 | 4.859 | 4.949 | no |
| 3_syncops | nonzero_n165122_true100_us | 1000 | 16.27 | 30.024 | no |
| 3_syncops | nonzero_n165122_true2000_us | 1000 | 23.348 | 32.719 | no |
| 3_syncops | int_sum_n165122_us | 1000 | 4.047 | 4.147 | no |
| 3_syncops | mask_index_len20_us | 1000 | 3.106 | 3.156 | no |
| 3_syncops | mask_index_len311_us | 1000 | 3.577 | 3.617 | no |
| 3_syncops | setitem_scalar_n165122_k20_us | 1000 | 1.402 | 1.442 | no |
| 3_syncops | setitem_tensor_n165122_k20_us | 1000 | 1.152 | 1.172 | no |
| 3_syncops | setitem_scalar_n165122_k200_us | 1000 | 1.683 | 1.713 | no |
| 3_syncops | setitem_tensor_n165122_k200_us | 1000 | 1.433 | 1.463 | no |
| 4_h2d | pageable_as_tensor_len10_us | 1000 | 0.831 | 0.842 | no |
| 4_h2d | pageable_as_tensor_len100_us | 1000 | 0.822 | 0.832 | no |
| 4_h2d | pageable_as_tensor_len1000_us | 1000 | 0.821 | 0.832 | no |

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
   "median": 1.72,
   "p90": 2.7,
   "mean": 2.561,
   "min": 1.131,
   "chunk_median_ratio": 1.16,
   "p90_over_median": 1.57,
   "varies_2x": false
  },
  "run_mean_ms": {
   "n": 10,
   "median": 2.529,
   "p90": 3.018,
   "mean": 2.561,
   "min": 1.75,
   "chunk_median_ratio": 1.81,
   "p90_over_median": 1.19,
   "varies_2x": false
  },
  "stall_threshold_ms": 5.0,
  "stall_steps": 187,
  "stall_fraction": 0.094,
  "stall_median_ms": 10.949,
  "steady_mean_ms": 1.71,
  "stall_ms_per_step": 1.011
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
   "median": 1.73,
   "p90": 2.308,
   "mean": 1.835,
   "min": 1.165,
   "chunk_median_ratio": 1.29,
   "p90_over_median": 1.33,
   "varies_2x": false
  },
  "run_mean_ms": {
   "n": 10,
   "median": 1.786,
   "p90": 2.108,
   "mean": 1.835,
   "min": 1.489,
   "chunk_median_ratio": 1.41,
   "p90_over_median": 1.18,
   "varies_2x": false
  },
  "stall_threshold_ms": 5.0,
  "stall_steps": 17,
  "stall_fraction": 0.0085,
  "stall_median_ms": 10.913,
  "steady_mean_ms": 1.757,
  "stall_ms_per_step": 0.094
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
   "median": 0.566,
   "p90": 0.787,
   "mean": 0.596,
   "min": 0.344,
   "chunk_median_ratio": 1.28,
   "p90_over_median": 1.39,
   "varies_2x": false
  },
  "run_mean_ms": {
   "n": 10,
   "median": 0.605,
   "p90": 0.656,
   "mean": 0.596,
   "min": 0.526,
   "chunk_median_ratio": 1.15,
   "p90_over_median": 1.09,
   "varies_2x": false
  },
  "stall_threshold_ms": 5.0,
  "stall_steps": 0,
  "stall_fraction": 0.0,
  "stall_median_ms": null,
  "steady_mean_ms": 0.596,
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
   "median": 0.549,
   "p90": 0.8,
   "mean": 0.589,
   "min": 0.336,
   "chunk_median_ratio": 1.37,
   "p90_over_median": 1.46,
   "varies_2x": false
  },
  "run_mean_ms": {
   "n": 10,
   "median": 0.57,
   "p90": 0.679,
   "mean": 0.589,
   "min": 0.477,
   "chunk_median_ratio": 1.28,
   "p90_over_median": 1.19,
   "varies_2x": false
  },
  "stall_threshold_ms": 5.0,
  "stall_steps": 0,
  "stall_fraction": 0.0,
  "stall_median_ms": null,
  "steady_mean_ms": 0.589,
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
  "ms_per_step_incl_setup": 1.958,
  "walls_s": [
   3.916,
   3.9,
   3.996
  ],
  "per_step_interval_ms": {
   "n": 5997,
   "median": 1.684,
   "p90": 2.206,
   "mean": 1.912,
   "min": 0.845,
   "chunk_median_ratio": 1.23,
   "p90_over_median": 1.31,
   "varies_2x": false
  }
 },
 "numpy": {
  "bio_s": 0.2,
  "repeats": 3,
  "ms_per_step_incl_setup": 2.084,
  "walls_s": [
   4.863,
   3.744,
   4.168
  ],
  "per_step_interval_ms": {
   "n": 5997,
   "median": 1.695,
   "p90": 2.132,
   "mean": 2.072,
   "min": 0.739,
   "chunk_median_ratio": 1.19,
   "p90_over_median": 1.26,
   "varies_2x": false
  }
 },
 "reference_ms_per_step_bench_pack": 3.327
}
```

## Budget (cuda)

### rng=numpy: predicted 1.406 ms vs measured mean 1.835 ms (ratio 0.766), steady-state mean 1.757 ms (ratio 0.8); stalls 0.094 ms/step (0.9% of steps)

| category | ms/step |
|---|---|
| python+dispatch | 0.168 |
| kernel launch | 0.4534 |
| sync round trips (D2H) | 0.6355 |
| scalar index_put_ (hidden H2D+sync) | 0.1254 |
| as_tensor H2D (pageable) | 0.0236 |

```
{
 "unit_costs_us": {
  "c_disp": 1.433,
  "c_enq": 5.744,
  "c_launch": 4.311,
  "extra_nonzero": 354.137,
  "extra_index": 79.718,
  "extra__local_scalar_dense": 67.224,
  "extra_index_put_": 31.361,
  "extra__to_copy": 23.634
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

### rng=torch: predicted 1.468 ms vs measured mean 2.561 ms (ratio 0.573), steady-state mean 1.71 ms (ratio 0.858); stalls 1.011 ms/step (9.4% of steps)

| category | ms/step |
|---|---|
| python+dispatch | 0.1684 |
| kernel launch | 0.4589 |
| sync round trips (D2H) | 0.7152 |
| scalar index_put_ (hidden H2D+sync) | 0.1254 |
| as_tensor H2D (pageable) | 0.0 |

```
{
 "unit_costs_us": {
  "c_disp": 1.433,
  "c_enq": 5.744,
  "c_launch": 4.311,
  "extra_nonzero": 354.137,
  "extra_index": 79.718,
  "extra__local_scalar_dense": 67.224,
  "extra_index_put_": 31.361,
  "extra__to_copy": 23.634
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
