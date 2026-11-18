# Week 4 T1 overhead budget: DESKTOP-4C13T5V_20261118-1036

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
| 1_launch | enqueue_us_per_op | 100 | 5.25 | 12.807 | **yes** |
| 1_launch | throughput_us_per_op_with_sync | 100 | 5.442 | 13.502 | **yes** |
| 1_launch | single_op_interval_us | 100000 | 4.109 | 5.873 | no |
| 2_sync | synchronize_idle_us | 2000 | 1.941 | 1.98 | no |
| 2_sync | add_then_item_us | 2000 | 79.34 | 86.377 | no |
| 2_sync | item_only_us | 2000 | 42.013 | 44.224 | no |
| 3_syncops | nonzero_n165122_true0_us | 1000 | 98.672 | 108.676 | no |
| 3_syncops | nonzero_n165122_true100_us | 1000 | 101.193 | 116.105 | no |
| 3_syncops | nonzero_n165122_true2000_us | 1000 | 100.644 | 114.637 | no |
| 3_syncops | int_sum_n165122_us | 1000 | 99.774 | 108.211 | no |
| 3_syncops | mask_index_len20_us | 1000 | 108.83 | 133.398 | no |
| 3_syncops | mask_index_len311_us | 1000 | 92.136 | 127.307 | no |
| 3_syncops | setitem_scalar_n165122_k20_us | 1000 | 36.268 | 43.983 | no |
| 3_syncops | setitem_tensor_n165122_k20_us | 1000 | 5.491 | 5.98 | no |
| 3_syncops | setitem_scalar_n165122_k200_us | 1000 | 36.122 | 39.892 | no |
| 3_syncops | setitem_tensor_n165122_k200_us | 1000 | 5.45 | 5.921 | no |
| 4_h2d | pageable_as_tensor_len10_us | 1000 | 30.112 | 37.522 | no |
| 4_h2d | pageable_as_tensor_len10_with_sync_us | 1000 | 30.667 | 36.882 | no |
| 4_h2d | pinned_nonblocking_len10_enqueue_us | 1000 | 5.4 | 6.011 | no |
| 4_h2d | pinned_nonblocking_len10_with_sync_us | 1000 | 37.69 | 44.292 | no |
| 4_h2d | pageable_as_tensor_len100_us | 1000 | 35.245 | 43.413 | no |
| 4_h2d | pageable_as_tensor_len100_with_sync_us | 1000 | 34.785 | 44.102 | no |
| 4_h2d | pinned_nonblocking_len100_enqueue_us | 1000 | 5.451 | 7.652 | no |
| 4_h2d | pinned_nonblocking_len100_with_sync_us | 1000 | 49.494 | 54.416 | no |
| 4_h2d | pageable_as_tensor_len1000_us | 1000 | 30.339 | 33.93 | no |
| 4_h2d | pageable_as_tensor_len1000_with_sync_us | 1000 | 30.118 | 34.28 | no |
| 4_h2d | pinned_nonblocking_len1000_enqueue_us | 1000 | 5.612 | 6.402 | no |
| 4_h2d | pinned_nonblocking_len1000_with_sync_us | 1000 | 50.695 | 51.905 | no |
| 5_graph | kernels | | 50 | | |
| 5_graph | eager_50_with_sync_us | 1000 | 333.96 | 591.201 | no |
| 5_graph | eager_50_enqueue_us | 1000 | 286.265 | 325.202 | no |
| 5_graph | graph_replay_with_sync_us | 1000 | 81.631 | 89.807 | no |
| 5_graph | graph_replay_enqueue_us | 1000 | 13.406 | 16.624 | no |
| 5_graph | capture_ok | | True | | |

## Micro-benchmarks (cpu)

| item | measurement | n | median | p90 | 2x flag |
|---|---|---|---|---|---|
| 1_launch | n_ops | | 10000 | | |
| 1_launch | batches | | 100 | | |
| 1_launch | enqueue_us_per_op | 100 | 1.488 | 1.508 | no |
| 1_launch | throughput_us_per_op_with_sync | 100 | 1.489 | 1.508 | no |
| 1_launch | single_op_interval_us | 100000 | 1.448 | 1.478 | no |
| 2_sync | add_then_item_us | 2000 | 1.569 | 1.58 | no |
| 2_sync | item_only_us | 2000 | 0.151 | 0.161 | no |
| 3_syncops | nonzero_n165122_true0_us | 1000 | 4.908 | 4.999 | no |
| 3_syncops | nonzero_n165122_true100_us | 1000 | 16.315 | 30.286 | no |
| 3_syncops | nonzero_n165122_true2000_us | 1000 | 31.071 | 35.617 | no |
| 3_syncops | int_sum_n165122_us | 1000 | 3.903 | 4.103 | no |
| 3_syncops | mask_index_len20_us | 1000 | 3.36 | 5.079 | no |
| 3_syncops | mask_index_len311_us | 1000 | 3.621 | 5.593 | no |
| 3_syncops | setitem_scalar_n165122_k20_us | 1000 | 1.388 | 1.409 | no |
| 3_syncops | setitem_tensor_n165122_k20_us | 1000 | 1.157 | 1.177 | no |
| 3_syncops | setitem_scalar_n165122_k200_us | 1000 | 1.609 | 1.63 | no |
| 3_syncops | setitem_tensor_n165122_k200_us | 1000 | 1.408 | 1.438 | no |
| 4_h2d | pageable_as_tensor_len10_us | 1000 | 0.815 | 0.845 | no |
| 4_h2d | pageable_as_tensor_len100_us | 1000 | 0.815 | 0.825 | no |
| 4_h2d | pageable_as_tensor_len1000_us | 1000 | 0.825 | 0.844 | no |

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
   "median": 1.788,
   "p90": 2.552,
   "mean": 2.26,
   "min": 1.202,
   "chunk_median_ratio": 1.3,
   "p90_over_median": 1.43,
   "varies_2x": false
  },
  "run_mean_ms": {
   "n": 10,
   "median": 2.025,
   "p90": 2.721,
   "mean": 2.26,
   "min": 1.519,
   "chunk_median_ratio": 1.78,
   "p90_over_median": 1.34,
   "varies_2x": false
  },
  "stall_threshold_ms": 5.0,
  "stall_steps": 99,
  "stall_fraction": 0.0497,
  "stall_median_ms": 11.081,
  "steady_mean_ms": 1.818,
  "stall_ms_per_step": 0.533
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
   "median": 1.699,
   "p90": 2.269,
   "mean": 2.065,
   "min": 1.132,
   "chunk_median_ratio": 1.08,
   "p90_over_median": 1.34,
   "varies_2x": false
  },
  "run_mean_ms": {
   "n": 10,
   "median": 1.874,
   "p90": 2.565,
   "mean": 2.065,
   "min": 1.565,
   "chunk_median_ratio": 1.65,
   "p90_over_median": 1.37,
   "varies_2x": false
  },
  "stall_threshold_ms": 5.0,
  "stall_steps": 82,
  "stall_fraction": 0.0412,
  "stall_median_ms": 10.704,
  "steady_mean_ms": 1.701,
  "stall_ms_per_step": 0.434
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
   "median": 0.519,
   "p90": 0.779,
   "mean": 0.557,
   "min": 0.346,
   "chunk_median_ratio": 1.32,
   "p90_over_median": 1.5,
   "varies_2x": false
  },
  "run_mean_ms": {
   "n": 10,
   "median": 0.554,
   "p90": 0.636,
   "mean": 0.557,
   "min": 0.462,
   "chunk_median_ratio": 1.34,
   "p90_over_median": 1.15,
   "varies_2x": false
  },
  "stall_threshold_ms": 5.0,
  "stall_steps": 0,
  "stall_fraction": 0.0,
  "stall_median_ms": null,
  "steady_mean_ms": 0.557,
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
   "median": 0.562,
   "p90": 0.801,
   "mean": 0.595,
   "min": 0.337,
   "chunk_median_ratio": 1.36,
   "p90_over_median": 1.43,
   "varies_2x": false
  },
  "run_mean_ms": {
   "n": 10,
   "median": 0.592,
   "p90": 0.676,
   "mean": 0.595,
   "min": 0.499,
   "chunk_median_ratio": 1.22,
   "p90_over_median": 1.14,
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

## Bench-pack method with main's sim/ (0.2 s, setup included)

```
{
 "torch": {
  "bio_s": 0.2,
  "repeats": 3,
  "ms_per_step_incl_setup": 2.447,
  "walls_s": [
   4.893,
   8.262,
   4.739
  ],
  "per_step_interval_ms": {
   "n": 5997,
   "median": 1.809,
   "p90": 10.089,
   "mean": 2.926,
   "min": 0.836,
   "chunk_median_ratio": 1.14,
   "p90_over_median": 5.58,
   "varies_2x": true
  }
 },
 "numpy": {
  "bio_s": 0.2,
  "repeats": 3,
  "ms_per_step_incl_setup": 2.856,
  "walls_s": [
   4.092,
   6.573,
   5.712
  ],
  "per_step_interval_ms": {
   "n": 5997,
   "median": 1.76,
   "p90": 9.344,
   "mean": 2.666,
   "min": 0.8,
   "chunk_median_ratio": 1.1,
   "p90_over_median": 5.31,
   "varies_2x": true
  }
 },
 "reference_ms_per_step_bench_pack": 3.327
}
```

## Budget (cuda)

### rng=numpy: predicted 1.131 ms vs measured mean 2.065 ms (ratio 0.548), steady-state mean 1.701 ms (ratio 0.665); stalls 0.434 ms/step (4.1% of steps)

| category | ms/step |
|---|---|
| python+dispatch | 0.1745 |
| kernel launch | 0.4158 |
| sync round trips (D2H) | 0.3942 |
| scalar index_put_ (hidden H2D+sync) | 0.1231 |
| as_tensor H2D (pageable) | 0.0232 |

```
{
 "unit_costs_us": {
  "c_disp": 1.488,
  "c_enq": 5.442,
  "c_launch": 3.954,
  "extra_nonzero": 95.751,
  "extra_index": 81.252,
  "extra__local_scalar_dense": 72.41,
  "extra_index_put_": 30.777,
  "extra__to_copy": 23.182
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

### rng=torch: predicted 1.194 ms vs measured mean 2.26 ms (ratio 0.528), steady-state mean 1.818 ms (ratio 0.657); stalls 0.533 ms/step (5.0% of steps)

| category | ms/step |
|---|---|
| python+dispatch | 0.1749 |
| kernel launch | 0.4209 |
| sync round trips (D2H) | 0.4755 |
| scalar index_put_ (hidden H2D+sync) | 0.1231 |
| as_tensor H2D (pageable) | 0.0 |

```
{
 "unit_costs_us": {
  "c_disp": 1.488,
  "c_enq": 5.442,
  "c_launch": 3.954,
  "extra_nonzero": 95.751,
  "extra_index": 81.252,
  "extra__local_scalar_dense": 72.41,
  "extra_index_put_": 30.777,
  "extra__to_copy": 23.182
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
