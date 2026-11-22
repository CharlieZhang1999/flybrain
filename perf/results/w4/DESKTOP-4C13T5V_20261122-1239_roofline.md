# Week 4 T4: compute roofline, DESKTOP-4C13T5V_20261122-1239 (5 runs)

torch.matmul, square n x n, sizes [1024, 2048, 4096, 8192], 100 timed repeats per size (CUDA events, not rescaled; host clock in the diagnostics columns). Peak of a run = best size's median. Every value: median across runs [min-max across runs].

| dtype | peak TFLOP/s | best n per run | spec TFLOP/s (derived) | % of spec | ridge FLOP/B at 382 GB/s |
|---|---|---|---|---|---|
| fp64 | 0.347 [0.347-0.347] | [2048, 2048, 4096, 4096, 4096] | 0.37 | 93.8% [93.8-93.8%] | 0.908 |
| fp32 | 16.879 [16.867-17.265] | [8192, 8192, 8192, 8192, 8192] | 23.68 | 71.3% [71.2-72.9%] | 44.186 |
| tf32 | 24.052 [24.050-24.241] | [8192, 8192, 8192, 8192, 8192] | 23.68 | 101.5% [101.5-102.3%] | 62.963 |
| bf16 | 48.294 [48.237-48.639] | [8192, 8192, 8192, 8192, 8192] | 47.37 | 102.0% [101.8-102.7%] | 126.424 |
| fp16 | 47.668 [47.667-48.084] | [8192, 4096, 4096, 4096, 4096] | 47.37 | 100.6% [100.6-101.5%] | 124.785 |

fp32 / fp64 peak: 48.6. FP16 tensor with FP16 accumulate (not used by torch.matmul): 94.74 TFLOP/s.

## Per size (TFLOP/s; median of the per-run medians [min-max]; p90 of time = slow tail; host = wall of the whole repeat loop)

| dtype | n | median | p90 | median rescaled to host wall | host wall | host wall / event sum |
|---|---|---|---|---|---|---|
| fp64 | 1024 | 0.328 [0.327-0.331] | 0.327 [0.327-0.331] | 0.345 [0.332-0.352] | 0.345 [0.332-0.352] | 0.951 [0.930-0.996] |
| fp64 | 2048 | 0.345 [0.345-0.347] | 0.345 [0.345-0.347] | 0.361 [0.357-0.378] | 0.361 [0.357-0.378] | 0.956 [0.918-0.966] |
| fp64 | 4096 | 0.347 [0.347-0.347] | 0.347 [0.347-0.347] | 0.363 [0.360-0.366] | 0.363 [0.360-0.366] | 0.958 [0.949-0.966] |
| fp64 | 8192 | 0.346 [0.346-0.346] | 0.344 [0.344-0.346] | 0.362 [0.359-0.365] | 0.362 [0.359-0.366] | 0.956 [0.947-0.962] |
| fp32 | 1024 | 13.823 [13.774-13.823] | 13.643 [13.643-13.820] | 14.250 [13.629-14.378] | 14.100 [13.611-14.275] | 0.970 [0.961-1.014] |
| fp32 | 2048 | 15.311 [15.208-15.567] | 15.146 [15.078-15.492] | 15.972 [15.528-16.082] | 15.980 [15.523-16.098] | 0.959 [0.950-1.002] |
| fp32 | 4096 | 15.643 [15.608-16.007] | 15.624 [15.548-15.985] | 16.308 [16.002-16.507] | 16.303 [16.002-16.504] | 0.957 [0.949-1.000] |
| fp32 | 8192 | 16.879 [16.867-17.265] | 16.851 [16.839-17.240] | 17.637 [17.605-17.811] | 17.636 [17.612-17.814] | 0.956 [0.948-0.981] |
| tf32 | 1024 | 20.416 [13.804-20.416] | 20.403 [12.509-20.410] | 20.921 [14.313-21.151] | 20.817 [15.948-21.066] | 0.974 [0.965-1.019] |
| tf32 | 2048 | 21.059 [21.057-21.325] | 21.031 [21.031-21.272] | 22.040 [21.268-22.159] | 22.021 [21.246-22.158] | 0.958 [0.950-1.003] |
| tf32 | 4096 | 23.532 [23.523-23.833] | 23.515 [23.507-23.816] | 24.614 [23.820-24.808] | 24.611 [23.817-24.808] | 0.956 [0.949-1.000] |
| tf32 | 8192 | 24.052 [24.050-24.241] | 24.045 [24.041-24.233] | 25.359 [24.898-26.443] | 25.359 [24.897-26.443] | 0.949 [0.917-0.966] |
| bf16 | 1024 | 38.326 [15.404-38.348] | 38.228 [14.196-38.304] | 38.431 [10.997-40.462] | 38.125 [10.662-40.181] | 0.997 [0.948-1.401] |
| bf16 | 2048 | 41.660 [41.660-41.869] | 41.657 [41.589-41.865] | 43.713 [42.926-45.455] | 43.676 [42.907-45.447] | 0.953 [0.921-0.971] |
| bf16 | 4096 | 46.885 [46.870-47.282] | 46.868 [46.836-47.231] | 49.398 [48.499-51.536] | 49.392 [48.501-51.562] | 0.949 [0.917-0.967] |
| bf16 | 8192 | 48.294 [48.237-48.639] | 48.259 [48.220-48.613] | 50.193 [49.851-50.936] | 50.193 [49.849-50.939] | 0.963 [0.948-0.976] |
| fp16 | 1024 | 15.676 [15.520-39.780] | 14.929 [14.600-38.414] | 11.426 [11.149-41.944] | 12.412 [11.879-41.481] | 1.369 [0.948-1.401] |
| fp16 | 2048 | 46.254 [46.254-46.648] | 46.249 [46.127-46.511] | 48.533 [47.804-50.614] | 48.465 [47.728-50.556] | 0.953 [0.922-0.968] |
| fp16 | 4096 | 47.668 [47.667-47.871] | 47.633 [47.534-47.854] | 50.235 [49.473-52.182] | 50.217 [49.440-52.179] | 0.949 [0.917-0.964] |
| fp16 | 8192 | 47.473 [47.469-48.084] | 47.467 [47.461-48.073] | 50.243 [50.040-52.741] | 50.240 [50.038-52.737] | 0.948 [0.900-0.954] |

Host wall / event sum, all loops: 0.9564 [0.9001-1.4011]; event / host over a 2 s idle sleep: 1.0542 [1.0000-1.1111]. Copy bandwidth this session: 385.9 [385.7-386.1] GB/s (Day 2: 382.0 GB/s; NVIDIA: 448.0 GB/s). TF32 check, relative error vs float64 at n=1024: {'fp32': 3.336090481893801e-07, 'tf32': 0.0002940723978356156}.

## LIF decay (sim/lif_torch._decay) per neuron

| dtype | FLOPs | eager bytes (aten ops) | eager FLOP/B | fused bytes | fused FLOP/B |
|---|---|---|---|---|---|
| float64 | 8 | 244 (14 ops) | 0.0328 | 49.0001 | 0.1633 |
| float32 | 8 | 140 (14 ops) | 0.0571 | 33.0001 | 0.2424 |

## Variability (> 2x across runs)

- bf16 n=1024 tflops_median: 38.326 [15.404-38.348], 2.49x
- fp16 n=1024 tflops_median: 15.676 [15.52-39.78], 2.56x

## Over spec by > 10% (stop condition)

- none (median peak)

## Spec derivation

```
{
 "sources": {
  "product": "https://www.nvidia.com/en-us/geforce/graphics-cards/compare/#50-series (RTX 5060 Ti column: 4608 CUDA cores, boost 2.57 GHz, 759 AI TOPS, 448 GB/sec)",
  "whitepaper": "https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf (Appendix C, Table 6, RTX 5070 column; FP64 = 1/64 FP32: GB202 section)"
 },
 "sms": 36,
 "boost_ghz": 2.57,
 "dense_flop_per_sm_per_clock": {
  "fp64": 4,
  "fp32": 256,
  "tf32": 256,
  "bf16": 512,
  "fp16": 512
 },
 "tflops": {
  "fp64": 0.37,
  "fp32": 23.685,
  "tf32": 23.685,
  "bf16": 47.37,
  "fp16": 47.37
 },
 "fp16_acc_tflops": 94.74,
 "bandwidth_GBps": 448.0,
 "note": "Derived, not published for the 5060 Ti: whitepaper RTX 5070 rates per SM per clock x 36 SMs x 2.57 GHz boost. FP64 = FP32 / 64 (whitepaper, GB202). torch fp16/bf16 matmul uses FP32 accumulate."
}
```

## Environment

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
 "triton": "3.6.0",
 "sms": 36,
 "vram_gb": 15.9,
 "l2_mb": 32.0,
 "fp16_reduced_precision_reduction": true,
 "bf16_reduced_precision_reduction": true,
 "smi_clocks": "3090 MHz, 14001 MHz, 180.00 W"
}
```
