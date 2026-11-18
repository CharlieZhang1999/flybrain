5 runs: DESKTOP-4C13T5V_20261118-1022, DESKTOP-4C13T5V_20261118-1029, DESKTOP-4C13T5V_20261118-1031, DESKTOP-4C13T5V_20261118-1034, DESKTOP-4C13T5V_20261118-1036
| device | item | measurement | medians (us) per run | max/min | runs flagged 2x |
|---|---|---|---|---|---|
| cuda | 1_launch | enqueue_us_per_op | 11.025 / 5.641 / 11.094 / 8.225 / 5.25 | 2.11 | 3/5 |
| cuda | 1_launch | throughput_us_per_op_with_sync | 11.728 / 5.744 / 11.708 / 8.52 / 5.442 | 2.16 | 4/5 |
| cuda | 1_launch | single_op_interval_us | 4.209 / 4.242 / 4.512 / 4.228 / 4.109 | 1.10 | 0/5 |
| cuda | 2_sync | synchronize_idle_us | 1.949 / 1.933 / 1.92 / 1.953 / 1.941 | 1.02 | 0/5 |
| cuda | 2_sync | add_then_item_us | 72.408 / 74.401 / 78.902 / 72.648 / 79.34 | 1.10 | 0/5 |
| cuda | 2_sync | item_only_us | 40.341 / 42.229 / 45.732 / 44.355 / 42.013 | 1.13 | 0/5 |
| cuda | 3_syncops | nonzero_n165122_true0_us | 103.5 / 360.012 / 99.298 / 98.277 / 98.672 | 3.66 | 0/5 |
| cuda | 3_syncops | nonzero_n165122_true100_us | 355.873 / 359.881 / 98.727 / 102.179 / 101.193 | 3.65 | 1/5 |
| cuda | 3_syncops | nonzero_n165122_true2000_us | 99.547 / 99.26 / 105.29 / 98.376 / 100.644 | 1.07 | 2/5 |
| cuda | 3_syncops | int_sum_n165122_us | 98.144 / 97.057 / 105.145 / 94.76 / 99.774 | 1.11 | 2/5 |
| cuda | 3_syncops | mask_index_len20_us | 108.755 / 92.704 / 111.602 / 99.86 / 108.83 | 1.20 | 0/5 |
| cuda | 3_syncops | mask_index_len311_us | 116.916 / 91.206 / 107.171 / 91.639 / 92.136 | 1.28 | 0/5 |
| cuda | 3_syncops | setitem_scalar_n165122_k20_us | 45.384 / 36.821 / 42.274 / 37.456 / 36.268 | 1.25 | 0/5 |
| cuda | 3_syncops | setitem_tensor_n165122_k20_us | 7.852 / 5.46 / 8.154 / 5.601 / 5.491 | 1.49 | 0/5 |
| cuda | 3_syncops | setitem_scalar_n165122_k200_us | 44.567 / 36.24 / 43.345 / 36.525 / 36.122 | 1.23 | 0/5 |
| cuda | 3_syncops | setitem_tensor_n165122_k200_us | 8.063 / 5.479 / 7.755 / 5.44 / 5.45 | 1.48 | 0/5 |
| cuda | 4_h2d | pageable_as_tensor_len10_us | 33.987 / 30.811 / 31.908 / 31.37 / 30.112 | 1.13 | 0/5 |
| cuda | 4_h2d | pageable_as_tensor_len10_with_sync_us | 30.878 / 31.002 / 33.029 / 30.077 / 30.667 | 1.10 | 0/5 |
| cuda | 4_h2d | pinned_nonblocking_len10_enqueue_us | 5.496 / 5.55 / 5.413 / 5.39 / 5.4 | 1.03 | 0/5 |
| cuda | 4_h2d | pinned_nonblocking_len10_with_sync_us | 49.842 / 47.992 / 50.254 / 45.406 / 37.69 | 1.33 | 0/5 |
| cuda | 4_h2d | pageable_as_tensor_len100_us | 30.477 / 30.943 / 33.865 / 30.938 / 35.245 | 1.16 | 0/5 |
| cuda | 4_h2d | pageable_as_tensor_len100_with_sync_us | 31.069 / 31.103 / 35.23 / 30.167 / 34.785 | 1.17 | 0/5 |
| cuda | 4_h2d | pinned_nonblocking_len100_enqueue_us | 5.496 / 5.51 / 8.024 / 5.37 / 5.451 | 1.49 | 0/5 |
| cuda | 4_h2d | pinned_nonblocking_len100_with_sync_us | 62.142 / 48.362 / 54.326 / 46.709 / 49.494 | 1.33 | 1/5 |
| cuda | 4_h2d | pageable_as_tensor_len1000_us | 86.598 / 30.737 / 34.69 / 33.649 / 30.339 | 2.85 | 0/5 |
| cuda | 4_h2d | pageable_as_tensor_len1000_with_sync_us | 100.241 / 30.908 / 35.105 / 36.314 / 30.118 | 3.33 | 0/5 |
| cuda | 4_h2d | pinned_nonblocking_len1000_enqueue_us | 48.331 / 5.61 / 8.225 / 8.256 / 5.612 | 8.62 | 0/5 |
| cuda | 4_h2d | pinned_nonblocking_len1000_with_sync_us | 102.92 / 50.231 / 53.45 / 50.827 / 50.695 | 2.05 | 0/5 |
| cuda | 5_graph | eager_50_with_sync_us | 349.07 / 556.511 / 394.604 / 339.265 / 333.96 | 1.67 | 2/5 |
| cuda | 5_graph | eager_50_enqueue_us | 266.982 / 238.947 / 225.126 / 284.976 / 286.265 | 1.27 | 1/5 |
| cuda | 5_graph | graph_replay_with_sync_us | 176.763 / 81.262 / 76.073 / 81.495 / 81.631 | 2.32 | 1/5 |
| cuda | 5_graph | graph_replay_enqueue_us | 10.108 / 10.609 / 10.35 / 13.035 / 13.406 | 1.33 | 1/5 |
| cpu | 1_launch | enqueue_us_per_op | 1.415 / 1.433 / 1.525 / 1.414 / 1.488 | 1.08 | 0/5 |
| cpu | 1_launch | throughput_us_per_op_with_sync | 1.415 / 1.433 / 1.525 / 1.414 / 1.489 | 1.08 | 0/5 |
| cpu | 1_launch | single_op_interval_us | 1.388 / 1.402 / 1.48 / 1.383 / 1.448 | 1.07 | 0/5 |
| cpu | 2_sync | add_then_item_us | 1.539 / 1.573 / 1.592 / 1.573 / 1.569 | 1.03 | 0/5 |
| cpu | 2_sync | item_only_us | 0.13 / 0.151 / 0.152 / 0.151 / 0.151 | 1.17 | 0/5 |
| cpu | 3_syncops | nonzero_n165122_true0_us | 6.597 / 4.859 / 4.927 / 7.825 / 4.908 | 1.61 | 0/5 |
| cpu | 3_syncops | nonzero_n165122_true100_us | 28.867 / 16.27 / 16.323 / 29.081 / 16.315 | 1.79 | 0/5 |
| cpu | 3_syncops | nonzero_n165122_true2000_us | 31.115 / 23.348 / 23.536 / 31.149 / 31.071 | 1.33 | 0/5 |
| cpu | 3_syncops | int_sum_n165122_us | 3.958 / 4.047 / 3.995 / 3.918 / 3.903 | 1.04 | 0/5 |
| cpu | 3_syncops | mask_index_len20_us | 3.109 / 3.106 / 3.376 / 3.126 / 3.36 | 1.09 | 0/5 |
| cpu | 3_syncops | mask_index_len311_us | 3.619 / 3.577 / 3.67 / 3.616 / 3.621 | 1.03 | 0/5 |
| cpu | 3_syncops | setitem_scalar_n165122_k20_us | 1.409 / 1.402 / 1.419 / 1.402 / 1.388 | 1.02 | 0/5 |
| cpu | 3_syncops | setitem_tensor_n165122_k20_us | 1.799 / 1.152 / 1.156 / 1.172 / 1.157 | 1.56 | 0/5 |
| cpu | 3_syncops | setitem_scalar_n165122_k200_us | 2.498 / 1.683 / 1.663 / 1.674 / 1.609 | 1.55 | 0/5 |
| cpu | 3_syncops | setitem_tensor_n165122_k200_us | 2.099 / 1.433 / 1.409 / 1.422 / 1.408 | 1.49 | 0/5 |
| cpu | 4_h2d | pageable_as_tensor_len10_us | 0.84 / 0.831 / 0.841 / 0.822 / 0.815 | 1.03 | 0/5 |
| cpu | 4_h2d | pageable_as_tensor_len100_us | 0.83 / 0.822 / 0.831 / 0.831 / 0.815 | 1.02 | 0/5 |
| cpu | 4_h2d | pageable_as_tensor_len1000_us | 0.83 / 0.821 / 0.831 / 0.821 / 0.825 | 1.01 | 0/5 |

| run | rng | predicted ms | measured mean ms | steady mean ms | stall fraction | pred/mean | pred/steady |
|---|---|---|---|---|---|---|---|
| DESKTOP-4C13T5V_20261118-1022 | numpy | 2.037 | 1.859 | 1.735 | 1.4% | 1.096 | 1.174 |
| DESKTOP-4C13T5V_20261118-1022 | torch | 2.123 | 2.039 | 1.984 | 0.6% | 1.041 | 1.07 |
| DESKTOP-4C13T5V_20261118-1029 | numpy | 1.406 | 1.835 | 1.757 | 0.9% | 0.766 | 0.8 |
| DESKTOP-4C13T5V_20261118-1029 | torch | 1.468 | 2.561 | 1.71 | 9.4% | 0.573 | 0.858 |
| DESKTOP-4C13T5V_20261118-1031 | numpy | 1.773 | 2.317 | 1.68 | 7.3% | 0.765 | 1.055 |
| DESKTOP-4C13T5V_20261118-1031 | torch | 1.851 | 2.224 | 1.724 | 5.4% | 0.832 | 1.074 |
| DESKTOP-4C13T5V_20261118-1034 | numpy | 1.418 | 2.201 | 1.654 | 6.2% | 0.644 | 0.858 |
| DESKTOP-4C13T5V_20261118-1034 | torch | 1.481 | 2.008 | 1.676 | 3.8% | 0.738 | 0.884 |
| DESKTOP-4C13T5V_20261118-1036 | numpy | 1.131 | 2.065 | 1.701 | 4.1% | 0.548 | 0.665 |
| DESKTOP-4C13T5V_20261118-1036 | torch | 1.194 | 2.26 | 1.818 | 5.0% | 0.528 | 0.657 |
