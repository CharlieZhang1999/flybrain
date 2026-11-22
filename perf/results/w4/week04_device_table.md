| device and mode | dtype | steady ms/step | all steps ms/step | bio-s per wall-s | source |
|---|---|---|---|---|---|
| PC CPU (6 threads), reference | f64 | 0.55 [0.52-0.59] | 0.55 [0.52-0.59] | 0.181 [0.169-0.191] | Day 6, `perf/results/w4/DESKTOP-4C13T5V_20261122-1053_graph.json` |
| 5060 Ti, reference eager | f64 | 1.73 [1.57-1.85] | 1.79 [1.74-1.98] | 0.058 [0.054-0.064] | Day 6, `perf/results/w4/DESKTOP-4C13T5V_20261122-1053_graph.json` |
| 5060 Ti, reference eager + torch.compile | f64 | - | 2.07 (worst 2.17) | 0.048 (all steps) | Day 2, `perf/results/w4/DESKTOP-4C13T5V_20261116-2309.json` |
| 5060 Ti, sync-free eager | f64 | 1.75 [1.48-1.93] | 1.84 [1.27-1.93] | 0.057 [0.052-0.068] | Day 6, `perf/results/w4/DESKTOP-4C13T5V_20261122-1053_graph.json` |
| 5060 Ti, sync-free CUDA graph | f64 | 0.49 [0.43-0.60] | 0.45 [0.42-0.63] | 0.205 [0.168-0.234] | Day 6, `perf/results/w4/DESKTOP-4C13T5V_20261122-1053_graph.json` |
| 5060 Ti, sync-free CUDA graph | f32 | 0.44 [0.43-0.56] | 0.44 [0.43-0.45] | 0.225 [0.177-0.235] | Day 6, `perf/results/w4/DESKTOP-4C13T5V_20261122-1053_graph.json` |
| Mac CPU | - | not run | not run | not run | not measured in Week 4 |
| Mac MPS | - | not run | not run | not run | not measured in Week 4 |
