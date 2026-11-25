# W5 T10: attribution by hop distance (PR #21)

Figures and summary tables from `scripts/w5_hop_heatmap.py`. The full outputs, including per-fold caches, are on the project mount at `results/w5/hidden_state/hop_heatmap/`. Write-up: `progress/week5/hop_heatmap.md`.

- `heatmap_any_raw_share.png`: share of total attribution per hop (any synapse counts as an edge).
- `heatmap_any_per_neuron.png`: the same, divided by the number of neurons at each hop.
- `heatmap_strong_*.png`: the same, counting only the strongest 10% of connections as edges.
- `hop_histogram.csv`: decoder input neurons per hop.
- `heatmap_values.csv`: the numbers behind every panel.
- `accuracy.json`: test balanced accuracy of the attention decoder.
- `hypotheses*.{csv,json}`: H1–H3 tests.
- `spotcheck.csv`: manual BFS spot checks.

"calib" is the real wiring; "shuffle" is the degree-preserving shuffled control.
