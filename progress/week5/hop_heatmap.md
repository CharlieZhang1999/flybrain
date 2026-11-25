# W5 T10: which neurons does the decoder read? Attribution by hop distance, calib vs shuffle

Dataset v1 (PR #19), the same 5 outer folds as PR #19 / #20. Script `scripts/w5_hop_heatmap.py`;
results in `results/w5/hidden_state/hop_heatmap/` (`hop_histogram.csv`, `spotcheck.csv`,
`heatmap_values.csv`, `hypotheses_h1_h2.csv`, `hypothesis_h3_spearman.csv`, `accuracy.json`,
`figures/heatmap_{any,strong}_{raw_share,per_neuron}.png`).

**Hops.** W is post x pre (`lif_torch` adds column j to all rows when j spikes; `shuffle_W`
permutes the column = presynaptic index). BFS from each driven set along pre -> post edges of that
cfg's chemical W. With any synapse, every decoder input is within 4 hops and almost all within 3:
calib hunger 1/2/3/4 = 82/2863/4593/14, thirst 33/1388/6016/115, arousal 1181/5806/565/0
(of 7,552); shuffle hunger 142/6193/989/0, thirst 72/5524/1728/0, arousal 2989/4335/0/0 (of
7,324). So "deep" = hop >= 3, and the shuffle is shallower: 88% of its inputs are at hops 1-2
(calib 50%). Strong edges (|w| >= p90) push everything ~1 hop deeper (16 / 12 unreachable).
Spot check: 10 neurons, path re-read from W, min hop of presynaptic partners = hop - 1, all pass.

**Decoder.** Perceiver-style: token = neuron embedding + piecewise-linear count embedding, 16 latent
queries cross-attend per 50 ms bin, bi-GRU over the 80 bins. Balanced accuracy (pooled held-out):
calib **0.935** (folds .952/.944/.972/.835/.941) vs PR #20 best gru_bi 0.883; shuffle **0.778**
(.760/.812/.826/.803/.699) vs gru_bi 0.808 (-0.031). Both within 0.05. Four shuffle runs stuck at
chance and were restarted with a new seed (rule: best val < 0.35; no calib run triggered it).

**Attribution** (mean of 5 folds, any edges, mean over the 3 states)

| method | shuffle: hops 1-2 (raw) | calib hop >= 3 (raw) | shuffle hop >= 3 (raw) | paired t (5 folds) | calib / shuffle hop >= 3, per neuron |
|---|---|---|---|---|---|
| attention (state - baseline) | 0.90 | 0.38 | 0.10 | p = 0.0002 | 0.14 / 0.08 (p = 0.18) |
| gradient x input | 0.88 | 0.33 | 0.12 | p = 0.001 | 0.07 / 0.08 (p = 0.66) |
| causal (mean-impute hop group) | 0.98 | 0.00 | 0.02 | p = 0.85 | 0.00 / 0.01 (p = 0.86) |
| share of input neurons (reference) | 0.88 | 0.50 | 0.12 | | |

Causal: the hop-1 group carries 67-100% of the recall drop for every state in both cfgs (for
hunger and arousal, imputing it usually removes nearly all of the state's recall). Hop 2 matters
only for thirst (0.33 of the drop in calib, 0.27 in shuffle), and imputing hop >= 3 changes
nothing. Wilcoxon p = 0.031 (the minimum for n = 5) for both raw-share rows.

**Hypotheses**
- **H1 (Lead) holds.** 88-98% of the shuffle's attribution mass is at hops 1-2 for all three
  methods (per neuron 92-99%). But 88% of the shuffle's inputs are at hops 1-2 anyway.
- **H2 (Charlie) partly holds, raw share only.** Calib puts more raw attention and gradient mass
  at hop >= 3 than the shuffle (+0.28 and +0.21, t-test p = 0.0002 and 0.001). That gap matches the hop histogram
  (the calib has 50% of its inputs at hop >= 3, the shuffle 12%), and it disappears per neuron and
  under the causal test. In both wirings the deep neurons get less than their share of mass: the
  readable signal is concentrated at hop 1.
- **H3 partly holds.** Per neuron, the methods agree (median Spearman 0.93-1.0 over cfg x state,
  any edges; strong edges 0.89-0.92, min 0.6). On raw shares, attention and gradient agree (median
  1.0) but the causal ranking does not (median 0.45-0.68). Only 3-4 hop groups are non-empty, so
  the rank correlations are coarse.

**Caveats.** The models use early stopping on the PR #20 validation split without refitting, and
four shuffle folds needed a restart. Hop groups are defined from chemical W only: gap junctions
and the sensory presyn are ignored. The causal mass is clipped at 0 and is not additive across
groups. Hop 4 holds 14-115 calib neurons, so its per-neuron values are noisy.
