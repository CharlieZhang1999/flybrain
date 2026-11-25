# Scout one-pager for P1 (2026-11-25): batched multi-trial event-driven LIF on GPU

Research only; unverified items are marked. Summary by Lead of the Scout report.

- **GeNN batch mode** shares parameters and sparse connectivity across trials, state per trial, trial-major
  (B x N) layout ([GeNN docs](https://genn-team.github.io/genn/documentation/5/_sources/building_networks.rst.txt),
  [genn#574](https://www.github.com/genn-team/genn/issues/574)); its sparse presynaptic kernels are latency-bound
  ([genn#410](https://www.github.com/genn-team/genn/issues/410)).
- **eonsystemspbc/fly-brain** runs the same Shiu/FlyWire LIF model (~138k neurons, dt 0.1 ms) on 6 backends
  ([repo](https://github.com/eonsystemspbc/fly-brain)). Published no-I/O timings, float32
  ([timing_summary.csv](https://raw.githubusercontent.com/eonsystemspbc/fly-brain/main/data/results/nature_2026_07/no_io/timing_summary.csv)):
  GeNN 0.054 / 0.073 / 0.103 / 0.82 / 1.57 ms/step at B = 1/4/8/16/32; PyTorch dense-spikes x sparse SpMM
  0.65 / 2.9 / 4.5 / 17.8 ms/step at B = 1/4/8/32 (linear in B). GPU unverified ("tested on RTX 4070").
  Our single-trial graph is 0.49 ms/step float64.
- **SpMM vs event gather:** at ~19 spikes/step, SpMM over all columns wastes work; keep the gather event-driven.
- **float64 atomics:** native since CC 6.x; at ~8 x 6k adds/step fp64 throughput is unlikely to bind (estimate).
  Atomic order is nondeterministic and fp64 add is not associative
  ([PyTorch randomness notes](https://docs.pytorch.org/docs/2.15/notes/randomness.html)), so count identity is
  observed, not guaranteed (consistent with Week 4 T5b). Docs disagree on whether CUDA `index_add_` has a
  deterministic mode; check the installed version.
- **RNG in graphs:** `graphsafe_get_state`/`graphsafe_set_state` and `register_generator_state`
  ([CUDA notes](https://docs.pytorch.org/docs/2.15/notes/cuda.html)); newer PyTorch limits distinct generator
  states per graph (`num_rng_states`, [docs](https://docs.pytorch.org/docs/main/generated/torch.cuda.CUDAGraph.html)),
  so one generator per trial may not scale to B = 32 on torch 2.11 (check). Alternative: counter-based Philox
  keyed on (trial seed, step, neuron) in one kernel; batch invariance holds by construction if B = 1 uses it too.

## Recommendations folded into the brief
1. Kernel count independent of B: B x N state, one event gather over a flattened (trial, spike) list.
2. Per-trial RNG stream independent of B and slot (counter-based, or per-trial generators if supported).
3. If atomic order breaks batch invariance, use a fixed-order accumulation in both B = 1 and B > 1.
4. No full SpMM. 5. GeNN float32 numbers above are a reference ceiling only (different GPU, precision, simulator).
