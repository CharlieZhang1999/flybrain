# Week 2 Days 2–3: eLN–PN gap junctions (3 Nov 2026)

## Why
Yaksi & Wilson 2010 (Neuron) tested the eLN→PN connection pharmacologically and genetically. It survived Cd²⁺ and the nicotinic blockers mecamylamine and d-tubocurarine. It was abolished in shakB², the gap-junction mutant. The coupling is small (Fig. 4C axis goes up to 0.04) and passes depolarization better than hyperpolarization. PN→eLN is mixed chemical and electrical, and eLN→iLN is mostly cholinergic chemical. The Shiu model turns all eLN output into chemical excitation.

## What was changed (all optional, off by default)
- **Simulator:** `sim/lif_numpy.py` takes a new `gap=` argument. After the exact LIF update, each coupled, non-refractory neuron gets `v_i += dt/τm · Σ_j c_ij (v_j − v_i)`. LIF neurons have no spike waveform, so a spike also gives each partner a spikelet `c_ij · v_spike`. With `gap=None`, spikes are bit-identical to the old code (checked on FlyWire sugar, and with an all-zero gap matrix). A two-neuron test behaves as expected: the partner doesn't fire at c = 0.01, fires at 16 Hz at c = 0.05 and at 54 Hz at c = 0.1.
- **Coupling:** `sim/gap.py`. The connectome has no gap junctions, so they go on eLN–PN pairs with any chemical contact (24,942 pairs in MaleCNS, 23,803 in FlyWire), with `c_ij = c0 · min(s_ij/50, 1)` where s_ij is the contact's synapse count. **This is an assumption.**
- **Chemical edges:** cholinergic LN→PN set to 0. PN→LN, LN→LN and LN→GABA LN kept.
- **Base model:** as in C8 for MaleCNS (brain only, monoamines removed, w × 0.581, rate APL). Shiu's model as published for FlyWire.

## Results (stimulus 0–1 s at 150 Hz, silence to 1.6 s; persistent = more than 20 neurons firing in the last 100 ms)

"Target PN" means the PNs of the stimulated glomeruli: DM1/DM4 for vinegar, VP2 for hot. "Other PN" is all remaining PNs. Rates are from 0.5–1.0 s.

### First attempt: 40 mV spikelets (c · full spike height), c0 = 0.01–0.04. Fails
| | MaleCNS persistent | MaleCNS tail | FlyWire persistent | FlyWire tail |
|---|---|---|---|---|
| base | 4/4 | 3,340 | 4/4 | 7,860 |
| chemical LN→PN 0, no gap | 4/4 | ~460 | 4/4 | ~740 |
| gap c0 0.01 | 4/4 | ~3,030 | 4/4 | ~2,700 |
| gap c0 0.02 | 4/4 | ~5,000 | 4/4 | ~9,200 |
| gap c0 0.04 | 4/4 | ~5,600 | 3/4 | ~11,300 |

A spikelet equal to c × 40 mV, delivered instantly, gives far too much charge. A real spike lasts about 1 ms, and the junction and membrane (τm 20 ms) filter it, so the charge moved is about c × 40 mV × 1/20 ≈ **c × 2 mV**. The first attempt overstated that by about 20×.

### Second attempt: charge-matched 2 mV spikelets, c0 = 0.04 / 0.1 / 0.2
| Condition | Connectome | Persistent (hot+vinegar) | Target PN (vinegar / hot) | Other PN | eLN during odor | Sugar MN9 | No stimulus |
|---|---|---|---|---|---|---|---|
| base | MaleCNS | 4/4 | 205 / 56 Hz | 82 Hz (560 active) | 177 Hz | n/a | n/a |
| gap 0.04 | MaleCNS | 2/4 (vinegar only, ~320 neurons) | 163 / 51 Hz | 1–4 Hz | 2–66 Hz | n/a | n/a |
| gap 0.1 | MaleCNS | **0/4** | 143 / 48 Hz | 1 Hz (~20 active) | ~0–1 Hz | n/a | n/a |
| gap 0.2 | MaleCNS | **0/4** | 117 / 42 Hz | 0.5–0.8 Hz | ~0 Hz | n/a | n/a |
| base | FlyWire | 4/4 | 218 / 120 Hz | 84 Hz (548 active) | 200 Hz | 82 Hz | silent |
| gap 0.04 | FlyWire | 3/4 | 173 / 76 Hz | 1.1–1.5 Hz | 1–132 Hz | 82 Hz | silent |
| gap 0.1 | FlyWire | 2/4 (vinegar ~540) | 182 / 78 Hz | 1.3–1.6 Hz | 0–121 Hz | 82 Hz | silent |
| gap 0.2 | FlyWire | **0/4** | 130 / 70 Hz | 0.8–1.1 Hz | ~0–1 Hz | 82 Hz | silent |

**At c0 = 0.2, the same rule abolishes the attractor on both connectomes.** Odor responses become glomerulus-specific: the stimulated glomeruli's PNs fire at 40–130 Hz, while the other PNs fall from about 83 Hz to under 1 Hz. Sugar→MN9 (82 Hz) and silence without a stimulus are unchanged on FlyWire. Odor no longer drives MN9 on MaleCNS.

Is c0 = 0.2 consistent with the paper? Each strong pair's coupling coefficient depends on the receiver's total coupling as well as on c. A passive estimate is cc ≈ c_ij/(1 + Σ_k c_ik). That gives a median eLN→PN coupling coefficient of **0.016 / 0.021 / 0.024** for strong pairs at c0 = 0.04 / 0.1 / 0.2, inside Yaksi & Wilson's range of up to 0.04. PN→eLN comes out at about 0.01. This is my estimate, not a measured value.

## Caveats
1. **It works mainly by shunting the eLNs.** Each eLN touches about 210 PNs, so its total coupling is several times its leak (Σc ≈ 4.5–9 at c0 0.1–0.2). That clamps eLNs near PN voltage, and they nearly stop firing (about 0 Hz during odor). In the fly, eLNs respond broadly to odors. So the attractor is gone, but the eLNs are switched off. That is too strong, and it is close to what the "eLN off" cut did on Day 1.
2. **The result depends on the spikelet assumption.** Full-height spikelets made things worse, and charge-matched spikelets fix it. Report both.
3. **Where the gap junctions go is a guess.** They follow chemical contact, because EM does not resolve gap junctions.
4. The rectification Yaksi & Wilson saw (depolarization passed more easily) is not modeled; the coupling is symmetric.

## Next
- **Running now:** all 17 words, sugar+bitter and no-stimulus on MaleCNS at c0 0.1 and 0.2 (`scripts/w2d3c_gap_allwords.py`, results in `results/malecns/w2d3c_allwords.csv`). This checks the non-olfactory words and the looming tail.
- **Fixing caveat 1:** normalize each neuron's total coupling (cap Σc at about 1), or put gap junctions only on the strongest contacts (≥ 50 synapses, about 12 per PN). Either gives fewer, stronger junctions. Then check that eLNs still respond to odors (tens of Hz) and the tail stays at zero.
- After that, on to Days 4–5: the 16-word readout and the shuffled control, on whichever calibrated version passes.

Scripts: `scripts/w2d3_gap_malecns.py`, `scripts/w2d3_gap_flywire.py`, `scripts/w2d3b_gap_charge.py`, `scripts/w2d3c_gap_allwords.py`; module `sim/gap.py`. Logs: `results/malecns/w2d3*_log.txt`, `results/flywire/w2d3*_log.txt`.

## Update: all words on MaleCNS (seed 0; `results/malecns/w2d3c_allwords.csv`)

Brain neurons still active after the stimulus (persistent if more than 20):

| Word | base | gap 0.1 | gap 0.2 |
|---|---|---|---|
| vinegar, fruit, co2, mold, hot, dry | ~3,350 each | 0–2 | 0 |
| pheromone | 3,357 | 168 | 0 |
| cold | 3,361 | 247 | 220 |
| humid | 3,389 | 206 | 172 |
| wind | 3,321 | 420 | 395 (none in the antennal lobe) |
| looming | 502 | 502 | 502 (none in the antennal lobe) |
| sugar+bitter | 0 | 0 | 78 |
| sugar, bitter, water, salt, umami, sound, none | 0 | 0 | 0 (78 for sugar+bitter is the only exception) |

- The big olfactory/thermo/hygro attractor (about 3,350 neurons for 10 words) is gone. Small residual tails in the antennal lobe remain for cold and humid (about 50–60 AL neurons).
- **A second loop outside the antennal lobe** sustains looming (502) and part of wind (about 400). The gap-junction change doesn't touch it, as expected. It needs its own SCC search (the C1 tools).
- The gustatory words are unchanged by the gap junctions. Note that MaleCNS sugar→MN9 is only 14 Hz in this brain-only config; FlyWire gives 82 Hz.
- Pheromone (DA1/VA1v) still drives MN9 at about 70 Hz without any attractor. That pathway is MaleCNS-specific and worth a look.
- Odors now activate 30–110 AL neurons, compared with about 930 at baseline.
