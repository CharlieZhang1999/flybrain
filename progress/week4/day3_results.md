# Week 4 Day 3: VNC sanity checks and calibration

## Built
- `sim/gap.py`: `build_gf_gap(meta, spikelet_mV)`, which adds rectifying giant-fiber junctions GF → TTMn and GF → PSI (ipsilateral pairs, one-way rows of C, so the targets do not load the GF). `merge_gap` joins them with the eLN–PN junctions.
- `sim/cns.py`: `cns_model(..., gf_gap=None, mn_glu=False)` and `W4_CALIB = dict(gf_gap=8.0, mn_glu=True)`, the day-3 decisions below. The defaults are unchanged so days 1–2 stay reproducible. **From day 4 on, pass `**W4_CALIB`.** GF junctions are added only for `scope='cns'`.
- Scripts: `w4d3_dn_panel.py` (+ `w4d3_panel_summary.py`), `w4d3_vnc_sign_audit.py`, `w4d3_looming_tail.py`, `w4d3_tail_loops.py`, `w4d3_loop_lesions.py`, `w4d3_tail_cut_ascending.py`. Outputs are in `results/w4/sci/`.

All runs: full CNS, PyTorch on CPU, float32, device RNG, 3 seeds. DNs and looming were driven at 150 Hz Poisson. Rates are means over seeds.

## (a) Descending-neuron panel, real vs shuffled
Drive 0–0.3 s, rates over 0–0.3 s. `gf8` means 8 mV GF junctions and `mnglu` means MN outputs set to glutamatergic (see b, c).

| stimulus | readout | calib | shuffle | calib+gf8+mnglu | shuffle+gf8+mnglu |
|---|---|---|---|---|---|
| GF (DNp01) | TTMn, Hz | 11.1 (right side only) | 0 | 120.6 | 120.6* |
| GF | PSI, Hz | 0 | 0 | 120.6 | 120.6* |
| GF | DLMn (via PSI), Hz | 0 | 0 | 6.2 | 0 |
| GF | TTMn spikes within 3 ms of a GF spike | 7% | 0 | 88% | 88%* |
| MDN | hind / front leg pools, Hz | 1.87 / 0.03 | 0.01 / 0.33 | 1.82 / 0.03 | 0.01 / 0.33 |
| DNg100 | front / mid / hind leg pools, Hz | 1.7 / 1.6 / 6.4 | 0 / 0 / 0.7 | 1.7 / 1.8 / 5.6 | 0 / 0 / 0.7 |
| looming | TTMn / DLMn, Hz | 68 / 97 | 0 / 0 | 277 / 105 | 0 / 0 |
| none | active MNs | 0 | 0 | 0 | 0 |

\* By construction: the gap junctions are not shuffled (like eLN–PN), so GF → TTMn/PSI survives the shuffle. The chemical PSI → DLMn step does not.

Active motor neurons (any spike in 0–0.3 s), real vs shuffled: MDN 44 vs 2, DNg100 74 vs 2, DNa02 L 18 vs 0, DNa02 R 6 vs 2, looming 181 vs 11.

Reading:
- **MDN passes.** It drives the hind legs about 60× more than the front legs (trochanter-femur 12–19 Hz, thorax-coxa 6–7 Hz), consistent with MDN acting mainly on the hind legs for backward walking. The shuffled control spreads a little activity over the wrong legs.
- **DNg100 partly passes.** It reaches all three leg pairs (hind > front ≈ mid), as a walking command should. It also drives wing pleurosternal and hg steering MNs at about 17 Hz, which we did not expect. There is no rhythm, because the model has no pattern-generating mechanism.
- **DNa02 fails as a turning check.** One-sided drive gives weak and inconsistent asymmetry: leg asymmetry (ipsi − contra)/(ipsi + contra) is +0.16 for the left and +0.52 for the right, at < 0.5 Hz, while the neck is slightly ipsilateral (2.8 vs 2.3 Hz) for the left only. In the fly, DNa02 changes the stepping of an already walking fly. Without ongoing walking, the panel cannot test it. We should re-test it on top of DNg100 drive.
- **Looming gives a takeoff pattern only with the real wiring** (DLM 105 Hz, TTM, DVM). The shuffled connectome reaches 11 MNs and no TTMn/DLMn. **Caveat:** looming drives the GF at about 275 Hz. In the fly, the GF fires one or a few spikes at takeoff. With the 8 mV junction, TTMn now follows the GF at 277 Hz. That is a GF calibration problem (no adaptation, and every looming-sensitive input is driven continuously). It is not something the junction causes.

## (b) GF → TTMn / PSI gap junctions
The chemical synapse alone (GF→TTMn 70 synapses right, 20 left) gives 7% one-to-one following, on one side only. We swept the spikelet size; the rest-to-threshold gap is 7 mV.

| spikelet | TTMn Hz (GF at about 160 Hz) | PSI Hz | DLMn Hz | TTMn following |
|---|---|---|---|---|
| none (calib) | 11.1 | 0 | 0 | 7% |
| 4 mV | 58.9 | 52.2 | 1.0 | 58% |
| 8 mV | 120.6 | 120.6 | 6.8 | 88% |
| 12 mV | 120.6 | 120.6 | 6.8 | 88% |

**Decision: 8 mV (`GF_GAP_MV`).** It is the smallest value that saturates following. The remaining 12% are probably TTMn refractory misses at GF inter-spike intervals under 2.2 ms, plus spikes beyond the 3 ms window. This is the reliable one-to-one GF → TTMn relay described in the fly, now on both sides. MDN, DNg100 and DNa02 are unchanged by it. PSI → DLMn stays weak (6 Hz) because DLMn receives strong inhibition. We leave that as it is.

## (c) VNC transmitter-sign audit
Synapse-weighted presynaptic transmitter shares (`results/w4/sci/w4d3_sign_audit.txt`):

| edges | synapses | ACh | GABA | Glu | unclear (→ +1) | consensus missing (fell back to prediction) |
|---|---|---|---|---|---|---|
| brain → brain | 96.8M | 59.8% | 18.3% | 18.8% | 0.8% | 1.6% |
| VNC → VNC | 21.5M | 55.3% | 31.8% | 11.3% | 1.1% | 1.8% |
| brain → VNC (DNs) | 2.8M | 64.5% | 20.7% | 10.7% | 1.4% | 3.9% |
| any → MN | 2.7M | 57.8% | 28.8% | 12.1% | 0.9% | 1.4% |
| **MN → any** | **0.03M** | 32.5% | 5.8% | 25.6% | **33.9%** | 43.8% |

- The VNC is no less certain than the brain. "Unclear" is about 1% of VNC synapses, with the same share of low-confidence predictions. No change is needed there.
- Glutamate onto MNs (12%) comes from the expected hemilineages (21A, 08A, 02A), and is inhibitory under the Shiu convention. 21A splits ACh/Glu by type, which matches the per-body predictions. We keep the convention.
- **MN outputs are mis-signed.** Fly motor neurons are glutamatergic, but the per-body prediction labels their few central synapses (30k, 0.14% of VNC synapses) 34% unclear (+1) and 33% ACh. That made an excitatory DLMn ↔ DLMn loop (see d). **Decision: `mn_glu=True`**, so every MN output is −1. Its effect on the panel is small (MDN hind legs 1.87 → 1.82 Hz, DNa02 asymmetry a little stronger).

## (d) The looming tail: brain-side or a VNC loop?
Looming 0–0.4 s, run to 1.0 s. The table shows active neurons in 0.6–1.0 s per seed (seeds 0 / 1 / 2).

| model | brain, Day 1 calib | VNC, Day 1 calib | brain, W4_CALIB | VNC, W4_CALIB |
|---|---|---|---|---|
| brain only (Week 3) | 29 / 2 / 2 | – | 29 / 2 / 2 | – |
| full CNS | 2 / 565 / 525 | 47 / 77 / 71 | 532 / 531 / 538 | 75 / 73 / 71 |
| full CNS, VNC → brain cut | 29 / 2 / 2 | 47 / 46 / 47 | 29 / 2 / 2 | 46 / 47 / 47 |
| full CNS, brain → VNC cut | 29 / 2 / 2 | 0 | 522 / 370 / 2 | 0 |

It is both, and neither is the Week 3 tail:
1. **A small loop that survives without the brain (about 47 neurons in the VNC).** Excitatory components among the tail neurons (`w4d3_tail_loops.py`): the bilateral **DNg33** pair (mutual excitation; DN somata are in the brain, axons in the VNC), an **IN19B040** quartet, and **DLMn ↔ DLMn** through the mis-signed MN outputs. Cutting all three removes the VNC-only tail (seed 0: 47 → 0). Each cut on its own leaves part of it (`w4d3_loop_lesions.txt`).
2. **A large brain state ignited by ascending input (about 530 brain neurons, plus about 25 VNC neurons it drives).** It is centred on the central complex (EPG 46, Delta7 42, PEN 42, EL, PEG, ER). About 220 neurons are central complex, and the rest include 69 DNs that keep driving the VNC. The brain-only model never enters this state, and cutting VNC → brain prevents it. But **once ignited it is self-sustaining.** Using day-2 resume, cutting VNC → brain at 0.6 s leaves it intact at 1.0–1.2 s (seed 1: 613 → 584 brain neurons, central complex 217 → 221; seed 2: 631 → 581). With W4_CALIB it ignites in 3 of 3 seeds. With brain → VNC cut, the GF junctions alone feed TTMn/PSI → ascending neurons, and that ignites it in 2 of 3 seeds. So the trigger is ascending feedback from the flight/jump circuit.

The whole EPG/Delta7/PEN ring being on is not a heading bump. It is the same kind of runaway recurrent state as the Week 1 attractors (`progress/week1/locate-brain-attractor.md`), now reachable through the VNC.

## Consequences for day 4 (regression dataset)
- Use `cns_model('cns', cfg, **W4_CALIB)`.
- **Episodes cannot be chained freely.** After any looming-like drive, the central complex state persists, and its descending outputs keep about 70 VNC neurons active. Either start each episode from rest (no `state=` carry-over across episodes), or mark episodes after ignition and report R² with and without them. Better still: find the ascending neurons that ignite the state and decide on a calibration (as with the Week 1 attractors) before generating the 200-episode set.
- Cap or adapt GF drive, so that looming episodes do not saturate TTM/DLM at 100–280 Hz and dominate pool variance.
- Re-test DNa02 on top of DNg100 (turning during walking) as part of the pilot.
