# Side-kick: remove the cold/humid lasting state (9 Nov 2026)

Evidence: `scripts/w3d1b_cold_tail.py` (`results/loop/w3d1b_log.txt`, `w3d1b_tail_*.csv`) and `scripts/w3d1c_cold_loop.py` (`results/loop/w3d1c_log.txt`, `w3d1c_lesions.csv`). Calibrated MaleCNS, seed 0.

## What is stuck, and where
- After 0.5 s of cold, about **244 neurons are still firing at 3.0 s**, with no decay from 1.0 s on. Humid leaves about 211 neurons, and it is the **same state**. Hot leaves 0 neurons by 1.0 s.
- The core sits in the VP3 (cold) glomerulus. At its center are the **cold sensory neurons themselves: TRN_VP3a keeps firing at about 165 Hz with no stimulus.** Around them are VP3+_l2PN (170 Hz), M_l2PN10t19 (93 Hz), v2LN4 (cholinergic LN, 115 Hz) and l2LN22 (163 Hz, transmitter "unclear", so modelled as excitatory). Everything else (about 200 neurons: WED/AOTU/PLP/LAL cells and 36 DNs) is downstream. All tail neurons with outgoing loops form one excitatory strongly connected component of about 127 neurons.
- **The loop, in synapse counts:** TRN_VP3a → l2LN22 (2,365) → TRN_VP3a (1,467), and TRN_VP3a → v2LN4 (1,377) → TRN_VP3a (790), plus TRN_VP3a → TRN_VP3a (1,335). The model treats central synapses onto sensory axon terminals as ordinary excitation that can make the sensory neuron spike, so the cold neurons re-excite themselves through two LNs.
- **Why cold and not hot:** TRN_VP3a gets strong excitatory feedback from l2LN22 and v2LN4. Humid gets in because HRN_VP5 excites v2LN4 (446) and l2LN22 (35), which then light up TRN_VP3a. Hot (VP2) has no comparable feedback.

## Three single cuts each abolish it (seed 0, tail = neurons firing 1.5–2.0 s)
| Change | cold: active during stimulus / tail | humid: active / tail |
|---|---|---|
| calibrated (now) | 360 / **245** (TRN_VP3a 165 Hz) | 330 / **202** |
| A. no central chemical input onto any sensory neuron | 256 / 0 | 144 / 0 |
| B. no central chemical input onto thermo- and hygrosensory neurons only | 256 / 0 | 144 / 0 |
| C. l2LN22 made inhibitory (FlyWire calls it GABA) | 193 / 0 | 132 / 0 |

## Is it a must-do?
Not for the loop itself: the episode-trained readout already decodes 100%, including all 7 hot-after-cold/humid windows. But it is a real model artifact. Sensory neurons firing at 165 Hz forever after the stimulus is gone is not biology, and it would carry into every LLM → fly round. The fix looks small, so it's worth doing **before Day 3**, because the action readout will be trained on whatever lasting state the model has.

**Recommended rule: A** (no central chemical input onto sensory neurons). In the fly, synapses onto sensory axon terminals adjust transmitter release (presynaptic inhibition, GABA-B; Olsen & Wilson 2008). They don't make the sensory neuron fire spikes. A follows from that principle for every sensory neuron, so it isn't a cold-specific patch. C is the backup: it depends on one uncertain transmitter call, though it matches FlyWire and the C7b audit. The extra cost of A is that it removes the feedback that boosted cold and humid during the stimulus (active neurons 360 → 256 and 330 → 144). That is acceptable if the words still decode.

## Plan (about half a day of compute, 3 checkpoints)
1. **Make it a calibration option.** Add `sensory_input=False` to the model setup (`w3d1_common.calib_model`), which zeros the rows of W for neurons whose superclass contains "sensory". Gap junctions are unaffected (they are eLN–PN only). Call this config `calib2`.
2. **Regression checks on calib2.** All 17 words + none, seed 0, 1.6 s trials, as in w2d3c:
   - tail ≤ 20 neurons for cold, humid, and every word that was clean before;
   - check whether the looming (~500) and wind (~395) tails change (JO and visual inputs are sensory too). Report it either way; this rule is not meant to fix them;
   - vinegar/hot glomerular specificity (target PN vs other PN rates) close to Week 2;
   - FlyWire with the same rule: sugar → MN9 still about 82 Hz, and no-stimulus still silent.
   *Checkpoint:* if A breaks a taste or odor word, fall back to B, then C.
3. **Re-run the readout data on calib2.** 17 words × 10 seeds (`w2d4_word_trials.py calib2`), plus the readout (`w2d5_readout.py`). Then the Day 1 episodes at 0 ms spacing only (40 episodes, about 55 min on 4 processes). Target: isolated readout ≥ 99% in episodes with no hot→cold errors, and gap windows reading `none`.
4. **Adopt or not.** If 2 and 3 pass, calib2 becomes the substrate for Day 2 onward, and the Explorer gets a calib2 view later (Day 6). If not, keep calib with the episode-trained readout and list the cold state as a limitation.

Out of scope: the non-AL looming/wind loop (separate SCC search, later), eLN shunting.
