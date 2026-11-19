# Week 4 Day 5 (T2c): brain bursts and post-`wind` persistence in the Day 5 episodes

**Diagnosis only. No fix has been implemented.**
- Nothing under `sim/` or `stimuli/` was changed. `git status` shows only the new `scripts/w4d5_burst_*.py` and this report.
- Every ablation acts on a copy of W or of the gap dict inside the scripts.

**Revision 1 (after QA).** QA's scripts and outputs are in `/tmp/claude-0/qa-burst/`.
- **Corrections M1–M5 and the low-priority items:** applied below.
- **Wind-state wording:** now *metastable*. "Bistable" is withdrawn.
- **New in this revision:**
  - the loop-gain scan (candidate fix B), `w4d5_burst_runs.py gain`;
  - the analysis part `gain`.

**Model and runs.**
- **Model:** `cns_model('cns', cfg, **W4_CALIB)`, CPU float32, `rng='torch'`, fixed seeds.
- **Simulation used:** about 45 min of wall time over 70 run segments, plus model builds. One simulation ran at a time.
- **Reproducibility check:** the lesion script re-runs Day 5 ep009 to 2.0 s, saves the state and resumes it with the intact W. The result is **spike-for-spike identical** to the saved `ep009_calib.npz` (189,587 spikes).

**Scripts:**
- `scripts/w4d5_burst_common.py`: masks, AL-event detector, bin counts.
- `scripts/w4d5_burst_runs.py`: all simulations. Modes `al_rate`, `al_head`, `al_ablate`, `wind`, `lesion`, `gain`.
- `scripts/w4d5_burst_analyze.py`: parts `census`, `al`, `wind`, `lesion`, `gain`.
- `scripts/w4d5_burst_loop.py`: connectome view of the loop.
- `scripts/w4d5_burst_metric.py`: the Q4 metric.

**Outputs:** `results/w4/regression/burst_diag/`:
- `runs/*.npz`, `runs_log.txt`
- `analyze_*.txt`
- `metric.csv`, `metric.txt`
- `scripts_backup/`: a copy of the scripts on the mount.

**Definitions:**
- **Brain bin:** distinct `brain_mask` neurons active in a 100 ms bin, word-driven neurons excluded (as in `stop_check.py`).
- **AL event:** at least 100 AL neurons (all ALPN + ALLN, 1,106 neurons) fire within a 10 ms window.
- **GNG core:** DNge027, GNG423, DNg21, DNg62, DNge044, DNge022, DNge078, DNge019, AN01A014, SAD093, DNge011/039/012 and DNg12_a–h.
- **LOOP5:** DNge027, GNG423, DNg21, DNg62, DNge044. Edges among them: 101 W entries.
- **OL:** `ol_intrinsic` neurons.

## Answer in brief
These are two unrelated phenomena, and both are most likely model artifacts.

1. **Post-`wind` persistence.**
   - **What it is:** a self-exciting descending-neuron loop in the GNG, among DNge027 (an ACh pair), GNG423, DNg21, DNg62 and DNge044.
   - **How wind enters:** mainly JO-C/E → SAD093 → GNG423 / DNg62.
   - **What is downstream, not part of it:** the optic-lobe tail (Mi18, Dm17, Pm8) and the VNC/MN tail.
   - **Behaviour:** the state is **metastable**. With no input after word offset it lasts 0.44 s to ≥ 1.0 s, then collapses abruptly. Background *before* offset clearly lengthens it: with no background at all it dies within 0.3 s. Background *after* offset helps in one comparison and not in another.
   - **Week 3b never tested `wind`.**
   - **Partial fix:** scaling the LOOP5 edges to 0.3 removed persistence in 3 of 3 wind episodes. It left non-wind episodes spike-identical and cut wind-word MN spikes by about 15–19%.
2. **AL bursts.**
   - **What they are:** a relaxation oscillation of the gap-coupled eLN–PN network under ORN background.
   - **Role of the eLN–PN junctions:** they do not create synchrony out of nothing. They shunt the eLNs, which otherwise fire tonically at about 200 Hz, so the coupled network charges slowly as a whole and then fires once, all together.
   - **Removing the junctions** does not abolish the activity. It turns the episodic events into continuous AL activity.

## Week 3b check: what it actually showed for `wind`
`scripts/w4d3b_word_check.py` tested **only** sugar, bitter, vinegar, looming, sound and hot (`WORDS = [...]` in the script; `results/w4/sci/w4d3b_word_check.txt`). It used brain scope, seed 0, a 0.5 s drive and no background. **`wind` was never tested,** so the claim that every word returns to silence did not cover it.

## Q1: wind persistence
### Test matrix (C2; `w4d5_burst_runs.py wind`, `analyze_wind.txt`)
**Specs:**
- **A:** Day 5 ep009 exactly: wind 0.857–1.771 s, plus humid 1.639–2.0 s, background 2.76 Hz, T 2.5 s.
- **B:** wind alone, 110 Hz, 0.3–1.0 s, background 2.5 Hz class_v2, T 2.0 s, Poisson seeds 1 and 2.

**Method:** the background switch is made at wind offset by state/resume. Words are never changed.

**Cells:** the end of the state (last DNge027 spike, or the bin where brain counts drop to background) and brain bins (word-driven neurons excluded) after offset.

| condition | A (ep009) | B seed 1 | B seed 2 |
|---|---|---|---|
| bg on (as dataset) | 474–482 to 2.5 s (≥ 0.73 s) | ~460, ends 1.86 s (0.86 s) | ~470, ends 1.50 s (0.50 s) |
| bg during word, off after | ends about 2.05 s, 50 ms after humid ends (**not input-free; see M2**) | 392–419 **to 2.0 s** (≥ 1.0 s, no input) | 405–445, ends 1.51 s (0.51 s, no input) |
| no bg at all | 437 → 90 → 33 → 0 (≤ 0.3 s) | 418 → 138 → 0 | 99 → 0 |
| no bg during word, bg on after | 472–765 to 2.5 s | 446–479 to 2.0 s | 428–512 to 2.0 s |
| **shuffle**, bg on | 15–35 (no state) | 16–29 | 18–32 |

**Resume from inside the established ep009 state at 2.0 s, background off** (`lesion`, `bg_off`): the state holds 350–426 brain neurons per 100 ms, then **collapses at 2.443 s**. QA's brain counts per 10 ms from 2.40 s were 144, 188, 193, 162, 35, 21, 2, 0. It lasted 0.44 s with no input. It does **not** persist to 2.5 s; my earlier 2.2–2.5 s count hid the collapse.

**Findings:**
- **Wind alone ignites the loop.** DNge027 fires at **104–134 Hz** during the word in all 12 Day 5 wind episodes (QA) and in no bg-only episode. It does not ignite on shuffle.
- **Without background at all it dies within 0.1–0.3 s** of offset (3 of 3).
- **With no input after offset, the state lasts 0.44 s to ≥ 1.0 s** (B seeds 1 and 2; the ep009 resume). It is therefore self-sustaining for a limited and variable time, i.e. **metastable, not bistable.**
- **Background after offset (M4): mixed.**
  - **Holds:** no background vs background after offset only: dies within ≤ 0.3 s vs lasts to the end in 3 of 3.
  - **Does not hold:** background throughout vs background off at offset, spec B: 0.86 s vs ≥ 1.0 s (seed 1) and 0.50 s vs 0.51 s (seed 2).
  - **Conclusion:** background **before** offset clearly matters. Its effect after offset is not established.
- **Lasting to the end (M1):** 4 of 6 runs with background after offset (A bg on; bg-after A, B1, B2). B seeds 1 and 2 with background on end at 1.86 s and 1.50 s.
  - Day 5: ep005/009/010/011/016 run to the end. ep002 is borderline: its last DNge027 spike is at 2.445 s.
- **Termination is abrupt and stochastic.**
  - It ended early in Day 5 ep003, 007, 008, 013, 015 and 017, and in B seed 2.
  - It does not track the episode's background rate: ep015 at 1.04 Hz held 0.7 s, while ep007 at 2.96 Hz stopped at offset.
  - Its inhibitors (GNG281, GNG166, DNg98, GNG292, IN27X001) fire tonically inside the state.
  - **What ends it was not identified.**

### Loop and hop path (`w4d5_burst_loop.py`, `analyze_loop.txt`; QA `static.out`)
Synapse numbers in the format "A → B N syn" are **whole-type-pair sums**. Single cell-to-cell edges in the loop carry **1–273 synapses** (up to about 44 mV per spike).

**Entry from JO-C/E.** There are no direct JO-C/E → LOOP5 synapses. The main route is:
- JO-C/E → **SAD093** (1,108 syn in total; 69 JO cells with an edge of ≥ 1 mV);
- SAD093 → **GNG423** (470 syn) and → **DNg62** (422 syn).
- SAD093 does not target DNge027.

The route JO-EV6 → CB2348 → DNge027 that I gave earlier is negligible: CB2348 → DNge027 has only 7 synapses (1.1 mV). Other minor 2-hop routes go via DNge132 → DNge044.

**Core loop.** Type-pair sums, with mean-field input to DNge027 in the tail state (an estimate, not a membrane potential):

| pair | into DNge027 | out of DNge027 |
|---|---|---|
| GNG423 | 908 syn, 29.7 mV | 429 syn |
| DNg21 | 825 syn, 24.0 mV | 366 syn |
| DNg62 | 404 syn, 15.5 mV | 331 syn |
| DNge044 | 213 syn, 9.0 mV | 498 syn |
| DNge027 L↔R | 162 / 200 syn | — |

**Downstream read-outs:**
- **DNg12 (a, c, e, f) and DNge019:** driven by DNge027 (10–16 mV), with DNge019 and GNG423.
- **Optic lobe:** DNge027 → GNG282 (31 mV) → MeVC4a/b, MeVC1 and MeVCMe1 (visual centrifugal, 24–25 mV) → Mi18 (MeVC4a/b), Dm17 (MeVC1, 39 mV) and Pm8.
- **VNC:** DNg12, DNge0xx and other DNs → INs/MNs. In the tail, the tergopleural promotor MN fires at about 80 Hz and MDN at about 18 Hz.

### Edge cuts on a copy of W (resume 2.0 → 2.5 s from ep009; distinct neurons active 2.2–2.5 s)

| variant | brain | GNG core | OL | VNC non-sens | MN |
|---|---|---|---|---|---|
| intact (= saved ep009) | 546 | 73 | 119 | 355 | 43 |
| background off (**collapses at 2.443 s**) | 437 | 73 | 118 | 261 | 39 |
| **cut all DNge027 outputs (837 entries)** | **141** | **0** | **0** | 99 | 2 |
| **cut edges inside the GNG core** | **133** | **0** | **0** | 94 | 1 |
| cut DNge027 L↔R only | 520 | 71 | 110 | 353 | 41 |
| DNge027 L↔R × 0.3 / × 0.1 | 541 / 528 | 73 / 72 | 127 / 114 | 410 / 360 | 48 / 44 |
| cut VNC → brain (ascending) | 522 | 72 | 102 | 402 | 47 |
| cut brain → VNC | 546 | 69 | 105 | **67** | **1** |
| cut central → OL intrinsic | 459 | 73 | **0** | 375 | 42 |

Brain counts of 133–141 after the decisive cuts are the background level (74–95 per 100 ms). QA's cut of only the 5-type LOOP5 edges also ended the state (`/tmp/claude-0/qa-burst/lesion_cut_loop5.npz`).

- **Maintained by:** the multi-neuron GNG DN loop through DNge027.
- **Not maintained by:**
  - **the DNge027 pair alone:** it is not a two-cell clique like DNg33;
  - **ascending feedback;**
  - **the optic lobe or the VNC:** both are downstream.

### Verdict on Q1
`wind` plus the model's wiring creates the state; background does not.
- **Behaviour:** the state is metastable. It lasts 0.44 s to ≥ 1.0 s with no input after offset.
- **Role of background:** it is required for the state to outlast offset by more than 0.3 s (no background at all: ≤ 0.3 s), with background before offset clearly mattering. Its effect after offset is mixed.
- **Shuffle:** the state never forms.

## Q2: AL bursts
**What an event is** (ep205–208): a single synchronous volley within about 10 ms.
- **Size:** 366–440 brain neurons, mostly one spike each: about 135 ALPN, about 100 ALLN and about 170 untyped `cb_intrinsic`.
- **Leaders:** il3LN6, lLN2P_c and lLN2F_a.
- **Between events the AL is nearly silent:** ACh LNs 0 Hz, PNs 0.02–0.04 Hz, GABA LNs 3.6–4.2 Hz.
- **Inter-event intervals:** 0.38–1.83 s.

### Rate × seed (class_v2 background only, T 2.0 s; AL events per run; `analyze_al.txt`)

| Hz | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 1.5 | 0 | 0 | 0 |
| 2.0 | 0 | 0 | 0 |
| 2.5 | 2 | 0 | 0 |
| 3.0 | **2** | 0 | 0 |
| 3.5 | 0 | **1** | 0 |

**What can be said about the rate dependence:**
- **Events are stochastic and absent in most seeds.**
- **They occur at 2.5–3.5 Hz,** so the Day 5 absence at 3.0 Hz is not a sharp upper edge.
- **Day 5 (T 2.5 s) had events in 4 of 4 episodes at 2.11–2.78 Hz,** more often than these 2 s runs at 2.5–3.0 Hz.
- **No events in 6 runs at 1.5–2.0 Hz.**
- **Whether a true rate window exists is not settled** by these numbers, in either direction.

Earlier runs (`analyze_census.txt`): head-only 5 Hz 0 of 2 runs, head-only 2 Hz 0 of 2 (1 s each).

### Which background drives them (2.5 Hz, seeds 0–2)

| background | events per seed |
|---|---|
| olfactory ORNs only (2,639) | 0, 1, 1 |
| class_v2 minus ORNs (11,049) | **0, 0, 0** |
| all head sensory | 3, 1, 1 |
| earlier head-only 3 Hz (bg_diag, QA) | 1, 1, 1 |

ORN background appears **necessary and sufficient.**
- **Hedge:** n = 3 per arm. Because the base rate is low, the decisive comparison is ORN-only with events vs no-ORN without, i.e. essentially one informative contrast.

### Ablations (Day 5 ep205 and ep207 bg-only specs, same Poisson seed; the saved runs have 3 events each)

| ablation | events (ep205 / ep207) | brain bin median (ep205 / ep207) |
|---|---|---|
| none | 3 / 3 | 56 / 74 |
| `apl=None` | 3 / 3, identical spike trains | 56 / 74 |
| `presyn=None` | 2 / 3 (shifted) | 60 / 76 |
| AL gap junctions × 0.5 (GF kept) | continuous from 0.2–0.3 s | **283 / 303** |
| AL gap junctions removed | continuous from 0.03 s | **374 / 392** |

**With the junctions removed,** 100 of 144 ACh LNs fire tonically at a median of 208 Hz, plus 72 GABA LNs.

- **APL:** irrelevant. In this model it acts only on KCs, and KCs are not in the events.
- **Presynaptic inhibition:** not the cause.
- **Gap junctions:** they do not cause the bursts; they hold them back.
  - eLNs have a total coupling of about 9× their leak (`sum_c_LN_median` 8.97). The ACh LN → PN chemical synapses are cut and replaced by the junctions.
  - Coupled to mostly silent PNs, the eLNs are shunted below threshold, so the whole network integrates ORN input slowly.
  - When it crosses threshold, eLNs, PNs and LNs fire together, then reset and recharge.

### Why the all-sensory 3 Hz runs show none
Re-scoring the existing T2a runs (`analyze_census.txt`):

| run (all sensory, 3 Hz) | events |
|---|---|
| intact (seeds 0–2) | 0 / 0 / 0 |
| VNC → brain cut, seed 0 | 2 |
| VNC → brain cut, seed 1 | 0 |
| VNC inhibition × 2 | 1 |
| VNC sensory output × 0.5 | 1 |
| brain → VNC cut, seed 0 | 1 |

Head-only 3 Hz gave 1 event in 3 of 3 runs.
- **What this suggests:** the bristle/abdominal-driven VNC activity, which class_v2 removed, suppresses events through ascending input.
- **Strength of the evidence:** single seeds per perturbation, and no ascending path was identified.

## Q3: biological or artifact?
**Not re-checked:** I could not re-open the primary papers in this session. Every literature claim below is from memory and should be checked before it is cited.

- **ORN spontaneous rates:** about 1–20 Hz, receptor-dependent, often about 8–10 Hz (de Bruyne, Foster & Carlson 2001, Neuron 30:537; Hallem & Carlson 2006, Cell 125:143).
  - The model's 2–3 Hz is therefore low.
  - The model events are a property of the model's regime, not a measured property.
- **Spontaneous AL population bursts:** PN spontaneous firing is driven by ORN input and is irregular (Kazama & Wilson 2009, Nat Neurosci 12:1136). eLNs are electrically coupled to PNs (Yaksi & Wilson 2010, Neuron 67:1034; Huang et al. 2010, Neuron 67:1021).
  - I know of no report of AL-wide synchronous volleys at rest. This absence is **unverified**.
  - The model facts point to an artifact: an AL that is silent between volleys, and a volley that needs the shunting effect of one large, uniform junction calibration.
- **Persistence after wind or JO stimulation:** JO-C/E encode wind and gravity (Yorozu et al. 2009, Nature 458:201). Antennal mechanosensory DNs track wind during the stimulus (Suver et al. 2019, Neuron).
  - **The volume and page numbers for Suver et al. 2019 are uncertain.** I gave 101:1186 earlier; QA recalls 102:828. Check before citing.
  - I know of no report that DNge027, DNg12 or Mi18 hold activity after wind ends. This is **unverified**.
  - **The model state:** DNge027 at 104–134 Hz, about 40 MNs and 350 VNC INs active, and a flat 440–480 brain neurons, ending abruptly at a random time.
  - **Verdict: most likely an artifact.** It comes from strong reciprocal ACh DN–DN wiring in a LIF without adaptation or depression (the same root cause as the DNg33 and looming cases in Week 3b).

## Q4: a pathology metric that ignores word-driven bins
`scripts/w4d5_burst_metric.py`; results in `metric.csv` / `metric.txt`. QA re-implemented it and got the same flags.

- **Eligible bins:** 100 ms bins with no word on, starting at least 0.1 s after the most recent word offset.
- **Brain count:** word-driven neurons excluded.
- **θ = 200:**
  - above every non-event bg-only class_v2 bin in the Day 5 data (max 124 at 3.0 Hz);
  - below both pathologies (AL events 366–440; post-wind state 400–800).
  - **Margin:** non-event bins reach 175 at 3.5 Hz (al_rate runs), so the margin shrinks at higher background. The choice is empirical.
- **Flags:** `persist` (≥ 2 consecutive eligible bins > θ); `burst` (an isolated eligible bin > θ). AL events in eligible bins are counted separately. A burst bin need not be an AL event: ep018 has 2 word-free burst bins but only 1 AL event in them.
- **Score:** `P_wf` = fraction of eligible bins > θ.

| set | n | persist | burst | ok | old metric (any bin > 300) | mean P_wf |
|---|---|---|---|---|---|---|
| calib mixtures k 0–19 | 20 | 8 | 6 | 6 | 19 | 0.189 |
| calib bg-only k 200–209 | 10 | 0 | 4 (AL events) | 6 | 4 | 0.044 |
| shuffle bg-only | 10 | 0 | 0 | 10 | 0 | 0 |

- **What changes from the old metric:** the old metric flagged 19 of 20 mixtures, largely because words legitimately drive many brain neurons. The maximum per word bin is 246–1,842 by QA's definition (the earlier "1,000–2,150" depended on the definition). The new metric flags 14 of 20.
  - The 8 `persist` episodes are all wind episodes: ep002, 005, 009, 010, 011, 015, 016, 017.
  - The 6 `burst` episodes are ep001, 004, 018 and 019 (AL events) and ep003 and 008 (short post-wind tails cut by the eligibility window).
- **Gap in the metric:** it cannot see persistence that overlaps a later word. A neuron-specific flag would catch that: DNge027 spikes in any eligible bin. DNge027 has 0 spikes in 19 of 20 bg-only episodes (1 spike in calib ep206) and in every non-wind mixture.

## Loop-gain scan (revision; candidate fix B; diagnostic only, on a copy of W)
**Set-up:**
- **Edges scaled:** only the 101 W entries whose pre- and postsynaptic neurons both belong to LOOP5 (DNge027, GNG423, DNg21, DNg62, DNge044).
- **Episodes:** exact Day 5 specs (calib, class_v2 background, same Poisson seed, T 2.5 s).
- **Baseline:** g = 1.0 is the saved Day 5 run.
- **Window:** the wind word.
- **Persistence:** the last DNge027 spike minus wind offset, with the brain time course checked by hand.

`analyze_gain.txt`:

| ep (wind window) | g | persistence after offset | DNge027 Hz in word | LOOP5 mean Hz in word | MN spikes in word (vs g 1) | brain active in word (vs g 1) |
|---|---|---|---|---|---|---|
| ep009 (0.86–1.77) | 1.0 | to end (≥ 0.73 s) | 129 | 97 | 1,935 | 1,674 |
| | 0.5 | 0.01 s | 68 | 65 | 1,639 (−15%) | 1,599 (−4%) |
| | 0.3 | 0.01 s | 36 | 51 | 1,561 (−19%) | 1,588 (−5%) |
| | 0.1 | 0.01 s | 7 | 45 | 1,482 (−23%) | 1,449 (−13%) |
| ep002 (0.75–1.39) | 1.0 | 1.05 s (last spike 2.445 s) | 126 | 96 | 1,481 | 1,357 |
| | 0.5 | **0.25 s** (DNge027 16, 14, 9 spikes per 100 ms; brain 360–453) | 60 | 60 | 1,260 (−15%) | 1,254 (−8%) |
| | 0.3 | 0.05 s | 30 | 52 | 1,280 (−14%) | 1,252 (−8%) |
| | 0.1 | ≤ 0.15 s (1–2 sporadic DNge027 spikes) | 6 | 44 | 1,281 (−14%) | 1,197 (−12%) |
| ep016 (1.21–1.96) | 1.0 | to end (≥ 0.54 s) | 125 | 94 | 1,568 | 1,466 |
| | 0.5 | 0.02 s | 56 | 58 | 1,409 (−10%) | 1,407 (−4%) |
| | 0.3 | 0.09 s (brain 342 in the first post-offset bin, then 32) | 32 | 51 | 1,339 (−15%) | 1,317 (−10%) |
| | 0.1 | 0.17 s (brain 281, 217, then 44) | 3 | 44 | 1,357 (−13%) | 1,300 (−11%) |

**Brain bin 2.4–2.5 s:** 396–464 at g = 1.0 and 23–100 at every reduced gain (background level).

**Non-wind check at g = 0.3:** ep000 (looming + dry) and ep019 (sound + bitter + looming + humid) are **spike-for-spike identical** to the saved Day 5 runs (170,706 and 185,681 spikes). LOOP5 is silent without wind, so other words are unaffected.

**Highest gain that removes persistence:**
- **g = 0.5:** removes it in 2 of 3. It leaves a 0.25 s tail in ep002 (from 1.05 s).
- **g = 0.3:** the highest tested gain that removes it in 3 of 3 (≤ 0.1 s, similar to normal word decay).
- **g = 0.1:** no better than 0.3, and costs more of the word response.

**Recommendation, if Charlie chooses fix B: g = 0.3 on the 101 LOOP5 entries,** the same value as CLIQUE_GAIN.

**Risks:**
- **Small sample:** 3 episodes, 1 seed each. Persistence is stochastic, so a rare long tail at 0.3 cannot be excluded. A larger check, e.g. all 12 Day 5 wind specs, is advisable before adoption.
- **Changed wind response:**
  - wind-word MN spikes drop 14–19% and brain recruitment 5–10%;
  - DNge027 itself falls from about 125 Hz to about 30–36 Hz during the word;
  - there is no biological reference for either rate.
- **Patching one loop:** this patches one loop. The Week 3b warning applies, and other latent loops may surface. The lasting fix remains adaptation or depression (fix A).
- **Untested gains:** between 0.3 and 0.5 are untested.

## Hypotheses (C1)

| hypothesis | verdict | deciding evidence |
|---|---|---|
| **H1:** wind drives a near-bistable recurrent loop; background keeps it above threshold; Week 3b missed it by testing without background | **Partly holds** | See the points below. |
| **H2:** an olfactory-background effect, synchronised by the eLN–PN gap junctions and/or the APL; the rate window may be n = 1 noise | **Partly holds** | See the points below. |
| **H3:** the 300-neuron threshold is the wrong metric during words | **Holds** | Old metric flags 19 of 20 mixtures, the word-free metric 14 of 20, and the remaining flags are the two real pathologies. Bg-only flags are unchanged (4 calib, 0 shuffle). |

**H1, what holds:**
- The loop is real and ignites in every wind episode.
- It persists on calib, not on shuffle.
- With no background at all it dies within 0.3 s.

**H1, what does not hold:**
- The loop is a GNG DN loop (LOOP5), not an ascending or optic-lobe loop.
- It is metastable, not bistable: 0.44 s to ≥ 1.0 s with no input after offset.
- Background after offset does not consistently prolong it; background before offset does.
- Week 3b missed it because `wind` was not in its word list at all.

**H2, what holds:**
- ORN background appears necessary and sufficient (ORN-only 2 of 3 seeds; no-ORN 0 of 3; n = 3).

**H2, what does not hold:**
- Removing or halving the AL gap junctions does not abolish the activity. It turns it into continuous AL activity.
- APL has no effect (identical spike trains), and presyn has little.

**H2, still open:**
- The rate window is unresolved. Day 5 had 4 of 4 at 2.11–2.78 Hz; the 2 s runs had events at 2.5–3.5 Hz in 3 of 9 runs and none at 1.5–2.0 Hz.

## Root cause, ranked
1. **Post-wind persistence (the larger problem):** a metastable excitatory DN loop in the GNG (LOOP5), driven from JO-C/E mainly through SAD093 → GNG423 / DNg62.
   - It carries hundreds of reciprocal ACh synapses per type pair (up to 273 per cell pair; 20–45 mV per spike).
   - It persists because the LIF has no adaptation or depression (the same root cause as DNg33/IN19B040 and the looming state in Week 3b).
   - It drives about 120 optic-lobe neurons via GNG282 → MeVC, and about 40 MNs via DNs. It therefore corrupts the regression target (MN output) as well as its inputs, in about 8 of 20 episodes.
   - Background before offset prolongs it, but background does not cause it.
2. **AL bursts (smaller, rare):** a relaxation oscillation of the gap-coupled eLN–PN network under ORN background.
   - No MN effect (bg-only MN spikes 5–35).
   - **Possibly why they appear only now:** class_v2 sets bristle background to 0 Hz, and removing the VNC-driven ascending input may unmask them. This rests on single-seed perturbations of the earlier all-sensory runs.
3. **Metric:** the 300-neuron any-bin criterion counted legitimate word drive (H3). It is not a model cause.

## Candidate fixes (none implemented)

| fix | expected effect | risk |
|---|---|---|
| A. **Generic spike-frequency adaptation or short-term depression** in the LIF | Should end the DN loop and other latent loops; should blunt the AL volley. | Changes every response; needs re-validation of Weeks 1–3 and the DN panel; the largest effort. |
| B. **Targeted loop gain:** LOOP5 internal edges (101 entries) × 0.3 | Scan: persistence ≤ 0.1 s in 3 of 3 wind episodes; non-wind episodes spike-identical. | Wind-word MN spikes −14–19%; DNge027 rate in the word about −75%; n = 3 with 1 seed; patches one loop. |
| C. **Exclude post-wind bins from the regression,** or flag episodes with the Q4 metric / DNge027 activity | No model change. | Discards about 40% of wind episodes' tails; MN output during wind remains loop-driven. |
| D. **Class-specific head background:** drop ORN background, or lower it (≤ 2 Hz: 0 of 6 runs) | Removes AL events (no-ORN class_v2: 0 of 3). | Below real ORN spontaneous rates (about 8–10 Hz, unverified); the region above the event band is not established. |
| E. **Re-calibrate the eLN–PN junctions** (c0) together with eLN threshold or adaptation | Could give tonic, non-synchronous AL rest activity. | Junction × 0.5 or × 0 gave runaway tonic eLNs (about 200 Hz); needs a joint re-fit of the Week 2 AL calibration. |
| F. **Leave the AL events as biological** | No work. | Not supported by the literature known to me (unverified); about 400 brain neurons in one volley would enter the regression inputs. |

## Assumptions and what is unverified
- **AL-event threshold** (≥ 100 AL neurons in 10 ms) and **metric θ = 200** were chosen by hand. θ has a smaller margin at 3.5 Hz.
- **Literature (Q3)** is from memory and was not re-read in this session; the Suver et al. 2019 reference details are uncertain.
- **The wind state's termination** is not identified. Which part of the pre-offset background lengthens the state is not identified either (JO-C/E self-background is one candidate).
- **Ascending suppression of AL events** in all-sensory runs, and the claim that class_v2 unmasks them, rest on single-seed re-scoring.
- **Seed coverage:**
  - wind matrix: 1 spec × 1 seed (A) + 1 spec × 2 seeds (B);
  - AL rate table: 3 seeds × 2 s per rate;
  - gain scan: 3 wind episodes, 1 seed each;
  - non-wind check: 2 episodes at g = 0.3 only.
