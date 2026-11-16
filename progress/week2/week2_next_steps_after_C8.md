# Week 2: next steps after C8

*2 Nov 2026. Based on the Neuro folder (plan.md, plan_C7_C11.md, Stage_after_C5_C6.md, the C1–C8 logs and the sim/ code) and a deep-research run with sources checked by 3-vote verification (sources at the end).*

## The short version

Stop trying to fix the attractor by weakening connections. C8's stopping rule fired for a real reason. Week 2 can still be finished this week if you do three things:

1. Run the same odor protocol on FlyWire v783 to find out whether this is a MaleCNS problem or a model problem. It takes minutes.
2. Give the model one piece of stabilizing physiology that the Shiu equations leave out: spike-frequency adaptation or short-term depression. Give it a fixed time budget of about a day and a half.
3. Build the 16-word readout on the stimulus window only, whatever happens in step 2, and report persistence as a documented model limitation.

## What C1–C8 now tell us

- **The attractor is not one antennal-lobe loop.** C8 removed every edge among the 80 cholinergic LNs, and persistence only dropped from about 3,340 to 3,035 neurons. C7c showed why: 49–78% of each LN group's excitatory input comes from outside the group.
- **It is not only an olfactory problem either.** In ckpt2b, sugar, water and salt leave no tail. But sound (2,094), looming (2,108), wind, hot, cold, dry and humid all persist. Thermo- and hygrosensory neurons project into antennal-lobe glomeruli, which explains those stimuli. Sound and looming do not, so there is at least one more self-sustaining loop outside the antennal lobe. A local edge correction cannot fix that.
- **The rates are not physiological.** lLN1_bc sits near 300 Hz and other LNs near 235 Hz after the odor ends. With a 2.2 ms refractory period, the model saturates around 450 Hz, and nothing in it slows a neuron that has been firing for a long time.
- **Odor drives MN9 at about 50 Hz in every C8 condition.** MN9 is a proboscis motor neuron, and in the fly vinegar shouldn't drive it like that. This is a second sign that activity spreads too freely.

**Biology says the model is missing the brakes, not that the weights are wrong** (confidence: high):
- In vivo, excitatory cholinergic LNs exist, but they excite PNs mainly through **ShakB gap junctions**, not cholinergic chemical synapses. Nicotinic blockers didn't reduce the eLN→PN excitation, and the shakB2 mutation abolished it (Yaksi & Wilson 2010; Huang 2010; Olsen et al. 2007; Shang et al. 2007). The model turns all of it into chemical excitation, which builds a PN↔LN loop the real fly mostly doesn't have. That loop was left intact in C8.
- The antennal lobe is also held down by **presynaptic GABA-B inhibition** on ORN terminals, which scales with total input (Olsen & Wilson 2008), and by **short-term depression**. The Shiu model has none of these.
- **There is a precedent.** A FlyCircuit whole-brain LIF model ran away into activity that never stopped at every I/E ratio from 0.1 to 100. More inhibition only delayed it, and short-term depression stabilized it (Huang et al. 2018). Note that this paper's τD threshold numbers did *not* survive verification, so cite only the qualitative result.
- **Shiu et al. never tested this regime.** The paper used 1 s gustatory and Johnston's-organ drive and never looked at activity after the stimulus ends. Its silence is not evidence that the model is stable after offset.

## Plan for the rest of Week 2

### Day 1: two cheap diagnostic runs (no new model features)

**D1. FlyWire control (this is the old C7d, still not run).** Use the Shiu repo on v783: vinegar ORNs (or the same "hot" set) at 150 Hz for 1 s, then 0.6 s of silence, seeds 0 and 1. Also run sound and looming, and a no-stimulus trial. It takes about 2 minutes per trial.
- *FlyWire stays clean:* the problem is MaleCNS-specific (synapse counts or transmitter calls). FlyWire becomes the primary model for the Week 2 deliverable, and the comparison between the two is itself a result.
- *FlyWire also persists:* the problem is in the model class, and Day 2 is the fix.

**D2. One targeted edge test: cholinergic LN→PN and PN→LN chemical edges at 0.5 and 0.** This is the loop that biology says should mostly be electrical. Run hot and vinegar on 2 seeds (8 runs per level). Keep it to this one test, not another sweep series. If 0 abolishes persistence, the result has a clean biological story: "gap-junction coupling modeled as chemical excitation". If it doesn't, it rules that story out.

### Days 2–3: add one stabilizer, with a hard time limit

Both options are cheap in the NumPy simulator (one state variable per neuron, not per synapse). Try adaptation first, because it also fixes the 300 Hz rates.

**Option A: spike-frequency adaptation** (current-based version of Kao & Lo 2020)
```
dv/dt = (v0 - v + g - a) / t_mbr
da/dt = -a / tau_a              tau_a = 250 ms   (cited range 200-300 ms)
on spike: a += delta_a          sweep delta_a ∈ {0.5, 1, 2, 4} mV
```
Apply it to all neurons first, since the second loop lies outside the antennal lobe. If that hurts the evoked responses, apply it only to cholinergic LNs and PNs.

**Option B: short-term depression on excitatory outputs** (Kao & Lo 2020; Liu et al. 2021)
```
per presynaptic neuron j:  dD_j/dt = (1 - D_j) / tau_D,   on spike: D_j *= (1 - U)
effective outgoing weight = W[:, j] * D_j
U = 0.5, tau_D = 450 ms   (cited range U 0.3-0.5, tau_D 160-450 ms)
```

**Accept a setting only if all four checks pass:**
1. Activity decays to fewer than 20 neurons within about 300 ms after offset, for hot, vinegar, sound and looming.
2. FlyWire sugar→MN9 still follows the Shiu dose–response curve. Use your Week 1 Brian2 reference (`results/flywire_sugar_brian2_fresh.npy`), and aim for roughly ±20% at each rate.
3. Odor-period antennal-lobe population and PN rates stay within about 15% of baseline.
4. No-stimulus runs stay silent.

**Time limit:** if no setting passes by the end of Day 3, freeze the model as it is and go to Days 4–5. Don't open a C9-style series.

### Days 4–5: the Week 2 deliverable (do this whatever happens above)

- **Stimuli:** the 16 words (plan.md, Week 2 step 3), 500 ms of drive, 10 Poisson seeds each.
- **Readout window:** spike counts during the stimulus only (0–500 ms, plus 100 ms bins as a second feature set). This is defensible because Shiu's own validation used only the drive window.
- **Decoder:** multinomial logistic regression with cross-validation across seeds. Report accuracy and the confusion matrix.
- **Shuffled-connectome control:** shuffle the network while keeping each neuron's in- and out-degree and its sign, then run the same stimuli, readout and cross-validation. A sensible build is a version of `sim/lesion.py` that writes a shuffled copy of W.
- **Attractor check:** also decode from a post-offset window (for example 1.0–1.6 s). If accuracy there is near chance and the persistent pattern is the same across stimuli, the attractor carries no word identity and is cleanly kept out of the stimulus-window readout. Put that number in the report.
- **Week 2 step 2 is already done:** ckpt2b has the MaleCNS sugar/bitter→MN9 numbers (sugar 78 Hz, bitter 0, no tail for gustatory stimuli). Put them next to the FlyWire numbers.

### Days 6–7: Connectome Graph Explorer artifact (the Week 2 final product, required before Week 3)

The full specification is in `docs/plan.md`, Week 2 step 6. In short: an interactive Artifact that draws the NumPy weight matrix as a graph. Each word gets a subgraph of up to about 3k neurons, layered from sensory input to DNs/MN9, and there is a cell-type overview. The NumPy spike activity is overlaid on a time slider. Toggles switch between base and gap-junction calibrated, MaleCNS and FlyWire, and real and shuffled. It also has panels for the attractor core and the readout confusion matrices. Day 6: write `scripts/export_graph.py` and export the JSON views. Day 7: build and publish the page. The write-up below goes into its "how to read this" panel and the README.

Write-up content:

Write a single week2 report in the style of `docs/week1_report.html`:
- the FlyWire versus MaleCNS persistence comparison;
- the outcome of the stabilizer test, positive or negative;
- 16-word decoding versus the shuffled control;
- **the limitation, stated plainly:** with fixed-sign current synapses and no gap junctions, adaptation or short-term depression, MaleCNS has a self-sustained state after odor offset, with saturating LN rates. Tie it to the biology cited above. Describe any fix as a *model calibration*, not a correction to the connectome.

## Not worth doing now

- More γ or β sweeps, or lesions of more cell groups (C8's stopping rule).
- The 2.7 GB transmitter download for more neurons (C7b moved the residual but not the main state).
- Re-checking the refractory handling (settled in C0/C7a).
- Explaining it by sex dimorphism: antennal-lobe dimorphism is limited to pheromone ORN counts and one PN type (Berg et al., male CNS paper).

## Open questions this leaves

- Are MaleCNS synapse counts per connection systematically higher than FlyWire's for matched cell types, so that one global 0.581 factor under-corrects dense recurrent excitation? No verified source covered this. A type-matched edge comparison would answer it (half a day, optional).
- Where is the loop behind the sound and looming tails? C1's SCC tools can find it if Day 2 doesn't remove it anyway.

## Sources

- Shiu et al., whole-brain LIF preprint: https://www.biorxiv.org/content/10.1101/2023.05.02.539144.full.pdf
- Shiu repo: https://github.com/philshiu/Drosophila_brain_model ; Eon fork: https://github.com/eonsystemspbc/fly-brain
- Huang et al. 2018, FlyCircuit runaway activity and STD: https://www.frontiersin.org/articles/10.3389/fninf.2018.00099/full
- Yaksi & Wilson 2010, eLN→PN coupling through gap junctions: https://wilson.hms.harvard.edu/sites/projects.iq.harvard.edu/files/wilson-lab/files/yaksiwilson2010.pdf
- Huang et al. 2010: https://pubmed.ncbi.nlm.nih.gov/20869598/ ; Olsen, Bhandawat & Wilson 2007: https://pmc.ncbi.nlm.nih.gov/articles/PMC2048819 ; Shang et al. 2007: https://einstein.elsevierpure.com/en/publications/excitatory-local-circuits-and-their-implications-for-olfactory-pr/
- Olsen & Wilson 2008, presynaptic GABA-B inhibition: https://wilson.hms.harvard.edu/sites/projects.iq.harvard.edu/files/wilson-lab/files/olsenwilson2008.pdf
- Kao & Lo 2020, STD and AHP parameters: https://webmail.life.nthu.edu.tw/~lablcc/publications/Lo-C-2020a.pdf
- Liu et al. 2021, Tsodyks–Markram STP in the antennal lobe: https://www.frontiersin.org/articles/10.3389/fncom.2021.730431/full
- Berg et al., male CNS (dimorphism): https://pubmed.ncbi.nlm.nih.gov/41279223/
