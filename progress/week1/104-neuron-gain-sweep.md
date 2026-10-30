# Final debugging plan after C1 and C2

## What C1 and C2 now establish

C1 and C2 substantially change the diagnosis. You no longer have evidence for a single “bad loop” or one dominant neuron that can be removed to cure the attractor. Instead, your results support a **distributed recurrent excitatory network** in which many overlapping cycles can substitute for one another.

C1 is especially informative because the strong recurrent component cannot be broken by removing a tiny handful of neurons: your cycle-breaking set contains **104 neurons**, including every `lLN1_bc` plus several other cholinergic local-neuron classes, some projection neurons, and a smaller set whose transmitter identity is unclear. In graph terms, that is what one expects when feedback is redundant: cutting one path leaves many alternate paths available.

C2 then supplies the causal result that C1 alone could not. When you allow the attractor to form normally and only then silence the 104-neuron cycle-breaking set, the established state collapses within roughly 10 ms. A size-matched control does not. That means the C1 set is not merely correlated with high activity: **activity flowing through that recurrent component is necessary for maintenance of the attractor in your simulation**.

But this still does **not** show that one of those 104 neurons is the biological “root cause.” A distributed network can be necessary as a group even when no member is individually necessary. Your earlier B2 result already points in that direction: removing excitatory→excitatory LN edges reduced activity but left the attractor alive, implying that alternate routes can carry the recurrence.

The most useful working hypothesis is therefore:

> **The attractor is maintained by excessive effective gain distributed across an antennal-lobe recurrent network, rather than by a single short pathological motif.**

That hypothesis is also biologically plausible in the sense that Drosophila antennal-lobe excitatory local neurons participate in recurrent interactions with projection neurons and with other local neurons, while inhibitory local neurons provide counterbalancing gain control. Experimental work has found reciprocal excitation involving excitatory LNs and PNs, and excitatory LNs can also recruit inhibitory LNs. citeturn391274search8turn391274search9

The important qualification is that the Shiu model is intentionally very simple: it represents neurons as leaky integrate-and-fire units whose synaptic strengths come from connectomic weights and predicted neurotransmitter identity, and it omits several biological mechanisms, including gap junctions, non-spiking neurons, internal state and neuropeptides. citeturn391274search0

## First priority: verify the simulator before changing the biology

There is one important correction to make before doing more lesion experiments.

Earlier, the hypothesis was raised that synaptic input arriving during the refractory period might **accumulate and be delivered after refractoriness**, helping saturated neurons immediately fire again. For the published Shiu Brian2 implementation, that interpretation is incorrect.

The reference code declares both membrane voltage `v` and synaptic variable `g` with `(unless refractory)`, and synaptic input is delivered with `on_pre='g += w'`. citeturn671338view1 Brian2 explicitly states that a variable marked `(unless refractory)` is read-only while the neuron is refractory, so incoming synaptic writes have no effect during that interval. citeturn671338view2

Therefore, in the **reference Shiu implementation**, refractory-period synaptic events to `g` should be discarded rather than accumulated.

That gives you an immediate diagnostic experiment before proceeding further:

**Run the same C2 attractor using the original Brian2 implementation, or make a tiny unit test in your implementation that sends synaptic events to a neuron during its 2.2-ms refractory period.** The event should not survive into the post-refractory state if you are reproducing Shiu semantics. The published model uses a 2.2-ms refractory period, a 1.8-ms synaptic delay and a 5-ms synaptic time constant, with a single synaptic-weight scaling parameter. citeturn671338view1turn391274search0

If your simulator instead accumulates those events, fix that **before interpreting any more circuit lesions**. A refractory mismatch is exactly the kind of error that can disproportionately amplify neurons already firing near the ceiling.

I would make this a new **C0 sanity check** and require it to pass before continuing.

## The next experiment should be a sign audit, not another broad lesion

Your C1 result makes the unclear-transmitter neurons much more important than they were previously.

MaleCNS does not treat “unclear” as equivalent to acetylcholine. In the current MaleCNS release, neurons are marked `unclear` when neurotransmitter confidence is insufficient; the dataset's `consensusNt` field is specifically recommended for most analyses. Neurons with fewer than 50 presynapses or with prediction confidence below 0.5 are assigned `unclear`. citeturn391274search3

The official MaleCNS download also provides a 42-MB neuron-level neurotransmitter table and, separately, a 2.7-GB file containing transmitter probabilities at individual presynaptic sites. citeturn391274search4 Synapse-level transmitter classification is useful but imperfect: the underlying classifier achieved about 87% accuracy at individual synapses and higher aggregate accuracy at neuron/type level, and antennal-lobe local neurons are specifically among classes where prediction errors have been observed. citeturn671338view4

So the next experiment should be **C3: sign sensitivity of the unclear members of the causal core**.

Run the exact C2 maintenance experiment under three assignments for only the approximately 12 unclear neurons inside the 104-neuron cycle-breaking set:

| C3 condition | Treatment of the unclear core neurons | Interpretation |
|---|---|---|
| C3a | current model assignment | reference |
| C3b | outgoing weights = 0 | asks whether their assumed excitation is necessary |
| C3c | outgoing weights inhibitory | extreme sign-sensitivity test |

Do not change the confirmed cholinergic neurons yet.

Measure the same quantities you already trust: collapse/non-collapse, tail-active count, mean rate of the 104-neuron set, `lLN1_bc` rate, PN rate, and normal stimulus response.

The key question is not whether activity decreases a little. It is whether **the attractor crosses from self-sustaining to decaying**.

If C3b or C3c abolishes persistence, stop there and resolve those transmitter identities before touching gains. Using “unclear → excitatory” would then be the likeliest modeling error.

This is biologically important because acetylcholine is generally excitatory in the fly CNS, whereas GABA is inhibitory and glutamate is often inhibitory in the antennal lobe through glutamate-gated chloride channels. citeturn705190search0turn705190search1

## Then measure the actual recurrent-gain threshold

If correcting or neutralizing unclear transmitters does **not** remove the attractor, the next experiment should no longer ask “which edge class do I delete?” It should ask:

**How much recurrent excitation can this network tolerate before it becomes self-sustaining?**

Call this **C4: core gain bifurcation**.

Take only the excitatory outgoing connections whose presynaptic neurons belong to the 104-neuron maintenance set and multiply those weights by a factor \( \alpha_\text{core} \). Leave all other network weights unchanged.

Run:

\[
\alpha_\text{core}
=
1.0,\ 0.9,\ 0.8,\ 0.7,\ 0.6,\ 0.5
\]

If the transition occurs between two values, refine that interval, for example 0.75, 0.725, 0.70.

This is fundamentally different from your previous LN→LN or LN→PN experiments. Those split the network by anatomical edge class. C4 asks whether the **whole causal recurrent component has crossed a dynamical gain threshold**.

The Shiu framework has one free per-synapse gain parameter, and each downstream effect scales with the signed synapse count. citeturn705190search0turn671338view1 That makes gain sensitivity particularly important when moving the same simple dynamical rule onto a different connectome.

For each value of \( \alpha_\text{core} \), record two families of measurements simultaneously:

**Stability:** silent-tail population, decay time, maximum core firing rate, and whether activity remains after 300–500 ms without stimulus.

**Function:** antennal-lobe response during the stimulus, stimulus specificity/overlap, and your downstream readouts such as MN9.

You are looking for a sharp transition such as:

\[
\alpha=0.80:\ \text{persistent}
\]

\[
\alpha=0.75:\ \text{decays normally}
\]

If such a transition exists, that is strong evidence that the pathology is **excess recurrent gain**, rather than a single incorrect edge.

## Find out which parts of the 104-neuron set supply that gain

Once you know the critical \( \alpha_\text{core} \), do **C5: gain decomposition**.

Do not return to binary cuts. Instead, partition the 104 neurons into biologically meaningful groups and reduce each group's outgoing excitation only enough to equal the amount of gain reduction identified in C4.

The first groups I would test are:

1. all 30 `lLN1_bc`;
2. the other confirmed cholinergic local interneurons;
3. cholinergic projection neurons inside the core;
4. the unclear-sign neurons;
5. combinations of the first three.

Your C2 result predicts that many individual groups may fail to abolish the attractor because the web has redundant feedback paths. That is not a failed experiment; it is exactly what a distributed attractor should do.

The useful result would look something like this:

- scaling `lLN1_bc` alone reduces rate but persistence survives;
- scaling other cholinergic LNs alone reduces rate but persistence survives;
- scaling PNs alone reduces spread but persistence survives;
- scaling `lLN1_bc + other cholinergic LNs` crosses the transition.

That would tell you where most of the **effective recurrent gain** resides without pretending that one neuron class is uniquely causal.

Also keep the maintenance timing at 1.3 s for these experiments. Because the perturbation occurs after the attractor is already established, you are measuring **maintenance**, rather than merely weakening odor input or ignition.

Physiologically, this division is reasonable: antennal-lobe excitatory LNs have been shown to interact reciprocally with PNs and to spread excitation between glomerular channels, while also helping recruit inhibitory gain control. citeturn391274search8turn391274search9

## The likely permanent fix should follow a hierarchy

At this point I would use a strict order of fixes rather than tuning until the simulation looks good.

**First, fix implementation mismatches.** Refractory handling must reproduce Brian2 semantics. The published equations clamp both `v` and `g` during refractoriness, and Brian2 makes such variables insensitive to incoming synaptic modification while refractory. citeturn671338view1turn671338view2

**Second, fix transmitter assignments.** Do not give an `unclear` MaleCNS neuron an excitatory sign merely because the simulator requires a sign. MaleCNS deliberately marks low-confidence cases unclear and provides both neuron-level and presynapse-level transmitter data for resolving them. citeturn391274search3turn391274search4

**Third, recalibrate gain only if the attractor survives those corrections.** The Shiu model deliberately reduces physiology to a signed synaptic count multiplied by a single free synaptic-weight parameter. citeturn705190search0 A gain that worked for the original FlyWire model is therefore not guaranteed to be dynamically appropriate for every densely recurrent subnetwork in MaleCNS; this is an inference from the model structure, not a statement made by Shiu et al. citeturn391274search0turn391274search3

If C4 demonstrates a clean critical gain, I would prefer a **principled regional or cell-class calibration** over deleting individual edges. For example, if all confirmed cholinergic antennal-lobe recurrent outputs become stable at \(0.78\times\) while odor responses remain largely intact, that is a much more defensible fix than deleting 104 neurons.

There is another reason not to overinterpret individual LN→PN chemical edges. Physiological studies show that some excitatory LN–PN coupling in the antennal lobe involves electrical coupling through gap junctions, and different experiments have found mixed electrical and cholinergic contributions. citeturn391274search8turn391274search9 The Shiu model explicitly does not model gap junctions. citeturn391274search0 Thus a static chemical-connectome LIF model cannot be expected to reproduce every antennal-lobe feedback mechanism quantitatively.

## A concrete final sequence

I would now stop doing arbitrary edge-class lesions and run this sequence:

1. **C0 — refractory semantics test.** Confirm that synaptic input to `g` during refractoriness is discarded exactly as in Brian2. If not, correct the simulator and repeat C1/C2. citeturn671338view1turn671338view2
2. **C3 — unclear-sign audit.** Test the approximately 12 unclear neurons within the 104-neuron causal set as current/default, zero-output and inhibitory. MaleCNS explicitly treats low-confidence transmitter assignments as unclear rather than cholinergic. citeturn391274search3
3. **C4 — core gain sweep.** Scale only confirmed excitatory outputs from the 104-neuron maintenance set from 1.0 downward and locate the persistence/decay transition.
4. **C5 — decompose that gain.** Repeat the critical reduction separately for `lLN1_bc`, other cholinergic LNs, PNs and combinations, using post-1.3-s maintenance perturbations.
5. **C6 — rescue.** Start from the first stable condition and restore each component toward 1.0. A genuine causal gain mechanism should bring persistence back when enough recurrent excitation is restored.
6. **C7 — validation.** Apply the chosen correction without lesions during the run and test hot, vinegar, the other odor inputs, no-stimulus activity, stimulus overlap, tail decay and downstream behavioral readouts across multiple seeds.

The stopping rule should be strict: the winning model should not merely suppress the 349-neuron antennal-lobe tail. It should allow a substantial odor-evoked response during stimulation, return toward silence afterward, preserve stimulus discrimination, and avoid spontaneous activity without input.

Your current evidence is already enough to make one conclusion fairly strong: **the 104-neuron core is a causal maintenance backbone, but probably not a 104-neuron “disease module” that should simply be removed.** The next question is whether that backbone is unstable because of an implementation mismatch, incorrect signs, or excessive gain. C0 → C3 → C4 is the shortest path to distinguishing those possibilities.