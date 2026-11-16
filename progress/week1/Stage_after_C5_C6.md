# Deep research assessment of C5/C6 and the next debugging stage

## What C5 and C6 have actually established

C5 and C6 are much more decisive than the earlier broad lesions. They show that the persistent antennal-lobe state is supported by **redundant recurrent excitatory subnetworks**, not by one fragile motif. In C5, no single subgroup lesion removes the attractor, but removing all 80 cholinergic local neurons causes an approximately 85% reduction, and removing all 104 cycle-breaking neurons eliminates it. In C6, reintroducing specific subsets after silencing all 104 restores persistence to different degrees. That is strong evidence that the recurrent maintenance mechanism resides primarily in the local-neuron population rather than in projection neurons.

The strongest causal candidate is clearly `lLN1_bc`. Restoring only those 29 neurons is sufficient to sustain roughly 1,500 brain neurons at full strength and nearly 900 even at half strength. The other 51 cholinergic local neurons form another self-sustaining module, but one much closer to its critical threshold because half strength fails while full strength succeeds. The unclear-transmitter group produces a much smaller persistent state. The 11 projection neurons, by contrast, cannot sustain activity when the local-neuron modules remain silenced.

I would slightly refine one interpretation, though: C6 supports **at least two strong self-sufficient recurrent modules plus one smaller persistent module**, rather than proving that there are exactly “three separate loops.” A rescue experiment establishes sufficiency under that intervention, but graphically distinct loops would require showing that each subset contains an internally strongly connected recurrent component and that its persistence does not depend on edges through neurons outside the restored subset. Likewise, the fact that projection-neuron rescue produces zero firing shows that those 11 PNs are not self-sufficient in this setting; it does not prove that PNs never participate in recurrent paths when the local network is intact.

This interpretation fits the MaleCNS wiring. The Male CNS Cell Type Explorer reports 30 `lLN1_bc` neurons, predicts acetylcholine with 75.1% confidence, and shows that essentially all of their input and output is in the antennal lobe. Most importantly, `lLN1_bc` receives an enormous amount of input from its own class: the explorer reports roughly 2,895 connections per cell from `lLN1_bc` back onto `lLN1_bc`, representing about 34.7% of the type's listed input. That is exactly the kind of architecture that can produce a high-gain recurrent module in a simplified integrate-and-fire model. citeturn603658view0

The same pattern exists in other cholinergic local-neuron types. `lLN2X12`, for example, receives substantial recurrent excitation from `lLN1_bc`, from itself, and from other cholinergic LN types, while also contacting inhibitory populations. citeturn603658view1 This makes your “dense overlapping excitatory web” interpretation much more plausible than the earlier idea of one short pathological cycle.

## The most important thing to do next is not another lesion

Before tuning gains further, I would insert a **model-audit stage**. There is a potentially important discrepancy between the refractory-period behavior you previously suspected and the actual Shiu/Brian2 implementation.

The public Shiu code defines both membrane voltage and the synaptic drive variable `g` with `(unless refractory)`, resets `g` to zero on every spike, and delivers synaptic events using `g += w`. citeturn775340view1turn775340view2 Brian2's documentation states that variables marked `(unless refractory)` are held constant during the refractory period and are read-only to incoming synaptic events while the neuron is refractory. citeturn603658view3

That means the hypothesis that “inputs accumulate during refractoriness without decay and then arrive all at once afterward” is **not how the reference Brian2 implementation behaves**. If your MaleCNS implementation does accumulate such inputs, then it differs materially from Shiu's reference model, and that difference could strongly exaggerate recurrent saturation.

This is important enough that I would make it the immediate next experiment.

Run a one-neuron unit test with exactly the same simulation engine you used for C5/C6. Force it to spike, then send several presynaptic events during its 2.2-ms refractory period. Record `g` every timestep. Under the reference behavior, those events should not accumulate into a post-refractory burst. Then repeat the same test directly in Brian2 using Shiu's equations. The two traces should match essentially exactly.

If they do not match, **stop the lesion program and fix the simulator first**. A recurrence problem caused by altered refractory handling should not be “corrected” by weakening biological connections.

## The next biological/model-data check should be the transmitter-sign audit

If the simulator passes the refractory test, the next stage should be the transmitter audit you had postponed.

C6 makes this more urgent than before because the 13 unclear-transmitter neurons are now causally relevant: restoring them alone produces about 200 persistent brain neurons, and adding them to `lLN1_bc` or the other local-neuron group substantially increases the recovered state. Their signs therefore affect the location of the attractor boundary.

The official MaleCNS release provides both a small neuron-level neurotransmitter table and, crucially, a **2.7-GB per-presynapse neurotransmitter-probability table**. citeturn775340view0 The underlying transmitter-classification work reports about 87% accuracy at individual synapses and about 94% at the neuron level, so uncertain neurons should not simply be treated as definitely excitatory. citeturn800053search6turn800053search7

I would therefore define the next experiment as **C7: sign-resolved core**.

For the 13 unclear neurons in the 104-neuron set, extract their per-presynapse probabilities only; you do not need to inspect the entire dataset manually. For each neuron, calculate the aggregate probability mass for ACh, GABA, glutamate, and modulatory transmitters over its outgoing synapses. Then rerun three versions of the core: your current default-excitatory assignment, zero-output assignment for only genuinely unresolved neurons, and the best-supported per-neuron or per-synapse sign assignment.

The key output is not simply “does persistence disappear?” It is how much the critical recurrent gain changes. If correcting those 13 neurons removes only the ~200-neuron residual but leaves the `lLN1_bc` attractor intact, then the main problem is still the cholinergic LN web. If it substantially shifts the threshold of `lLN1_bc` or the 51-neuron module, then sign uncertainty has been amplifying the main attractor.

## After that, measure the attractor threshold rather than deleting neurons

Assuming the simulator and signs check out, I would move to **C8: module-specific critical-gain curves**.

You no longer need broad αLL or αLP sweeps. C6 has already narrowed the recurrent mechanism enough that the informative variables are now the gains *inside the identified recurrent modules*.

Use three gain parameters:

\[
\gamma_{1} = \text{gain on recurrent edges among the 29 }lLN1\_bc
\]

\[
\gamma_{2} = \text{gain on recurrent edges among the other 51 cholinergic LNs}
\]

\[
\gamma_{12} = \text{gain on cross-connections between those two groups}
\]

Keep their outputs to PNs and inhibitory neurons unchanged initially.

For each parameter, start at 1.0 and step downward more finely than before—something like 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, then bracket the transition with 0.05 or 0.025 increments. Use the same hot and vinegar stimuli and several seeds once you approach the boundary.

The question is now:

> **At what internal recurrent gain does the module stop being self-sustaining after the odor disappears, while still participating normally during odor stimulation?**

That is much more informative than removing entire cell classes.

Your C6 data already indicate that `lLN1_bc` is well above threshold because it survives at β = 0.5, while the other 51-neuron module lies between β = 0.5 and β = 1.0. Therefore, the first curve I would measure is `lLN1_bc → lLN1_bc`, followed by its cross-excitation with the other cholinergic LNs.

Do not initially weaken `lLN1_bc → PN`, `LN → GABA`, or sensory input. Those pathways are likely part of normal odor processing. Classical Drosophila physiology shows that excitatory antennal-lobe LNs participate in broad lateral excitation and reciprocal interactions with PNs and inhibitory LNs, meaning that deleting all of their outward drive can degrade legitimate olfactory responses. citeturn800053search8turn800053search9turn800053search10

## The decisive experiment is a two-dimensional gain map

Once you know approximate thresholds, run a small factorial map over `lLN1_bc` self-recurrence and cross-LN recurrence.

For example:

| γ₁: lLN1_bc internal | γ₁₂: lLN1_bc ↔ other cholinergic LNs | What you learn |
|---|---|---|
| 1.0 | 1.0 | Current pathological baseline |
| 0.75 | 1.0 | Whether lLN1_bc self-gain alone is enough to move below threshold |
| 0.5 | 1.0 | Stronger internal correction |
| 1.0 | 0.75 | Whether cross-module coupling is the main amplifier |
| 1.0 | 0.5 | Strong cross-module reduction |
| 0.75 | 0.75 | Whether modest distributed correction works |
| 0.5 | 0.75 | Test below the expected lLN1_bc threshold |
| 0.75 | 0.5 | Alternative balanced correction |

At each point measure four quantities: silent-tail population size, firing rate of the 29 `lLN1_bc` cells, antennal-lobe population during the odor, and downstream behavioral readouts such as MN9 where relevant.

A good correction is not the point where firing is smallest. It is the point just below the **bifurcation** where the persistent state disappears while the evoked response remains close to baseline.

Your C6 result strongly suggests that such a threshold exists because one module already changes qualitatively between β = 0.5 and β = 1.0. That is exactly what one expects near a recurrent-gain transition.

## One biological caveat may explain why the simplified model is unusually prone to runaway excitation

The Shiu network is intentionally simple: it uses one generic leaky integrate-and-fire model for all neurons and converts connectomic synapse counts into fixed signed weights. citeturn800053search2turn900779search0 It does not reproduce every biophysical mechanism that shapes antennal-lobe recurrence.

This matters particularly for excitatory local neurons. Experimental work has reported reciprocal excitation among excitatory LNs and PNs, but also substantial electrical coupling and specialized mixed electrical/chemical interactions in the antennal lobe. citeturn800053search8turn800053search10 Electrical coupling can redistribute and shunt currents rather than acting like a simple positive chemical weight; reviews of these circuits have explicitly noted that such properties may help prevent runaway excitation. citeturn800053search10

Therefore, if you find that only a large reduction of biologically predicted cholinergic recurrent weights fixes the attractor, the conclusion should not be “the connectome is wrong.” A more defensible conclusion would be that **the fixed-sign, fixed-gain LIF approximation is over-amplifying a densely recurrent antennal-lobe circuit because it omits stabilizing physiology**.

That distinction matters because your final correction may need to be described as a **model calibration**, not a biological lesion.

## Final debugging and fixing sequence

The shortest path from where you are now is:

1. **C7a: refractory unit test.** Verify that your implementation matches Shiu/Brian2 exactly during refractoriness. If inputs are accumulating while `g` is refractory, correct that before doing anything else. citeturn775340view1turn603658view3
2. **C7b: transmitter sign audit.** Resolve the 13 unclear members of the 104-neuron set using MaleCNS per-presynapse transmitter probabilities. citeturn775340view0
3. **C8: internal recurrent-gain sweep.** Measure the critical gain of `lLN1_bc → lLN1_bc`, the 51-neuron local-LN module, and the cross-connections between them. Do **not** start by weakening all LN outputs.
4. **C9: ignition-versus-maintenance at the winning gain.** Present the odor normally, then switch only the recurrent internal gain after the odor ends. If the established state collapses while the stimulus-period response was normal, you have specifically fixed maintenance rather than sensory transmission.
5. **C10: add-back validation.** Start from the corrected stable model and restore the suspect recurrent edges toward 1.0. Persistence should reappear near the same threshold seen in the downward sweep. This gives you a reversible causal result rather than a one-direction lesion.
6. **C11: full-network validation.** Only after the above passes, run the complete brain-plus-VNC network over multiple seeds and all sensory inputs. The correction should abolish the large olfactory persistent state, preserve odor responses, preserve relevant motor outputs, leave no-stimulus trials silent, and avoid creating a new attractor elsewhere.

The central conclusion from C5/C6 is therefore not “find one more neuron to cut.” You have already reached the point where neuron deletion has done its job diagnostically. The remaining question is **why the gain of this biologically dense recurrent LN network crosses the self-sustaining threshold in your model**. The fastest route to that answer is now: verify simulator semantics, resolve uncertain signs, then measure the recurrent gain threshold specifically inside `lLN1_bc` and its coupled cholinergic-LN web.