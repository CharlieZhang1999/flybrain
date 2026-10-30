# Final debugging plan for the antennal-lobe recurrent core

## What B1 and B2 have established

Your B1 result is much more informative than Stage A. With projection-neuron edges cut, a **349-neuron antennal-lobe population still sustains itself during the silent tail**, and the most extreme members are cholinergic local interneurons such as `lLN1_bc`, which are running at or near the model’s refractory-limited maximum. That means the large brain-wide persistent state is not merely being driven by projection neurons feeding back from downstream brain regions; there is already a **locally self-sustaining antennal-lobe core** in your simulation. The fact that projection neurons remain active at lower rates is consistent with them being downstream participants or secondary contributors rather than the sole engine.

B2 then rules out another simple story. Cutting all excitatory-LN→excitatory-LN edges reduces activity, but only modestly, while the excitatory interneuron population continues firing strongly. Cutting excitatory-LN→GABA edges makes the state worse, which is exactly what you would expect if those edges recruit inhibition that normally restrains the recurrent activity. Cutting excitatory-LN→glutamatergic neurons barely changes the result, and cutting excitatory-LN→unclear neurons gives only a small improvement. In other words, **no broad transmitter-class split isolates the engine**.

That is not a failure. It means the root cause is probably **not a single gross edge class such as “all cholinergic LN→LN.”** The next step should move from *cell-class lesions* to *the exact active directed graph formed by the 349 neurons*.

There is also a biologically important reason not to expect a simple LN→LN answer. Excitatory local neurons in the Drosophila antennal lobe can participate in reciprocal excitation with projection neurons, while local inhibitory circuits regulate lateral excitation; classical physiology also indicates that excitatory LNs can recruit inhibitory LNs, producing direct excitation and indirect inhibition at the same time. citeturn649138search7turn649138search8turn649138search10 Your E2 result—removing excitation onto GABA neurons makes persistence markedly worse—is therefore qualitatively consistent with an excitatory population that also recruits its own brakes.

The strongest conclusion I would make now is:

> **You have localized the runaway state to a compact antennal-lobe recurrent dynamical core, but you have not yet localized the minimal causal cycle inside that core.**

The next experiments should identify that minimal causal cycle directly.

## A critical correction about the refractory-period hypothesis

Before doing another large lesion screen, I would perform one model-level sanity check because it could change the interpretation of the 400 Hz saturation.

The published Shiu implementation defines both membrane voltage and the synaptic-drive variable `g` with `(unless refractory)`:

```text
dv/dt = (...) : volt (unless refractory)
dg/dt = -g/tau : volt (unless refractory)
```

and resets `g` to zero when the neuron spikes. citeturn649138search0 Brian2 documents `(unless refractory)` as clamping such state variables during refractoriness; variables marked this way are treated as read-only while the neuron is refractory, so incoming synaptic writes to those variables do not simply accumulate normally during that period. citeturn649138search1turn199182search1

So I would **not currently describe the Shiu mechanism as “synaptic input accumulates during refractory, does not decay, and then arrives all at once.”** The source code and Brian2 documentation do not support that description as written. The model can still produce a refractory-limited excitatory loop, but the mechanism is more likely to be the timing and strength of repeated post-refractory excitation than a backlog of synaptic input stored during refractoriness. The Shiu model uses a 2.2 ms refractory period, 1.8 ms transmission delay, 5 ms synaptic time constant, and fixed connection weights derived from synapse counts, which is quite capable of supporting tightly timed recurrent firing when a strongly connected subgraph is sufficiently excited. citeturn606641search0turn606641search3

This matters enough that I would make it **Debug Test Zero**:

Run a tiny two- or three-neuron toy network using the exact same Brian2 equations and log, at 0.1 ms resolution, `v`, `g`, presynaptic spike times, refractory status, and postsynaptic spike times. Confirm exactly what happens to an `on_pre='g += w'` event that lands during refractoriness.

Do this before changing the full model. It is cheap and removes an important uncertainty.

## Stop doing broad edge-class cuts and find the actual recurrent kernel

The central analysis now should be a graph analysis of the **349 persistently active antennal-lobe neurons**, not another family-level lesion sweep.

Build a directed weighted graph containing only those neurons and all edges among them. For each edge, retain at least the presynaptic cell type, transmitter assignment, synapse count, model weight, postsynaptic cell type, and the measured tail firing rates of both neurons.

Then calculate the graph’s **strongly connected components**, or SCCs.

An SCC is a group where every neuron can ultimately reach every other neuron through directed connections. In plain language, it is a part of the network in which signals can travel around loops. Directed strongly connected subgraphs are exactly the structures that can support recurrent circulation in excitable networks; theoretical work on excitable-refractory networks has specifically shown that strongly connected subnetworks can support persistent dynamical attractors. citeturn649138search6

Your 349 neurons are unlikely to constitute one equally important mass. My expectation is that you will find something like:

**large active population → one dense recurrent SCC → several downstream satellites.**

The satellites may fire continuously because the core drives them, but destroying satellite edges will not end the attractor.

For every SCC, calculate:

- number of neurons;
- total excitatory internal weight;
- total inhibitory internal weight;
- excitatory/inhibitory ratio;
- mean and maximum firing rate;
- number of near-ceiling neurons;
- incoming weight from outside the SCC;
- outgoing weight to the rest of the 349;
- shortest directed cycles;
- neurons that participate in the largest number of high-weight cycles.

Do **not** rank neurons only by eigenvector centrality this time. Eigenvector centrality tells you who belongs to an influential region of the graph; it does not tell you which neuron is causally indispensable. A highly redundant recurrent network can have many high-centrality neurons while tolerating the removal of any one of them.

The object you now want is closer to a **feedback vertex set**: a small set of neurons whose removal breaks the important directed cycles.

Conceptually:

```text
          A ─────→ B
          ↑        │
          │        ↓
          D ←───── C

```

Removing a strong outgoing edge from B may reduce activity.

But if an alternative loop exists,

```text
          A ─────→ B
          ↑       ↙ ↓
          │      E  C
          └──────←──┘
```

the activity survives.

That is exactly the pattern your B2 results suggest: **you are cutting families of edges, while the recurrent graph has alternative routes around them.**

## The decisive experiment is a maintenance-lesion screen inside the core

Once you have the SCCs, I would no longer lesion cells before odor presentation as the primary assay.

Let the odor start normally. Let the recurrent state become established. Then, during the silent tail, switch off a candidate neuron or edge group.

This is the cleanest experiment for asking:

> **What is actually required to keep this state alive right now?**

For example:

```text
odor           recurrent tail
██████████│████████████████████████
          ↑
          odor ends

                     ↓ lesion at 300 ms
██████████│██████████│..............
                     ↑
              does activity collapse?
```

If you silence one candidate *after* the attractor is already running and the whole 349-neuron core collapses within a few tens of milliseconds, that neuron or edge set is part of the **maintenance mechanism**.

If nothing happens, it may still help ignite the attractor, but it is not indispensable once the state exists.

I would screen candidates in this order.

First, lesion the **whole dominant excitatory SCC** during maintenance. If that immediately kills the tail, you have verified that the SCC is the engine rather than merely a correlated population.

Second, divide that SCC structurally—not by transmitter class—using communities or recurrent motifs. Lesion each subcommunity during maintenance.

Third, test individual high-cycle-participation neurons, starting with `lLN1_bc`, `lLN2T_a`, `lLN2X04`, `lLN2X05`, `lLN2X11`, `lLN2X12`, and `lLN2T_e`.

Fourth, test **pairs and small combinations**. Your earlier `lLN1_bc` lesion was not decisive because 30 neurons of one annotated type may be only one branch of a redundant recurrent network.

The important readout should now be **time to collapse**, not merely active-neuron count at the end.

For example:

```text
maintenance intervention       outcome

candidate X off                349 → 320 → 305 → 290
                               probably modulation

candidate Y off                349 → 280 → 70 → 0
                               candidate causal breaker

candidate Y+Z off              349 → 0 within 15 ms
                               strong redundant-pair evidence
```

This will tell you much more than another 100-ms endpoint table.

## Search for the smallest cycle that can reproduce the attractor

After identifying the necessary maintenance set, reverse the logic.

Take a network in which the recurrent core has been disabled and **add pieces back**.

This is the sufficiency test.

Suppose graph analysis identifies a candidate motif such as:

```text
lLN1_bc
   ↓
lLN2X12
   ↓
lLN2T_a
   ↓
lLN1_bc
```

or a motif involving PNs:

```text
eLN → PN
 ↑     ↓
 └─────┘
```

Test whether restoring only that motif recreates high-frequency persistent firing.

Physiologically, reciprocal excitation between excitatory local neurons and PNs has been reported in Drosophila, and PNs can provide cholinergic excitation back to LNs; therefore PN↔LN recurrence is biologically plausible even though your projection-edge lesion shows that some local persistence survives without normal PN pathways. citeturn649138search7turn649138search10

The key distinction is:

**necessary** = removing it stops the attractor.

**sufficient** = restoring it can bring the attractor back.

Your strongest causal result would be:

```text
baseline core               persistent
remove motif                no persistence
restore motif               persistence returns
```

That is much stronger than “these neurons have the highest eigenvector score.”

I would call that **Stage C-revised: kernel identification**, and I would postpone the old gain sweep until this is done.

## Audit transmitter signs only where they affect the kernel

I would still perform your proposed unclear-neuron test, but not across the entire antennal lobe first.

Run condition `U`: silence the outputs of the 46 unclear interneurons.

If the recurrent core barely changes, stop worrying about their transmitter identities for now.

If it collapses or changes substantially, identify **which unclear neurons actually lie inside the causal SCC** and audit only those first.

That is much more efficient than immediately downloading or processing every per-synapse transmitter probability.

The MaleCNS transmitter predictions come from synapse-level EM classification, and the recent MaleCNS description explicitly treats those transmitter probabilities as predictions rather than ground-truth synaptic signs. citeturn174895search2turn174895search3 This matters particularly for ambiguous neurons. Furthermore, transmitter identity is not identical to postsynaptic effect: acetylcholine is generally excitatory and GABA generally inhibitory in fly central circuits, but glutamate can be either excitatory or inhibitory depending on receptor context. In the antennal lobe specifically, glutamatergic local neurons have experimentally been shown to exert inhibition through glutamate-gated chloride channels. citeturn174895search3turn174895search4

So I would **not** reinterpret your ~40 glutamatergic core neurons as excitatory without receptor evidence. Under the Shiu sign convention they are inhibitory, and olfactory physiology provides real support for inhibitory glutamatergic signaling in this circuit. citeturn174895search4

## The final sequence I recommend

Your old Stage B gain sweep is no longer the highest-value next experiment. B1/B2 have already shown that coarse edge classes are too broad.

I would replace the remaining pipeline with this one:

| New stage | Experiment | Question answered | Stop/go criterion |
|---|---|---|---|
| **C0** | Verify exact refractory/`g` behavior in a 2–3 neuron Shiu toy model | Is the saturation partly a simulator-rule misunderstanding? | Must understand this before interpreting 400 Hz |
| **C1** | Construct the induced graph of the 349 tail neurons and find SCCs | Where can activity actually circulate? | Identify dominant recurrent SCC(s) |
| **C2** | Maintenance lesion of each SCC after odor offset | Which SCC actually maintains the state? | One lesion should cause rapid global collapse |
| **C3** | Within winning SCC, rank neurons/edges by cycle participation and lesion combinations | What is the smallest necessary recurrent kernel? | Find smallest set that reliably ends tail |
| **C4** | Add back candidate motif into the stable lesion network | Is that kernel sufficient? | Restoration recreates persistence |
| **C5** | Audit unclear transmitter signs only inside that kernel | Is persistence caused by a sign assignment error? | Fix signs if biologically unsupported |
| **C6** | Only now perform a local gain sweep on the identified causal edges | What is the smallest intervention that stabilizes the model? | Find highest gain with no runaway |
| **C7** | Full MaleCNS, multiple seeds and all sensory inputs | Does the fix generalize without breaking behavior? | Stable tails + preserved sensory/MN outputs |

For C6, do **not** scale all LN→LN or all LN→PN edges. Scale only the causal kernel discovered in C2–C4. That turns a very blunt intervention into a mechanistically interpretable one.

For example, if the culprit turns out to be 18 neurons connected by 73 high-weight recurrent edges, your final parameter would be something like

\[
W_{ij}^{new}=\alpha_{\mathrm{core}}W_{ij}
\]

only for those 73 edges.

Then try

\[
\alpha_{\mathrm{core}}=
0.9,\;0.8,\;0.7,\;0.6,\ldots
\]

until the persistent state disappears.

The goal is **not** “make the antennal lobe quieter.”

The goal is:

> **Find the smallest biologically questionable or dynamically excessive recurrent motif whose correction converts a self-sustaining attractor back into a transient odor response.**

That is now the shortest route to the root cause.

## What I think is most likely happening

Based on the evidence so far, I would rank the hypotheses this way.

The leading hypothesis is a **redundant, strongly connected excitatory LN kernel** in which several cholinergic LN subclasses can substitute for one another. That explains why removing `lLN1_bc` alone helps but does not cure the state, why removing all excitatory→excitatory LN edges reduces firing without eliminating it, and why broad classes of other edges can be removed while the network finds another recurrent route.

A second possibility is a **mixed LN–PN recurrent kernel** where PNs are not required for the smallest residual core but greatly amplify and broadcast it. This fits your finding that cutting LN→PN collapses the large tail from thousands of neurons to hundreds, while B1 still leaves a 349-neuron self-sustaining population.

A third possibility is that one or a few **incorrect transmitter assignments** create a positive-feedback shortcut that should biologically be inhibitory. The unclear neurons therefore remain worth auditing, but your current data do not make them the leading explanation.

A fourth possibility is a **model-regime problem rather than a connectome error**: fixed PSP-per-synapse scaling plus an integrate-and-fire model lacking adaptation and other stabilizing cellular mechanisms may turn a real recurrent olfactory circuit into an unrealistically stable high-rate attractor. The Shiu model deliberately simplifies neurons and assigns connection strength from synapse count and transmitter sign; it was designed as a useful connectome-level computational model, not a biophysically complete account of every local circuit. citeturn606641search0turn606641search2 If you discover that the minimal recurrent kernel corresponds to a biologically plausible antennal-lobe circuit but only becomes pathological at your simulation gain, then the correct fix may be **model calibration**, not deleting the anatomical circuit.

The most diagnostically valuable next result is therefore not another global tail number. It is a trace like:

```text
Attractor established
        ↓
causal SCC lesion applied
        ↓
lLN1_bc: 400 → 0 Hz
lLN2X12: 340 → 0 Hz
GABA LNs: 280 → 0 Hz
PNs:      40 → 0 Hz
whole core collapses in 12 ms
```

Once you can produce that experiment, you will have moved from **“the antennal lobe is involved”** to **“this particular recurrent loop is the maintenance engine.”**