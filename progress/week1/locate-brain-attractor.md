# Locating the Self-Sustaining Brain Attractor in the MaleCNS Shiu Model

## What your new experiments establish

Your new results substantially change the diagnosis.

| Condition | Persistent stimuli | Brain neurons left active | Nerve-cord neurons left active | Interpretation |
|---|---:|---:|---:|---|
| Full network | 14 / 18 | 4,376 | 1,013 | Baseline pathological persistence |
| Brain only | **15 / 18** | **4,310** | 0 | The brain alone is sufficient |
| No nerve-cord→nerve-cord weights | 14 / 18 | **4,670** | 747 | Removing VNC recurrence does not suppress the brain state |
| No abdominal output | 14 / 18 | 4,360 | 980 | Abdominal output is essentially irrelevant |

The decisive result is **brain only**. Removing the entire nerve cord reduces the persistent brain population from 4,376 to 4,310—only about **1.5%**—while the number of stimuli producing persistence actually increases from 14 to 15. Likewise, deleting abdominal output changes the brain count by only about **0.4%**. Removing VNC→VNC connectivity makes brain persistence **larger**, by about **6.7%**.

So I would now reject the earlier working hypothesis that the nerve cord is the source of the pathological attractor.

The strongest interpretation of your data is:

> **A recurrent state inside the MaleCNS brain is sufficient to maintain thousands of active neurons after the sensory input has ended. The nerve cord is neither necessary nor the principal sustaining circuit.**

There is even a weak suggestion that cord circuitry may exert some *net stabilizing or inhibitory influence* on the brain, because brain persistence rises from 4,376 to 4,670 when VNC→VNC recurrence is removed and from 14 to 15 persistent stimuli when the VNC is removed altogether. I would not regard that secondary effect as established until you repeat trials with matched random seeds and confidence intervals, but it points in the opposite direction from the original VNC hypothesis.

That makes the next task much more specific: **find the relatively small recurrent brain circuit that ignites and maintains the 4,000-neuron state rather than repeatedly lesioning large anatomical divisions.**

This is also a situation where the Shiu model's assumptions matter greatly. The published model is a homogeneous leaky-integrate-and-fire network in which synaptic strength is proportional to connectomic synapse count, with neurotransmitter identity supplying the sign of the connection. The global per-synapse strength \(W_{\rm syn}=0.275\) mV is explicitly described as a **free parameter**, rather than a universal biological synaptic constant. citeturn20search0turn27search0 Consequently, transferring the same dynamics to a different connectome can legitimately move particular recurrent circuits across a stability boundary even if a global synapse-density correction looks reasonable.

My recommended investigation is therefore:

**sanity-check the port → fingerprint the persistent population → identify recurrent cores → causally lesion those cores → locally recalibrate only the offending recurrence.**

## First rule out a simulation artifact before searching anatomy

Before looking for a biological-looking "attractor circuit," I would run four very cheap tests. They distinguish a genuine recurrent brain state from a porting error that can produce exactly the same phenotype.

### The most decisive experiment: cut brain recurrence at stimulus offset

Take one of the 15 stimuli that produces persistence in the brain-only model.

Run it normally during the stimulus. At the exact instant the stimulus ends, set **all brain→brain synaptic weights to zero**, while leaving neuronal membrane variables untouched.

Do **not** remove the connections from the beginning of the trial. Remove them only at stimulus offset.

Then monitor the next 300–500 ms.

The logic is simple:

\[
\text{stimulus}
\quad\longrightarrow\quad
\boxed{\text{brain recurrence ON}}
\quad
\xrightarrow[T_{\rm off}]{}
\quad
\boxed{\text{brain recurrence OFF}}
\]

In Shiu's published model, the synaptic decay time constant is about 5 ms, while the membrane time constant is about 20 ms; the threshold is −45 mV and resting/reset potential is −52 mV. citeturn26search3turn27search0 Therefore, after all new recurrent synaptic events are stopped, residual synaptic state should rapidly decay. After roughly 100 ms, a 5-ms synaptic exponential has undergone 20 time constants; there should not be thousands of neurons continuing to fire hundreds of milliseconds later merely because of stored synaptic conductance.

This experiment gives an unusually clean diagnostic:

| Result after cutting brain→brain weights at \(T_{\rm off}\) | Meaning |
|---|---|
| Activity rapidly dies | **Real recurrent attractor under your model** |
| Thousands continue firing for the final 300 ms | Something other than recurrence is still driving them |
| Activity decreases but a large subset remains | Recurrence plus an additional persistent input/state source |

If the second outcome occurs, **stop the anatomical search**. The most likely problem would then be an external-drive or state-management bug.

### Run a completely unstimulated fresh network

Instantiate the brain-only model from a completely fresh state and supply **zero Poisson input for the entire trial**.

Expected result:

\[
N_{\rm active}(t)=0
\]

apart from any deliberately implemented spontaneous source.

The Shiu model is a simplified LIF model driven by explicit stimulation; it is not a detailed model containing all endogenous biological currents and neuromodulatory processes. citeturn20search0turn22search1 Therefore spontaneous firing in an otherwise unforced port needs an explicit explanation in your implementation.

This test catches several common implementation failures:

previous-trial spikes left in a queue; membrane or synaptic state not reset between words; a Poisson generator that remains active; reused simulator objects; or an external rate that is set to "stimulus off" in your code while the actual event generator continues running.

### Verify that the stimulus really ends

This deserves special attention because your experiment contains a "silent final 300 ms," whereas the original Shiu framework was designed principally around responses under defined Poisson activation conditions. citeturn20search0turn22search1

At the exact stimulus-off timestamp, log:

\[
\sum_i \text{external spikes}_i(t)
\]

rather than merely logging your stimulus flag.

You want the external event count itself to become exactly zero.

A particularly strong control is:

> **At \(T_{\rm off}\), simultaneously set every external Poisson rate to zero and all brain→brain weights to zero.**

If anything is still firing in the last 100–300 ms, the problem is almost certainly in simulator state/reset semantics rather than connectome recurrence.

### Confirm what "silencing" means in your port

For the coming lesion search, distinguish three manipulations:

\[
\begin{aligned}
\text{output lesion:}&\quad W_{i\rightarrow *}=0\\
\text{input lesion:}&\quad W_{*\rightarrow i}=0\\
\text{full node deletion:}&\quad W_{i\rightarrow *}=W_{*\rightarrow i}=0
\end{aligned}
\]

They answer different questions.

For locating a sustaining recurrent circuit, **output silencing is usually the cleanest first test**: the candidate neuron can still receive activity, but cannot propagate it.

Do not casually mix this with deleting both incoming and outgoing weights. A full deletion can remove feed-forward sensory pathways and make a candidate appear important simply because the original stimulus never reached the rest of the circuit.

This distinction becomes especially important when comparing your implementation with published Shiu-style perturbations, because the experimental logic is to suppress propagation from selected neurons rather than arbitrarily alter every aspect of their state. The published computational work uses connectome-defined directed connectivity and targeted circuit perturbation to assess causal pathways. citeturn20search0turn20search9

**I would not proceed to brain-region lesions until these four controls pass.** In particular, the recurrence-at-offset experiment can save many hours of blind searching.

## Build a fingerprint of the attractor before lesioning anything

The fact that approximately 4,300 brain neurons remain active does **not** imply that all 4,300 constitute the attractor.

Most may be downstream followers.

A much smaller recurrent core could continuously reactivate thousands of downstream cells:

\[
\boxed{\text{50–500 driver neurons}}
\rightarrow
\boxed{\text{recurrent amplification}}
\rightarrow
\boxed{\text{4,000+ active followers}}
\]

This possibility is consistent with whole-brain connectome analyses showing that dominant connectivity modes can be concentrated in comparatively sparse subsets of neurons. Pospisil and colleagues used eigenmodes of the signed FlyWire connectome to identify "eigencircuits": sparse neuron populations predicted to contribute disproportionately to recurrent dynamics. In their FlyWire analysis, the highest-ranked modes were often far smaller than the full brain-wide population affected by them. They also explicitly warn that connectome eigenmodes are hypotheses about dynamics, not substitutes for the true physiological Jacobian. citeturn21view2

So the next step should use your **spike data**, not anatomy, to shrink 4,300 neurons into a candidate driver set.

### Separate common-core neurons from stimulus-specific followers

For every one of your 18 word stimuli, run repeated trials—ideally with the same set of random seeds for every lesion condition.

For neuron \(i\), calculate a persistence probability:

\[
P_i =
\frac{\text{number of persistent trials in which neuron }i
\text{ fires in the silent tail}}
{\text{number of persistent trials}}
\]

Do this first using the **last 100 ms**, not the entire final 300 ms.

Why? A neuron that spikes 10 ms after stimulus offset may simply be finishing a transient response. A neuron firing 250–300 ms after input termination is much stronger evidence of participation in the maintained state.

I would calculate two versions:

\[
P_i^{\rm any}=
P(\geq 1\text{ spike in final 100 ms})
\]

and

\[
P_i^{\rm sustained}=
P(\text{rate exceeds your chosen low threshold in final 100 ms}).
\]

Then divide the population conceptually into:

| Population | Recommended working definition | Interpretation |
|---|---|---|
| Common core | Persistent in ≥80–90% of persistent stimuli/trials | Candidate global attractor machinery |
| Frequent shell | Persistent in ~40–80% | Likely recruited by the core |
| Stimulus-specific | Persistent only for one/few words | Word pathway or attractor-specific branch |
| Early-only | Active after input but absent late | Transient follower |

The 80–90% cutoff is a **diagnostic heuristic**, not a biological threshold. Repeat the analysis at several thresholds so the conclusion does not depend on a single arbitrary cutoff.

### Measure whether the 15 persistent words enter the same attractor

This is extremely important.

For every pair of stimuli \(a,b\), calculate the Jaccard overlap of late persistent populations:

\[
J(a,b)=
\frac{|P_a\cap P_b|}
{|P_a\cup P_b|}.
\]

Also compare the **firing-rate vectors**, not merely active/inactive sets, using cosine similarity or correlation.

You may discover either:

**One attractor**

\[
\text{sugar}\rightarrow C,\quad
\text{hot}\rightarrow C,\quad
\text{word X}\rightarrow C
\]

where almost every stimulus collapses onto essentially the same late brain state;

or:

**Several attractors**

\[
\begin{aligned}
\text{sugar, salt, X}&\rightarrow C_1\\
\text{hot, Y, Z}&\rightarrow C_2.
\end{aligned}
\]

Your earlier observation that different words lose stimulus specificity while producing very large overlapping active populations makes the first possibility particularly plausible, but the actual spike identities should decide it.

### Find who fires first when the runaway state ignites

Now take the neurons in the common persistent population and go backward in time.

Bin spikes at approximately 5–10 ms resolution around the transition from normal sensory activity into runaway activity.

For each candidate neuron ask:

> Does this neuron become active **before** the large population explosion, or only afterward?

For a trial with runaway onset \(t_{\rm ignite}\), define an onset delay:

\[
d_i=t_i^{\rm first\ sustained}-t_{\rm ignite}.
\]

The strongest driver candidates will have:

\[
P_i\text{ high}
\quad+\quad
d_i<0
\quad+\quad
\text{strong recurrent output into other persistent neurons}.
\]

A neuron that always turns on 20–50 ms **after** thousands of others is much more likely a passenger.

This dynamic ordering is much more informative than simply ranking the 4,300 neurons by firing rate.

### The output from this stage should be a small table

At the end, generate something like:

| bodyId | Type | Neuropil | NT | Persistence probability | Median onset relative to ignition | Recurrent E output to core |
|---|---|---|---|---:|---:|---:|
| A | … | … | ACh | 1.00 | −18 ms | high |
| B | … | … | ACh | 0.98 | −14 ms | high |
| C | … | … | GABA | 0.96 | −11 ms | high |
| … | | | | | | |

That table—not "brain" versus "VNC"—should become the basis of the next lesion round.

## Locate the recurrent core directly in the MaleCNS graph

MaleCNS gives you exactly the data required to do this without guessing regions. Janelia provides the complete segment-to-segment connection graph, curated neuron annotations, neuron-level neurotransmitter predictions, synaptic partner tables with `primary_post` neuropil assignments, and even neurotransmitter probabilities at individual presynaptic sites. citeturn21view1

I would construct a **small induced brain graph** around the persistent population rather than eigendecompose the entire MaleCNS CNS initially.

### Start with the late persistent population plus its precursors

Let

\[
C = \{\text{late persistent brain neurons}\}
\]

and

\[
Q =
\{\text{neurons active shortly before runaway ignition}\}.
\]

Start with

\[
G_{\rm candidate}=C\cup Q.
\]

If that misses obvious upstream recurrence, add one presynaptic hop:

\[
G_{\rm candidate}
=
C\cup Q\cup
\operatorname{Pre}(C).
\]

Then retain only **brain→brain** edges.

For each edge, construct the exact weight that your simulation uses:

\[
W_{ij}
=
n_{ij}\,
s_i\,
w_{\rm syn},
\]

where \(n_{ij}\) is synapse count, \(s_i\) is the transmitter-derived sign in your port, and \(w_{\rm syn}\) is your current global scale.

That guarantees the graph analysis investigates **your simulation's attractor**, not an abstract anatomical network.

### Find strongly connected components before looking at brain regions

A self-maintaining recurrent loop requires directed feedback.

So compute **strongly connected components (SCCs)**.

A strongly connected component means that every neuron in that component can eventually reach every other neuron through directed paths.

For example:

\[
A\rightarrow B\rightarrow C\rightarrow A
\]

is strongly connected.

But:

\[
A\rightarrow B\rightarrow C
\]

is not.

The first structure can repeatedly circulate activity. The second cannot sustain itself indefinitely without another feedback route.

Rank SCCs by a combination of:

\[
\text{persistent participation}
\times
\text{internal excitatory weight}
\times
\text{early ignition participation}
\times
\text{component size}.
\]

I would perform this twice:

1. on the **positive/excitatory subgraph**, to find direct recurrent excitation;
2. on the **full signed graph**, because an attractor can also involve disinhibitory loops rather than a simple all-excitatory cycle.

Do not simply pick the largest SCC. Large connectomes naturally contain enormous strongly connected regions. What matters is a component unusually enriched for **late-active, early-igniting neurons and strong internal positive feedback**.

### Then use eigenmodes as a ranking tool

On each promising induced component, compute the leading eigenvectors/eigenvalues of your **signed simulation weight matrix**.

Pospisil et al. provide particularly relevant precedent: they use scaled signed connectomic weights and show that high-loading neurons in dominant eigenmodes provide candidate "eigencircuits" for disproportionately influential recurrent dynamics. They also show that dominant modes of the fly connectome can be sparse. citeturn21view2

But there is an important restriction:

> **Do not conclude that an eigenvalue above some arbitrary value proves instability.**

The true Shiu model is nonlinear, thresholded, refractory and temporally filtered. Pospisil et al. explicitly note that a signed connectome is generally only a first-order proxy for the dynamical Jacobian; membrane constants, voltage state, synaptic physiology and other parameters can alter the relationship between anatomy and dynamics. citeturn21view2

Use spectral analysis to answer:

> **Which neurons and edges should I lesion first?**

not:

> "Has linear algebra proven that this is the attractor?"

A useful ranking for neuron \(i\) is the product of its participation in your measured persistent state and its loading in the dominant candidate mode. For even more targeted intervention, the product of left- and right-eigenvector loadings can identify edges/nodes to which the dominant eigenvalue is particularly sensitive.

### Map anatomy only after identifying the graph core

Once you have, say, the top 50–500 candidate neurons, use the MaleCNS annotations and the `primary_post` neuropil field to ask:

\[
\text{Where are these neurons actually concentrated?}
\]

MaleCNS supplies manually refined brain neuropil segmentation plus cell classes/types and connection-level primary neuropil assignments, so you can localize candidate recurrence quantitatively rather than assuming in advance that it must be mushroom body, central complex, superior medial protocerebrum, and so on. citeturn21view1

For each neuropil \(R\), compute something like:

\[
F_R =
\frac{
\text{candidate-core recurrent synapses occurring in }R
}{
\text{all candidate-core recurrent synapses}
}.
\]

You may find that "4,300 brain neurons" is actually driven by a few hundred neurons whose recurrent synapses are heavily concentrated in one or two neuropils.

That is the brain attractor you want to lesion.

## Prove causality with maintenance and ignition lesions

Once the dynamic and graph analyses give you candidate SCCs or eigencircuits, switch from large anatomical ablations to **targeted causal tests**.

This is the stage that will distinguish a genuine driver circuit from neurons that merely happen to be highly active.

### Use two temporally different lesions

For candidate circuit \(C^\*\), perform:

#### Maintenance lesion

Run the stimulus normally. Let the abnormal state form.

Exactly at stimulus offset:

\[
W_{C^\*\rightarrow *}=0.
\]

Question:

> Can the already-established attractor survive without this circuit?

If 4,300 late neurons collapse to tens or hundreds, \(C^\*\) is required for **maintenance**.

#### Ignition lesion

Silence \(C^\*\) during stimulus presentation, but restore it at stimulus offset.

Question:

> Is the circuit required to push the network into the attractor basin in the first place?

This separates two roles that a lesion from \(t=0\) cannot distinguish.

You could obtain:

| Ignition lesion | Maintenance lesion | Interpretation |
|---|---|---|
| Stops persistence | Stops persistence | Core attractor machinery |
| Stops persistence | No effect once established | Trigger / gateway |
| Little effect | Stops established activity | Maintenance circuit |
| Neither | Correlated follower |

That distinction will make your search much faster.

### Lesion edges as well as neurons

Suppose candidate neurons A, B and C form:

\[
A\rightarrow B\rightarrow C\rightarrow A.
\]

Deleting A completely might also destroy A's normal sensory function.

Instead, selectively delete the **internal recurrent edges**:

\[
W_{A\rightarrow B}
=
W_{B\rightarrow C}
=
W_{C\rightarrow A}
=0,
\]

while preserving:

\[
\text{sensory input}\rightarrow A
\]

and

\[
C\rightarrow\text{normal downstream target}.
\]

If the sensory response remains but persistence disappears, you have much stronger evidence that **recurrent gain itself**, rather than the biological identity of the neurons, causes the problem.

### Use a staged lesion search rather than one neuron at a time

Once candidates are ranked, I would test approximately this sequence:

| Experiment | What it tells you |
|---|---|
| Top candidate SCC | Is this recurrent component necessary? |
| Top 50 ranked neurons | Is the driver compact? |
| Top 25 | Narrowing |
| Top 10 | Narrowing |
| Top 5 | Narrowing |
| Individual neurons/edges | Precise causal locus |
| Restore removed edges | Rescue test |

A particularly efficient method is **binary partitioning**.

Suppose a candidate SCC contains 200 neurons:

\[
200\rightarrow100\rightarrow50\rightarrow25\rightarrow\cdots
\]

Lesion half. Keep subdividing the half whose removal suppresses persistence.

If neither half alone works but the full group works, you have evidence for distributed redundancy rather than a single bottleneck.

### Require rescue, not just lesion

After finding a candidate recurrent subnetwork:

1. remove its internal recurrent edges → persistence disappears;
2. put those exact edges back → persistence returns.

That A→B reversal is substantially stronger than a single lesion result.

Even better, vary only those internal excitatory weights:

\[
\alpha W_{C^\*\rightarrow C^\*},
\qquad
\alpha =
1.0,\;0.9,\;0.8,\;0.7,\ldots
\]

and plot final silent-tail population versus \(\alpha\).

You may see:

\[
\begin{array}{c|c}
\alpha & N_{\rm persistent}\\
\hline
1.00 & 4310\\
0.90 & 3980\\
0.85 & 3510\\
0.80 & 410\\
0.75 & 35
\end{array}
\]

Those numbers are illustrative, but a sharp transition of this kind would be very informative. It would show that MaleCNS placed a particular brain recurrent circuit just above a dynamical threshold.

That gives you a **specific correction** instead of the unsuccessful global \(1/1.43\) rescaling.

## Why your previous fixes could fail, and what I would change

### Global \(1/1.43\) rescaling is probably too coarse

The Shiu synaptic scaling parameter is a global free parameter in the original FlyWire-derived model. citeturn27search0 There is no reason to expect a single average MaleCNS/FlyWire density ratio to correct every local recurrent circuit equally.

Imagine two regions:

\[
\text{Region A: MaleCNS/FlyWire recurrent gain}=1.05
\]

\[
\text{Region B: MaleCNS/FlyWire recurrent gain}=2.3.
\]

A global division by 1.43 would make A too weak while B could remain strongly supercritical.

Your experiment is consistent with that scenario: global scaling reduced activity in your earlier runs but did not restore normal post-stimulus extinction.

The better comparison is **local recurrent gain**, preferably matched by cell type or neuropil.

For candidate component \(C^\*\), calculate:

\[
G_E(C^\*)=
\sum_{\substack{i,j\in C^\*\\W_{ij}>0}}W_{ij}
\]

and compare it with:

\[
G_I(C^\*)=
\left|
\sum_{\substack{i,j\in C^\*\\W_{ij}<0}}W_{ij}
\right|.
\]

Then examine incoming E/I as well:

\[
G_E^{\rm in}(j),\qquad
G_I^{\rm in}(j)
\]

for each candidate neuron.

The most informative MaleCNS-vs-FlyWire comparison would be at the **matched type or neuropil level**, not CNS-wide.

### Audit the neurotransmitter mapping before touching weights again

This is potentially one of the highest-value checks in the entire project.

The original FlyWire/Shiu framework was constructed around neurotransmitter predictions used to sign connections; the published Shiu methods classify predicted GABAergic and glutamatergic neurons as inhibitory in the model. citeturn25search0

MaleCNS, however, exposes a richer neurotransmitter annotation resource. Its release includes aggregate neuron-level predictions as well as neurotransmitter probabilities at each presynaptic T-bar, and the MaleCNS publication describes transmitter annotations beyond the small set commonly used in the original FlyWire model, including glycine, histamine, nitric oxide and tyramine in addition to acetylcholine, dopamine, GABA, glutamate, octopamine and serotonin. citeturn21view1turn22search2

Therefore inspect your code for anything equivalent to:

```python
if nt in ["GABA", "glutamate"]:
    sign = -1
else:
    sign = +1
```

That reproduces one simplified rule only if the transmitter vocabulary and intended treatment are the same.

With MaleCNS, an `else = excitatory` fallback can silently convert **new, uncertain or unsupported transmitter categories** into fast positive LIF synapses.

I would explicitly generate this audit table:

| MaleCNS NT class | Number of brain neurons | Sign used by your simulator | Confidence handling | Intended by exact Shiu convention? |
|---|---:|---:|---|---|
| ACh | | + | | |
| GABA | | − | | |
| Glutamate | | − | | |
| Dopamine | | | | |
| Octopamine | | | | |
| Serotonin | | | | |
| Histamine | | | | |
| Glycine | | | | |
| Tyramine | | | | |
| Nitric oxide / other | | | | |
| Uncertain / missing | | | | |

Then determine what fraction of **candidate-attractor internal positive weight** comes from each class.

This matters much more than simply repeating a "monoamines removed" test across the whole network. If a small recurrent core is incorrectly signed, global removal experiments can obscure the mechanism.

MaleCNS provides synapse-level transmitter probability data, so a useful robustness test is to rebuild the candidate component using only high-confidence transmitter assignments and see whether its persistence survives. citeturn21view1

### APL was never a strong whole-brain stabilizer test

Your failure to cure the runaway state by adding one APL inhibitory-feedback neuron now makes much more sense.

APL is an inhibitory feedback neuron associated with the **mushroom body**, not a generic global inhibitory controller. Experimental work further shows that adult APL is non-spiking and that its activity and inhibition can be spatially localized within mushroom-body compartments. citeturn22search0turn22search5

So two things follow.

First, if your pathological attractor lives elsewhere in the brain, strengthening APL may have very little leverage over it.

Second, representing APL as an ordinary homogeneous Shiu-style spiking LIF neuron is itself a substantial physiological simplification, because the experimental APL is non-spiking and compartmentalized. citeturn22search0

Therefore:

> **Do not keep increasing APL inhibition in an attempt to compensate for an unidentified brain recurrence problem.**

Locate the unstable recurrent component first.

### The most likely final correction is local, not global

If the implementation sanity checks pass and you identify a genuine MaleCNS recurrent core, I would test fixes in this order:

| Priority | Correction | Why |
|---|---|---|
| **Highest** | Fix any external-input/reset error | Otherwise there is no biological attractor to correct |
| **Highest** | Correct transmitter/sign mapping | A wrong sign directly manufactures positive feedback |
| **High** | Correct anomalously strong internal recurrent weights in identified SCC | Directly targets the pathological loop |
| **High** | Type/neuropil-specific MaleCNS→FlyWire normalization | Better than CNS-wide 1/1.43 |
| Medium | Cap extreme connection weights / fan-in after validating connectome distributions | Useful if a few very large edges dominate |
| Medium | Add biologically appropriate local inhibition | Only after identifying the affected circuit |
| Low | More global APL inhibition | Not anatomically targeted |
| Low | Further uniform whole-network scaling | Risks destroying normal responses before removing the attractor |

The MaleCNS dataset gives you the annotations, full graph, primary neuropil assignments and neurotransmitter probabilities needed for this targeted normalization. citeturn21view1

## The experiment sequence I would run now

The fastest path from your current result to the actual cause is the following funnel.

### Immediate diagnostic funnel

| Run | Manipulation | Measurement | Decision |
|---|---|---|---|
| **A** | Fresh brain, zero stimulus | Final 300-ms spikes | Must be essentially silent |
| **B** | Persistent word; at stimulus offset set external drive = 0 **and brain→brain W = 0** | Final 300 ms | If persistence remains → implementation/state bug |
| **C** | Persistent word; at offset external drive = 0, recurrence unchanged | Reproduce ~4,310 | Confirms phenotype |
| **D** | Same as C across repeated seeds | Identity of late-active neurons | Build persistent core |

Run B before anything computationally expensive.

### Attractor-localization funnel

For the 15 persistent stimuli:

\[
\boxed{
\text{final-100-ms persistent neurons}
}
\]

↓

calculate cross-stimulus persistence probability

↓

\[
\boxed{
\text{common late core}
}
\]

↓

find neurons firing immediately **before** population ignition

↓

\[
\boxed{
\text{core + precursor population}
}
\]

↓

extract brain→brain MaleCNS subgraph

↓

strongly connected components

↓

rank by internal positive weight + persistence + early onset

↓

signed-connectome eigenmode ranking

↓

\[
\boxed{
\text{top 50–500 candidate driver neurons}
}
\]

↓

maintenance lesions at stimulus offset

↓

binary split / neuron and edge lesions

↓

restore edges for rescue

↓

locally scale recurrent E weights

↓

\[
\boxed{
\text{minimal attractor circuit + correction}
}
\]

This approach follows an important lesson from connectome dynamics work: structural connectivity is excellent for **prioritizing candidate recurrent circuits**, but causal perturbation is needed to determine which circuit actually maintains the dynamics. Pospisil et al. make exactly this distinction between connectomic priors/eigencircuits and the underlying causal "effectome." citeturn21view2

### What would count as a convincing solution

I would not define success as merely reducing the number of active neurons.

Your current history already shows why that is dangerous: global changes can reduce 15,000 neurons to 10,000 or 6,000 while leaving the fundamental pathology intact.

Instead, require all of these outcomes simultaneously:

\[
\boxed{
\text{large post-stimulus activity}
\rightarrow
\text{approximately zero}
}
\]

for normally transient words;

\[
\boxed{
14\text{–}15/18\ \text{persistent stimuli}
\rightarrow
\text{near zero pathological persistence}
}
\]

while preserving:

\[
\boxed{
\text{stimulus-dependent differences during input}
}
\]

and preserving the outputs you care about—for example the appropriate MN9 responses rather than simply silencing the network.

For the candidate attractor itself, I would regard the evidence as strong when:

**internal recurrent-edge lesion kills the silent-tail state; feed-forward sensory activation remains; restoring those edges restores persistence; and modestly reducing only those edges crosses a clear persistence threshold.**

That would establish much more than "this brain region was active." It would show that you have located the recurrent mechanism responsible for the MaleCNS failure.

The central change from the previous strategy is therefore:

> **Do not ask which large anatomical compartment contains the runaway state. Your brain-only experiment has already answered that. Ask which small set of brain neurons is active late across stimuli, fires before the runaway transition, forms strongly recurrent signed connectivity, and is causally necessary to maintain the state after stimulation stops.**

Given your present data, my strongest working hypothesis is no longer "MaleCNS has too much recurrence globally." It is more specific:

> **One or a few MaleCNS brain recurrent components are above the stability boundary under the transferred Shiu parameters and/or transmitter-sign mapping. Those components recruit thousands of downstream followers. The correct fix is likely a sign correction or local recurrent-gain correction in those components, rather than another whole-CNS scaling factor or stronger APL feedback.** This is an inference from your lesion results combined with the structure and assumptions of the Shiu model, and it should now be tested with the offset-recurrence cut and core/SCC lesion sequence above. citeturn27search0turn21view1turn21view2