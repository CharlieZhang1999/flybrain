# Diagnosis and Fix for Stimulus-Independent Firing in MaleCNS

## Executive diagnosis

Your results are much more consistent with a **supercritical recurrent attractor** than with an error in the sensory-word mapping. The strongest clue is not simply that thousands of neurons fire; it is that, in your screenshot, **14 of 18 stimuli remain active after the stimulus ends**, with roughly **6,000–9,500 neurons continuing to fire during the nominally silent final 300 ms**. The large overlap between words is then a consequence: many different inputs are pushing the CNS into approximately the same self-sustaining state.

There are **two separate issues that should be treated in order**.

First, verify that your MaleCNS simulator reproduces some unusually important details of Shiu et al.'s Brian2 implementation. In the reference code, a spike does **not** merely reset membrane voltage: it explicitly resets the synaptic state variable `g` to zero. The reference model also uses a 1.8-ms transmission delay, a 2.2-ms refractory period, a 5-ms synaptic time constant, and weights equal to signed synapse count × 0.275 mV. citeturn21view5turn21view6 Missing the `g = 0` reset in particular can make recurrence considerably more persistent than the published Shiu dynamics.

Second, even if your implementation is exact, **the FlyWire-calibrated Shiu parameters should not be transplanted unchanged to MaleCNS**. MaleCNS is a different specimen, reconstructed with a different pipeline, contains the VNC that FlyWire lacks, and has substantially more edges. An independent 2026 implementation that reproduced the Shiu equations and tick ordering reports precisely the phenomenon you are seeing when it runs the model on MaleCNS: stimulus-triggered activity can move into the VNC, become concentrated in abdominal neuromeres, and persist after input disappears. citeturn21view0turn21view1turn22view0

Most importantly, your **1.43 density correction is probably too small**. The MaleCNS authors themselves use a factor of **0.581** to align MaleCNS edge weights to FlyWire for cross-dataset comparisons. Their analysis attributes the systematic weight difference to biological and technical differences between the datasets. citeturn25search0turn26search1turn29search1

You currently use

\[
\frac{1}{1.43}=0.6993
\]

of the original Shiu weight. The MaleCNS/FlyWire comparison instead suggests starting at

\[
0.581
\]

of the FlyWire-calibrated weight. Therefore, **from your current 1/1.43 setting, multiply every MaleCNS synaptic weight by another ≈0.831**:

\[
\frac{0.581}{0.6993}=0.831.
\]

Equivalently, instead of \(0.275/1.43 \approx 0.1923\) mV per MaleCNS synapse, test

\[
\boxed{w_\text{MaleCNS}=0.275\times 0.581\approx0.1598\ {\rm mV/synapse}}
\]

as the first principled baseline.

**I would not expect that change alone necessarily to solve the full-CNS problem**, because the 0.581 factor was derived for dataset alignment, not for dynamical stability, and the VNC has no FlyWire counterpart. But it is a substantially better starting point than 1/1.43, and your screenshot suggests you may already be sitting quite close to the dynamical transition: reducing the weight has greatly reduced active-neuron count and overlap without yet eliminating the persistent state. In recurrent networks, another ~17% reduction can plausibly move the system across that transition.

## What your screenshot is actually telling us

Your experiments already contain an important localization clue.

At full or stronger weighting, you were seeing roughly 15,000–19,000 active neurons. After scaling, the median falls to **10,825** under your Shiu convention. Removing monoamines reduces it further to **8,107**. Likewise, overlap drops from 0.78 to 0.49 and then to 0.43.

That means both interventions **are changing network gain**. They are not failing completely.

But the decisive statistic is:

> **14 of 18 stimuli still remain active after the stimulus has ended.**

That number is unchanged by monoamine removal. The persistent population only gets smaller, from roughly 9,500 neurons to ~6,500–7,200 neurons. Thus the perturbations are reducing the size of the attractor without eliminating its basin.

That is exactly what one expects when recurrent gain is still on the wrong side of a stability boundary.

Your clean stimuli are particularly informative. Sugar, salt and water reportedly activate only ~220–1,300 neurons and decay to zero. So the network is **not spontaneously unstable from its resting state**. Instead:

\[
\text{resting state}
\xrightarrow{\text{sufficient perturbation}}
\text{high-activity recurrent state}
\]

and some inputs cross that separatrix while others do not.

This distinction matters. If the network fired 10,000 neurons with no input from \(t=0\), I would first suspect a sign, initialization or numerical-integration bug. Your observation that some stimuli remain clean while others trigger persistent activity is much more characteristic of a **stimulus-triggered attractor/bifurcation**.

There is striking independent evidence for the same behavior. A separate MaleCNS implementation of the Shiu model reports that a sweet-taste stimulus drove the fraction of spikes occurring in the VNC from **28.6% in the first 100 ms to 66.9% in the last 100 ms**. Of 85 abdominal neurons exceeding 100 Hz near the end, 69 were in abdominal neuromeres and 55 were specifically in A7–A10. citeturn22view0turn22view4

More importantly, when that implementation switched the sensory stimulus off after 500 ms, **25 of 30 trials remained in the high-VNC state**. Silencing outgoing synapses from all 2,156 abdominal VNC neurons prevented the VNC takeover; meanwhile brain activity itself fell substantially after stimulus removal. citeturn22view0turn22view1

That independent result is unusually close to your “14/18 words stay on after input ends” phenotype.

I would therefore treat the hypothesis

\[
\boxed{\text{the persistent state is primarily maintained by recurrent VNC circuitry}}
\]

as your leading explanation until disproved.

## Why the FlyWire parameters do not transfer cleanly

Shiu et al.'s published model is deliberately simple: a leaky integrate-and-fire neuron for every included neuron, signed connectome connectivity, and one fitted synaptic-weight parameter. The paper demonstrated that such a model can recover meaningful sensorimotor relationships from FlyWire. citeturn2view1

The reference constants are:

| Quantity | Shiu reference |
|---|---:|
| Rest potential | −52 mV |
| Reset potential | −52 mV |
| Threshold | −45 mV |
| Membrane time constant | 20 ms |
| Synaptic time constant | 5 ms |
| Refractory period | 2.2 ms |
| Synaptic delay | 1.8 ms |
| Weight per anatomical synapse | 0.275 mV |

These values are directly visible in the authors' code. citeturn20view0turn21view5

The critical point is that **0.275 mV is a fitted effective parameter, not an independently measured universal conductance of a Drosophila synapse**. Consequently, “one EM-counted MaleCNS synapse = one FlyWire synapse × the same 0.275 mV” is not automatically justified when the reconstruction pipeline and synapse-count statistics change.

The datasets differ substantially. The independent MaleCNS/Shiu port reports approximately **127,400 neurons and 14.69 million directed edges in its FlyWire v630 pack**, versus **166,700 neurons and 24.47 million edges in MaleCNS v1.0**; it explicitly warns that Shiu's constants were fitted to FlyWire and that spike counts should not simply be compared between the packs. citeturn21view1turn21view2 MaleCNS itself provides the complete segment-to-segment connection-strength graph and separate neurotransmitter annotations. citeturn21view3turn21view4

The difference is not captured well by a single “MaleCNS has 1.43× more synapses” argument. For the two packs above, total directed-edge count is about 1.67× larger, while directed edges per neuron are about 1.27× larger. More importantly, **stability is controlled by the strongest recurrent loops and the distribution of incoming/outgoing weights, not by the average density alone**.

The MaleCNS paper provides a much more relevant empirical clue. For matched male/female central-brain comparisons, the project website reports:

> `weight (M, scaled)` = male CNS connection weight multiplied by **0.581** to align it with FlyWire.

citeturn25search0turn25search6

The associated paper describes deriving that factor from systematic differences between datasets and applying it to MaleCNS edge weights for cross-dataset comparison. citeturn26search1turn29search1

That does **not** mean 0.581 is guaranteed to be the correct physiological LIF gain. It does mean that if your goal is to carry a parameter fitted to FlyWire synapse counts onto MaleCNS synapse counts, **0.581 is a substantially better first-order normalization than 0.699**.

There is a second transfer problem: FlyWire is brain-only in this comparison, whereas MaleCNS contains the brain **plus the ventral nerve cord**. citeturn21view1turn21view2 Thus the Shiu model's fitted gain never had to stabilize the MaleCNS recurrent brain↔VNC and VNC↔VNC loops.

That is probably why a global weight reduction helps but does not completely fix your simulations.

## The highest-priority implementation checks

Before tuning any more biology, I would run an exact **Shiu parity audit**. A subtle simulator discrepancy could otherwise make every later gain correction misleading.

### Make sure `g` is reset on every postsynaptic spike

This is the most important code-level check.

The published implementation evolves

\[
\frac{dv}{dt} = \frac{v_0-v+g}{\tau_m},
\qquad
\frac{dg}{dt}=-\frac{g}{\tau_s},
\]

and on a spike executes:

```text
v = v_rst
g = 0
```

with the source code actually specifying `v = v_rst; ...; g = 0*mV`. citeturn21view5

Synaptic events then execute

```text
g += w
```

after a 1.8-ms delay. citeturn21view6

A conventional LIF implementation often resets only \(v\) and allows synaptic current/conductance to continue decaying. **That is not Shiu's implementation.**

If yours does this:

```python
if spike:
    v[i] = V_RESET
    # g[i] unchanged
```

change it to the Shiu behavior:

```python
if spike:
    v[i] = V_RESET
    g[i] = 0.0
```

This should be checked before any weight fitting.

I would test it with a tiny deterministic circuit rather than the whole connectome: one presynaptic neuron, one postsynaptic neuron and a predefined train of spikes. Verify the `g` trajectory and exact postsynaptic spike times against Brian2.

### Verify that MaleCNS edge strengths are counted once

Shiu creates **one connection for each pre/post edge** and then multiplies its signed anatomical connectivity value by \(w_\text{syn}\). citeturn21view6

MaleCNS's downloadable `connectome-weights` table is already a **segment-to-segment connection-strength graph**. citeturn21view4

Therefore, a potentially catastrophic conversion bug would be doing both:

1. creating one simulated edge per anatomical synapse, **and**
2. assigning each one the aggregate pre→post connection count.

That squares the effective multiplicity.

For every pair \(j\rightarrow i\), you want either:

\[
\text{one simulated edge with }w_{ji}=N_{ji}w_0
\]

or:

\[
N_{ji}\text{ individual edges, each of weight }w_0,
\]

**not both**.

Also check that autapses, duplicates produced by ROI joins, and polyadic partner rows are not accidentally counted multiple times. The MaleCNS documentation explicitly notes that an individual presynaptic site can connect to multiple postsynaptic sites, so care is needed when constructing the graph from lower-level synapse tables. citeturn21view4

### Verify the remaining Shiu timing semantics

Your port should reproduce all of the following simultaneously:

\[
\tau_m=20{\rm ms},\quad
\tau_s=5{\rm ms},\quad
t_{\rm ref}=2.2{\rm ms},\quad
d=1.8{\rm ms}.
\]

citeturn20view0turn21view5

A zero-delay or one-timestep-delay recurrent implementation is not equivalent. With strongly recurrent circuits, timing changes can substantially alter synchronization and the threshold for a self-sustaining state.

Also note that Shiu's Poisson stimulation is unusual: the Poisson input is delivered directly to the target neuron's membrane variable, and stimulated neurons have their refractory period set to zero. citeturn21view5 So for parity testing, reproduce that behavior rather than implementing sensory drive as an ordinary weak synapse.

## The fix I would implement

I would not keep adding stronger APL inhibition globally. Instead I would make the correction in stages, because each stage answers a different scientific question.

### Start with the MaleCNS-to-FlyWire weight correction

Your current effective synaptic scale is:

\[
w_{\rm current}
=
0.275/1.43
=
0.1923\ {\rm mV}.
\]

Test:

\[
\boxed{
w_{\rm base}=0.275\times0.581
=0.1598\ {\rm mV}
}
\]

for every ordinary fast MaleCNS connection.

That is another **16.9% reduction relative to your present setting**.

In code, conceptually:

```python
SHIU_W = 0.275       # mV / FlyWire-counted synapse
MALE_TO_FLYWIRE = 0.581

w0 = SHIU_W * MALE_TO_FLYWIRE   # 0.159775 mV

weight_ij = nt_sign[j] * synapse_count_ij * w0
```

The 0.581 normalization has an empirical MaleCNS/FlyWire basis; the MaleCNS project itself uses it when aligning male edge weights with FlyWire. citeturn25search0turn26search1

Because the model is evidently near a threshold, do not judge this from mean spike counts alone. Judge it primarily from **whether activity dies after input removal**.

### Then separate brain and VNC recurrence

This is the experiment I expect to be most diagnostic.

Run each troublesome word under these four conditions:

| Condition | What it tests |
|---|---|
| Full MaleCNS | Current phenotype |
| Brain only / remove VNC nodes | Whether FlyWire-style brain dynamics are stable |
| Full CNS but set VNC→VNC recurrent weights to zero | Whether the cord is maintaining the attractor |
| Full CNS but only abdominal-VNC outgoing weights set to zero | Tests the strongest independent MaleCNS hypothesis |

The reason to prioritize this is unusually strong: the independent Shiu-on-MaleCNS implementation found the persistent state concentrated in abdominal VNC, and silencing outgoing connections from the abdominal population prevented the VNC takeover. citeturn22view0turn22view1

If your 14 problematic words suddenly decay normally when VNC→VNC recurrence is removed, you have essentially identified the problem.

At that point, **do not globally keep weakening the brain**. That risks destroying exactly the sensory-to-MN9 transformations you are trying to preserve.

Instead introduce a VNC recurrent gain:

\[
w_{ij}'=
\begin{cases}
0.581\,w_{ij}^{\rm Shiu}, & \text{brain / brain↔VNC}\\[4pt]
\alpha_{\rm VNC}\,0.581\,w_{ij}^{\rm Shiu},
& i,j\in{\rm VNC}.
\end{cases}
\]

Then find the **largest** \(\alpha_{\rm VNC}\le1\) for which every unwanted post-stimulus state decays.

I would determine \(\alpha_{\rm VNC}\) by binary search rather than choosing an arbitrary inhibition strength.

For example:

```text
alpha_VNC = 1.00
           0.75
           0.875
           0.8125
           ...
```

and optimize against a stability criterion rather than against the number of active neurons during the stimulus.

This preserves as much connectomic gain as possible while eliminating the pathological basin.

### Separate excitatory and inhibitory VNC gain if needed

If scaling all VNC recurrence weakens motor output too much, do the next refinement:

\[
w_E'=\alpha_{E,r}w_E,
\qquad
w_I'=\alpha_{I,r}w_I.
\]

Keep inhibition at the dataset-corrected baseline initially,

\[
\alpha_{I,r}=1,
\]

and reduce only recurrent excitation:

\[
\alpha_{E,\rm VNC}<1.
\]

This is preferable to inserting a fictitious global inhibitory neuron because it changes the gain of the actual recurrent loops responsible for the problem.

APL in particular is not a whole-CNS stabilizer. The APL neuron is a mushroom-body feedback-inhibitory neuron acting on Kenyon cells, and experimentally its activity and inhibitory effects are spatially localized within mushroom-body compartments. citeturn20view3 So adding “one APL inhibitory feedback” may sparsify a mushroom-body-like population but has no anatomical reason to neutralize an attractor sustained in abdominal VNC circuitry.

That explains why your APL experiment did not cure the phenotype.

### Keep monoamines out of the fast signed matrix for now

Your monoamine-removal result is useful: it reduces persistent population size, but does not change the fact that **14/18 words remain persistent**.

I would therefore keep your “monoamines removed” configuration as a conservative baseline while solving the stability problem, but I would not treat monoamines as the root cause.

The MaleCNS dataset provides neurotransmitter predictions separately from its connectivity graph. citeturn21view4 A Shiu-style LIF model reduces neurotransmission to an instantaneous signed synaptic increment, whereas monoaminergic effects are not naturally represented by a single universal fast ± sign. Thus, excluding them from the **fast** synaptic matrix is a reasonable modeling control until you explicitly implement receptor-dependent neuromodulation.

Your data already show that this alone cannot repair the recurrent instability.

## A more principled normalization if the global factor is insufficient

There is a deeper reason a single global multiplier can fail.

Suppose MaleCNS has two neurons:

- neuron A receives approximately the same recurrent excitation as its FlyWire homolog;
- neuron B receives two or three times as much effective recurrent excitation.

A global 0.581 multiplier may make A underactive while B remains supercritical.

The MaleCNS project provides cross-brain cell-type mappings and explicitly shows male versus FlyWire connection weights for matched cell types. citeturn25search0turn25search6 That enables a much better normalization strategy: **match effective incoming drive by homologous cell type**.

For each MaleCNS postsynaptic neuron \(i\), let

\[
E_i^{MC}=\sum_j \max(0,W_{ji}^{MC}),
\]

and for its matched FlyWire cell type \(T(i)\), estimate a typical FlyWire excitatory input

\[
E_{T(i)}^{FW}.
\]

Then use a postsynaptic gain such as

\[
s_i=
\frac{E_{T(i)}^{FW}}
     {E_i^{MC}}
\]

with sensible clipping, and apply

\[
W_{ji}'=0.275\ {\rm mV}\times
N_{ji}\times {\rm sign}_j\times s_i.
\]

I would actually compute separate factors for excitation and inhibition,

\[
s_i^E=
\frac{E_{T(i)}^{FW}}{E_i^{MC}},
\qquad
s_i^I=
\frac{I_{T(i)}^{FW}}{I_i^{MC}},
\]

because preserving E/I structure is preferable to matching only total degree.

This would change the question from

> “How much denser is MaleCNS on average?”

to

> “How much effective recurrent drive does this particular MaleCNS neuron receive compared with the FlyWire neuron/type on which the Shiu parameterization was calibrated?”

That is much closer to the quantity controlling LIF stability.

For unmatched VNC neurons, there is no FlyWire homologous VNC population, so I would use region-specific normalization derived from the stability experiment rather than pretending FlyWire supplies a reference.

In practical terms, the final parameterization might look like:

\[
\boxed{
W_{ji}
=
0.275\,{\rm mV}
\times0.581
\times N_{ji}
\times{\rm sign}_j
\times s_i^{E/I}
\times \alpha_{r(j),r(i)}
}
\]

where:

- **0.581** corrects the MaleCNS/FlyWire systematic edge-weight difference;
- \(s_i^{E/I}\) corrects cell-specific or cell-type-specific incoming drive;
- \(\alpha_{r(j),r(i)}\) is needed mainly for CNS regions, especially VNC recurrence, absent from the FlyWire calibration.

I would add those terms in that order, not simultaneously, so you can identify which correction actually removes the attractor.

## Validation protocol and the result I would expect

The key change I would make to your evaluation is to make **post-stimulus decay the primary objective**, rather than active-neuron count.

For every word, run at least:

\[
300\text{–}500\ {\rm ms\ pre}
\rightarrow
200\text{–}500\ {\rm ms\ stimulus}
\rightarrow
500\text{–}1000\ {\rm ms\ no\ input}.
\]

Then calculate separately for brain, VNC and abdominal VNC:

\[
N_{\rm active}(t),
\qquad
R(t)=\text{population spike rate},
\qquad
R_E(t),
\qquad
R_I(t).
\]

Also calculate the ratio

\[
G(t)=
\frac{\text{recurrently generated spikes}}
     {\text{spikes in preceding time window}},
\]

or a comparable perturbation-amplification measure.

Operationally, you want:

\[
G<1
\]

after external drive ends, so perturbations contract rather than grow. I would treat this as an empirical stability test rather than relying only on the spectral radius of the anatomical matrix, because thresholding, refractory dynamics and signed connections make the LIF system nonlinear.

### The decisive ablation experiment

For one word that reliably triggers your 6,000–9,500-neuron persistent state:

1. stimulate normally until the high state develops;
2. stop sensory stimulation;
3. at that exact moment, zero only VNC→VNC outgoing recurrent weights.

If the high state collapses immediately, the VNC is maintaining it.

Repeat with:

- abdominal VNC only;
- brain→VNC only;
- VNC→brain only;
- excitatory VNC→VNC only.

That identifies the recurrent block responsible for the eigenmode/attractor far more efficiently than trying dozens of global inhibitory-feedback constants.

The independent MaleCNS reproduction suggests **abdominal VNC outgoing recurrence is the first block to test**. citeturn22view0turn22view1

### Preserve the behavioral readout while stabilizing

Your existing MN9 results give you a valuable constraint. Do not simply turn the gain down until everything becomes silent.

At each candidate gain, track both:

\[
\text{post-stimulus residual activity}
\]

and

\[
\text{stimulus-specific MN9 response}.
\]

Your desired regime looks conceptually like this:

| Stimulus | During stimulus | After stimulus |
|---|---|---|
| Sugar | appropriate sensory population + MN9 feeding response | rapid decay |
| Salt | expected MN9 response | rapid decay |
| Bitter | no inappropriate MN9 feeding | rapid decay |
| Hot | low/nonfeeding MN9 response | rapid decay |
| No stimulus | quiescent | quiescent |

The solution is therefore **the largest recurrent gain that preserves stimulus discrimination while making unwanted persistent states subcritical**, not the gain producing the fewest spikes.

### A parameter sweep I would run first

Given your current results, this is the most informative small sweep:

| MaleCNS scale relative to Shiu 0.275 mV | Effective mV/synapse | Purpose |
|---:|---:|---|
| 0.699 | 0.192 | Your current 1/1.43 condition |
| **0.581** | **0.160** | MaleCNS paper's FlyWire-alignment starting point |
| 0.55 | 0.151 | slightly below alignment |
| 0.50 | 0.138 | stability test |
| 0.45 | 0.124 | determines whether global gain alone can eliminate attractor |

For each value, record:

\[
\{\text{persistent words}/18,\,
N_{\rm silent},\,
\text{word overlap},\,
MN9_{\rm sugar},\,
MN9_{\rm bitter},\,
MN9_{\rm hot},\,
MN9_{\rm salt}\}.
\]

The shape of that curve will tell you a great deal.

If persistence changes sharply, for example:

\[
14/18\rightarrow12/18\rightarrow3/18\rightarrow0/18,
\]

while MN9 remains sensible, you simply crossed a recurrent-gain bifurcation.

If persistence survives even at 0.45–0.50, but disappears with VNC recurrence removed, then the right fix is **region-specific VNC scaling**, not further whole-network attenuation.

If persistence remains even after VNC recurrence removal, return to the implementation audit—particularly `g` reset, edge duplication and neurotransmitter signs.

## Recommended configuration

Putting all of the evidence together, my recommended next version is:

\[
\boxed{
\textbf{Shiu dynamics exactly}
+
\textbf{MaleCNS weight correction}
+
\textbf{region-specific VNC stabilization}
}
\]

More concretely:

```text
Membrane:
    Vrest       = -52 mV
    Vreset      = -52 mV
    Vthreshold  = -45 mV
    tau_m       = 20 ms

Synapse:
    tau_g       = 5 ms
    delay       = 1.8 ms
    refractory  = 2.2 ms

On presynaptic event:
    g_post += weight

On postsynaptic spike:
    v = -52 mV
    g = 0                     # essential Shiu behavior

Base MaleCNS weight:
    0.275 mV × 0.581
    = 0.159775 mV per MaleCNS synapse

Fast transmitter model:
    keep clearly fast signed transmitters
    leave monoamines out initially

Then:
    diagnose VNC→VNC recurrence
    reduce VNC excitatory recurrent gain only if required
```

The reference equations and reset behavior come directly from Shiu's public implementation. citeturn21view5turn21view6 The 0.581 factor comes from the MaleCNS/FlyWire comparison rather than an ad hoc density ratio. citeturn25search0turn26search1 The recommendation to localize the remaining correction to VNC recurrence is supported by an independent implementation of exactly this model/connectome combination that found persistent post-stimulus activity concentrated in abdominal VNC and strongly reduced by abdominal-VNC silencing. citeturn22view0turn22view1

**My strongest prediction is that the fix is not “more APL.”** Your screenshot shows APL/monoamine/global-gain manipulations reducing the size of the pathological state without removing it. APL itself is a localized mushroom-body inhibitory mechanism, not a whole-CNS global stabilizer. citeturn20view3 The most likely solution is instead:

\[
\boxed{\text{verify Shiu's }g\text{-reset}
\;\rightarrow\;
\text{replace }1/1.43\text{ with }0.581
\;\rightarrow\;
\text{identify and downscale recurrent VNC excitation}}
\]

That sequence has the best combination of fidelity to Shiu, fidelity to the MaleCNS dataset, and direct empirical support for the exact failure mode visible in your screenshot.