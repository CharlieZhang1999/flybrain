# Locating and Breaking the MaleCNS Antennal-Lobe Attractor

## What your last experiment is actually telling you

The most important result is not simply that “the antennal lobe is active.” Your analysis has narrowed the persistent state to a much more specific candidate mechanism:

> **A recurrent excitatory network dominated by cholinergic antennal-lobe local interneurons, with projection neurons participating in the loop.**

That is biologically plausible as a *location* for strong recurrent interactions: the antennal lobe is the first major olfactory relay, and Drosophila has both inhibitory and excitatory local interneurons connecting olfactory channels. Cholinergic excitatory local interneurons have been demonstrated experimentally, while GABAergic and glutamatergic local interneurons provide important inhibitory interactions. citeturn2view3turn4search0turn4search2

But there is an important distinction:

**Your experiment does not yet prove that lLN1_bc, or any other named cell type, is the causal source of the runaway state.**

It tells you which neurons sit most strongly inside the candidate recurrent mode. That is exactly what the lesion stage now needs to test.

### A beginner's interpretation of each row

| Your result | Beginner interpretation |
|---|---|
| **lLN1_bc — acetylcholine — 15 of top 200** | A surprisingly large fraction of the neurons most strongly associated with the runaway circuit are the same type of excitatory antennal-lobe local neuron. This makes lLN1_bc your strongest *family-level* suspect. |
| **lLN2T_a, lLN2X04, lLN2X05, lLN2X12, lLN1_a — acetylcholine** | Other excitatory local interneurons are also embedded in the same candidate circuit, suggesting this may not be a one-cell-type problem. |
| **DA1_lPN, DL2d_adPN, VC3_adPN, etc. — acetylcholine** | Projection neurons are participating too. They normally carry olfactory information onward from the antennal lobe, but in your model they may also be part of feedback paths that help re-excite the local circuit. Olfactory PNs are a major output route from the antennal lobe. citeturn0search2turn11view3 |
| **lLN2P_a–c, lLN2R_a — GABA** | The model contains inhibitory local neurons in the same region, but their negative influence is apparently not sufficient to stop the persistent state once it begins. |
| **v2LN30 — transmitter unclear — 3,626 mV** | This neuron is potentially extremely important, but you currently do **not know the sign of its effect reliably enough to treat it as excitatory**. It should be isolated as a separate uncertainty experiment rather than lumped together with confirmed cholinergic neurons. |

The finding that v2LN30 is an antennal-lobe local neuron is consistent with independent connectomic work; it has been classified as an ALLN, an antennal-lobe local neuron. citeturn11view3 That does **not**, however, establish that its MaleCNS transmitter assignment is excitatory. This distinction matters because neurotransmitter predictions from EM are statistical predictions rather than direct physiological measurements, and recent analyses often impose confidence cutoffs before using such predictions. citeturn2view3turn11view3

### What “highest eigenvector loading” means

A useful beginner analogy is to imagine that the runaway activity is a song being sung by a choir.

Some neurons barely participate in that song.

Other neurons sing very loudly and are tightly synchronized with the pattern.

An **eigenvector loading** measures something like:

> “How strongly does this neuron belong to this particular network-wide feedback pattern?”

So when lLN1_bc cells have exceptionally high loadings, that means their connections are heavily represented in the recurrent pattern you identified.

This is a good method for generating lesion candidates. Pospisil and colleagues explicitly interpret high-loading neurons in a connectome eigenvector as neurons whose connectivity helps define a putative dynamical “eigencircuit.” But they also emphasize that the signed connectome is only an approximation to the real dynamical effectome; the true neuronal Jacobian depends on neuronal state, synaptic physiology and other properties that the anatomical connectome does not contain. citeturn10search1turn11view4

So:

**High eigenvector loading = strong suspect.**

It does **not** mean:

**High eigenvector loading = proven causal driver.**

That is exactly why the next stage should be lesions.

## What the very large “mV output” means

This part is easy to misinterpret.

When you say an lLN1_bc neuron sends approximately **1,000–1,500 mV of excitation into the candidate set**, that does **not** mean one postsynaptic neuron is suddenly receiving a one-volt depolarization.

It is an **aggregate network quantity**: you are summing the modeled excitatory weights from that neuron across many downstream connections in the candidate population.

Under the original Shiu convention, the single free synaptic-weight parameter is \(W_{\rm syn}=0.275\) mV per anatomical synaptic contact. The model's resting potential is −52 mV and spike threshold is −45 mV, a difference of only 7 mV; incoming synaptic events are filtered by the model's synaptic and membrane dynamics rather than simply being added instantaneously to one cell. citeturn3view0

For scale, **if** those totals were calculated using the unmodified 0.275-mV Shiu conversion:

| Aggregate excitation | Rough equivalent anatomical contacts |
|---|---:|
| 1,000 mV | ~3,636 |
| 1,500 mV | ~5,455 |
| v2LN30's 3,626 mV | ~13,185 |

Those are only “synapse equivalents” under the original Shiu coefficient; because you have already rescaled MaleCNS weights, they should not be interpreted as your literal MaleCNS synapse counts.

What matters is the structural picture:

```text
                 EXCITATION
        ┌─────────────────────────┐
        │                         ↓
 Cholinergic LN ────────→ other cholinergic LNs
        │                         │
        │                         │
        ↓                         │
 Projection neurons ──────────────┘
        │
        ↓
 downstream brain
```

If the return pathways are strong enough, stimulation only has to **start** the loop.

After that:

```text
LN fires
   ↓
PN / other LNs fire
   ↓
they excite recurrent partners
   ↓
original LN population is activated again
   ↓
repeat
```

That explains your peculiar observation that many completely different stimuli eventually produce a similar persistent population.

They do not necessarily have to encode the same sensory information.

They merely have to push the network across the same dynamical threshold.

This interpretation also fits an important limitation of the Shiu model. It treats each neuron as globally excitatory or inhibitory, ignores morphology and receptor dynamics, omits gap junctions and neuromodulation, and starts every neuron with zero baseline firing. The authors explicitly caution that absolute firing rates from this simplified model should not be taken literally. citeturn3view5

That matters enormously when moving from FlyWire to a different, denser connectome.

## Why your previous APL intervention probably did not solve this

Your earlier experiment added or strengthened **APL inhibition**.

Now that the runaway circuit has been localized to the antennal lobe, there is a strong reason that intervention could fail even if it added a lot of inhibitory activity.

APL is primarily the giant inhibitory feedback neuron of the **mushroom body**, particularly the Kenyon-cell system. Experiments show that APL provides localized GABAergic feedback within the mushroom body rather than functioning as the principal inhibitory controller of antennal-lobe LN recurrence. citeturn13search0turn13search3

In other words, you may have been doing something roughly equivalent to this:

```text
PROBLEM

Antennal lobe
   E → E → E
   ↑       │
   └───────┘
       │
       ↓
Projection neurons → Mushroom body
                         │
                       APL
                         └── inhibition
```

You strengthened a brake **downstream of the source of the instability**.

That can reduce downstream activity without necessarily stopping the AL feedback engine itself.

Your new candidate analysis gives you a much better intervention point:

```text
Antennal lobe
   E ──→ E
   ↑     │
   └─────┘
     ↑
     │
  intervene HERE
```

This is one reason I would **not add another global inhibitory mechanism yet**. First determine exactly which recurrent edges are necessary.

## The lesion stage I would run

The most important methodological change is:

> **Do not begin by deleting every interneuron→interneuron or interneuron→projection-neuron connection.**

That experiment is useful eventually, but it is too coarse as the first lesion. If it works, you will know that antennal-lobe recurrence matters but will not know **which component was pathological**.

Instead, use a progressively narrower causal-lesion ladder.

### Establish a fixed readout before lesioning

Use exactly the same stimulus set and simulation duration for every condition. Your most informative readout remains activity during the **silent final 300 ms**, because there is no external stimulus during that interval.

For every condition I would save at least:

| Metric | Why it matters |
|---|---|
| Number of stimuli persistent out of 18 | Primary runaway measure |
| AL neurons active in final 300 ms | Direct measure of the candidate attractor |
| Whole-brain neurons active in final 300 ms | Whether stopping AL recurrence stops the global state |
| Median firing rate of lLN1_bc | Direct candidate measurement |
| Median firing rate of all cholinergic AL LNs | Whether the excitatory LN population remains regenerative |
| Median firing of GABAergic AL LNs | Whether inhibition is being recruited |
| Stimulus-period activity | Ensures the lesion has not simply killed olfactory responsiveness |
| Word-to-word population overlap | Tests whether stimulus specificity is restored |
| MN9 or other task output | Ensures the original Shiu-style sensorimotor behavior survives |

Shiu's own computational-silencing experiments used the conceptually similar intervention of eliminating all output from individual candidate neurons while sensory neurons remained stimulated. citeturn3view5

Also add a **no-stimulus control**. Because the Shiu model has a 0-Hz baseline, a no-input simulation should remain silent unless your implementation initializes or externally drives activity differently. citeturn2view0turn3view5

### Start with the lLN1_bc family

This should be the first causal experiment because your own ranking gives it the strongest combination of evidence:

**15 representatives among the top 200 + highest loading + very large excitatory output.**

Do not physically delete the neurons initially.

Instead:

> **Set all outgoing synaptic weights of lLN1_bc neurons to zero.**

Keep their incoming edges.

That distinction is very useful.

The neurons can still tell you:

> “The rest of the network is trying to activate me.”

But they can no longer tell downstream neurons:

> “Fire.”

So your first experiment becomes:

```text
lLN1_bc receives input normally
          ↓
       can fire
          ↓
          X    outgoing synapses blocked
```

This is cleaner than removing the cell completely because the network topology upstream of the neuron remains intact.

Your decisive comparison is:

| Condition | Prediction if lLN1_bc is causal |
|---|---|
| Baseline | ~14–15/18 persistent |
| lLN1_bc outputs = 0 | persistence should collapse strongly |
| Stimulus-period AL response | should remain at least partly present |
| Final 300 ms | should approach zero or a small transient remainder |

If persistence falls from, say, 15/18 to 1–3/18 while sensory responses survive, **that is an extremely strong causal result**.

If it remains 14–15/18, lLN1_bc is prominent inside the attractor but is not individually necessary.

### Then split the cholinergic-LN population

Next test the remaining candidate cholinergic LN types as a group:

```text
lLN2T_a
lLN2X04
lLN2X05
lLN2X12
lLN1_a
...
```

The antennal lobe genuinely contains excitatory local circuits; electrophysiological and anatomical studies have identified excitatory local interneurons and excitatory interactions between olfactory channels. citeturn4search0turn4search12

Use three conditions:

| Lesion | Interpretation |
|---|---|
| lLN1_bc only | Is the dominant family necessary? |
| Other cholinergic AL LNs only | Is the instability distributed across several LN families? |
| All cholinergic AL LNs | Is excitatory local recurrence as a class necessary? |

This gives much more information than immediately deleting every AL interneuron connection.

### Treat v2LN30 as its own uncertainty experiment

I would move **v2LN30 near the top of the lesion queue**, but I would *not* automatically classify it as excitatory.

It deserves special treatment because your candidate analysis says it has the largest individual excitatory output, yet the transmitter sign is coming from a default rather than a confident assignment.

That is dangerous in this model.

The Shiu simulation converts neurotransmitter identity directly into the **sign of every output edge from the neuron**. Therefore a transmitter-sign mistake on one highly connected neuron does not create a small local error: it can reverse thousands of modeled synaptic effects. citeturn3view0turn3view2

Recent FlyWire work also explicitly treats neurotransmitter predictions probabilistically, and one 2026 study restricted analysis to fast transmitters with prediction scores above 62%, illustrating why uncertain predictions should not be silently converted into hard signs. citeturn11view3

Run:

| v2LN30 condition | What it asks |
|---|---|
| **Outputs = 0** | Is v2LN30 necessary at all? |
| **Outputs excitatory** | Reproduce your current assumption |
| **Outputs inhibitory** | What happens if the sign is wrong? |

This one experiment could be disproportionately informative.

If:

```text
Current excitatory assumption → massive persistence
v2LN30 = 0              → much less persistence
v2LN30 inhibitory       → persistence disappears
```

then I would stop treating the problem as primarily a “MaleCNS is denser” scaling problem.

It would instead point toward a **neurotransmitter-sign assignment problem involving a high-degree local neuron**.

## The edge-specific gain experiment I recommend

After the cell-type lesions, your suggestion to manipulate **interneuron→interneuron** and **interneuron→projection-neuron** edges is exactly the right next stage.

But I would **scale them gradually rather than delete them immediately**.

Define two parameters:

\[
\alpha_{LL}
\]

for:

**cholinergic antennal-lobe local neuron → antennal-lobe local neuron**

and:

\[
\alpha_{LP}
\]

for:

**cholinergic antennal-lobe local neuron → projection neuron.**

Then transform only those edge classes:

\[
W'_{LN\rightarrow LN}=\alpha_{LL} W_{LN\rightarrow LN}
\]

\[
W'_{LN\rightarrow PN}=\alpha_{LP} W_{LN\rightarrow PN}.
\]

Everything else remains unchanged.

### The first gain sweep

Run:

| Condition | \(\alpha_{LL}\) | \(\alpha_{LP}\) |
|---|---:|---:|
| Baseline | 1.00 | 1.00 |
| Reduce LN→LN slightly | 0.75 | 1.00 |
| Reduce LN→LN strongly | 0.50 | 1.00 |
| Reduce LN→LN heavily | 0.25 | 1.00 |
| Remove LN→LN | 0 | 1.00 |
| Reduce LN→PN slightly | 1.00 | 0.75 |
| Reduce LN→PN strongly | 1.00 | 0.50 |
| Reduce LN→PN heavily | 1.00 | 0.25 |
| Remove LN→PN | 1.00 | 0 |
| Both halfway | 0.50 | 0.50 |

This is much better than immediately zeroing both pathways.

Why?

Suppose you discover:

```text
αLL = 1.00 → 15/18 persistent
αLL = 0.75 → 14/18
αLL = 0.50 → 5/18
αLL = 0.25 → 0/18
```

while changing \(\alpha_{LP}\) barely matters.

You have learned that the crucial engine is:

> **LN → LN recurrent excitation**

rather than:

> LN → PN amplification.

Conversely, this result:

```text
LN→LN reduction: little effect
LN→PN reduction: persistence collapses
```

would imply that the loop probably travels through the projection-neuron population:

```text
LN
 ↓
PN
 ↓
some return pathway
 ↓
LN
```

That is a completely different mechanism.

### Then run the small factorial grid

Once you know the rough transition region, run:

| \(\alpha_{LL}\) \\ \(\alpha_{LP}\) | 1.0 | 0.75 | 0.5 | 0.25 | 0 |
|---|---:|---:|---:|---:|---:|
| **1.0** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **0.75** | ✓ |  | ✓ |  |  |
| **0.5** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **0.25** | ✓ |  | ✓ |  |  |
| **0** | ✓ |  | ✓ |  |  |

You do not actually need every combination initially.

You are looking for a **boundary** between two regimes:

```text
          persistent
              │
              │
──────────────┼──────── critical boundary
              │
              │
            stable
```

The most interesting solution is not \(\alpha=0\).

The interesting solution is the **largest gain that restores stability while leaving stimulus-evoked olfactory activity intact**.

That gives you a calibration parameter rather than an arbitrary lesion.

## Controls that will tell you whether you found the actual mechanism

Three controls are especially important.

### GABAergic-LN lesion as a negative control

Do **not** expect removing lLN2P_a–c or lLN2R_a to rescue the network if they are functioning as inhibitory neurons.

In fact, if those GABAergic LNs really oppose the runaway circuit, removing them should either:

**increase persistence**, increase firing, or leave the phenotype roughly unchanged if their modeled inhibition is already overwhelmed.

The antennal lobe is known to use GABAergic inhibition for lateral and gain-control computations, and connectome-wide analyses likewise find inhibitory neurons frequently incorporated into feedback motifs. citeturn4search1turn13search14

Thus this is a valuable causal control:

```text
Remove excitatory LN output
        ↓
persistence improves

Remove inhibitory LN output
        ↓
persistence worsens
```

That would strongly support an excitation-versus-inhibition explanation.

### Projection-neuron-only lesion

Do not eliminate PNs entirely. Instead eliminate only their **return output into the candidate attractor population**.

That distinguishes:

```text
LN → PN → higher brain
```

which is ordinary feed-forward olfactory output,

from:

```text
LN → PN → candidate AL neurons → LN
```

which is a recurrent feedback loop.

Olfactory PNs normally relay AL information to higher-order olfactory regions, so preserving their outward pathway while selectively blocking candidate feedback gives you a much cleaner mechanistic experiment. citeturn0search2turn11view3

### Add-back experiments

This is the most powerful final localization step.

Suppose:

> removing all cholinergic LN→LN edges fixes the network.

Now reverse the experiment.

Start from the stable lesioned network and restore:

```text
lLN1_bc → lLN1_bc
```

Then:

```text
lLN1_bc → other cholinergic LN
```

Then:

```text
other cholinergic LN → lLN1_bc
```

Then particular PN feedback classes.

You are asking:

> **Which edge set is sufficient to bring the attractor back?**

This gives you two complementary causal tests:

**Necessity**

```text
remove pathway
↓
attractor disappears
```

and **sufficiency**

```text
restore pathway
↓
attractor returns
```

That is much stronger evidence than eigenvector ranking alone.

## The exact experiment order I would use

Given everything you have found so far, this is the sequence I recommend:

| Priority | Experiment | Why |
|---:|---|---|
| **First** | lLN1_bc outgoing weights → 0 | Highest-confidence candidate family |
| **Second** | v2LN30 outputs → 0 | Huge candidate output and uncertain sign |
| **Third** | v2LN30 excitatory vs inhibitory | Directly tests a potentially catastrophic sign assumption |
| **Fourth** | All other cholinergic AL-LN outputs → 0 | Tests distributed excitatory-LN mechanism |
| **Fifth** | All cholinergic AL-LN outputs → 0 | Determines whether excitatory LNs are necessary as a class |
| **Sixth** | LN→LN gain sweep | Tests direct recurrent excitation |
| **Seventh** | LN→PN gain sweep | Tests PN-mediated amplification |
| **Eighth** | PN→LN candidate return-edge sweep | Tests whether PNs close the loop |
| **Ninth** | GABAergic-LN output lesion | Negative/control experiment |
| **Tenth** | Add back individual families/edge classes | Identifies minimal sufficient attractor circuit |

I would **not** globally reduce all excitatory synapses again before completing these experiments.

You already reduced the MaleCNS global weight based on its greater synaptic density and the attractor persisted. That is evidence that the problem may be **topological or cell-class-specific rather than simply a global synapse-count scale factor**. That conclusion is an inference from your results, rather than something established by the published MaleCNS data.

There is also a deeper reason to prefer targeted scaling. Shiu's \(W_{\rm syn}=0.275\) mV was a free parameter fitted in the FlyWire network by choosing a value that made 100-Hz sugar-GRN stimulation produce roughly 80% of maximal MN9 firing. It was **not derived from a universal biological conversion saying that every anatomical synapse in every connectome should equal 0.275 mV**. citeturn3view0

That point may ultimately explain why simply copying the FlyWire scalar into MaleCNS—even after a density correction—does not guarantee equivalent network dynamics.

## What outcome would convince me that you have found the fix

The goal is **not merely to reduce the number of neurons firing**.

A good repair needs to satisfy several things simultaneously.

Consider three possible outcomes.

### Bad fix: just suppress everything

```text
Before:
stimulus → 19,000 neurons → persistent

After:
stimulus → 400 neurons → 300 neurons persistent
```

This looks better numerically, but if sugar no longer activates the appropriate downstream circuit or MN9, you have merely made the network too weak.

### Bad fix: smaller attractor

```text
Before:
stimulus → 19,000 → 9,000 persistent

After:
stimulus → 8,000 → 3,000 persistent
```

That is still an attractor.

It is just smaller.

### Good fix

You want something qualitatively more like:

```text
                   stimulus ON             stimulus OFF

sugar        → specific population ─────→ decay toward baseline
water        → different population ─────→ decay toward baseline
bitter       → different population ─────→ decay toward baseline
hot          → different population ─────→ decay toward baseline
```

with the appropriate behavioral-output differences remaining.

The best lesion/gain condition would therefore show:

| Property | Desired outcome |
|---|---|
| Persistent words | near 0/18, unless there is a biologically justified persistent response |
| Final silent-period AL activity | collapses strongly |
| Word-to-word overlap | substantially decreases |
| Stimulus-period responses | remain present |
| Sugar → MN9 | preserved |
| Bitter → MN9 | remains suppressed under your validated convention |
| Different stimuli | produce meaningfully different populations |
| No-stimulus run | remains silent |
| lLN1_bc population | stops self-recruiting after stimulus offset |
| GABAergic LNs | need not be artificially amplified |

The critical conceptual shift is this:

> **Stop asking “How much should I globally weaken MaleCNS?” and start asking “Which recurrent antennal-lobe edge class makes the effective loop gain exceed the self-sustaining threshold?”**

Your present results point first to **cholinergic LN→LN recurrence**, second to a possible **LN→PN→LN loop**, and separately to **v2LN30's uncertain transmitter sign**.

My strongest recommendation for the very next simulations is therefore this compact set:

```text
A. Baseline
B. lLN1_bc outputs = 0
C. v2LN30 outputs = 0
D. v2LN30 forced inhibitory
E. all cholinergic AL LN → LN edges = 0
F. all cholinergic AL LN → PN edges = 0
```

Do **not** combine E and F initially.

Those six conditions should tell you far more than another global weight reduction. If **E eliminates persistence while F does not**, you have essentially localized the runaway engine to **excitatory local-interneuron recurrence**. If **F eliminates it instead**, follow the PN return paths. If **C or D alone dramatically fixes it**, investigate **v2LN30's sign before changing the rest of the model**. If **B alone fixes it**, lLN1_bc becomes your prime causal family rather than merely your highest-ranked eigenvector family.

That would move you from “there is an attractor somewhere in the antennal lobe” to a testable statement of the form:

> **“The MaleCNS runaway state requires cholinergic lLN1_bc-mediated LN→LN recurrent excitation,”**

or whatever pathway actually survives these causal tests.