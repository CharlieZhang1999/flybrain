# Stage A changes the strategy: proceed to Stage B, but do not run the original sweep unchanged

## Bottom line

**Yes, proceed toward Stage B, but modify it first.** Your Stage A result is much more informative than the original plan anticipated.

The strongest result is not merely that condition F improved things. It is the *contrast between E, F, and G*:

- Cutting cholinergic **LN→LN** connections did **not** rescue the attractor; it made the tail about **14% worse**.
- Cutting cholinergic **LN→PN** connections reduced the large tail by about **86.5%**, from 4,310 to ~580 neurons.
- Removing **GABAergic LN output** made the tail about **71% worse**.
- Individual removal of `lLN1_bc` or `v2LN30` helped only partially.

That pattern says the leading hypothesis should now be:

> **The pathological gain is associated with excitatory antennal-lobe local-neuron → projection-neuron transmission, probably embedded in an LN↔PN feedback loop. It is not primarily an excitatory LN→LN attractor.**

That is biologically plausible. Drosophila excitatory local neurons are known to provide broad lateral excitation to projection neurons, whereas GABAergic local neurons constrain antennal-lobe activity; projection neurons can in turn excite local neurons, providing a potential feedback route. citeturn6search12turn6search14turn8search2

But I would **not immediately run the old 12-condition Stage B**. I would insert two very cheap causal tests first and then concentrate the gain sweep almost entirely on **LN→PN and the PN return path**.

There are also two important corrections to the previous plan:

1. **Do not treat “all LN→LN” as one edge class anymore.** Your E lesion mixes potentially destabilizing excitatory-LN→excitatory-LN connections with potentially stabilizing excitatory-LN→GABA-LN connections.
2. **Do not use 0/18 persistent inputs as the Stage B success criterion.** You already know bitter, umami, looming and sound contain separate persistent states outside this antennal-lobe phenomenon. For this stage, judge the **olfactory attractor specifically**.

## What Stage A is actually telling you

Here is the experiment interpreted as a causal screen rather than simply a table of firing counts.

| Condition | Silent-tail change vs baseline | Main interpretation |
|---|---:|---|
| **B: `lLN1_bc` output = 0** | 4,310 → 3,250, **−24.6%** | Important contributor, but nowhere near sufficient |
| **C: `v2LN30` output = 0** | 4,310 → 3,940, **−8.6%** | Not a dominant single-neuron engine |
| **D: `v2LN30` inhibitory** | 4,310 → 3,480, **−19.3%** | Added inhibition suppresses activity, but has severe collateral effects |
| **E: cholinergic LN→LN = 0** | 4,310 → 4,930, **+14.4%** | Broad LN→LN removal is actually destabilizing |
| **F: cholinergic LN→PN = 0** | 4,310 → ~580, **−86.5%** | This is the dominant hit |
| **G: GABA-LN output = 0** | 4,310 → 7,370, **+71.0%** | GABA LNs strongly oppose the runaway state |

The first conclusion is therefore quite strong: **`lLN1_bc` matters, but the attractor is distributed.** Its complete output lesion removes about one-quarter of the tail, while ~3,250 neurons still remain. `lLN1_bc` is indeed annotated in MaleCNS as an antennal-lobe local neuron and cholinergic neuron, so its sign is much better supported than your uncertain `v2LN30` assignment. citeturn16search0turn16search1

The `v2LN30` result is even clearer. Silencing it removes only ~8.6% of the pathological tail. Making it inhibitory removes more (~19.3%), but **Hot MN9 collapses from 61 Hz to 1 Hz**. That is exactly why condition D should *not* be interpreted as evidence that "`v2LN30` ought to be inhibitory." Flipping its sign does something much stronger than deleting it: every downstream target that previously received your model's positive contribution now receives negative drive. The enormous off-target functional effect says D is a perturbation, not a plausible correction.

The F result is the centerpiece. Removing LN→PN excitation eliminates approximately **3,730 of the 4,310** persistent neurons. But it simultaneously reduces the antennal-lobe stimulus response from 971 to 377 neurons, a **61.2% loss**, and eliminates the Hot→MN9 response in this setup. That means F has identified a **high-leverage edge class**, not a usable final lesion. This makes biological sense: excitatory antennal-lobe local neurons normally redistribute and amplify odor-related excitation onto PNs; the literature specifically describes broad eLN responses and diffuse excitation of PNs rather than these connections being merely pathological feedback. citeturn6search12turn6search27

The G result is almost as important as F. Removing GABA-LN output causes a huge increase in the persistent state, consistent with the established role of GABAergic antennal-lobe local neurons in suppressing and shaping PN responses. citeturn6search14turn18search14

### Why E getting worse is especially informative

I would **stop referring to E simply as an “LN→LN recurrence lesion.”**

An antennal-lobe local-neuron population contains excitatory, inhibitory and other cell types. Whole-connectome descriptions explicitly note that ALLNs interact with receptor neurons, projection neurons and other ALLNs, with both excitatory and inhibitory interactions. citeturn18search21

Suppose your cholinergic LN has these two outputs:

```text
                ┌──→ excitatory LN ──→ PN
cholinergic LN ─┤
                └──→ GABA LN ─| PN
```

Your E experiment removes **both arrows**.

You intended to remove:

```text
eLN → eLN
```

which might weaken positive feedback.

But you may simultaneously have removed:

```text
eLN → GABA LN
```

which weakens negative feedback.

Your G experiment tells you that GABAergic feedback is extremely valuable: removing it adds about 3,060 neurons to the tail. Therefore, **E becoming worse is entirely compatible with accidentally disconnecting recruitment of the inhibitory network.**

That is not proven yet, but it is a much better explanation than “LN→LN recurrence is strong” because total LN→LN deletion moves the phenotype in the opposite direction.

## Why F points to an LN↔PN loop rather than simply “too much LN gain”

The crucial next question is:

> **Are LN→PN connections actually maintaining the attractor, or do they merely push the network hard enough during the stimulus to enter an attractor maintained somewhere else?**

Those are very different mechanisms.

### Possibility one: LN→PN is the maintenance engine

Something like this is happening:

```text
         excitation
   eLN ───────────────→ PN
    ↑                   │
    │                   │
    └───────────────────┘
         PN return
```

Odor input briefly starts the circuit.

Then:

```text
eLN → PN → LN → PN → LN → ...
```

continues after the stimulus ends.

This hypothesis deserves serious consideration because PNs are themselves predominantly cholinergic/excitatory in the olfactory pathway, and classic physiological work shows that PNs can provide cholinergic excitation back to local neurons. citeturn8search2turn18search4

The connectomic data also make a PN→LN return path concrete rather than hypothetical. Virtual Fly Brain's current connectivity tutorial, for example, identifies `v2LN30` and several other local-neuron types among strong downstream partners of DA1 lPNs; the MaleCNS/FlyWire resources explicitly support weighted partner tracing of these paths. citeturn9search1

### Possibility two: LN→PN is only the ignition amplifier

An alternative is:

```text
odor
 │
 ▼
eLN ─────→ PN ─────→ some other recurrent circuit
                        ▲                │
                        └────────────────┘
```

F might work because deleting LN→PN makes the initial odor response too weak to cross the attractor's threshold.

Once the real attractor has started, however, LN→PN might no longer be required.

That distinction matters enormously. If LN→PN is just the ignition gain, reducing it until the attractor disappears could produce exactly the trade-off you already see:

> less runaway activity **because you weakened the real odor response too much**.

That would explain the ~61% reduction in antennal-lobe responses under F.

So before spending time searching α values, I would bring one piece of your planned Stage D **forward**.

## Revised Stage B: add a causal gate before the gain sweep

I recommend a short **Stage A½** consisting of two or three experiments on one smell that reliably generates the ~4,300-neuron state.

### First test: maintenance lesion

Run the normal network during the odor.

Do **not** touch LN→PN during stimulation.

At odor offset:

```text
stimulus               silent tail
████████████████│────────────────────────

LN→PN:
    α = 1.0      │       α = 0
                 ↑
             switch here
```

Even better, perform one variant where you wait ~50–100 ms into the silent tail so you know the persistent state has actually formed, and *then* cut LN→PN:

```text
odor ends        attractor established        lesion
    │                   │                       │
████│───────────────────│───────────────────────│
LN→PN = 1               1                       0
```

**Interpretation:**

If the existing ~4,300-neuron state rapidly collapses when LN→PN is removed, you have very strong evidence that **LN→PN transmission participates in maintenance of the attractor**.

Then Stage B should absolutely sweep α_LP.

If the persistent state continues despite the post-stimulus LN→PN lesion, then F primarily prevented **ignition**, and you should trace the PN return/recurrent pathway before tuning LN→PN as the final correction.

### Second test: ignition-only lesion

Do the inverse:

```text
stimulus               silent tail
████████████████│────────────────────────

LN→PN:
    α = 0        │       α = 1.0
```

Now ask whether persistence appears after the normal edges are restored.

This produces a very useful causal matrix:

| LN→PN manipulation | Result | Meaning |
|---|---|---|
| Post-offset lesion collapses established tail | Strong effect | **LN→PN is part of maintenance** |
| Stimulus-only lesion prevents persistence | Strong effect | **LN→PN participates in ignition** |
| Both happen | Both effects | LN→PN participates in both ignition and maintenance |
| Only stimulus lesion works | Ignition only | Follow PN/recurrent return path before adopting LP gain correction |
| Neither works | Unlikely given F | Re-check exactly which edges F actually removed |

This is essentially your Stage D maintenance-versus-ignition experiment moved forward because **Stage A has now made it the most informative experiment you can run**.

It may save you an entire gain sweep.

### Third test: verify that F is actually antennal-lobe-specific

This is particularly important because **Hot MN9 goes from 61 Hz to 0 Hz under F**.

If “all cholinergic interneuron→projection-neuron edges” literally means that classification across the whole brain, F is too broad. MaleCNS distinguishes antennal-lobe local neurons (`ALLN`) and antennal-lobe projection neurons (`ALPN`), so the diagnostic lesion can be restricted specifically to:

```text
presynaptic class = ALLN
presynaptic transmitter = acetylcholine
postsynaptic class = ALPN
```

rather than every cholinergic interneuron→projection-neuron connection in the CNS. MaleCNS/VFB annotations explicitly identify ALPNs separately, and `lLN1_bc`, for example, is classified as ALLN. citeturn15search0turn16search0

**If F was already restricted this way, keep it.**

**If it was not, rerun F in this anatomically restricted form before Stage B.**

The disappearance of a non-olfactory Hot→MN9 response is a warning that F may currently encompass more circuitry than the attractor itself.

## The gain sweep I would run now

Assuming the post-offset experiment confirms that LN→PN is required for maintenance, **Stage B should become an LP-first sweep**, not LL-first.

### Do not start with the old αLL sweep

Your old sequence was:

```text
αLL: 0.75, 0.50, 0.25, 0
then
αLP: 0.75, 0.50, 0.25, 0
then
PN return path
```

I would change that to:

```text
1. αLP
2. αPL
3. small αLP × αPL grid
4. LL only after splitting eLN→eLN from eLN→iLN
```

The reason is empirical: α_LL=0 made the relevant phenotype worse, while α_LP=0 produced by far the largest rescue.

### Sweep LN→PN first

Let

\[
\alpha_{LP}
\]

multiply only cholinergic **ALLN→ALPN** edges.

Keep all other weights at their current value.

I would initially run:

| αLP | Purpose |
|---:|---|
| 1.00 | baseline |
| 0.80 | mild correction |
| 0.60 | moderate |
| 0.40 | stronger |
| 0.20 | near-lesion |
| 0.00 | your F endpoint |

Your old `0.75, 0.5, 0.25, 0` is also perfectly usable. I prefer approximately 0.2 spacing because F tells us there is a large dynamic range between α=1 and α=0 and we do not yet know where the transition lies.

Once you find:

```text
α = 0.8     persistent
α = 0.6     not persistent
```

do not run the rest of a generic grid.

Refine:

```text
0.75
0.70
0.65
```

and find the **largest stable α**.

That gives you something closer to a bifurcation threshold:

```text
persistent attractor
       │
       │
───────●────────────── αLP
      ~0.67
       │
       │
stable decay
```

The exact shape may not be a mathematical bifurcation in this finite stochastic spiking model, so I would treat that term descriptively rather than formally.

### Change what you measure during Stage B

Do **not** use the 15/18 persistent-input count as your main objective anymore.

For each smell, record at least:

| Metric | Why it matters |
|---|---|
| **AL neurons active during odor** | Did you preserve sensory processing? |
| **AL neurons active 0–100 ms after odor** | Immediate persistence |
| **AL neurons active 100–200 ms after odor** | Decay versus maintenance |
| **AL neurons active 200–300 ms after odor** | True late persistence |
| **Total AL tail spikes** | More informative than “fired once” |
| **eLN tail rate** | Is the LN population still maintaining activity? |
| **PN tail rate** | Same question for PNs |
| **LN–PN tail overlap** | Does the recurrent pair disappear together? |
| **Across-odor tail overlap** | Are different smells collapsing into the same attractor? |
| **Stimulus-window identity/separation** | Did the correction preserve odor specificity? |

This distinction is important because:

```text
580 neurons each firing once and decaying
```

and

```text
580 neurons firing continuously for 300 ms
```

are completely different dynamical states even though your current “leftover neurons” metric may report **580** for both.

I would particularly calculate the population activity over consecutive tail windows:

```text
odor        0–100       100–200      200–300 ms
██████│────────────│────────────│────────────
      900          300           40            2
```

That is a decaying response.

Compare it with:

```text
██████│────────────│────────────│────────────
      900          650          620          610
```

That is a persistent state.

### Do not optimize overlap by itself

F changes stimulus-window overlap from 0.29 to 0.17.

That looks attractive, but it occurs while the AL response falls from 971 to 377 neurons.

A lower overlap can mean:

> “odor representations became more distinct”

or simply:

> “most neurons stopped responding.”

Therefore use overlap only alongside response magnitude and, ideally, pairwise decoding/separability.

### Suggested acceptance window

For this Stage B screen I would define a provisional “good region” as something like:

\[
\text{late AL tail} < 10\% \text{ of baseline}
\]

while retaining approximately:

\[
\text{stimulus AL response} \gtrsim 75\text{–}80\% \text{ of baseline}
\]

and avoiding major degradation of the downstream sentinel responses.

The exact 75–80% is an engineering criterion, not a known biological constant. The important principle is to search for a **window**, not simply minimize firing.

Your desired curve is:

```text
                             persistent
AL tail                         ███████
                               ███████
                        ███████
                   █████
              ████
         ██
    ██
────┬────┬────┬────┬────┬──── αLP
   .2   .4   .6   .8   1.0

AL odor response
████████████████████████████
      increases smoothly →
```

You are looking for the region where the pathological tail undergoes a sharp change while the legitimate stimulus response changes much less.

That separation would be strong evidence that the MaleCNS transfer needs a **class-specific gain correction**, rather than simply a globally weaker connectome.

## Follow the PN return path immediately after LP

If α_LP has a clean stable window, excellent.

But the more mechanistically satisfying explanation is likely to involve the **return arm**.

Define:

\[
\alpha_{PL}
\]

for:

```text
AL projection neuron → cholinergic AL local neuron
```

and, separately if useful,

```text
PN → GABA LN
```

rather than lumping all PN→LN edges together.

Physiology already supports reciprocal PN–LN relationships in the fly antennal lobe, including cholinergic PN excitation of local neurons. citeturn8search2

The connectome likewise contains substantial PN→LN connectivity; VFB's DA1-lPN example lists `v2LN30` and other local neurons among important downstream partners. citeturn9search1

So a candidate pathological cycle is:

```text
                    ┌───────────────────┐
                    │                   │
                    ▼                   │
              cholinergic LN            │
                    │                   │
                 αLP│                   │αPL
                    ▼                   │
                    PN ─────────────────┘
```

while inhibition is trying to oppose it:

```text
PN ───→ GABA LN ─| PN
```

Condition G says that second pathway matters enormously.

### A useful small factorial

Once you find the approximate LP transition, do **not** run a huge grid.

Suppose:

```text
αLP ≈ 0.65
```

is the transition.

Test perhaps:

| αLP | αPL |
|---:|---:|
| 0.6 | 1.0 |
| 0.7 | 1.0 |
| 0.8 | 0.8 |
| 0.8 | 0.6 |
| 0.9 | 0.6 |
| 0.9 | 0.4 |

The important question becomes:

> Can mild reduction of both halves of the loop preserve more genuine odor drive than severe reduction of LN→PN alone?

For example, hypothetically:

```text
αLP=.55, αPL=1.0
tail = 20
odor response = 650
```

versus:

```text
αLP=.80, αPL=.60
tail = 15
odor response = 850
```

The second solution is much better because you have broken **loop gain** without crippling its forward sensory arm.

That is the result I would hope to find.

## Split the LN→LN experiment instead of sweeping it globally

I would replace most of the old α_LL sweep with a decomposition of E.

Use at least:

```text
E1: cholinergic eLN → cholinergic eLN = 0

E2: cholinergic eLN → GABA LN = 0

E3: cholinergic eLN → unclear-sign LN = 0
```

This is much more informative than:

```text
all cholinergic LN → all LN = 0
```

because E currently mixes mechanisms of opposite sign.

The likely possibilities are:

| Result | Interpretation |
|---|---|
| E1 worsens | eLN→eLN is not the runaway engine |
| E1 improves but E2 worsens strongly | Original E hid an excitatory attractor because it simultaneously removed inhibitory recruitment |
| E2 strongly resembles G | eLNs help recruit GABA stabilization |
| E3 changes everything | Sign uncertainty is contaminating the conclusion |

There is good biological reason to preserve inhibitory LN recruitment. GABA-mediated inhibition shapes PN response magnitude and spatiotemporal odor patterns, and cholinergic mechanisms can modulate GABAergic AL neurons. citeturn6search14turn2search4

So at this point I would **not increase GABA gain globally** either. G says inhibition helps, but “make all GABA stronger” is the inhibitory equivalent of your F full cut: it may suppress the pathology by suppressing legitimate computation.

First correct the excessive positive loop.

## The sign audit is easier than the previous plan suggested

There is one important update from the current MaleCNS release that changes your Stage C logistics.

You **do not need to download the 2.7 GB per-synapse neurotransmitter table first.**

The official MaleCNS download page provides:

- a roughly **42 MB neuron-level neurotransmitter file**, `body-neurotransmitters-male-cns-v1.0.feather`, containing aggregate transmitter predictions for each neuron;
- and the much larger roughly **2.7 GB** `tbar-neurotransmitters-male-cns-v1.0.feather`, containing neurotransmitter probabilities at individual presynaptic sites. citeturn12search0

So I would turn Stage C into two levels.

### Lightweight sign audit first

Load the 42 MB neuron-level table and inspect your 46 “unclear” interneurons.

For every candidate record:

```text
body ID
type
ACh probability / aggregate assignment
GABA
glutamate
serotonin
dopamine
octopamine
confidence / ambiguity
```

This should be cheap enough that I would do it **in parallel with Stage B**, not wait until afterward.

For `lLN1_bc`, independent MaleCNS/VFB annotation already identifies it as cholinergic, so it should not be treated like an uncertain default-sign neuron. citeturn16search0

For cells such as `v2LN30`, where your present pipeline gives an unclear/default sign, use the actual MaleCNS probabilities before making a biological conclusion.

### Only use the 2.7 GB file when necessary

Go to the per-T-bar table if the aggregate result looks like:

```text
ACh 0.37
GABA 0.31
glutamate 0.27
```

or otherwise has poor confidence, or if you suspect transmitter heterogeneity across the neuron's presynaptic sites.

That is exactly what the per-synapse file is designed to resolve. citeturn12search0

This is preferable to arbitrarily declaring all 46 unclear neurons excitatory or inhibitory.

### Do not let the v2LN30 audit block Stage B

Your Stage A results do **not** meet the old decision rule:

> “C or D fixes it → do sign stage first.”

Neither C nor D fixes it.

C gives 3,940 persistent neurons.

D gives 3,480 and virtually destroys Hot→MN9.

Therefore `v2LN30` sign uncertainty is a data-quality issue worth resolving, but it is no longer your main mechanistic lead.

F is.

## Why a class-specific gain correction is plausible in MaleCNS

There is a broader modeling issue worth keeping in view.

Shiu's original implementation uses a simple LIF model with a **single global synaptic weight scale**. The reference implementation uses a resting potential of −52 mV, threshold −45 mV, membrane time constant 20 ms, synaptic time constant 5 ms, and a global `w_syn = 0.275 mV`; edge weights are essentially signed connectivity multiplied by this global factor. citeturn5view0

The original model is intentionally simplified. Shiu and colleagues explicitly note that it ignores gap junctions and neuromodulatory/neuropeptide effects, assumes a 0 Hz basal firing rate, and therefore should not be expected to reproduce absolute biological firing rates. citeturn13search3

Meanwhile, the MaleCNS project's **0.581** factor was introduced to scale MaleCNS edge weights into alignment with FlyWire for cross-dataset comparison; the MaleCNS site explicitly labels it as a scaled structural connection weight. citeturn3view3turn10search1

That does **not** establish that:

\[
0.581\times\text{MaleCNS synapse count}
\]

has exactly the same dynamical stability under the Shiu LIF equations as:

\[
\text{FlyWire synapse count}.
\]

That latter conclusion would require dynamical validation. Your experiment is effectively showing that **global scaling appears insufficient for at least one local circuit**.

A modest class-specific LP/PL correction is therefore not unreasonable as a model-transfer calibration if it is reproducible and preserves sensory function.

There is also a biological reason the simplified dynamics could be especially sensitive in the AL. Real PN odor responses can show rapid accommodation, and antennal-lobe processing involves fast and slower inhibitory mechanisms; a static synapse-count × global-gain model lacks many mechanisms that constrain persistent excitation biologically. citeturn6search8turn6search14

I would therefore use this priority order:

```text
correct wrong transmitter signs
        ↓
correct a demonstrably over-gained edge class
        ↓
only then consider adding new synaptic/adaptation dynamics
```

rather than immediately altering membrane time constants or adding arbitrary global inhibition.

## Revised sequence from here

I would now change your stages to the following.

| Next step | Experiment | Decision |
|---|---|---|
| **A½-1** | F restricted specifically to cholinergic ALLN→ALPN, if it was not already | Verify anatomical specificity |
| **A½-2** | Cut LN→PN only **after** attractor establishment | Does LP maintain the state? |
| **A½-3** | Cut LN→PN only **during stimulus** | Does LP ignite the state? |
| **B1** | αLP coarse sweep | Find approximate transition |
| **B2** | αLP fine sweep around transition | Find largest stable αLP |
| **B3** | αPL sweep | Test PN→LN return arm |
| **B4** | Small αLP×αPL grid | Break loop with minimum signal loss |
| **B5** | Split eLN→eLN vs eLN→GABA-LN | Explain why E worsened |
| **C-light** | 42 MB neuron-level MaleCNS NT audit | Resolve uncertain signs cheaply |
| **C-full, only if needed** | 2.7 GB per-T-bar NT probabilities | Resolve genuinely ambiguous neurons |
| **D** | Subtype add-back and rescue | Locate minimal sufficient recurrent loop |
| **E** | Full CNS, ≥3 seeds | Validate transfer |

For Stage D, the add-back should now be based on **F**, not E.

Start from a stable LN→PN-reduced condition and restore:

```text
lLN1_bc → PN
```

then:

```text
lLN2T / lLN2X / lLN1_a → PN
```

then:

```text
v2LN30 → PN
```

then other cholinergic ALLN subclasses.

Because silencing all `lLN1_bc` output already gives a 24.6% reduction, I would put **`lLN1_bc→ALPN` at the top of that add-back list**.

A particularly convincing result would look like:

```text
Stable corrected network:
αLP = 0.65
tail ≈ 0

restore lLN1_bc→PN to α=1:
tail = 800

restore other eLN→PN:
tail = 3,800

reduce PN→lLN1_bc:
tail returns ≈ 0
```

That would identify an actual feedback motif rather than merely an edge category.

Finally, I would **change your Stage E success rule**. For this antennal-lobe project, do not require the AL correction itself to eliminate the known bitter, umami, looming and sound attractors. Those are separate localization problems by your own lesion results.

The appropriate Stage E bar for the current problem is:

> **All smell-triggered large antennal-lobe persistent states disappear; the normal stimulus response is substantially preserved; downstream functional responses are not catastrophically suppressed; no-stimulus remains silent; and the unrelated small non-olfactory attractors do not become worse.**

The other persistent states can then become the next localization project.

So the immediate answer is:

> **Proceed to Stage B, but do not do the old LL→LP sweep. First run the post-stimulus LN→PN maintenance lesion. If it collapses an already-established attractor, make αLP your primary Stage B variable, then test the PN→LN return gain. Decompose LN→LN rather than weakening it globally.**

Your Stage A result has actually narrowed the search considerably: **the strongest causal handle is no longer “antennal-lobe excitatory local neurons” in general. It is the transmission from those neurons into the PN population, with the PN return path now the leading candidate for closing the pathological loop.**