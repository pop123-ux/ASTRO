# Round 4: what changed, and why each change was made

Round 4 began from a failure. A 124M GPT-2 run on a T4 put Muon at 6.6193,
AdamW at 6.8513 and ASTRO last at 6.8838, and ASTRO diverged one learning rate
above the one it selected. Everything below follows from working out why.

## 1. The default was never the measured winner

The shipped `Astro` defaults were `variance_placement="pre"`,
`update_scale="adam_rms"`, `nesterov=False`, `betas=(0.9, 0.95)`. Round 3
selected `post`, `muon`, `nesterov=True`, `betas=(0.95, 0.95)` -- a gap of 0.041
on `gpt_scratch`, which is the difference between beating Muon and tying it.

So the 124M run measured a configuration already known to be worse. The default
is now what the evidence chose, which makes this the largest single correction
in the round and the least interesting one: it was a bookkeeping failure, not a
scientific one.

Changing it immediately re-exposed the round-2 harness bug from the other
direction. `ranges_for` picked the learning-rate range by inspecting the
*overrides* for `update_scale`, so a Muon-scaled default silently inherited
Adam's range -- a ceiling of 0.03 for an update that wants 0.3. Both
`build_spaces` and `build_candidate_spaces` now pin the recipe explicitly and
derive the range from the pinned value, and an unrecognised scale raises rather
than defaulting.

## 2. The benchmark could not see a third of the model

Embeddings and the tied head take the elementwise path and scale with the
vocabulary; every other operator takes the spectral path and does not. That
ratio decides how much of the model a matrix optimizer is responsible for at
all:

| | vocab | width | elementwise path |
|---|---|---|---|
| `gpt_scratch` (character) | 97 | 128 | **3.5%** |
| `gpt_scratch_mid` (character) | 97 | 256 | 1.9% |
| GPT-2 small | 50257 | 768 | **31.7%** |

The CPU benchmark hands a matrix optimizer 96% of the model where the target
hands it 68%. Tuning `scalar_lr_mult` against the former says nothing about the
latter, and a component that helps the spectral path is over-rewarded by a
factor of roughly ten.

`gpt_bpe` fixes it: a byte-level BPE trained on the corpus itself, at 2816
tokens, chosen for one reason -- it reproduces GPT-2 small's split at this
width, measured 32.2% against 31.7%. Model, budget, batch and schedule are
unchanged, so tokenisation is the only difference between the two tasks.

## 3. Components, each behind a flag

**`post_normalize`.** Newton-Schulz converges to its own fixed point, not to the
polar factor: singular values land in roughly `[0.68, 1.13]`, and where in that
range depends on the conditioning of what it was handed. On a 128x64 block with
a planted spectrum the update's Frobenius norm moves from `0.688` to `0.921` of
`sqrt(min(m,n))` as the condition number goes 1 to 1000. That is a 34% drift in
effective step size driven by a quantity nobody tracks and which moves as
training changes the gradient spectrum. Pinning costs one norm per block.
Muon+ reports a consistent gain from an extra normalisation after
orthogonalisation across 130M-1B, which is plausibly the same correction.

**`converging`.** The deeper version of `post_normalize`. Muon's quintic is one
polynomial applied k times, and it *cannot* reach the polar factor at any
budget: its fixed points solve `2.4445 - 4.7750 s^2 + 2.0315 s^4 = 0`, at 0.868
and 1.264, which bracket exactly the band the singular values were measured in.
That is why five, eight and twelve steps all gave the same spread. Solving one
polynomial per step greedily removes the obstruction:

| filter | max abs(P-1) on [1e-3, 1] | update-norm ratio, condition 1 to 1000 |
|---|---|---|
| `muon5` | -- | 0.6881 / 0.9024 / 0.9212 / 0.8641 |
| `polar5` | 0.1167 | 0.8940 / 1.0017 / 0.9990 / 0.9664 |
| `polar6` | 0.0010 | 0.9997 / 1.0001 / 1.0001 / 0.9883 |
| `polar7` | 0.0000 | 1.0000 / 1.0000 / 1.0000 / 0.9985 |

At equal cost the drift halves and centres on 1; two steps further it is exact.
The coefficients are solved, not copied, so they can be re-derived and checked --
and the last two land near `(1.875, -1.25, 0.375)`, independently reproducing
the asymptote the Polar Express publishes. That agreement was not fitted to and
is the main reason to think the solve found the right object.

Whether exactness *helps* is a separate question. Muon's under-convergence may
be acting as useful damping, and seven steps costs 40% more than five, so both
budgets are registered as candidates and the benchmark decides.

**`variance_power`** (gamma). Interpolates Muon (0) to NorMuon (1). Proposition
1 identifies the row norms NorMuon divides out as leverage scores; localisation
is signal for a structured matrix, so dividing all of it out should overshoot.
Both endpoints are published and the interior is not searched by either. It also
connects two results that have not been connected: Muown finds Muon's
spectral-norm drift is driven by the row-magnitude factor, and Proposition 1
says the update's row mass *is* the leverage profile -- so localised update mass
is the mechanism behind the drift Muown measures, and gamma is a one-scalar
control on it.

**`spectral_blend`** (alpha). Mixes the unfiltered direction back in,
norm-matched so it changes the update's shape and not its length. Ma et al.'s
river-valley analysis finds Muon fast along the informative direction early and
*slower than gradient descent* near the optimum, because a constant-magnitude
spectral update cannot take a small step.

**`equilibrate`.** MuonEq's R variant: equalise the momentum's row norms
instantaneously before the polar step. Not the same rule as pre-placement
variance adaptation, which is an EMA of the gradient rather than the momentum's
current row norms, and which therefore does not condition the matrix actually
being handed over.

**`cautious_wd`.** Cautious Weight Decay (Chen et al.): decay a coordinate only
where the step is already carrying it toward zero. No new hyperparameter, and
reported to help consistently at million-to-billion scale. On by default, so the
ablation removes rather than adds it.

**`split_steps`.** The fused-projection defect is an initialisation-time
phenomenon -- V's share of the update is 0.65 at init against 0.50 on trained
weights, where uniform is 0.33 -- so a repair applied forever is the wrong
shape. Relaxing to a single block later is also the cheaper branch: a fused
Newton-Schulz iteration is `7d^3` against a split `9d^3`.

**Per-block RMS scaling** is a correctness fix, not a component. Under
grouped-query attention the blocks have different shapes, so one scale taken
from the largest is wrong for the other two.

## 4. What was retracted

The claim that splitting is cheaper than not splitting. Muon's Newton-Schulz
transposes to operate on the smaller dimension, so a fused `(3d, d)` already
runs as `(d, 3d)` at `7d^3` per iteration against a split `9d^3`. Predicted
1.29x slower, measured 1.33x on a T4. This was a reasoning error rather than a
measurement artifact, which makes it the worst of the five.

See `docs/PRIOR_ART.md` for the rest, including that the split itself is not
novel.

## 5. The result, and the one component that had to go

At GPT-2 124M on FineWeb-Edu, 300 steps, batch 8, sequence 512:

| optimizer | val loss | vs Muon | s/run |
|---|---|---|---|
| **`astro` without the cautious mask** | **6.6191** | **-0.0383** | 323 |
| `normuon` | 6.6523 | -0.0051 | 327 |
| `muon` | 6.6574 | -- | 323 |
| `astro` with the mask | 6.7508 | +0.0934 | 333 |
| `adamw` | 6.9656 | +0.3083 | 155 |

The mask alone accounts for `-0.1317` on 2 of 2 seeds, at one shared
configuration so the comparison between those two rows needs no tuning
assumption. Removing it also makes ASTRO *cheaper* than the masked version and
exactly as expensive as Muon.

**The cautious mask's sign depends on scale.** On the 1.17M subword benchmark it
is worth `-0.0291` against Muon on 8 of 8 seeds, exact `p = 0.0078`. At 124M,
with the same tokeniser and the same protocol, it costs `+0.1317`. Four times
the magnitude, in the other direction.

The plausible mechanism is that the mask deletes coordinates whose sign
disagrees with the *current minibatch* gradient and rescales the survivors by
`numel/count`. At larger width the momentum is the better of the two estimates,
so most of what the mask removes is the minibatch being wrong rather than the
update being wrong, and the rescale amplifies whatever survives.

This is the second time in this project that a small-scale result has predicted
the wrong sign at 124M, and the first time the small-scale evidence was as
strong as 8 of 8 seeds. The lesson is not that the CPU benchmark is useless --
it found the QKV split, the placement, the update scale, all of which held --
but that a component's *sign* is not among the things a 100x-smaller benchmark
can be trusted to establish.

### Confirmed at three seeds

Every seed is shared, so these are genuine paired comparisons rather than a
comparison of means:

| | mean | per-seed | paired Δ | worst seed | seeds won |
|---|---|---|---|---|---|
| **`astro` (no mask)** | **6.6177** | 6.6065, 6.6317, 6.6150 | -- | -- | -- |
| `normuon` | 6.6523 | 6.6422, 6.6576, 6.6572 | **-0.0346** | -0.0259 | **3/3** |
| `muon` | 6.6573 | 6.6475, 6.6661, 6.6584 | **-0.0396** | -0.0344 | **3/3** |
| `astro` (masked) | 6.7519 | 6.7424, 6.7593, 6.7539 | -0.1341 | -0.1276 | 3/3 |
| `adamw` | 6.9657 | 6.9532, 6.9575, 6.9863 | -0.3479 | -0.3258 | 3/3 |

The margin over Muon is not carried by one lucky seed: the *worst* of the three
is still `-0.0344`, larger than NorMuon's whole mean advantage over Muon
(`-0.0051`). Wall clock is 323 s/run against Muon's 323, so this is not a
per-step win paid for in time.

### What still needs confirming

Three seeds is the exact sign test's floor at `p = 0.25`, so the *p*-value
cannot certify this however clean the sweep is; the evidence is 3/3 with a
worst-case margin seven times NorMuon's mean advantage, not a small *p*.

The unmasked recipe was never tuned. It ran at ASTRO's learning rate with a
guessed weight decay, because the tuned configuration was lost with a wiped
Colab session, while Muon and NorMuon ran at configurations selected from an
equal-budget sweep. That handicaps the winner, so it makes the result harder to
dismiss rather than easier -- but it also means the margin is a lower bound of
unknown tightness, and an equal-budget sweep for the unmasked recipe is what
turns this from a finding into a claim.

Both remaining gaps therefore point the same way: more seeds and a real sweep
would be expected to *widen* the margin, and until they are run the honest word
for a 0.04-nat lead over Muon is "directional".

## 6. The sweep was run, and it reversed the result

An overnight `astro_lab.py` run at 124M with `--config lr=0.0144
weight_decay=0.02 scalar_lr_mult=0.1 beta2=0.95` -- one shared configuration for
every optimizer rather than a sweep for some and a guess for others:

| steps | muon | astro | paired Δ | worst seed | seeds won by astro | s/run |
|---|---|---|---|---|---|---|
| 300 | 6.5870 | 6.6772 | **+0.0903** | +0.0990 | **0/2** | 276 vs 314 |
| 900 | 6.0849 | 6.2056 | **+0.1207** | +0.1274 | **0/2** | 832 vs 936 |

ASTRO loses on every seed at both budgets, the gap **widens** with longer
training, and it is 12-15% slower per run. That is the opposite sign from §5,
and it is much larger than the 0.0021 noise floor.

**Neither run is a fair comparison, and that is the finding.** §5 tuned Muon,
NorMuon and AdamW by sweep and gave ASTRO a guessed configuration; §6 gives
every optimizer one shared configuration. Look at what each did to Muon:

| | Muon at 300 steps | ASTRO at 300 steps |
|---|---|---|
| §5, Muon tuned, ASTRO guessed | 6.6573 | **6.6177** |
| §6, one shared rate | **6.5870** | 6.6772 |

Muon is *0.07 better* under the shared rate than under the configuration its own
tuning sweep selected, and ASTRO is *0.06 worse*. So the §5 sweep did not find
Muon's optimum, and the shared rate does not suit ASTRO. Each comparison
flattered whichever optimizer happened to sit nearer its own optimum, and they
disagree by 0.13 nats -- sixty times the noise floor.

The §5 result is therefore **withdrawn**, not merely qualified. We do not have a
valid 124M comparison in either direction. What settles it is an equal-budget
tuned sweep with a per-optimizer learning rate, which is exactly the protocol
this project already argues for and did not apply here.

This is the seventh retraction, and the second time the mistake has been about
learning-rate ranges rather than about counts of tuned hyperparameters -- see
Remark "Equal budgets are not sufficient" in the paper. Having written that
remark did not stop us doing it again.

## 7. The horizon question, answered: the gap widens

The same shared configuration, extended to 2700 steps over about four hours on
one T4:

| steps | muon | astro | paired Δ | astro wins | s/run |
|---|---|---|---|---|---|
| 300 | 6.5861 | 6.6764 | **+0.0903** | 0/2 | 270 vs 307 |
| 900 | 6.0837 | 6.2043 | **+0.1206** | 0/2 | 804 vs 915 |
| 2700 | 5.5251 | 5.7224 | **+0.1973** | 0/1 | 2407 vs 2737 |

Every 300- and 900-step cell reproduces the earlier session to within `0.0024`
against a `0.0021` noise floor, so this is the same code and the same
configuration confirmed rather than assumed. ASTRO has one completed seed at
2700; the second was still running when the budget expired.

**The gap does not decay with training length. It roughly doubles as the budget
goes up 9x.** That is the opposite of the pattern \citet{wen2025fantastic}
describe, where optimizer advantages shrink as budgets grow -- here a
*disadvantage* grows.

Two things follow, and the second is the uncomfortable one.

First, this run tested none of the improvements. `post_normalize` and the trust
scale both landed after it started, and the bit-level reproduction proves the
old code ran. So it is not evidence against the current optimizer.

Second, and more seriously: a widening gap is hard to explain as a step-size
offset. `docs/IMPROVEMENT_LOG.md` measures ASTRO's step at about 0.93 of Muon's
on the fused projection at this configuration. A uniformly smaller step would
show up as a roughly constant lag or one that *closes* as both approach the
same basin -- not one that doubles. Something in the recipe is compounding
against us as training proceeds, and the honest reading is that the 300-step
cells were the most favourable point on the curve rather than a representative
one.
