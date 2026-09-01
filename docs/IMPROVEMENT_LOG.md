# Improving ASTRO: every step, with the measurement that motivated it

Written after the 124M shared-rate run put ASTRO **behind** Muon by `+0.0903` at
300 steps and `+0.1207` at 900, on 0 of 2 seeds, while running 12-15% slower.
That run and the earlier one that had ASTRO ahead by `-0.0396` disagree by 0.13
nats -- sixty times the noise floor -- so the first job is not to add anything.
It is to find out why one learning rate does not mean one step size.

Each section below states the defect, the measurement that found it, the change,
and what would falsify the change. Nothing here is adopted on argument alone.

---

## Step 0 -- the diagnosis: a shared learning rate is not a shared step size

**Question.** ASTRO and Muon were compared at `lr = 0.0144`. Does that put them
at the same step size?

**Measurement** (`scripts/diagnose_step.py`, CPU, seconds). Feed both optimizers
the *same* sequence of 40 anisotropic gradients -- heavy-tailed row norms,
decaying spectrum, decaying scale, i.e. the structure real gradients have -- at
GPT-2 124M shapes, and record the Frobenius norm of the update each returns.

A perfect polar factor of an `m x n` block has `min(m, n)` singular values equal
to 1, so its Frobenius norm is exactly `sqrt(min(m, n))`. That is the number the
learning rate is implicitly calibrated against. What the two actually deliver on
the fused QKV projection, split into three `768 x 768` blocks:

| | per-block norm | fraction of the `sqrt(768) = 27.71` target |
|---|---|---|
| ASTRO, as shipped | 21.38 | **0.771** |
| ASTRO, `post_normalize=True` | 27.71 | **1.000** |

and the ratio between ASTRO's whole-tensor update and Muon's, over one 40-step
run:

| | mean | min | max | **drift within the run** |
|---|---|---|---|---|
| ASTRO as shipped | 0.9263 | 0.8847 | 0.9318 | **0.0471** |
| ASTRO, `post_normalize=True` | 1.1132 | 1.1088 | 1.1469 | 0.0381 |

**Three things follow.**

1. Neither optimizer reaches its theoretical step. ASTRO's split blocks deliver
   77% of it; Muon's fused tensor delivers about 83%.
2. The *ratio between them is not even constant*. It moves by 4.7% over 40
   steps, because Newton-Schulz's landing point depends on the conditioning of
   the matrix handed to it and the conditioning changes as training proceeds.
   So a single learning rate does not hold the two at a fixed relative step size
   for the length of a run, let alone at the same one.
3. Once each block is pinned to its target, ASTRO takes an **11% larger** step
   than Muon at equal `lr`. So the shared-rate comparison was never a comparison
   of directions. It was a comparison of two different step sizes, and which one
   won depended on which happened to sit nearer its own optimum.

**This is the whole explanation for the contradiction between the two 124M
runs**, and it is why no amount of extra components would have fixed the result.

**Falsifier.** If pinning the norm leaves the per-optimizer tuned learning rates
in the same ratio as before, the drift was irrelevant and this step is cosmetic.

---

## Step 1 -- pin the update norm to its theoretical value

**Change.** `post_normalize` defaults to `True`. After the spectral filter, each
block is rescaled so `||Z||_F = sqrt(min(m, n))` exactly.

**Why this and not "more Newton-Schulz steps".** They are different claims and
only one of them survives the literature. NVIDIA
([2606.00371](https://arxiv.org/abs/2606.00371)) tested *polar accuracy* -- a
truncated Polar Express, the Jordan quintic, a cubic schedule and an exact FP32
SVD polar factor -- and found final loss nearly indistinguishable on GPT-2
Small. So making the singular values individually closer to 1 buys nothing.

What we are fixing is not accuracy, it is **calibration**: the total norm, which
is the quantity the learning rate multiplies. A rescale is one Frobenius norm
per block and changes no singular value's *relative* position.

We flag the overlap honestly: a uniform rescale does move the whole spectrum,
so it is not fully orthogonal to what NVIDIA measured. The distinguishing
prediction is that pinning should matter for *learning-rate transfer across
shapes and across training* rather than for final loss at one tuned rate.

**Cost.** One norm and one scalar multiply per block per step. Measured below.

**Falsifier.** If a tuned ASTRO with pinning is no better than a tuned ASTRO
without it, and the tuned learning rates coincide, the pin does nothing.

---

## Step 2 -- give each layer a trust ratio

**Defect.** Muon's `max(1, m/n)^0.5` scale is a function of the *shape* of a
tensor and nothing else. Two layers of identical shape whose weights differ in
norm by 5x receive the same update magnitude, so the *relative* change they
undergo differs by 5x. Nothing in Muon or NorMuon controls this.

**Why it is likely to matter.** Hyperball's central claim is that weight decay
in a normalised network is not regularisation but an indirect controller of the
angular learning rate `||W_hat_{t+1} - W_hat_t||_F`; making it explicit yields
20-30% token-equivalent speedup at 1.2B where plain MuonW gives about 10%, and
cuts learning-rate drift across widths and depths from 2-4x to about 1.4x.
OrScale ([2605.07815](https://arxiv.org/abs/2605.07815)) applies layer-wise
trust-ratio scaling to orthogonalised updates directly. This is the
best-supported ingredient ASTRO does not have.

**Rule.** With `trust_ratio > 0`, scale the spectral update so that the step is
a fixed fraction of the layer's own size:

```
u  <-  u * clip( trust_ratio * ||W||_F / (||u||_F + eps),  lo,  hi )
```

clipped to `[1/trust_clip, trust_clip]` so an atypical layer -- one whose weights
are near zero at initialisation -- cannot produce an unbounded step. `lo`/`hi`
exist because the unclipped rule is exactly the failure mode that made the
post-placement variance division dangerous before it was renormalised.

**Interaction.** This *replaces* the shape scale rather than composing with it,
because both set the step length and applying both makes the learning rate mean
nothing. `update_scale="trust"` selects it.

**Falsifier.** If tuned `trust_ratio` loses to tuned `update_scale="muon"` at
equal budget, it is cut, not defaulted-on-and-unmentioned.

---

## Step 3 -- schedule the QKV split instead of applying it forever

**Evidence already in hand.** The fused-projection defect is an
initialisation-time phenomenon: V's share of the update is 0.651 at random
init (0.76 on the GPT-2 family) against 0.502 on trained weights, where uniform
is 0.333. The literature independently reports that head-wise splitting
"reduces validation loss faster early in training, but is later overtaken by
full-matrix Muon".

Splitting also **costs** 1.29x by operation count and 1.33x measured, so a
repair applied forever is paid for forever.

**Change.** `split_steps` already exists and has never been measured. It is now
a swept dimension rather than a flag nobody sets.

**Falsifier.** If `split_steps = infinity` (always split) beats every finite
value at equal budget, the decay does not matter operationally and the schedule
is dropped.

---

## Step 4 -- search the interior between Muon and NorMuon

**Measurement.** How far does the post-placement variance division actually move
the update, as `variance_power` (gamma) goes from Muon's 0 to NorMuon's 1, on
`mlp_fc (3072, 768)` with anisotropic gradients:

| gamma | cos(ASTRO, Muon), mean over 60 steps |
|---|---|
| 0.00 | 1.0000 |
| 0.25 | 0.9994 |
| 0.50 | 0.9972 |
| 1.00 | 0.9869 |

NorMuon's published advantage over Muon is `0.0051` in our own measurement, and
`gamma = 1` moves the direction by 1.3%. The interior is unsearched by either
endpoint's authors and is one scalar.

**Falsifier.** If the tuned optimum sits at `gamma = 0` or `gamma = 1`, there is
no interior worth reporting.

---

## Step 5 -- the experiment that actually settles it

Everything above is preparation for the run that has never happened: an
**equal-budget sweep with a per-optimizer learning rate** at 124M.

Every 124M comparison so far has been one of two invalid designs:

| run | Muon | ASTRO | result |
|---|---|---|---|
| `colab_bench` | tuned by sweep | **guessed** | ASTRO ahead by 0.0396 |
| `astro_lab --config` | **one shared rate** | **same shared rate** | ASTRO behind by 0.0903 |

Neither tuned ASTRO. Given Step 0 -- that the same `lr` is an 11% different step
size once norms are pinned, and that the ratio drifts within a run -- neither
design can answer the question, and they disagree by sixty times the noise
floor, which is what an unanswerable question looks like.

`scripts/astro_lab.py --trials N` gives every optimizer the same number of
trials from ranges that follow its own update scale. That is the protocol this
project has argued for since round 2 and has still never applied at 124M.

**This is the deliverable.** Until it runs, ASTRO neither beats nor loses to
Muon at 124M; we simply do not know, and the paper says so.
