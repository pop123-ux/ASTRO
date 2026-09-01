# A plan to actually beat Muon, and why the current approach could not

## What went wrong, stated plainly

ASTRO adds six things to Muon at once. Not one of them has ever been measured
in isolation at 124M. When the assembled optimizer lost by `+0.1206` at 900
steps we had no way to attribute a single nat of that to a single component,
so every response was a guess.

Worse: two of the six differences are not components at all. They are
**unintended asymmetries** — places where ASTRO and Muon run different
algorithms for reasons nobody chose. Both were found by reading the two classes
side by side, and both bias against ASTRO, and one of them compounds.

That is the actual explanation for the last four hours of GPU time: we were
comparing a six-change bundle containing two accidents, at one shared
configuration, and reading the total as a verdict on the design.

---

## The audit: every difference between ASTRO and Muon

Read off the two vendored classes in `scripts/astro_lab.py`, which is the code
that produced every 124M number.

### Unintended asymmetries — these are bugs in the comparison

| # | difference | Muon | ASTRO | effect |
|---|---|---|---|---|
| **A1** | weight-decay coverage | decays **100%** of coordinates | cautious mask decays **38%** | **compounds multiplicatively**: predicted weight inflation 1.06× at 300 steps, 1.17× at 900, **1.62× at 2700** |
| **A2** | scalar-path `beta1` | `0.9` | `0.95` | applies to **31.7%** of GPT-2 124M (embeddings + tied head). Different Adam on a third of the model |

A1 is the only mechanism found so far whose *shape* matches the widening gap.
A2 is smaller but is a genuine "different optimizer on a third of the
parameters" problem, and it has never been controlled.

Neither is a property of ASTRO's design. Both must be removed before any
comparison means anything.

### Checked and cleared

**Momentum normalisation.** Muon accumulates `buf = μ·buf + g` unnormalised;
ASTRO uses an EMA `m = βm + (1−β)g`. These look different but are not: the
Nesterov combination gives the same ratio of fresh gradient to history in both
(`2.16` at `β = 0.95`), and orthogonalisation discards the overall scale. Not a
difference.

**Scalar learning rate.** Both give the elementwise path `lr × scalar_lr_mult`.
Same.

### Real components — these are the design, and each is unmeasured at 124M

| # | component | what it does | measured at 124M? |
|---|---|---|---|
| **C1** | QKV split | orthogonalise Q, K, V separately | **never** |
| **C2** | post-variance, γ=1 | NorMuon's neuron-wise second moment | never (NorMuon itself: +0.005 over Muon) |
| **C3** | `post_normalize` | pin each block to `sqrt(min(m,n))` | never (added after the last run) |
| **C4** | cautious update mask | Liang et al. | yes: **+0.1341**, removed |
| **C5** | `update_scale="trust"` | angular learning rate | never |

### The C1 argument I now think is wrong

The case for splitting was that a fused projection gives V 65% of the update
mass at initialisation against 33% for parity. But **uniform is not obviously
the right target**, and the split does something stronger than reallocating
mass:

`msgn` is scale-invariant *per matrix it is applied to*. Applied to the fused
stack, the relative magnitudes of the Q, K and V gradient blocks survive into
the update — V gets a bigger step because its gradient is bigger. Applied to
three separate blocks, each is normalised independently, so **Q, K and V
receive equal update magnitude regardless of their gradients.**

The split does not just rebalance. It *destroys the relative gradient
magnitude information between Q, K and V.* If V genuinely warrants a larger
update — which is exactly what the softmax-Jacobian argument says, since Q and K
saturate and V is linear — then equalising is actively harmful, and the harm is
systematic at every step, which is a candidate for compounding.

Measured on real gradients: splitting changes the update direction from Muon's
by `cos = 0.83`. That is by far the largest direction change of any component,
and it has never been tested at 124M.

Note also that Aurora, the leverage-aware optimizer this project's Proposition 1
overlaps with, applies row-uniformity to **tall MLP projections** — not to QKV.
We may have applied a good idea to the wrong tensor.

---

## The strategy

Three sessions, each **under 3 hours**, each answering one question. No session
depends on a guess from the previous one.

### Principle 1 — decompose before adding

The next thing built should be nothing. Six changes are already on the table
and none is attributed. Ablate first.

### Principle 2 — 900 steps is the workhorse, not 2700

The gap is unambiguous at 900 (`+0.1206`, 0/2 seeds, 60× the noise floor) and a
900-step run costs 15 minutes against 46. Three times the information per hour,
and the horizon behaviour is already measured.

### Principle 3 — one seed for ablation, seeds for the verdict

We are hunting effects of 0.05–0.12 nats against a noise floor of 0.002. One
seed resolves that at 25:1. Seeds are for the final head-to-head, not for
attribution.

### Principle 4 — the learning rate is not a detail

Muon at `lr = 0.0144` reaches `6.586`; Muon at its own earlier "tuned" setting
reached `6.657`. That is `0.07` nats from the learning rate alone — the same
order as the gap we are chasing. **ASTRO has never been tuned at 124M.** Any
verdict before that is premature.

---

### The budget is now enforced, because it was not

`--max-minutes 240` produced a four-hour run that returned nothing. The check
only ran *between* runs, so a 46-minute 2700-step cell starting at minute 239
finished at 285. The script now estimates the next run's cost from timings
already in the state file and **refuses to start one that will not fit**:

```
[124M 900st] muon seed 101  (~15 min, 15/40 used)
skipping [124M 900st] muon seed 102: needs ~16 min, only 10 left of the budget.
```

Two tests pin it. Measured costs at 124M, batch 8, sequence 512, on a T4:

| steps | Muon | ASTRO-class |
|---|---|---|
| 300 | 4.5 min | 5.1 min |
| 900 | 13.4 min | **15.2 min** |
| 2700 | 40.1 min | 45.6 min |

Every session below is sized against those numbers with `--max-minutes 165`,
which leaves 15 minutes of slack inside a 3-hour cap.

## Session A — attribute the gap (≈105 min)

Strip ASTRO back toward Muon, one change at a time, at 900 steps, one seed, at
the same configuration as every previous run so the numbers are directly
comparable.

| run | removes | tests |
|---|---|---|
| `astro` | — | reference (known: 6.2043) |
| `astro_plain_wd` | A1 | the decay confound |
| `astro_muon_betas` | A2 | the scalar-path confound |
| `astro_nosplit` | C1 | **the split — highest prior** |
| `astro_gamma0` | C2 | the NorMuon component |
| `astro_unpinned` | C3 | the pin |
| `muon` | everything | reference (known: 6.0837) |

```bash
!git clone -q https://github.com/pop123-ux/knsa-knee-normality-kaggle
%cd knsa-knee-normality-kaggle/astro
!pip -q install transformers datasets

!python scripts/astro_lab.py --mode scaling --sizes 124M --steps 900 \
    --optimizers astro astro_plain_wd astro_muon_betas astro_nosplit \
                 astro_gamma0 astro_unpinned muon \
    --config lr=0.0144 weight_decay=0.02 scalar_lr_mult=0.1 beta2=0.95 \
    --seeds 1 --max-minutes 165
```

Then, in a **separate cell** so it survives the output scrolling:

```bash
!cat astro_lab_report.md
!cat astro_lab_state.json
```

Read: any variant landing near `6.084` identifies its removed change as the
cause. The deltas should roughly sum to `+0.1206`; if they do not, the
components interact and that is itself worth knowing.

Note this re-runs `astro` and `muon` at 900 even though we have them, because a
single session with one code version is worth more than a cross-session
comparison at a 0.002 noise floor — and it costs 29 of the 165 minutes.

**Outcome:** a ranked list of what each change costs, which is the first
attribution this project has ever had at 124M. It is also a paper table on its
own — a component-by-component decomposition at the scale that matters.

## Session B — tune the survivor (≈150 min)

Take the winning recipe from A — call it `astro_v2` — and give it and Muon the
same tuning budget, which is the protocol this project argues for and has never
applied at 124M.

```bash
!python scripts/astro_lab.py --mode scaling --sizes 124M --steps 900 \
    --optimizers muon astro_v2 --trials 5 --seeds 1 --max-minutes 165
```

5 trials × 2 optimizers × ~14.3 min = 143 min, plus 2 evaluation runs. The
budget guard will stop cleanly and the state file resumes, so if it does not
all fit, re-running the identical command in a second session finishes it.

This is the first tuned ASTRO at 124M. Given that Muon alone moves `0.07` nats
between learning rates, this is where a win comes from if one exists.

## Session C — the verdict, only if B reaches parity (≈150 min)

```bash
!python scripts/astro_lab.py --mode scaling --sizes 124M --steps 900 \
    --optimizers muon normuon astro_v2 --trials 5 --seeds 3 --max-minutes 165
```

Tuning is cached from B, so this is 9 evaluation runs at ~14.3 min = 129 min.
Three seeds against Muon **and** NorMuon, all tuned at equal budget. That is
the table the paper prints, and the first one in this project that a referee
could not dismiss on protocol.

Three seeds floors the exact sign test at `p = 0.25`; a fourth session of 3 more
seeds each on the two-way comparison would reach `p = 0.03` at 6 seeds.

---

## What I honestly expect

I am not going to promise a win, because the evidence does not support
promising one and a promise is worth nothing to a referee.

What I can say:

- **A1 alone predicts a large fraction of the widening.** Its predicted weight
  inflation at 900 steps is 1.17×, and weight norm is the quantity Hyperball
  identifies as controlling the angular learning rate. This is a real
  confound with the right shape.
- **C1 has the largest unmeasured direction change** (`cos = 0.83`) and an
  argument for being actively harmful that I did not have before.
- **The learning rate is worth 0.07 nats** and has never been tuned for ASTRO.

Those three together are the same order as the `0.1206` gap. That is a credible
path to parity or better. It is not a guarantee, and the honest form of this
plan is that Session A tells us which of the three it is — and if the answer is
"none of them", the paper reports a decomposition showing a matrix optimizer
whose additions do not survive at scale, which is a real and publishable result
in the current climate.

## What makes it verifiable by a referee

Every claim above is checkable without a GPU:

- the two asymmetries are visible by reading two classes side by side, and
  `tests/test_optimizer.py` now pins the decay one with a measured fraction;
- the leverage identity and the quintic fixed points are CPU verifications that
  run in seconds;
- every figure writes the numbers it drew to a sibling JSON;
- every run checkpoints to `astro_lab_state.json` and the report can be rebuilt
  from it with `--report-only`, no GPU.

The training results are 124M on FineWeb-Edu, which is the scale the curvature
literature uses and defends explicitly. What we cannot claim is anything above
124M or past 2700 steps, and the limitations section says exactly that.
