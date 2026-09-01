# The run that settles it

Every 124M comparison in this project has been decided by the tuner rather than
by the optimizer. That is now measured, not suspected, and it is the reason this
runbook was rewritten.

| session | design | what it produced |
|---|---|---|
| `colab_bench` | tuned Muon vs **guessed** ASTRO | ASTRO ahead by 0.0396 |
| `astro_lab --config` | one shared rate for both | ASTRO behind by 0.0903 |
| Session A | one shared rate, component ablation | ASTRO ahead by 0.0571 over an **untuned** baseline |
| Session B | tuned at 900 | tuning alone was worth **0.1184** to Muon |
| Session C | tuned at 300, evaluated at 900 | the 300-step tuner **picked the wrong configuration**, worth **0.1335** |

Read the last two rows together. The tuner moves the answer by more than any
difference between the optimizers, and a cheaper tuner moves it further. No
amount of extra seeds fixes that, because it is not noise in the evaluation --
it is variance in the *selection*.

## What Session C actually showed

I recommended `--tune-steps 300` to make the sweep affordable. It is unsafe.

NorMuon's 300-step tuning ranked trial 3 above trial 5. At 900 steps that
ranking is inverted:

| NorMuon at 900 steps | loss | where |
|---|---|---|
| trial-3 configuration (what the 300-step tuner chose) | 6.1301 | Session C, seed 100 |
| trial-5 configuration | **5.9966** | Session B, seed 0 |

That is not a seed effect. Muon ran the trial-5 configuration at 900 three
times across two sessions and two seeds -- 6.0206, 6.0208, 6.0157 -- a total
spread of 0.0051 against an inversion cost of 0.1335, twenty-six times larger.

So the headline "NorMuon loses to Muon by 0.1144 at 900 steps" is an artifact of
the tuning horizon. **At the shared trial-5 configuration NorMuon beats Muon at
900 by 0.024**, same seed, same session. `--tune-steps` stays in the script
because decoupling the budgets is sometimes what you want, but it is withdrawn
as a protocol for this comparison, and `artifacts/measured.json` says so.

---

## The design that does not have a tuner in it

Stop asking "does tuned ASTRO beat tuned Muon" -- that question routes the
entire result through a selection step with more variance than the effect. Ask
instead:

> Drawn from a **common** hyperparameter range, at **every** configuration, does
> ASTRO beat Muon?

Pointwise dominance over a shared grid is strictly stronger than a tuned-vs-tuned
win, and it is immune to the objection that has invalidated every previous
result here: it cannot be the tuner finding a lucky draw, because there is no
selection. It also implies the tuned comparison, since the maximum over a grid
follows the pointwise order.

Two things make it affordable and make it valid.

**Common random numbers.** `draw_configs` seeds from a fixed constant, so every
optimizer sharing a search space is offered the *identical* configurations. The
sweep is therefore a paired design, and `report_configs` reads it back as one --
at no GPU cost, from runs that used to be discarded. Optimizers with their own
space (`adamw`, `astro_trust`) contribute no pairs rather than wrong ones; the
report pairs on the recorded configuration, never on the trial index.

**Pinning the shared subsystem.** `scalar_lr_mult` scales the elementwise path:
embeddings and the tied head, 31.7% of GPT-2 124M, running *identical AdamW code*
in Muon, NorMuon, AdaMuon and ASTRO. It is not part of what distinguishes them,
and Session C measured the cost of letting the tuner gamble on it at 0.1335.
`--pin scalar_lr_mult=0.4369` holds it at **Muon's own tuned value** -- the
baseline's optimum, which biases against ASTRO -- and spends the search density
on the matrix path, where the actual question lives.

---

## Before you start

The repository is private, so `git clone` and `raw.githubusercontent.com` both
need a token from inside Colab. `scripts/astro_lab.py` is self-contained --
standard library and torch only -- so upload that one file to Drive instead.

```python
!pip -q install transformers datasets
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

Then pass `--work-dir`. The script mounts Drive itself, creates the directory,
and runs there, so the state file, the corpus cache and the report all land
somewhere the runtime cannot take with it:

```
--work-dir /content/drive/MyDrive/astro
```

**This is not advice, it is enforced.** A Colab session with an ephemeral
working directory is refused before the first run starts. Session D ran eight
900-step trials from `/content` -- two hours of a T4 -- and lost all eight when
the runtime was reclaimed, because the state file the resume logic depends on
went with it. A warning would have scrolled past in the first second of a
two-hour cell.

The first line of output is now a fingerprint of the script's own source:

```
astro_lab 413ce5cb7bdf (73,713 bytes)
search space per optimizer (pinned values marked *):
  muon             lr=[0.002,0.1]  weight_decay=[0.001,0.3]  scalar_lr_mult=*0.4369
```

Quote that line back with any result. Session C's drawn values had to be
reverse-engineered to work out whether the file that ran contained a range
widening committed before it, and the answer was still only ~90% confident. The
hash above is the file at the commit that wrote this page; a later commit gives
a different one, which is the point.

## What the first two shared configurations showed

Run twice, in two sessions, on two runtimes. The paired deltas moved by at most
**0.005** between them -- that is the noise floor of this design, and it is
sixteen times smaller than `astro_v2`'s margin at the good configuration.

Session D was killed before it finished, and its numbers were lost with the
runtime -- but they were in the log. Transcribed:

| optimizer | config 1 `lr=0.0102` | config 2 `lr=0.0505` | degrades by |
|---|---|---|---|
| `astro_v2` | **5.9218** | **5.9389** | **+0.0171** |
| `muon` | 6.0026 | 6.4344 | +0.4318 |
| `adamuon` | 6.0040 | 6.2589 | +0.2549 |
| `normuon` | 6.0146 | 6.4239 | +0.4093 |

Two things, and the second is the more interesting one.

`astro_v2` wins both configurations -- by 0.0808 at the good learning rate,
sixteen times the noise floor, and by 0.4955 at the bad one.

And the reason it wins the second is that **it barely notices the learning
rate**. A 4.9x increase costs Muon 0.43 and ASTRO 0.017: a factor of 25. That is
a different kind of claim than "lower loss at the tuned point", it is exactly
what the shared-grid design was built to detect, and it is invisible to any
protocol that reports only each optimizer's best.

Two configurations is not six. The sign test cannot go below 0.5 at `n = 2`, and
config 2 is a mediocre rate for everyone, so the half-nat margin is a robustness
result rather than a peak-performance one. Four more configurations are drawn
and unrun. **A single configuration where `astro_v2` loses falsifies the
pointwise claim**, and that is the thing to watch for.

## Budget

Measured on the T4 in Session D: **~15.5 minutes per 900-step run**, all five
optimizers within 3% of each other (~1.00 it/s). Earlier 13.4/15.2 estimates
were optimistic. Five optimizers is therefore **~78 minutes per configuration**.

### Which ASTRO

Two, not one. Session A ranked `astro_muon_betas` (6.0241) best of six variants,
and `astro_v2` -- which removes the same asymmetry *and* the cautious-decay one
-- has never been run at 900. If the two fixes are additive it lands near 6.008;
if they interact it may be worse than either. Running both costs one extra
column and removes a guess from the headline.

---

## One configuration per cell -- the same command six times (~78 min each)

```bash
!python astro_lab.py --mode scaling --sizes 124M --steps 900 \
    --optimizers muon normuon adamuon astro_v2 astro_muon_betas \
    --trials 1 --seeds 0 --pin scalar_lr_mult=0.4369 \
    --work-dir /content/drive/MyDrive/astro \
    --stop-after 6 --max-minutes 165
```

Then `--trials 2`, `3`, `4`, `5`, `6`. Nothing else changes.

**The sweep runs configuration-major**: configuration 1 for all five
optimizers, then configuration 2, and so on. Any stopping point is therefore
balanced to within the one configuration in flight, and the paired table is
readable at every moment.

That ordering is not cosmetic. Session E walked optimizers in the outer loop,
stopped after six runs, and produced a report comparing Muon, NorMuon and
AdaMuon with **both ASTRO columns simply absent** -- the numbers were fine and
the table answered a question nobody asked. The reference is conventionally
listed first, so the truncation always falls on the thing being tested.

`--stop-after 6` is the belt to `--max-minutes`'s braces. The clock budget does
not help when what ends the cell is Colab reclaiming the runtime rather than the
budget expiring; a run count is predictable, and the sixth slot absorbs a
leftover run from an interrupted previous cell.

**Trial `k` is the same point at every value of `--trials`** -- the draw is
prefix-stable and `tests/test_sweep_plumbing.py` pins it -- so each cell reads
the previous ones back from the state file and runs only what is new. Nothing is
ever repeated. `--seeds 0` means "run the shared grid, evaluate nothing extra".

`--pin scalar_lr_mult=0.4369` is Muon's own value tuned at 900 steps in Session
B -- the baseline's optimum at the evaluation horizon, which biases against
ASTRO.

Six configurations floors the exact sign test at `2/2^6 = 0.031`, the first
value below 0.05.

## Then seeds on the winner (~78 min per cell)

Only once the six configurations are in. The configurations are already cached,
so this is pure evaluation on fresh seeds -- `--seeds 1` per cell, then 2:

```bash
!python astro_lab.py --mode scaling --sizes 124M --steps 900 \
    --optimizers muon normuon adamuon astro_v2 astro_muon_betas \
    --trials 6 --seeds 1 --pin scalar_lr_mult=0.4369 \
    --work-dir /content/drive/MyDrive/astro \
    --stop-after 6 --max-minutes 165
```

That gives the paper two independent paired axes: **across six configurations at
one seed**, and **across seeds at one configuration**. A result that survives
both is not a tuner artifact and is not a seed artifact, and those are the only
two ways every previous result here has died.

---

## Reading the results back

The output scrolls away and Colab wipes the container. **Run these in a separate
cell**, not appended to the training cell:

```bash
!cat /content/drive/MyDrive/astro/astro_lab_report.md
!cat /content/drive/MyDrive/astro/astro_lab_state.json
```

Both are current after every completed run *and after every completed tuning
trial*, so whenever the runtime dies, everything that finished is on disk. If
the report is missing entirely, rebuild it with no GPU:

```bash
!python astro_lab.py --report-only --sizes 124M --steps 900 \
    --optimizers muon normuon adamuon astro_v2 astro_muon_betas \
    --work-dir /content/drive/MyDrive/astro
```

`--report-only` touches no GPU and downloads nothing, so it costs a second.

---

## What each outcome means

Written down first, so no result is a surprise I get to reinterpret afterwards.

| outcome | reading |
|---|---|
| either ASTRO beats Muon at 6/6 shared configurations | the first result here that cannot be a tuner artifact. `p = 0.031`. Then Session F for the seed axis |
| ASTRO beats Muon at 4-5 of 6 | a real but configuration-dependent effect. Report the per-configuration table, not a single number |
| ASTRO ties (paired Δ under 0.005) | at the noise floor. The optimizer claim is dropped; the leverage, inversion, tune-transfer and protocol findings stand on their own |
| ASTRO loses at most configurations | the components do not help at 124M. The paper reports the decomposition and the negative result |
| NorMuon or AdaMuon beats both | the honest headline is that an existing method wins, and the contribution is the protocol that showed it |

Rows three through five are live. `docs/PRIOR_ART.md` records that both
structural contributions are substantially prior art, and NVIDIA measured that an
exact SVD polar factor buys nothing at GPT-2 Small -- a null here would be
consistent with the literature rather than a shock.

What cannot be claimed from a T4 is anything above 124M or past 2700 steps. The
limitations section says exactly that.
