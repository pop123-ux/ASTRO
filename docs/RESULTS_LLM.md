# ASTRO on GPT-2: setup, protocol, and results

Companion to `docs/paper/paper.pdf`. The paper derives the algorithm and reports
component-level measurements; this document reports the end-to-end language-model
comparison — the thing Section 8 of the paper explicitly declined to claim.

---

## 1. What is being compared, and against what

Seven optimizers, all implemented in this repository so that no comparison
crosses a version boundary:

| optimizer | family | what it contributes to the comparison |
|---|---|---|
| `adamw` | scalar | the control. Everything is measured against it |
| `sgd` | scalar | momentum + Nesterov; the floor |
| `muon` | spectral | the reference matrix-structured method |
| `normuon` | spectral + variance | neuron-wise second moment **after** orthogonalisation |
| `ademamix` | scalar + slow EMA | reported as the strongest scalar method in recent benchmarks |
| `cautious` | scalar + sign mask | C-AdamW; the cheapest published AdamW improvement |
| `astro` | spectral + variance + anchor | this work; variance adaptation **before** orthogonalisation |

## 2. The model

`astro/src/astro/bench/gpt.py` is Andrej Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT) `model.py`, vendored with the
training, sampling and HuggingFace-loading machinery removed and **nothing else
changed**: pre-norm blocks, fused QKV projection, 4×-expansion GELU MLP, learned
position embeddings, weight tying between `wte` and `lm_head`, and the
`N(0, 0.02/√(2L))` scaled initialisation on residual projections.

Only the shape is reduced, and only because of the compute budget:

| | benchmark | GPT-2 (124M) |
|---|---|---|
| layers | 4 | 12 |
| heads | 4 | 12 |
| width | 128 | 768 |
| context | 128 | 1024 |
| non-embedding parameters | 806K | 124M |

At 806K parameters this is a real transformer, not a lookalike — but it is three
orders of magnitude below the scale the Muon/SOAP literature makes its claims
at, and optimizer advantages are known to *shrink* with scale (that is the
central finding of Wen et al., arXiv:2509.02046). Section 6 states what that
does and does not permit.

## 3. The data

| corpus | role | source |
|---|---|---|
| WikiText-2 | pretraining | `pytorch/examples` |
| tinyshakespeare | fine-tuning | `karpathy/char-rnn` |

Encyclopedic prose to Early Modern verse and dialogue: a large stylistic shift
over a shared alphabet, which is what makes a transferred representation useful
without being sufficient.

**Character level, with a vocabulary shared across both corpora.** Shared because
otherwise the embedding table cannot transfer and the fine-tuning task would be
measuring re-initialisation rather than adaptation. Character rather than BPE
because GPT-2's 50257-token vocabulary at width 128 would put an order of
magnitude more parameters in the embedding table than in the transformer blocks
— and the embedding table is routed to the *scalar* path, so the benchmark would
mostly be measuring AdamW on a lookup table instead of the matrix path under
test. `scripts/bench_gpu_llm.py` uses real GPT-2 BPE, where the width makes that
the right trade.

WikiText-2 carries a long tail of CJK and symbol characters: 283 distinct
characters, of which 187 account for 0.02% of the text. Characters below 100
occurrences collapse to one replacement symbol, leaving **97** with 99.98%
coverage.

## 4. Protocol

Enforced in `astro/src/astro/bench/protocol.py`, not by convention:

- **Equal tuning budget.** 16 trials per optimizer, drawn log-uniformly, from an
  RNG seeded identically per optimizer so draw *k* is equally lucky for
  everyone. `tune()` raises if the optimizers do not all tune the **same number**
  of hyperparameters — three each here. Sweeping the proposed method over three
  knobs and the baseline over one is the commonest way an optimizer result is
  inflated, so it is a hard error.
- **Tuning and evaluation are separated.** Tuning runs on seed 0; the winning
  configuration is then re-run on seeds 100–104, so tuning cannot select on the
  noise it is scored against.
- **Paired statistics.** Optimizers are compared seed-by-seed, not by comparing
  means of independent runs, and intervals are percentile bootstraps. A
  difference that does not survive the interval is not reported as a difference.
- **Wall-clock alongside step counts.** A per-step win that costs more time than
  it saves is recorded as a loss.
- **Tuning traces are published.** A best-so-far curve still descending at trial
  16 is direct evidence that the optimizer was under-tuned, and it is printed
  rather than hidden.

Gradient clipping at 1.0 (nanoGPT's default) applies to every optimizer equally.
Without it, a badly scaled draw diverges and scores infinity, which flatters
whichever optimizer's search range happens to be better centred.

### Task calibration

The fine-tuning task can only detect feature disruption if the pretrained
initialisation genuinely matters. Measured under matched AdamW:

| initialisation | Shakespeare val loss after fine-tuning |
|---|---|
| AdamW-pretrained on WikiText-2 | **2.146** |
| random | 2.622 |

A 0.48-nat gap. Over the *full* 1M-token Shakespeare corpus the same gap is only
0.23, because there is enough data to relearn from scratch — the 40K-token pool
is what makes the task sensitive at all. 120 steps at batch 16 × block 128 is
roughly six epochs over that pool, which is the low-data regime where Qu et al.
locate the effect.

## 5. A routing bug that GPT-2 exposed

Worth recording because it would have corrupted the comparison silently, and
because it broke *in favour* of the method being proposed.

GPT-2 ties `lm_head.weight` to `transformer.wte.weight` — they are one tensor.
`named_parameters()` therefore reports it **once**, under the `wte` name. Two
consequences:

1. A router keyed on names never sees `lm_head.weight`, so the tied token
   embedding stayed on the spectral path — orthogonalising a lookup table whose
   rows are indexed by token, which Muon's own guidance forbids. The position
   embedding `wpe` went the same way, being shape `(block_size, n_embd)` and
   indistinguishable from a dense matrix by shape alone.
2. The fallback heuristic "the last 2-D parameter is the classifier head" then
   selected the final block's **MLP projection** — a genuine linear operator that
   belongs on the spectral path — and excluded it.

Three tensors misrouted, in both directions. Because `Muon` and `NorMuon` share
the same router, they were affected identically, so the bug did not simply
handicap one method — it changed what every matrix optimizer was doing.

`classify_module` now resolves parameters **by identity** rather than by name,
and detects `nn.Embedding` weights and the output head structurally from the
module graph. `tests/test_llm.py` pins each of the three cases against this
model.

---

## 6. Results

### 6.1 Headline

`astro_normuon_cautious` — the QKV split, NorMuon's post-orthogonalisation
placement, and cautious masking — beats both industry standards on **gpt_scratch**
at 10 seeds, on a step budget *and* at matched wall-clock. Paired exact Wilcoxon,
Holm-corrected within each family.

| budget | vs NorMuon | vs Muon | vs AdamW |
|---|---|---|---|
| **equal steps** | −0.0138, Holm **0.0195** | −0.0407, Holm **0.0117** | −0.4297, Holm **0.0117** |
| **equal wall-clock** | −0.0163, Holm **0.0293** | −0.0171, Holm **0.0117** | −0.2649, Holm **0.0078** |

It carries no time penalty at this scale — 91.0 ms/step against NorMuon's 92.2.

**The explanation originally attached to that number was wrong** and is
withdrawn. We wrote that splitting the fused QKV "replaces a single 384×128
Newton–Schulz with three 128×128 ones, which is cheaper." Newton–Schulz
transposes to operate on the smaller dimension, so the fused tensor already runs
as 128×384 at `7d³` per iteration against `9d³` split: splitting is **1.29×**
more expensive by operation count, and 1.33× measured on a T4. The timing above
is real — our step is cheaper than NorMuon's — but it is cheaper for other
reasons, and the saving we credited it to does not exist.

### 6.2 Decomposition

Each row adds one component. The **replica** row is the control that makes the
rest interpretable: it reproduces NorMuon to within a negligible effect
(+0.0021, d = 0.19, p = 0.77), so deltas measured on top of it are attributable
to the component added rather than to an implementation gap.

| configuration | loss | adds |
|---|---|---|
| Muon | 1.8396 | — |
| `astro_nosplit` | 1.8399 | ASTRO's defaults, indistinguishable from Muon |
| `astro_split` | 1.8158 | **+ QKV split** (−0.024) |
| NorMuon | 1.8127 | (row normalisation) |
| `astro_normuon_replica` | 1.8148 | *control: reproduces NorMuon* |
| **`astro_normuon_cautious`** | **1.7989** | **+ cautious mask** (−0.016) |

Splitting alone reaches NorMuon's level (+0.0031, p = 0.77), which is what the
leverage argument in §5 predicts: splitting removes the cause of the row-mass
collapse that NorMuon's row normalisation treats after the fact.

### 6.3 Corrections

Three claims made during this work were withdrawn after further measurement. They
are recorded because the pattern — every one caught by a control disagreeing with
a headline, none by reading the code — is itself a result.

**A five-seed fine-tuning win did not replicate.** ASTRO beat AdamW on
`gpt_finetune` by −0.0111 on 5/5 seeds with a bootstrap interval below zero. At 10
seeds the effect fell to −0.0079, 8/10 seeds, exact p = 0.084, Holm 0.168 — not
significant. With n = 5 the smallest attainable two-sided p is 2/2⁵ = 0.0625, so a
perfect sweep is the *only* impressive-looking outcome available and is not
unlikely for a small real effect plus noise.

**That fine-tuning claim then inverted on time.** AdamW's step costs 89 ms against
ASTRO's 137, so in equal seconds it takes 120 steps to ASTRO's 78 and wins by
+0.0315.

**"At matched wall-clock, plain Muon beats NorMuon" was wrong.** It came from a
run carrying the config-resolution bug below, and rested on a single per-step cost
calibration that measured 80.2 ms in one run and 84.1 ms in the next — enough to
move Muon's step allocation from 457 to 438 and with it the whole conclusion.
Corrected, Muon and NorMuon are indistinguishable: +0.0007, d = 0.04, 4/10 seeds,
p = 1.0000.

### 6.4 Two bugs in the measurement code

Both produced publishable-looking numbers, and neither was visible in any single
output — only in the inconsistency between outputs.

**Learning-rate ranges must follow the update scale.** Candidates using
`update_scale="muon"` produce Muon-magnitude updates but were searching Adam's
range, whose ceiling (0.03) sits barely above the 0.0198 NorMuon itself tuned to.
The protocol enforces an equal *number* of tuned hyperparameters — the standard
fairness recipe — and says nothing about whether the *ranges* are comparable. The
NorMuon replica missing by 0.034 is what exposed it.

**Candidate configurations resolved by name, not recency.** The wall-clock harness
globbed candidate artifacts alphabetically and merged with `setdefault`, so a
candidate measured in two rounds got whichever file sorted first — here, the
superseded pre-fix configuration. It produced an apparent regression to 1.8260 that
was purely the wrong hyperparameters. A 0.027 swing from a 0.9% per-step cost
difference is not arithmetically possible, which is what prompted the check.

### 6.5 Held-out validation, and the scope of each component

`gpt_scratch` was held out from ASTRO's original design but candidate rounds 1–3
iterated on it, making it in-sample for the candidate search. The held-out check
is therefore `gpt_finetune`, 10 seeds, against the two industry standards:

| candidate | vs Muon | vs AdamW | seeds won | Holm p |
|---|---|---|---|---|
| `astro_normuon_cautious` | +0.0023 (d = 0.10) | +0.0016 (d = 0.07) | 4/10 | 1.0000 |
| `astro_split` | −0.0071 (d = −0.49) | −0.0078 (d = −0.62) | 8/10 | 0.4219 |

**The two components have different scope, and the paper should not conflate
them.**

*The QKV split transfers.* It is directionally better in both regimes —
significant from scratch (beats Muon, Holm p = 0.0293) and 8/10 seeds with a
medium effect when fine-tuning, though not significant after correction. That
matches what it is: a structural repair of a pathology present in any transformer
with a fused QKV projection, not a tuning choice.

*The full winning recipe is regime-specific.* NorMuon's post-orthogonalisation
placement, Nesterov momentum and Muon scaling win decisively from scratch and add
nothing when fine-tuning (a tie with both baselines, negligible effect). This is
consistent with every other regime split measured here: post-placement wins from
scratch, pre-placement wins fine-tuning.

Nothing regressed — the from-scratch winner is never worse on fine-tuning, it
simply stops helping.

**A fourth correction.** On seeing the partial number 2.1943 against AdamW's
2.1927, this project's notes initially described the from-scratch winner as
"worse than AdamW" on the held-out task. That was reading a raw mean difference
as a result; the paired test gives d = 0.07 over 4/10 seeds at p = 0.92, which is
a tie. A 0.0016 difference should not have been characterised before the test was
run.

The component doing most of the work — cautious masking — is Liang et al.'s, not
ours. The contributions here are the QKV diagnosis and fix, the composition, and
the protocol that caught four wrong claims.

### 6.6 Reproducing

**The paper's Section 5 tables are generated from the benchmark JSON** by
`scripts/make_results.py`, so they cannot disagree with the artifacts.

That is deliberate. Section 5 of the paper is generated from the benchmark JSON by
`scripts/make_results.py`, so it cannot disagree with the artifacts. A second
hand-maintained copy in this file could disagree with both, and a reader would
have no way to tell which was stale. To regenerate after a re-run:

```bash
python -m astro.bench.run --task gpt_finetune --trials 16 --seeds 5 \
    --out artifacts/bench_llm/v2_finetune
python scripts/make_results.py --inject   # replaces the <!--RESULTS--> marker
python scripts/build_paper.py
```

The raw records — every trial, every seed, every selected configuration — are in
`artifacts/bench_llm/*/`. They are gitignored because they are regenerable, so
re-run the commands above rather than looking for them in a clone.

## 7. What this document is for

The paper reports the results and the argument. This file records the parts of
the setup that are settled independently of how the comparison turned out — the
model, the corpora, the protocol, the measured task calibration, and the routing
bug — so that a reader who wants to check whether the *experiment* was built
correctly does not have to reconstruct it from the paper's narrative.

