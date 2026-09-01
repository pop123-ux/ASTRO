# ASTRO — Anchored Spectral Trust-Region Optimizer

A matrix-structured optimizer in the Muon/SOAP family, plus the benchmark harness used to
evaluate it. **Start with [`docs/paper/paper.tex`](docs/paper/paper.tex)** — it carries the
derivation, the pseudocode, and a precise account of what has and has not been measured.
(`paper.pdf` is built from the superseded `paper.md` draft and is one round behind.)

Self-contained: depends only on `torch`. Nothing here imports from the surrounding repository.

## Research status

ASTRO is being developed as a compute-constrained research project. The objective is not to claim universal optimizer superiority from a single small run. The current program is to identify and falsify mechanisms, freeze the smallest recipe supported by the evidence, and validate it on a real Transformer training task with reproducible comparisons.

**Current priority:** finish the shared-configuration study in [`scripts/astro_lab.py`](scripts/astro_lab.py), then perform multi-seed end-task validation on a 124M Transformer derived from modded-nanoGPT. See [`docs/RESEARCH_RELEASE_PLAN.md`](docs/RESEARCH_RELEASE_PLAN.md) and [`docs/MODDED_NANOGPT_VALIDATION.md`](docs/MODDED_NANOGPT_VALIDATION.md).

### Existing directional result

At GPT-2 **124M** on FineWeb-Edu, 300 steps, ASTRO beats Muon, NorMuon and AdamW on **3 of 3
shared seeds** at Muon's wall-clock:

| optimizer | val loss | paired Δ | worst seed | s/run |
|---|---|---|---|---|
| **astro** | **6.6177** | — | — | 323 |
| normuon | 6.6523 | −0.0346 | −0.0259 | 327 |
| muon | 6.6573 | −0.0396 | −0.0344 | 323 |
| adamw | 6.9657 | −0.3479 | −0.3258 | 155 |

**Read that as directional, not established.** Three seeds is the exact sign test's floor at
`p = 0.25`, so no arrangement of these numbers yields a small *p*. ASTRO's row is also the only
one that was never tuned — its tuned configuration was lost with a reclaimed Colab session, so
it ran at a guessed weight decay while every baseline it beats came out of an equal-budget sweep. Both gaps bias *against* the margin, but neither substitutes for running the sweep. The
measured cross-session noise floor is **0.0021**, so the margin over Muon is 18× noise.

The newer `astro_lab.py` study is the authoritative path for resolving this limitation. Everything above 124M, and every horizon past 300 steps, is **unmeasured** until that study is complete.

## What the work produced besides the optimizer

- **the leverage identity** — the squared row norms of a Muon update are exactly the leverage
  scores of the momentum's row space. Two corollaries follow: row normalisation is provably
  inert on wide matrices, and a *fused* QKV projection hands ~85% of its update to `V` while
  passing every orthogonality check. Measured on 8 pretrained checkpoints, 124M–1.4B: it is an
  **initialisation-time** defect that decays (V's share 0.65 at init → 0.50 trained → 0.333 uniform);
- **Muon's quintic cannot converge** — its fixed points solve `2.4445 − 4.7750s² + 2.0315s⁴ = 0`
  at `s = 0.868` and `1.264`, so singular values never reach 1 at any step budget, and the
  update's norm drifts 34% with the conditioning of what it was handed. A solved per-step
  schedule removes the obstruction and is exact at 7 steps; its tail independently reproduces
  the published Polar Express asymptote, which was not fitted to;
- **a component whose sign inverts with scale** — cautious masking is worth −0.0291 on **8/8
  seeds, exact p = 0.0078** at 1.17M parameters and costs **+0.1341** at 124M, same tokeniser
  and protocol. It is now off by default. A clean small-scale sweep predicted the wrong sign;
- **a routing bug** that GPT-2's weight tying exposes in *every* matrix optimizer using the
  standard name-based policy — the tied token embedding and position embedding go onto the
  spectral path while a genuine MLP projection is excluded, silently, in Muon and NorMuon alike;
- **six retractions and three measurement bugs**, all recorded. The split-is-cheaper claim was
  a one-line arithmetic error published without checking; splitting is 1.29× more expensive by
  operation count and 1.33× measured.

Components **off by default**, because they did not earn being on:

| component | why it is off |
|---|---|
| `cautious` | −0.0291 at 1.17M (8/8 seeds) but **+0.1341 at 124M** — sign inverts with scale |
| `anchor` | failed at three scales in two formulations; with a free choice the tuner switched it off every time |
| `dead_zone` | excellent spectral behaviour, independently verified; costs +1.9% on end-task loss |
| `norm_control="hyperball"` | +4.5% against plain decoupled weight decay |

## What is here

| file | what it is |
|---|---|
| `src/astro/optimizer.py` | ASTRO itself |
| `src/astro/polar.py` | spectral filters: Newton–Schulz, and the dead-zone filter with its minimax solver |
| `src/astro/routing.py` | which tensors are genuine linear operators and which are not |
| `src/astro/baselines/` | Muon, NorMuon, **SOAP**, Hyperball, AdEMAMix, Cautious AdamW — implemented in-repo so comparisons have no version skew |
| `src/astro/bench/` | equal-tuning-budget protocol, seven tasks, runner |
| `src/astro/bench/gpt.py` | nanoGPT's GPT-2, vendored faithfully, for the language-model tasks |
| `src/astro/bench/corpora.py` | WikiText-2 and tinyshakespeare loading and tokenisation |
| `scripts/astro_lab.py` | self-contained Colab/T4 lab for shared configurations, ablations, and scale/horizon checks |
| `docs/MODDED_NANOGPT_VALIDATION.md` | controlled 124M modded-nanoGPT-derived end-task validation protocol |
| `docs/RESEARCH_RELEASE_PLAN.md` | seven-day, compute-constrained research release plan |
| `docs/paper/paper.md` | the paper source; `build_paper.py` renders it to PDF |

The benchmark tasks, in the order they answer questions:

| task | what it asks |
|---|---|
| `quadratic` | how much does Muon's isotropic-input assumption cost, against a known optimum |
| `mlp`, `convnet` | from-scratch training, with the shape zoo that makes routing matter |
| `finetune` | AdamW-pretrained CNN, fully fine-tuned on a shifted distribution |
| `transformer` | from-scratch attention on a procedurally generated language |
| `gpt_scratch` | **GPT-2 from scratch on WikiText-2** — the regime the literature's claims come from |
| `gpt_finetune` | **GPT-2 pretrained with AdamW, fully fine-tuned on Shakespeare** — the regime where matrix optimizers are documented to lose |

## Use

```python
from astro import Astro

optimizer = Astro.from_model(model, lr=3e-4)      # routing decided automatically
```

`from_model` classifies every parameter and sends genuine linear operators (dense convs,
attention projections, hidden linears) down the spectral path, while norms, biases, gains,
depthwise kernels, the stem convolution and the output layer take an AdamW path. To see why
each tensor was routed where:

```python
from astro.routing import classify_module

for name, spec in classify_module(model).items():
    print(f"{name:30s} {spec.kind.value:10s} {spec.reason}")
```

To enable the default-off components — none of which improved end-task loss in our
measurements, so turn them on only to reproduce those measurements:

```python
Astro.from_model(model, lr=3e-4, anchor=True, anchor_mode="elastic", anchor_strength=1e-2)
Astro.from_model(model, lr=3e-4, dead_zone=0.1, ns_steps=10)
```

## Reproduce

```bash
pip install -e ".[dev]"
pytest                                                       # 220 tests

python -m astro.bench.run --task all --trials 16 --seeds 5   # CPU comparison
python -m astro.bench.run --task finetune --ablation         # component ablation
```

Every table in the paper is generated from the JSON those commands write, so after a re-run:

```bash
python scripts/make_results.py --inject   # replaces the <!--RESULTS--> marker in paper.md
python scripts/build_paper.py
```

The two language-model tasks need their corpora fetched once (12 MB, from
`raw.githubusercontent.com`; nothing is downloaded at run time, because a task that touches the
network cannot be trusted to be reproducible):

```bash
python scripts/fetch_llm_data.py
python -m astro.bench.run --task gpt_finetune --trials 16 --seeds 5
```

On a GPU, the same protocol at sizes where the comparison is worth making:

```bash
python scripts/bench_gpu.py     --task finetune-convnext --data DIR   # real backbone
python scripts/bench_gpu_llm.py --task gpt_finetune --size gpt2-small # real GPT-2, real BPE
```

The protocol enforces equal tuning budgets in code: `astro.bench.protocol.tune` raises if the
optimizers under comparison do not all tune the same number of hyperparameters. That is the
single most common way optimizer results are inflated, so it is a hard error rather than a
convention.

## Focused research validation

The next-stage experimental instructions are deliberately separated from the general benchmark:

- [`docs/RESEARCH_RELEASE_PLAN.md`](docs/RESEARCH_RELEASE_PLAN.md) — the seven-day release plan and decision gates;
- [`docs/MODDED_NANOGPT_VALIDATION.md`](docs/MODDED_NANOGPT_VALIDATION.md) — the T4-safe 124M Transformer validation protocol;
- [`scripts/astro_lab.py`](scripts/astro_lab.py) — shared-configuration and ablation harness.

The modded-nanoGPT validation is derived from the public optimization track, but T4 experiments are reported as controlled research validation rather than official speedrun submissions unless they satisfy that benchmark's published rules. citeturn362378search0turn362378search4

## Rebuild the paper

```bash
pip install -e ".[paper]"      # needs Node for KaTeX, and a Chromium binary
python scripts/build_paper.py
```

No TeX distribution required: markdown → KaTeX → MathML → headless Chromium print-to-PDF.
Set `CHROMIUM_BINARY` if Chromium is not on the usual paths.

## Re-fit the dead-zone filter

```bash
pip install -e ".[design]"
python -m astro.polar 0.1 10   # tau, steps -> coefficients + pass/stop-band diagnostics
```

Minutes, not seconds — it is a multi-start global optimisation. The shipped coefficients are
cached in `polar.py`, so training never runs the solver.
