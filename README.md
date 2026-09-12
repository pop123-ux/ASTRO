# ASTRO

**ASTRO is a matrix-structured optimizer research project built around a simple question:**
which apparent improvements over Muon survive fair tuning, longer training, independent seeds,
larger models, and wall-clock accounting?

The repository contains the current ASTRO implementation used in experiments, Muon-family
baselines, a resumable Colab/T4 benchmark harness, the complete experiment ledger, and a
reproducible paper/figure pipeline.

> **Research status — September 2026:** ASTRO-v2 is promising at GPT-2 124M / 900 steps, but the
> headline is not frozen. Horizon, independent-seed replication, and the first 355M scale check
> are in progress. The manuscript deliberately marks unfinished cells as unfinished.

Start here:

- [`docs/paper/paper.md`](docs/paper/paper.md) — readable living manuscript;
- [`docs/paper/paper.tex`](docs/paper/paper.tex) — two-column LaTeX source;
- [`docs/PAPER_WORKFLOW.md`](docs/PAPER_WORKFLOW.md) — experiment → figure → paper workflow;
- [`docs/FIGURES.md`](docs/FIGURES.md) — visual conventions and figure inventory;
- [`scripts/astro_lab.py`](scripts/astro_lab.py) — current self-contained optimizer/benchmark lab.

---

## Current empirical picture

### Targeted 124M / 900-step mechanism run

Three shared hyperparameter configurations, one training seed per configuration:

| variant | mean paired Δ vs Muon | configurations won |
|---|---:|---:|
| **ASTRO-v2** | **−0.2208** | **3/3** |
| ASTRO-MB | −0.2137 | 3/3 |
| ASTRO-v2 (`variance_power=0`) | −0.2047 | 3/3 |

This is encouraging, but the unit of replication is **configuration**, not seed. The result is
best read as sensitivity/robustness evidence.

The largest separation occurs at the aggressive configuration `lr ≈ 0.0505`: Muon reaches
`6.4527` validation loss while ASTRO-v2 remains at `5.9436`. That makes **hyperparameter
robustness** a leading hypothesis for the current v2 signal.

ASTRO-v2 takes about **6.4% more wall-clock** than Muon in this targeted 900-step experiment, so
a step-matched loss win is not yet a compute-efficiency win.

### One shared configuration against modern Muon variants

At `lr=0.0102059788`, `weight_decay=0.0010597179`, `scalar_lr_mult=0.4369`:

| optimizer | validation loss | Δ vs Muon |
|---|---:|---:|
| **ASTRO-v2** | **5.9208** | **−0.0827** |
| ASTRO-MB | 5.9226 | −0.0809 |
| AdaMuon | 6.0028 | −0.0007 |
| Muon | 6.0035 | 0.0000 |
| NorMuon | 6.0155 | +0.0121 |

This is a **pointwise shared-configuration comparison**, not a tuned global leaderboard.

### Negative evidence is kept

In the expanded tuned GPT-2 124M / 300-step benchmark, the earlier/plain ASTRO recipe is slightly
*worse* than Muon:

| optimizer | mean validation loss | Δ vs Muon |
|---|---:|---:|
| NorMuon | **6.6280** | **−0.0077** |
| Muon | 6.6357 | 0.0000 |
| earlier ASTRO | 6.6538 | +0.0180 |
| AdamW | 6.8445 | +0.2087 |

The repository does not rewrite this history to make v2 look inevitable.

---

## What ASTRO-v2 currently is

The reference implementation used by the scale campaign lives directly in
[`scripts/astro_lab.py`](scripts/astro_lab.py):

```python
class Astro(torch.optim.Optimizer):
    ...
```

It subclasses `torch.optim.Optimizer` for PyTorch's parameter-group/state interface, but its
actual update rule is implemented explicitly in this repository.

The current spectral path includes:

1. Nesterov-style momentum;
2. optional Q/K/V block splitting for fused attention projections;
3. a Newton–Schulz/polar transform;
4. a neuron-wise second moment on the **post-polar** direction;
5. norm-preserving adaptive redistribution;
6. Muon-style aspect-ratio update scaling;
7. a separate Adam-like scalar path for embeddings, tied heads, norms, biases, and other
   non-operator parameters.

ASTRO-v2 is a **recipe of the same `Astro` class**, not a separate class:

```python
"astro_v2": {
    "betas": (0.9, 0.95),
    "cautious_wd": False,
}
```

The two changes matter because earlier ASTRO used `beta1=0.95` on the scalar path even though
Muon's auxiliary Adam path uses `0.9`, and cautious weight decay changes the effective amount of
decay by masking coordinates.

`astro_muon_betas` isolates the beta alignment. `astro_v2_gamma0` additionally removes the
post-polar variance-power effect. Their near-tie with v2 is why the paper does **not** currently
claim that variance adaptation is the main mechanism.

---

## Research questions now being tested

```text
HORIZON
Does the v2 effect survive 300 → 600 → 900 → 2700 steps?

REPLICATION
Does a frozen 124M/900 result survive independent random seeds?

SCALE
Does the result survive 124M → 355M before we spend money on larger GPUs?

COMPUTE
Does the lower step-matched loss compensate for ASTRO-v2's extra per-step cost?

MECHANISM
Which part of v2 actually causes the result?
```

The current horizon state is complete at 300 and 600 steps, partial at 900, and not yet measured
at 2700 under the new v2 campaign. See `artifacts/paper_results.json` for the current paper-facing
snapshot.

**CLI seed semantics:** in `astro_lab.py`, `--seeds` is a **count** beginning at seed 100.
Therefore `--seeds 3` evaluates seeds 100, 101, and 102. `--seeds 100` means one hundred seeds.

---

## Mechanistic evidence: fused QKV allocation

The project also studies what the spectral transform does inside fused attention projections.
Examples from checkpoint probes:

| checkpoint | fused Q/K/V | split Q/K/V |
|---|---|---|
| GPT-2 | 0.244 / 0.226 / **0.530** | 0.318 / 0.347 / 0.336 |
| GPT-2 Medium | 0.245 / 0.266 / **0.489** | 0.308 / 0.369 / 0.323 |
| Pythia-410M | 0.218 / 0.248 / **0.534** | 0.356 / 0.300 / 0.344 |

Splitting Q, K, and V before the polar transform substantially balances the allocation. Random
initialization controls also show V-skew, so the claim is **not** that training creates the entire
effect. The defensible finding is that fused treatment can produce strongly imbalanced allocation
and block-wise treatment repairs much of it.

Splitting costs extra compute at width, which is why scheduled splitting remains an open
experiment rather than a free default.

---

## Repository layout

```text
ASTRO/
├── README.md
├── Makefile
├── pyproject.toml
├── artifacts/
│   ├── measured.json          # historical experiment ledger; never erase
│   └── paper_results.json     # current paper-facing evidence snapshot
├── docs/
│   ├── FIGURES.md
│   ├── PAPER_WORKFLOW.md
│   ├── RESEARCH_RELEASE_PLAN.md
│   ├── MODDED_NANOGPT_VALIDATION.md
│   └── paper/
│       ├── paper.md
│       ├── paper.tex
│       └── paper.pdf          # regenerated when a TeX build is available
├── scripts/
│   ├── astro_lab.py           # current self-contained ASTRO + baselines + T4 harness
│   ├── figures/               # publication Matplotlib scripts
│   └── paper/
│       ├── run_trajectory.py  # periodic validation for learning-curve figures
│       ├── wandb_sync.py      # optional W&B bridge
│       └── build.py           # figures + LaTeX build
├── src/astro/
│   ├── baselines/
│   └── bench/
└── tests/
```

**Important packaging note:** the current research implementation is vendored/self-contained in
`scripts/astro_lab.py`. A clean reusable `src/astro/optimizer.py` API is a release task, not
something this README claims already exists.

---

## Run the T4 research harness

A fixed shared configuration:

```bash
python scripts/astro_lab.py \
  --mode scaling \
  --sizes 124M \
  --steps 900 \
  --optimizers muon normuon adamuon astro_muon_betas astro_v2 \
  --config lr=0.010205978810672643 \
           weight_decay=0.001059717889298923 \
           scalar_lr_mult=0.4369 \
  --seeds 3 \
  --work-dir /content/drive/MyDrive/astro/frozen_124m_900
```

A shared-configuration grid with no seed evaluation:

```bash
python scripts/astro_lab.py \
  --mode scaling \
  --sizes 124M \
  --steps 900 \
  --optimizers muon normuon adamuon astro_muon_betas astro_v2 \
  --trials 5 \
  --seeds 0 \
  --pin scalar_lr_mult=0.4369 \
  --work-dir /content/drive/MyDrive/astro/shared_grid
```

Completed runs are written immediately to `astro_lab_state.json`, so a reclaimed Colab session
can resume against the same persistent work directory.

---

## Paper figures

The source of record is **committed JSON + Matplotlib**, not a manually edited chart and not a
W&B screenshot.

Install figure dependencies:

```bash
pip install -e '.[paper]'
```

Build all figures:

```bash
python scripts/figures/make_all.py
```

Build only the current empirical spine:

```bash
python scripts/figures/make_all.py \
  --only optimizer_journey horizon_v2 efficiency_v2 training_trajectories
```

Every figure writes:

```text
artifacts/figures/<figure>.png
artifacts/figures/<figure>.pdf
artifacts/figures/<figure>.json
```

The vector PDF is what LaTeX uses.

### Validation-loss trajectories

After configurations are frozen:

```bash
python scripts/paper/run_trajectory.py \
  --size 124M --steps 900 --seed 100 --eval-every 75 \
  --optimizers muon normuon adamuon astro_muon_betas astro_v2 \
  --config lr=0.010205978810672643 \
           weight_decay=0.001059717889298923 \
           scalar_lr_mult=0.4369 \
  --cache /content/drive/MyDrive/astro/paper_trajectory_cache.pt \
  --out /content/drive/MyDrive/astro/trajectories.json
```

Then copy the JSON into `artifacts/trajectories.json` and regenerate
`fig_training_trajectories`.

### Optional W&B

```bash
pip install -e '.[tracking]'
wandb login
python scripts/paper/wandb_sync.py --project astro-paper \
  --input artifacts/trajectories.json
```

W&B is for interactive inspection. A reviewer never needs it to rebuild the paper.

---

## Build the paper

```bash
make paper
```

or:

```bash
python scripts/paper/build.py
```

The builder regenerates the primary figures and then uses `latexmk` or `pdflatex`. If an optional
measurement is missing, the living LaTeX draft renders a visible placeholder instead of silently
reusing a stale plot.

The visual style is defined once in `scripts/figures/style.py`: two-column paper sizing, serif
text, vector output, restrained grids, stable optimizer colours, and direct quantitative labels.
See [`docs/FIGURES.md`](docs/FIGURES.md) for the full convention.

---

## Test

```bash
pip install -e '.[paper,dev]'
pytest
python -m compileall scripts src
python scripts/figures/make_all.py --only optimizer_journey horizon_v2 efficiency_v2
```

The paper-facing tests recompute deltas from the raw JSON, assert that incomplete horizon cells
remain marked incomplete, and exercise the current figure builders under a non-interactive
Matplotlib backend.

---

## Research policy

ASTRO is intentionally developed under falsifiable rules:

- a configuration sweep is not called a seed replication;
- a short-horizon win is not extrapolated to longer training;
- a 124M win is not extrapolated to 355M or frontier scale;
- a step-matched win is not called an efficiency win when the step is more expensive;
- a component is not credited merely because its mathematical story sounds attractive;
- negative experiments and protocol errors stay in the repository.

The goal is not to make ASTRO impossible to falsify. The goal is to make it obvious when it has
been falsified — and equally obvious when an effect survives increasingly hard controls.
