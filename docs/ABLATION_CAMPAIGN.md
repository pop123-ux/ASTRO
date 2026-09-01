# ASTRO Ablation Campaign

This document defines the expanded optimizer evaluation suite. Its purpose is to make a narrow result difficult to dismiss as a hyperparameter accident while preserving negative and null results as first-class evidence.

## Research standard

The paper should distinguish four claims:

1. **Configuration robustness:** ASTRO remains competitive across a broad, pre-specified configuration region rather than at one selected point.
2. **Component attribution:** individual ASTRO changes have separately measurable effects or are shown to be unnecessary.
3. **Transfer:** a configuration selected on one cell is tested without retuning across seeds, training horizons, and model scales.
4. **End-to-end usefulness:** the result survives on a real GPT-style language model rather than only optimizer diagnostics.

The launcher never calls a method by a name that is not currently wired into `astro_lab.py`. The current direct comparison surface is AdamW, Muon, NorMuon and AdaMuon plus 17 ASTRO variants. SOAP and other optimizers remain candidates for a later adapter once their exact benchmark interface is pinned.

## Optimizers in scope

### Direct baselines

- AdamW
- Muon
- NorMuon
- AdaMuon

### ASTRO variants

| Variant | Hypothesis tested |
|---|---|
| `astro` | complete current recipe |
| `astro_pinned` | fixed post-normalized update norm |
| `astro_trust` | layer-norm/trust-region scaling |
| `astro_cautious` | cautious gradient-sign mask |
| `astro_converging` | convergent per-step polar coefficients |
| `astro_gamma0` | removes the adaptive row-moment effect |
| `astro_gamma25` | quarter-strength row moment |
| `astro_gamma50` | half-strength row moment |
| `astro_equil` | momentum row-norm equilibration |
| `astro_plain_wd` | ordinary decay instead of cautious decay |
| `astro_wd_rescaled` | cautious decay with survivor-count rescaling |
| `astro_muon_betas` | Muon-compatible `(0.9, 0.95)` beta pair on scalar path |
| `astro_v2` | current beta-asymmetry recipe with plain decay |
| `astro_v2_gamma0` | `astro_v2` with variance power 0 |
| `astro_nosplit` | fused QKV orthogonalized as one block |
| `astro_split100` | QKV split only through step 100 |
| `astro_split300` | QKV split only through step 300 |

The exact implementation remains single-sourced in `scripts/astro_lab.py`.

## Suite A: dense configuration coverage

**Target:** 124M GPT-2-shaped model, 900 steps, seed 0.

The direct baselines and 14 major ASTRO variants are exposed to a dense grid:

- learning rate: `0.003, 0.006, 0.012, 0.024, 0.048, 0.096`
- weight decay: `0, 0.003, 0.01, 0.03, 0.1`
- scalar-path multiplier: `0.25, 0.35, 0.4369, 0.55, 0.75`

That is 150 explicit points per method in the dense layer. The purpose is to expose whether an advantage occupies a region or one narrow ridge.

A second layer gives AdamW, Muon, NorMuon, AdaMuon, `astro`, `astro_muon_betas`, `astro_v2`, and `astro_trust` a 12-trial native tuning control.

## Suite B: component ablations

**Target:** 124M; 300, 900 and 2700 steps; seeds 100, 101, 102.

The same nominal recipe is used to isolate:

```text
AdamW / Muon / NorMuon / AdaMuon
ASTRO complete recipe
beta asymmetry
cautious masking
cautious weight decay
weight-decay rescaling
convergent polar schedule
row equilibration
trust scaling
fused QKV split
scheduled QKV split
variance-power interpolation
```

The horizon dimension is intentional. A component that helps at 300 steps but hurts at 2700 is a regime-dependent effect, not a universal improvement.

## Suite C: learning-rate robustness

The 124M reference learning rate is multiplied by:

```text
0.25, 0.5, 0.75, 1.0, 1.33, 2.0, 4.0
```

for Muon, NorMuon, AdaMuon, `astro_v2`, and `astro_muon_betas`, each on three seeds.

## Suite D: scale × horizon transfer

The symmetric matrix is:

```text
models: 45M, 124M, 355M, 774M
steps:  300, 900, 2700
seeds:  100, 101, 102
```

for AdamW, Muon, NorMuon, AdaMuon, ASTRO, ASTRO-Muon-Betas, and ASTRO-v2.

The configuration is frozen from the discovery protocol. The resulting tables must report tokens seen and wall-clock time, not only optimizer steps.

## Suite E: stability envelope

124M and 2700 steps probe:

- LR `0.0015, 0.003, 0.006, 0.012, 0.024, 0.048, 0.096, 0.192`;
- weight decay `0, 0.01, 0.1, 0.3`;
- AdamW, Muon, NorMuon, AdaMuon, ASTRO, ASTRO-Muon-Betas and ASTRO-v2.

The output is a stability region, not a leaderboard: divergence and excessive sensitivity are useful findings.

## Statistical protocol

Do not treat hundreds of searched configurations as hundreds of independent confirmations. The final manuscript should report the search space, selection rule, held-out seeds, per-seed values, effect sizes, worst-case seed, and uncertainty.

The key separation is:

```text
DISCOVERY: broad search on tuning data
        ↓
FREEZE: selected optimizer configuration
        ↓
EVALUATION: fresh seeds and transfer cells
```

The paper should never select a configuration after looking at the final held-out seeds.

## Three-Colab deployment

Mount Drive once per runtime:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then use disjoint shards with a shared persistent root:

```bash
# Instance 1 — direct baseline + ASTRO configuration coverage
python scripts/ablation_suite.py --suite primary --shard 0/3 \
  --work-root /content/drive/MyDrive/astro/ablation --max-jobs 18

# Instance 2 — component and horizon ablations
python scripts/ablation_suite.py --suite mechanisms --shard 1/3 \
  --work-root /content/drive/MyDrive/astro/ablation --max-jobs 18

# Instance 3 — scale/horizon transfer
python scripts/ablation_suite.py --suite robustness --shard 2/3 \
  --work-root /content/drive/MyDrive/astro/ablation --max-jobs 18
```

The `18` is a session-management cap. After measuring actual T4 runtime, increase it or rerun the same shard. Every successful job is recorded in `ablation_audit.jsonl`.

## Aggregating results

Once jobs accumulate under the persistent root, build one paper-ready summary without using the GPU:

```bash
python scripts/analyze_ablation.py \
  --root /content/drive/MyDrive/astro/ablation
```

This produces `ablation_summary.json` and `ablation_summary.md` containing normalized records, aggregate statistics, coverage, and read warnings.

## Paper tables enabled

The campaign is designed to populate:

1. Main 124M comparison: tuned AdamW / Muon / NorMuon / AdaMuon / ASTRO.
2. Configuration sensitivity: dense loss surfaces and percentile summaries.
3. Component ablation: ASTRO variants across 300/900/2700 steps.
4. Scale transfer: 45M → 774M at frozen configuration.
5. Horizon transfer: 300 → 2700 steps.
6. Stability envelope: divergence/sensitivity region.
7. Compute efficiency: wall-clock and tokens/sec alongside loss.
8. Failure analysis: explicit negative results and regime boundaries.

No headline claim should be selected until the search, held-out evaluation and failure cases have all been inspected together.
