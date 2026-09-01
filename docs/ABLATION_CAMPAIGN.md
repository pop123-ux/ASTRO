# ASTRO Ablation Campaign

This document defines the expanded optimizer evaluation suite added after the initial 124M result. Its purpose is to make a narrow win difficult to dismiss as a hyperparameter accident, while preserving negative and null results as first-class evidence.

## Research standard

The paper should distinguish four claims:

1. **Configuration robustness:** ASTRO remains competitive across a broad, pre-specified configuration region rather than at one selected point.
2. **Component attribution:** individual ASTRO changes have separately measurable effects (or are shown to be unnecessary).
3. **Transfer:** a configuration selected on one cell is tested without retuning across seeds, training horizons, and model scales.
4. **End-to-end usefulness:** the result survives on a real GPT-style language model rather than only synthetic or optimizer-specific diagnostics.

No claim of universal SOTA is made until a baseline has been tested with a reasonable configuration budget. Conversely, a baseline is not declared weaker merely because its single default loses.

## Optimizers in scope

### Primary baselines

- Muon
- NorMuon
- AdaMuon
- SOAP

### ASTRO variants

The campaign exercises the variants already exposed by `astro_lab.py`:

| Variant | Hypothesis tested |
|---|---|
| `astro` | complete current recipe |
| `astro_pinned` | fixed post-normalized update norm |
| `astro_trust` | layer-norm/trust-region scaling instead of Muon aspect scaling |
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

The exact variant semantics remain in `scripts/astro_lab.py`; the launcher does not duplicate optimizer code.

## Suite A: dense configuration coverage

**Target:** 124M GPT-2-shaped model, 900 optimization steps, seed 0.

For the principal baselines and major ASTRO variants, the launcher enumerates:

- learning rate: `0.003, 0.006, 0.012, 0.024, 0.048, 0.096`
- weight decay: `0, 0.003, 0.01, 0.03, 0.1`
- scalar-path multiplier: `0.25, 0.35, 0.4369, 0.55, 0.75`

That is 150 explicit points per optimizer before continuous tuning. The purpose is to expose whether an apparent advantage only occupies a thin ridge.

The suite then gives Muon, NorMuon, AdaMuon, ASTRO, `astro_muon_betas`, `astro_v2`, and `astro_trust` a separate 12-trial tuning budget.

The dense grid is a sensitivity map, not the final held-out result. The final statistical claim must freeze a configuration and evaluate it on fresh seeds.

## Suite B: component ablations

**Target:** 124M; 300, 900 and 2700 steps; three held-out seeds.

The same nominal configuration is used to isolate structural differences:

```text
Muon
NorMuon
AdaMuon
ASTRO
ASTRO + beta asymmetry
ASTRO +/− cautious masking
ASTRO +/− cautious weight decay
ASTRO +/− decay rescaling
ASTRO + convergent polar schedule
ASTRO + row equilibration
ASTRO + trust scaling
ASTRO + fused QKV split
ASTRO + scheduled QKV split
ASTRO + variance-power changes
```

This produces horizon curves rather than one ablation bar. That matters because the repository already contains a scale-dependent sign inversion for cautious masking.

## Suite C: learning-rate sensitivity

The nominal 124M reference LR is multiplied by:

```text
0.25, 0.5, 0.75, 1.0, 1.33, 2.0, 4.0
```

for Muon, NorMuon, AdaMuon, `astro_v2`, and `astro_muon_betas`, on three seeds. A gain that survives a factor-of-four LR perturbation is qualitatively different from a gain at one exact point.

## Suite D: scale × horizon transfer

The symmetric matrix is:

```text
models: 45M, 124M, 355M, 774M
steps:  300, 900, 2700
seeds:  100, 101, 102
```

for Muon, NorMuon, AdaMuon, SOAP, ASTRO, ASTRO-Muon-Betas, and ASTRO-v2.

The configuration is frozen from the discovery stage. Because batch size decreases with model size on a 16 GB T4, the paper must report **tokens seen** as well as steps.

## Suite E: stability envelope

124M and 2700 steps probe LR from `0.0015` through `0.192` and weight decay `0, 0.01, 0.1, 0.3` for Muon, NorMuon, AdaMuon, ASTRO, ASTRO-Muon-Betas, and ASTRO-v2.

This is not a leaderboard. It records divergence, instability, and sensitivity so the paper can state whether ASTRO widens or narrows the stable region.

## Statistical protocol

Do not treat many configurations as independent confirmation of one hypothesis. A large search creates multiple-comparisons pressure. The paper should report the complete search space, selection rule, held-out seeds, per-seed values, paired deltas where valid, worst-case seed, and uncertainty/variance estimates.

A sign test is useful for transparent paired evidence, but three seeds are insufficient for strong statistical claims by themselves. The final manuscript should emphasize effect size and replication as well.

## What counts as convincing?

The strongest evidence chain is:

```text
broad configuration region
        ↓
repeatable component effect
        ↓
mechanistic measurement agrees
        ↓
frozen-config transfer across seeds
        ↓
benefit survives end-to-end GPT-2 training
```

A negative result is also useful. If ASTRO wins at 124M and disappears at 355M, that establishes a regime boundary rather than invalidating the work.

## Three-Colab deployment

Mount Drive once per runtime:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then launch disjoint shards with the same persistent root. The launcher records an auditable manifest and skips completed jobs on rerun.

```bash
# Colab 1 — primary configuration coverage
python scripts/ablation_suite.py --suite primary --shard 0/3 \
  --work-root /content/drive/MyDrive/astro/ablation --max-jobs 18

# Colab 2 — component/mechanism coverage
python scripts/ablation_suite.py --suite mechanisms --shard 1/3 \
  --work-root /content/drive/MyDrive/astro/ablation --max-jobs 18

# Colab 3 — scale/horizon transfer
python scripts/ablation_suite.py --suite robustness --shard 2/3 \
  --work-root /content/drive/MyDrive/astro/ablation --max-jobs 18
```

The `18` is only a session-management cap; adjust it after measuring actual T4 runtime. More jobs are intentionally defined than a single four-hour window can finish.

## Paper tables enabled

The campaign is designed to populate:

1. Main 124M comparison: tuned Muon / NorMuon / AdaMuon / SOAP / ASTRO.
2. Configuration sensitivity: dense loss maps or percentile summaries.
3. Component ablation: ASTRO variants at 300/900/2700 steps.
4. Scale transfer: 45M → 774M at frozen configuration.
5. Horizon transfer: 300 → 2700 steps.
6. Stability envelope: divergence/sensitivity region.
7. Compute efficiency: wall-clock and tokens/sec alongside validation loss.
8. Failure analysis: explicit negative results and regime boundaries.

No headline claim should be selected until these artifacts are inspected together.
