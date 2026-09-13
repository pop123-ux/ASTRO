# ASTRO: Auditing and Improving Matrix-Structured Optimizers for Language Model Training

**Pop Alexandru · September 2026 · living technical report**  
Code, experiment states, plotting scripts, and LaTeX source live in this repository.

> **Current status.** The 124M shared-configuration horizon is complete through **900 steps**. The 2700-step cell is pending. Seven frozen-config Muon seeds are measured while matched ASTRO-v2 seeds remain pending. A clean 355M rerun is also pending because the first partial 355M run used a different `astro_lab.py` fingerprint and is excluded from paper-facing evidence.

---

## Abstract

Matrix-structured optimizers such as **Muon**, **NorMuon**, and **AdaMuon** transform matrix-valued updates rather than treating every parameter independently. This report documents the development of **ASTRO**, a Muon-family optimizer, together with an experimental framework for testing whether an apparent optimizer improvement survives shared hyperparameters, longer horizons, independent seeds, model scale, and wall-clock accounting.

The current **ASTRO-v2** recipe keeps Muon-style spectral updates, splits fused Q/K/V blocks before the polar transform, applies a norm-preserving row-wise second moment after that transform, and routes non-operator parameters through an Adam-like scalar path. At GPT-2 124M and 900 steps, the now-complete five-setting shared grid gives a mean paired validation-loss difference of **−0.0816** versus Muon, with ASTRO-v2 better in **3/5** settings. The best ASTRO-v2 setting in that grid reaches **5.8102** validation loss versus Muon’s best **5.9286**, although those minima occur at different hyperparameters and are therefore not a paired comparison.

A separate targeted three-setting mechanism study gives mean paired deltas of **−0.2208** for ASTRO-v2, **−0.2137** for ASTRO-MB, and **−0.2047** when the post-polar variance power is removed. The same study measures roughly **6.4% more wall-clock** per 900-step ASTRO-v2 run than Muon.

The evidence is promising but not yet a universal optimizer claim. The earlier ASTRO recipe loses to Muon in a tuned 300-step benchmark; the 2700-step horizon is unfinished; seven Muon replication seeds are measured while matched ASTRO-v2 seeds are pending; and a clean 355M comparison must be rerun under a matching code fingerprint. The emerging hypothesis is therefore narrower: **ASTRO-v2 may improve robustness to hyperparameter choice, while a simpler subset of its recipe carries most of the gain.**

---

## 1. Motivation

Muon made a simple idea practical: hidden weight matrices can be optimized as matrices rather than only as collections of scalar coordinates. Later work strengthened Muon’s scaling behavior and added adaptivity after orthogonalization. That makes the baseline landscape much stronger than “AdamW versus a new optimizer.”

It also makes optimizer research unusually easy to overstate. A method can appear better because its learning-rate range is better centered, because an auxiliary Adam-like parameter group was tuned differently, because the comparison ended at a favorable intermediate point, or because the method simply performs more computation per step.

ASTRO began as an attempt to add additional control to Muon-family updates. The project changed direction as experiments invalidated parts of the original explanation. This paper therefore treats the **research path itself** as part of the artifact: negative runs are retained, shared configurations are distinguished from shared seeds, code fingerprints are recorded, incomplete cells remain visibly incomplete, and step-matched loss is separated from compute-matched efficiency.

### Contributions

1. A PyTorch implementation of ASTRO that subclasses `torch.optim.Optimizer` for state and parameter-group management while implementing the matrix update explicitly.
2. A modern 124M comparison to Muon, NorMuon, and AdaMuon, including the path from earlier ASTRO variants to ASTRO-MB and ASTRO-v2.
3. A completed 300/600/900-step shared-configuration horizon showing negative mean paired deltas at all three measured horizons, without hiding settings where ASTRO-v2 loses.
4. A mechanism study showing that ASTRO-MB captures most of the current ASTRO-v2 gain and that removing variance-power adaptation preserves most of it.
5. A reproducible paper pipeline in which committed JSON is the evidence layer, Matplotlib emits vector figures, W&B is optional, and CI compiles the paper.

---

## 2. Current ASTRO-v2

### 2.1 Routing

The reference implementation is `class Astro(torch.optim.Optimizer)` in `scripts/astro_lab.py`.

- hidden 2-D operators → **spectral path**;
- embeddings, tied output head, biases, norms, and other non-operators → **Adam-like scalar path**;
- GPT-2 `Conv1D` weights are interpreted in operator orientation;
- fused attention projections can be split into Q, K, and V blocks before the spectral transform.

### 2.2 Spectral path

For a matrix gradient \(G_t\), ASTRO maintains momentum

\[
M_t = \beta_1 M_{t-1} + (1-\beta_1)G_t.
\]

After the approximate polar transform produces \(Q_t\), ASTRO tracks a row-wise second moment

\[
v_t = \beta_2 v_{t-1} + (1-\beta_2)\operatorname{rowmean}(Q_t^2),
\]

redistributes the orthogonalized direction,

\[
\widetilde Q_t = Q_t \oslash \widehat v_t^{\gamma/2},
\]

and restores the original Frobenius norm,

\[
D_t = \widetilde Q_t\frac{\lVert Q_t\rVert_F}{\lVert\widetilde Q_t\rVert_F}.
\]

The intended effect is therefore mainly to change **where update energy goes**, rather than simply making the global step larger.

### 2.3 V2 is a recipe, not a second optimizer class

The harness instantiates the same `Astro` class with:

```python
"astro_v2": {
    "betas": (0.9, 0.95),
    "cautious_wd": False,
}
```

The earlier recipe used `beta1=0.95` on both spectral and scalar groups, even though Muon’s auxiliary Adam-like path uses `0.9`. ASTRO-v2 removes that asymmetry and disables cautious weight decay.

`astro_muon_betas` (**ASTRO-MB**) isolates beta alignment. `astro_v2_gamma0` additionally removes the post-polar variance-power effect.

---

## 3. Experimental protocol

The principal experiments train GPT-2-style causal language models from scratch on FineWeb-Edu with GPT-2 BPE. The main model is approximately 124M parameters, with 12 layers, 12 heads, width 768, and sequence length 512. Validation is pinned to the same first **400,000 tokens** at every horizon.

We distinguish two experimental units:

- **shared configurations** — optimizers see the same learning rate, weight decay, and scalar-path multiplier; this measures sensitivity and robustness;
- **shared seeds** — frozen optimizer-specific configurations are rerun under identical stochastic seeds; this measures reproducibility.

Three configurations are not three seeds.

The current 124M horizon and replication evidence use `astro_lab.py` fingerprint **`7b321ff7fda3`**. A partial 355M run used **`56581ade7398`**. Because the fingerprints differ, the partial 355M numbers are retained only as a research log and are not used in the scale figure.

---

## 4. Results

### 4.1 Negative evidence: earlier ASTRO at 300 steps

| Optimizer | Mean validation loss | Δ vs Muon |
|---|---:|---:|
| **NorMuon** | **6.6280** | **−0.0077** |
| Muon | 6.6357 | 0.0000 |
| Earlier ASTRO | 6.6538 | +0.0180 |
| AdamW | 6.8445 | +0.2087 |

The earlier ASTRO recipe does **not** beat Muon in this tuned short-horizon benchmark.

### 4.2 Optimizer journey at one shared 124M/900 setting

At one shared configuration:

| Optimizer | Validation loss | Δ vs Muon |
|---|---:|---:|
| **ASTRO-v2** | **5.9208** | **−0.0827** |
| ASTRO-MB | 5.9226 | −0.0809 |
| AdaMuon | 6.0028 | −0.0007 |
| Muon | 6.0035 | 0.0000 |
| NorMuon | 6.0155 | +0.0121 |

The publication figure is generated by `scripts/figures/fig_optimizer_journey.py` and intentionally shows the progression rather than only the final winner.

### 4.3 Complete five-setting 124M/900 grid

The shared-grid 900-step cell is now complete.

| Quantity | Muon | ASTRO-v2 |
|---|---:|---:|
| Best validation loss in the five-setting grid | 5.9286 | **5.8102** |
| Settings won | 2/5 | **3/5** |
| Mean paired Δ ASTRO-v2 − Muon |  | **−0.0816** |

The individual paired deltas are approximately:

```text
config 1   -0.091
config 2   +0.067
config 3   -0.108
config 4   +0.006
config 5   -0.283
```

The new dumbbell plot `fig_paired_900` shows every pairing directly. This matters because the result is clearly **not** a uniform shift: two settings are neutral or unfavorable while others are substantially better.

> **Current interpretation:** the strongest evidence is consistent with improved **hyperparameter robustness** rather than a claim that ASTRO-v2 dominates Muon at every setting.

### 4.4 Targeted mechanism study

| Variant | Mean paired Δ vs Muon | Settings won |
|---|---:|---:|
| **ASTRO-v2** | **−0.2208** | **3/3** |
| ASTRO-MB | −0.2137 | 3/3 |
| ASTRO-v2 (`gamma=0`) | −0.2047 | 3/3 |

The largest gap occurs around `lr≈0.0505`, where Muon reaches **6.4527** while ASTRO-v2 remains at **5.9436**.

Most of the gain survives removal of variance-power adaptation and is already present in ASTRO-MB. The most mathematically distinctive part of ASTRO-v2 is therefore not automatically the part responsible for the observed improvement.

### 4.5 Horizon through 900 steps

| Steps | Completed settings | Mean paired Δ | Settings won |
|---:|---:|---:|---:|
| 300 | 5/5 | −0.0500 | 2/5 |
| 600 | 5/5 | −0.0541 | 3/5 |
| 900 | 5/5 | **−0.0816** | 3/5 |
| 2700 | 0/5 | pending | pending |

The paper figure uses small points for individual configurations and diamonds for cell means. The 2700 marker is deliberately open and grey: it is an experiment slot, not an estimate.

### 4.6 Compute cost

In the targeted 124M/900 study:

```text
Muon       ≈ 787 s / run
ASTRO-v2   ≈ 838 s / run
```

That is roughly **6.4% wall-clock overhead** for ASTRO-v2.

Therefore the current claim is **better loss at equal steps**, not “faster training.” A time-matched trajectory or time-to-target experiment is required before making an efficiency claim.

### 4.7 Independent-seed replication: half complete

Seven Muon seeds are already measured under a frozen configuration:

```text
100  6.0043
101  5.9831
102  6.0032
103  6.0067
104  5.9893
105  5.9680
106  6.0124
```

Mean: **5.9953**  
Sample SD: **0.0158**

Matched ASTRO-v2 seeds are still pending, so the paper makes **no paired seed claim yet**. `fig_seed_replication.py` is designed to render the Muon distribution now and automatically switch to paired seed connectors once ASTRO-v2 values are added.

### 4.8 Scale: intentionally not inferred

A partial 355M run exists, but it used a different source fingerprint from the 124M evidence. It is excluded from the paper-facing scale result.

The `fig_scale_status.py` plot therefore contains:

```text
124M   measured
355M   clean rerun pending
774M   not started
```

Open markers are experiment slots, **not extrapolations**.

---

## 5. Fused QKV allocation

| Checkpoint | Fused Q/K/V | Split Q/K/V |
|---|---|---|
| GPT-2 | .244/.226/**.530** | .318/.347/.336 |
| GPT-2 Medium | .245/.266/**.489** | .308/.369/.323 |
| Pythia-410M | .218/.248/**.534** | .356/.300/.344 |

The corresponding gradient V/Q ratios are approximately **2.65×**, **3.27×**, and **3.28×**. Random-initialization controls also show V-skew, so training is not claimed as the sole cause.

The defensible statement is narrower: **fused spectral treatment can produce strongly imbalanced Q/K/V allocation, while block-wise treatment restores substantially more balanced allocation.**

---

## 6. Research journey and corrections

**Shared learning rate was not enough.** Early runs forced Muon and ASTRO to share one learning rate despite different spectral and scalar-path calibration. Opposite conclusions appeared at different fixed rates, motivating optimizer-specific tuning and shared-configuration robustness analysis.

**The scalar path mattered more than expected.** ASTRO originally used `beta1=0.95` for both matrix and scalar groups, whereas Muon’s auxiliary Adam path uses `0.9`. Aligning this behavior produced a larger change than several more elaborate spectral modifications.

**Cautious weight decay was not a free safety mechanism.** A masked decay rule changes effective regularization unless survivors are rescaled. ASTRO-v2 disables it rather than treating it as a harmless default.

**Variance power is not yet the headline mechanism.** Setting `gamma=0` loses only about 0.016 validation loss on average relative to full ASTRO-v2 in the targeted run while preserving most of the advantage versus Muon.

**Provenance is part of the experiment.** The mismatched partial 355M run stays in the research log but is excluded from the paper-facing scale figure.

---

## 7. What remains before submission

1. Finish all five **124M/2700** shared settings.
2. Run **ASTRO-v2 seeds 100–106** to pair with the existing seven Muon seeds.
3. Complete a same-fingerprint **355M/900** paired scale study.
4. Generate frozen validation trajectories for **loss vs step** and **loss vs wall-clock**.

Until those are finished, ASTRO-v2 should be described as **promising, not proven**.

---

## 8. Figure workflow

The paper uses Matplotlib as the reproducible source of record. Every figure writes:

```text
artifacts/figures/<name>.png   preview
artifacts/figures/<name>.pdf   vector paper figure
artifacts/figures/<name>.json  exact plotted numbers
```

Current paper-facing figure scripts:

```text
fig_optimizer_journey.py
fig_paired_900.py
fig_horizon_v2.py
fig_efficiency_v2.py
fig_seed_replication.py
fig_scale_status.py
fig_training_trajectories.py
```

W&B may be used for interactive exploration, but the paper does not depend on W&B exports for reproducibility.

---

## 9. Conclusion

ASTRO-v2 is currently **promising, not proven**. The strongest new result is a completed 124M/900 shared grid with a mean paired delta of **−0.0816** versus Muon and **3/5** settings favoring ASTRO-v2. A separate targeted study suggests that most of the gain is already present in a simpler beta-aligned variant and that the largest separation occurs in an aggressive learning-rate regime.

That evidence supports a testable robustness hypothesis more strongly than a claim of universal optimizer dominance.

The final paper will show the journey, not only the winner. Negative results, protocol corrections, provenance mismatches, compute overhead, and unfinished cells remain visible because an optimizer comparison is most useful when it is allowed to fail.
