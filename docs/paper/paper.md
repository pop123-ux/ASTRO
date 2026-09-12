# ASTRO: Auditing and Improving Matrix-Structured Optimizers for Language Model Training

**Pop Alexandru · September 2026 · living research draft**  
Code, experiment states, plotting scripts, and LaTeX source are contained in this repository.  
**Important:** the 124M horizon, independent-seed replication, and 355M scale checks are still in progress. This draft separates measured findings from pending claims rather than filling unfinished cells with extrapolation.

---

## Abstract

Matrix-structured optimizers such as **Muon**, **NorMuon**, and **AdaMuon** improve language-model training by transforming matrix-valued updates rather than treating every parameter independently. This paper documents the development of **ASTRO**, a Muon-family optimizer and, equally importantly, an experimental framework for determining which apparent optimizer improvements survive fair tuning, longer horizons, larger models, independent seeds, and wall-clock accounting.

The current ASTRO-v2 recipe keeps Muon-style spectral updates, separates fused Q/K/V blocks, applies a neuron-wise second moment after the polar step with norm preservation, routes embeddings and other non-operator parameters through an Adam-like scalar path, and removes two choices that ablation showed were harmful or accidental: the scalar path no longer inherits a spectral-path $\beta_1=0.95$, and cautious weight decay is disabled. At GPT-2 124M and 900 steps, a targeted three-configuration experiment gives mean paired validation-loss differences versus Muon of **−0.2208** for ASTRO-v2, **−0.2137** for the beta-aligned ASTRO-MB variant, and **−0.2047** when the post-polar variance power is removed. ASTRO-v2 wins all three tested shared configurations, but these are configurations, not independent seeds; the corresponding exact sign test is therefore not a significance claim. The same run measures roughly **6.4% wall-clock overhead** for ASTRO-v2 relative to Muon.

The result is promising but deliberately not presented as universal. In an independently tuned 300-step 124M benchmark, the earlier/plain ASTRO recipe is **+0.0180 validation loss worse than Muon**, while NorMuon is slightly better. The current horizon study is complete at 300 and 600 steps but still incomplete at 900 and 2700, independent-seed replication is being rerun under a frozen configuration, and the first 355M check is in progress. This mixed evidence makes the paper’s central question sharper: **is ASTRO-v2 a generally better optimizer, or does it mainly enlarge the region of stable hyperparameters?**

A second line of evidence concerns fused QKV projections. Across GPT-2-family and Pythia checkpoints, treating Q, K, and V as one matrix produces strongly non-uniform update allocation, while splitting the three operators before the polar transform restores substantially more balanced allocation. Random-initialization controls also show V-skew, so we do not attribute the entire effect to training. Together, the experiments motivate a broader methodological conclusion: optimizer research should report the path by which a result was obtained—including negative runs and withdrawn explanations—not only the final best row.

---

## 1. Introduction

Muon changed the optimizer discussion for language-model pretraining by making a simple idea practical: for hidden weight matrices, use the geometry of the matrix update rather than applying only coordinate-wise scaling. Subsequent methods such as NorMuon and AdaMuon add adaptivity after orthogonalization, while large benchmarking studies have shown that optimizer rankings depend strongly on tuning budget, model scale, and the point in training at which the comparison is made.

That makes a new optimizer unusually easy to overstate. A method can appear better because its learning-rate range happened to be better centered, because an auxiliary Adam-like parameter group was tuned differently, because the comparison ended early, because one method took more compute per step, or because a favorable hyperparameter setting was mistaken for stochastic replication.

ASTRO began as an attempt to add additional control to Muon-family updates. The project changed direction as experiments invalidated parts of the original story. The current contribution is therefore twofold:

1. **ASTRO-v2**, a concrete matrix-structured optimizer recipe whose strongest current 124M/900 results are substantially better than Muon under three shared settings;
2. **an auditable research process** that preserves negative evidence, distinguishes configurations from seeds, records wall-clock cost, and makes unfinished experiments visibly unfinished in the paper itself.

The paper is intentionally a living draft. We state what the current evidence licenses and list the experiments required before a final headline can be frozen.

### Contributions

- **A current ASTRO-v2 recipe** implemented directly in PyTorch by subclassing `torch.optim.Optimizer`; the matrix update itself is implemented from scratch rather than delegated to a built-in optimizer.
- **A controlled comparison to modern Muon-family baselines**: Muon, NorMuon, and AdaMuon, with shared-configuration analyses that expose robustness rather than only each optimizer’s best draw.
- **A component-level result** showing that ASTRO-MB captures almost all of ASTRO-v2’s present 124M/900 gain, while setting the post-polar variance power to zero preserves most of it. This weakens the original hypothesis that variance-power adaptation is the main cause.
- **A QKV allocation analysis** showing that fused spectral treatment can strongly bias update mass toward one operator and that splitting restores a much more balanced allocation.
- **A reproducible paper pipeline** in which Matplotlib figures are regenerated from committed JSON evidence, W&B is optional, missing measurements render as missing rather than estimated, and wall-clock is shown alongside step-matched loss.
- **A documented research journey** including runs in which the earlier ASTRO recipe loses to Muon. Those results are evidence about the optimizer, not debris to be deleted.

---

## 2. Background and contemporary baselines

### 2.1 Muon

For a matrix-valued gradient, Muon maintains momentum and applies a small number of Newton–Schulz-style matrix iterations that approximately replace the singular values of the momentum matrix by values near one. The result behaves like an approximate polar factor. Muon is normally used on hidden matrix parameters, while embeddings, biases, normalization parameters, and output heads are handled by an AdamW-like auxiliary path.

This detail matters: a GPT-2 124M model has a substantial fraction of parameters on the auxiliary path. A comparison that changes that path is not “only” comparing two spectral transforms.

### 2.2 NorMuon

NorMuon adds neuron-wise adaptivity **after** orthogonalization. It maintains one second-moment statistic per output row, divides the polar-transformed update by the resulting row scale, and rescales the result so the total update norm is preserved. The method therefore changes the distribution of update energy across neurons without intentionally changing the global step magnitude.

### 2.3 AdaMuon

AdaMuon pushes the same idea to element-wise adaptivity: instead of one second-moment statistic per row, it can maintain one per weight in the orthogonalized update. This makes it a useful baseline for ASTRO because the current experiments suggest that **how adaptation is placed after the spectral transform, and how much freedom it has, may matter for hyperparameter robustness**.

### 2.4 Why horizon and scale are mandatory

Wen et al. show that fair optimizer comparisons require optimizer-specific tuning, evaluation at the intended training budget, and multiple model scales; rankings can change during training and apparent speedups shrink with scale. We therefore treat 124M/300, 124M/900, and 355M as different empirical questions rather than assuming one result transfers automatically.

---

## 3. ASTRO in the current repository

The current Colab/reference implementation is the `Astro` class in `scripts/astro_lab.py`:

```python
class Astro(torch.optim.Optimizer):
    ...
```

Subclassing `torch.optim.Optimizer` supplies parameter groups, optimizer state, serialization, and PyTorch interoperability. The update rule itself—momentum, polar iteration, QKV handling, post-polar variance adaptation, norm restoration, scaling, and weight decay—is explicitly implemented by ASTRO.

### 3.1 Parameter routing

ASTRO first decides whether a parameter represents a genuine linear operator.

- hidden 2-D projection matrices → **spectral path**;
- embeddings, tied output heads, biases, norms, and other non-operator tensors → **scalar/Adam-like path**;
- HuggingFace GPT-2 `Conv1D` projections are interpreted in operator orientation rather than storage orientation;
- fused attention projections can be split into Q, K, and V blocks before the spectral transform.

This structural routing replaces fragile name-only heuristics, which are especially dangerous under GPT-2 weight tying.

### 3.2 Spectral path

For a matrix gradient $G_t$, ASTRO maintains momentum

$$
M_t = \beta_1 M_{t-1} + (1-\beta_1)G_t.
$$

It forms a Nesterov-style lookahead, applies the approximate polar transform block-by-block, and obtains a direction $Q_t$. ASTRO then tracks a row-wise second moment on the **orthogonalized direction**,

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)\,\operatorname{rowmean}(Q_t^2),
$$

and redistributes the direction using

$$
\widetilde Q_t = Q_t \oslash \widehat v_t^{\gamma/2}.
$$

The total norm is restored afterward,

$$
D_t = \widetilde Q_t\frac{\lVert Q_t\rVert_F}{\lVert\widetilde Q_t\rVert_F},
$$

so the variance term primarily changes **where** update energy goes rather than simply increasing or shrinking the whole step.

### 3.3 ASTRO-v2 is a recipe, not a second optimizer class

ASTRO-v2 uses the same `Astro` implementation with a different configuration. In the current experiment harness:

```python
"astro_v2": {
    "betas": (0.9, 0.95),
    "cautious_wd": False,
}
```

The change is small enough to be scientifically useful. The original ASTRO path used $\beta_1=0.95$ on both spectral and scalar groups even though Muon’s Adam-like auxiliary path uses $\beta_1=0.9$. That meant roughly a third of the model could be trained by a different scalar optimizer before any matrix-geometry difference was considered. ASTRO-v2 removes that asymmetry and disables cautious weight decay, which previous measurements showed could act as a hidden change in effective decay strength.

`astro_muon_betas` isolates the beta alignment while retaining cautious weight decay. `astro_v2_gamma0` additionally removes the post-polar variance-power effect. These variants are central because they let the data decide which part of “v2” is actually doing work.

---

## 4. Experimental protocol

### 4.1 Model and data

The current scale campaign uses GPT-2-style causal language models trained from scratch on FineWeb-Edu with GPT-2 BPE. The principal model is GPT-2 124M (12 layers, 12 heads, width 768). The first scaling check uses a 355M configuration. Sequence length is 512 in the T4 campaign.

Validation is pinned to a fixed 400,000-token slice so changing the training horizon cannot silently change the evaluation text.

### 4.2 What is paired

We use two distinct forms of pairing and never merge their interpretation:

- **shared configuration pairing**: Muon and ASTRO see the same learning rate, weight decay, and scalar-path multiplier. This measures sensitivity and robustness across settings;
- **shared seed pairing**: frozen optimizer configurations are rerun with the same stochastic seeds. This measures stochastic reproducibility.

Three configurations are not three seeds.

### 4.3 Current tuning variables

For Muon-family optimizers the T4 harness exposes:

- spectral learning rate;
- weight decay;
- scalar-path learning-rate multiplier.

The scalar multiplier proved particularly important because it controls the auxiliary path containing embeddings and the tied head. Earlier experiments that fixed it near 0.1 were not neutral.

### 4.4 Step-matched and time-matched evidence

A more expensive optimizer can win at a fixed number of steps while losing at a fixed amount of compute. Every final claim must therefore distinguish:

1. validation loss after the same number of training steps;
2. validation loss at matched wall-clock or time-to-target.

The present targeted run measures ASTRO-v2 at about 6.4% more wall-clock than Muon. A full time-matched trajectory is therefore required before calling ASTRO-v2 more *efficient*.

---

## 5. Results: what is established now

### 5.1 Negative evidence: the earlier ASTRO recipe at 300 steps

An expanded tuned end-to-end 124M benchmark gives:

| optimizer | mean validation loss | paired Δ vs Muon |
|---|---:|---:|
| NorMuon | **6.6280** | **−0.0077** |
| Muon | 6.6357 | 0.0000 |
| earlier/plain ASTRO | 6.6538 | +0.0180 |
| AdamW | 6.8445 | +0.2087 |

The evaluation uses seeds 100, 101, and 102. At this short horizon the earlier ASTRO recipe does **not** beat Muon.

> **Observation 1.** The current ASTRO-v2 story cannot be back-projected onto earlier ASTRO runs; at 124M/300, the earlier recipe is worse than Muon under the expanded tuned benchmark.

### 5.2 Modern Muon-family comparison at one 124M/900 configuration

At the shared configuration

```text
lr = 0.010205978810672643
weight_decay = 0.001059717889298923
scalar_lr_mult = 0.4369
```

the measured losses are:

| optimizer | validation loss | Δ vs Muon |
|---|---:|---:|
| **ASTRO-v2** | **5.9208** | **−0.0827** |
| ASTRO-MB | 5.9226 | −0.0809 |
| AdaMuon | 6.0028 | −0.0007 |
| Muon | 6.0035 | 0.0000 |
| NorMuon | 6.0155 | +0.0121 |

This is a pointwise comparison, not a tuned global ranking. Its value is that every method sees the same configuration and the result includes two strong contemporary Muon variants.

**Figure: optimizer journey** — generated by `scripts/figures/fig_optimizer_journey.py`.

> **Observation 2.** At this shared 124M/900 setting, almost the entire separation from Muon is already present in ASTRO-MB; ASTRO-v2 adds only about 0.002 validation loss beyond it.

### 5.3 Three-config targeted mechanism experiment

The targeted run compares Muon, ASTRO-MB, ASTRO-v2, and ASTRO-v2 with $\gamma=0$ over three shared settings.

| variant | mean paired Δ vs Muon | settings won |
|---|---:|---:|
| **ASTRO-v2** | **−0.2208** | **3/3** |
| ASTRO-MB | −0.2137 | 3/3 |
| ASTRO-v2 ($\gamma=0$) | −0.2047 | 3/3 |

The largest separation occurs at the aggressive setting $lr\approx0.0505$, where Muon reaches 6.4527 while ASTRO-v2 remains at 5.9436. At the milder settings the gaps are much smaller.

That shape is important. A method that improves peak tuned loss and a method that prevents failure under aggressive hyperparameters are different contributions.

> **Observation 3.** Current evidence points more strongly toward **hyperparameter robustness** than toward the post-polar variance-power term as the main source of ASTRO-v2’s advantage.

### 5.4 Compute cost

Across the three targeted settings, Muon takes about 787 seconds per 900-step run and ASTRO-v2 about 838 seconds, or roughly **6.4% overhead**. ASTRO-MB is slower still in this measurement.

**Figure: compute efficiency** — generated by `scripts/figures/fig_efficiency_v2.py`.

> **Observation 4.** ASTRO-v2’s step-matched loss advantage is not free. Until time-to-target or time-matched loss is measured, the correct claim is “better loss at equal steps,” not “faster optimization.”

### 5.5 Horizon: intentionally incomplete

The current five-configuration horizon experiment gives:

| horizon | completed settings | mean paired Δ ASTRO-v2 − Muon | settings won |
|---|---:|---:|---:|
| 300 | 5/5 | −0.0500 | 2/5 |
| 600 | 5/5 | −0.0541 | 3/5 |
| 900 | **2/5** | −0.0118 so far | 1/2 so far |
| 2700 | **0/5** | pending | pending |

Two facts are already useful. First, one favorable configuration remains around −0.09 from 300 through 900 steps. Second, another configuration remains unfavorable, so the result is not uniformly positive across the search space.

The 900 mean is **not a result yet** because three settings are missing. The 2700 cell is not drawn as data at all.

**Figure: horizon** — generated by `scripts/figures/fig_horizon_v2.py`; open markers denote incomplete cells.

> **Observation 5.** The current horizon evidence is consistent with a persistent effect at some configurations, but it is insufficient to determine the 900- or 2700-step average trend.

---

## 6. Mechanism: fused QKV allocation

A fused attention projection stores Q, K, and V in one larger matrix. A spectral transform applied to the whole fused matrix is mathematically valid for that matrix, but it need not distribute update energy evenly among the three operators inside it.

Measured examples:

| checkpoint | fused Q/K/V share | split Q/K/V share | gradient V/Q |
|---|---|---|---:|
| GPT-2 | 0.244 / 0.226 / **0.530** | 0.318 / 0.347 / 0.336 | 2.65× |
| GPT-2 Medium | 0.245 / 0.266 / **0.489** | 0.308 / 0.369 / 0.323 | 3.27× |
| Pythia-410M | 0.218 / 0.248 / **0.534** | 0.356 / 0.300 / 0.344 | 3.28× |

Early GPT-2 layers can be much more extreme than the model average. However, random-initialization controls also show V-skew, so training cannot be claimed as the sole cause.

The defensible result is narrower and stronger:

> **Observation 6.** Fused spectral treatment can produce strongly imbalanced Q/K/V update allocation, while block-wise treatment restores substantially more balanced allocation. The imbalance is architecture- and stage-dependent rather than purely learned during training.

The split is not free: prior timing in this repository shows that split orthogonalization can cost materially more at larger widths. This motivates a scheduled split as a future efficiency experiment rather than assuming permanent splitting is always optimal.

---

## 7. The research journey

The development history is part of the evidence because several attractive explanations failed.

### 7.1 Shared learning rate was not a fair comparison

Earlier runs forced Muon and ASTRO to use the same learning rate even though their spectral and scalar update magnitudes were not calibrated identically. Opposite conclusions appeared under different fixed rates. This motivated optimizer-specific tuning and shared-configuration grids.

### 7.2 Scalar-path beta alignment mattered more than expected

ASTRO originally used $\beta_1=0.95$ for both matrix and scalar groups. Muon’s auxiliary Adam path uses $\beta_1=0.9$. Aligning that scalar-path behavior produced a much larger improvement than the more elaborate spectral changes in the current ablation.

### 7.3 Cautious weight decay was not a free safety improvement

Masking weight decay deletes decay on coordinates that fail the mask. It therefore changes effective regularization unless the surviving decay is rescaled. ASTRO-v2 disables the mechanism rather than treating it as a harmless default.

### 7.4 Post-polar variance adaptation is not yet the headline mechanism

Setting $\gamma=0$ in the targeted experiment loses only about 0.016 validation loss on average relative to full v2 while preserving a large advantage versus Muon. That is useful negative attribution evidence: the most mathematically distinctive component need not be the component carrying the result.

### 7.5 Why we keep the failures

A paper that shows only the final ASTRO-v2 row would make the development path look inevitable. It was not. The repository retains `artifacts/measured.json` as a historical ledger and uses the smaller `artifacts/paper_results.json` for the current paper-facing state.

---

## 8. Experiments required before the headline is frozen

### 8.1 Finish the 124M horizon

Complete the remaining 900-step configurations and all five 2700-step configurations. This decides whether the advantage is transient, persistent, or mainly a robust region around particular settings.

### 8.2 Independent-seed replication

Freeze a configuration **before** looking at new seeds and evaluate it on seeds 100–102 or more. In the current CLI, `--seeds 3` means seeds 100, 101, and 102; `--seeds 100` means one hundred seeds.

### 8.3 First scaling falsification at 355M

Run Muon and ASTRO-v2 at 355M/900 over shared settings. If the margin disappears, the paper must say the effect is small-model specific. If it survives, larger-GPU validation becomes justified.

### 8.4 Time-matched comparison

Record frozen validation-loss trajectories and compare both validation loss at equal wall-clock and time-to-target. The paper repository already contains `scripts/paper/run_trajectory.py` and `fig_training_trajectories.py` for this purpose.

---

## 9. Limitations

**The strongest ASTRO-v2 result is not yet an independent-seed result.** Three shared configurations reveal robustness but do not provide three stochastic replications.

**The scale is still small.** 124M is useful for falsification and mechanism work but is below the scales at which a new pretraining optimizer should ultimately be judged.

**The horizon is unfinished.** The current 900 cell contains only two of five settings and 2700 contains none in the new v2 study.

**ASTRO-v2 costs more per step.** The 6.4% overhead is moderate, not negligible.

**Some historical experiments used superseded protocols.** They remain in the repository as an audit trail but are not used as current headline evidence.

**The current mechanism attribution is incomplete.** ASTRO-MB’s near-tie with v2 narrows the hypothesis, but we still need to inspect and isolate every implementation difference before claiming that “beta alignment” alone is causal.

---

## 10. Conclusion

ASTRO-v2 is currently **promising, not proven**. At GPT-2 124M/900 it produces a large improvement over Muon across three shared configurations, including an especially large advantage in an aggressive-learning-rate regime. Yet the evidence also says what the result is *not*: the earlier ASTRO recipe loses at a tuned 300-step benchmark, the post-polar variance-power term does not explain most of v2’s gain, and the current step-matched result comes with roughly 6% additional runtime.

That combination makes the project scientifically more interesting than a simple leaderboard claim. The emerging hypothesis is that ASTRO-v2 may improve the **robustness of matrix-structured optimization to hyperparameter choice**, while a smaller subset of the recipe may carry most of the effect. The ongoing horizon, seed, and 355M experiments are designed to falsify that hypothesis quickly.

The methodological commitment is simpler: the final paper will show the journey, not just the winner. Every important negative result, protocol correction, and unfinished cell remains visible because optimizer research is most useful when the comparison can fail.

---

## References

1. Keller Jordan et al. **Muon: An optimizer for hidden layers in neural networks.** 2024.
2. Zichong Li et al. **NorMuon: Making Muon more efficient and scalable.** arXiv:2510.05491, 2025.
3. Chongjie Si, Debing Zhang, Wei Shen. **AdaMuon: Adaptive Muon Optimizer.** arXiv:2507.11005, 2025.
4. Kaiyue Wen, David Hall, Tengyu Ma, Percy Liang. **Fantastic Pretraining Optimizers and Where to Find Them.** arXiv:2509.02046, 2025.
5. Noah Amsel, David Persson, Christopher Musco, Robert Gower. **The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm.** arXiv:2505.16932, 2025.
6. Shuche Wang, Fengzhuo Zhang, Jiaxiang Li, Dirk Bergemann, Zhuoran Yang. **Why Muon Outperforms Adam: A Curvature Perspective.** arXiv:2606.04662, 2026.
7. Kaiyue Wen, Xingyu Dang, Kaifeng Lyu, Tengyu Ma, Percy Liang. **Fantastic Pretraining Optimizers and Where to Find Them II: Hyperball Optimization.** arXiv:2606.16899, 2026.
