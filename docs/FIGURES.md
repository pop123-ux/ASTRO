# ASTRO paper figure system

The paper treats every figure as a reproducible result. The visual source of
record is **Matplotlib + committed JSON**. Weights & Biases is optional and is
used as an experiment browser, not as a dependency of the manuscript.

```bash
pip install -e '.[paper]'
python scripts/figures/make_all.py
```

Every successful plot writes both:

- `artifacts/figures/<name>.pdf` — vector source used by LaTeX;
- `artifacts/figures/<name>.png` — convenient preview for GitHub;
- `artifacts/figures/<name>.json` — the exact values rendered.

This prevents a common paper failure mode: a plot is manually edited after the
numbers change and silently stops matching the experiment.

---

## 1. Figure hierarchy

The current paper has a **primary empirical spine** and a **supporting mechanism
suite**. Historical plots are retained because the research journey matters, but
they are explicitly labeled as legacy rather than allowed to compete with the
current ASTRO-v2 evidence.

### Primary ASTRO-v2 figures

| figure | purpose | current input |
|---|---|---|
| `fig_optimizer_journey` | documents the progression Muon → NorMuon/AdaMuon → ASTRO-MB → ASTRO-v2, then shows the three-config mechanism comparison | `artifacts/paper_results.json` |
| `fig_training_trajectories` | validation loss through training, shown against both steps and wall-clock | `artifacts/trajectories.json` |
| `fig_horizon_v2` | asks whether the paired ASTRO-v2 margin survives 300 → 600 → 900 → 2700 steps; incomplete cells are visibly incomplete | `artifacts/paper_results.json` |
| `fig_efficiency_v2` | makes the compute cost visible beside the loss improvement | `artifacts/paper_results.json` |

### Supporting figures

| figure | purpose |
|---|---|
| `fig1_leverage` | Q/K/V allocation and the leverage interpretation of the polar update |
| `fig2_quintic` | numerical behavior of Muon's repeated quintic versus a converging schedule |
| `fig3_curvature` | optional curvature decomposition at matched validation loss |
| `fig5_inversion` | documents the cautious-mask sign inversion across scale |
| `fig7_drift` | optional update-norm drift measured inside real training |

`fig4_results` and the original `fig6_horizon` are retained as **legacy research
history**. They describe earlier protocols and must not be used as the headline
ASTRO-v2 result.

---

## 2. The optimizer journey is part of the paper

We do not want a final paper that jumps straight from Muon to the best ASTRO-v2
number. The path is scientifically useful:

1. **Muon** — spectral reference;
2. **NorMuon** — row-wise post-orthogonalization adaptation;
3. **AdaMuon** — element-wise post-orthogonalization adaptation;
4. **earlier ASTRO recipes** — useful negative evidence at 300 steps;
5. **ASTRO-MB** — removes the scalar-path beta asymmetry;
6. **ASTRO-v2** — the current recipe;
7. **ASTRO-v2 γ=0** — asks whether the post-polar variance term is actually
   responsible for the improvement.

The journey figure must always distinguish **pointwise shared-configuration
comparisons** from **independent-seed replications**. A hyperparameter
configuration is not a seed and must never be used to imply a significance test.

---

## 3. Validation-loss trajectories

Final validation loss is not enough for an optimizer paper. We also want to see
*how* the ordering develops.

Run the dedicated trajectory instrument only after configurations are frozen:

```bash
python scripts/paper/run_trajectory.py \
  --size 124M \
  --steps 900 \
  --seed 100 \
  --eval-every 75 \
  --optimizers muon normuon adamuon astro_muon_betas astro_v2 \
  --config lr=0.010205978810672643 \
           weight_decay=0.001059717889298923 \
           scalar_lr_mult=0.4369 \
  --cache /content/drive/MyDrive/astro/paper_trajectory_cache.pt \
  --out /content/drive/MyDrive/astro/trajectories.json
```

Copy the resulting JSON to `artifacts/trajectories.json` and run:

```bash
python scripts/figures/make_all.py --only training_trajectories
```

The trajectory runner periodically evaluates the same pinned validation slice.
Probe time is subtracted from its training clock so the wall-clock panel is not
artificially penalized by the act of measuring it. These trajectory runs are for
visualizing learning dynamics; the headline runtime comparison still comes from
the uninstrumented benchmark.

For three independent evaluation seeds use the normal benchmark semantics:
`--seeds 3` means seeds **100, 101, and 102**. `--seeds 100` means one hundred
seeds and should not be used when the intent is literal seed 100.

---

## 4. W&B is optional

The committed JSON is the evidence. W&B is useful for interaction and run
inspection:

```bash
pip install wandb
wandb login
python scripts/paper/wandb_sync.py \
  --project astro-paper \
  --input artifacts/trajectories.json
```

A reviewer must never need W&B access to reproduce a paper figure. If a W&B
chart is useful, export its underlying history to JSON and render the final paper
version with the Matplotlib scripts in this repository.

---

## 5. Visual language

The reference aesthetic is the compact two-column style used in contemporary
optimizer papers such as *Why Muon Outperforms Adam: A Curvature Perspective*:
small multiples, restrained typography, direct quantitative annotations, and
plots that make the comparison readable before the caption is read.

The rules are encoded in `scripts/figures/style.py`:

- serif type throughout;
- vector PDF output for LaTeX;
- one optimizer = one colour everywhere;
- ASTRO-v2 is the strongest accent, not every ASTRO ablation;
- no top/right plot spines;
- light solid grids, not dark dashboard grids;
- direct values when they help interpretation;
- legends only when direct labels would collide;
- uncertainty is drawn from the uncertainty we actually have;
- missing measurements are marked as pending rather than interpolated.

The paper itself uses a two-column LaTeX template. Full-width multi-panel figures
use `figure*`; compact one-panel figures use `figure`. Tables use `booktabs` and
avoid vertical rules.

---

## 6. What a figure is allowed to claim

**Shared configurations.** Show pointwise robustness and sensitivity. Do not call
them independent replications.

**Independent seeds.** Support statements about stochastic reproducibility. With
three seeds, the minimum two-sided sign-test p-value is 0.25, so the paper should
show all seed points rather than manufacture precision from an error bar.

**Wall-clock.** ASTRO-v2 currently costs more per step than Muon in the targeted
124M/900 run. A step-matched loss win therefore does not automatically imply a
compute win. `fig_efficiency_v2` exists so this caveat cannot disappear from the
paper layout.

**Incomplete horizon cells.** Open markers and explicit `pending` annotations.
No line should visually imply a completed 2700-step result before it exists.

---

## 7. Recommended final paper order

1. **Figure 1 — optimizer journey / headline empirical result**
2. **Figure 2 — training trajectories, step and wall-clock**
3. **Figure 3 — horizon and scale**
4. **Figure 4 — QKV allocation mechanism**
5. **Figure 5 — ablation / ASTRO-MB / γ=0 attribution**
6. **Figure 6 — compute efficiency or time-to-target**
7. optional appendix figures: quintic, curvature, drift, withdrawn protocols

This ordering makes the paper read as a research argument rather than a gallery
of plots: **what happens → does it persist → why might it happen → what does it
cost → where does it fail?**
