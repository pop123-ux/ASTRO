# ASTRO paper figure system

The paper treats every figure as a reproducible result. The visual source of record is **Matplotlib + committed JSON**. Weights & Biases is optional and is used as an experiment browser, not as a dependency of the manuscript.

```bash
pip install -e '.[paper]'
python scripts/figures/make_all.py
```

Every successful plot writes:

- `artifacts/figures/<name>.pdf` — vector source used by LaTeX;
- `artifacts/figures/<name>.png` — GitHub/quick-look preview;
- `artifacts/figures/<name>.json` — exact values rendered.

This prevents a common paper failure mode: a plot is manually edited after the numbers change and silently stops matching the experiment.

---

## 1. Primary paper figures

| figure | purpose | current status |
|---|---|---|
| `fig_optimizer_journey` | headline progression from Muon/NorMuon/AdaMuon to ASTRO-MB and ASTRO-v2, plus the targeted mechanism comparison | measured |
| `fig_paired_900` | all five 124M/900 shared settings shown as paired dumbbells instead of hiding the search behind one winner | measured |
| `fig_horizon_v2` | 300 → 600 → 900 → 2700 horizon; small points are configurations, diamonds are means | measured through 900; 2700 pending |
| `fig_efficiency_v2` | validation loss against measured runtime for the targeted mechanism study | measured |
| `fig_seed_replication` | independent-seed replication; automatically changes from pending-aware view to paired connectors once ASTRO-v2 seeds exist | Muon 100–106 measured; ASTRO-v2 pending |
| `fig_scale_status` | scale-study evidence without extrapolation; open markers are experiment slots | 124M measured; clean 355M + 774M pending |
| `fig_training_trajectories` | validation loss against both training step and wall-clock | waiting for frozen trajectory run |

Supporting mechanistic and historical figures remain available (`fig_leverage`, `fig_quintic`, curvature, drift, inversion, and legacy result plots), but they do not replace the current ASTRO-v2 empirical spine.

---

## 2. Aesthetic target

The visual target is the compact technical-report style used by strong optimizer papers: **small number of high-information figures, restrained typography, obvious comparison direction, and no dashboard decoration**.

The current LaTeX manuscript mirrors that style on page one:

1. strong horizontal rule;
2. centered paper title and technical-report label;
3. compact full-width abstract;
4. a full-width headline empirical figure immediately below the abstract;
5. two-column body afterward.

The house style in `scripts/figures/style.py` enforces:

- serif typography;
- vector PDF output;
- one optimizer = one colour throughout the paper;
- ASTRO-v2 as the strongest accent;
- no top/right plot spines;
- light neutral grids;
- direct numerical labels only when they help interpretation;
- legends only when direct labels would collide;
- uncertainty based only on measurements that exist;
- open neutral markers for pending experiments rather than interpolation.

The palette is deliberately print-safe and restrained. Muon is orange, ASTRO-v2 is deep purple, ASTRO-MB and gamma ablations use progressively lighter related accents, and pending evidence is neutral grey.

---

## 3. The optimizer journey is part of the result

The paper should not jump straight from Muon to one best ASTRO number. The development path is scientifically useful:

1. **Muon** — spectral reference;
2. **NorMuon** — row-wise post-orthogonalization adaptation;
3. **AdaMuon** — finer post-orthogonalization adaptation;
4. **earlier ASTRO recipes** — negative evidence at 300 steps;
5. **ASTRO-MB** — beta-aligned recipe;
6. **ASTRO-v2** — current recipe;
7. **ASTRO-v2 gamma=0** — tests whether post-polar variance adaptation carries the result.

`fig_optimizer_journey` is therefore the first-page headline figure, not an appendix artifact.

---

## 4. Why `fig_paired_900` matters

The completed 124M/900 grid has five shared configurations. Showing only the best Muon and best ASTRO-v2 values would hide the shape of the result because the minima occur at different settings.

The dumbbell plot shows all five pairings directly:

```text
config 1   Δ ≈ -0.091
config 2   Δ ≈ +0.067
config 3   Δ ≈ -0.108
config 4   Δ ≈ +0.006
config 5   Δ ≈ -0.283
mean       Δ ≈ -0.082
```

This makes the emerging robustness hypothesis visible: ASTRO-v2 does not win everywhere, but several settings improve substantially and the mean paired margin is negative.

---

## 5. Horizon, seeds, and scale must show missing evidence honestly

### Horizon

300, 600, and 900 are complete. 2700 is not. `fig_horizon_v2` therefore renders an **open grey diamond** at 2700 and does not connect the measured trend into that slot.

### Independent seeds

Seven Muon seeds (100–106) are measured. ASTRO-v2 is pending. `fig_seed_replication` currently renders the Muon distribution and an open pending ASTRO-v2 marker. Once seven ASTRO-v2 losses are added to `paper_results.json`, the same script automatically draws matched seed connectors.

### Scale

A partial 355M run used a different `astro_lab.py` fingerprint from the 124M evidence. It is excluded from the paper-facing scale result. `fig_scale_status` renders 355M as a clean-rerun slot rather than turning mismatched provenance into a datapoint.

---

## 6. Validation-loss trajectories

Final validation loss is not enough for an optimizer paper. We also want to see *how* the ordering develops.

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

The trajectory runner evaluates the same pinned validation slice. Probe time is removed from the reported training clock so the wall-clock panel is not penalized by the act of measuring it.

---

## 7. W&B is optional

The committed JSON is the evidence. W&B is useful for interactive run inspection:

```bash
pip install wandb
wandb login
python scripts/paper/wandb_sync.py \
  --project astro-paper \
  --input artifacts/trajectories.json
```

A reviewer must never need W&B access to reproduce a paper figure. If a W&B chart is useful during exploration, export the underlying history and render the final paper version with Matplotlib.

---

## 8. What a figure is allowed to claim

**Shared configurations.** They show sensitivity and robustness. They are not independent replications.

**Independent seeds.** They support stochastic reproducibility claims. The paper should show every seed, not only a mean/error bar.

**Wall-clock.** ASTRO-v2 currently costs more per step than Muon in the targeted 124M/900 run. A step-matched loss win does not automatically imply a compute win.

**Incomplete horizon/scale cells.** Open markers and explicit pending annotations. No line should visually imply a result that does not exist.

**Mismatched provenance.** Keep it in the research log; do not place it in the same scientific curve.

---

## 9. Recommended final paper order

1. **Figure 1 — optimizer journey / headline empirical result**
2. **Figure 2 — five paired 900-step shared configurations**
3. **Figure 3 — horizon through 2700 once complete**
4. **Figure 4 — independent-seed replication**
5. **Figure 5 — scale**
6. **Figure 6 — loss-vs-step and loss-vs-wall-clock trajectories**
7. **Figure 7 — mechanism/QKV evidence**
8. appendix: quintic, curvature, drift, failed hypotheses, and superseded protocols

The paper should read as a research argument rather than a gallery of plots:

**what happens → does it persist → does it replicate → does it scale → what does it cost → why might it happen → where does it fail?**
