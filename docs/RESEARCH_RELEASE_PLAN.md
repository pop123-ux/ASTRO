# ASTRO — 7-day research release plan

## Purpose

This plan turns the existing ASTRO work into a small, reproducible research release without requiring billion-parameter training.

The target is **not** to claim universal SOTA superiority. The target is to demonstrate a clear hypothesis, a controlled optimizer comparison, component attribution, and an end-to-end Transformer result that another researcher can reproduce on modest hardware.

> **Compute policy:** free Colab T4 is the primary experimental machine. Prefer more seeds and controlled ablations over larger models.

## What already exists

ASTRO already contains the optimizer, routing logic, spectral analysis, baseline implementations, ablation machinery, GPU probes, statistical reporting, and a paper source. `scripts/astro_lab.py` now provides a self-contained scaling/shared-configuration lab with Adam-family and Muon-family baselines.

The current 124M shared-configuration experiment is the immediate gate before any larger validation.

## Critical path

```text
shared configurations
        ↓
attribute the surviving ASTRO change(s)
        ↓
freeze the candidate recipe
        ↓
5-seed end-task validation
        ↓
modded-nanoGPT-derived 124M validation
        ↓
mechanistic figure(s)
        ↓
paper + reproducibility audit
        ↓
release
```

## Day 1 — finish attribution

**Goal:** determine whether the current ~0.083 nat ASTRO-v2 margin persists across the sampled configuration space.

Run the remaining shared configurations in `astro_lab.py` using the same pinned `scalar_lr_mult=0.4369` protocol.

Required optimizers:

- `muon`
- `normuon`
- `adamuon`
- `astro_v2`
- `astro_muon_betas`

Do not change the protocol after seeing results.

**Deliverable:** `astro_lab_report.md` containing the shared-configuration table.

**Decision:**

- If ASTRO wins consistently, continue to confirmation.
- If the effect is configuration-dependent, document that and make the paper about the boundary/attribution rather than claiming a universal win.

## Day 2 — lock the candidate and reproduce

**Goal:** freeze the recipe before end-task evaluation.

Select the smallest ASTRO recipe justified by Day 1. Do not add a new component merely to improve a headline number.

Run the key ASTRO vs Muon comparison on **5 seeds** at 124M/900 steps using the fixed, auditable configuration. Include NorMuon when the budget allows.

Record:

- validation loss per seed
- mean and standard deviation
- paired delta against Muon
- worst seed
- wall-clock time
- peak VRAM if available

**Deliverable:** a frozen `candidate` configuration and a seed-level JSON/Markdown table.

## Day 3 — component ablation

**Goal:** show which mechanism is responsible for the improvement.

Prioritize the existing ASTRO variants rather than inventing new ones:

```text
Muon
ASTRO-v2
ASTRO-v2 without the surviving beta change
ASTRO-v2 without the QKV split
ASTRO-v2 without post-normalization
ASTRO-v2 without cautious weight decay
```

One seed is acceptable for attribution; reserve the expensive multi-seed budget for the final comparison.

**Deliverable:** one ablation table and one plot showing contribution/removal effects.

## Day 4 — modded-nanoGPT-derived validation

**Goal:** validate the candidate on a standard 124M Transformer training setup derived from modded-nanoGPT.

The upstream optimization track fixes architecture, dataset, and batch size and seeks to minimize training steps to 3.28 validation loss. Its official experiments target multi-GPU A100/H100 systems; a T4 run should therefore be treated as a **research replication/validation**, not an official speed record. See `docs/MODDED_NANOGPT_VALIDATION.md`.

On the T4, keep the scientific comparison fair by holding the following constant across optimizers:

- model initialization
- tokenizer/data format
- sequence length
- batch/microbatch policy
- training steps
- LR schedule
- evaluation set
- seed

Only the optimizer recipe should change.

**Deliverable:** ASTRO vs Muon loss curves and final validation losses on the same 124M-derived task.

## Day 5 — mechanism and efficiency

**Goal:** explain the result, not just report it.

Use the existing measurement tools where they support the final story:

```text
scripts/measure_curvature.py
scripts/measure_drift.py
scripts/bench_gpu.py
scripts/colab_probe.py
```

Choose at most two mechanistic measurements that directly connect the hypothesis to the observed end-task effect.

Also report the cost of the candidate relative to Muon:

- seconds/step or seconds/run
- peak VRAM where measurable
- optimizer-state footprint if measurable

Do not turn efficiency into a separate optimization project. The objective is to establish whether the improvement is worth its compute overhead.

## Day 6 — paper integration

**Goal:** turn the experimental record into a coherent scientific argument.

Paper structure:

1. Problem and motivation
2. What is known about the relevant Muon/spectral behavior
3. Hypothesis
4. ASTRO update rule
5. Experimental protocol
6. Shared-configuration attribution
7. End-task results
8. Ablations
9. Mechanistic analysis
10. Limitations
11. Conclusion

The limitations must explicitly state the scale and compute boundary of the experiments.

Update the results from JSON artifacts rather than manually typing numbers into the paper.

**Deliverable:** paper source with every reported number traceable to an artifact.

## Day 7 — reproducibility and release freeze

**Goal:** make the project independently auditable.

Run:

```bash
pip install -e ".[dev]"
pytest
python scripts/make_results.py --inject
python scripts/build_paper.py
```

Then perform a clean-machine/clean-Colab replay of the smallest end-to-end experiment.

Freeze:

- optimizer code
- candidate hyperparameters
- benchmark protocol
- result JSONs
- figures
- paper PDF
- README claims

Tag the release only after the README, paper, and artifacts agree.

## Release acceptance criteria

The release is ready when all of the following are true:

- [ ] ASTRO's candidate recipe is frozen.
- [ ] Muon, NorMuon, and AdaMuon are represented where relevant.
- [ ] At least one multi-seed head-to-head exists at 124M.
- [ ] The strongest claimed effect survives shared-configuration checking.
- [ ] The main ablation identifies which component(s) matter.
- [ ] At least one modded-nanoGPT-derived 124M experiment is complete.
- [ ] Every headline result has a stored raw artifact.
- [ ] Negative/unmeasured components remain explicitly labelled.
- [ ] The paper states the compute/scale limitations.
- [ ] The repository passes the test suite.

## What not to do during this week

Do not add 1B+ models, a new dataset, a new optimizer family, a new architectural modification, or a large hyperparameter search unless an existing result makes it scientifically necessary.

Do not optimize for a leaderboard position. Optimize for an **auditable research claim**.

## Final research question

The release should answer one sentence clearly:

> **Does the mechanism identified by ASTRO produce a reproducible improvement over strong spectral/Adam-family baselines in controlled Transformer optimization, and can the improvement be attributed to a specific change rather than to tuning or protocol asymmetry?**

A negative answer is still a valid result if the decomposition is rigorous and reproducible.
