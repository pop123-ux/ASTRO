# ORBIT: exact four-Colab runbook

This is the canonical launch sequence for `orbit-paper-campaign`. Do not edit the optimizer
mid-phase. If the branch changes after a phase starts, finish or discard that phase; the
merger rejects mixed code digests.

## 0. Start four GPU Colabs

Use the same runtime type on all four instances. T4 is the baseline target. In **each**
notebook run:

```bash
!nvidia-smi
!git clone -b orbit-paper-campaign https://github.com/pop123-ux/ASTRO.git
%cd ASTRO
!python -m pip install -q -e . "transformers>=4.56,<5" "datasets>=3,<5" "matplotlib>=3.8" "pytest>=8"
```

Then mount the same Google Drive in each notebook:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Use one shared work directory:

```bash
%env ORBIT_WORK=/content/drive/MyDrive/orbit_paper
```

Before spending GPU time, run on **every** instance:

```bash
!python -m pytest tests/test_orbit.py
!python scripts/orbit_novelty_gate.py --strict
```

All four should print the same package versions when the campaign starts. If merging later
reports an environment mismatch, do not override it; recreate the mismatching runtime.

## 1. Build the pinned corpus once

Run this only on Colab 0 and wait for it to finish before starting the other workers:

```bash
!python scripts/orbit_campaign.py \
  --work-dir "$ORBIT_WORK" \
  --prepare-data
```

It reuses the inherited paper campaign's pinned FineWeb-Edu v1.0.0 shared token cache.

## 2. Phase A — discovery/tuning, four workers

Run exactly one command per instance. Re-running the same command is safe: completed task IDs
are skipped.

**Colab 0**
```bash
!python scripts/orbit_campaign.py --phase discovery --work-dir "$ORBIT_WORK" --shard-id 0 --num-shards 4 --trials 5 --max-minutes 330
```

**Colab 1**
```bash
!python scripts/orbit_campaign.py --phase discovery --work-dir "$ORBIT_WORK" --shard-id 1 --num-shards 4 --trials 5 --max-minutes 330
```

**Colab 2**
```bash
!python scripts/orbit_campaign.py --phase discovery --work-dir "$ORBIT_WORK" --shard-id 2 --num-shards 4 --trials 5 --max-minutes 330
```

**Colab 3**
```bash
!python scripts/orbit_campaign.py --phase discovery --work-dir "$ORBIT_WORK" --shard-id 3 --num-shards 4 --trials 5 --max-minutes 330
```

Default discovery optimizers are:

```text
AdamW
Muon
NorMuon
faithful AdaMuon reference
ASTRO-v2
ORBIT
```

Each gets five deterministic tuning trials. ORBIT uses Muon's search space; it does not receive
extra tuning dimensions for its new mechanism.

After all four workers finish, run on one instance:

```bash
!python scripts/orbit_merge.py --work-dir "$ORBIT_WORK" --phase discovery
!cat "$ORBIT_WORK/merged/best_configs.json"
```

This freezes one configuration per optimizer. **Do not manually pick a different configuration
after seeing held-out results.**

## 3. Phase B — five held-out seeds at 124M / 900 steps

Launch the same four-way split:

**Colab 0**
```bash
!python scripts/orbit_campaign.py --phase confirm --work-dir "$ORBIT_WORK" --shard-id 0 --num-shards 4 --max-minutes 330
```

**Colab 1**
```bash
!python scripts/orbit_campaign.py --phase confirm --work-dir "$ORBIT_WORK" --shard-id 1 --num-shards 4 --max-minutes 330
```

**Colab 2**
```bash
!python scripts/orbit_campaign.py --phase confirm --work-dir "$ORBIT_WORK" --shard-id 2 --num-shards 4 --max-minutes 330
```

**Colab 3**
```bash
!python scripts/orbit_campaign.py --phase confirm --work-dir "$ORBIT_WORK" --shard-id 3 --num-shards 4 --max-minutes 330
```

Then:

```bash
!python scripts/orbit_merge.py --work-dir "$ORBIT_WORK" --phase confirm
!python scripts/orbit_plot.py --work-dir "$ORBIT_WORK"
```

**Decision gate:** do not move to expensive phases merely because one ORBIT seed wins. Check the
mean, seed spread, wall-clock overhead, and whether the effect is larger than the held-out noise.

## 4. Phase C — mechanism ablations

This tests whether the actual RoPE-functional mechanism matters.

Run on all four workers with shard IDs 0, 1, 2, 3 respectively:

```bash
!python scripts/orbit_campaign.py --phase ablation --work-dir "$ORBIT_WORK" --shard-id 0 --num-shards 4 --max-minutes 330
```

Change only `--shard-id` on the other three Colabs.

The phase compares:

```text
orbit          full RoPE-aware 2x2 functional metric
orbit_norope   Q/K covariance adaptation without relative-position rotations
orbit_diag     per-frequency metric without off-diagonal phase coupling
orbit_identity same routing/code path with functional preconditioning disabled
```

Merge:

```bash
!python scripts/orbit_merge.py --work-dir "$ORBIT_WORK" --phase ablation
!python scripts/orbit_plot.py --work-dir "$ORBIT_WORK"
```

If `orbit_identity`, `orbit_norope`, or `orbit_diag` matches ORBIT within noise, weaken or reject
the corresponding mechanism claim before running scale experiments.

## 5. Phase D — long horizon, 124M / 2700 steps

Only run if confirmation + ablations justify it.

```bash
!python scripts/orbit_campaign.py --phase horizon --work-dir "$ORBIT_WORK" --shard-id 0 --num-shards 4 --max-minutes 330
```

Again use shard IDs 0–3 across the four instances, then:

```bash
!python scripts/orbit_merge.py --work-dir "$ORBIT_WORK" --phase horizon
```

The long-horizon set is intentionally smaller: Muon, NorMuon and ORBIT, two frozen-config seeds.

## 6. Phase E — 355M transfer

Only run after the long-horizon gate:

```bash
!python scripts/orbit_campaign.py --phase scale --work-dir "$ORBIT_WORK" --shard-id 0 --num-shards 4 --max-minutes 330
```

Use shard IDs 0–3 across the workers. The 355M model enables gradient checkpointing.

Merge and regenerate all figures:

```bash
!python scripts/orbit_merge.py --work-dir "$ORBIT_WORK" --phase scale
!python scripts/orbit_merge.py --work-dir "$ORBIT_WORK" --phase all
!python scripts/orbit_plot.py --work-dir "$ORBIT_WORK"
```

## 7. Generate the LaTeX manuscript

From any machine with the repository checked out:

```bash
python scripts/orbit_build_paper.py --work-dir "$ORBIT_WORK" --no-compile
```

This writes `docs/orbit/paper/generated_results.tex` and copies generated PDF figures into the
paper tree. On a machine with TeX Live / `latexmk` installed:

```bash
python scripts/orbit_build_paper.py --work-dir "$ORBIT_WORK"
```

The final PDF will be `docs/orbit/paper/main.pdf`.

## Useful commands

Preview exactly what one worker owns without running it:

```bash
python scripts/orbit_campaign.py --phase discovery --work-dir "$ORBIT_WORK" --shard-id 2 --num-shards 4 --list-tasks
```

One-task GPU sanity run before the full discovery campaign:

```bash
python scripts/orbit_campaign.py --phase discovery --work-dir "$ORBIT_WORK/sanity" --shard-id 0 --num-shards 1 --trials 1 --max-tasks 1 --steps-override 20
```

CPU-only synthetic plumbing test:

```bash
python scripts/orbit_campaign.py --phase discovery --work-dir /tmp/orbit-smoke --num-shards 1 --shard-id 0 --trials 1 --max-tasks 1 --steps-override 2 --seq 32 --synthetic --device cpu
```

## Non-negotiable provenance rules

- Never merge results generated with different `code_digest` values.
- Never edit `best_configs.json` after confirmation begins.
- Never reuse a result directory after changing the optimizer equations.
- Keep failed/negative runs; do not delete them to improve the story.
- Run the exact-equation prior-art search in `PRIOR_ART_GATE.md` again immediately before any
  public novelty claim.
