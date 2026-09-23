# ORBIT post-scale validation: 3-Colab runbook

This is the canonical continuation after the legacy `scale` phase. It addresses the four remaining paper-grade questions without invalidating the existing 84 runs.

The existing core campaign remains frozen. New runs record both the unchanged core digest and a separate post-scale orchestration digest.

## 0. Update all three Colabs

On every instance:

```bash
%cd /content/orbit_project
!git fetch origin orbit-paper-campaign
!git reset --hard origin/orbit-paper-campaign
!python -m pip install -q -e . "transformers>=4.56,<5" "datasets>=3,<5" "matplotlib>=3.8" "pytest>=8"
%env ORBIT_WORK=/content/drive/MyDrive/orbit_paper
```

Verify the shared Drive and frozen core digest:

```bash
!cat "$ORBIT_WORK/__shared_probe.txt"
!python - <<'PY'
import sys
sys.path.insert(0, 'scripts')
import orbit_campaign
print('core digest:', orbit_campaign.code_digest()[:12])
PY
```

The core digest should remain `de8b994a7342` for the already-produced campaign implementation.

Run the tests once on one instance:

```bash
!python -m pytest tests/test_orbit.py tests/test_orbit_postscale.py
```

## How to use 4 logical shards on 3 Colabs

For the larger phases, start shard IDs `0`, `1`, and `2` on the three instances. When any instance finishes, run shard `3` on that freed instance. A shard can run on any physical Colab; only the shard ID matters. Never run the same phase/shard ID simultaneously on two instances.

All new outputs are written to:

```text
$ORBIT_WORK/shards/shard-N/<phase>.jsonl
```

Rerunning the same command is safe: completed task IDs for the exact core + orchestration digest and environment are skipped.

## Phase F — 2x2 cross-configuration isolation

Purpose: separate optimizer mechanism from hyperparameter recipe.

This reruns the four cells on the same five held-out seeds:

```text
Muon  @ Muon config
Muon  @ ORBIT config
ORBIT @ Muon config
ORBIT @ ORBIT config
```

Launch shards `0`, `1`, `2`; then `3` on the first free instance:

```bash
!python scripts/orbit_postscale.py --phase xconfig --work-dir "$ORBIT_WORK" --shard-id 0 --num-shards 4 --max-minutes 330
```

Change only `--shard-id` to `1`, `2`, and later `3`.

Merge:

```bash
!python scripts/orbit_postscale_merge.py --work-dir "$ORBIT_WORK" --phase xconfig
!cat "$ORBIT_WORK/merged/xconfig_summary.json"
!cat "$ORBIT_WORK/merged/xconfig_analysis.json"
```

The analysis reports the mechanism effect under each frozen configuration, the recipe effect for each optimizer, and the optimizer-by-config interaction.

## Phase G — matched shared-grid tuning at the actual 900-step target

Purpose: remove random-search luck. Muon and ORBIT receive the exact same ten LR / scalar-LR / weight-decay candidate triples.

```bash
!python scripts/orbit_postscale.py --phase matched_tune --work-dir "$ORBIT_WORK" --shard-id 0 --num-shards 4 --max-minutes 330
```

Run shard IDs `0`, `1`, `2`, then `3`.

Merge and freeze the winners before any held-out evaluation:

```bash
!python scripts/orbit_postscale_merge.py --work-dir "$ORBIT_WORK" --phase matched_tune
!cat "$ORBIT_WORK/merged/matched_best_configs.json"
```

Do not edit `matched_best_configs.json` after seeing later results.

## Phase H — matched-grid held-out confirmation, 10 seeds

Purpose: test the two configs selected from the identical candidate set on ten new seeds `500..509`.

```bash
!python scripts/orbit_postscale.py --phase matched_confirm --work-dir "$ORBIT_WORK" --shard-id 0 --num-shards 4 --max-minutes 330
```

Run shard IDs `0`, `1`, `2`, then `3`.

Merge:

```bash
!python scripts/orbit_postscale_merge.py --work-dir "$ORBIT_WORK" --phase matched_confirm
!cat "$ORBIT_WORK/merged/matched_confirm_summary.json"
!cat "$ORBIT_WORK/merged/matched_confirm_analysis.json"
```

The paired analysis reports ORBIT-minus-Muon per seed, win counts, and a paired 95% t interval.

## Phase I — expanded mechanism ablation, 10 seeds

Purpose: determine which parts of ORBIT survive a larger seed set under the matched-tuned ORBIT recipe.

Variants:

```text
orbit
orbit_norope
orbit_diag
orbit_identity
```

Seeds are `600..609`.

```bash
!python scripts/orbit_postscale.py --phase ablation_ext --work-dir "$ORBIT_WORK" --shard-id 0 --num-shards 4 --max-minutes 330
```

Run shard IDs `0`, `1`, `2`, then `3`.

Merge:

```bash
!python scripts/orbit_postscale_merge.py --work-dir "$ORBIT_WORK" --phase ablation_ext
!cat "$ORBIT_WORK/merged/ablation_ext_summary.json"
!cat "$ORBIT_WORK/merged/ablation_ext_analysis.json"
```

## Phase J — add ASTRO-v2 to the 2700-step horizon cell

The Muon/NorMuon/ORBIT horizon runs already exist and remain valid. This phase runs only ASTRO-v2 on the same seeds `400,401` and then produces a combined four-optimizer table.

Only shards `0` and `1` own tasks here:

```bash
!python scripts/orbit_postscale.py --phase astro_horizon --work-dir "$ORBIT_WORK" --shard-id 0 --num-shards 4 --max-minutes 330
!python scripts/orbit_postscale.py --phase astro_horizon --work-dir "$ORBIT_WORK" --shard-id 1 --num-shards 4 --max-minutes 330
```

Merge:

```bash
!python scripts/orbit_postscale_merge.py --work-dir "$ORBIT_WORK" --phase astro_horizon
!cat "$ORBIT_WORK/merged/horizon_with_astro_summary.json"
!cat "$ORBIT_WORK/merged/horizon_with_astro_analysis.json"
```

## Phase K — add ASTRO-v2 to the 355M scale cell

Again, only the missing ASTRO-v2 runs are executed, using the same scale seeds `300,301`.

```bash
!python scripts/orbit_postscale.py --phase astro_scale --work-dir "$ORBIT_WORK" --shard-id 0 --num-shards 4 --max-minutes 330
!python scripts/orbit_postscale.py --phase astro_scale --work-dir "$ORBIT_WORK" --shard-id 1 --num-shards 4 --max-minutes 330
```

Merge:

```bash
!python scripts/orbit_postscale_merge.py --work-dir "$ORBIT_WORK" --phase astro_scale
!cat "$ORBIT_WORK/merged/scale_with_astro_summary.json"
!cat "$ORBIT_WORK/merged/scale_with_astro_analysis.json"
```

## Final post-scale integrity merge

After all six phases are complete:

```bash
!python scripts/orbit_postscale_merge.py --work-dir "$ORBIT_WORK" --phase all
```

At that point the empirical core needed to begin the manuscript is complete. The next work is analysis/figure generation and paper writing, not another optimizer search.
