# ORBIT paper artifact pipeline

This document starts after the final `ablation_ext` shard and merge complete.

The paper pipeline is intentionally strict. It does **not** use the legacy
`merged/summary.json` as the source of truth. Instead it validates the exact final JSONLs,
freezes a provenance manifest, recomputes the paper-facing paired effects, then generates
figures and LaTeX tables in the evidence order used by the manuscript.

## 1. Finish and verify the final expanded ablation

```bash
%cd /content/orbit_project
%env ORBIT_WORK=/content/drive/MyDrive/orbit_paper

!python scripts/orbit_postscale_merge.py \
  --work-dir "$ORBIT_WORK" \
  --phase ablation_ext

!ls -lh "$ORBIT_WORK/merged/ablation_ext"*
```

Required:
- `merged/ablation_ext.jsonl`
- `merged/ablation_ext_summary.json`
- `merged/ablation_ext_analysis.json`

## 2. Freeze the evidence

Update the code on the Colab that will generate paper artifacts:

```bash
%cd /content/orbit_project
!git fetch origin orbit-paper-campaign
!git reset --hard origin/orbit-paper-campaign
!python -m pip install -q -e . "matplotlib>=3.8"

!python scripts/orbit_paper_artifacts.py \
  --work-dir "$ORBIT_WORK"
```

A successful strict freeze writes:

```text
$ORBIT_WORK/paper_artifacts/
  manifest.json
  paper_results.json
```

The validator checks:
- exact task counts and expected seeds;
- unique task IDs;
- frozen core digest `de8b994a7342...`;
- successful finite losses/runtimes;
- one software environment per phase;
- identical matched Muon/ORBIT configuration;
- expanded-ablation configuration consistency.

If any required phase is missing, paper generation stops rather than silently using stale
results.

## 3. Generate publication figures

```bash
!python scripts/orbit_plot.py \
  --work-dir "$ORBIT_WORK"
```

The main figures are:

```text
paper_artifacts/figures/
  matched_effect.pdf
  matched_seed_pairs.pdf
  xconfig_interaction.pdf
  ablation_effects.pdf
  transfer_horizon.pdf
  transfer_scale.pdf
  broad_confirmation.pdf
  matched_training_curves.pdf
  matched_efficiency.pdf
  metric_diagnostics.pdf
```

The primary plots use paired effects rather than zero-based loss bars. The n=2 transfer
figures expose individual seeds and intentionally avoid a high-powered-significance visual
claim.

## 4. Generate paper tables, claim ledger, and manuscript scaffold

```bash
!python scripts/orbit_build_paper.py \
  --work-dir "$ORBIT_WORK" \
  --no-compile
```

This produces:
- `$ORBIT_WORK/paper_artifacts/claim_ledger.md`;
- generated LaTeX macros/tables under `docs/orbit/paper/generated/`;
- synchronized PDF figures under `docs/orbit/paper/figures/`.

The claim ledger explicitly records supported claims and prohibited overclaims.

## 5. Compile the manuscript

If TeX Live / latexmk is installed:

```bash
!python scripts/orbit_build_paper.py \
  --work-dir "$ORBIT_WORK"
```

The resulting manuscript is:

```text
docs/orbit/paper/main.pdf
```

## Evidence order used in the paper

1. matched-hyperparameter confirmation;
2. optimizer/configuration cross-over;
3. expanded mechanism ablation;
4. long-horizon transfer;
5. 355M transfer;
6. broad independently tuned comparison as secondary context.

This ordering prevents the independently tuned headline gap from being confused with the
mechanism-isolated effect.
