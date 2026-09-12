# Paper workflow

This repository treats the manuscript as a generated research artifact: results,
figures, tables, and prose must all trace back to versioned experiment outputs.
The paper is a **living draft** until the horizon, replication, scale, and
compute-matched checks are frozen.

## Repository contract

```text
artifacts/
  measured.json             historical experiment ledger; never erase
  paper_results.json         current paper-facing snapshot
  trajectories.json         optional frozen validation trajectories
  figures/                  generated PNG/PDF + figure data JSON

docs/
  FIGURES.md                 visual conventions and figure inventory
  PAPER_WORKFLOW.md          this file
  paper/
    paper.md                 readable manuscript
    paper.tex                submission source
    paper.pdf                compiled artifact when available
scripts/
  astro_lab.py               benchmark / ablation harness
  figures/                   deterministic Matplotlib renderers
  paper/
    run_trajectory.py        validation-loss trajectory instrument
    wandb_sync.py            optional W&B upload
    build.py                 render figures + compile LaTeX
```

`artifacts/measured.json` is intentionally historical. It contains experiments
that were later superseded and is part of the audit trail. The paper should use
`artifacts/paper_results.json`, which contains only the evidence currently in
scope for the manuscript.

## Current paper state

As of 2026-09-13 the strongest current ASTRO-v2 signal is at GPT-2 124M / 900
steps under shared configurations. In the targeted mechanism experiment,
ASTRO-v2 is below Muon on all three tested settings with a mean paired delta of
about -0.221 validation loss. ASTRO-MB is nearly identical and γ=0 retains most
of the effect, so the paper must not attribute the improvement mainly to the
post-polar variance-power mechanism.

The evidence is still **directional**, not final:

- the three targeted settings are configurations, not independent seeds;
- a tuned 300-step benchmark of the earlier/plain ASTRO recipe is slightly worse
  than Muon;
- the 900/2700 horizon study is incomplete;
- the 355M scale check is running;
- ASTRO-v2 costs roughly 6% more wall-clock than Muon in the targeted 900-step
  run, so a time-matched comparison is required before an efficiency claim.

The abstract and conclusion should remain conservative until those checks land.

## Experiment → paper pipeline

### 1. Run experiments in persistent work directories

Every Colab run should point to Drive:

```bash
--work-dir /content/drive/MyDrive/astro/<experiment-name>
```

Never use `/content` for a multi-hour result. Preserve the printed
`astro_lab <fingerprint>` line with the output.

### 2. Freeze the experiment before making a headline figure

A figure is paper-facing only after all of the following are known:

- exact source fingerprint;
- model size and token budget;
- optimizer names;
- hyperparameter configurations;
- whether variation is across seeds or across configurations;
- validation slice;
- wall-clock measurement convention.

Do not combine values from different validation slices or superseded source
fingerprints in one statistical comparison.

### 3. Update `artifacts/paper_results.json`

This file is a deliberately small paper snapshot. Add measurements, not prose
that guesses what an unfinished experiment will do. The historical ledger remains
in `artifacts/measured.json`.

### 4. Render figures

```bash
python scripts/figures/make_all.py
```

Or only the current empirical spine:

```bash
python scripts/figures/make_all.py \
  --only optimizer_journey horizon_v2 efficiency_v2 training_trajectories
```

### 5. Compile the manuscript

```bash
python scripts/paper/build.py
```

The builder renders figures first and then uses `latexmk` or `pdflatex` when one
is installed. A missing optional measurement may skip its figure, but LaTeX uses
a visible placeholder instead of silently failing or using an obsolete image.

## The final evidence matrix

Before calling the manuscript submission-ready, the following rows should be
filled:

| axis | minimum evidence | current state |
|---|---|---|
| horizon | 124M at 300/600/900/2700 under shared settings | running |
| stochastic replication | frozen 124M/900 comparison on seeds 100–102 or more | running / needs verification |
| scale | 355M/900 shared settings | running |
| state-of-the-art baselines | Muon, NorMuon, AdaMuon under the same protocol | partial but instrumented |
| compute | loss-vs-step and loss-vs-wall-clock/time-to-target | same-step timing measured; time-matched still needed |
| mechanism | ASTRO-MB, v2, γ=0, QKV split | strong exploratory evidence |
| trajectories | frozen validation-loss curves | runner ready; measurement pending |

If 355M or 2700 steps falsify the headline, the paper changes its thesis rather
than hiding those cells.

## W&B policy

W&B is optional. Use it to compare runs interactively and to share a dashboard,
but do not paste dashboard screenshots into the paper as the only source of a
claim. The final figure must be generated by Matplotlib from a committed JSON
artifact.

```bash
python scripts/paper/wandb_sync.py --project astro-paper \
  --input artifacts/trajectories.json
```

## Statistical language

Use the unit that was actually replicated.

- **five configurations** → “won 3/5 shared configurations,” not “3/5 seeds”;
- **three seeds** → show every seed and call the sign test directional;
- **one seed, many configs** → sensitivity/robustness evidence only;
- **incomplete cells** → mark them incomplete in table and figure.

Prefer paired deltas whenever optimizers saw the same seed/configuration/data.
Report the worst paired case next to the mean so one extreme favorable setting
cannot carry the story invisibly.

## Writing style

The main text should follow the compact empirical style used by strong optimizer
papers:

1. **Experiment setup.** What was held fixed, what varied, and why the comparison
   is fair.
2. **Experiment findings.** The actual measured pattern.
3. **Observation.** One sentence stating exactly what the evidence licenses.

Avoid “SOTA” unless the comparison really includes the relevant contemporary
baselines at the same budget. For the current project, “modern Muon-family
baselines” is the safer phrase until the final campaign is complete.

## Final pre-submission checks

```bash
python -m compileall scripts src
pytest
python scripts/figures/make_all.py
python scripts/paper/build.py
```

Then inspect the PDF manually for:

- unreadably small axis labels;
- captions split from figures;
- tables exceeding column width;
- stale `pending` labels;
- claims whose numbers are absent from `paper_results.json`;
- any figure rasterized when a vector PDF exists;
- any result described as a seed replication when it is actually a configuration
  sweep.
