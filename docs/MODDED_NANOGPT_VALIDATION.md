# ASTRO validation on modded-nanoGPT

## Scope

This document defines the **end-to-end Transformer validation** for ASTRO under the project's compute constraint: a single free Google Colab Tesla T4.

The goal is scientific validation, not an official modded-nanoGPT speedrun entry.

The upstream modded-nanoGPT optimization track fixes the model architecture, dataset, and batch size and asks contributors to minimize the number of optimization steps needed to reach 3.28 validation loss. The published reference setup uses multi-GPU A100/H100 systems. A T4 cannot reproduce that hardware regime, so ASTRO should not report a T4 wall-clock number as a speedrun record. It can, however, provide a controlled optimizer comparison using the same model family and data semantics. 

## Why use it

ASTRO's existing lab is intentionally small and diagnostic. modded-nanoGPT supplies a recognizable 124M Transformer training environment for the final end-task question:

> Does the ASTRO update rule improve real Transformer language-model optimization once the optimizer is placed inside a standard training loop rather than a synthetic or tiny benchmark?

This is the bridge between the mechanism experiments in ASTRO and a real language-model result.

## Required fairness

For the main comparison, hold these fixed across all optimizers:

| Item | Fixed value |
|---|---|
| model architecture | modded-nanoGPT-derived 124M GPT-style model |
| initialization | identical seed and initialization procedure |
| tokenizer | identical |
| dataset | identical FineWeb shard/token stream |
| sequence length | identical |
| global batch policy | identical within the T4 research profile |
| microbatching | identical |
| number of optimizer steps | identical |
| LR schedule | identical schedule shape unless the optimizer requires a separately justified scale |
| validation set | identical |
| precision policy | identical |
| random seed | recorded explicitly |

Only optimizer-specific hyperparameters may differ, and those differences must be part of a declared tuning protocol.

## T4 profile

The upstream trainer uses a large global batch and is designed around one or more A100/H100 GPUs. Do not blindly run that configuration on a T4.

Create a **T4 research profile** that keeps the model/data/task semantics but reduces the per-step memory footprint enough to run reliably on 16 GB VRAM. The profile should use:

- one CUDA device;
- smaller microbatches with gradient accumulation when necessary;
- the same effective global batch for every optimizer in the comparison;
- FP16/autocast where required by T4 compatibility;
- gradient checkpointing if necessary;
- a fixed sequence length;
- no architectural changes beyond memory accommodations.

The profile must be recorded in the experiment artifact so that the result is reproducible.

## Do not compare against the official 3.28 target directly

The official optimization benchmark requires the baseline architecture/data/batch and uses a multi-GPU environment. A T4 research run with altered microbatching or accumulation is not an apples-to-apples submission to that benchmark.

The ASTRO paper should instead report:

1. final validation loss at a fixed step budget;
2. validation-loss trajectory;
3. optimizer comparison at matched settings;
4. seed-to-seed variation;
5. training time on the stated T4 profile, clearly labelled as local hardware cost.

If the T4 profile is later made identical to the official baseline configuration, the run can be discussed separately as a possible benchmark comparison. Do not imply that merely using the same 124M model makes it an official record.

## Main comparison

The recommended final comparison is:

```text
AdamW
Muon
NorMuon
AdaMuon
ASTRO candidate
```

Use the candidate selected from `astro_lab.py`; do not tune ASTRO differently after looking at the final seed results.

### Recommended budget

For a compute-constrained first paper:

```text
1 shared protocol
1 candidate recipe
5 seeds for the final comparison
1–2 ablation seeds per component
1 fixed validation slice
```

More seeds are preferable to a larger model when the T4 is the limiting resource.

## Integration boundary

Do not vendor the whole modded-nanoGPT repository into ASTRO.

Keep the projects separate:

```text
ASTRO/
  src/astro/
  scripts/astro_lab.py
  docs/MODDED_NANOGPT_VALIDATION.md
  artifacts/

modded-nanogpt/
  records/track_3_optimization/train_gpt_simple.py
  data/
```

The ASTRO repository owns the optimizer and the research protocol. The external repository owns the model/data trainer. A small adapter or patch file should be added only when the exact T4 integration has been tested.

## Adapter requirements

When the integration script is added, it should:

1. import or link to `astro.Astro` rather than copy the optimizer implementation;
2. construct the optimizer from the model's actual parameter/module structure;
3. preserve all parameter coverage checks;
4. print a routing summary at startup;
5. record the exact ASTRO commit and modded-nanoGPT commit;
6. record PyTorch/CUDA/GPU information;
7. record the seed and all hyperparameters;
8. write results to a JSON artifact;
9. fail loudly on missing or duplicated parameters.

The official modded-nanoGPT trainer itself logs its source code and environment, which is useful precedent for this provenance requirement.

## Suggested experiment sequence

### Smoke test

Run each optimizer for a short budget to catch OOM, NaNs, routing mistakes, and throughput problems.

### Main run

Run the frozen candidate and baselines for the same step budget.

### Seeds

Repeat the main run across five seeds.

### Ablation

Run only the most important already-implemented ASTRO removals. Prefer one carefully chosen ablation per mechanism rather than a large Cartesian grid.

## Results to save

Each completed run should produce a machine-readable record containing at least:

```json
{
  "optimizer": "astro",
  "astro_commit": "...",
  "modded_nanogpt_commit": "...",
  "seed": 0,
  "steps": 1000,
  "model_parameters": 124000000,
  "sequence_length": 1024,
  "global_batch_tokens": 0,
  "precision": "fp16",
  "gpu": "Tesla T4",
  "final_val_loss": 0.0,
  "best_val_loss": 0.0,
  "training_seconds": 0.0,
  "peak_vram_gb": 0.0
}
```

Replace placeholders with measured values. Never hand-enter final results into the paper without retaining the raw record.

## Interpretation rules

A result counts as supportive evidence when the candidate improves validation loss under the **same declared protocol** and the improvement survives the seed analysis.

A result does not count as proof of universal superiority when it depends on:

- changing the model architecture;
- changing the data stream;
- giving one optimizer a larger batch;
- giving one optimizer extra forward/backward passes;
- tuning on the final test seeds;
- silently changing precision or accumulation policy;
- comparing T4 wall-clock time to the official H100 speedrun.

## External benchmark context

The modded-nanoGPT optimization track explicitly allows academic optimizer results that improve knowledge even when they do not establish a new record. That is the appropriate framing for this project if ASTRO does not reach the official target or lacks the hardware needed for an official comparison.

## Paper language

Recommended wording:

> We evaluate ASTRO on a 124M Transformer derived from the modded-nanoGPT optimization setup under a single-GPU T4 research profile. Because the official benchmark fixes a different hardware regime and batch configuration, we use this experiment as controlled end-task validation rather than as a speedrun submission.

Do not write "ASTRO beats modded-nanoGPT SOTA" unless the experiment actually satisfies the benchmark's published rules and the comparison is made against the relevant current record under the same protocol.
