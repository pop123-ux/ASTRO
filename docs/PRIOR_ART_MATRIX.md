# ASTRO prior-art boundary (paper campaign)

**As of 2026-09-16.** This is a claim-safety document, not a substitute for the
final literature review. Its purpose is to prevent the paper from calling a
known component novel merely because ASTRO arrived at it independently.

## What is already prior art

| mechanism | prior art | claim ASTRO must **not** make |
|---|---|---|
| matrix momentum + polar / Newton--Schulz update | Muon | "ASTRO invents matrix orthogonalized optimization" |
| post-orthogonalization neuron/row second moment with norm-preserving rescaling | **NorMuon** (arXiv:2510.05491) | "ASTRO invents post-polar row adaptation" |
| elementwise second moment on orthogonalized directions, sign-stabilized before orthogonalization, Adam-aligned RMS scaling | **AdaMuon** (arXiv:2507.11005) | "ASTRO is the first adaptive Muon" |
| splitting a fused QKV tensor before Muon orthogonalization | used in current Muon implementations / Megatron-LM; related chunking is central to **CMuon** (arXiv:2608.02502) | "ASTRO invents QKV splitting" |
| ordinary AdamW path for embeddings, head, norms and biases | standard Muon practice | "ASTRO invents hybrid matrix/scalar routing" |

The official AdaMuon implementation is the reference for the paper baseline:
<https://github.com/Chongjie-Si/AdaMuon>. The clean harness names this baseline
`adamuon_ref`; the older local `adamuon` variant is retained only for historical
reproduction.

## What ASTRO can investigate as its contribution

The defensible target is **not** novelty of either row adaptation or QKV
splitting in isolation. The paper campaign tests the more specific recipe and
interaction:

> **operator-aware spectral factorization followed by adaptive post-spectral
> redistribution under a preserved global update budget.**

In the current ASTRO-v2 implementation, fused semantic operators are split for
the polar transform, the resulting directions are concatenated, row-wise
second-moment redistribution is applied, and the total Frobenius magnitude is
restored before the final aspect scaling. This is a more specific update
construction than "Muon + split QKV" or "NorMuon" considered separately.

Whether that composition is sufficiently novel for the final title/abstract is
a **literature-review question**, not something a benchmark can prove. The
experiments can establish that the composition has a measurable effect and that
its benefit is not explained by beta, scheduling, or a weak baseline.

## The minimum attribution experiment

The paper uses one six-condition, frozen-hyperparameter experiment:

| condition | matrix beta | post-polar row redistribution | QKV treatment | purpose |
|---|---:|---:|---|---|
| Muon | .95 | off | fused | published reference recipe |
| Muon-M90 | .90 | off | fused | remove beta confound |
| ASTRO-γ0-NoSplit | .90 | off | fused | implementation-parity control |
| ASTRO-γ0 | .90 | off | split | split-only cell |
| ASTRO-v2-NoSplit | .90 | on | fused | row-only cell |
| ASTRO-v2 | .90 | on | split | full composition |

The last four cells form the required **2×2 factorial**:

```text
                         fused QKV          split Q/K/V
row redistribution off  γ0 + no split      γ0 + split
row redistribution on   v2 + no split      full ASTRO-v2
```

This is intentionally smaller and more informative than a large ablation zoo.
It measures the two relevant main effects and their interaction directly.

## Claim rules after the experiment

- If `ASTRO-γ0-NoSplit` materially differs from `Muon-M90` under the same
  configuration and seeds, stop and inspect the implementations before making
  mechanism claims; those two cells are intended as a parity control.
- If splitting helps but row redistribution does not, describe the result that
  way; do not credit row adaptation.
- If row redistribution helps but splitting does not, likewise narrow the claim.
- If the full cell improves beyond both one-mechanism cells, the paper can
  present evidence for a useful interaction/composition, while still citing the
  prior art for each ingredient.
- Negative ablations remain in the paper. They are evidence about the recipe,
  not failed experiments to be hidden.

## Why the baseline set is small

The clean headline experiment uses **AdamW, Muon, NorMuon, faithful AdaMuon, and
ASTRO-v2**. This mirrors the comparison logic of modern optimizer papers without
spending the T4 budget on a long list of weak or tangential optimizers:

- AdamW: standard adaptive reference;
- Muon: direct matrix-orthogonalization parent;
- NorMuon: closest row-adaptive prior art;
- AdaMuon: closest elementwise-adaptive Muon prior art;
- ASTRO-v2: proposed composition.

Additional optimizers belong in an appendix only if they answer a specific
reviewer question. They are not required merely to make the table longer.
