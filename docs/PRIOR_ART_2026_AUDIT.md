# ASTRO prior-art audit — September 2026

This document defines what ASTRO may and may not claim as novel before the paper is written.
It is a research prior-art review, **not** a legal/patent novelty opinion. Absence from this search
is not proof that no earlier work exists.

## Bottom line

The broad ASTRO-v2 recipe is **not defensibly novel merely because it combines Muon,
post-orthogonalisation adaptation, norm preservation, and QKV splitting**. Each of those ideas,
and even several of their pairwise combinations, already exists publicly.

The candidate contribution that remains plausibly distinct is narrower:

> **Independently polarise semantically distinct fused operator blocks (Q/K/V), then recombine
> them and perform one global post-spectral row-adaptive redistribution with one global
> Frobenius-magnitude restoration across the fused operator.**

This differs from the now-common prior-art pattern in which Q/K/V blocks are split and then
normalised/rescaled independently, matching the update they would receive as separate parameters.
The paper campaign therefore contains an explicit `astro_v2_blockwise` control to test whether
ASTRO's global redistribution after semantic splitting has measurable value.

If that control does not help, the strongest defensible paper framing should move away from
"new optimizer architecture" and toward architecture-aware optimizer analysis / benchmark evidence.

## Prior art that overlaps ASTRO

### Muon

Keller Jordan's Muon supplies momentum followed by Newton–Schulz orthogonalisation for hidden
matrices, with a separate AdamW path for embeddings, heads, gains and biases.

- https://github.com/KellerJordan/Muon

Not novel for ASTRO: matrix momentum, Newton–Schulz/polar geometry, auxiliary AdamW routing,
Muon-style aspect-ratio scaling.

### AdaMuon — 2025

AdaMuon adds elementwise second-moment adaptation after orthogonalisation, a sign transform
before the polar step, and RMS-aligned rescaling.

- Paper: https://arxiv.org/abs/2507.11005
- Reference implementation: https://github.com/Chongjie-Si/AdaMuon

Not novel for ASTRO: post-orthogonalisation adaptive second moments in general, norm/RMS restoration
as a way to decouple redistribution from total update magnitude.

### NorMuon — 2025

NorMuon adds neuron/row-wise second-moment normalisation after orthogonalisation and rescales the
result to preserve the update norm.

- Paper: https://arxiv.org/abs/2510.05491

Not novel for ASTRO: row-wise post-polar adaptation followed by Frobenius-norm preservation.
This is the closest prior art to ASTRO's redistribution stage.

### Split / chunked QKV Muon — public before this paper

By 2026, operator-aware splitting of fused QKV before orthogonalisation is public in several places.

- CMuon: https://arxiv.org/abs/2608.02502
- NVIDIA Emerging Optimizers: https://github.com/NVIDIA-NeMo/Emerging-Optimizers
- Microsoft Dion Muon/NorMuon implementation: https://github.com/microsoft/dion
- Muonium: https://github.com/tonyf/muonium

These implementations explicitly support splitting fused QKV into semantic row blocks before
Newton–Schulz. Some support NorMuon together with split blocks.

Not novel for ASTRO: detecting fused QKV, splitting Q/K/V before orthogonalisation, or claiming
that fused treatment can couple functionally distinct operators.

### Schedule-free NorMuon — 2026

SF-NorMuon combines NorMuon's row-wise spectral adaptation with schedule-free optimisation.

- Paper: https://arxiv.org/abs/2605.23061

Not directly overlapping ASTRO's candidate contribution, but it demonstrates that composing NorMuon
with other optimizer machinery is itself an active and crowded design space.

## Candidate ASTRO distinction

The inspected ASTRO path does the following for a fused QKV operator:

1. split the Nesterov direction into semantic Q/K/V row blocks;
2. independently apply the polar/Newton–Schulz transform to each block;
3. concatenate those filtered blocks back into one fused direction;
4. update one row-wise second-moment vector across the concatenated direction;
5. redistribute rows by that adaptive statistic;
6. restore **one global Frobenius norm** after redistribution;
7. apply aspect scaling per semantic block.

The important distinction is steps 3–6. A conventional split-NorMuon implementation instead treats
blocks independently through both orthogonalisation and norm-preserving adaptation, so each block
retains its own magnitude budget. ASTRO permits adaptive movement of update energy across semantic
blocks while preserving only the total fused-operator magnitude.

This is the exact hypothesis the paper must test. Do not call the whole recipe novel if the evidence
only shows that QKV splitting or NorMuon-style adaptation helps.

## Required novelty-control experiment

The paper campaign registers:

- `astro_v2` — split polar geometry, then global redistribution/restoration;
- `astro_v2_blockwise` — same beta, scalar path, split geometry, row statistics, decay and schedule,
  but row adaptation and Frobenius restoration are performed independently inside each block;
- `astro_v2_nosplit` — removes semantic splitting;
- `astro_v2_gamma0` — removes adaptive redistribution while preserving the rest of the recipe.

The cleanest evidence for the candidate contribution is therefore:

```text
ASTRO-v2             vs ASTRO-v2-Blockwise   -> global cross-block redistribution
ASTRO-v2             vs ASTRO-v2-NoSplit     -> semantic split geometry
ASTRO-v2             vs ASTRO-v2-gamma0      -> row adaptation itself
```

A positive result on `astro_v2` vs `astro_v2_blockwise` would establish empirical value for the
narrow distinction identified here. It still would not, by itself, prove exhaustive literature
novelty; the final paper should repeat/update the search immediately before submission.

## Claim rules for the paper

Allowed only if measured:

- "ASTRO-v2 achieved lower validation loss than the tested baselines under the declared protocol."
- "Global post-split redistribution improved over the matched blockwise control across X/Y settings."
- "The candidate design differs from the prior-art split-NorMuon pattern by restoring magnitude
  globally after recombining semantic blocks."

Do not write without stronger evidence:

- "ASTRO invented QKV splitting."
- "ASTRO invented post-polar row adaptation."
- "ASTRO is the first optimizer to combine Muon and NorMuon ideas."
- "ASTRO is universally better than Muon/NorMuon/AdaMuon."
- "The literature proves no previous method performs global cross-block redistribution."

## Final pre-submission novelty check

Immediately before submission, repeat searches for at least:

```text
Muon global redistribution QKV
NorMuon fused QKV global norm
split QKV row-wise adaptive Muon
cross-block NorMuon
operator-aware Muon adaptive redistribution
fused projection orthogonalization global Frobenius
```

Record the search date and any newly surfaced papers/repositories in this document.
