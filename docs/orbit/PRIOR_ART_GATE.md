# ORBIT prior-art gate

**Audit date:** 2026-09-19

This file exists to prevent a fourth optimizer restart caused by discovering a close
precedent after expensive training. It is deliberately conservative. ORBIT may not be
presented as novel merely because the repository contains a new implementation.

## Ideas that are already occupied

The paper must **not** claim novelty for any of these by themselves:

1. optimizing the collapsed QK/OV products rather than Q/K/V/O factors;
2. fixed-rank or quotient-manifold optimization of QK/VO attention products;
3. mapping a product-space/tangent update back into low-rank factors;
4. gauge-equivariant optimization of factorized matrices;
5. learning or changing RoPE frequencies;
6. Muon-style spectral descent or Newton--Schulz polar updates;
7. architecture-aware or block-structured adaptive optimization.

The closest direct collision is Nicholas Knight, *Riemannian Gradient Descent for
Low-Rank Architectures*, arXiv:2606.02328. It explicitly treats QK and VO products as the
rank-constrained optimization objects while retaining a factorized implementation and also
covers MHA/GQA. This rules out the earlier plain `CircuitOpt` direction.

Other mandatory rechecks before submission include the 2026 work on faster Query--Key
learning / collapsed QK--OV dynamics (arXiv:2608.06776), gauge-sensitive optimizer dynamics
(arXiv:2608.05136), LoRA tangent/spectral optimizers (arXiv:2609.02734 and 2609.12123),
learnable RoPE-frequency work (arXiv:2607.10134), and symmetry/gauge-compatible Transformer
optimization (including arXiv:2606.29176).

## Frozen ORBIT-v0 novelty candidate

ORBIT-v0 does **not** optimize a single product `W_Q W_K^T`. RoPE makes the functional QK
operator relative-position dependent:

\[
C_\Delta = W_Q R_\Delta W_K^\top.
\]

Instead of materializing this family, ORBIT approximates the local attention-logit metric
frequency by frequency. For each 2D RoPE pair it accumulates Q/K covariances and averages the
corresponding rotated 2x2 sensitivity matrices over a fixed relative-position grid. These
metrics precondition the Q/K Muon directions jointly, followed by one joint norm restoration.
The model weights and inference graph remain standard.

The precise candidate claim is therefore narrow:

> **Training-time Q/K preconditioning whose metric is derived from the
> RoPE-conditioned relative-position bilinear interaction, implemented with frequency-local
> 2x2 statistics and no inference change.**

This wording is provisional until the exact implementation is searched again immediately
before any public novelty claim.

## Final novelty search checklist

Before the first public preprint, search all of the following combinations again across
arXiv, OpenReview, NeurIPS, ICML/PMLR, ACL, ICLR proceedings, Semantic Scholar/Google Scholar,
and public code:

- `RoPE optimizer query key`
- `rotary attention optimizer`
- `relative position query key preconditioner`
- `attention logits trust region optimizer`
- `RoPE natural gradient`
- `rotary natural gradient attention`
- `frequency-wise query key optimizer`
- `RoPE frequency preconditioner`
- `joint Q K preconditioning`
- `position-conditioned QK optimization`
- `attention logit metric optimizer`
- `function space optimizer attention`
- `2x2 RoPE covariance optimizer`
- `rotary bilinear optimization`

Also search the exact final equations, not only keywords.

## Go/no-go novelty rule

A paper may use a strong novelty statement only if no earlier work is found that simultaneously
satisfies all of the following:

1. the optimizer acts during pretraining, not only post-hoc editing;
2. it jointly treats Q and K;
3. its metric/update depends explicitly on RoPE relative-position rotations or an equivalent
   frequency-resolved relative-position QK representation;
4. this geometry changes the optimizer update rather than merely learning RoPE parameters;
5. it preserves standard inference-time Transformer weights.

If a direct match is found, stop and reframe before additional large-scale runs.
