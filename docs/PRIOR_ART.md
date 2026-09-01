# Prior art, including work that overlaps claims this project made

Written after a literature pass that post-dates the first draft of the paper.
It is kept as its own file because the honest version of this project's
contribution is smaller than the first draft implied, and the record of *why*
is worth more than a quietly edited abstract.

## The fused-QKV split is not novel

The first draft presented splitting a fused QKV projection before
orthogonalisation as a new fix for a new defect. It is neither.

- Production Muon implementations already "split along the output dimension,
  run Polar Express per Q/K/V slice, and concatenate". Fusing is understood to
  cross-contaminate the three subspaces.
- *When and Why Grouping Attention Heads Accelerates Muon Optimization*
  (arXiv:2605.08933) studies head grouping directly.
- *CMuon: Accelerating and Stabilizing Diffusion Transformer Training via
  Chunked Momentum Orthogonalization* (arXiv:2608.02502) is the same idea for
  diffusion transformers.

**What survives.** Not the fix, and not the observation that fusing is bad. What
we have that the above do not is a *quantitative account of how bad, and when*:

- the identity in Proposition 1 (squared row norms of the polar factor are
  leverage scores of the momentum's row space, summing to `n`), which says the
  mass split is a leverage-localisation phenomenon rather than a scaling
  accident;
- the mechanism -- Q and K reach the loss through the softmax Jacobian and
  receive gradients 2.4-4.0x smaller than V on trained checkpoints;
- the measurement across eight pretrained checkpoints and both fusion layouts,
  including the finding below.

## The defect is largest at initialisation and attenuates with training

Measured on a T4 over eight checkpoints (`scripts/colab_probe.py`, part A),
V's share of the fused update:

| condition | V share |
|---|---|
| uniform, no defect | 0.333 |
| trained checkpoints, mean of 7 | 0.502 |
| random init, same architectures | 0.651 |
| random init, GPT-2 and SmolLM2 only | 0.762 |
| this project's 806K benchmark, step 1 from random init | 0.85 |

The 0.85 the first draft reported as a general property is an
initialisation-time value. On trained weights the localisation is roughly half
that.

This independently explains an observation in the literature that we did not
predict and only found afterwards: head-wise splitting "reduces validation loss
faster early in training, but is later overtaken by full-matrix Muon", a
stage-dependent trade-off. If the defect the split repairs is severe at
initialisation and decays, a split that is worth applying early is worth
relaxing later -- which is a testable prediction rather than a post-hoc story,
and is why `split_schedule` exists.

## The cost claim was wrong

The first draft claimed the split "costs less than not splitting". Muon's
Newton-Schulz transposes to operate on the smaller dimension, so a fused
`(3d, d)` already runs as `(d, 3d)`:

| | FLOPs per NS iteration |
|---|---|
| fused `(3d, d)` | `7d^3` |
| split `3 x (d, d)` | `9d^3` |

Predicted 1.29x slower; measured 1.33x mean on a T4 across `d` in
768-4096. The CPU measurement that supported the original claim was an artifact.

## Strong recent baselines this project has not beaten

- **Muown** (arXiv:2605.10797) decomposes the spectral norm into row-magnitude
  and row-coherence factors, identifies row magnitude as the driver of Muon's
  spectral-norm drift, and treats it as an explicit optimizer variable under
  l-infinity geometry. Reported to beat Muon, SOAP, AdamW and Lion on
  FineWeb-Edu from 124M to 2.7B, to widen the near-optimal learning-rate
  plateau, and to reduce weight-decay sensitivity. This is the closest work to
  our row-norm analysis and it is both more general and far better validated.
- **NorMuon** (arXiv:2510.05491), neuron-wise second moment after
  orthogonalisation -- the `variance_power = 1` endpoint here.
- **AdaMuon** (arXiv:2507.11005), elementwise second moment on the
  *orthogonalised* update, explicitly because the raw gradient carries the
  ill-conditioning the polar step exists to remove. This is the argument for
  post-placement, which this project reached by measurement and then shipped
  the wrong way round for several rounds.
- **Cautious Weight Decay** (arXiv:2510.12402), adopted here.

## Second literature pass, August 2026: two more contributions lost

This pass was run late -- after the figure suite was already built -- which is
itself the lesson. Both remaining "structural" contributions turn out to be
substantially prior art, and one of them has been directly tested and found not
to matter.

### Leverage-aware spectral optimization already exists

**Aurora: A Leverage-Aware Spectral Optimizer** (arXiv:2606.27715, Tilde
Research, 26 June 2026) "enforces row-uniformity of matrix parameter updates
while respecting Muon's polar factor geometry", starting from the observation
that "for tall matrix parameters ... the Muon update can have row norms that are
arbitrarily non-uniform". It names the failure mode -- neuron death in SwiGLU
MLPs, a "death spiral" where under-updated neurons receive ever less signal --
and reports state-of-the-art results on the modded-nanoGPT speedrun and the
lowest final loss at 1.1B over ~100B tokens. Riemannian-Aurora treats
orthogonality and equal row leverage as a single joint constraint rather than
post-hoc normalisation.

That is our Proposition 1, its motivation, and a far better validated remedy,
published seven weeks before this pass. **The leverage framing is not ours.**

What is left of it is thin and should be stated as thin: Aurora is about *tall*
matrices, so the wide-matrix corollary (for `m <= n` every row norm of `UV^T` is
exactly 1, hence row normalisation is provably inert there) may still be
unremarked -- but it is a one-line consequence of an identity someone else
published. The QKV application is a different target from Aurora's MLP one, and
is separately prior art anyway (above).

### Muon's non-convergence is known, intentional, and measured not to matter

Worse than scooped -- answered.

- The non-convergence is **documented design intent**, not a discovery. The
  modula documentation states that after repeated composition "singular values
  are allowed to lie in a band around one rather than converging to machine
  precision", and that the iteration "oscillates and in fact does not converge
  -- the cursed quintic iteration sacrifices convergence for speed."
- **How Much Orthogonalization Does Muon Need?** (arXiv:2606.00371, Hua Huang,
  NVIDIA, June 2026) tests the fix directly and finds "training quality is not
  governed monotonically by polar-decomposition accuracy": truncated Polar
  Express, the Muon-Jordan quintic, a cubic Newton-Schulz schedule, **and an
  explicit FP32 SVD polar factor reach nearly indistinguishable final loss on
  GPT-2 Small**, with cubic5 matching quintic Muon to about `1e-3` validation
  loss on 1B-4B hybrid MoE/Mamba models. An exact SVD polar factor is the
  perfect version of what our solved schedule approximates, at our benchmark
  scale, and it buys nothing.
- The deviation may be *load-bearing*: Muon with Newton-Schulz "implicitly
  down-weight[s] noise-dominated singular-vector directions", correlating with
  improved performance. Our `converging` flag may be removing something useful.
- **Spectral Scaling Laws of Muon** (arXiv:2606.04058) already tracks singular
  value quantiles of the momentum buffer across layers from 77M to 2.8B, which
  overlaps what `scripts/measure_drift.py` was built to measure, and finds the
  quantiles stabilise after a short burn-in.

`converging` was already default-off pending measurement, and ROUND4 already
recorded that "Muon's under-convergence may be acting as useful damping", so
nothing shipped on the strength of it. But it can no longer be sold as a
contribution, and the derivation of the fixed points at `0.868` and `1.264` is
at best a tidy closed form for a fact the field states qualitatively.

### Further baselines and neighbours found in this pass

Aurora (2606.27715); NVIDIA cubic5 (2606.00371); Spectral Scaling Laws of Muon
(2606.04058); The Spectral Dynamics and Noise Geometry of Muon (2606.08388);
Spectral Flattening Is All Muon Needs (2605.13079, max stable step size scales
with the *average* singular value); Beyond the Ideal: Analyzing the Inexact Muon
Update (2510.19933, inexactness couples to optimal step size and momentum);
HTMuon (2603.10067); Muon^p (2606.13867); Schatten-p (2605.19781); MuonEq
(2603.28254); Magma (2602.15322, masking that beats Cautious-Adam and Muon at
1B); Insights on Muon from Simple Quadratics (2602.11948).

## What this project can still honestly claim

Revised twice. In descending order of how much survives contact with the
literature:

1. **A controlled instance of small-scale sign inversion.** Cautious masking is
   worth `-0.0291` on 8 of 8 seeds at exact `p = 0.0078` at 1.17M and costs
   `+0.1341` at 124M, under one tokeniser and one protocol, with the
   elementwise/spectral parameter split matched to 0.5 percentage points
   *specifically to rule out the obvious confound* -- and the inversion survived
   that control. This is directly relevant to **Small-Scale Experiments: Are We
   There Yet?** (arXiv:2608.11859), which argues small-scale unreliability
   "stems from confounding hyperparameter factors rather than a fundamental
   limitation": our case is one where a confound was controlled and the
   inversion persisted.
2. **The protocol and its corrections record.** Six withdrawn claims and three
   measurement bugs, with the pattern that almost every one was caught by a
   control disagreeing with a headline rather than by inspection -- including
   the ranges-versus-counts bug, where equal tuning budgets were satisfied in
   letter while pairing an update scale with the wrong learning-rate range.
3. **The weight-tying routing bug**, which affects every matrix optimizer using
   the standard name-based policy.
4. **The stage dependence of the fused-projection defect**, measured on eight
   real checkpoints with a random-init control. The split is not ours and the
   defect is not ours; the decay with training, and the schedule that follows
   from it, still appear unremarked.
5. `variance_power`: the interior between Muon and NorMuon, which neither
   endpoint's authors search.

Items 1 and 2 are methodological. That is now the honest centre of gravity of
this work, and the paper should be organised around it rather than around an
optimizer.
