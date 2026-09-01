"""ASTRO -- Anchored Spectral Trust-Region Optimizer.

Motivation
----------
Matrix-structured optimizers (Muon, Shampoo, SOAP) beat AdamW when training from
scratch, and the honest margin is roughly 1.1-1.4x rather than the 2x often
claimed [Wen et al. 2025]. But there is a regime where they *lose*: fully
fine-tuning a model whose weights were pretrained with Adam. Qu et al. (ICML
2026) show this "optimizer mismatch" disrupts pretrained knowledge, and that the
disruption scales with **update strength**. Their fix is LoRA, which caps update
strength by confining the update to a rank-``r`` subspace.

That is a blunt instrument: it buys protection by giving up full-rank updates.
ASTRO caps update strength *directly* -- a trust region around the pretrained
weights -- and keeps the update full-rank.

The update
----------
For a parameter reshaped to its operator matrix ``W`` (see :mod:`astro.routing`):

1. **Momentum first.** ``M <- b1 M + (1-b1) G``. Momentum before orthogonalisation
   is a spectral denoiser: it widens the gap between signal and noise singular
   values, which stabilises the subspaces the next step relies on. Nesterov is
   deliberately absent -- NVIDIA measured it as no help at scale.
2. **Row-wise variance adaptation.** ``M <- M / (sqrt(v) + eps)`` with ``v`` a
   per-output-neuron second moment. Frans et al. isolate variance adaptation as
   the ingredient that explains SOAP's edge over Muon, and it costs ``m`` floats
   rather than ``m*n``.
3. **Spectral normalisation.** Apply a :class:`~astro.polar.SpectralFilter`,
   i.e. approximately set the singular values to one.
4. **Update-RMS matching.** Rescale so the update has the same expected Frobenius
   norm as an Adam update on the same tensor. Without this, a learning rate
   cannot be transferred between the spectral and scalar paths and every
   comparison against AdamW silently confounds "better algorithm" with
   "different effective step size".
5. **Norm control** -- decoupled weight decay, or Hyperball's explicit radius
   constraint.
6. **Anchored trust region** -- project the result back into a ball of radius
   ``rho * ||W_0||`` around the pretrained weights.

Setting ``anchor=False``, ``norm_control="none"``, ``cautious=False`` and
``variance="none"`` recovers Muon exactly (``post_normalize`` is off by
default). That reduction is asserted elementwise against an independently written reference in
``tests/test_optimizer.py`` and is what makes the ablation meaningful.

What the measurements changed
-----------------------------
Defaults here are set by experiment, and several reverse an earlier choice. The
full chain of evidence is in ``docs/IMPROVEMENT_LOG.md``; in brief:

* ``cautious=False``. Once the headline default, now off, because its sign
  depends on scale: worth ``-0.0291`` against Muon on 8 of 8 seeds at 1.17M
  (exact ``p = 0.0078``) and ``+0.1341`` at 124M under the same tokeniser and
  protocol.
* ``variance_placement="post"``, ``nesterov=True``, ``update_scale="muon"``,
  ``betas=(0.95, 0.95)`` -- selected by the round-3 sweep, and worth 0.041 on
  ``gpt_scratch`` against the shipped defaults they replaced.
* ``post_normalize=False``, **reverted**. It was defaulted on from a step-size
  argument -- Muon's iteration delivers only 0.77-0.83 of the theoretical
  update norm and the ratio drifts 4.7% over a 40-step run, so pinning it makes
  the learning rate name a fixed step size. The loss disagreed anyway: at 124M
  over 900 steps, pinning costs **0.0222**.
* ``anchor=False``. Elastic beats hard by 2.2%, so the restoring-force
  formulation is the better one; it is still not better than having no anchor,
  across three independent measurements.

``update_scale="trust"`` is available and **not** on by default: it sets each
layer's step from its own norm rather than from its shape, which is the angular-learning-rate control Hyperball argues weight decay has been doing by accident. It is registered as a swept variant and will be defaulted on only if it wins at equal budget.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any, Literal

import torch
from torch import nn
from torch.optim import Optimizer

from astro.polar import SpectralFilter, deadzone_filter, muon_filter, polar_filter
from astro.routing import classify_module, fused_block_sizes, matrix_view

__all__ = [
    "Astro",
    "apply_weight_decay",
    "astro_matrix_update",
    "cautious_mask",
    "rms_match_scale",
]

VarianceAxis = Literal["row", "col", "none"]
VariancePlacement = Literal["pre", "post", "both"]
NormControl = Literal["none", "wd", "hyperball"]
TrustScope = Literal["global", "param"]
AnchorMode = Literal["hard", "elastic"]
UpdateScale = Literal["adam_rms", "muon", "trust"]


def cautious_mask(update: torch.Tensor, grad: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    mask = (update * grad > 0).to(update.dtype)
    return update * mask * (mask.numel() / mask.sum().clamp_min(eps))


def apply_weight_decay(
    param: torch.Tensor, update: torch.Tensor, rate: float, *, cautious: bool,
    rescale: bool = False, eps: float = 1e-12
) -> None:
    if rate == 0.0:
        return
    if not cautious:
        param.mul_(1.0 - rate)
        return
    agrees = update * param > 0
    if rescale:
        kept = float(agrees.sum())
        if kept <= 0.0:
            return
        rate = rate * agrees.numel() / max(kept, eps)
    param.add_(torch.where(agrees, param, torch.zeros_like(param)), alpha=-rate)


def rms_match_scale(rows: int, cols: int, beta1: float) -> float:
    return math.sqrt((1.0 - beta1) / (1.0 + beta1)) * math.sqrt(max(rows, cols))


def _filter_block(
    spectral_filter: SpectralFilter, block: torch.Tensor, post_normalize: bool
) -> torch.Tensor:
    filtered = spectral_filter(block)
    if not post_normalize:
        return filtered
    target = math.sqrt(min(block.size(0), block.size(1)))
    return filtered * (target / filtered.norm().clamp_min(1e-12))


@torch.no_grad()
def astro_matrix_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    variance: torch.Tensor | None,
    spectral_filter: SpectralFilter,
    *,
    variance_post: torch.Tensor | None = None,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
    variance_axis: VarianceAxis,
    variance_placement: VariancePlacement,
    rms_match: bool,
    normalize_direction: bool,
    cautious: bool,
    nesterov: bool = False,
    update_scale: UpdateScale = "adam_rms",
    variance_power: float = 1.0,
    spectral_blend: float = 0.0,
    post_normalize: bool = False,
    equilibrate: bool = False,
    blocks: tuple[int, ...] | int = 1,
    weight: torch.Tensor | None = None,
    trust_clip: float = 1e3,
) -> torch.Tensor:
    momentum.lerp_(grad, 1.0 - beta1)
    axis = 1 if variance_axis == "row" else 0
    adapt = variance_axis != "none"

    update = grad.lerp(momentum, beta1) if nesterov else momentum
    if adapt and variance is not None and variance_placement in ("pre", "both"):
        variance.mul_(beta2).add_(grad.square().mean(dim=axis), alpha=1.0 - beta2)
        denom = (variance / (1.0 - beta2**step)).sqrt().add_(eps)
        update = momentum / (denom.unsqueeze(1) if variance_axis == "row" else denom.unsqueeze(0))

    if equilibrate:
        norms = update.norm(dim=1, keepdim=True).clamp_min(eps)
        balanced = update / norms
        update = balanced * (update.norm() / balanced.norm().clamp_min(eps))

    sizes = (
        blocks if isinstance(blocks, tuple) else (update.size(0) // max(1, blocks),) * blocks
    )
    if len(sizes) > 1:
        direction = torch.cat(
            [_filter_block(spectral_filter, c, post_normalize)
             for c in update.split(list(sizes), dim=0)],
            dim=0,
        )
    else:
        direction = _filter_block(spectral_filter, update, post_normalize)

    if adapt and variance_post is not None and variance_placement in ("post", "both"):
        variance_post.mul_(beta2).add_(direction.square().mean(dim=axis), alpha=1.0 - beta2)
        moment = variance_post / (1.0 - beta2**step)
        denom = moment.pow(0.5 * variance_power).add_(eps)
        scaled = direction / (denom.unsqueeze(1) if variance_axis == "row" else denom.unsqueeze(0))
        direction = scaled * (direction.norm() / scaled.norm().clamp_min(1e-12))

    if spectral_blend > 0.0:
        reference = update * (direction.norm() / update.norm().clamp_min(1e-12))
        direction = direction.lerp(reference, spectral_blend)

    if cautious:
        direction = cautious_mask(direction, grad, eps)

    if normalize_direction:
        return direction / (direction.norm() + eps)
    if rms_match:
        cols = direction.size(1)
        if update_scale == "trust":
            if weight is None:
                raise ValueError("update_scale='trust' needs the weight tensor")
            magnitude = direction.norm().clamp_min(1e-12)
            factor = (weight.norm() / magnitude).clamp(1.0 / trust_clip, trust_clip)
            return direction * factor

        def scale_for(rows: int) -> float:
            if update_scale == "muon":
                return max(1.0, rows / cols) ** 0.5
            return rms_match_scale(rows, cols, beta1)

        if len(sizes) > 1:
            direction = torch.cat(
                [block * scale_for(block.size(0))
                 for block in direction.split(list(sizes), dim=0)],
                dim=0,
            )
        else:
            direction = direction * scale_for(direction.size(0))
    return direction


class Astro(Optimizer):
    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.95, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        *,
        variance: VarianceAxis = "row",
        variance_placement: VariancePlacement = "post",
        cautious: bool = False,
        split_fused: bool = True,
        split_steps: int | None = None,
        nesterov: bool = True,
        update_scale: UpdateScale = "muon",
        variance_power: float = 1.0,
        spectral_blend: float = 0.0,
        post_normalize: bool = False,
        equilibrate: bool = False,
        trust_clip: float = 1e3,
        cautious_wd_rescale: bool = False,
        converging: bool = False,
        cautious_wd: bool = True,
        dead_zone: float = 0.0,
        ns_steps: int = 5,
        anchor: bool = False,
        anchor_mode: AnchorMode = "elastic",
        rho: float = 0.1,
        anchor_strength: float = 1e-2,
        rho_warmup_steps: int = 0,
        norm_control: NormControl = "wd",
        rms_match: bool = True,
        trust_scope: TrustScope = "global",
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"betas must lie in [0, 1), got {betas}")
        if rho <= 0:
            raise ValueError(f"rho must be positive, got {rho}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            spectral=True, variance=variance,
            variance_placement=variance_placement, cautious=cautious,
            split_fused=split_fused, split_steps=split_steps,
            nesterov=nesterov, update_scale=update_scale,
            variance_power=variance_power, spectral_blend=spectral_blend,
            post_normalize=post_normalize, equilibrate=equilibrate,
            trust_clip=trust_clip, cautious_wd=cautious_wd,
            cautious_wd_rescale=cautious_wd_rescale, anchor=anchor,
            anchor_mode=anchor_mode, rho=rho,
            anchor_strength=anchor_strength, rho_warmup_steps=rho_warmup_steps,
            norm_control=norm_control, rms_match=rms_match,
        )
        super().__init__(params, defaults)

        if trust_scope not in ("global", "param"):
            raise ValueError(f"trust_scope must be 'global' or 'param', got {trust_scope!r}")
        self.trust_scope: TrustScope = trust_scope
        self.dead_zone = float(dead_zone)
        self.ns_steps = int(ns_steps)
        if dead_zone > 0:
            self._filter: SpectralFilter = deadzone_filter(dead_zone, ns_steps)
        elif converging:
            self._filter = polar_filter(ns_steps, 1e-3)
        else:
            self._filter = muon_filter(ns_steps)

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        lr: float = 3e-4,
        *,
        scalar_lr_mult: float = 1.0,
        routing_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "Astro":
        specs = classify_module(model, **(routing_kwargs or {}))
        spectral, scalar = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            (spectral if specs[name].is_spectral else scalar).append(param)

        groups: list[dict[str, Any]] = []
        if spectral:
            groups.append({"params": spectral, "spectral": True, "lr": lr})
        if scalar:
            groups.append({"params": scalar, "spectral": False, "lr": lr * scalar_lr_mult})
        optimizer = cls(groups, lr=lr, **kwargs)
        optimizer.routing = specs  # type: ignore[attr-defined]
        optimizer._names = {id(p): n for n, p in model.named_parameters()}  # type: ignore[attr-defined]
        return optimizer

    def _blocks_for(self, param: torch.Tensor) -> tuple[int, ...]:
        name = getattr(self, "_names", {}).get(id(param), "")
        return fused_block_sizes(name, matrix_view(param))

    @staticmethod
    def _blocks_at(
        group: dict[str, Any], state: dict[str, Any], step: int
    ) -> tuple[int, ...]:
        limit = group.get("split_steps")
        if limit is not None and step > limit:
            return (sum(state["blocks"]),)
        return state["blocks"]

    def _rho_at(self, group: dict[str, Any], step: int) -> float:
        warmup = int(group["rho_warmup_steps"])
        rho = float(group["rho"])
        return rho if warmup <= 0 else rho * min(1.0, step / warmup)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps = group["lr"], group["eps"]
            spectral = bool(group["spectral"])
            norm_control: NormControl = group["norm_control"]
            use_anchor = bool(group["anchor"])

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("Astro does not support sparse gradients")

                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(param)
                    state["blocks"] = (matrix_view(param).size(0),)
                    if spectral:
                        view = matrix_view(param)
                        if group["split_fused"]:
                            state["blocks"] = self._blocks_for(param)
                        if group["variance"] != "none":
                            size = view.size(0) if group["variance"] == "row" else view.size(1)
                            placement = group["variance_placement"]
                            if placement in ("pre", "both"):
                                state["variance"] = torch.zeros(size, dtype=param.dtype, device=param.device)
                            if placement in ("post", "both"):
                                state["variance_post"] = torch.zeros(size, dtype=param.dtype, device=param.device)
                        if norm_control == "hyperball":
                            state["radius"] = param.detach().norm().clamp_min(eps)
                    else:
                        state["variance"] = torch.zeros_like(param)
                    if use_anchor:
                        state["anchor"] = param.detach().clone()
                        state["anchor_norm"] = state["anchor"].norm().clamp_min(eps)
                state["step"] += 1
                step = state["step"]

                if not spectral:
                    state["blocks"] = (param.size(0),)
                    self._adamw_step(param, grad, state, lr, beta1, beta2, eps, group)
                    if use_anchor and group["anchor_mode"] == "elastic":
                        self._elastic_pull(param, state, lr * group["anchor_strength"])
                    elif use_anchor and self.trust_scope == "param":
                        self._project_to_trust_region(param, state, self._rho_at(group, step), eps)
                    continue

                view_grad = matrix_view(grad)
                view_mom = matrix_view(state["momentum"])
                hyperball = norm_control == "hyperball"

                update = astro_matrix_update(
                    view_grad, view_mom, state.get("variance"), self._filter,
                    variance_post=state.get("variance_post"), beta1=beta1, beta2=beta2,
                    eps=eps, step=step, variance_axis=group["variance"],
                    variance_placement=group["variance_placement"],
                    rms_match=bool(group["rms_match"]), normalize_direction=hyperball,
                    cautious=bool(group["cautious"]), nesterov=bool(group["nesterov"]),
                    blocks=self._blocks_at(group, state, step), update_scale=group["update_scale"],
                    variance_power=group["variance_power"], spectral_blend=group["spectral_blend"],
                    post_normalize=group["post_normalize"], equilibrate=group["equilibrate"],
                    weight=param.view(view_mom.shape) if param.ndim > 2 else param,
                    trust_clip=group["trust_clip"],
                )

                flat = param.view(update.shape) if param.ndim > 2 else param
                if hyperball:
                    radius = state["radius"]
                    flat.add_(update, alpha=-lr * float(radius))
                    flat.mul_(radius / flat.norm().clamp_min(eps))
                else:
                    if norm_control == "wd":
                        apply_weight_decay(
                            flat, update, lr * group["weight_decay"],
                            cautious=group["cautious_wd"], rescale=group["cautious_wd_rescale"],
                        )
                    flat.add_(update, alpha=-lr)

                if use_anchor and group["anchor_mode"] == "elastic":
                    self._elastic_pull(flat, state, lr * group["anchor_strength"])
                elif use_anchor and self.trust_scope == "param":
                    self._project_to_trust_region(flat, state, self._rho_at(group, step), eps)

        if self.trust_scope == "global" and any(
            g["anchor"] and g["anchor_mode"] == "hard" for g in self.param_groups
        ):
            self._project_globally()
        return loss

    @staticmethod
    @torch.no_grad()
    def _elastic_pull(flat: torch.Tensor, state: dict[str, Any], strength: float) -> None:
        if strength <= 0.0:
            return
        anchor = state["anchor"].view(flat.shape)
        flat.add_(flat - anchor, alpha=-strength)

    @torch.no_grad()
    def _project_globally(self) -> None:
        total_drift_sq = 0.0
        total_anchor_sq = 0.0
        tracked: list[tuple[torch.Tensor, torch.Tensor]] = []
        rho = None

        for group in self.param_groups:
            if not group["anchor"]:
                continue
            for param in group["params"]:
                state = self.state[param]
                if "anchor" not in state:
                    continue
                if rho is None:
                    rho = self._rho_at(group, state["step"])
                anchor = state["anchor"]
                total_drift_sq += float((param - anchor).pow(2).sum())
                total_anchor_sq += float(anchor.pow(2).sum())
                tracked.append((param, anchor))

        if rho is None or not tracked or total_anchor_sq <= 0.0:
            return
        drift = math.sqrt(total_drift_sq)
        limit = rho * math.sqrt(total_anchor_sq)
        if drift <= limit or drift <= 0.0:
            return
        scale = limit / drift
        for param, anchor in tracked:
            param.copy_(anchor + (param - anchor) * scale)

    @staticmethod
    @torch.no_grad()
    def _project_to_trust_region(
        flat: torch.Tensor, state: dict[str, Any], rho: float, eps: float
    ) -> None:
        anchor = state["anchor"].view(flat.shape)
        limit = rho * float(state["anchor_norm"])
        drift = flat - anchor
        distance = float(drift.norm())
        if distance > limit and distance > eps:
            flat.copy_(anchor + drift * (limit / distance))

    @staticmethod
    def _adamw_step(
        param: torch.Tensor,
        grad: torch.Tensor,
        state: dict[str, Any],
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        group: dict[str, Any],
    ) -> None:
        step = state["step"]
        momentum, variance = state["momentum"], state["variance"]
        momentum.lerp_(grad, 1.0 - beta1)
        variance.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        bias1 = 1.0 - beta1**step
        bias2 = 1.0 - beta2**step
        denom = (variance / bias2).sqrt_().add_(eps)
        update = (momentum / bias1) / denom
        apply_weight_decay(
            param, update, lr * group["weight_decay"], cautious=group["cautious_wd"],
            rescale=group["cautious_wd_rescale"],
        )
        param.add_(update, alpha=-lr)
