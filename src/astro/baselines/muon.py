"""Muon and NorMuon, implemented faithfully as published.

These are *baselines*, and the single most common way optimizer papers overstate
their results is by comparing against a baseline that is not quite the published
algorithm or is not tuned as hard as the proposed method [Wen et al. 2025]. So
these follow the reference implementations rather than being re-expressed as
special cases of :class:`~astro.optimizer.Astro`, even though several of them
are mathematically reachable that way.

Differences from ASTRO worth noting, because they are deliberate:

* Muon uses **Nesterov** momentum by default and the scaling
  ``max(1, m/n) ** 0.5``, not the RMS-matching scale. Both come from the
  reference implementation at github.com/KellerJordan/Muon.
* NorMuon applies its neuron-wise second moment **after** orthogonalisation,
  which is what the paper specifies; ASTRO applies variance adaptation before.
  That ordering is one of the things the ablation measures.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from astro.polar import muon_filter
from astro.routing import classify_module, matrix_view

__all__ = ["Muon", "NorMuon"]


class Muon(Optimizer):
    """MomentUm Orthogonalized by Newton-schulz, with an AdamW path for the rest.

    Parameters
    ----------
    params:
        Parameter groups; each may set ``spectral`` to override routing.
    lr:
        Learning rate for the spectral path, in units of spectral norm per step.
    momentum:
        Momentum coefficient. 0.95 is the reference default.
    nesterov:
        Reference default is ``True``. NVIDIA's scaling report found it made no
        difference at large batch and dropped it; it is exposed so that can be
        checked rather than assumed.
    ns_steps:
        Newton-Schulz iterations.
    adamw_lr:
        Learning rate for non-matrix parameters. Muon's README is explicit that
        embeddings, heads, gains and biases must not be orthogonalised.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        *,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        neuron_norm: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            spectral=True,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        super().__init__(params, defaults)
        self.filter = muon_filter(ns_steps)
        self.neuron_norm = bool(neuron_norm)

    @classmethod
    def from_model(cls, model: nn.Module, lr: float = 0.02, **kwargs: Any) -> Muon:
        """Route parameters with :mod:`astro.routing`, as ASTRO does.

        Using identical routing for Muon and ASTRO is deliberate: it isolates the
        update rule from the routing policy, so a win cannot be attributed to
        routing alone.
        """
        specs = classify_module(model)
        spectral, scalar = [], []
        for name, param in model.named_parameters():
            if param.requires_grad:
                (spectral if specs[name].is_spectral else scalar).append(param)
        groups: list[dict[str, Any]] = []
        if spectral:
            groups.append({"params": spectral, "spectral": True})
        if scalar:
            groups.append({"params": scalar, "spectral": False})
        return cls(groups, lr=lr, **kwargs)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            spectral = bool(group["spectral"])
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(param)
                    if not spectral:
                        state["variance"] = torch.zeros_like(param)
                    elif self.neuron_norm:
                        state["neuron"] = torch.zeros(
                            matrix_view(param).size(0), dtype=param.dtype, device=param.device
                        )
                state["step"] += 1

                if not spectral:
                    self._adamw(param, state, group)
                    continue

                grad = matrix_view(param.grad)
                buf = matrix_view(state["momentum"])
                buf.lerp_(grad, 1.0 - group["momentum"])
                update = grad.lerp(buf, group["momentum"]) if group["nesterov"] else buf

                update = self.filter(update)
                # Reference scaling: preserves update RMS across aspect ratios.
                update = update * max(1.0, update.size(-2) / update.size(-1)) ** 0.5

                if self.neuron_norm:
                    update = self._normuon_scale(update, grad, state, group)

                flat = param.view(update.shape) if param.ndim > 2 else param
                if group["weight_decay"]:
                    flat.mul_(1.0 - group["lr"] * group["weight_decay"])
                flat.add_(update, alpha=-group["lr"])
        return loss

    def _normuon_scale(
        self,
        update: torch.Tensor,
        grad: torch.Tensor,
        state: dict[str, Any],
        group: dict[str, Any],
    ) -> torch.Tensor:
        """NorMuon's row-wise second moment, applied after orthogonalisation.

        Muon gives every singular value the same magnitude, which leaves the
        per-neuron update norms unequal. NorMuon rebalances them, then rescales
        so the overall update norm is preserved.
        """
        beta2 = group["adamw_betas"][1]
        neuron = state["neuron"]
        neuron.mul_(beta2).add_(update.square().mean(dim=1), alpha=1.0 - beta2)
        corrected = neuron / (1.0 - beta2 ** state["step"])
        scaled = update / (corrected.sqrt().add_(group["adamw_eps"]).unsqueeze(1))
        return scaled * (update.norm() / scaled.norm().clamp_min(1e-12))

    @staticmethod
    def _adamw(param: torch.Tensor, state: dict[str, Any], group: dict[str, Any]) -> None:
        grad = param.grad
        beta1, beta2 = group["adamw_betas"]
        lr, eps = group["adamw_lr"], group["adamw_eps"]
        momentum, variance = state["momentum"], state["variance"]
        momentum.lerp_(grad, 1.0 - beta1)
        variance.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        bias1 = 1.0 - beta1 ** state["step"]
        bias2 = 1.0 - beta2 ** state["step"]
        if group["weight_decay"]:
            param.mul_(1.0 - lr * group["weight_decay"])
        param.addcdiv_(momentum / bias1, (variance / bias2).sqrt_().add_(eps), value=-lr)


class NorMuon(Muon):
    """Muon with neuron-wise second-moment normalisation after orthogonalisation."""

    def __init__(self, params: Iterable[Any], **kwargs: Any) -> None:
        kwargs.setdefault("neuron_norm", True)
        super().__init__(params, **kwargs)
