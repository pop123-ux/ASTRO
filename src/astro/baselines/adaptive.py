"""Scalar-path baselines: AdEMAMix and Cautious AdamW.

Both are strong, cheap, widely-reported AdamW variants, and both are included
because they attack the problem on an axis ASTRO does not. If a spectral method
cannot beat a well-tuned *scalar* variant, the extra machinery is not paying for
itself, and the benchmark should be able to say so.

AdEMAMix (Pareyre-style dual EMA, Ablin et al.)
    Keeps two first-moment EMAs -- one fast, one very slow (``beta3 ~ 0.9999``)
    -- and mixes them. The slow one lets gradients from tens of thousands of
    steps ago still contribute, which is why the reported gains grow with run
    length: at 1.3B parameters, AdEMAMix on 101B tokens matches AdamW on 197B.

Cautious AdamW
    Masks the update wherever it disagrees in sign with the current gradient,
    then renormalises. A one-line change that consistently beats vanilla AdamW
    in the 2026 benchmark sweeps.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch.optim import Optimizer

__all__ = ["AdEMAMix", "CautiousAdamW"]


class AdEMAMix(Optimizer):
    """AdamW with an additional slow first-moment EMA.

    Parameters
    ----------
    betas:
        ``(beta1, beta2, beta3)``. ``beta3`` drives the slow EMA and should be
        much closer to 1 than ``beta1``.
    alpha:
        Mixing weight on the slow EMA.
    beta3_warmup, alpha_warmup:
        Steps over which ``beta3`` and ``alpha`` ramp to their final values.
        Without the ramp the slow EMA is badly biased early and destabilises the
        first few thousand steps.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 5.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        beta3_warmup: int = 0,
        alpha_warmup: int = 0,
    ) -> None:
        super().__init__(
            params,
            dict(
                lr=lr,
                betas=betas,
                alpha=alpha,
                eps=eps,
                weight_decay=weight_decay,
                beta3_warmup=beta3_warmup,
                alpha_warmup=alpha_warmup,
            ),
        )

    @staticmethod
    def _linear_ramp(final: float, warmup: int, step: int) -> float:
        """Linear ramp used for ``alpha``."""
        if warmup <= 0 or step >= warmup:
            return final
        return final * (step / warmup)

    @staticmethod
    def _beta3_ramp(start: float, final: float, warmup: int, step: int) -> float:
        """Ramp ``beta3`` from ``start`` to ``final`` in log-half-life space.

        Interpolating linearly between 0.9 and 0.9999 would spend almost the
        whole ramp at an EMA horizon indistinguishable from the endpoint. The
        published schedule interpolates the *half-life* instead, which makes the
        horizon grow smoothly.
        """
        if warmup <= 0 or step >= warmup:
            return final
        if start >= 1.0 or final >= 1.0:
            return final
        progress = step / warmup
        log_start, log_final = math.log(start), math.log(final)
        denominator = (1.0 - progress) * log_final + progress * log_start
        if denominator == 0.0:
            return final
        return math.exp(log_start * log_final / denominator)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3_final = group["betas"]
            lr, eps = group["lr"], group["eps"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["fast"] = torch.zeros_like(param)
                    state["slow"] = torch.zeros_like(param)
                    state["variance"] = torch.zeros_like(param)
                state["step"] += 1
                step = state["step"]

                beta3 = self._beta3_ramp(beta1, beta3_final, group["beta3_warmup"], step)
                alpha = self._linear_ramp(group["alpha"], group["alpha_warmup"], step)

                fast, slow, variance = state["fast"], state["slow"], state["variance"]
                fast.lerp_(grad, 1.0 - beta1)
                slow.lerp_(grad, 1.0 - beta3)
                variance.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias1 = 1.0 - beta1**step
                bias2 = 1.0 - beta2**step
                denom = (variance / bias2).sqrt_().add_(eps)

                if group["weight_decay"]:
                    param.mul_(1.0 - lr * group["weight_decay"])
                # The slow EMA is deliberately left un-bias-corrected: it starts
                # near zero and grows in, which is the intended warm-in.
                param.addcdiv_(fast / bias1 + alpha * slow, denom, value=-lr)
        return loss


class CautiousAdamW(Optimizer):
    """AdamW that zeroes update components disagreeing in sign with the gradient."""

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps = group["lr"], group["eps"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(param)
                    state["variance"] = torch.zeros_like(param)
                state["step"] += 1
                step = state["step"]

                momentum, variance = state["momentum"], state["variance"]
                momentum.lerp_(grad, 1.0 - beta1)
                variance.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias1 = 1.0 - beta1**step
                bias2 = 1.0 - beta2**step
                update = (momentum / bias1) / (variance / bias2).sqrt_().add_(eps)

                # Keep only components that agree with the current gradient, then
                # rescale so the masked update keeps the unmasked mean magnitude.
                mask = (update * grad > 0).to(update.dtype)
                mask = mask * (mask.numel() / mask.sum().clamp_min(1.0))

                if group["weight_decay"]:
                    param.mul_(1.0 - lr * group["weight_decay"])
                param.add_(update * mask, alpha=-lr)
        return loss
