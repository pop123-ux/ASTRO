"""SOAP: ShampoO with Adam in the Preconditioner's eigenbasis.

Vyas et al., arXiv:2409.11321. Implemented here because a comparison that omits
it is not a comparison against the state of the art: NVIDIA's 2026 scaling
report (arXiv:2607.20548) finds KL-SOAP edges out Muon at multi-billion scale
and recommends it over Muon whenever the memory can be afforded.

The idea in one line
--------------------
Shampoo preconditions a matrix gradient with Kronecker factors ``L^{-1/4} G
R^{-1/4}``. SOAP instead uses those factors only to find a *basis* -- the
eigenvectors ``Q_L``, ``Q_R`` of the row and column covariances -- and then runs
ordinary Adam on the gradient expressed in that basis::

    G'     = Q_L^T G Q_R          rotate in
    N      = Adam(G')             elementwise, in the rotated space
    dW     = Q_L N Q_R^T          rotate out

When ``Q_L`` and ``Q_R`` are identities this is exactly AdamW, which is why SOAP
transfers hyperparameters from Adam so readily. When the second moments are
switched off it reduces to Shampoo's whitening. That places it precisely on the
axis this repository cares about: Frans et al. decompose matrix-whitening into
*spectral normalisation* plus *variance adaptation*, and SOAP is the method that
implements both, which is the standing explanation for why it beats Muon
per step.

Cost
----
Two eigendecompositions per matrix, amortised by refreshing the basis only every
``precondition_frequency`` steps, plus four matmuls per step to rotate in and
out. Memory is the expensive part: ``L``, ``R``, ``Q_L``, ``Q_R`` are
``in x in`` and ``out x out``, so a wide layer costs far more state than Muon's
single momentum buffer. That trade -- more memory and compute per step for fewer
steps -- is exactly what the benchmark's wall-clock column exists to adjudicate.

Deviation from the reference, stated plainly
--------------------------------------------
This is SOAP as published. It is **not** NVIDIA's KL-SOAP, which replaces the
covariance accumulation with a KL-regularised estimator and forces a per-step QR
with the current gradient included. Those changes fix a "slingshot" instability
that appears at very large batch sizes; at the batch sizes reachable on a CPU
they address a failure mode that does not arise, and implementing a variant
imprecisely is worse than implementing the canonical algorithm correctly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from astro.routing import classify_module, matrix_view

__all__ = ["SOAP"]


class SOAP(Optimizer):
    """Adam in the eigenbasis of Shampoo's Kronecker factors.

    Parameters
    ----------
    params:
        Parameter groups; each may set ``spectral`` to override routing. Tensors
        not on the matrix path get plain AdamW, exactly as for Muon.
    lr:
        Learning rate. SOAP inherits Adam's scale because an identity eigenbasis
        reduces it to Adam, so the same range that suits AdamW suits it.
    betas:
        ``(beta1, beta2)`` for the Adam moments computed in the rotated basis.
    shampoo_beta:
        EMA coefficient for the Kronecker factors ``L`` and ``R``. Slower than
        ``beta2`` on purpose: the basis should reflect persistent curvature
        structure rather than the current minibatch.
    precondition_frequency:
        Steps between eigenbasis refreshes. The reference implementation uses 10;
        NVIDIA found that staleness is what triggers instability at large batch,
        and that per-step refresh fixes it.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 3e-3,
        betas: tuple[float, float] = (0.95, 0.95),
        shampoo_beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        *,
        precondition_frequency: int = 10,
        max_precondition_dim: int = 4096,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"betas must lie in [0, 1), got {betas}")
        defaults = dict(
            lr=lr,
            betas=betas,
            shampoo_beta=shampoo_beta,
            eps=eps,
            weight_decay=weight_decay,
            spectral=True,
        )
        super().__init__(params, defaults)
        self.precondition_frequency = int(precondition_frequency)
        self.max_precondition_dim = int(max_precondition_dim)

    @classmethod
    def from_model(cls, model: nn.Module, lr: float = 3e-3, **kwargs: Any) -> SOAP:
        """Route parameters with :mod:`astro.routing`, as Muon and ASTRO do.

        Identical routing across every matrix method is deliberate: it isolates
        the update rule from the routing policy, so a win cannot be attributed to
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

    # -- preconditioner ------------------------------------------------------

    def _init_state(self, param: torch.Tensor, state: dict[str, Any], spectral: bool) -> None:
        state["step"] = 0
        state["momentum"] = torch.zeros_like(param)
        state["variance"] = torch.zeros_like(param)
        if not spectral:
            return
        view = matrix_view(param)
        rows, cols = view.shape
        # A dimension too large to eigendecompose cheaply falls back to identity
        # on that side, which degrades SOAP to one-sided preconditioning rather
        # than making it unaffordable.
        state["left"] = (
            torch.zeros(rows, rows, dtype=param.dtype, device=param.device)
            if rows <= self.max_precondition_dim
            else None
        )
        state["right"] = (
            torch.zeros(cols, cols, dtype=param.dtype, device=param.device)
            if cols <= self.max_precondition_dim
            else None
        )
        state["q_left"] = (
            torch.eye(rows, dtype=param.dtype, device=param.device)
            if state["left"] is not None
            else None
        )
        state["q_right"] = (
            torch.eye(cols, dtype=param.dtype, device=param.device)
            if state["right"] is not None
            else None
        )

    @staticmethod
    def _eigenbasis(factor: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        """Eigenvectors of a symmetric factor, ordered by descending eigenvalue.

        Falls back to the previous basis if the decomposition fails, which it can
        under fp32 on a near-singular factor early in training. Losing a refresh
        is a far smaller error than propagating NaNs into every subsequent step.
        """
        try:
            _, vectors = torch.linalg.eigh(factor)
        except Exception:  # noqa: BLE001 - LinAlgError is backend-dependent
            return fallback
        if not torch.isfinite(vectors).all():
            return fallback
        return vectors.flip(-1)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform one optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps = group["lr"], group["eps"]
            spectral = bool(group["spectral"])

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("SOAP does not support sparse gradients")

                state = self.state[param]
                if not state:
                    self._init_state(param, state, spectral)
                state["step"] += 1
                step = state["step"]

                if not spectral:
                    self._adamw(param, state, lr, beta1, beta2, eps, group["weight_decay"])
                    continue

                grad = matrix_view(param.grad)
                left, right = state["left"], state["right"]
                q_left, q_right = state["q_left"], state["q_right"]

                # Kronecker factors: row and column gradient covariances.
                shampoo_beta = group["shampoo_beta"]
                if left is not None:
                    left.mul_(shampoo_beta).add_(grad @ grad.T, alpha=1.0 - shampoo_beta)
                if right is not None:
                    right.mul_(shampoo_beta).add_(grad.T @ grad, alpha=1.0 - shampoo_beta)

                # Refresh the basis on a schedule. The Adam moments live in the
                # rotated frame, so they must be carried across a basis change or
                # they would suddenly describe different directions.
                if step == 1 or step % self.precondition_frequency == 0:
                    momentum = matrix_view(state["momentum"])
                    variance = matrix_view(state["variance"])
                    old_left, old_right = q_left, q_right
                    if left is not None:
                        q_left = self._eigenbasis(left, old_left)
                    if right is not None:
                        q_right = self._eigenbasis(right, old_right)
                    if step > 1:
                        momentum.copy_(
                            self._rotate(self._unrotate(momentum, old_left, old_right),
                                         q_left, q_right)
                        )
                        # The second moment is a variance, so it rotates through
                        # the same change of basis on the *squared* quantity;
                        # rotating its square root keeps it non-negative.
                        variance.copy_(
                            self._rotate(
                                self._unrotate(variance.sqrt(), old_left, old_right),
                                q_left, q_right,
                            ).square()
                        )
                    state["q_left"], state["q_right"] = q_left, q_right

                rotated = self._rotate(grad, q_left, q_right)

                momentum = matrix_view(state["momentum"])
                variance = matrix_view(state["variance"])
                momentum.lerp_(rotated, 1.0 - beta1)
                variance.mul_(beta2).addcmul_(rotated, rotated, value=1.0 - beta2)

                bias1 = 1.0 - beta1**step
                bias2 = 1.0 - beta2**step
                normalized = (momentum / bias1) / ((variance / bias2).sqrt() + eps)

                update = self._unrotate(normalized, q_left, q_right)

                flat = param.view(update.shape) if param.ndim > 2 else param
                if group["weight_decay"]:
                    flat.mul_(1.0 - lr * group["weight_decay"])
                flat.add_(update, alpha=-lr)
        return loss

    @staticmethod
    def _rotate(
        matrix: torch.Tensor, q_left: torch.Tensor | None, q_right: torch.Tensor | None
    ) -> torch.Tensor:
        """``Q_L^T M Q_R``, skipping either side whose basis is unavailable."""
        if q_left is not None:
            matrix = q_left.T @ matrix
        if q_right is not None:
            matrix = matrix @ q_right
        return matrix

    @staticmethod
    def _unrotate(
        matrix: torch.Tensor, q_left: torch.Tensor | None, q_right: torch.Tensor | None
    ) -> torch.Tensor:
        """``Q_L M Q_R^T``: the inverse of :meth:`_rotate` for orthonormal bases."""
        if q_left is not None:
            matrix = q_left @ matrix
        if q_right is not None:
            matrix = matrix @ q_right.T
        return matrix

    @staticmethod
    def _adamw(
        param: torch.Tensor,
        state: dict[str, Any],
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
    ) -> None:
        """Plain decoupled-weight-decay Adam, for tensors off the matrix path."""
        step = state["step"]
        momentum, variance = state["momentum"], state["variance"]
        grad = param.grad
        momentum.lerp_(grad, 1.0 - beta1)
        variance.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        bias1 = 1.0 - beta1**step
        bias2 = 1.0 - beta2**step
        if weight_decay:
            param.mul_(1.0 - lr * weight_decay)
        param.addcdiv_(momentum / bias1, (variance / bias2).sqrt_().add_(eps), value=-lr)
