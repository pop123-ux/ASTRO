"""Hyperball: constrain weight norms and update norms to constants.

From Wen et al. 2026 (*Fantastic Pretraining Optimizers II*). The observation is
that in a normalised network the loss is invariant to the scale of a weight
matrix, ``L(cW) = L(W)``, so penalising ``||W||`` cannot be regularisation in the
classical sense. What decoupled weight decay actually does is set an equilibrium
radius, and *that* radius sets the angular step size

    eta_ang = || W_hat_{t+1} - W_hat_t ||_F ,   W_hat = W / ||W||_F,

which is the quantity that governs how fast the block moves in function space.
Hyperball controls it directly instead:

    W_{t+1} = R * Normalize( W_t - eta_t * R * Normalize(u_t) ),   R = ||W_0||_F.

Implemented as a wrapper so it composes with any base optimizer, which is how the
paper frames it and is what lets the benchmark run Adam-Hyperball and
Muon-Hyperball without duplicating either.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch.optim import Optimizer

__all__ = ["Hyperball"]


class Hyperball(Optimizer):
    """Wrap ``base`` so that constrained parameters keep a constant Frobenius norm.

    Parameters
    ----------
    base:
        Any optimizer. Its ``step`` produces the direction; Hyperball rescales
        the displacement and re-projects onto the sphere.
    params:
        The subset of parameters to constrain. Parameters not listed are left to
        the base optimizer untouched -- embeddings, gains and biases should not
        be constrained, because for them the norm carries meaning.
    lr:
        Angular learning rate: the step length is ``lr * R``. Because the radius
        and the update length are both fixed, this directly sets the relative
        update size, which is why the paper finds it transfers across widths and
        depths far better than a weight-decay-tuned learning rate.

    Notes
    -----
    The base optimizer's own learning rate still scales its update, but Hyperball
    normalises the resulting displacement, so only the *direction* survives. Set
    the base learning rate to 1.0 to make that explicit.
    """

    def __init__(
        self,
        base: Optimizer,
        params: Iterable[torch.Tensor],
        lr: float = 0.02,
        eps: float = 1e-12,
    ) -> None:
        constrained = list(params)
        super().__init__([{"params": constrained}], dict(lr=lr))
        self.base = base
        self.eps = float(eps)
        self._radius: dict[torch.Tensor, torch.Tensor] = {}
        for param in constrained:
            self._radius[param] = param.detach().norm().clamp_min(eps).clone()

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        lr = float(group["lr"])
        before = {p: p.detach().clone() for p in group["params"] if p.grad is not None}

        self.base.step()

        for param, previous in before.items():
            radius = self._radius[param]
            direction = param.detach() - previous
            norm = direction.norm()
            if float(norm) <= self.eps:
                param.copy_(previous)
                continue
            # Unconstrained step of length lr * R along the normalised direction,
            # then radial projection back onto the sphere of radius R.
            trial = previous + direction * (lr * float(radius) / float(norm))
            param.copy_(trial * (radius / trial.norm().clamp_min(self.eps)))
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base.zero_grad(set_to_none=set_to_none)
        super().zero_grad(set_to_none=set_to_none)
