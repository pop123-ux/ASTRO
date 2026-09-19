from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import torch


def newton_schulz(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Muon's quintic polar iteration in float32 for reproducibility on T4."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix.float()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.T if transposed else x


def inverse_metric_power(
    metric: torch.Tensor,
    *,
    power: float,
    eps: float,
    condition_cap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """M^{-power/2} for batches of 2x2 SPD matrices."""
    values, vectors = torch.linalg.eigh(metric.float())
    values = values.clamp_min(eps)
    largest = values[..., -1:]
    floor = largest / max(condition_cap, 1.0)
    values = torch.maximum(values, floor)
    factors = values.pow(-0.5 * power)
    transform = (vectors * factors.unsqueeze(-2)) @ vectors.transpose(-1, -2)
    condition = values[..., -1] / values[..., 0].clamp_min(eps)
    return transform, condition


class Orbit(torch.optim.Optimizer):
    """ORBIT-v0: RoPE-aware functional Q/K preconditioning on top of Muon.

    Non-Q/K hidden matrices use a standard Muon matrix update. Embeddings, the
    tied LM head, biases and normalization vectors use AdamW-style updates. Q/K
    matrices first receive the same Muon candidate direction, then each 2-row
    RoPE frequency pair is left-preconditioned by an inverse power of the local
    2x2 attention-logit sensitivity metric accumulated by ``RotaryAttention``.
    A joint Frobenius restoration keeps the total Q+K update budget unchanged,
    so the mechanism changes functional geometry rather than silently obtaining
    a larger step.

    Variants are intentionally first-class for falsification:
      orbit          full RoPE-rotated 2x2 metric
      orbit_norope   same covariance metric without relative-position rotations
      orbit_diag     remove the 2x2 phase/off-diagonal coupling
      orbit_identity disable the functional preconditioner (Muon control)
    """

    VALID_VARIANTS = {"orbit", "orbit_norope", "orbit_diag", "orbit_identity"}

    def __init__(
        self,
        model,
        *,
        lr: float = 0.02,
        adamw_lr: float = 3e-4,
        weight_decay: float = 0.05,
        momentum: float = 0.95,
        ns_steps: int = 5,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        functional_power: float = 1.0,
        metric_eps: float = 1e-5,
        metric_condition_cap: float = 100.0,
        deltas: Iterable[int] = (1, 2, 4, 8, 16, 32, 64, 128),
        variant: str = "orbit",
    ) -> None:
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"unknown ORBIT variant {variant!r}")
        params = list(model.parameters())
        super().__init__(
            params,
            dict(
                lr=lr,
                adamw_lr=adamw_lr,
                weight_decay=weight_decay,
                momentum=momentum,
                ns_steps=ns_steps,
                betas=adamw_betas,
                eps=eps,
            ),
        )
        self.model = model
        self.functional_power = float(functional_power)
        self.metric_eps = float(metric_eps)
        self.metric_condition_cap = float(metric_condition_cap)
        self.deltas = tuple(int(d) for d in deltas)
        self.variant = variant
        self._meta: dict[int, str] = {}
        self._pairs = []
        self._diagnostic_sums: defaultdict[str, float] = defaultdict(float)
        self._diagnostic_count = 0

        excluded = {id(model.wte.weight), id(model.lm_head.weight)}
        for pair in model.orbit_qk_pairs():
            self._meta[id(pair["q"])] = "q"
            self._meta[id(pair["k"])] = "k"
            self._pairs.append(pair)
        for param in params:
            if id(param) in self._meta:
                continue
            if param.ndim < 2:
                self._meta[id(param)] = "aux_nodecay"
            elif id(param) in excluded:
                self._meta[id(param)] = "aux_decay"
            else:
                self._meta[id(param)] = "spectral"

    def _spectral_candidate(self, param: torch.Tensor, group: dict) -> torch.Tensor:
        grad = param.grad
        state = self.state[param]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(grad)
        momentum = state["momentum_buffer"].mul_(group["momentum"]).add_(grad)
        lookahead = grad.add(momentum, alpha=group["momentum"])
        update = newton_schulz(lookahead, group["ns_steps"])
        update.mul_(max(1.0, update.size(0) / update.size(1)) ** 0.5)
        return update.to(param.dtype)

    @torch.no_grad()
    def _precondition_pair(self, pair, q_update: torch.Tensor, k_update: torch.Tensor):
        if self.variant == "orbit_identity" or self.functional_power == 0.0:
            return q_update, k_update
        rotate = self.variant != "orbit_norope"
        mq, mk = pair["module"].orbit_metrics(
            self.deltas, rotate=rotate, eps=self.metric_eps
        )
        mq, mk = mq.to(q_update.device), mk.to(k_update.device)
        if self.variant == "orbit_diag":
            mq = torch.diag_embed(torch.diagonal(mq, dim1=-2, dim2=-1))
            mk = torch.diag_embed(torch.diagonal(mk, dim1=-2, dim2=-1))
        tq, cq = inverse_metric_power(
            mq,
            power=self.functional_power,
            eps=self.metric_eps,
            condition_cap=self.metric_condition_cap,
        )
        tk, ck = inverse_metric_power(
            mk,
            power=self.functional_power,
            eps=self.metric_eps,
            condition_cap=self.metric_condition_cap,
        )
        h = pair["module"].n_head
        f = pair["module"].n_freq
        qv = q_update.float().view(h, f, 2, -1)
        kv = k_update.float().view(h, f, 2, -1)
        before = torch.sqrt(qv.square().sum() + kv.square().sum()).clamp_min(1e-12)
        qv = torch.einsum("hfab,hfbi->hfai", tq, qv)
        kv = torch.einsum("hfab,hfbi->hfai", tk, kv)
        after = torch.sqrt(qv.square().sum() + kv.square().sum()).clamp_min(1e-12)
        restore = before / after
        qv.mul_(restore)
        kv.mul_(restore)
        self._diagnostic_sums["metric_condition_q"] += float(cq.mean().cpu())
        self._diagnostic_sums["metric_condition_k"] += float(ck.mean().cpu())
        self._diagnostic_sums["joint_restore"] += float(restore.cpu())
        self._diagnostic_count += 1
        return qv.reshape_as(q_update).to(q_update.dtype), kv.reshape_as(k_update).to(k_update.dtype)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None if closure is None else closure()
        group = self.param_groups[0]
        candidates: dict[int, torch.Tensor] = {}

        for param in group["params"]:
            if param.grad is None:
                continue
            role = self._meta[id(param)]
            if role in {"q", "k", "spectral"}:
                candidates[id(param)] = self._spectral_candidate(param, group)

        for pair in self._pairs:
            q, k = pair["q"], pair["k"]
            if id(q) not in candidates or id(k) not in candidates:
                continue
            candidates[id(q)], candidates[id(k)] = self._precondition_pair(
                pair, candidates[id(q)], candidates[id(k)]
            )

        beta1, beta2 = group["betas"]
        for param in group["params"]:
            if param.grad is None:
                continue
            role = self._meta[id(param)]
            if role in {"q", "k", "spectral"}:
                if group["weight_decay"]:
                    param.mul_(1.0 - group["lr"] * group["weight_decay"])
                param.add_(candidates[id(param)], alpha=-group["lr"])
                continue

            state = self.state[param]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)
            state["step"] += 1
            step = state["step"]
            grad = param.grad
            exp_avg = state["exp_avg"].mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq = state["exp_avg_sq"].mul_(beta2).addcmul_(
                grad, grad, value=1.0 - beta2
            )
            update = (exp_avg / (1.0 - beta1**step)) / (
                (exp_avg_sq / (1.0 - beta2**step)).sqrt().add_(group["eps"])
            )
            if role == "aux_decay" and group["weight_decay"]:
                param.mul_(1.0 - group["adamw_lr"] * group["weight_decay"])
            param.add_(update, alpha=-group["adamw_lr"])
        return loss

    def diagnostics(self) -> dict[str, float]:
        if not self._diagnostic_count:
            return {}
        return {
            key: value / self._diagnostic_count
            for key, value in sorted(self._diagnostic_sums.items())
        }
