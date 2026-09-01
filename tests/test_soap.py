"""Tests for the SOAP baseline.

A baseline that is subtly wrong is worse than no baseline: it makes the proposed
method look good for a reason that has nothing to do with the proposed method.
So these check the two properties that define SOAP, not just that loss goes down.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from astro.baselines import SOAP

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(24, 32), nn.GELU(), nn.Linear(32, 16), nn.GELU(),
                         nn.Linear(16, 5))


def _train(optimizer, model, steps: int = 40) -> list[float]:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(32, 24, generator=generator)
    y = torch.randint(0, 5, (32,), generator=generator)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def test_soap_reduces_the_loss() -> None:
    model = _model()
    losses = _train(SOAP.from_model(model, lr=3e-3), model)
    assert losses[-1] < losses[0]


def test_identity_eigenbasis_reduces_soap_to_adamw() -> None:
    """The defining reduction: with Q = I, rotating in and out is a no-op and the
    update is exactly AdamW's. If this fails, SOAP is not SOAP."""
    torch.manual_seed(0)
    weight = nn.Parameter(torch.randn(12, 12))
    reference = nn.Parameter(weight.detach().clone())

    # max_precondition_dim=0 leaves both factors unbuilt, so rotate/unrotate are
    # no-ops and nothing but the Adam machinery runs. That is the reduction.
    soap = SOAP([{"params": [weight], "spectral": True}], lr=1e-2, betas=(0.9, 0.95),
                weight_decay=0.0, max_precondition_dim=0)
    adam = torch.optim.AdamW([reference], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.0)

    generator = torch.Generator().manual_seed(1)
    for _ in range(5):
        grad = torch.randn(12, 12, generator=generator)
        weight.grad, reference.grad = grad.clone(), grad.clone()
        soap.step()
        adam.step()
    assert torch.allclose(weight, reference, atol=1e-5)


def test_eigenbasis_is_orthonormal() -> None:
    """Rotate-in and rotate-out must be inverses, or the update is not a change
    of basis at all."""
    torch.manual_seed(0)
    model = nn.Linear(16, 16, bias=False)
    optimizer = SOAP.from_model(model, lr=1e-3, precondition_frequency=1)
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        model(torch.randn(8, 16)).sum().backward()
        optimizer.step()

    state = optimizer.state[model.weight]
    for key in ("q_left", "q_right"):
        basis = state[key]
        assert basis is not None
        assert torch.allclose(basis.T @ basis, torch.eye(basis.size(0)), atol=1e-4)


def test_rotate_and_unrotate_are_inverses() -> None:
    torch.manual_seed(0)
    matrix = torch.randn(8, 12)
    q_left, _ = torch.linalg.qr(torch.randn(8, 8))
    q_right, _ = torch.linalg.qr(torch.randn(12, 12))
    roundtrip = SOAP._unrotate(SOAP._rotate(matrix, q_left, q_right), q_left, q_right)
    assert torch.allclose(roundtrip, matrix, atol=1e-5)


def test_basis_refresh_carries_the_moments_across() -> None:
    """The Adam moments live in the rotated frame. If a basis change does not
    carry them, they suddenly describe different directions and the step is
    garbage for several iterations afterwards."""
    torch.manual_seed(0)
    model = nn.Linear(16, 16, bias=False)
    optimizer = SOAP.from_model(model, lr=1e-3, precondition_frequency=3)
    losses = []
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(16, 16, generator=generator)
    target = torch.randn(16, 16, generator=generator)
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        loss = ((model(x) - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    # No spike at a refresh boundary: monotone enough that the worst step-to-step
    # increase stays small relative to the starting loss.
    worst_jump = max(b - a for a, b in zip(losses, losses[1:], strict=False))
    assert worst_jump < 0.05 * losses[0]
    assert losses[-1] < losses[0]


def test_routing_keeps_embeddings_off_the_matrix_path() -> None:
    """SOAP uses the same router as Muon and ASTRO, so a win cannot come from
    routing differences."""
    torch.manual_seed(0)

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(50, 16)
            self.hidden = nn.Linear(16, 16)
            self.out = nn.Linear(16, 50)

    optimizer = SOAP.from_model(Net(), lr=1e-3)
    spectral = [g for g in optimizer.param_groups if g["spectral"]]
    assert len(spectral[0]["params"]) == 1  # only `hidden.weight`


def test_sparse_gradients_are_rejected_explicitly() -> None:
    parameter = nn.Parameter(torch.randn(10, 10))
    parameter.grad = torch.sparse_coo_tensor(
        torch.tensor([[0], [0]]), torch.tensor([1.0]), (10, 10)
    )
    with pytest.raises(RuntimeError, match="sparse"):
        SOAP([{"params": [parameter], "spectral": True}], lr=1e-3).step()


def test_eigendecomposition_failure_falls_back_to_the_old_basis() -> None:
    """A non-finite factor must not propagate NaNs into every later step."""
    fallback = torch.eye(4)
    result = SOAP._eigenbasis(torch.full((4, 4), float("nan")), fallback)
    assert torch.allclose(result, fallback)
