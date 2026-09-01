"""The curvature probe measures what it says it measures.

The probe is the load-bearing part of the curvature figure: it claims to
decompose a loss decrease into a first-order gain and a Hessian penalty. On a
quadratic the decomposition is not an approximation but an identity, so that is
where it is checked -- if the probe is wrong, the identity fails exactly.

It also has to leave training untouched. A measurement that perturbs the
trajectory it is measuring produces numbers for a run nobody did.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("transformers")

import measure_curvature as mc  # noqa: E402


class Quadratic(torch.nn.Module):
    """loss = 1/2 x^T A x - b^T x, whose Hessian is exactly A."""

    def __init__(self, matrix: torch.Tensor, offset: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(matrix.size(0)))
        self.register_buffer("matrix", matrix)
        self.register_buffer("offset", offset)

    def forward(self, _ignored=None, labels=None):
        value = 0.5 * self.weight @ self.matrix @ self.weight - self.offset @ self.weight
        return type("Output", (), {"loss": value})()


def _spd(size: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    root = torch.randn(size, size, generator=generator, dtype=torch.float64)
    return root @ root.T / size + torch.eye(size, dtype=torch.float64)


def test_hessian_vector_product_is_the_hessian() -> None:
    matrix = _spd(12)
    model = Quadratic(matrix, torch.zeros(12, dtype=torch.float64)).double()
    params = [model.weight]
    vector = [torch.randn(12, generator=torch.Generator().manual_seed(3),
                          dtype=torch.float64)]

    product, _, _ = mc.hessian_vector_product(model, None, vector, params)
    assert torch.allclose(product[0], matrix @ vector[0], atol=1e-10)


def test_the_decomposition_is_exact_on_a_quadratic() -> None:
    """On a quadratic, realized decrease == first order - curvature penalty."""
    size = 12
    matrix = _spd(size)
    offset = torch.randn(size, generator=torch.Generator().manual_seed(1),
                         dtype=torch.float64)
    model = Quadratic(matrix, offset).double()
    params = [model.weight]
    optimizer = torch.optim.SGD(params, lr=0.05)

    def step_fn(_batch, _step):
        optimizer.zero_grad(set_to_none=True)
        loss = model().loss
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    record, update = mc.probe(model, optimizer, None, None, params, step_fn, 0)

    assert record["realized"] == pytest.approx(record["predicted"], abs=1e-9)
    assert record["update_norm_sq"] == pytest.approx(float(update[0] @ update[0]))
    assert record["nds"] == pytest.approx(
        float(update[0] @ matrix @ update[0]) / float(update[0] @ update[0]))


def test_nds_is_the_rayleigh_quotient_so_it_ignores_step_size() -> None:
    """NDS is scale-free by construction; that is why it separates direction
    from step size, which is the whole point of reporting it."""
    matrix = _spd(10)
    model = Quadratic(matrix, torch.zeros(10, dtype=torch.float64)).double()
    params = [model.weight]
    direction = torch.randn(10, generator=torch.Generator().manual_seed(7),
                            dtype=torch.float64)

    values = []
    for scale in (0.01, 1.0, 100.0):
        vector = [direction * scale]
        product, _, _ = mc.hessian_vector_product(model, None, vector, params)
        values.append(float(vector[0] @ product[0]) / float(vector[0] @ vector[0]))
    assert values[0] == pytest.approx(values[1])
    assert values[1] == pytest.approx(values[2])


def test_the_probe_leaves_the_parameters_where_the_step_put_them() -> None:
    """The probe rewinds to take curvature at W_before; if it fails to restore
    W_after, every subsequent step trains a different model than we report."""
    size = 8
    model = Quadratic(_spd(size), torch.randn(
        size, generator=torch.Generator().manual_seed(2), dtype=torch.float64)).double()
    params = [model.weight]
    optimizer = torch.optim.SGD(params, lr=0.1)

    def step_fn(_batch, _step):
        optimizer.zero_grad(set_to_none=True)
        loss = model().loss
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    start = model.weight.detach().clone()
    with torch.no_grad():  # what one plain step does, for reference
        expected = start - 0.1 * (model.matrix @ start - model.offset)

    mc.probe(model, optimizer, None, None, params, step_fn, 0)
    assert torch.allclose(model.weight.detach(), expected, atol=1e-12)


def test_probing_does_not_change_the_trajectory() -> None:
    """Ten probed steps must land exactly where ten plain steps land."""
    size = 8
    matrix, offset = _spd(size), torch.randn(
        size, generator=torch.Generator().manual_seed(5), dtype=torch.float64)

    def trajectory(with_probe: bool) -> torch.Tensor:
        model = Quadratic(matrix, offset).double()
        params = [model.weight]
        optimizer = torch.optim.SGD(params, lr=0.05, momentum=0.9)

        def step_fn(_batch, _step):
            optimizer.zero_grad(set_to_none=True)
            loss = model().loss
            loss.backward()
            optimizer.step()
            return float(loss.detach())

        for step in range(10):
            if with_probe:
                mc.probe(model, optimizer, None, None, params, step_fn, step)
            else:
                step_fn(None, step)
        return model.weight.detach().clone()

    assert torch.allclose(trajectory(True), trajectory(False), atol=1e-12)
