"""Tests for the baseline optimizers and the benchmark protocol.

The baselines carry as much weight as the proposed method: a benchmark whose
baselines are subtly broken proves nothing. The protocol tests cover the
fairness machinery -- equal tuning budgets, paired statistics -- because that is
what the headline claims rest on.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from astro import AdEMAMix, CautiousAdamW, Hyperball, Muon, NorMuon
from astro.bench.protocol import (
    EvaluationSummary,
    SearchSpace,
    TaskResult,
    bootstrap_ci,
    paired_comparison,
    tune,
)


def _model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.Conv2d(16, 32, 1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 24),
        nn.Linear(24, 5),
    )


def _train(factory, steps: int = 25) -> list[float]:
    model = _model()
    optimizer = factory(model)
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randn(12, 3, 12, 12, generator=generator)
    targets = torch.randint(0, 5, (12,), generator=generator)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(inputs), targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


@pytest.mark.parametrize(
    "factory",
    [
        lambda m: Muon.from_model(m, lr=0.02, adamw_lr=3e-3),
        lambda m: NorMuon.from_model(m, lr=0.02, adamw_lr=3e-3),
        lambda m: AdEMAMix(m.parameters(), lr=3e-3),
        lambda m: CautiousAdamW(m.parameters(), lr=3e-3),
    ],
    ids=["muon", "normuon", "ademamix", "cautious"],
)
def test_baseline_reduces_the_loss(factory) -> None:
    losses = _train(factory)
    assert losses[-1] < losses[0]
    assert all(math.isfinite(v) for v in losses)


def test_normuon_differs_from_muon() -> None:
    """If the neuron normalisation were a no-op the ablation would be vacuous."""
    muon = _train(lambda m: Muon.from_model(m, lr=0.02, adamw_lr=3e-3))
    normuon = _train(lambda m: NorMuon.from_model(m, lr=0.02, adamw_lr=3e-3))
    assert muon != normuon


def test_muon_leaves_non_matrix_parameters_to_adamw() -> None:
    """Orthogonalising a depthwise kernel is the failure NVIDIA reported.

    Three layers, not two: the last weight matrix is treated as the output layer
    and routed to AdamW, so a two-layer stack would leave nothing on the spectral
    path and the assertion would pass vacuously.
    """
    model = nn.Sequential(
        nn.Conv2d(8, 8, 3, groups=8), nn.Conv2d(8, 16, 1), nn.Conv2d(16, 4, 1)
    )
    optimizer = Muon.from_model(model, lr=0.02)
    spectral = {id(p) for g in optimizer.param_groups if g["spectral"] for p in g["params"]}
    assert id(model[0].weight) not in spectral, "depthwise kernel must not be orthogonalised"
    assert id(model[2].weight) not in spectral, "output layer must not be orthogonalised"
    assert id(model[1].weight) in spectral


def test_hyperball_holds_the_norm_constant() -> None:
    torch.manual_seed(0)
    model = nn.Linear(32, 32, bias=False)
    initial = float(model.weight.norm())
    base = torch.optim.SGD(model.parameters(), lr=1.0)
    optimizer = Hyperball(base, model.parameters(), lr=0.05)

    generator = torch.Generator().manual_seed(0)
    inputs = torch.randn(16, 32, generator=generator)
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        model(inputs).pow(2).mean().backward()
        optimizer.step()
        assert float(model.weight.norm()) == pytest.approx(initial, rel=1e-4)


def test_ademamix_beta3_ramp_is_monotone_and_reaches_its_target() -> None:
    values = [AdEMAMix._beta3_ramp(0.9, 0.9999, 100, s) for s in range(0, 101, 10)]
    assert values == sorted(values)
    assert values[0] >= 0.9
    assert values[-1] == pytest.approx(0.9999)


def test_cautious_masks_disagreeing_updates() -> None:
    """A gradient that reverses sign must not be followed on the reversal step."""
    parameter = nn.Parameter(torch.zeros(4))
    optimizer = CautiousAdamW([parameter], lr=0.1)
    for _ in range(5):
        parameter.grad = torch.ones(4)
        optimizer.step()
    descended = parameter.detach().clone()

    parameter.grad = -torch.ones(4)
    optimizer.step()
    # Momentum still points the old way, so the cautious mask suppresses it.
    assert torch.all(parameter.detach() >= descended - 1e-6)


# -- protocol ----------------------------------------------------------------


def _toy_task(factory, seed: int) -> TaskResult:
    torch.manual_seed(seed)
    model = nn.Linear(8, 4, bias=False)
    optimizer = factory(model)
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(32, 8, generator=generator)
    targets = torch.randn(32, 4, generator=generator)
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        F.mse_loss(model(inputs), targets).backward()
        optimizer.step()
    with torch.no_grad():
        return TaskResult(final=float(F.mse_loss(model(inputs), targets)), steps=20)


def _space(name: str, count: int) -> SearchSpace:
    keys = ["lr", "weight_decay", "beta2"][:count]
    ranges = {"lr": (1e-3, 1e-1), "weight_decay": (1e-5, 1e-1), "beta2": (0.9, 0.999)}
    return SearchSpace(
        name=name,
        build=lambda m, c: torch.optim.AdamW(m.parameters(), lr=c["lr"]),
        ranges={k: ranges[k] for k in keys},
    )


def test_unequal_tuning_budgets_are_rejected() -> None:
    """The central fairness guarantee: three knobs versus one is not a comparison."""
    with pytest.raises(ValueError, match="same number of hyperparameters"):
        tune(_toy_task, [_space("a", 3), _space("b", 1)], trials=2)


def test_equal_budgets_are_accepted_and_traces_are_monotone() -> None:
    records = tune(_toy_task, [_space("a", 2), _space("b", 2)], trials=4)
    assert set(records) == {"a", "b"}
    for record in records.values():
        assert record.trace == sorted(record.trace, reverse=True)
        assert record.best_value == record.trace[-1]
        assert record.trials == 4


def test_a_space_must_tune_something() -> None:
    with pytest.raises(ValueError, match="at least one"):
        SearchSpace(name="x", build=lambda m, c: None, ranges={})


def test_diverged_trials_are_recorded_not_raised() -> None:
    """A hyperparameter draw that explodes is data, not a crash."""

    def exploding(factory, seed: int) -> TaskResult:
        raise RuntimeError("diverged")

    records = tune(exploding, [_space("a", 2)], trials=3)
    assert records["a"].best_value == math.inf


def test_bootstrap_interval_brackets_the_mean() -> None:
    values = [1.0, 1.1, 0.9, 1.05, 0.95]
    low, high = bootstrap_ci(values)
    assert low <= sum(values) / len(values) <= high
    assert bootstrap_ci([2.0]) == (2.0, 2.0)


def test_paired_comparison_uses_within_seed_differences() -> None:
    """A consistent per-seed win must register even when the spread is large."""
    treatment = EvaluationSummary("t", [10.0, 20.0, 30.0], [1, 1, 1], [None] * 3)
    control = EvaluationSummary("c", [11.0, 21.0, 31.0], [1, 1, 1], [None] * 3)
    result = paired_comparison(treatment, control)
    assert result["mean_delta"] == pytest.approx(-1.0)
    assert result["win_rate"] == 1.0
    assert result["significant"] == 1.0


def test_paired_comparison_requires_matching_seeds() -> None:
    with pytest.raises(ValueError, match="same seeds"):
        paired_comparison(
            EvaluationSummary("t", [1.0, 2.0], [1, 1], [None, None]),
            EvaluationSummary("c", [1.0], [1], [None]),
        )
