"""Tests for the spectral filters.

The central invariant is that the matrix path and the scalar path compute the
same function. That is what lets :meth:`SpectralFilter.response` serve as the
test oracle everywhere else -- it is cheap, exact and needs no SVD.
"""

from __future__ import annotations

import pytest
import torch

from astro.polar import (
    MUON_QUINTIC,
    SpectralFilter,
    deadzone_filter,
    muon_filter,
    power_iteration,
)


def _planted(
    rows: int = 48, cols: int = 72, signal: int = 6, tail: float = 0.004, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """A matrix with a known spectrum: ``signal`` large values and a flat tail."""
    generator = torch.Generator().manual_seed(seed)
    left, _ = torch.linalg.qr(torch.randn(rows, rows, generator=generator))
    right, _ = torch.linalg.qr(torch.randn(cols, cols, generator=generator))
    values = torch.cat(
        [torch.linspace(1.0, 0.5, signal), torch.full((rows - signal,), tail)]
    )
    return left @ torch.diag(values) @ right[:, :rows].T, values


@pytest.mark.parametrize("name", ["muon", "deadzone"])
def test_matrix_path_matches_scalar_response(name: str) -> None:
    """The matrix iteration applies exactly the scalar map to each singular value."""
    spectral_filter = muon_filter(5) if name == "muon" else deadzone_filter(0.1, 10)
    matrix, _ = _planted()
    normalised = matrix / matrix.norm()

    observed = torch.linalg.svdvals(spectral_filter(matrix)).sort(descending=True).values
    predicted = (
        spectral_filter.response(torch.linalg.svdvals(normalised))
        .sort(descending=True)
        .values.to(observed.dtype)
    )
    assert torch.allclose(observed, predicted, atol=1e-4)


def test_muon_lifts_the_entire_spectrum_including_noise() -> None:
    """Muon's filter drives even a 0.004 noise tail up towards one.

    This is by design -- its quintic has slope 3.44 at the origin -- and it is
    the behaviour the dead-zone filter exists to change. Asserting it here keeps
    the comparison in ``docs/paper/paper.md`` honest if the reference
    coefficients are ever changed.
    """
    matrix, _ = _planted()
    values = torch.linalg.svdvals(muon_filter(5)(matrix))
    assert values[:6].mean() > 0.9
    assert values[6:].mean() > 0.5


def test_deadzone_suppresses_the_tail_and_preserves_the_head() -> None:
    matrix, _ = _planted()
    values = torch.linalg.svdvals(deadzone_filter(0.1, 10)(matrix))
    assert values[:6].mean() == pytest.approx(1.0, abs=1e-3)
    assert values[6:].max() < 1e-3


def test_deadzone_response_is_a_step_and_stays_bounded() -> None:
    """Pass band maps to 1, stop band to 0, and no iterate blows up on [0, 1]."""
    spectral_filter = deadzone_filter(0.1, 10)
    grid = torch.linspace(0.0, 1.0, 501, dtype=torch.float64)

    response = spectral_filter.response(grid)
    assert response.abs().max() < 1.05

    assert spectral_filter.response(torch.tensor([0.2, 0.5, 1.0])).sub(1.0).abs().max() < 1e-3
    assert spectral_filter.response(torch.tensor([1e-5, 1e-3, 0.03])).abs().max() < 1e-3


def test_unknown_deadzone_configuration_is_reported_not_guessed() -> None:
    with pytest.raises(KeyError, match="no cached dead-zone filter"):
        deadzone_filter(0.42, 3)


def test_filter_handles_zero_and_batched_input() -> None:
    spectral_filter = muon_filter(3)
    assert torch.isfinite(spectral_filter(torch.zeros(8, 12))).all()

    batched = spectral_filter(torch.randn(4, 8, 12))
    assert batched.shape == (4, 8, 12)


def test_filter_rejects_vectors() -> None:
    with pytest.raises(ValueError, match="expected a matrix"):
        muon_filter(2)(torch.randn(10))


def test_transpose_invariance() -> None:
    """The filter must not depend on which side the Gram matrix is taken."""
    matrix = torch.randn(16, 40, generator=torch.Generator().manual_seed(3))
    spectral_filter = muon_filter(5)
    assert torch.allclose(
        spectral_filter(matrix), spectral_filter(matrix.T).T, atol=1e-5
    )


def test_power_iteration_estimates_the_top_singular_value() -> None:
    matrix, _ = _planted()
    normalised = matrix / matrix.norm()
    true = float(torch.linalg.svdvals(normalised)[0])
    estimate = float(power_iteration(normalised, iters=12))
    # Power iteration converges from below, which keeps a relative threshold
    # built on it conservative rather than over-aggressive.
    assert estimate <= true + 1e-5
    assert estimate == pytest.approx(true, rel=0.02)


def test_muon_quintic_is_the_published_triple() -> None:
    assert MUON_QUINTIC == (3.4445, -4.7750, 2.0315)
    assert muon_filter(5).steps == tuple([MUON_QUINTIC] * 5)


def test_custom_filter_composition_is_ordered() -> None:
    """Steps apply in listed order, so a composition is not silently commutative."""
    first = SpectralFilter(steps=((2.0, 0.0, 0.0), (1.0, -1.0, 0.0)), name="a")
    second = SpectralFilter(steps=((1.0, -1.0, 0.0), (2.0, 0.0, 0.0)), name="b")
    probe = torch.tensor([0.3], dtype=torch.float64)
    assert not torch.allclose(first.response(probe), second.response(probe))


# ---------------------------------------------------------------------------
# A converging polar iteration
# ---------------------------------------------------------------------------


def test_muon_quintic_has_fixed_points_away_from_one() -> None:
    """The reason repeating one polynomial cannot reach the polar factor.

    p(s) = a s + b s^3 + c s^5 fixes s when 2.4445 - 4.7750 s^2 + 2.0315 s^4 = 0,
    which has roots at 0.868 and 1.264 -- and those bracket the band the
    singular values are measured in.
    """
    import numpy as np

    from astro.polar import MUON_QUINTIC

    a, b, c = MUON_QUINTIC
    roots = np.roots([c, 0, b, 0, a - 1, 0])
    positive = sorted(float(r.real) for r in roots if abs(r.imag) < 1e-9 and r.real > 1e-9)
    assert positive == pytest.approx([0.8680, 1.2637], abs=1e-3)


def test_converging_filter_reaches_one_where_muon_cannot() -> None:
    from astro.polar import muon_filter, polar_filter

    torch.manual_seed(0)
    matrix = torch.randn(128, 64)

    muon = torch.linalg.svdvals(muon_filter(5)(matrix))
    assert float(muon.min()) < 0.75, "Muon's quintic should leave the tail short of 1"

    for steps, tolerance in ((6, 2e-3), (7, 1e-4)):
        values = torch.linalg.svdvals(polar_filter(steps, 1e-3)(matrix))
        assert float(values.min()) == pytest.approx(1.0, abs=tolerance)
        assert float(values.max()) == pytest.approx(1.0, abs=tolerance)


def test_converging_filter_removes_the_conditioning_drift() -> None:
    """Muon's update norm depends on the input's condition number; this is the
    quantity that makes its effective step size drift during training."""
    import math

    from astro.polar import muon_filter, polar_filter

    torch.manual_seed(0)
    rows, cols = 128, 64
    target = math.sqrt(min(rows, cols))

    def ratios(spectral_filter) -> list[float]:
        out = []
        for condition in (1.0, 10.0, 100.0, 1000.0):
            left, _ = torch.linalg.qr(torch.randn(rows, cols))
            right, _ = torch.linalg.qr(torch.randn(cols, cols))
            spectrum = torch.logspace(0, -math.log10(condition), cols)
            out.append(float(spectral_filter((left * spectrum) @ right.T).norm()) / target)
        return out

    muon = ratios(muon_filter(5))
    polar = ratios(polar_filter(7, 1e-3))
    assert max(muon) - min(muon) > 0.15
    assert max(polar) - min(polar) < 0.05
    assert all(value == pytest.approx(1.0, abs=0.05) for value in polar)


def test_solved_schedules_are_nested_prefixes() -> None:
    """Greedy solving means a shorter schedule is a prefix of a longer one."""
    from astro.polar import _POLAR_CACHE

    for steps in range(1, 7):
        assert _POLAR_CACHE[(steps, 1e-3)] == _POLAR_CACHE[(steps + 1, 1e-3)][:steps]


def test_late_solved_coefficients_match_the_published_asymptote() -> None:
    """An independent check that the solve found the right thing: the tail of
    the schedule approaches the (1.875, -1.25, 0.375) the Polar Express uses."""
    from astro.polar import _POLAR_CACHE

    last = _POLAR_CACHE[(7, 1e-3)][-1]
    assert last == pytest.approx((1.875, -1.25, 0.375), abs=0.2)


def test_unsolved_schedule_is_refused_rather_than_guessed() -> None:
    from astro.polar import polar_filter

    with pytest.raises(KeyError, match="no solved coefficients"):
        polar_filter(5, 1e-6)
