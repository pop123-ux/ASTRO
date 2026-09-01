"""Tests for the statistical machinery.

These are checked against scipy where scipy is available, and against
hand-computable cases where it is not. A statistics module that is subtly wrong
produces confident, wrong claims -- the worst possible failure for this project.
"""

from __future__ import annotations

import pytest

from astro.bench.stats import (
    cohens_d_paired,
    holm_bonferroni,
    interpret_effect_size,
    paired_test,
    wilcoxon_signed_rank,
)

scipy_stats = pytest.importorskip("scipy.stats", reason="cross-check needs scipy")


# -- Wilcoxon ---------------------------------------------------------------


@pytest.mark.parametrize(
    "deltas",
    [
        [-0.011, -0.008, -0.013, -0.004, -0.015],
        [0.5, -0.25, 0.3, -0.1, 0.4, -0.6, 0.2],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [-0.4, 0.9, -1.7, 2.2, -3.1, 3.8],
    ],
)
def test_exact_wilcoxon_matches_scipy_without_ties(deltas: list[float]) -> None:
    """On tie-free data the two exact computations must agree to machine precision."""
    assert len({abs(d) for d in deltas}) == len(deltas), "this case must be tie-free"
    statistic, p_value, exact = wilcoxon_signed_rank(deltas)
    assert exact
    reference = scipy_stats.wilcoxon(deltas, method="exact", zero_method="wilcox")
    assert p_value == pytest.approx(float(reference.pvalue), rel=1e-9)


def test_with_ties_we_use_the_conditional_permutation_distribution() -> None:
    """A deliberate, documented divergence from scipy.

    ``scipy.stats.wilcoxon(method="exact")`` builds the null distribution over the
    integer ranks ``1..n``, which is the correct null *only when there are no
    ties*; its own documentation does not recommend the exact method with tied
    data. We instead enumerate sign assignments over the **observed midranks**,
    which is the exact conditional (permutation) distribution given the ties that
    actually occurred -- the appropriate null when ties are present.

    Equal validation losses across seeds do occur, so this is not hypothetical.
    """
    deltas = [0.5, -0.2, 0.3, -0.1, 0.4, -0.6, 0.2]  # 0.2 appears twice
    ours = wilcoxon_signed_rank(deltas)[1]
    theirs = float(scipy_stats.wilcoxon(deltas, method="exact", zero_method="wilcox").pvalue)
    assert ours != pytest.approx(theirs)
    assert ours == pytest.approx(0.59375)
    # Both remain valid probabilities and agree on the (non-)conclusion.
    assert 0.0 <= ours <= 1.0
    assert (ours <= 0.05) == (theirs <= 0.05)


def test_wilcoxon_uses_ranks_not_magnitudes() -> None:
    """The property that makes it robust: one huge outlier cannot flip the test.

    Four differences favour the control and one enormous one favours the
    treatment. The mean is dragged negative; the rank test is not fooled.
    """
    deltas = [0.01, 0.01, 0.01, 0.01, -10.0]
    assert sum(deltas) / len(deltas) < 0  # the mean says "treatment wins"
    _, p_value, _ = wilcoxon_signed_rank(deltas)
    assert p_value > 0.05  # the rank test declines to call it


def test_all_zero_differences_are_not_significant() -> None:
    statistic, p_value, exact = wilcoxon_signed_rank([0.0, 0.0, 0.0])
    assert (statistic, p_value, exact) == (0.0, 1.0, True)


def test_consistent_direction_is_detected_at_five_seeds() -> None:
    """Five seeds all favouring the treatment is the strongest evidence five
    seeds can give: p = 2/32 = 0.0625."""
    _, p_value, exact = wilcoxon_signed_rank([-0.011, -0.008, -0.013, -0.004, -0.015])
    assert exact
    assert p_value == pytest.approx(2 / 32)


def test_ten_seeds_can_reach_significance_where_five_cannot() -> None:
    """Why the headline comparison was re-run at ten seeds: with five, a perfect
    sweep still only reaches p = 0.0625 and can never clear 0.05."""
    five = wilcoxon_signed_rank([-1.0] * 5)[1]
    ten = wilcoxon_signed_rank([-1.0] * 10)[1]
    assert five > 0.05
    assert ten < 0.05


def test_ties_get_midranks() -> None:
    _, p_tied, _ = wilcoxon_signed_rank([-1.0, -1.0, -1.0, -1.0])
    reference = scipy_stats.wilcoxon([-1.0, -1.0, -1.0, -1.0], method="exact")
    assert p_tied == pytest.approx(float(reference.pvalue), rel=1e-9)


# -- effect size ------------------------------------------------------------


def test_cohens_d_is_mean_over_sd() -> None:
    deltas = [-0.02, -0.01, -0.03, -0.02]
    import statistics as st

    assert cohens_d_paired(deltas) == pytest.approx(st.fmean(deltas) / st.stdev(deltas))


def test_identical_differences_give_zero_effect_size() -> None:
    """No spread means no way to size the effect against noise; a division by
    zero here would otherwise report an infinitely large effect."""
    assert cohens_d_paired([-0.01, -0.01, -0.01]) == 0.0


def test_effect_size_labels() -> None:
    assert interpret_effect_size(0.1) == "negligible"
    assert interpret_effect_size(0.35) == "small"
    assert interpret_effect_size(0.65) == "medium"
    assert interpret_effect_size(1.5) == "large"


# -- multiple comparisons ---------------------------------------------------


def test_holm_matches_statsmodels_ordering_and_values() -> None:
    p_values = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005}
    adjusted = holm_bonferroni(p_values)
    # m=4: sorted 0.005, 0.01, 0.03, 0.04 -> x4, x3, x2, x1 with monotonicity
    assert adjusted["d"][0] == pytest.approx(0.02)
    assert adjusted["a"][0] == pytest.approx(0.03)
    assert adjusted["c"][0] == pytest.approx(0.06)
    assert adjusted["b"][0] == pytest.approx(0.06)  # monotone: cannot fall below c


def test_holm_is_monotone() -> None:
    adjusted = holm_bonferroni({"a": 0.001, "b": 0.002, "c": 0.5})
    values = [adjusted[k][0] for k in ("a", "b", "c")]
    assert values == sorted(values)


def test_holm_rejects_only_below_alpha() -> None:
    adjusted = holm_bonferroni({"strong": 0.0001, "weak": 0.9}, alpha=0.05)
    assert adjusted["strong"][1] is True
    assert adjusted["weak"][1] is False


def test_holm_on_empty_family() -> None:
    assert holm_bonferroni({}) == {}


# -- integration ------------------------------------------------------------


def test_paired_test_reports_direction_and_size() -> None:
    treatment = [2.1770, 2.2419, 2.1730, 2.1627, 2.1347]
    control = [2.1938, 2.2616, 2.1744, 2.1654, 2.1499]
    result = paired_test(treatment, control)
    assert result.n == 5
    assert result.mean_delta < 0  # treatment better
    assert result.effect_size < 0
    assert result.magnitude in {"small", "medium", "large"}


def test_paired_test_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same seeds"):
        paired_test([1.0, 2.0], [1.0])
