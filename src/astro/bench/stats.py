"""Statistical tests for optimizer comparisons.

Written in the standard library so the package keeps its single dependency, and
because the tests that matter at this sample size are small enough to do exactly.

Why these three
---------------
A bootstrap interval on the paired differences (in :mod:`astro.bench.protocol`)
answers "how uncertain is the mean difference". It does not answer two other
questions a reader should ask, and both are cheap:

**Is the difference consistent, or driven by one seed?**
    The exact Wilcoxon signed-rank test uses only the ranks of the paired
    differences, so a single large outlier cannot manufacture significance the
    way it can for a mean. At :math:`n \\le 20` the null distribution is obtained
    by enumerating all :math:`2^n` sign assignments -- no normal approximation,
    which is worth doing because the approximation is poor exactly in the range
    optimizer papers use (5 to 10 seeds).

**How big is the difference relative to run-to-run noise?**
    Cohen's :math:`d` for paired samples, :math:`\\bar{d}/s_d`. A statistically
    significant difference that is small compared to seed variance is a real but
    unimportant difference, and the two should not be reported as if they were
    the same thing.

**Are we significant because we ran many comparisons?**
    Comparing one method against six baselines gives six chances to clear
    :math:`p < 0.05`. Holm-Bonferroni controls the family-wise error rate without
    assuming independence, and is uniformly more powerful than plain Bonferroni.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

__all__ = [
    "PairedTest",
    "cohens_d_paired",
    "holm_bonferroni",
    "interpret_effect_size",
    "wilcoxon_signed_rank",
]

#: Above this sample size, enumerating 2^n sign assignments stops being cheap and
#: the normal approximation is accurate enough to use instead.
EXACT_LIMIT = 20


@dataclass(frozen=True)
class PairedTest:
    """Outcome of comparing two optimizers seed by seed."""

    n: int
    mean_delta: float
    #: Cohen's d for paired samples. Negative favours the treatment.
    effect_size: float
    #: Wilcoxon signed-rank statistic (sum of ranks of positive differences).
    statistic: float
    #: Two-sided p-value; exact for ``n <= EXACT_LIMIT``.
    p_value: float
    exact: bool

    @property
    def magnitude(self) -> str:
        return interpret_effect_size(self.effect_size)


def _ranks(values: Sequence[float]) -> list[float]:
    """Ranks of ``values``, averaging over ties -- the standard midrank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
            stop += 1
        midrank = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[order[position]] = midrank
        index = stop + 1
    return ranks


def wilcoxon_signed_rank(deltas: Sequence[float]) -> tuple[float, float, bool]:
    """Two-sided Wilcoxon signed-rank test on paired differences.

    Returns ``(statistic, p_value, exact)``. Zero differences are dropped, which
    is Wilcoxon's original handling and the conservative choice: it reduces the
    sample size rather than inventing a direction for a tie.

    For ``n <= EXACT_LIMIT`` the p-value is exact, computed by enumerating every
    assignment of signs to the observed ranks under the null hypothesis that each
    is equally likely. Beyond that it falls back to the normal approximation with
    a continuity correction.
    """
    nonzero = [d for d in deltas if d != 0.0]
    n = len(nonzero)
    if n == 0:
        return (0.0, 1.0, True)

    ranks = _ranks([abs(d) for d in nonzero])
    positive = sum(r for r, d in zip(ranks, nonzero, strict=True) if d > 0)

    if n <= EXACT_LIMIT:
        total = 0
        at_least_as_extreme = 0
        # Under the null each rank is equally likely to carry a + or a - sign.
        mean_statistic = sum(ranks) / 2.0
        observed_deviation = abs(positive - mean_statistic)
        for signs in product((0, 1), repeat=n):
            statistic = sum(r for r, s in zip(ranks, signs, strict=True) if s)
            total += 1
            if abs(statistic - mean_statistic) >= observed_deviation - 1e-12:
                at_least_as_extreme += 1
        return (positive, at_least_as_extreme / total, True)

    mean_statistic = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    if variance <= 0:
        return (positive, 1.0, False)
    z = (abs(positive - mean_statistic) - 0.5) / math.sqrt(variance)
    p = 2.0 * (1.0 - _standard_normal_cdf(z))
    return (positive, min(1.0, max(0.0, p)), False)


def _standard_normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def cohens_d_paired(deltas: Sequence[float]) -> float:
    """Cohen's :math:`d` for paired samples: mean difference over its own sd.

    Returns 0 when every difference is identical, which is a degenerate case the
    formula would otherwise divide by zero on. That is the right answer in the
    sense that matters here -- no spread means no way to judge the size of the
    effect against noise, so reporting a huge effect size would be misleading.
    """
    if len(deltas) < 2:
        return 0.0
    spread = statistics.stdev(deltas)
    return statistics.fmean(deltas) / spread if spread > 0 else 0.0


def interpret_effect_size(d: float) -> str:
    """Cohen's conventional labels. Conventions, not laws -- reported as such."""
    magnitude = abs(d)
    if magnitude < 0.2:
        return "negligible"
    if magnitude < 0.5:
        return "small"
    if magnitude < 0.8:
        return "medium"
    return "large"


def paired_test(treatment: Sequence[float], control: Sequence[float]) -> PairedTest:
    """Full paired comparison: mean delta, effect size, and an exact rank test."""
    if len(treatment) != len(control):
        raise ValueError("paired test needs the same seeds on both sides")
    deltas = [t - c for t, c in zip(treatment, control, strict=True)]
    statistic, p_value, exact = wilcoxon_signed_rank(deltas)
    return PairedTest(
        n=len(deltas),
        mean_delta=statistics.fmean(deltas),
        effect_size=cohens_d_paired(deltas),
        statistic=statistic,
        p_value=p_value,
        exact=exact,
    )


def holm_bonferroni(
    p_values: dict[str, float], *, alpha: float = 0.05
) -> dict[str, tuple[float, bool]]:
    """Holm-Bonferroni step-down correction for a family of comparisons.

    Returns ``{name: (adjusted_p, reject)}``. Controls the family-wise error rate
    without assuming the tests are independent, and is uniformly more powerful
    than plain Bonferroni -- it only applies the full ``m`` penalty to the
    smallest p-value.

    Comparing one proposed method against six baselines is six chances to clear
    ``p < 0.05``; without a correction, one of them clearing it means very little.
    """
    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, tuple[float, bool]] = {}
    running_max = 0.0
    for index, (name, p) in enumerate(ordered):
        # Step-down: enforce monotonicity so an adjusted p can never decrease.
        value = min(1.0, max(running_max, (m - index) * p))
        running_max = value
        adjusted[name] = (value, value <= alpha)
    return adjusted
