"""Seed resolution for candidate comparisons.

A paired signed-rank test on values from different seeds is not a weaker test,
it is a different and meaningless one, and nothing in the numbers reveals the
mismatch. This project's two evaluation entry points historically used different
seed ranges -- ``astro.bench.run`` starts at 100, ``scripts/headline.py`` at
200 -- and ``candidate.py`` paired against whichever it found. Earlier rounds
escaped it only because a headline artifact happened to exist.

The fix is not to validate the convention but to remove it: candidates run on
whatever seeds the reference recorded. These tests pin that.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from candidate import resolve_paired_seeds  # noqa: E402


def test_seeds_come_from_the_reference() -> None:
    seeds = resolve_paired_seeds(
        {"muon": list(range(100, 110)), "adamw": list(range(100, 110))},
        ["muon", "adamw"], 10,
    )
    assert seeds == list(range(100, 110))


def test_a_different_reference_range_is_followed_not_overridden() -> None:
    """headline.py evaluates on 200+; the candidate must follow it there."""
    seeds = resolve_paired_seeds({"muon": list(range(200, 205))}, ["muon"], 5)
    assert seeds == list(range(200, 205))


def test_unrecorded_seeds_are_refused() -> None:
    with pytest.raises(SystemExit, match="do not record which seeds"):
        resolve_paired_seeds({"muon": []}, ["muon"], 5)

    with pytest.raises(SystemExit, match="do not record which seeds"):
        resolve_paired_seeds({}, ["muon"], 5)


def test_baselines_evaluated_on_different_seeds_are_refused() -> None:
    """Pairing one candidate against two differently-seeded baselines would make
    the two comparisons incomparable even if each were individually valid."""
    with pytest.raises(SystemExit, match="different seeds"):
        resolve_paired_seeds(
            {"muon": list(range(100, 105)), "adamw": list(range(200, 205))},
            ["muon", "adamw"], 5,
        )


def test_too_few_reference_seeds_are_refused() -> None:
    with pytest.raises(SystemExit, match="only 3 seeds"):
        resolve_paired_seeds({"muon": [100, 101, 102]}, ["muon"], 10)


def test_extra_reference_seeds_are_truncated_consistently() -> None:
    seeds = resolve_paired_seeds(
        {"muon": list(range(100, 120)), "adamw": list(range(100, 115))},
        ["muon", "adamw"], 10,
    )
    assert seeds == list(range(100, 110))


def test_only_the_requested_baselines_are_consulted() -> None:
    """An unused baseline with odd seeds must not block a valid comparison."""
    seeds = resolve_paired_seeds(
        {"muon": list(range(100, 105)), "soap": list(range(900, 905))},
        ["muon"], 5,
    )
    assert seeds == list(range(100, 105))
