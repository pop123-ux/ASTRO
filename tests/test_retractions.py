"""Withdrawn claims stay withdrawn, and the paper counts them correctly.

This file exists because of a specific failure. The claim that splitting a fused
QKV projection is *cheaper* than not splitting it was retracted in
``docs/ROUND4.md`` and went on being asserted as fact in ``paper.tex``'s
conclusion, in its methods section, and in ``RESULTS_LLM.md`` -- for days, while
the abstract two pages above advertised the retraction. Nothing caught it,
because prose has no compiler.

So the retracted claims are pinned here. A withdrawn claim may appear in the
docs only where it is being withdrawn; asserting it again fails the suite.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "docs" / "paper" / "paper.tex"

# Documents that make claims to a reader. paper.md is the superseded Markdown
# draft the PDF was built from and is not maintained; it is excluded
# deliberately rather than by omission.
PROSE = [
    ROOT / "docs" / "paper" / "paper.tex",
    ROOT / "docs" / "RESULTS_LLM.md",
    ROOT / "docs" / "ROUND4.md",
    ROOT / "docs" / "PRIOR_ART.md",
    ROOT / "docs" / "COLAB.md",
    ROOT / "README.md",
]

# Words that mark a sentence as withdrawing a claim rather than making it. A
# banned phrase is allowed within RETRACTION_WINDOW lines of one of these.
WITHDRAWAL = re.compile(
    r"withdraw|retract|was wrong|is wrong|no longer|corrected|incorrect|"
    r"does not exist|we first reported|originally|mistaken",
    re.IGNORECASE,
)
RETRACTION_WINDOW = 400  # characters, measured on the collapsed document

# (why it is banned, pattern that would only match the claim being asserted)
#
# Patterns run against the document with all whitespace collapsed, because
# every real instance of the split-cost claim was wrapped across two lines and
# a line-by-line guard saw none of them.
BANNED = [
    (
        "splitting the fused QKV is cheaper than not splitting it "
        "(it is 1.29x more expensive by operation count, 1.33x measured)",
        re.compile(
            # "...splitting ... is cheaper", either order, across a sentence
            r"split\w*[^.]{0,200}\bcheaper\b"
            r"|\bcheaper\b[^.]{0,200}split\w*"
            # "the fix is also cheaper than the defect"
            r"|\bfix\b[^.]{0,60}\bcheaper\b"
            # "costs less than not fixing/splitting it"
            r"|costs? less than not\b"
            # "three 128x128 ... which is cheaper"
            r"|which is cheaper",
            re.IGNORECASE,
        ),
    ),
    (
        "GPU runs are reproducible given a seed "
        "(measured spread 0.0021 across sessions on identical seeds)",
        # The negated form -- "not bit-reproducible given a seed" -- is the
        # corrected statement and must not trip the guard, so the negation is
        # excluded in the pattern rather than by widening WITHDRAWAL.
        re.compile(r"(?<!not )(?<!not bit-)"
                   r"(deterministic|reproducible|bit-identical|bit-exact)"
                   r"[^.]{0,80}(given|for) (a|the) seed", re.IGNORECASE),
    ),
]


def audit(text: str) -> list[tuple[str, str]]:
    """Return (reason, quote) for every withdrawn claim asserted in ``text``.

    Whitespace is collapsed first so a claim wrapped across lines is still one
    string, and the withdrawal context is a character window around the match
    rather than a line window.
    """
    flat = re.sub(r"\s+", " ", text)
    found: list[tuple[str, str]] = []
    for reason, pattern in BANNED:
        for match in pattern.finditer(flat):
            low = max(0, match.start() - RETRACTION_WINDOW)
            context = flat[low:match.end() + RETRACTION_WINDOW]
            if not WITHDRAWAL.search(context):
                found.append((reason, flat[low:match.end() + 80]))
    return found


@pytest.mark.parametrize("path", PROSE, ids=lambda p: p.name)
def test_withdrawn_claims_are_not_asserted_again(path: Path) -> None:
    problems = audit(path.read_text(encoding="utf-8"))
    assert not problems, "\n".join(
        f"{path.name} asserts a withdrawn claim -- {reason}\n  ...{quote}...\n"
        "If this is a retraction, say so within a few hundred characters of it."
        for reason, quote in problems
    )


def test_the_guard_catches_the_text_it_was_written_for() -> None:
    """Every one of these shipped. A guard that misses them is worse than none."""
    shipped = [
        "The fix is also \\emph{cheaper} than the defect: three $128\\times128$ "
        "Newton--Schulz iterations cost less than one $384\\times128$ iteration.",
        "splitting the fused QKV replaces a single 384x128 Newton-Schulz with "
        "three 128x128 ones, which is cheaper.",
        "Splitting the projection fixes it, costs less than not fixing it, and "
        "transfers across regimes.",
        "Runs are deterministic given a seed, so a baseline recorded in an "
        "earlier session is directly comparable.",
    ]
    for claim in shipped:
        assert audit(claim), f"guard does not catch: {claim[:70]}"


def test_the_guard_allows_the_corrected_statements() -> None:
    corrected = [
        "GPU training here is not bit-reproducible given a seed, so we treat "
        "any cross-session gap below 0.005 as a tie.",
        "We first reported that splitting is cheaper than not splitting. That "
        "is wrong: it is 1.29x more expensive by operation count.",
    ]
    for claim in corrected:
        assert not audit(claim), f"guard false-positives on: {claim[:70]}"


# ---------------------------------------------------------------------------
# The paper's own bookkeeping
# ---------------------------------------------------------------------------


def _corrections_paragraphs() -> list[str]:
    text = PAPER.read_text(encoding="utf-8")
    start = text.index(r"\section{Corrections}")
    end = text.index(r"\section{Limitations}", start)
    return re.findall(r"\\paragraph\{(.+?)\}", text[start:end], re.DOTALL)


def test_the_abstract_counts_what_the_section_contains() -> None:
    """The abstract advertised five retractions while the section held four."""
    paragraphs = _corrections_paragraphs()
    bugs = [p for p in paragraphs if p.startswith("Bug")]
    retractions = [p for p in paragraphs if not p.startswith("Bug")]

    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    abstract = PAPER.read_text(encoding="utf-8")
    claimed = re.search(
        r"We report (\w+) claims this work made and withdrew and (\w+) bugs",
        abstract,
    )
    assert claimed, "the abstract no longer states its retraction count"

    assert words[claimed.group(1)] == len(retractions), (
        f"abstract says {claimed.group(1)} retractions, "
        f"section has {len(retractions)}: {retractions}"
    )
    assert words[claimed.group(2)] == len(bugs), (
        f"abstract says {claimed.group(2)} bugs, section has {len(bugs)}: {bugs}"
    )


def test_bugs_are_numbered_consecutively() -> None:
    bugs = [p for p in _corrections_paragraphs() if p.startswith("Bug")]
    numbers = [int(re.match(r"Bug (\d+)", p).group(1)) for p in bugs]
    assert numbers == list(range(1, len(bugs) + 1)), numbers


def test_the_split_cost_is_stated_as_measured() -> None:
    """The replacement claim carries the number, so it can be checked."""
    text = PAPER.read_text(encoding="utf-8")
    assert "1.29" in text and "1.33" in text, (
        "the corrected split cost lost its predicted/measured figures"
    )


def test_the_measurement_record_is_actually_in_the_repository() -> None:
    """A paper's numbers are only checkable if the record of them ships.

    ``artifacts/`` was ignored by the *root* .gitignore, which matches at any
    depth, so ``artifacts/measured.json`` -- the provenance for every measured
    figure the paper prints -- was never committed for the whole project. It
    existed only in the working container.
    """
    import subprocess

    root = PAPER.parent.parent.parent
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "artifacts/measured.json"],
        cwd=root, capture_output=True, text=True)
    assert tracked.returncode == 0, (
        "artifacts/measured.json is not tracked by git; the paper cites numbers "
        f"that no reader can retrieve. {tracked.stderr.strip()}"
    )
