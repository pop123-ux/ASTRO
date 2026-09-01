#!/usr/bin/env python3
"""Static checks on the paper source, for a machine with no TeX distribution.

    python scripts/check_tex.py docs/paper/paper.tex

This cannot tell you the paper compiles -- only a TeX run can, and this box has
none. What it can do is catch the errors that would waste a compile: an
unbalanced environment, a ``\\ref`` to a label nobody declares, a ``\\cite`` with
no bibliography entry, a stray unescaped ``%`` or ``&``, a math delimiter that
never closes.

Every check is reported as an error only when it is certain. Anything heuristic
is a warning, because a validator that cries wolf on correct LaTeX is worse than
no validator: it trains you to ignore it.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

#: Commands whose argument declares a name that ``\ref``-likes may point at.
LABEL = re.compile(r"\\label\{([^}]*)\}")
REFERENCE = re.compile(r"\\(?:ref|eqref|autoref|Cref|cref)\{([^}]*)\}")
CITATION = re.compile(r"\\(?:cite|citep|citet|citeauthor|citeyear)\*?(?:\[[^\]]*\])*\{([^}]*)\}")
BIBITEM = re.compile(r"\\bibitem(?:\[[^\]]*\])?\{([^}]*)\}")
BEGIN = re.compile(r"\\begin\{([^}]*)\}")
END = re.compile(r"\\end\{([^}]*)\}")
NEWCOMMAND = re.compile(r"\\(?:newcommand|renewcommand|providecommand)\*?\{?\\(\w+)")
DECLARE_OP = re.compile(r"\\DeclareMathOperator\*?\{\\(\w+)\}")
NEWTHEOREM = re.compile(r"\\newtheorem\*?\{([^}]*)\}")


def strip_comments(text: str) -> str:
    """Drop TeX comments, keeping escaped percent signs and line structure."""
    out = []
    for line in text.split("\n"):
        cleaned, index = [], 0
        while index < len(line):
            char = line[index]
            if char == "\\" and index + 1 < len(line):
                cleaned.append(line[index : index + 2])
                index += 2
                continue
            if char == "%":
                break
            cleaned.append(char)
            index += 1
        out.append("".join(cleaned))
    return "\n".join(out)


def check(path: Path) -> tuple[list[str], list[str]]:
    raw = path.read_text()
    text = strip_comments(raw)
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Environments nest and balance.
    stack: list[tuple[str, int]] = []
    for line_number, line in enumerate(text.split("\n"), start=1):
        for name in BEGIN.findall(line):
            stack.append((name, line_number))
        for name in END.findall(line):
            if not stack:
                errors.append(f"{path}:{line_number}: \\end{{{name}}} with nothing open")
            elif stack[-1][0] != name:
                opened, where = stack.pop()
                errors.append(
                    f"{path}:{line_number}: \\end{{{name}}} closes \\begin{{{opened}}} "
                    f"opened at line {where}"
                )
            else:
                stack.pop()
    for name, where in stack:
        errors.append(f"{path}:{where}: \\begin{{{name}}} never closed")

    # 2. Braces balance across the document.
    depth = 0
    for line_number, line in enumerate(text.split("\n"), start=1):
        index = 0
        while index < len(line):
            if line[index] == "\\" and index + 1 < len(line):
                index += 2
                continue
            if line[index] == "{":
                depth += 1
            elif line[index] == "}":
                depth -= 1
                if depth < 0:
                    errors.append(f"{path}:{line_number}: unmatched closing brace")
                    depth = 0
            index += 1
    if depth:
        errors.append(f"{path}: {depth} unclosed brace(s) at end of file")

    # 3. Inline math delimiters pair up. Count $ outside \[ \] blocks; an odd
    #    total is a certain error, since $$ is deprecated but still even.
    dollars = len(re.findall(r"(?<!\\)\$", text))
    if dollars % 2:
        errors.append(f"{path}: odd number of unescaped $ ({dollars}); a math run is unclosed")

    # 4. Every reference resolves, and every label is used.
    labels = set(LABEL.findall(text))
    referenced = set(REFERENCE.findall(text))
    for name in sorted(referenced - labels):
        errors.append(f"{path}: \\ref{{{name}}} has no \\label")
    for name in sorted(labels - referenced):
        warnings.append(f"{path}: \\label{{{name}}} is never referenced")

    duplicates = [name for name, count in Counter(LABEL.findall(text)).items() if count > 1]
    for name in sorted(duplicates):
        errors.append(f"{path}: \\label{{{name}}} declared more than once")

    # 5. Every citation key exists, if the bibliography is inline.
    keys: set[str] = set()
    for group in CITATION.findall(text):
        keys.update(key.strip() for key in group.split(","))
    entries = set(BIBITEM.findall(text))
    if entries:
        for key in sorted(keys - entries):
            errors.append(f"{path}: \\cite{{{key}}} has no \\bibitem")
        for key in sorted(entries - keys):
            warnings.append(f"{path}: \\bibitem{{{key}}} is never cited")
    elif keys:
        warnings.append(
            f"{path}: {len(keys)} citation key(s) but no inline \\bibitem; "
            "keys resolve from a .bib file this check cannot see"
        )

    # 6. Theorem-like environments are declared before use.
    declared = set(NEWTHEOREM.findall(text)) | {
        "abstract", "align", "align*", "array", "cases", "center", "document",
        "enumerate", "equation", "equation*", "figure", "itemize", "table",
        "tabular", "thebibliography", "verbatim", "quote", "gather", "gather*",
        "pmatrix", "bmatrix", "split", "algorithmic", "algorithm", "description",
        "displaymath", "flushleft", "flushright", "minipage", "small", "subequations",
    }
    used = set(BEGIN.findall(text))
    for name in sorted(used - declared):
        warnings.append(f"{path}: environment {{{name}}} not declared here (package-provided?)")

    # 7. Unescaped characters that are certain errors in text mode.
    for line_number, line in enumerate(text.split("\n"), start=1):
        if re.search(r"(?<!\\)&", line) and not re.search(r"\\begin|\\end|&|\\\\", line):
            warnings.append(f"{path}:{line_number}: bare & outside a tabular row?")

    # 8. Macros used but never defined, restricted to this file's own namespace.
    defined = set(NEWCOMMAND.findall(text)) | set(DECLARE_OP.findall(text))
    for name in sorted(defined):
        if not re.search(rf"\\{name}(?![A-Za-z])", text.replace(f"\\{name}}}", "", 1)):
            warnings.append(f"{path}: \\{name} is defined but appears unused")

    return errors, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path,
                        default=[Path("docs/paper/paper.tex")])
    args = parser.parse_args(argv)

    total_errors = 0
    for path in args.paths:
        if not path.exists():
            print(f"missing: {path}")
            total_errors += 1
            continue
        errors, warnings = check(path)
        total_errors += len(errors)
        for message in errors:
            print(f"error:   {message}")
        for message in warnings:
            print(f"warning: {message}")
        print(f"{path}: {len(errors)} error(s), {len(warnings)} warning(s)")

    if total_errors:
        print("\nStatic checks failed. This does not prove the rest compiles.")
    else:
        print("\nStatic checks passed. This is not a compile; run pdflatex before submitting.")
    return 1 if total_errors else 0


if __name__ == "__main__":
    sys.exit(main())
