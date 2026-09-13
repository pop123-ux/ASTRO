#!/usr/bin/env python3
"""Render the ASTRO paper figures and compile docs/paper/paper.tex.

The build is intentionally boring: figures first, LaTeX second. A figure script
may explicitly skip a measurement that is not available yet; the LaTeX source
uses visible placeholders for missing optional figures so a living draft still
compiles without pretending the experiment exists.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "docs" / "paper"

PRIMARY_FIGURES = [
    "optimizer_journey",
    "paired_900",
    "horizon_v2",
    "efficiency_v2",
    "seed_replication",
    "scale_status",
    "training_trajectories",
    "leverage",
    "quintic",
]


def run(command: list[str], *, cwd: Path = ROOT) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    run([sys.executable, "scripts/figures/make_all.py", "--only", *PRIMARY_FIGURES])

    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")
    if latexmk:
        run([latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "paper.tex"],
            cwd=PAPER)
    elif pdflatex:
        run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "paper.tex"], cwd=PAPER)
        run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "paper.tex"], cwd=PAPER)
    else:
        print("No TeX compiler found. Figures were rendered successfully.")
        print("Install latexmk/pdflatex or upload docs/paper/paper.tex to Overleaf.")
        return 2

    output = PAPER / "paper.pdf"
    print(f"\nPaper built: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
