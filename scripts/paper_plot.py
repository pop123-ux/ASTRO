#!/usr/bin/env python3
"""Collect confirmatory states and build the figure appropriate to a milestone.

This is the user-facing plotting entry point for the paper campaign.  It keeps
plotting separate from training and never reads historical ``paper_results.json``.

Examples::

    python scripts/paper_plot.py --stage main
    python scripts/paper_plot.py --stage mechanism
    python scripts/paper_plot.py --stage horizon
    python scripts/paper_plot.py --stage scale
    python scripts/paper_plot.py --stage all

The default paths match ``docs/PAPER_PROOF_CAMPAIGN.md`` and can be overridden.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = Path("/content/drive/MyDrive/astro")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("main", "mechanism", "horizon", "scale", "all"),
        required=True,
    )
    parser.add_argument(
        "--main-state",
        type=Path,
        default=DEFAULT_ROOT / "paper_main" / "astro_lab_state.json",
    )
    parser.add_argument(
        "--mechanism-state",
        type=Path,
        default=DEFAULT_ROOT / "paper_mechanism" / "paper_mechanism_state.json",
    )
    parser.add_argument(
        "--horizon-state",
        type=Path,
        default=DEFAULT_ROOT / "paper_horizon_2700" / "paper_transfer_state.json",
    )
    parser.add_argument(
        "--scale-state",
        type=Path,
        default=DEFAULT_ROOT / "paper_scale_355m" / "paper_transfer_state.json",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "artifacts" / "paper_campaign.json",
    )
    args = parser.parse_args()

    if not args.main_state.is_file():
        raise SystemExit(f"headline state not found: {args.main_state}")

    collect = [
        sys.executable,
        str(ROOT / "scripts" / "paper_collect.py"),
        "--main",
        str(args.main_state),
        "--out",
        str(args.artifact),
    ]
    if args.mechanism_state.is_file():
        collect += ["--mechanism", str(args.mechanism_state)]
    if args.horizon_state.is_file():
        collect += ["--horizon", str(args.horizon_state)]
    if args.scale_state.is_file():
        collect += ["--scale", str(args.scale_state)]
    subprocess.run(collect, cwd=ROOT, check=True)

    figure = {
        "main": "paper_main",
        "mechanism": "paper_ablation",
        "horizon": "paper_horizon",
        "scale": "paper_scale",
    }
    plot = [
        sys.executable,
        str(ROOT / "scripts" / "figures" / "make_all.py"),
    ]
    if args.stage == "all":
        plot += ["--paper"]
    else:
        plot += ["--only", figure[args.stage]]
    subprocess.run(plot, cwd=ROOT, check=True)

    print(f"\nplot stage: {args.stage}")
    print(f"data artifact: {args.artifact}")
    print(f"figures: {ROOT / 'artifacts' / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
