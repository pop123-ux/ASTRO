"""Build ASTRO paper figures that have data, and name the ones that do not.

The confirmatory paper campaign is deliberately isolated from historical plots.
Use ``--paper`` after the final campaign or ``--only`` after each milestone.

    python scripts/figures/make_all.py --paper
    python scripts/figures/make_all.py --only paper_main paper_ablation
"""

from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from style import OUT  # noqa: E402

FIGURES = {
    # Clean confirmatory campaign -----------------------------------------
    "paper_main": (
        "fig_paper_main",
        "five held-out seeds plus measured runtime for the headline baselines",
        True,
    ),
    "paper_ablation": (
        "fig_paper_ablation",
        "beta, row-adaptation, QKV-split, and global-vs-blockwise controls",
        True,
    ),
    "paper_scale": (
        "fig_paper_scale",
        "frozen Muon/ASTRO-v2 transfer from 124M to 355M",
        True,
    ),
    "paper_horizon": (
        "fig_paper_horizon",
        "frozen Muon/ASTRO-v2 durability from 900 to 2700 steps",
        True,
    ),

    # Historical research figures retained for audit/history --------------
    "optimizer_journey": (
        "fig_optimizer_journey",
        "historical optimizer progression",
        False,
    ),
    "paired_900": (
        "fig_paired_900",
        "historical 124M/900 shared-configuration pairings",
        False,
    ),
    "horizon_v2": (
        "fig_horizon_v2",
        "historical pre-confirmatory horizon study",
        False,
    ),
    "efficiency_v2": (
        "fig_efficiency_v2",
        "historical same-step runtime study",
        False,
    ),
    "seed_replication": (
        "fig_seed_replication",
        "historical seed-replication rendering",
        False,
    ),
    "scale_status": (
        "fig_scale_status",
        "historical scale-status figure",
        False,
    ),
    "training_trajectories": (
        "fig_training_trajectories",
        "historical trajectory figure",
        True,
    ),
    "leverage": (
        "fig_leverage",
        "row norms of a Muon update are leverage scores",
        False,
    ),
    "quintic": (
        "fig_quintic",
        "Muon quintic behaviour",
        False,
    ),
    "curvature": (
        "fig_curvature",
        "direction versus step-size mechanism probe",
        True,
    ),
    "results_legacy": (
        "fig_results",
        "earlier 124M comparison",
        False,
    ),
    "inversion": (
        "fig_inversion",
        "historical component scale reversal",
        False,
    ),
    "horizon_legacy": (
        "fig_horizon",
        "superseded horizon study",
        False,
    ),
    "drift": (
        "fig_drift",
        "spectral update-norm drift probe",
        True,
    ),
}

PAPER_FIGURES = ("paper_main", "paper_ablation", "paper_scale", "paper_horizon")


def load_builder(module_name: str):
    return importlib.import_module(module_name).build


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", nargs="*", choices=sorted(FIGURES),
                        help="build a named subset")
    parser.add_argument("--paper", action="store_true",
                        help="build only the clean confirmatory paper figures")
    args = parser.parse_args()
    if args.only and args.paper:
        parser.error("use either --paper or --only, not both")
    wanted = list(args.only or (PAPER_FIGURES if args.paper else FIGURES))

    def stamps() -> dict[Path, float]:
        return {
            path: path.stat().st_mtime_ns
            for path in (OUT.glob("*.png") if OUT.exists() else ())
        }

    built, skipped, failed = [], [], []
    for name in wanted:
        module_name, claim, needs_measurement = FIGURES[name]
        print(f"\n{name}: {claim}")
        before = stamps()
        try:
            load_builder(module_name)()
        except ModuleNotFoundError as exc:
            print(f"  FAILED {name}: missing dependency {exc.name!r}")
            failed.append(name)
            continue
        except Exception:
            traceback.print_exc()
            failed.append(name)
            continue
        after = stamps()
        if any(time != before.get(path) for path, time in after.items()):
            built.append(name)
        else:
            skipped.append((name, needs_measurement))

    print("\n" + "=" * 70)
    print(f"built {len(built)}: {', '.join(built) or 'none'}")
    if skipped:
        print("waiting on input:")
        for name, needs_measurement in skipped:
            where = "confirmatory measurement required" if needs_measurement else "missing input"
            print(f"  {name:22s} -- {where}")
    if failed:
        print(f"FAILED {len(failed)}: {', '.join(failed)}")
    print(f"output: {OUT}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
