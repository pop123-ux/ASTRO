"""Build ASTRO paper figures that have data, and name the ones that do not.

The final paper campaign is deliberately isolated from the historical figures.
Use ``--paper`` after each experimental milestone; only figures backed by
``artifacts/paper_campaign.json`` will render.

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

# name -> (module, what it supports, whether it needs a measurement upload)
FIGURES = {
    # Clean paper-proof campaign -------------------------------------------
    "paper_main": (
        "fig_paper_main",
        "five held-out seeds comparing Muon, faithful AdaMuon, and ASTRO-v2",
        True,
    ),
    "paper_convergence": (
        "fig_paper_convergence",
        "training-loss convergence versus optimizer steps and measured wall-clock",
        True,
    ),
    "paper_ablation": (
        "fig_paper_ablation",
        "the minimal beta / row-adaptation / QKV-split causal ladder",
        True,
    ),
    "paper_scale": (
        "fig_paper_scale",
        "frozen 124M recipe transfer to 355M",
        True,
    ),
    "paper_horizon": (
        "fig_paper_horizon",
        "frozen-recipe durability from 900 to 2700 steps",
        True,
    ),

    # Historical ASTRO-v2 research spine ---------------------------------
    "optimizer_journey": (
        "fig_optimizer_journey",
        "the progression from Muon/NorMuon/AdaMuon to ASTRO-MB and ASTRO-v2",
        False,
    ),
    "paired_900": (
        "fig_paired_900",
        "all five historical 124M/900 shared-configuration pairings",
        False,
    ),
    "horizon_v2": (
        "fig_horizon_v2",
        "the historical ASTRO-v2 paired margin across training horizons",
        False,
    ),
    "efficiency_v2": (
        "fig_efficiency_v2",
        "historical validation-loss gain and runtime cost",
        False,
    ),
    "seed_replication": (
        "fig_seed_replication",
        "historical independent-seed replication rendering",
        False,
    ),
    "scale_status": (
        "fig_scale_status",
        "historical scale-status figure",
        False,
    ),
    "training_trajectories": (
        "fig_training_trajectories",
        "historical validation loss versus step and wall-clock",
        True,
    ),

    # Mechanistic / historical figures retained as supporting evidence ----
    "leverage": (
        "fig_leverage",
        "row norms of a Muon update are leverage scores",
        False,
    ),
    "quintic": (
        "fig_quintic",
        "Muon's repeated quintic does not converge to the exact polar factor",
        False,
    ),
    "curvature": (
        "fig_curvature",
        "the advantage is a direction effect, not only a step-size effect",
        True,
    ),
    "results_legacy": (
        "fig_results",
        "the earlier 124M comparison, retained as research history",
        False,
    ),
    "inversion": (
        "fig_inversion",
        "a component whose sign inverts with scale",
        False,
    ),
    "horizon_legacy": (
        "fig_horizon",
        "the superseded/earlier horizon study",
        False,
    ),
    "drift": (
        "fig_drift",
        "spectral update-norm drift observed during real training",
        True,
    ),
}

PAPER_FIGURES = (
    "paper_main",
    "paper_convergence",
    "paper_ablation",
    "paper_scale",
    "paper_horizon",
)


def load_builder(module_name: str):
    module = importlib.import_module(module_name)
    return module.build


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", nargs="*", choices=sorted(FIGURES),
                        help="build a named subset")
    parser.add_argument("--paper", action="store_true",
                        help="build only the clean paper-proof campaign figures")
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
        module_name, claim, needs_upload = FIGURES[name]
        print(f"\n{name}: {claim}")
        before = stamps()
        try:
            builder = load_builder(module_name)
            builder()
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
            skipped.append((name, needs_upload))

    print("\n" + "=" * 70)
    print(f"built {len(built)}: {', '.join(built) or 'none'}")
    if skipped:
        print("waiting on a measurement or input:")
        for name, needs_upload in skipped:
            where = "paper campaign measurement required" if needs_upload else "missing input"
            print(f"  {name:22s} -- {where}")
    if failed:
        print(f"FAILED {len(failed)}: {', '.join(failed)}")
    print(f"output: {OUT}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
