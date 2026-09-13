"""Build paper figures that have data, and name the ones that do not.

Figure modules are imported lazily. This matters because some mechanistic
figures use PyTorch for exact matrix measurements, while the current ASTRO-v2
empirical plots need only NumPy + Matplotlib. Requesting one lightweight plot
should not force every optional scientific dependency to be installed.

    python scripts/figures/make_all.py
    python scripts/figures/make_all.py --only optimizer_journey paired_900 horizon_v2
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
    # Current ASTRO-v2 paper spine -----------------------------------------
    "optimizer_journey": (
        "fig_optimizer_journey",
        "the progression from Muon/NorMuon/AdaMuon to ASTRO-MB and ASTRO-v2",
        False,
    ),
    "paired_900": (
        "fig_paired_900",
        "all five 124M/900 shared-configuration pairings",
        False,
    ),
    "horizon_v2": (
        "fig_horizon_v2",
        "whether ASTRO-v2's paired margin survives longer training",
        False,
    ),
    "efficiency_v2": (
        "fig_efficiency_v2",
        "the validation-loss gain together with ASTRO-v2's measured runtime cost",
        False,
    ),
    "seed_replication": (
        "fig_seed_replication",
        "independent-seed replication with pending-aware rendering",
        False,
    ),
    "scale_status": (
        "fig_scale_status",
        "paper-eligible scale evidence with explicit empty experiment slots",
        False,
    ),
    "training_trajectories": (
        "fig_training_trajectories",
        "validation loss versus both training step and wall-clock",
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


def load_builder(module_name: str):
    module = importlib.import_module(module_name)
    return module.build


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", choices=sorted(FIGURES),
                        help="build a subset")
    args = parser.parse_args()
    wanted = args.only or list(FIGURES)

    def stamps() -> dict[Path, float]:
        return {path: path.stat().st_mtime_ns
                for path in (OUT.glob("*.png") if OUT.exists() else ())}

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
            where = "GPU run / trajectory file required" if needs_upload else "missing input"
            print(f"  {name:22s} -- {where}")
    if failed:
        print(f"FAILED {len(failed)}: {', '.join(failed)}")
    print(f"output: {OUT}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
