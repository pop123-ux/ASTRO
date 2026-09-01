"""Build every figure that has data, and name the ones that do not.

The point of running this is not only the PNGs. It prints a manifest of which
claims currently have a figure behind them and which are waiting on a
measurement, so the gap between what the paper says and what it can show is
visible in one command instead of being reconstructed from memory.

    python scripts/figures/make_all.py
    python scripts/figures/make_all.py --only leverage quintic
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fig_curvature  # noqa: E402
import fig_drift  # noqa: E402
import fig_horizon  # noqa: E402
import fig_inversion  # noqa: E402
import fig_leverage  # noqa: E402
import fig_quintic  # noqa: E402
import fig_results  # noqa: E402
from style import OUT  # noqa: E402

# name -> (builder, what it supports, whether it needs a measurement upload)
FIGURES = {
    "leverage": (fig_leverage.build,
                 "row norms of a Muon update are leverage scores", False),
    "quintic": (fig_quintic.build,
                "Muon's iteration cannot reach the polar factor", False),
    "curvature": (fig_curvature.build,
                  "the advantage is a direction effect, not a step-size effect", True),
    "results": (fig_results.build,
                "the 124M comparison, with its noise floor", False),
    "inversion": (fig_inversion.build,
                  "a component whose sign inverts with scale", False),
    "horizon": (fig_horizon.build,
                "whether the margin survives longer training", False),
    "drift": (fig_drift.build,
              "the non-convergence observed during real training", True),
}


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
        builder, claim, needs_upload = FIGURES[name]
        print(f"\n{name}: {claim}")
        # Modification times, not existence: a stale PNG left by an earlier run
        # or by the test suite would otherwise be reported as freshly built.
        before = stamps()
        try:
            builder()
        except Exception:  # a broken figure must not hide the working ones
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
        print("waiting on a measurement:")
        for name, needs_upload in skipped:
            where = "GPU run, then copy the JSON here" if needs_upload else "unknown"
            print(f"  {name:10s} -- {where}")
    if failed:
        print(f"FAILED {len(failed)}: {', '.join(failed)}")
    print(f"output: {OUT}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
