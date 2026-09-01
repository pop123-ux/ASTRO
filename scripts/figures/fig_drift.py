"""Figure 7: the non-convergence, observed during real training.

Reads the JSON from ``scripts/measure_drift.py``. The synthetic result in
Figure 2 says Muon's update norm depends on the conditioning of the momentum;
this says whether real momentum matrices are conditioned differently enough for
that to matter.

  (a) update-norm ratio over training, one line per tracked tensor
  (b) the spread across tensors at each step -- the quantity a step size is
      implicitly assuming is zero
  (c) ratio against the momentum's condition number, which is the mechanism
      from Figure 2 measured on real matrices rather than planted ones
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
from style import (
    COLORS,
    LABELS,
    figure,
    grid,
    label_panels,
    missing,
    read_data,
    reference_line,
    save,
    write_data,
)

PRODUCED_BY = ("python scripts/measure_drift.py --optimizers muon astro "
               "--steps 300 --every 10 --out drift.json")


def build(source: str = "drift.json") -> None:
    payload = read_data(source)
    if payload is None:
        missing("fig7_drift", PRODUCED_BY)
        return

    by_optimizer: dict[str, list[dict]] = defaultdict(list)
    for record in payload["records"]:
        by_optimizer[record["optimizer"]].append(record)

    fig, axes = figure(3, height=2.2)
    left, middle, right = axes
    summary: dict[str, dict[str, float]] = {}

    for name, records in sorted(by_optimizer.items()):
        color = COLORS.get(name, "#333")
        per_tensor: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for record in records:
            per_tensor[record["tensor"]].append((record["step"], record["ratio"]))

        # (a) one faint line per tensor, plus the mean --------------------
        for points in per_tensor.values():
            points.sort()
            steps, ratios = zip(*points, strict=True)
            left.plot(steps, ratios, color=color, linewidth=0.5, alpha=0.35,
                      zorder=2)
        steps_axis = sorted({record["step"] for record in records})
        means = [np.mean([r["ratio"] for r in records if r["step"] == step])
                 for step in steps_axis]
        left.plot(steps_axis, means, color=color, linewidth=1.6, zorder=4,
                  label=LABELS.get(name, name))

        # (b) spread across tensors --------------------------------------
        spreads = [max(r["ratio"] for r in records if r["step"] == step)
                   - min(r["ratio"] for r in records if r["step"] == step)
                   for step in steps_axis]
        middle.plot(steps_axis, spreads, color=color, linewidth=1.4, zorder=3,
                    label=LABELS.get(name, name))

        # (c) ratio against the conditioning that produced it -------------
        conditions = [record.get("momentum_condition") for record in records]
        ratios = [record["ratio"] for record in records]
        usable = [(c, r) for c, r in zip(conditions, ratios, strict=True)
                  if c is not None and np.isfinite(c)]
        if usable:
            xs, ys = zip(*usable, strict=True)
            right.scatter(xs, ys, s=3, color=color, alpha=0.4, linewidths=0,
                          zorder=3, label=LABELS.get(name, name))

        summary[name] = {
            "mean_ratio": float(np.mean(ratios)),
            "min_ratio": float(np.min(ratios)),
            "max_ratio": float(np.max(ratios)),
            "max_spread_across_tensors": float(np.max(spreads)),
        }

    reference_line(left, 1.0, "polar factor")
    left.set_xlabel("training step")
    left.set_ylabel(r"$\|Z\|_F / \sqrt{\min(m,n)}$")
    left.legend(loc="best", fontsize=6)
    left.annotate("faint: one tensor\nbold: mean", (0.03, 0.06),
                  xycoords="axes fraction", fontsize=6, color="#666666")
    grid(left)

    middle.set_xlabel("training step")
    middle.set_ylabel("spread across tensors")
    middle.set_ylim(bottom=0)
    middle.legend(loc="best", fontsize=6)
    grid(middle)

    right.set_xscale("log")
    reference_line(right, 1.0, "polar factor")
    right.set_xlabel("momentum condition number")
    right.set_ylabel(r"$\|Z\|_F / \sqrt{\min(m,n)}$")
    right.legend(loc="best", fontsize=6, markerscale=2.5)
    grid(right)

    label_panels(axes)
    fig.tight_layout(pad=0.4, w_pad=1.5)
    saved = save(fig, "fig7_drift")
    write_data("fig7_drift", summary)

    print(f"  wrote {saved}")
    for name, values in summary.items():
        print(f"    {name:8s} ratio {values['min_ratio']:.4f}-{values['max_ratio']:.4f} "
              f"(mean {values['mean_ratio']:.4f}), "
              f"worst spread {values['max_spread_across_tensors']:.4f}")
    if "muon" in summary and "astro" in summary:
        if summary["muon"]["max_spread_across_tensors"] < 0.02:
            print("    NOTE: Muon's ratio is nearly flat here. The "
                  "non-convergence is real but would not be costing it "
                  "anything at this scale -- report that, do not bury it.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="drift.json")
    args = parser.parse_args()
    build(args.data)


if __name__ == "__main__":
    main()
