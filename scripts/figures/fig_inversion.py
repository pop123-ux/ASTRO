"""Figure 5: a component whose sign inverts between 1.17M and 124M.

The claim this figure has to carry is uncomfortable and specific: a clean
small-scale sweep -- 8 of 8 seeds, exact p = 0.0078 -- predicted the wrong sign
at 100x the scale. Drawing it as a bar chart of two numbers would make it look
like a curiosity. The panels are built to close the obvious escape routes:

  (a) the effect itself, crossing zero, with seeds-won and exact p on each bar
  (b) the escape route we tried and lost -- the two benchmarks agree on the
      elementwise/spectral parameter split to within 0.5 points, so the split
      does not explain the inversion
  (c) what the small benchmark got *right*, so the conclusion is "sign does not
      transfer", not "small benchmarks are useless"

Reads ``artifacts/measured.json``.
"""

from __future__ import annotations

import argparse

import numpy as np
from style import (
    COLORS,
    annotate_value,
    figure,
    grid,
    label_panels,
    missing,
    read_data,
    save,
    write_data,
)

# Components the small benchmark selected that survived the jump to 124M. Kept
# beside the one that did not, because the honest lesson depends on both.
TRANSFERRED = [
    ("QKV split", "ordering held"),
    ("variance placement", "ordering held"),
    ("update scale", "ordering held"),
    ("cautious mask", "sign inverted"),
]


def build(source: str = "measured.json") -> None:
    payload = read_data(source)
    if payload is None:
        missing("fig5_inversion", "artifacts/measured.json is missing from the repo")
        return
    block = payload["scale_inversion"]
    points = block["points"]

    fig, axes = figure(3, height=2.25)
    left, middle, right = axes

    # (a) the sign flip -------------------------------------------------
    positions = np.arange(len(points))
    deltas = [point["delta_vs_muon"] for point in points]
    colors = [COLORS["astro"] if value < 0 else COLORS["astro_cautious"]
              for value in deltas]
    left.bar(positions, deltas, width=0.5, color=colors, zorder=3)
    left.axhline(0, color="#333333", linewidth=0.9, zorder=4)
    for index, point in enumerate(points):
        value = point["delta_vs_muon"]
        annotate_value(left, index, value,
                       f"{value:+.4f}\n{point['seeds_won']}/{point['seeds']} seeds\n"
                       f"$p = {point['exact_p']:.4g}$",
                       dy=6 if value > 0 else -30, fontsize=6)
    left.set_xticks(positions)
    left.set_xticklabels([point["label"] for point in points])
    left.set_xlabel("model size")
    left.set_ylabel(r"$\Delta$ vs Muon from the mask")
    # Deep enough that the three-line label under the negative bar clears the
    # tick labels.
    left.set_ylim(min(deltas) * 4.4, max(deltas) * 1.75)
    left.annotate("helps", (0.02, 0.44), xycoords="axes fraction",
                  fontsize=6.5, color=COLORS["astro"])
    left.annotate("hurts", (0.02, 0.72), xycoords="axes fraction",
                  fontsize=6.5, color=COLORS["astro_cautious"])
    grid(left, axis="y")

    # (b) the explanation that failed -----------------------------------
    fractions = [point["elementwise_fraction"] for point in points]
    middle.bar(positions - 0.16, fractions, width=0.3,
               color=COLORS["muon"], zorder=3, label="elementwise path")
    middle.bar(positions + 0.16, [1 - value for value in fractions], width=0.3,
               color=COLORS["normuon"], zorder=3, label="spectral path")
    for index, fraction in enumerate(fractions):
        annotate_value(middle, index - 0.16, fraction, f"{fraction:.3f}",
                       dy=3, fontsize=6)
    middle.set_xticks(positions)
    middle.set_xticklabels([point["label"] for point in points])
    middle.set_ylim(0, 1.12)
    middle.set_ylabel("fraction of parameters")
    middle.legend(loc="upper center", fontsize=6, ncol=2, columnspacing=0.8)
    gap = abs(fractions[0] - fractions[1])
    middle.annotate(f"splits agree to {gap * 100:.1f} pts\n"
                    "-- so the split is not\nthe explanation",
                    (0.5, 0.06), xycoords="axes fraction", ha="center",
                    fontsize=6, color="#444444")
    grid(middle, axis="y")

    # (c) what did transfer ---------------------------------------------
    labels = [name for name, _ in TRANSFERRED]
    held = [outcome == "ordering held" for _, outcome in TRANSFERRED]
    spots = np.arange(len(labels))
    right.barh(spots, [1] * len(labels), height=0.5, zorder=3,
               color=[COLORS["normuon"] if ok else COLORS["astro_cautious"]
                      for ok in held])
    for index, (_name, outcome) in enumerate(TRANSFERRED):
        right.annotate(outcome, (0.5, index), ha="center", va="center",
                       fontsize=6, color="white", zorder=5)
    right.set_yticks(spots)
    right.set_yticklabels(labels, fontsize=6.5)
    right.invert_yaxis()
    right.set_xticks([])
    right.set_xlabel(f"selected at 1.17M, checked at 124M\n"
                     f"{sum(held)} of {len(held)} transferred", fontsize=6.5)
    for spine in ("left", "bottom"):
        right.spines[spine].set_visible(False)

    label_panels(axes)
    fig.tight_layout(pad=0.4, w_pad=1.5)
    saved = save(fig, "fig5_inversion")
    write_data("fig5_inversion", {
        "points": points,
        "split_gap_points": gap * 100,
        "transferred": dict(TRANSFERRED),
    })

    print(f"  wrote {saved}")
    for point in points:
        print(f"    {point['label']:>6s}: {point['delta_vs_muon']:+.4f}  "
              f"{point['seeds_won']}/{point['seeds']} seeds  "
              f"exact p = {point['exact_p']}")
    print(f"    magnitude ratio {abs(deltas[1] / deltas[0]):.1f}x, opposite sign")
    print(f"    parameter splits differ by {gap * 100:.1f} points")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="measured.json")
    args = parser.parse_args()
    build(args.data)


if __name__ == "__main__":
    main()
