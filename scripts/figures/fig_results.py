"""Figure 4: the 124M result, drawn so its weaknesses are visible.

A bar chart of means would overstate this result. Three seeds cannot support a
small p-value, and one of the five configurations was never tuned, so the
figure is built to show the paired structure and the noise floor rather than to
flatter the ordering:

  (a) every seed drawn individually, with the runs joined -- a reader can see
      that the ordering holds seed by seed rather than on average
  (b) paired differences against ASTRO, with the measured cross-session noise
      floor shaded, so a bar inside the band reads as a tie
  (c) loss against wall-clock, because a per-step win paid for in time is not
      a win

Reads ``artifacts/measured.json``; no GPU needed to redraw.
"""

from __future__ import annotations

import argparse

import numpy as np
from style import (
    COLORS,
    LABELS,
    MARKERS,
    annotate_value,
    figure,
    grid,
    label_panels,
    missing,
    read_data,
    save,
    write_data,
)


def sign_test(deltas: list[float]) -> tuple[int, float]:
    """Two-sided exact sign test, and its floor.

    With n seeds the smallest attainable two-sided p is 2/2^n, so at n = 3 no
    arrangement of the data gives less than 0.25. Reporting the floor next to
    the result is the only honest way to show a 3/3 sweep.
    """
    from math import comb
    wins = sum(1 for value in deltas if value < 0)
    total = len(deltas)
    extreme = min(wins, total - wins)
    tail = sum(comb(total, k) for k in range(extreme + 1))
    return wins, min(1.0, 2.0 * tail / 2 ** total)


def build(source: str = "measured.json") -> None:
    payload = read_data(source)
    if payload is None:
        missing("fig4_results", "artifacts/measured.json is missing from the repo")
        return
    block = payload["gpt2_124m_fineweb"]
    floor = payload["_provenance"]["noise_floor"]["value"]
    losses = block["losses"]
    timing = block["seconds_per_run"]
    order = sorted(losses, key=lambda name: np.mean(losses[name]))
    seeds = block["seeds"]

    fig, axes = figure(3, height=2.3)
    left, middle, right = axes

    # (a) every seed, joined -------------------------------------------
    positions = np.arange(len(order))
    for index, name in enumerate(order):
        values = losses[name]
        color = COLORS.get(name, "#333")
        left.scatter([index] * len(values), values, s=14, color=color,
                     zorder=4, linewidths=0)
        left.plot([index - 0.22, index + 0.22],
                  [np.mean(values)] * 2, color=color, linewidth=1.6, zorder=3)
    for seed_index in range(len(seeds)):
        left.plot(positions, [losses[name][seed_index] for name in order],
                  color="#999999", linewidth=0.5, zorder=2, alpha=0.8)
    left.set_xticks(positions)
    left.set_xticklabels([LABELS.get(name, name) for name in order],
                         rotation=30, ha="right")
    left.set_ylabel("validation loss")
    left.annotate(f"{len(seeds)} shared seeds;\nthin lines join one seed",
                  (0.03, 0.93), xycoords="axes fraction", ha="left", va="top",
                  fontsize=6, color="#666666")
    grid(left, axis="y")

    # (b) paired differences against ASTRO ------------------------------
    best = order[0]
    others = [name for name in order if name != best]
    stats = {}
    for index, name in enumerate(others):
        deltas = [losses[best][i] - losses[name][i] for i in range(len(seeds))]
        wins, p_value = sign_test(deltas)
        stats[name] = {"mean": float(np.mean(deltas)), "worst": float(max(deltas)),
                       "won": wins, "of": len(deltas), "exact_p": p_value}
        color = COLORS.get(name, "#333")
        middle.barh(index, np.mean(deltas), height=0.55, color=color,
                    zorder=3, alpha=0.9)
        middle.scatter(deltas, [index] * len(deltas), s=9, color="#222222",
                       zorder=5, linewidths=0)
        # Outside the bar on the negative side: bars here span two orders of
        # magnitude, so a label placed inside fits the long ones and spills
        # out of the short ones.
        annotate_value(middle, min(deltas), index,
                       f"{np.mean(deltas):+.4f} ({wins}/{len(deltas)})",
                       dx=-4, dy=-2.5, ha="right", fontsize=6, color=color)
    middle.axvspan(-floor, floor, color="#999999", alpha=0.4, zorder=1)
    middle.axvline(0, color="#333333", linewidth=0.8, zorder=2)
    middle.set_yticks(range(len(others)))
    middle.set_yticklabels([LABELS.get(name, name) for name in others])
    middle.invert_yaxis()
    middle.set_xlabel(f"paired $\\Delta$ vs {LABELS.get(best, best)}")
    widest = min(min(losses[best][i] - losses[name][i]
                     for i in range(len(seeds))) for name in others)
    middle.set_xlim(widest * 1.85, abs(widest) * 0.10)
    # The noise band is drawn but is thinner than the zero line at this scale,
    # which is the point worth making in words rather than pixels.
    middle.annotate(f"noise floor $\\pm${floor:.4f}\n(narrower than the axis line)",
                    (0.03, 0.06), xycoords="axes fraction", fontsize=6,
                    color="#555555")
    grid(middle, axis="x")

    # (c) loss against wall-clock ---------------------------------------
    # The three spectral optimizers sit within 4 s and 0.04 nats of each other,
    # so inline labels collide; a legend keeps the crowded corner readable.
    for name in order:
        right.scatter([timing[name]], [np.mean(losses[name])], s=34,
                      color=COLORS.get(name, "#333"), zorder=4, linewidths=0,
                      marker=MARKERS.get(name, "o"),
                      label=LABELS.get(name, name))
    right.set_xlabel("seconds per run")
    right.set_ylabel("validation loss")
    right.set_xlim(min(timing.values()) - 45, max(timing.values()) + 55)
    span = max(np.mean(losses[n]) for n in order) - min(np.mean(losses[n]) for n in order)
    right.set_ylim(min(np.mean(losses[n]) for n in order) - 0.09 * span,
                   max(np.mean(losses[n]) for n in order) + 0.09 * span)
    right.legend(loc="upper right", fontsize=6, handletextpad=0.3)
    right.annotate("cheaper and better\n$\\leftarrow$ this corner",
                   (0.03, 0.06), xycoords="axes fraction", fontsize=6,
                   color="#666666")
    grid(right)

    label_panels(axes)
    fig.tight_layout(pad=0.4, w_pad=1.5)
    saved = save(fig, "fig4_results")
    write_data("fig4_results", {"paired_vs_" + best: stats,
                                "noise_floor": floor,
                                "untuned": block["untuned"],
                                "seeds": seeds})

    print(f"  wrote {saved}")
    print(f"    reference: {best} (mean {np.mean(losses[best]):.4f})")
    for name, values in stats.items():
        flag = "" if abs(values["mean"]) > 2 * floor else "   <- inside noise"
        print(f"    vs {name:16s} {values['mean']:+.4f}  worst {values['worst']:+.4f}  "
              f"{values['won']}/{values['of']}  exact p = {values['exact_p']:.3f}{flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="measured.json")
    args = parser.parse_args()
    build(args.data)


if __name__ == "__main__":
    main()
