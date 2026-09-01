"""Figure 6: does the advantage survive longer training and larger models?

This is the figure most likely to falsify the paper's optimizer claim, which is
why it is drawn before the data that would settle it exists. Optimizer margins
are documented to shrink with scale, and every training number we have is at
300 steps or below.

  (a) the measured decay at 806K: the gap against AdamW shrinks with budget,
      and the fitted power law says by how much
  (b) the same question at 124M, from ``astro_lab.py --mode scaling`` -- drawn
      only if that run exists, and labelled as absent if it does not
  (c) the region the paper cannot speak about, stated as a figure rather than
      buried in a limitations paragraph

Reads ``artifacts/measured.json`` and, optionally, ``astro_lab_state.json``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

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

SCALING_COMMAND = ("python scripts/astro_lab.py --mode scaling --size 124M "
                   "--steps 300 900 2700 --optimizers muon astro --seeds 100 101")


def power_law(steps, gaps):
    """Fit |gap| ~ a * steps^b in log space. b < 0 means the margin decays."""
    logs, logg = np.log(np.asarray(steps, float)), np.log(np.abs(np.asarray(gaps, float)))
    slope, intercept = np.polyfit(logs, logg, 1)
    return float(slope), float(np.exp(intercept))


def parse_scaling(state: dict) -> dict[tuple[str, int], dict[str, list[float]]]:
    """Group astro_lab's flat ``size|steps|name|seed`` state into curves."""
    grouped: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for key, value in state.items():
        parts = key.split("|")
        if len(parts) != 4:
            continue
        size, steps, name, _seed = parts
        # astro_lab writes {"value": ..., "seconds": ..., "config": ...}; older
        # states used "loss", and a bare float is accepted too. Reading only
        # "loss" silently produced an empty panel, which looks exactly like
        # "the run has not happened yet".
        if isinstance(value, dict):
            loss = value.get("value", value.get("loss"))
        else:
            loss = value
        if isinstance(loss, (int, float)):
            grouped[(size, int(steps))][name].append(float(loss))
    return grouped


def build(source: str = "measured.json", scaling: str = "astro_lab_state.json") -> None:
    payload = read_data(source)
    if payload is None:
        missing("fig6_horizon", "artifacts/measured.json is missing from the repo")
        return
    horizon = payload["horizon"]
    steps = horizon["steps"]
    gaps = horizon["delta_vs_adamw"]
    slope, scale = power_law(steps, gaps)

    fig, axes = figure(3, height=2.25)
    left, middle, right = axes

    # (a) the measured decay at 806K ------------------------------------
    left.plot(steps, np.abs(gaps), color=COLORS["astro"], marker="D", zorder=4,
              label="measured")
    dense = np.linspace(min(steps), max(steps) * 4, 100)
    left.plot(dense, scale * dense ** slope, color=COLORS["reference"],
              linestyle="--", linewidth=0.9, zorder=3,
              label=rf"fit $\propto N^{{{slope:.2f}}}$")
    left.set_xscale("log")
    left.set_yscale("log")
    # Default log minor ticks collide at this width; label the budgets we ran.
    left.set_xticks(list(steps) + [max(steps) * 4])
    left.set_xticklabels([str(value) for value in steps] + [str(max(steps) * 4)])
    left.minorticks_off()
    left.set_ylim(min(np.abs(gaps)) * 0.35, max(np.abs(gaps)) * 1.9)
    left.set_xlabel("training steps")
    left.set_ylabel(r"$|\Delta|$ vs AdamW (nats)")
    left.legend(loc="lower left", fontsize=6)
    for step, gap in zip(steps, gaps, strict=True):
        annotate_value(left, step, abs(gap), f"{gap:+.3f}", dy=7, fontsize=6)
    left.annotate("806K params\ntinyshakespeare", (0.97, 0.93),
                  xycoords="axes fraction", ha="right", va="top",
                  fontsize=6, color="#666666")
    grid(left)

    # (b) the same question at 124M -------------------------------------
    state = read_data(scaling)
    drawn = False
    scaling_summary: dict[str, float] = {}
    if state:
        # astro_lab nests the run map under "runs" alongside "tuned"; passing
        # the top level made parse_scaling iterate over those two keys and find
        # nothing, which is indistinguishable from "the run has not happened".
        grouped = parse_scaling(state.get("runs", state.get("results", state)))
        budgets = sorted({steps_ for (size, steps_) in grouped if size == "124M"})
        if budgets:
            reference = "muon"
            deltas = []
            for budget in budgets:
                block = grouped[("124M", budget)]
                if reference in block and "astro" in block:
                    deltas.append(np.mean(block["astro"]) - np.mean(block[reference]))
                else:
                    deltas.append(np.nan)
            if np.isfinite(deltas).any():
                middle.plot(budgets, deltas, color=COLORS["astro"], marker="D",
                            zorder=4)
                middle.axhline(0, color="#333333", linewidth=0.9, zorder=3)
                middle.set_xscale("log")
                # Same tick collision as panel (a): label the budgets we ran.
                middle.set_xticks(budgets)
                middle.set_xticklabels([str(value) for value in budgets])
                middle.minorticks_off()
                span = max(deltas) - min(min(deltas), 0.0)
                middle.set_ylim(min(min(deltas), 0.0) - 0.08 * span,
                                max(deltas) + 0.30 * span)
                middle.set_xlabel("training steps")
                middle.set_ylabel(r"$\Delta$ vs Muon (nats)")
                for budget, delta in zip(budgets, deltas, strict=True):
                    if np.isfinite(delta):
                        annotate_value(middle, budget, delta, f"{delta:+.4f}",
                                       dy=7, fontsize=6)
                        scaling_summary[str(budget)] = float(delta)
                middle.annotate("above 0 = ASTRO worse", (0.03, 0.94),
                                xycoords="axes fraction", ha="left", va="top",
                                fontsize=6, color=COLORS["astro"])
                middle.annotate("124M, FineWeb-Edu\nshared rate, not tuned",
                                (0.97, 0.06), xycoords="axes fraction",
                                ha="right", fontsize=6, color="#666666")
                grid(middle)
                drawn = True
    if not drawn:
        middle.set_xticks([])
        middle.set_yticks([])
        for spine in middle.spines.values():
            spine.set_visible(False)
        middle.annotate("not yet measured\n\nrun:\n" + SCALING_COMMAND.replace(" --", "\n  --"),
                        (0.5, 0.5), xycoords="axes fraction", ha="center",
                        va="center", fontsize=5.5, color="#999999",
                        family="monospace")
        middle.set_xlabel("124M horizon")

    # (c) what is and is not covered ------------------------------------
    covered = [("806K", 1600, True), ("1.17M", 400, True),
               ("124M", 300, True), ("124M", 2700, drawn),
               ("355M", 300, False), ("774M", 300, False)]
    spots = np.arange(len(covered))
    right.barh(spots, [np.log10(steps_) for _, steps_, _ in covered], height=0.55,
               color=[COLORS["normuon"] if ok else "#cccccc"
                      for _, _, ok in covered], zorder=3)
    for index, (_size, steps_, ok) in enumerate(covered):
        right.annotate("measured" if ok else "not measured",
                       (np.log10(steps_) - 0.05, index), ha="right", va="center",
                       fontsize=5.5, color="white" if ok else "#666666", zorder=5)
    right.set_yticks(spots)
    right.set_yticklabels([f"{size}, {steps_} steps" for size, steps_, _ in covered],
                          fontsize=6)
    right.invert_yaxis()
    right.set_xlabel(r"$\log_{10}$ steps")
    grid(right, axis="x")

    label_panels(axes)
    fig.tight_layout(pad=0.4, w_pad=1.5)
    saved = save(fig, "fig6_horizon")
    write_data("fig6_horizon", {
        "small_scale": {"steps": steps, "gap_vs_adamw": gaps,
                        "power_law_exponent": slope},
        "scaling_124m": scaling_summary or None,
        "coverage": [{"size": s, "steps": n, "measured": ok}
                     for s, n, ok in covered],
    })

    print(f"  wrote {saved}")
    print(f"    806K gap decays as steps^{slope:.3f} "
          f"({gaps[0]:+.4f} at {steps[0]} -> {gaps[-1]:+.4f} at {steps[-1]})")
    extrapolated = scale * (steps[-1] * 4) ** slope
    print(f"    that fit puts |gap| at {steps[-1] * 4} steps near {extrapolated:.4f} "
          "-- an extrapolation, not a measurement")
    if scaling_summary:
        print(f"    124M horizon: {scaling_summary}")
    else:
        print("    124M horizon: NOT MEASURED -- panel (b) says so on the figure")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="measured.json")
    parser.add_argument("--scaling", default="astro_lab_state.json")
    args = parser.parse_args()
    build(args.data, args.scaling)


if __name__ == "__main__":
    main()
