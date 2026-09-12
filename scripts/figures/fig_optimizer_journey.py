"""Paper figure: ASTRO's optimizer journey against modern Muon baselines.

Panel (a) is a pointwise 124M/900 comparison at one shared configuration.
Panel (b) shows how the current ASTRO variants behave across the three shared
configurations in the targeted mechanism run.  This is intentionally not a
leaderboard: it documents what changed, what survived, and where robustness
rather than peak loss appears to carry the signal.

Reads ``artifacts/paper_results.json`` and writes PNG, vector PDF, and the exact
numbers plotted to ``artifacts/figures``.
"""

from __future__ import annotations

import numpy as np

from style import (
    COLORS,
    LABELS,
    annotate_value,
    figure,
    grid,
    label_panels,
    missing,
    read_data,
    reference_line,
    save,
    write_data,
)


def build(source: str = "paper_results.json") -> None:
    payload = read_data(source)
    if payload is None:
        missing("fig_optimizer_journey", "add artifacts/paper_results.json")
        return

    shared = payload["shared_config_124m_900"]
    targeted = payload["targeted_mechanism_124m_900"]

    fig, axes = figure(2, height=2.45)
    left, right = axes

    # (a) One shared configuration: show the entire modern Muon family tested.
    order = ["muon", "normuon", "adamuon", "astro_muon_betas", "astro_v2"]
    losses = shared["loss"]
    x = np.arange(len(order))
    for i, name in enumerate(order):
        left.scatter(i, losses[name], s=45, color=COLORS[name],
                     marker={"muon": "s", "normuon": "^", "adamuon": "v",
                             "astro_muon_betas": "X", "astro_v2": "D"}[name],
                     linewidths=0, zorder=4)
        annotate_value(left, i, losses[name], f"{losses[name]:.4f}", dy=6,
                       color=COLORS[name], fontsize=6.4)
    left.set_xticks(x)
    left.set_xticklabels([LABELS[n] for n in order], rotation=24, ha="right")
    left.set_ylabel("validation loss ↓")
    left.set_title("124M / 900 steps / one shared configuration", fontsize=8.2)
    left.set_ylim(min(losses[n] for n in order) - 0.018,
                  max(losses[n] for n in order) + 0.022)
    grid(left, axis="y")
    left.annotate("pointwise comparison — not a tuned global ranking",
                  (0.02, 0.04), xycoords="axes fraction", fontsize=6.2,
                  color="#666666")

    # (b) Targeted mechanism run: delta against Muon across three settings.
    configs = targeted["configs"]
    variants = ["astro_muon_betas", "astro_v2", "astro_v2_gamma0"]
    config_x = np.arange(1, len(configs) + 1)
    plotted = {}
    for name in variants:
        deltas = [cfg["loss"][name] - cfg["loss"]["muon"] for cfg in configs]
        plotted[name] = deltas
        right.plot(config_x, deltas, marker="o", markersize=4.2,
                   color=COLORS[name], label=LABELS[name], zorder=3)
        right.scatter(config_x, deltas, s=18, color=COLORS[name], zorder=4,
                      linewidths=0)
    reference_line(right, 0.0, "Muon", axis="y")
    right.set_xticks(config_x)
    right.set_xticklabels([f"config {i}" for i in config_x])
    right.set_ylabel(r"paired $\Delta$ validation loss vs Muon ↓")
    right.set_title("Targeted mechanism run: three shared settings", fontsize=8.2)
    right.legend(loc="lower left", fontsize=6.5)
    grid(right, axis="y")
    right.annotate("largest separation occurs at the aggressive-LR setting",
                   (0.98, 0.93), xycoords="axes fraction", ha="right", va="top",
                   fontsize=6.2, color="#666666")

    label_panels(axes)
    fig.tight_layout(pad=0.45, w_pad=1.6)
    saved = save(fig, "fig_optimizer_journey")
    write_data("fig_optimizer_journey", {
        "shared_config": shared["config"],
        "shared_config_loss": {name: losses[name] for name in order},
        "targeted_delta_vs_muon": plotted,
        "targeted_mean_delta_vs_muon": targeted["mean_delta_vs_muon"],
        "caveat": "shared configurations are not independent seeds",
    })
    print(f"  wrote {saved}")


if __name__ == "__main__":
    build()
