"""Paper figure: ASTRO's optimizer journey against modern Muon baselines.

Panel (a) is a pointwise 124M/900 comparison at one shared configuration.
Panel (b) shows how the current ASTRO variants behave across the three shared
configurations in the targeted mechanism run. This is intentionally not a
leaderboard: it documents what changed, what survived, and where robustness
rather than peak loss appears to carry the signal.
"""

from __future__ import annotations

import numpy as np

from style import COLORS, LABELS, annotate_value, figure, grid, missing, read_data, save, write_data


def panel_header(ax, text: str) -> None:
    ax.text(0.0, 1.035, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.5, fontweight="bold", color="#222222")


def build(source: str = "paper_results.json") -> None:
    payload = read_data(source)
    if payload is None:
        missing("fig_optimizer_journey", "add artifacts/paper_results.json")
        return

    shared = payload["shared_config_124m_900"]
    targeted = payload["targeted_mechanism_124m_900"]

    fig, axes = figure(2, height=2.30)
    left, right = axes

    # (a) One shared configuration: the contemporary baselines plus ASTRO-v2.
    order = ["muon", "normuon", "adamuon", "astro_muon_betas", "astro_v2"]
    losses = shared["loss"]
    x = np.arange(len(order))
    markers = {"muon": "s", "normuon": "^", "adamuon": "v",
               "astro_muon_betas": "X", "astro_v2": "D"}
    for i, name in enumerate(order):
        left.scatter(i, losses[name], s=38, color=COLORS[name], marker=markers[name],
                     linewidths=0, zorder=4)
        annotate_value(left, i, losses[name], f"{losses[name]:.4f}", dy=6,
                       color=COLORS[name], fontsize=6.2)
    left.set_xticks(x)
    left.set_xticklabels([LABELS[n] for n in order], rotation=19, ha="right")
    left.set_ylabel("validation loss ↓")
    left.set_ylim(min(losses[n] for n in order) - 0.016,
                  max(losses[n] for n in order) + 0.020)
    grid(left, axis="y")
    panel_header(left, "(a)  One shared 124M / 900-step setting")
    left.text(0.02, 0.035, "pointwise comparison; not a tuned ranking",
              transform=left.transAxes, fontsize=6.0, color="#6A6A6A")

    # (b) Targeted mechanism run: delta against Muon across three settings.
    configs = targeted["configs"]
    variants = ["astro_muon_betas", "astro_v2", "astro_v2_gamma0"]
    config_x = np.arange(1, len(configs) + 1)
    plotted = {}
    for name in variants:
        deltas = [cfg["loss"][name] - cfg["loss"]["muon"] for cfg in configs]
        plotted[name] = deltas
        right.plot(config_x, deltas, marker="o", markersize=3.8,
                   color=COLORS[name], label=LABELS[name], zorder=3)

    right.axhline(0.0, color="#555555", linestyle="--", linewidth=0.8, zorder=1)
    right.text(0.98, 0.985, "Muon = 0", transform=right.transAxes, ha="right",
               va="top", fontsize=6.0, color="#555555")
    right.set_xticks(config_x)
    right.set_xticklabels([
        r"1  ($\eta=.0102$)",
        r"2  ($\eta=.0505$)",
        r"3  ($\eta=.00845$)",
    ])
    right.set_ylabel(r"paired $\Delta$ validation loss ↓")
    right.legend(loc="lower left", fontsize=6.1, handlelength=1.35)
    grid(right, axis="y")
    panel_header(right, "(b)  Shared-setting robustness")
    right.annotate("aggressive LR", (2, min(plotted["astro_v2"])),
                   textcoords="offset points", xytext=(0, -13), ha="center",
                   fontsize=5.9, color="#666666")

    fig.tight_layout(pad=0.5, w_pad=1.45)
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
