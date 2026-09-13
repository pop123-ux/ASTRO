"""Paper figure: all five paired 124M/900 shared configurations.

A horizontal dumbbell plot makes the comparison legible without collapsing the
search into one selected winner. Each row is one shared hyperparameter setting;
left/right endpoints are Muon and ASTRO-v2 and the annotation is the paired
validation-loss difference.
"""

from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D

from style import COLORS, column_figure, grid, missing, read_data, save, write_data


def build(source: str = "paper_results.json") -> None:
    payload = read_data(source)
    if payload is None:
        missing("fig_paired_900", "add artifacts/paper_results.json")
        return

    rows = payload["horizon_124m_v2"]["steps"]["900"].get("loss_by_config", [])
    if len(rows) != 5:
        missing("fig_paired_900", "complete the five 124M/900 shared configurations")
        return

    fig, ax = column_figure(height=2.58)
    y = np.arange(len(rows), 0, -1)

    for yi, row in zip(y, rows):
        muon = float(row["muon"])
        astro = float(row["astro_v2"])
        delta = float(row["delta"])
        ax.plot([muon, astro], [yi, yi], color="#D2D2D2", linewidth=1.35,
                solid_capstyle="round", zorder=1)
        ax.scatter(muon, yi, s=31, marker="s", color=COLORS["muon"],
                   edgecolor="white", linewidth=0.5, zorder=3)
        ax.scatter(astro, yi, s=38, marker="D", color=COLORS["astro_v2"],
                   edgecolor="white", linewidth=0.5, zorder=4)
        right = max(muon, astro)
        ax.annotate(f"{delta:+.3f}", (right, yi), textcoords="offset points",
                    xytext=(6, 0), ha="left", va="center", fontsize=5.9,
                    color=COLORS["astro_v2"] if delta < 0 else "#666666")

    mean_delta = float(np.mean([float(row["delta"]) for row in rows]))
    wins = sum(float(row["delta"]) < 0 for row in rows)

    ax.set_yticks(y)
    ax.set_yticklabels([f"config {row['config']}" for row in rows])
    ax.set_xlabel("validation loss ↓")
    ax.set_xlim(5.77, 6.205)
    ax.set_ylim(0.55, 5.55)
    grid(ax, axis="x")

    ax.set_title("Five shared configurations · 124M / 900 steps",
                 loc="left", pad=7, fontsize=7.7, fontweight="bold")

    handles = [
        Line2D([0], [0], marker="s", linestyle="None", markersize=5.2,
               markerfacecolor=COLORS["muon"], markeredgecolor="white", label="Muon"),
        Line2D([0], [0], marker="D", linestyle="None", markersize=5.4,
               markerfacecolor=COLORS["astro_v2"], markeredgecolor="white", label="ASTRO-v2"),
    ]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.995),
              ncol=1, fontsize=6.0, handlelength=0.8, borderaxespad=0.2)

    ax.text(0.98, 0.055,
            f"mean Δ {mean_delta:+.3f}   ·   {wins}/5 favor ASTRO-v2",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.9,
            color=COLORS["astro_v2"])
    ax.text(0.02, 0.055, "paired settings, not independent seeds",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=5.7,
            color="#777777")

    fig.tight_layout(pad=0.42)
    saved = save(fig, "fig_paired_900")
    write_data("fig_paired_900", {
        "rows": rows,
        "mean_delta": mean_delta,
        "wins": f"{wins}/5",
        "note": "paired configurations, not independent seeds",
    })
    print(f"  wrote {saved}")


if __name__ == "__main__":
    build()
