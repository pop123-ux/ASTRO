"""Paper figure: all five paired 124M/900 shared configurations.

A horizontal dumbbell plot makes the comparison legible without collapsing the
search into one selected winner. Each row is one shared hyperparameter setting;
left/right endpoints are Muon and ASTRO-v2 and the annotation is the paired
validation-loss difference.
"""

from __future__ import annotations

import numpy as np

from style import COLORS, LABELS, column_figure, grid, missing, read_data, save, write_data


def build(source: str = "paper_results.json") -> None:
    payload = read_data(source)
    if payload is None:
        missing("fig_paired_900", "add artifacts/paper_results.json")
        return

    rows = payload["horizon_124m_v2"]["steps"]["900"].get("loss_by_config", [])
    if len(rows) != 5:
        missing("fig_paired_900", "complete the five 124M/900 shared configurations")
        return

    fig, ax = column_figure(height=2.55)
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
        ax.annotate(f"Δ {delta:+.3f}", (right, yi), textcoords="offset points",
                    xytext=(6, 0), ha="left", va="center", fontsize=5.9,
                    color=COLORS["astro_v2"] if delta < 0 else "#666666")

    ax.set_yticks(y)
    ax.set_yticklabels([f"config {row['config']}" for row in rows])
    ax.set_xlabel("validation loss ↓")
    ax.set_xlim(5.77, 6.205)
    grid(ax, axis="x")

    ax.text(0.02, 0.965, "124M · 900 steps · paired shared settings",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.05,
            fontweight="bold", color="#333333")
    ax.text(0.02, 0.035, "■ Muon      ◆ ASTRO-v2",
            transform=ax.transAxes, fontsize=5.9, color="#555555")

    mean_delta = float(np.mean([float(row["delta"]) for row in rows]))
    wins = sum(float(row["delta"]) < 0 for row in rows)
    ax.text(0.98, 0.035, f"mean Δ {mean_delta:+.3f} · {wins}/5 favor ASTRO-v2",
            transform=ax.transAxes, ha="right", fontsize=5.9,
            color=COLORS["astro_v2"])

    fig.tight_layout(pad=0.35)
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
