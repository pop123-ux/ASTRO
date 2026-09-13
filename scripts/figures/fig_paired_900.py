"""Paper figure: all five paired 124M/900 shared configurations.

A horizontal dumbbell plot makes the comparison legible without collapsing the
search into one selected winner. Each row is one shared hyperparameter setting;
left/right endpoints are Muon and ASTRO-v2 and the annotation is the paired
validation-loss difference.
"""

from __future__ import annotations

import numpy as np

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
                 loc="left", pad=17, fontsize=7.7, fontweight="bold")
    ax.text(0.0, 1.018,
            f"mean Δ {mean_delta:+.3f} · {wins}/5 favor ASTRO-v2 · paired settings, not seeds",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=5.8,
            color="#666666")

    # Direct labels on the first pair keep the plot self-explanatory without a
    # legend competing for the same small amount of paper real estate.
    first = rows[0]
    ax.annotate("ASTRO-v2", (float(first["astro_v2"]), y[0]),
                textcoords="offset points", xytext=(0, -10), ha="center",
                va="top", fontsize=5.7, color=COLORS["astro_v2"])
    ax.annotate("Muon", (float(first["muon"]), y[0]),
                textcoords="offset points", xytext=(0, -10), ha="center",
                va="top", fontsize=5.7, color=COLORS["muon"])

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
