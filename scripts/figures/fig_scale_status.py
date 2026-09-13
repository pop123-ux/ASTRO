"""Paper figure shell for model-scale evidence.

The only paper-eligible quantitative point today is the complete 124M/900
shared-grid result. 355M and 774M remain deliberately blank until clean
same-fingerprint measurements exist. The script is designed so future scale
results can be inserted without restyling the paper.
"""

from __future__ import annotations

import numpy as np

from style import COLORS, column_figure, grid, missing, read_data, save, write_data


def build(source: str = "paper_results.json") -> None:
    payload = read_data(source)
    if payload is None:
        missing("fig_scale_status", "add artifacts/paper_results.json")
        return

    horizon = payload["horizon_124m_v2"]["steps"]["900"]
    status = payload.get("scale_status", {})

    sizes = [124, 355, 774]
    labels = ["124M", "355M", "774M"]
    values = [float(horizon["mean_delta"]), None, None]

    fig, ax = column_figure(height=2.34)
    x = np.arange(len(sizes), dtype=float)

    ax.scatter(x[0], values[0], s=48, marker="D", color=COLORS["astro_v2"],
               edgecolor="white", linewidth=0.6, zorder=4)
    ax.annotate(f"{values[0]:+.3f}", (x[0], values[0]), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=6.1, color=COLORS["astro_v2"])

    for i in (1, 2):
        ax.scatter(x[i], -0.145, s=42, marker="D", facecolor="white",
                   edgecolor=COLORS["pending"], linewidth=1.0, zorder=3)
        note = "clean rerun" if i == 1 else "not started"
        ax.text(x[i], -0.159, note, ha="center", va="top", fontsize=5.9,
                color="#777777")

    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"mean paired $\Delta$ loss vs Muon ↓")
    ax.set_xlabel("model size")
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(-0.185, 0.055)
    grid(ax, axis="y")

    ax.text(0.02, 0.965, "Scale study · paper-eligible evidence only",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.05,
            fontweight="bold", color="#333333")
    ax.text(0.02, 0.035, "open markers are experiment slots, not estimates",
            transform=ax.transAxes, fontsize=5.8, color="#666666")

    fig.tight_layout(pad=0.35)
    saved = save(fig, "fig_scale_status")
    write_data("fig_scale_status", {
        "sizes_millions": sizes,
        "mean_delta_vs_muon": values,
        "scale_status": status,
    })
    print(f"  wrote {saved}")


if __name__ == "__main__":
    build()
