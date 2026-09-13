"""Paper figure: ASTRO-v2 horizon evidence through 900 steps.

Each small point is one shared hyperparameter configuration and each diamond is
the mean paired delta (ASTRO-v2 minus Muon). Complete cells are connected with
a restrained line; unfinished horizons are rendered as open pending markers so
missing evidence can never look like an extrapolated trend.
"""

from __future__ import annotations

import numpy as np

from style import COLORS, column_figure, grid, missing, read_data, save, write_data


def build(source: str = "paper_results.json") -> None:
    payload = read_data(source)
    if payload is None:
        missing("fig_horizon_v2", "add artifacts/paper_results.json")
        return

    block = payload["horizon_124m_v2"]["steps"]
    labels = ["300", "600", "900", "2700"]
    xpos = np.arange(len(labels), dtype=float)

    fig, ax = column_figure(height=2.42)
    means_x: list[float] = []
    means_y: list[float] = []
    plotted: dict[str, list[float]] = {}

    for x, step in zip(xpos, labels):
        entry = block[step]
        deltas = [float(v) for v in entry.get("deltas_vs_muon", [])]
        plotted[step] = deltas

        if not deltas:
            # A deliberately visible hole in the evidence rather than a fake
            # trend line. The open diamond uses the same visual language as a
            # measured mean but is neutral grey and labelled pending.
            ax.scatter(x, -0.305, s=42, marker="D", facecolor="white",
                       edgecolor=COLORS["pending"], linewidth=1.0, zorder=4)
            ax.text(x, -0.326, "pending", ha="center", va="top",
                    fontsize=5.9, color="#777777")
            continue

        jitter = np.linspace(-0.075, 0.075, len(deltas)) if len(deltas) > 1 else np.array([0.0])
        ax.scatter(x + jitter, deltas, s=16, color=COLORS["astro_v2"],
                   alpha=0.55, linewidths=0, zorder=3)

        mean = float(np.mean(deltas))
        if entry.get("complete"):
            means_x.append(x)
            means_y.append(mean)
            ax.scatter(x, mean, s=46, marker="D", color=COLORS["astro_v2"],
                       edgecolor="white", linewidth=0.6, zorder=5)
            ax.annotate(f"{mean:+.3f}", (x, mean), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=6.0,
                        color=COLORS["astro_v2"])
        else:
            ax.scatter(x, mean, s=46, marker="D", facecolor="white",
                       edgecolor=COLORS["astro_v2"], linewidth=1.1, zorder=5)
            ax.annotate(f"partial {len(deltas)}/5", (x, mean),
                        textcoords="offset points", xytext=(0, 8), ha="center",
                        fontsize=5.9, color="#666666")

    if len(means_x) >= 2:
        ax.plot(means_x, means_y, color=COLORS["astro_v2"], linewidth=1.5,
                zorder=2)

    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=0.8, zorder=1)
    ax.text(0.985, 0.565, "Muon tie", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.9, color="#555555")

    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_xlabel("training steps")
    ax.set_ylabel(r"ASTRO-v2 $-$ Muon validation loss ↓")
    ax.set_xlim(-0.35, 3.35)
    ax.set_ylim(-0.35, 0.105)
    grid(ax, axis="y")

    ax.text(0.02, 0.965, "124M · five shared configurations per completed horizon",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.05,
            fontweight="bold", color="#333333")
    ax.text(0.02, 0.035, "small points = configurations   ◆ = mean",
            transform=ax.transAxes, fontsize=5.8, color="#666666")

    fig.tight_layout(pad=0.35)
    saved = save(fig, "fig_horizon_v2")
    write_data("fig_horizon_v2", {
        "deltas_vs_muon": plotted,
        "complete": {k: bool(v.get("complete")) for k, v in block.items()},
        "means": {k: (float(np.mean(v)) if v else None) for k, v in plotted.items()},
    })
    print(f"  wrote {saved}")


if __name__ == "__main__":
    build()
