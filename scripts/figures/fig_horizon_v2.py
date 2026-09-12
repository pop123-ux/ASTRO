"""Paper figure: current ASTRO-v2 horizon evidence.

Every point is a shared-configuration paired delta (ASTRO-v2 minus Muon). The
figure makes incomplete cells visually explicit: 300 and 600 have all five
planned configurations, 900 currently has only two, and 2700 is left blank
until measured.
"""

from __future__ import annotations

import matplotlib.ticker as mticker
import numpy as np

from style import COLORS, column_figure, grid, missing, read_data, save, write_data


def build(source: str = "paper_results.json") -> None:
    payload = read_data(source)
    if payload is None:
        missing("fig_horizon_v2", "add artifacts/paper_results.json")
        return

    block = payload["horizon_124m_v2"]["steps"]
    fig, ax = column_figure(height=2.35)

    full_steps: list[int] = []
    full_means: list[float] = []
    partial_steps: list[int] = []
    partial_means: list[float] = []
    plotted = {}

    for step_text in sorted(block, key=int):
        step = int(step_text)
        entry = block[step_text]
        deltas = entry.get("deltas_vs_muon", [])
        plotted[step_text] = deltas
        if not deltas:
            continue

        # Multiplicative jitter is visually even on the log x-axis.
        jitter = np.linspace(-0.030, 0.030, len(deltas)) if len(deltas) > 1 else np.array([0.0])
        ax.scatter(np.full(len(deltas), step, dtype=float) * (1 + jitter), deltas,
                   s=16, color=COLORS["astro_v2"], alpha=0.62,
                   linewidths=0, zorder=3)
        mean = float(np.mean(deltas))
        if entry.get("complete"):
            full_steps.append(step)
            full_means.append(mean)
            ax.scatter(step, mean, s=42, marker="D", color=COLORS["astro_v2"],
                       edgecolor="white", linewidth=0.55, zorder=5)
        else:
            partial_steps.append(step)
            partial_means.append(mean)
            ax.scatter(step, mean, s=42, marker="D", facecolor="white",
                       edgecolor=COLORS["astro_v2"], linewidth=1.15, zorder=5)
            ax.annotate(f"partial {len(deltas)}/5", (step, mean),
                        textcoords="offset points", xytext=(5, 6), fontsize=5.9,
                        color="#666666")

    if len(full_steps) >= 2:
        ax.plot(full_steps, full_means, color=COLORS["astro_v2"], linewidth=1.5, zorder=4)
    if full_steps and partial_steps:
        ax.plot([full_steps[-1], partial_steps[0]], [full_means[-1], partial_means[0]],
                color=COLORS["astro_v2"], linewidth=0.9, linestyle="--", alpha=0.65)

    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=0.8, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([300, 600, 900, 2700])
    ax.set_xticklabels(["300", "600", "900", "2700"])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("training steps")
    ax.set_ylabel(r"ASTRO-v2 $-$ Muon validation loss ↓")
    grid(ax, axis="y")

    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    ax.annotate("Muon tie", (2700, 0.0), textcoords="offset points",
                xytext=(-4, 5), ha="right", fontsize=5.9, color="#555555")
    if not block["2700"].get("deltas_vs_muon"):
        ax.annotate("pending", (2700, ymin + 0.08 * span),
                    ha="center", va="bottom", fontsize=6.0, color="#777777")
    ax.text(0.02, 0.035, "negative favors ASTRO-v2",
            transform=ax.transAxes, fontsize=5.9, color="#666666")

    fig.tight_layout(pad=0.35)
    saved = save(fig, "fig_horizon_v2")
    write_data("fig_horizon_v2", {
        "deltas_vs_muon": plotted,
        "complete": {k: bool(v.get("complete")) for k, v in block.items()},
    })
    print(f"  wrote {saved}")


if __name__ == "__main__":
    build()
