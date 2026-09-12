"""Paper figure: current ASTRO-v2 horizon evidence.

Every point is a shared-configuration paired delta (ASTRO-v2 minus Muon).  The
figure makes incomplete cells visually explicit: 300 and 600 have all five
planned configurations, 900 currently has only two, and 2700 is left blank
until measured.  A paper draft must never turn missing trials into a smooth
trend by accident.
"""

from __future__ import annotations

import numpy as np

from style import COLORS, annotate_value, column_figure, grid, missing, read_data, save, write_data


def build(source: str = "paper_results.json") -> None:
    payload = read_data(source)
    if payload is None:
        missing("fig_horizon_v2", "add artifacts/paper_results.json")
        return

    block = payload["horizon_124m_v2"]["steps"]
    fig, ax = column_figure(height=2.45)

    full_steps = []
    full_means = []
    partial_steps = []
    partial_means = []
    plotted = {}

    for step_text in sorted(block, key=int):
        step = int(step_text)
        entry = block[step_text]
        deltas = entry.get("deltas_vs_muon", [])
        plotted[step_text] = deltas
        if not deltas:
            continue

        jitter = np.linspace(-0.035, 0.035, len(deltas)) if len(deltas) > 1 else np.array([0.0])
        ax.scatter(np.full(len(deltas), step, dtype=float) * (1 + jitter), deltas,
                   s=18, color=COLORS["astro_v2"], alpha=0.72, linewidths=0, zorder=3)
        mean = float(np.mean(deltas))
        if entry.get("complete"):
            full_steps.append(step)
            full_means.append(mean)
            ax.scatter(step, mean, s=48, marker="D", color=COLORS["astro_v2"],
                       edgecolor="white", linewidth=0.6, zorder=5)
        else:
            partial_steps.append(step)
            partial_means.append(mean)
            ax.scatter(step, mean, s=48, marker="D", facecolor="white",
                       edgecolor=COLORS["astro_v2"], linewidth=1.2, zorder=5)
            ax.annotate(f"partial {len(deltas)}/5", (step, mean),
                        textcoords="offset points", xytext=(5, 7), fontsize=6.2,
                        color="#666666")

    if len(full_steps) >= 2:
        ax.plot(full_steps, full_means, color=COLORS["astro_v2"], linewidth=1.6, zorder=4)
    if full_steps and partial_steps:
        ax.plot([full_steps[-1], partial_steps[0]], [full_means[-1], partial_means[0]],
                color=COLORS["astro_v2"], linewidth=1.0, linestyle="--", alpha=0.65)

    ax.axhline(0.0, color="#4D4D4D", linestyle="--", linewidth=0.8, zorder=1)
    ax.text(0.99, 0.51, "Muon tie", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.2, color="#555555")
    ax.set_xscale("log")
    ax.set_xticks([300, 600, 900, 2700])
    ax.set_xticklabels(["300", "600", "900", "2700"])
    ax.set_xlabel("training steps")
    ax.set_ylabel(r"ASTRO-v2 $-$ Muon validation loss ↓")
    ax.set_title("Horizon study — filled diamonds are complete cells", fontsize=8.2)
    grid(ax, axis="y")

    if not block["2700"].get("deltas_vs_muon"):
        ymin, ymax = ax.get_ylim()
        ax.annotate("2700 pending", (2700, ymin + 0.08 * (ymax - ymin)),
                    ha="center", va="bottom", fontsize=6.3, color="#777777")
    ax.annotate("negative favors ASTRO-v2", (0.02, 0.04), xycoords="axes fraction",
                fontsize=6.2, color="#666666")

    fig.tight_layout(pad=0.4)
    saved = save(fig, "fig_horizon_v2")
    write_data("fig_horizon_v2", {
        "deltas_vs_muon": plotted,
        "complete": {k: bool(v.get("complete")) for k, v in block.items()},
    })
    print(f"  wrote {saved}")


if __name__ == "__main__":
    build()
