"""Paper figure: validation loss versus wall-clock for the targeted 124M run.

The figure exists to prevent a step-only optimizer claim.  Each configuration is
shown as a thin connector from Muon to the ASTRO variants.  Lower-left is better:
lower validation loss at lower wall-clock cost.
"""

from __future__ import annotations

import numpy as np

from style import COLORS, LABELS, column_figure, grid, missing, read_data, save, write_data


def build(source: str = "paper_results.json") -> None:
    payload = read_data(source)
    if payload is None:
        missing("fig_efficiency_v2", "add artifacts/paper_results.json")
        return

    configs = payload["targeted_mechanism_124m_900"]["configs"]
    names = ["muon", "astro_muon_betas", "astro_v2", "astro_v2_gamma0"]

    fig, ax = column_figure(height=2.55)
    points = {name: [] for name in names}

    for index, cfg in enumerate(configs):
        # Connect the same hyperparameter configuration with a quiet grey line.
        xs = [cfg["seconds"][name] for name in names]
        ys = [cfg["loss"][name] for name in names]
        ax.plot(xs, ys, color="#D0D0D0", linewidth=0.75, zorder=1)
        for name in names:
            x = cfg["seconds"][name]
            y = cfg["loss"][name]
            points[name].append({"config": index + 1, "seconds": x, "loss": y})
            ax.scatter(x, y, s=34 if name != "muon" else 30,
                       color=COLORS[name], marker="D" if name == "astro_v2" else "o",
                       linewidths=0, zorder=3,
                       label=LABELS[name] if index == 0 else None)

    ax.set_xlabel("wall-clock seconds / 900-step run")
    ax.set_ylabel("validation loss ↓")
    ax.set_title("Step quality versus measured compute cost", fontsize=8.2)
    grid(ax)
    ax.legend(loc="upper right", fontsize=6.3)
    ax.annotate("better", (0.04, 0.05), xycoords="axes fraction", fontsize=6.3,
                color="#666666")
    ax.annotate("↙", (0.105, 0.055), xycoords="axes fraction", fontsize=9,
                color="#666666")

    # Report mean runtime overhead directly on the panel.
    muon_t = np.mean([cfg["seconds"]["muon"] for cfg in configs])
    v2_t = np.mean([cfg["seconds"]["astro_v2"] for cfg in configs])
    overhead = v2_t / muon_t - 1.0
    ax.text(0.03, 0.95, f"ASTRO-v2 mean runtime overhead: {overhead*100:.1f}%",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.4,
            color=COLORS["astro_v2"])

    fig.tight_layout(pad=0.4)
    saved = save(fig, "fig_efficiency_v2")
    write_data("fig_efficiency_v2", {
        "points": points,
        "astro_v2_runtime_overhead": float(overhead),
        "note": "same-step wall-clock, not yet a full time-matched training comparison",
    })
    print(f"  wrote {saved}")


if __name__ == "__main__":
    build()
