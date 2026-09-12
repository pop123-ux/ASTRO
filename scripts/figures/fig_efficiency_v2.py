"""Paper figure: validation loss versus wall-clock for the targeted 124M run.

Each thin grey connector joins optimizers evaluated at the same hyperparameter
configuration. Lower-left is better: lower validation loss at lower wall-clock
cost. This is same-step timing, not yet a full time-matched training result.
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

    fig, ax = column_figure(height=2.40)
    points = {name: [] for name in names}

    for index, cfg in enumerate(configs):
        xs = [cfg["seconds"][name] for name in names]
        ys = [cfg["loss"][name] for name in names]
        ax.plot(xs, ys, color="#CECECE", linewidth=0.72, zorder=1)
        for name in names:
            x = cfg["seconds"][name]
            y = cfg["loss"][name]
            points[name].append({"config": index + 1, "seconds": x, "loss": y})
            ax.scatter(x, y, s=29 if name != "astro_v2" else 36,
                       color=COLORS[name], marker="D" if name == "astro_v2" else "o",
                       linewidths=0, zorder=3,
                       label=LABELS[name] if index == 0 else None)

        # Identify the shared setting once, at the Muon endpoint of each pair.
        ax.annotate(str(index + 1),
                    (cfg["seconds"]["muon"], cfg["loss"]["muon"]),
                    textcoords="offset points", xytext=(-7, 5), fontsize=5.7,
                    color="#777777")

    ax.set_xlabel("wall-clock seconds / 900-step run")
    ax.set_ylabel("validation loss ↓")
    grid(ax)
    ax.legend(loc="upper right", fontsize=5.9, handlelength=1.15, labelspacing=0.35)

    muon_t = np.mean([cfg["seconds"]["muon"] for cfg in configs])
    v2_t = np.mean([cfg["seconds"]["astro_v2"] for cfg in configs])
    overhead = v2_t / muon_t - 1.0

    ax.text(0.02, 0.975, "124M · 900 steps · shared settings",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.1,
            fontweight="bold", color="#333333")
    ax.text(0.98, 0.56, f"ASTRO-v2: +{overhead*100:.1f}% mean runtime",
            transform=ax.transAxes, ha="right", va="center", fontsize=6.0,
            color=COLORS["astro_v2"])
    ax.text(0.02, 0.035, "lower-left is better",
            transform=ax.transAxes, fontsize=5.9, color="#666666")

    fig.tight_layout(pad=0.35)
    saved = save(fig, "fig_efficiency_v2")
    write_data("fig_efficiency_v2", {
        "points": points,
        "astro_v2_runtime_overhead": float(overhead),
        "note": "same-step wall-clock, not yet a full time-matched training comparison",
    })
    print(f"  wrote {saved}")


if __name__ == "__main__":
    build()
