"""Paper figure: durability of the frozen ASTRO-v2 advantage.

The 900-step point comes from the five-seed main experiment.  The 2700-step
point is a two-seed frozen-configuration check whose purpose is only to test
whether the effect obviously collapses with longer training.
"""

from __future__ import annotations

import numpy as np

from paper_style import COLORS, column_figure, grid, missing, read_data, save, write_data


def _stage_delta(stage: dict) -> tuple[float, list[float]]:
    muon = stage["optimizers"]["muon"]
    astro = stage["optimizers"]["astro_v2"]
    m = {s: v for s, v in zip(muon["seeds"], muon["loss"])}
    a = {s: v for s, v in zip(astro["seeds"], astro["loss"])}
    seeds = sorted(set(m) & set(a))
    values = [float(a[s] - m[s]) for s in seeds]
    return float(np.mean(values)), values


def build(source: str = "paper_campaign.json") -> None:
    payload = read_data(source)
    if payload is None:
        missing("fig_paper_horizon", "python scripts/paper_collect.py ...")
        return
    main = payload.get("main_124m_900", {})
    horizon = payload.get("horizon_124m_2700", {})
    if not main.get("complete") or not horizon.get("complete"):
        missing("fig_paper_horizon", "finish the main run and the 124M/2700 frozen-horizon stage")
        return

    d900 = float(main["optimizers"]["astro_v2"]["paired_vs_muon"]["mean_delta"])
    d2700, d2700_all = _stage_delta(horizon)
    x = np.asarray([900, 2700], dtype=float)
    y = np.asarray([d900, d2700], dtype=float)

    fig, ax = column_figure(height=2.32)
    ax.axhline(0, color="#555555", linestyle="--", linewidth=0.8, zorder=1)
    ax.plot(x, y, color=COLORS["astro_v2"], linewidth=1.5, zorder=2)
    ax.scatter(x, y, s=47, marker="D", color=COLORS["astro_v2"],
               edgecolor="white", linewidth=0.55, zorder=3)
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:+.3f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=6.1,
                    color=COLORS["astro_v2"])

    ax.set_xticks(x)
    ax.set_xticklabels(["900\n5 seeds", "2700\n2 seeds"])
    ax.set_xlabel("training steps")
    ax.set_ylabel(r"ASTRO-v2 $-$ Muon validation loss ↓")
    grid(ax, axis="y")
    ax.set_title("Frozen-recipe long-horizon check · 124M",
                 loc="left", pad=7, fontsize=7.5, fontweight="bold")
    ax.text(0.02, 0.035, "negative = ASTRO-v2 lower loss",
            transform=ax.transAxes, fontsize=5.7, color="#666666")
    lo, hi = float(y.min()), float(y.max())
    pad = max(0.025, 0.25 * (hi - lo if hi > lo else abs(lo) + 0.01))
    ax.set_ylim(min(lo - pad, -0.02), max(hi + pad, 0.02))
    fig.tight_layout(pad=0.38)
    saved = save(fig, "fig_paper_horizon")
    write_data("fig_paper_horizon", {
        "delta_900": d900,
        "delta_2700": d2700,
        "delta_2700_per_seed": d2700_all,
    })
    print(f"  wrote {saved}")


if __name__ == "__main__":
    build()
