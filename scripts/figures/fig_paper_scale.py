"""Paper figure: frozen-recipe transfer from 124M to 355M.

This is deliberately a transfer test, not a retuned 355M leaderboard.  The
124M-selected configurations are frozen before the larger model is run.
"""

from __future__ import annotations

import numpy as np

from paper_style import COLORS, column_figure, grid, missing, read_data, save, write_data


def _paired_delta(muon: dict, astro: dict) -> tuple[float, list[float]]:
    m = {s: v for s, v in zip(muon["seeds"], muon["loss"])}
    a = {s: v for s, v in zip(astro["seeds"], astro["loss"])}
    common = sorted(set(m) & set(a))
    deltas = [float(a[s] - m[s]) for s in common]
    return float(np.mean(deltas)), deltas


def build(source: str = "paper_campaign.json") -> None:
    payload = read_data(source)
    if payload is None:
        missing("fig_paper_scale", "python scripts/paper_collect.py ...")
        return
    main = payload.get("main_124m_900", {})
    scale = payload.get("scale_355m_900", {})
    if not main.get("complete") or not scale.get("complete"):
        missing("fig_paper_scale", "finish the clean 124M main run and 355M frozen-transfer stage")
        return

    d124 = float(main["optimizers"]["astro_v2"]["paired_vs_muon"]["mean_delta"])
    d355, d355_all = _paired_delta(scale["optimizers"]["muon"], scale["optimizers"]["astro_v2"])
    deltas = [d124, d355]

    fig, ax = column_figure(height=2.35)
    x = np.arange(2, dtype=float)
    ax.axhline(0, color="#555555", linestyle="--", linewidth=0.8, zorder=1)
    ax.plot(x, deltas, color=COLORS["astro_v2"], linewidth=1.45, zorder=2)
    ax.scatter(x, deltas, s=46, marker="D", color=COLORS["astro_v2"],
               edgecolor="white", linewidth=0.55, zorder=3)
    for xi, delta in zip(x, deltas):
        ax.annotate(f"{delta:+.3f}", (xi, delta), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=6.1,
                    color=COLORS["astro_v2"])

    ax.set_xticks(x)
    ax.set_xticklabels(["124M\n5 seeds", "355M\n2 seeds"])
    ax.set_ylabel(r"ASTRO-v2 $-$ Muon validation loss ↓")
    ax.set_xlim(-0.42, 1.42)
    grid(ax, axis="y")
    ax.set_title("Frozen 124M recipe transferred without retuning",
                 loc="left", pad=7, fontsize=7.5, fontweight="bold")
    ax.text(0.02, 0.035, "negative = ASTRO-v2 lower loss",
            transform=ax.transAxes, fontsize=5.7, color="#666666")

    lo, hi = min(deltas), max(deltas)
    pad = max(0.025, 0.25 * (hi - lo if hi > lo else abs(lo) + 0.01))
    ax.set_ylim(min(lo - pad, -0.02), max(hi + pad, 0.02))
    fig.tight_layout(pad=0.38)
    saved = save(fig, "fig_paper_scale")
    write_data("fig_paper_scale", {
        "delta_124m": d124,
        "delta_355m": d355,
        "delta_355m_per_seed": d355_all,
        "note": "355M uses frozen 124M-selected configurations",
    })
    print(f"  wrote {saved}")


if __name__ == "__main__":
    build()
