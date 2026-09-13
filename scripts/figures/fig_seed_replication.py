"""Paper figure: independent-seed replication at 124M/900.

The plot is deliberately useful before the experiment is complete. Measured
Muon seeds render immediately; ASTRO-v2 is shown as a quiet pending column.
Once ASTRO-v2 values are added to paper_results.json, the same script switches
to paired seed connectors automatically.
"""

from __future__ import annotations

import numpy as np

from style import COLORS, column_figure, grid, missing, read_data, save, write_data


def build(source: str = "paper_results.json") -> None:
    payload = read_data(source)
    if payload is None or "replication_124m_900" not in payload:
        missing("fig_seed_replication", "add replication_124m_900 to artifacts/paper_results.json")
        return

    block = payload["replication_124m_900"]
    seeds = block["seeds_planned"]
    muon = np.asarray(block["muon"]["loss"], dtype=float)
    astro = np.asarray(block["astro_v2"]["loss"], dtype=float)

    fig, ax = column_figure(height=2.52)
    x_muon, x_astro = 0.0, 1.0

    # Deterministic jitter keeps individual seeds visible without implying a
    # density estimate from only seven observations.
    jitter = np.linspace(-0.065, 0.065, len(muon)) if len(muon) > 1 else np.array([0.0])
    ax.scatter(np.full(len(muon), x_muon) + jitter, muon, s=22,
               color=COLORS["muon"], alpha=0.72, linewidths=0, zorder=3)
    if len(muon):
        mean_m = float(muon.mean())
        ax.scatter(x_muon, mean_m, s=62, marker="_", color=COLORS["muon"],
                   linewidths=2.2, zorder=5)
        ax.annotate(f"{mean_m:.4f}", (x_muon, mean_m), textcoords="offset points",
                    xytext=(-5, 8), ha="right", fontsize=6.1, color=COLORS["muon"])

    paired = len(astro) == len(muon) == len(seeds) and len(astro) > 0
    if paired:
        for i, (m, a) in enumerate(zip(muon, astro)):
            ax.plot([x_muon, x_astro], [m, a], color="#D6D6D6", linewidth=0.75, zorder=1)
        jitter_a = np.linspace(-0.065, 0.065, len(astro))
        ax.scatter(np.full(len(astro), x_astro) + jitter_a, astro, s=24,
                   color=COLORS["astro_v2"], alpha=0.78, linewidths=0, zorder=3)
        mean_a = float(astro.mean())
        ax.scatter(x_astro, mean_a, s=62, marker="_", color=COLORS["astro_v2"],
                   linewidths=2.2, zorder=5)
        ax.annotate(f"{mean_a:.4f}", (x_astro, mean_a), textcoords="offset points",
                    xytext=(5, 8), ha="left", fontsize=6.1, color=COLORS["astro_v2"])
    else:
        # Open placeholder: visible enough to document study status, quiet
        # enough that it cannot be mistaken for a measurement.
        ymid = float(muon.mean()) if len(muon) else 6.0
        ax.scatter(x_astro, ymid, s=48, marker="D", facecolor="white",
                   edgecolor=COLORS["pending"], linewidth=1.0, zorder=3)
        ax.text(x_astro, ymid - 0.013, "pending", ha="center", va="top",
                fontsize=6.0, color="#777777")

    ax.set_xticks([x_muon, x_astro])
    ax.set_xticklabels(["Muon", "ASTRO-v2"])
    ax.set_ylabel("validation loss ↓")
    ax.set_xlim(-0.42, 1.42)
    if len(muon):
        lo = min(muon.min(), astro.min() if len(astro) else muon.min()) - 0.028
        hi = max(muon.max(), astro.max() if len(astro) else muon.max()) + 0.028
        ax.set_ylim(lo, hi)
    grid(ax, axis="y")

    ax.text(0.02, 0.965, f"124M · 900 steps · {len(muon)}/{len(seeds)} Muon seeds measured",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.05,
            fontweight="bold", color="#333333")
    ax.text(0.02, 0.035, "dots = seeds   horizontal tick = mean",
            transform=ax.transAxes, fontsize=5.8, color="#666666")

    fig.tight_layout(pad=0.35)
    saved = save(fig, "fig_seed_replication")
    write_data("fig_seed_replication", {
        "seeds": seeds,
        "muon": muon.tolist(),
        "astro_v2": astro.tolist(),
        "paired_complete": paired,
    })
    print(f"  wrote {saved}")


if __name__ == "__main__":
    build()
