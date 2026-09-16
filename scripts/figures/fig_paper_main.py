"""Main paper figure: five held-out seeds at 124M / 900 steps.

This plot reads only ``artifacts/paper_campaign.json``. Historical measurements
from the old harness can never leak into it.
"""

from __future__ import annotations

import numpy as np

from paper_style import COLORS, column_figure, grid, missing, read_data, save, write_data


def build(source: str = "paper_campaign.json") -> None:
    payload = read_data(source)
    if payload is None:
        missing("fig_paper_main", "python scripts/paper_collect.py ...")
        return
    block = payload.get("main_124m_900", {})
    if not block.get("complete"):
        missing("fig_paper_main", "finish the clean 124M/900 five-seed main campaign")
        return

    names = ["muon", "normuon", "adamuon_ref", "astro_v2"]
    fig, ax = column_figure(height=2.68)
    x = np.arange(len(names), dtype=float)

    summary = {}
    for xi, name in zip(x, names):
        row = block["optimizers"][name]
        values = np.asarray(row["loss"], dtype=float)
        jitter = np.linspace(-0.075, 0.075, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(
            np.full(len(values), xi) + jitter,
            values,
            s=23 if name != "astro_v2" else 27,
            color=COLORS[name],
            alpha=0.78,
            linewidths=0,
            zorder=3,
        )
        mean = float(values.mean())
        ax.scatter(xi, mean, marker="_", s=78, linewidths=2.2,
                   color=COLORS[name], zorder=5)
        ax.annotate(f"{mean:.4f}", (xi, mean), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=5.8, color=COLORS[name])
        summary[name] = {"mean": mean, "loss": values.tolist()}

    muon = block["optimizers"]["muon"]
    normuon = block["optimizers"]["normuon"]
    adamuon = block["optimizers"]["adamuon_ref"]
    astro = block["optimizers"]["astro_v2"]
    d_muon = float(astro["mean"] - muon["mean"])
    d_normuon = float(astro["mean"] - normuon["mean"])
    d_ada = float(astro["mean"] - adamuon["mean"])

    ax.set_xticks(x)
    ax.set_xticklabels(["Muon", "NorMuon", "AdaMuon", "ASTRO-v2"], fontsize=6.2)
    ax.set_ylabel("validation loss ↓")
    grid(ax, axis="y")
    ax.set_title("124M · 900 steps · five held-out seeds", loc="left",
                 pad=15, fontsize=7.7, fontweight="bold")
    ax.text(0.0, 1.015,
            f"ASTRO-v2 Δ: Muon {d_muon:+.3f} · NorMuon {d_normuon:+.3f} · AdaMuon {d_ada:+.3f}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=5.45,
            color="#666666")
    ax.text(0.02, 0.035, "dots = seeds   horizontal tick = mean",
            transform=ax.transAxes, fontsize=5.7, color="#666666")

    all_values = np.concatenate([
        np.asarray(block["optimizers"][name]["loss"], dtype=float) for name in names
    ])
    pad = max(0.02, 0.12 * float(all_values.max() - all_values.min()))
    ax.set_ylim(float(all_values.min() - pad), float(all_values.max() + pad))
    fig.tight_layout(pad=0.4)
    saved = save(fig, "fig_paper_main")
    write_data("fig_paper_main", {
        "summary": summary,
        "astro_v2_delta_vs_muon": d_muon,
        "astro_v2_delta_vs_normuon": d_normuon,
        "astro_v2_delta_vs_adamuon": d_ada,
    })
    print(f"  wrote {saved}")


if __name__ == "__main__":
    build()
