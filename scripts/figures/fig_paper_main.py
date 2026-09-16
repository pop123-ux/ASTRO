"""Headline paper figure: held-out validation loss and measured runtime.

Both panels use the same five 124M/900 held-out seeds.  Runtime is reported as a
cost measurement, not as a separate wall-clock-superiority claim.
"""

from __future__ import annotations

import numpy as np

from paper_style import COLORS, figure, grid, label_panels, missing, read_data, save, write_data


def build(source: str = "paper_campaign.json") -> None:
    payload = read_data(source)
    if payload is None:
        missing("fig_paper_main", "python scripts/paper_collect.py ...")
        return
    block = payload.get("main_124m_900", {})
    if not block.get("complete"):
        missing("fig_paper_main", "finish the clean 124M/900 five-seed headline campaign")
        return

    names = ["adamw", "muon", "normuon", "adamuon_ref", "astro_v2"]
    labels = ["AdamW", "Muon", "NorMuon", "AdaMuon", "ASTRO-v2"]
    fig, axes = figure(panels=2, height=2.55)
    ax_loss, ax_time = axes
    x = np.arange(len(names), dtype=float)
    summary = {}

    for xi, name in zip(x, names):
        row = block["optimizers"][name]
        values = np.asarray(row["loss"], dtype=float)
        seconds = np.asarray(row["seconds"], dtype=float)
        jitter = np.linspace(-0.065, 0.065, len(values)) if len(values) > 1 else np.array([0.0])

        ax_loss.scatter(np.full(len(values), xi) + jitter, values,
                        s=19 if name != "astro_v2" else 24,
                        color=COLORS[name], alpha=0.75, linewidths=0, zorder=3)
        mean_loss = float(values.mean())
        ax_loss.scatter(xi, mean_loss, marker="_", s=72, linewidths=2.0,
                        color=COLORS[name], zorder=5)
        ax_loss.annotate(f"{mean_loss:.3f}", (xi, mean_loss),
                         textcoords="offset points", xytext=(0, 7), ha="center",
                         fontsize=5.35, color=COLORS[name])

        time_mean = float(seconds.mean())
        time_sd = float(seconds.std(ddof=1)) if len(seconds) > 1 else 0.0
        ax_time.errorbar(xi, time_mean / 60.0, yerr=time_sd / 60.0,
                         fmt="D" if name == "astro_v2" else "o",
                         markersize=4.2, capsize=2.0, linewidth=0.9,
                         color=COLORS[name], zorder=3)
        ax_time.annotate(f"{time_mean/60:.1f}m", (xi, time_mean / 60.0),
                         textcoords="offset points", xytext=(0, 7), ha="center",
                         fontsize=5.3, color=COLORS[name])
        summary[name] = {
            "mean_loss": mean_loss,
            "loss": values.tolist(),
            "mean_seconds": time_mean,
            "seconds": seconds.tolist(),
        }

    astro = block["optimizers"]["astro_v2"]
    muon = block["optimizers"]["muon"]
    normuon = block["optimizers"]["normuon"]
    adamuon = block["optimizers"]["adamuon_ref"]
    d_muon = float(astro["mean"] - muon["mean"])
    d_normuon = float(astro["mean"] - normuon["mean"])
    d_ada = float(astro["mean"] - adamuon["mean"])
    runtime_overhead = float(astro["mean_seconds"] / muon["mean_seconds"] - 1.0)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=5.6)
        grid(ax, axis="y")
    ax_loss.set_ylabel("validation loss ↓")
    ax_time.set_ylabel("minutes / 900-step run ↓")
    ax_loss.set_title("held-out validation", loc="left", pad=6,
                      fontsize=7.2, fontweight="bold")
    ax_time.set_title("measured training cost", loc="left", pad=6,
                      fontsize=7.2, fontweight="bold")
    ax_loss.text(0.0, 1.02,
                 f"ASTRO-v2 Δ: Muon {d_muon:+.3f} · NorMuon {d_normuon:+.3f} · AdaMuon {d_ada:+.3f}",
                 transform=ax_loss.transAxes, ha="left", va="bottom",
                 fontsize=5.1, color="#666666")
    ax_time.text(0.02, 0.035,
                 f"ASTRO-v2 vs Muon runtime: {runtime_overhead*100:+.1f}%",
                 transform=ax_time.transAxes, fontsize=5.6, color="#666666")
    label_panels(axes)

    fig.tight_layout(pad=0.5, w_pad=1.0)
    saved = save(fig, "fig_paper_main")
    write_data("fig_paper_main", {
        "summary": summary,
        "astro_v2_delta_vs_muon": d_muon,
        "astro_v2_delta_vs_normuon": d_normuon,
        "astro_v2_delta_vs_adamuon": d_ada,
        "astro_v2_runtime_overhead_vs_muon": runtime_overhead,
        "note": "same-step runtime cost; not a separate time-matched superiority claim",
    })
    print(f"  wrote {saved}")


if __name__ == "__main__":
    build()
