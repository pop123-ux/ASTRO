#!/usr/bin/env python3
"""Publication figures for the ORBIT paper.

The plotting functions consume only the strict paper_results.json artifact emitted
by orbit_paper_artifacts.py. They do not inspect raw run logs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


METHOD_LABELS = {
    "adamw": "AdamW",
    "adamuon_ref": "AdaMuon",
    "muon": "Muon",
    "normuon": "NorMuon",
    "astro_v2": "ASTRO",
    "orbit": "ORBIT",
}


def load_results(path: Path) -> dict:
    return json.loads(path.read_text())


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_overview(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.6, 3.15))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 4.0)
    ax.axis("off")

    def box(
        x, y, w, h, title, subtitle=None, *,
        lw=1.2, facecolor="#FFFFFF", edgecolor="0.25"
    ):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.035,rounding_size=0.08",
            linewidth=lw,
            facecolor=facecolor,
            edgecolor=edgecolor,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h * 0.66,
            title,
            ha="center",
            va="center",
            fontsize=10.0,
            fontweight="bold",
            linespacing=1.0,
        )
        if subtitle:
            ax.text(
                x + w / 2,
                y + h * 0.28,
                subtitle,
                ha="center",
                va="center",
                fontsize=8.4,
                color="0.30",
                linespacing=1.1,
            )
        return (x, y, w, h)

    def arrow(a, b, *, yfrac=0.5):
        x1=a[0]+a[2]; y1=a[1]+a[3]*yfrac
        x2=b[0]; y2=b[1]+b[3]*yfrac
        ax.annotate("", xy=(x2-0.08,y2), xytext=(x1+0.08,y1),
                    arrowprops=dict(arrowstyle="->", lw=1.25, color="0.35"))

    b1 = box(
        0.20, 1.95, 1.65, 1.25, "Muon candidate", "Q/K gradients\n+ momentum",
        facecolor="#EDF5FA", edgecolor="#4F7C92",
    )
    b2 = box(
        2.20, 1.95, 1.70, 1.25, "Q/K statistics", "EMA $2\\times2$\nsecond moments",
        facecolor="#F6F4F0", edgecolor="#756F68",
    )
    b3 = box(
        4.25, 1.95, 1.80, 1.25, "RoPE transport", "optimizer orientation\n$R(+\\Delta)$",
        facecolor="#ECF7F5", edgecolor="#4E928E",
    )
    b4 = box(
        6.40, 1.95, 1.75, 1.25, "Local metric", "per-frequency\n$2\\times2$",
        facecolor="#EDF5FA", edgecolor="#4F7C92",
    )
    b5 = box(
        8.50, 1.95, 1.75, 1.25, "Precondition", r"analytic $M^{-1/2}$",
        facecolor="#FAF3E8", edgecolor="#8E7B5C",
    )
    b6 = box(
        10.60, 1.95, 1.40, 1.25, "Restore", "joint Q+K\nnorm",
        facecolor="#ECF7F5", edgecolor="#4E928E",
    )
    for a,b in zip((b1,b2,b3,b4,b5),(b2,b3,b4,b5,b6)):
        arrow(a,b)

    ax.text(6.10,3.62,"ORBIT: RoPE-informed Q/K optimizer update",
            ha="center",va="center",fontsize=12.0,fontweight="regular")
    ax.text(
        2.25,
        0.72,
        r"Other hidden matrices $\rightarrow$ standard Muon"
        "\n"
        r"Embeddings / biases / norms $\rightarrow$ auxiliary AdamW"
        "\nInference graph unchanged",
        ha="left",
        va="center",
        fontsize=8.5,
        color="black",
        linespacing=1.25,
    )
    ax.annotate("", xy=(11.30, 1.68), xytext=(11.30, 1.14),
                arrowprops=dict(arrowstyle="->", lw=1.15, color="0.35"))
    ax.text(11.30, 0.90, "parameter update", ha="center", va="center", fontsize=8.8)
    save(fig, out, "orbit_overview")


def plot_matched_effect(results: dict, out: Path) -> None:
    effect = results["matched_confirmation"]["orbit_vs_muon"]
    per_seed = {int(k): float(v) for k, v in effect["per_seed"].items()}
    seeds = sorted(per_seed)
    vals = [per_seed[s] for s in seeds]
    mean = float(effect["mean_delta"])
    lo, hi = effect["ci95"]

    x = list(range(len(seeds)))
    mean_x = len(seeds) + 0.75
    fig, ax = plt.subplots(figsize=(7.0, 3.60))
    ax.axhline(0.0, linewidth=1.0, linestyle="--", color="0.45")
    ax.scatter(x, vals, zorder=3, s=34)
    ax.errorbar(
        [mean_x], [mean],
        yerr=[[mean - lo], [hi - mean]],
        fmt="D", capsize=5, markersize=6.5, linewidth=1.8,
        label=r"paired mean $\pm$ 95% CI",
    )
    ax.set_xticks(x + [mean_x], [str(s) for s in seeds] + ["Mean"])
    ax.set_xlim(-0.55, mean_x + 0.65)
    ax.set_xlabel("Held-out run")
    ax.set_ylabel(r"$\Delta$ validation loss (ORBIT $-$ Muon)")
    ax.set_title("Matched ORBIT-Muon effect across held-out runs")
    ax.legend(frameon=False, fontsize=8)
    save(fig, out, "matched_effect")


def plot_xconfig(results: dict, out: Path) -> None:
    x = results["cross_configuration_isolation"]
    labels = ["Muon discovery recipe", "ORBIT discovery recipe"]
    orbit = [x["means"]["orbit_at_muon_config"], x["means"]["orbit_at_orbit_config"]]
    muon = [x["means"]["muon_at_muon_config"], x["means"]["muon_at_orbit_config"]]

    fig, ax = plt.subplots(figsize=(5.8, 3.45))
    idx = [0, 1]
    ax.plot(idx, muon, marker="o", linewidth=1.6, label="Muon update")
    ax.plot(idx, orbit, marker="o", linewidth=1.6, label="ORBIT update")
    ax.set_xticks(idx, labels)
    ax.set_ylabel("Validation loss")
    ax.set_title("Crossed optimizer and discovery recipe")
    ax.legend(frameon=False, fontsize=8)
    save(fig, out, "xconfig")


def plot_mechanism_summary(results: dict, out: Path) -> None:
    x = results["cross_configuration_isolation"]
    ab = results["mechanism_ablation"]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3))

    ax = axes[0]
    labels = ["Muon discovery recipe", "ORBIT discovery recipe"]
    orbit = [x["means"]["orbit_at_muon_config"], x["means"]["orbit_at_orbit_config"]]
    muon = [x["means"]["muon_at_muon_config"], x["means"]["muon_at_orbit_config"]]
    idx = [0, 1]
    ax.plot(idx, muon, marker="o", linewidth=1.6, label="Muon")
    ax.plot(idx, orbit, marker="o", linewidth=1.6, label="ORBIT")
    ax.set_xticks(idx, labels)
    ax.set_ylabel("Validation loss")
    ax.set_title("(a) Crossed optimizer and recipe")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    entries = [
        ("Identity", ab["orbit_vs_identity"]),
        ("No-RoPE", ab["orbit_vs_norope"]),
        ("Diagonal", ab["orbit_vs_diag"]),
    ]
    ys = list(range(len(entries)))
    means = [float(e[1]["mean_delta"]) for e in entries]
    lows = [float(e[1]["ci95"][0]) for e in entries]
    highs = [float(e[1]["ci95"][1]) for e in entries]
    xerr = [[m-l for m,l in zip(means,lows)], [h-m for m,h in zip(means,highs)]]
    ax.axvline(0, color="0.45", linestyle="--", linewidth=1.0)
    ax.errorbar(means, ys, xerr=xerr, fmt="o", capsize=4, linewidth=1.5)
    ax.set_yticks(ys, [e[0] for e in entries])
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta$ validation loss (full ORBIT $-$ control)")
    ax.set_title("(b) Mechanism ablations")

    fig.tight_layout()
    save(fig, out, "mechanism_summary")


def plot_transfer(results: dict, out: Path) -> None:
    horizon = results["long_horizon_transfer"]["methods"]
    scale = results.get("scale_transfer", {}).get("methods", {})
    methods = [m for m in ("muon", "normuon", "astro_v2", "orbit") if m in horizon]

    panels = [("124M / 2700 steps", horizon)]
    if scale:
        panels.append(("355M / 900 steps", scale))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 3.4), squeeze=False)
    for ax, (title, data) in zip(axes[0], panels):
        for i, method in enumerate(methods):
            row = data[method]
            vals = row.get("values", [])
            if vals:
                ax.scatter([i] * len(vals), vals, s=32)
            ax.hlines(row["mean_val_loss"], i - 0.22, i + 0.22, linewidth=2.1)
        ax.set_xticks(range(len(methods)), [METHOD_LABELS[m] for m in methods])
        ax.set_ylabel("Validation loss")
        ax.set_title(title)
    fig.tight_layout()
    save(fig, out, "transfer_summary")


def plot_broad(results: dict, out: Path) -> None:
    broad = results["broad_independently_tuned_context"]
    order = [m for m in ("adamw", "adamuon_ref", "muon", "normuon", "astro_v2", "orbit") if m in broad]
    means = [broad[m]["mean_val_loss"] for m in order]
    sds = [broad[m]["sd_val_loss"] for m in order]
    labels = [METHOD_LABELS[m] for m in order]

    fig, ax = plt.subplots(figsize=(6.2, 3.55))
    x = list(range(len(order)))
    ax.errorbar(x, means, yerr=sds, fmt="o", capsize=4, linewidth=1.5)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Validation loss")
    ax.set_title("Broader 124M optimizer context")
    fig.tight_layout()
    save(fig, out, "broad_confirmation")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = load_results(args.results)
    plot_overview(args.out)
    plot_matched_effect(results, args.out)
    plot_xconfig(results, args.out)
    plot_mechanism_summary(results, args.out)
    plot_transfer(results, args.out)
    plot_broad(results, args.out)


if __name__ == "__main__":
    main()
