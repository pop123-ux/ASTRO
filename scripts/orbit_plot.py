#!/usr/bin/env python3
"""Publication figures for the ORBIT paper evidence hierarchy."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from orbit_paper_artifacts import build as freeze_evidence


plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 240,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "cmr10",
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    }
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def grouped(rows: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        out[str(row.get(key))].append(row)
    return out


def save(fig, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_orbit_overview(out: Path) -> None:
    """Compact vector schematic of the paper-trained ORBIT update path."""
    fig, ax = plt.subplots(figsize=(11.6, 3.15))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 4.0)
    ax.axis("off")

    def box(x, y, w, h, title, subtitle, *, facecolor, edgecolor):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.035,rounding_size=0.08",
            linewidth=1.2,
            facecolor=facecolor,
            edgecolor=edgecolor,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
                fontsize=10.0, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center",
                fontsize=8.4, color="0.30", linespacing=1.1)
        return (x, y, w, h)

    def arrow(a, b):
        x1, y1 = a[0] + a[2], a[1] + a[3] * 0.5
        x2, y2 = b[0], b[1] + b[3] * 0.5
        ax.annotate("", xy=(x2 - 0.08, y2), xytext=(x1 + 0.08, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.25, color="0.35"))

    boxes = [
        box(0.20, 1.95, 1.65, 1.25, "Muon candidate", "Q/K gradients\n+ momentum",
            facecolor="#EDF5FA", edgecolor="#4F7C92"),
        box(2.20, 1.95, 1.70, 1.25, "Q/K statistics", "EMA $2\\times2$\nsecond moments",
            facecolor="#F6F4F0", edgecolor="#756F68"),
        box(4.25, 1.95, 1.80, 1.25, "RoPE transport", "optimizer orientation\n$R(+\\Delta)$",
            facecolor="#ECF7F5", edgecolor="#4E928E"),
        box(6.40, 1.95, 1.75, 1.25, "Local metric", "per-frequency\n$2\\times2$",
            facecolor="#EDF5FA", edgecolor="#4F7C92"),
        box(8.50, 1.95, 1.75, 1.25, "Precondition", r"analytic $M^{-1/2}$",
            facecolor="#FAF3E8", edgecolor="#8E7B5C"),
        box(10.60, 1.95, 1.40, 1.25, "Restore", "joint Q+K\nnorm",
            facecolor="#ECF7F5", edgecolor="#4E928E"),
    ]
    for a, b in zip(boxes[:-1], boxes[1:]):
        arrow(a, b)

    ax.text(6.10, 3.62, "ORBIT: RoPE-informed Q/K optimizer update",
            ha="center", va="center", fontsize=12.0)
    ax.text(
        6.10, 0.68,
        r"Other hidden matrices $\rightarrow$ standard Muon   |   "
        r"Embeddings / biases / norms $\rightarrow$ auxiliary AdamW   |   "
        r"Inference graph unchanged",
        ha="center", va="center", fontsize=8.3, color="black",
    )
    ax.annotate("", xy=(11.30, 1.68), xytext=(11.30, 1.14),
                arrowprops=dict(arrowstyle="->", lw=1.15, color="0.35"))
    ax.text(11.30, 0.90, "parameter update", ha="center", va="center", fontsize=8.8)
    save(fig, out, "orbit_overview")


def plot_mechanism_summary(results: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.55), gridspec_kw={"wspace": 0.34})

    ax = axes[0]
    sm = results["cross_configuration_isolation"]["summary"]
    labels = ["Muon discovery recipe", "ORBIT discovery recipe"]
    series = {
        "Muon": ["muon_at_muon_config", "muon_at_orbit_config"],
        "ORBIT": ["orbit_at_muon_config", "orbit_at_orbit_config"],
    }
    for name, keys in series.items():
        means = [sm[k]["mean_val_loss"] for k in keys]
        sds = [sm[k]["sd_val_loss"] for k in keys]
        ax.errorbar([0, 1], means, yerr=sds, marker="o", capsize=4, label=name)
    ax.set_xticks([0, 1], labels)
    ax.set_ylabel("Validation loss")
    ax.set_title(r"Optimizer $\times$ discovery recipe")
    ax.legend(frameon=False)
    ax.text(-0.14, 1.04, "a", transform=ax.transAxes, fontweight="bold", fontsize=11)

    ax = axes[1]
    ab = results["mechanism_ablation"]
    comparisons = [
        ("Identity", ab["orbit_vs_identity"]),
        ("No-RoPE", ab["orbit_vs_norope"]),
        ("Diagonal", ab["orbit_vs_diag"]),
    ]
    ax.axvline(0.0, linewidth=1.0, linestyle="--", color="0.45")
    for yi, (_, effect) in enumerate(comparisons):
        mean = float(effect["mean_delta"])
        lo, hi = effect["ci95"]
        ax.errorbar(mean, yi, xerr=[[mean - lo], [hi - mean]], fmt="o", capsize=4)
    ax.set_yticks(range(len(comparisons)), [x[0] for x in comparisons])
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta$ validation loss (ORBIT $-$ control)")
    ax.set_title("Mechanism ablations")
    ax.text(-0.14, 1.04, "b", transform=ax.transAxes, fontweight="bold", fontsize=11)
    save(fig, out, "mechanism_summary")


def plot_transfer_summary(horizon_rows: list[dict], _scale_rows: list[dict], out: Path) -> None:
    """Release long-horizon plot; the historical 355M development panel is omitted."""
    order = ["muon", "normuon", "astro_v2", "orbit"]
    names = {"muon": "Muon", "normuon": "NorMuon", "astro_v2": "ASTRO", "orbit": "ORBIT"}
    by = grouped(horizon_rows, "optimizer")

    fig, ax = plt.subplots(figsize=(6.2, 3.55))
    for xi, method in enumerate(order):
        vals = [float(x["val_loss"]) for x in sorted(by[method], key=lambda r: r["seed"])]
        offsets = [-0.055, 0.055] if len(vals) == 2 else [0.0] * len(vals)
        ax.scatter([xi + o for o in offsets], vals, s=32, zorder=3)
        mean = statistics.fmean(vals)
        ax.plot([xi - 0.14, xi + 0.14], [mean, mean], linewidth=2.0)
    ax.set_xticks(range(len(order)), [names[m] for m in order])
    ax.set_ylabel("Validation loss")
    ax.set_title("124M / 2700 steps")
    save(fig, out, "horizon_transfer")


def plot_broad_confirmation(rows: list[dict], out: Path) -> None:
    by = grouped(rows, "optimizer")
    order = [m for m in ("adamw", "adamuon_ref", "muon", "normuon", "astro_v2", "orbit") if m in by]
    names = {
        "adamw": "AdamW", "adamuon_ref": "AdaMuon", "muon": "Muon",
        "normuon": "NorMuon", "astro_v2": "ASTRO", "orbit": "ORBIT",
    }
    means, sds = [], []
    for method in order:
        vals = [float(x["val_loss"]) for x in by[method]]
        means.append(statistics.fmean(vals))
        sds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
    fig, ax = plt.subplots(figsize=(6.2, 3.55))
    xs = list(range(len(order)))
    ax.errorbar(xs, means, yerr=sds, fmt="o", capsize=4, linewidth=1.5)
    ax.set_xticks(xs, [names[m] for m in order], rotation=20, ha="right")
    ax.set_ylabel(r"Validation loss (mean $\pm$ SD)")
    ax.set_title("Broader 124M optimizer context")
    save(fig, out, "broad_confirmation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    results, _ = freeze_evidence(args.work_dir)
    merged = args.work_dir / "merged"
    out = args.work_dir / "paper_artifacts" / "figures"
    plot_orbit_overview(out)
    if "cross_configuration_isolation" in results and "mechanism_ablation" in results:
        plot_mechanism_summary(results, out)
    if "long_horizon_transfer" in results and "scale_transfer" in results:
        plot_transfer_summary(load_jsonl(merged / "horizon_with_astro.jsonl"),
                              load_jsonl(merged / "scale_with_astro.jsonl"), out)
    if "broad_independently_tuned_context" in results:
        plot_broad_confirmation(load_jsonl(merged / "confirm.jsonl"), out)


if __name__ == "__main__":
    main()
