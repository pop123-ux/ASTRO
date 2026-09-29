#!/usr/bin/env python3
"""Generate publication-oriented ORBIT figures from the frozen evidence hierarchy."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from orbit_paper_artifacts import build as freeze_evidence
from orbit_paper_artifacts import grouped, load_jsonl


plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 240,
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
    }
)


def save(fig, out: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def mean_sd(values: list[float]) -> tuple[float, float]:
    return (
        statistics.fmean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def plot_matched_effect(results: dict, out: Path) -> None:
    effect = results["matched_confirmation"]["orbit_vs_muon"]
    per_seed = {int(k): float(v) for k, v in effect["per_seed"].items()}
    seeds = sorted(per_seed)
    vals = [per_seed[s] for s in seeds]
    mean = float(effect["mean_delta"])
    lo, hi = effect["ci95"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(0.0, linewidth=1.0, linestyle="--", color="0.35")
    ax.scatter(seeds, vals, zorder=3)
    ax.axhspan(lo, hi, alpha=0.12)
    ax.axhline(mean, linewidth=2.2)
    ax.set_xticks(seeds)
    ax.set_xlabel("Held-out seed")
    ax.set_ylabel("Validation-loss difference (ORBIT − Muon)")
    ax.set_title("Matched hyperparameters: seed-wise ORBIT effect")
    ax.text(
        0.02,
        0.04,
        f"mean Δ = {mean:.4f}   95% CI [{lo:.4f}, {hi:.4f}]   "
        f"wins {effect['a_wins']}/{effect['n']}",
        transform=ax.transAxes,
        va="bottom",
    )
    save(fig, out, "matched_effect")


def plot_matched_pairs(rows: list[dict], out: Path) -> None:
    by = grouped(rows, "optimizer")
    mu = {int(x["seed"]): float(x["val_loss"]) for x in by["muon"]}
    orb = {int(x["seed"]): float(x["val_loss"]) for x in by["orbit"]}
    seeds = sorted(set(mu) & set(orb))

    fig, ax = plt.subplots(figsize=(5.9, 4.8))
    for seed in seeds:
        ax.plot([0, 1], [mu[seed], orb[seed]], marker="o", alpha=0.42, linewidth=1.0)
    ax.scatter(
        [0, 1],
        [statistics.fmean(mu.values()), statistics.fmean(orb.values())],
        marker="D",
        s=58,
        zorder=4,
        label="mean",
    )
    ax.set_xticks([0, 1], ["Muon", "ORBIT"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("Validation loss")
    ax.set_title("Matched confirmation: paired absolute losses")
    ax.legend(frameon=False)
    save(fig, out, "matched_seed_pairs")


def plot_xconfig(results: dict, out: Path) -> None:
    sm = results["cross_configuration_isolation"]["summary"]
    x = [0, 1]
    labels = ["Muon-selected config", "ORBIT-selected config"]
    series = {
        "Muon": ["muon_at_muon_config", "muon_at_orbit_config"],
        "ORBIT": ["orbit_at_muon_config", "orbit_at_orbit_config"],
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    for name, keys in series.items():
        means = [sm[k]["mean_val_loss"] for k in keys]
        sds = [sm[k]["sd_val_loss"] for k in keys]
        ax.errorbar(x, means, yerr=sds, marker="o", capsize=4, label=name)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Validation loss")
    ax.set_title("Optimizer × hyperparameter-recipe cross-over")
    ax.legend(frameon=False)
    ax.text(
        0.02,
        0.04,
        "Lower is better. Interaction is reported numerically in the paper table.",
        transform=ax.transAxes,
        va="bottom",
    )
    save(fig, out, "xconfig_interaction")


def plot_ablation_effects(results: dict, out: Path) -> None:
    ab = results["mechanism_ablation"]
    comparisons = [
        ("Identity control", ab["orbit_vs_identity"]),
        ("No RoPE rotation", ab["orbit_vs_norope"]),
        ("Diagonal metric", ab["orbit_vs_diag"]),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axvline(0.0, linewidth=1.0, linestyle="--", color="0.35")
    y = list(range(len(comparisons)))
    for yi, (label, effect) in zip(y, comparisons):
        mean = float(effect["mean_delta"])
        lo, hi = effect["ci95"]
        ax.errorbar(
            mean,
            yi,
            xerr=[[mean - lo], [hi - mean]],
            fmt="o",
            capsize=4,
        )
        ax.text(
            hi + 0.00025,
            yi,
            f"{effect['a_wins']}/{effect['n']}",
            va="center",
            fontsize=9,
        )
    ax.set_yticks(y, [x[0] for x in comparisons])
    ax.invert_yaxis()
    ax.set_xlabel("Validation-loss difference (full ORBIT − control)")
    ax.set_title("Mechanism ablation: paired effects with 95% CIs")
    ax.text(
        0.02,
        0.03,
        "Negative favors full ORBIT; annotation = seed wins.",
        transform=ax.transAxes,
        va="bottom",
    )
    save(fig, out, "ablation_effects")


def plot_transfer(rows: list[dict], out: Path, *, stem: str, title: str) -> None:
    by = grouped(rows, "optimizer")
    order = ["muon", "normuon", "astro_v2", "orbit"]
    display = {"muon": "Muon", "normuon": "NorMuon", "astro_v2": "ASTRO", "orbit": "ORBIT"}

    fig, ax = plt.subplots(figsize=(7.1, 4.5))
    for xi, name in enumerate(order):
        vals = [float(x["val_loss"]) for x in sorted(by[name], key=lambda r: r["seed"])]
        offsets = [-0.045, 0.045] if len(vals) == 2 else [0.0] * len(vals)
        ax.scatter([xi + o for o in offsets], vals, s=42, zorder=3)
        mean = statistics.fmean(vals)
        ax.plot([xi - 0.15, xi + 0.15], [mean, mean], linewidth=2.4)
    ax.set_xticks(range(len(order)), [display[x] for x in order])
    ax.set_ylabel("Validation loss")
    ax.set_title(title)
    ax.text(
        0.02,
        0.04,
        "Individual seeds shown; n=2 per optimizer. No significance claim from this panel.",
        transform=ax.transAxes,
        va="bottom",
    )
    save(fig, out, stem)


def plot_broad_confirmation(rows: list[dict], out: Path) -> None:
    by = grouped(rows, "optimizer")
    display = {
        "adamw": "AdamW",
        "adamuon_ref": "AdaMuon",
        "muon": "Muon",
        "normuon": "NorMuon",
        "astro_v2": "ASTRO",
        "orbit": "ORBIT",
    }
    stats = []
    for name, items in by.items():
        vals = [float(x["val_loss"]) for x in items]
        mean, sd = mean_sd(vals)
        stats.append((mean, sd, name))
    stats.sort(reverse=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    for yi, (mean, sd, name) in enumerate(stats):
        ax.errorbar(mean, yi, xerr=sd, fmt="o", capsize=4)
    ax.set_yticks(range(len(stats)), [display.get(x[2], x[2]) for x in stats])
    ax.set_xlabel("Validation loss (mean ± SD)")
    ax.set_title("Broad independently tuned confirmation")
    ax.text(
        0.02,
        0.03,
        "Context only: configurations were selected independently; not the primary causal test.",
        transform=ax.transAxes,
        va="bottom",
    )
    save(fig, out, "broad_confirmation")


def plot_training_curves(rows: list[dict], out: Path) -> None:
    curves = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for point in row.get("curve", []):
            curves[row["optimizer"]][int(point["step"])].append(float(point["train_loss"]))

    fig, ax = plt.subplots(figsize=(7.3, 4.6))
    for name in ("muon", "orbit"):
        steps = sorted(curves[name])
        if not steps:
            continue
        means = [statistics.fmean(curves[name][s]) for s in steps]
        sds = [
            statistics.stdev(curves[name][s]) if len(curves[name][s]) > 1 else 0.0
            for s in steps
        ]
        (line,) = ax.plot(steps, means, label=name.upper() if name == "orbit" else "Muon")
        color = line.get_color()
        lower = [m - s for m, s in zip(means, sds)]
        upper = [m + s for m, s in zip(means, sds)]
        ax.fill_between(steps, lower, upper, alpha=0.12, color=color)
    ax.set_xlabel("Optimizer steps")
    ax.set_ylabel("Training loss")
    ax.set_title("Matched-confirmation training trajectories")
    ax.legend(frameon=False)
    save(fig, out, "matched_training_curves")


def plot_efficiency(rows: list[dict], out: Path) -> None:
    by = grouped(rows, "optimizer")
    fig, ax = plt.subplots(figsize=(6.8, 4.7))
    for name in ("muon", "orbit"):
        xs = [float(x["seconds"]) / 60 for x in by[name]]
        ys = [float(x["val_loss"]) for x in by[name]]
        ax.scatter(xs, ys, label=name.upper() if name == "orbit" else "Muon", alpha=0.82)
    ax.set_xlabel("Wall-clock minutes")
    ax.set_ylabel("Validation loss")
    ax.set_title("Matched confirmation: quality vs wall-clock")
    ax.legend(frameon=False)
    save(fig, out, "matched_efficiency")


def plot_metric_diagnostics(rows: list[dict], out: Path) -> None:
    orbit = [
        x for x in rows
        if x["optimizer"] == "orbit" and x.get("optimizer_diagnostics")
    ]
    if not orbit:
        return
    specs = [
        ("metric_condition_q", "Q metric condition"),
        ("metric_condition_k", "K metric condition"),
        ("joint_restore", "Joint restore"),
    ]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for xi, (key, label) in enumerate(specs):
        vals = [float(x["optimizer_diagnostics"][key]) for x in orbit]
        offsets = [
            (j - (len(vals) - 1) / 2) * 0.012
            for j in range(len(vals))
        ]
        ax.scatter([xi + o for o in offsets], vals, alpha=0.72, s=28)
        mean = statistics.fmean(vals)
        ax.plot([xi - 0.14, xi + 0.14], [mean, mean], linewidth=2.5)
    ax.set_xticks(range(len(specs)), [x[1] for x in specs])
    ax.set_ylabel("Recorded diagnostic value")
    ax.set_title("ORBIT metric diagnostics across matched held-out seeds")
    save(fig, out, "metric_diagnostics")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    results, manifest = freeze_evidence(
        args.work_dir, allow_incomplete=args.allow_incomplete
    )
    merged = args.work_dir / "merged"
    out = args.out_dir or (args.work_dir / "paper_artifacts" / "figures")
    out.mkdir(parents=True, exist_ok=True)

    if "matched_confirmation" in results:
        rows = load_jsonl(merged / "matched_confirm.jsonl")
        plot_matched_effect(results, out)
        plot_matched_pairs(rows, out)
        plot_training_curves(rows, out)
        plot_efficiency(rows, out)
        plot_metric_diagnostics(rows, out)

    if "cross_configuration_isolation" in results:
        plot_xconfig(results, out)

    if "mechanism_ablation" in results:
        plot_ablation_effects(results, out)

    if "long_horizon_transfer" in results:
        plot_transfer(
            load_jsonl(merged / "horizon_with_astro.jsonl"),
            out,
            stem="transfer_horizon",
            title="Long-horizon transfer: 124M, 2700 steps",
        )

    if "scale_transfer" in results:
        plot_transfer(
            load_jsonl(merged / "scale_with_astro.jsonl"),
            out,
            stem="transfer_scale",
            title="Scale transfer: 355M, 900 steps",
        )

    if "broad_independently_tuned_context" in results:
        plot_broad_confirmation(load_jsonl(merged / "confirm.jsonl"), out)

    print(
        json.dumps(
            {
                "figures": str(out),
                "status": manifest["status"],
                "generated": sorted(p.name for p in out.glob("*.pdf")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
