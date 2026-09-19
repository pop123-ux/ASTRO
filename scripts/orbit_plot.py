#!/usr/bin/env python3
"""Generate deterministic ORBIT paper figures from merged JSONL only."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def group(rows: list[dict]):
    out = defaultdict(list)
    for row in rows:
        out[row["optimizer"]].append(row)
    return dict(out)


def save_confirm(rows: list[dict], out: Path) -> None:
    grouped = group(rows)
    names = sorted(grouped)
    means = [statistics.fmean(r["val_loss"] for r in grouped[n]) for n in names]
    errors = [
        statistics.stdev(r["val_loss"] for r in grouped[n]) if len(grouped[n]) > 1 else 0.0
        for n in names
    ]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(names, means, yerr=errors, capsize=4)
    ax.set_ylabel("Validation loss")
    ax.set_title("ORBIT held-out confirmation")
    ax.tick_params(axis="x", rotation=28)
    fig.tight_layout()
    fig.savefig(out / "confirm_final_loss.pdf")
    fig.savefig(out / "confirm_final_loss.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for name in names:
        xs = [r["seconds"] / 60 for r in grouped[name]]
        ys = [r["val_loss"] for r in grouped[name]]
        ax.scatter(xs, ys, label=name, alpha=0.8)
    ax.set_xlabel("Wall-clock minutes")
    ax.set_ylabel("Validation loss")
    ax.set_title("Efficiency, held-out runs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "confirm_efficiency.pdf")
    fig.savefig(out / "confirm_efficiency.png", dpi=180)
    plt.close(fig)

    curves = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for point in row.get("curve", []):
            curves[row["optimizer"]][int(point["step"])].append(float(point["train_loss"]))
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for name in names:
        steps = sorted(curves[name])
        if not steps:
            continue
        values = [statistics.fmean(curves[name][s]) for s in steps]
        ax.plot(steps, values, label=name)
    ax.set_xlabel("Optimizer steps")
    ax.set_ylabel("Training loss")
    ax.set_title("Mean held-out training trajectories")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "confirm_training_curves.pdf")
    fig.savefig(out / "confirm_training_curves.png", dpi=180)
    plt.close(fig)


def save_bars(rows: list[dict], title: str, stem: str, out: Path) -> None:
    grouped = group(rows)
    if not grouped:
        return
    names = sorted(grouped)
    means = [statistics.fmean(r["val_loss"] for r in grouped[n]) for n in names]
    errors = [
        statistics.stdev(r["val_loss"] for r in grouped[n]) if len(grouped[n]) > 1 else 0.0
        for n in names
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(names, means, yerr=errors, capsize=4)
    ax.set_ylabel("Validation loss")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=28)
    fig.tight_layout()
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=180)
    plt.close(fig)


def save_diagnostics(rows: list[dict], out: Path) -> None:
    orbit = [r for r in rows if r["optimizer"] == "orbit" and r.get("optimizer_diagnostics")]
    if not orbit:
        return
    keys = sorted(orbit[0]["optimizer_diagnostics"])
    vals = [statistics.fmean(r["optimizer_diagnostics"].get(k, 0.0) for r in orbit) for k in keys]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.bar(keys, vals)
    ax.set_title("ORBIT functional-metric diagnostics")
    ax.tick_params(axis="x", rotation=24)
    fig.tight_layout()
    fig.savefig(out / "orbit_metric_diagnostics.pdf")
    fig.savefig(out / "orbit_metric_diagnostics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path)
    args = p.parse_args()
    merged = args.work_dir / "merged"
    out = args.out_dir or (args.work_dir / "figures")
    out.mkdir(parents=True, exist_ok=True)

    confirm = load_jsonl(merged / "confirm.jsonl")
    if confirm:
        save_confirm(confirm, out)
        save_diagnostics(confirm, out)
    save_bars(load_jsonl(merged / "ablation.jsonl"), "ORBIT mechanism ablations", "ablations", out)
    save_bars(load_jsonl(merged / "horizon.jsonl"), "Long-horizon comparison", "horizon", out)
    save_bars(load_jsonl(merged / "scale.jsonl"), "355M scale transfer", "scale", out)
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
