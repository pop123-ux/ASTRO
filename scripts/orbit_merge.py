#!/usr/bin/env python3
"""Merge append-only ORBIT shard outputs and freeze discovery winners.

Historical rows are intentionally retained in the shard JSONL logs for audit
provenance. Merge only consumes rows produced by the *current* ORBIT code
digest, so a numerical/algorithmic patch does not require deleting old results.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from orbit_campaign import code_digest


DISCOVERY_TRIALS = 5
DISCOVERY_OPTIMIZERS = (
    "adamw",
    "muon",
    "normuon",
    "adamuon_ref",
    "astro_v2",
    "orbit",
)


def load_rows(work_dir: Path, phase: str, *, digest: str) -> tuple[list[dict], int]:
    """Load latest successful row per task for the current code digest only."""
    rows: dict[str, dict] = {}
    stale_ok_rows = 0
    for path in sorted((work_dir / "shards").glob(f"shard-*/{phase}.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") != "ok":
                continue
            if row.get("code_digest") != digest:
                stale_ok_rows += 1
                continue
            # Append-only files may contain a rerun of the same task. Keep the
            # latest successful current-digest row encountered for that task.
            rows[row["task_id"]] = row
    ordered = sorted(
        rows.values(),
        key=lambda r: (r["optimizer"], r["seed"], str(r.get("trial"))),
    )
    return ordered, stale_ok_rows


def summarize(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["optimizer"]].append(row)
    summary = {}
    for name, items in sorted(grouped.items()):
        losses = [float(x["val_loss"]) for x in items]
        times = [float(x["seconds"]) for x in items]
        summary[name] = {
            "n": len(items),
            "mean_val_loss": statistics.fmean(losses),
            "sd_val_loss": statistics.stdev(losses) if len(losses) > 1 else 0.0,
            "median_val_loss": statistics.median(losses),
            "mean_seconds": statistics.fmean(times),
            "mean_peak_cuda_gb": statistics.fmean(
                [float(x.get("peak_cuda_bytes", 0)) / (1024**3) for x in items]
            ),
        }
    return summary


def freeze_discovery(rows: list[dict]) -> dict:
    best = {}
    for row in rows:
        name = row["optimizer"]
        if name not in best or row["val_loss"] < best[name]["val_loss"]:
            best[name] = {
                "val_loss": row["val_loss"],
                "trial": row.get("trial"),
                "config": row["config"],
                "task_id": row["task_id"],
                "code_digest": row["code_digest"],
                "environment": row["environment"],
            }
    return best


def require_complete_discovery(rows: list[dict]) -> None:
    """Never freeze discovery configs from a partial current-digest campaign."""
    counts = Counter(row["optimizer"] for row in rows)
    problems = []
    for name in DISCOVERY_OPTIMIZERS:
        got = counts.get(name, 0)
        if got != DISCOVERY_TRIALS:
            problems.append(f"{name}={got}/{DISCOVERY_TRIALS}")
    extras = sorted(set(counts) - set(DISCOVERY_OPTIMIZERS))
    if extras:
        problems.append("unexpected=" + ",".join(extras))
    expected_total = len(DISCOVERY_OPTIMIZERS) * DISCOVERY_TRIALS
    if len(rows) != expected_total:
        problems.append(f"total={len(rows)}/{expected_total}")
    if problems:
        raise SystemExit(
            "discovery incomplete for current code digest; do not freeze configs yet: "
            + "; ".join(problems)
        )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument(
        "--phase",
        choices=("discovery", "confirm", "ablation", "horizon", "scale", "all"),
        default="all",
    )
    args = p.parse_args()
    phases = (
        ("discovery", "confirm", "ablation", "horizon", "scale")
        if args.phase == "all"
        else (args.phase,)
    )
    current_digest = code_digest()
    print(f"merge current digest={current_digest[:12]}")

    merged_dir = args.work_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    full_summary = {}
    existing_summary = merged_dir / "summary.json"
    if existing_summary.exists():
        full_summary.update(json.loads(existing_summary.read_text()))

    for phase in phases:
        rows, stale_ok_rows = load_rows(args.work_dir, phase, digest=current_digest)
        if stale_ok_rows:
            print(
                f"{phase}: ignored {stale_ok_rows} successful historical rows "
                "from stale code digests"
            )
        if not rows:
            print(f"{phase}: no completed rows for current digest {current_digest[:12]}")
            continue

        # This is mostly defensive because load_rows already filters the digest.
        digests = {row["code_digest"] for row in rows}
        if digests != {current_digest}:
            raise SystemExit(
                f"{phase}: internal digest filter failure: {sorted(digests)}"
            )
        envs = {json.dumps(row["environment"], sort_keys=True) for row in rows}
        if len(envs) != 1:
            raise SystemExit(
                f"{phase}: mixed Python/package environments detected. "
                "Recreate matching Colabs before merging."
            )

        if phase == "discovery":
            require_complete_discovery(rows)

        write_jsonl(merged_dir / f"{phase}.jsonl", rows)
        phase_summary = summarize(rows)
        full_summary[phase] = phase_summary
        (merged_dir / f"{phase}_summary.json").write_text(
            json.dumps(phase_summary, indent=2, sort_keys=True) + "\n"
        )
        if phase == "discovery":
            best = freeze_discovery(rows)
            (merged_dir / "best_configs.json").write_text(
                json.dumps(best, indent=2, sort_keys=True) + "\n"
            )
            print("discovery configurations frozen -> merged/best_configs.json")
        print(f"{phase}: merged {len(rows)} unique completed current-digest tasks")

    (merged_dir / "summary.json").write_text(
        json.dumps(full_summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
