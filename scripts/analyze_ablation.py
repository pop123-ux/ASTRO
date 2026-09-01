#!/usr/bin/env python3
"""Aggregate distributed ASTRO ablation outputs into paper-ready summaries.

The ablation launcher intentionally creates many small, isolated work
-directories so three or more Colab sessions can run independently. This tool
-turns that directory tree back into one machine-readable summary without
-touching the GPU.

It recursively discovers ``astro_lab_state.json`` files, extracts completed
runs and tuning trials, groups them by optimizer/size/steps/configuration, and
writes:

  ablation_summary.json   complete normalized records and aggregates
  ablation_summary.md     readable tables plus coverage diagnostics

Use the raw state files as the source of truth; this file never overwrites
individual experiment artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def walk_states(root: Path) -> list[Path]:
    return sorted(root.rglob("astro_lab_state.json"))


def parse_key(key: str):
    parts = key.split("|")
    if len(parts) >= 4:
        return parts[0], int(parts[1]), parts[2], parts[3]
    return None


def load_records(root: Path) -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    warnings: list[str] = []
    for path in walk_states(root):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"could not read {path}: {exc}")
            continue
        for section in ("runs", "trials"):
            table = state.get(section, {})
            if not isinstance(table, dict):
                warnings.append(f"{path}: {section} is not an object")
                continue
            for key, entry in table.items():
                parsed = parse_key(str(key))
                if parsed is None or not isinstance(entry, dict):
                    continue
                size, steps, optimizer, index = parsed
                value = entry.get("value")
                records.append({
                    "source": str(path),
                    "section": section,
                    "size": size,
                    "steps": steps,
                    "optimizer": optimizer,
                    "index": index,
                    "value": value,
                    "seconds": entry.get("seconds"),
                    "config": entry.get("config"),
                })
    return records, warnings


def numeric(values):
    return [float(v) for v in values
            if isinstance(v, (int, float)) and math.isfinite(float(v))]


def aggregate(records: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in records:
        if r["value"] is not None and r["section"] in ("runs", "trials"):
            groups[(r["section"], r["size"], r["steps"], r["optimizer"])].append(r)

    out = {}
    for key, rows in sorted(groups.items()):
        values = numeric([r["value"] for r in rows])
        seconds = numeric([r["seconds"] for r in rows])
        if not values:
            continue
        skey = "|".join(map(str, key))
        out[skey] = {
            "section": key[0],
            "size": key[1],
            "steps": key[2],
            "optimizer": key[3],
            "n": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "mean_seconds": statistics.fmean(seconds) if seconds else None,
            "values": values,
        }
    return out


def coverage(records: list[dict]) -> dict:
    keys = defaultdict(set)
    for r in records:
        if r["value"] is not None:
            keys[(r["section"], r["size"], r["steps"])].add(r["optimizer"])
    return {"|".join(map(str, k)): sorted(v) for k, v in sorted(keys.items())}


def render_markdown(records: list[dict], aggs: dict, warnings: list[str]) -> str:
    lines = [
        "# ASTRO Ablation Aggregate",
        "",
        f"Completed normalized records: **{len(records)}**",
        f"Aggregate cells: **{len(aggs)}**",
        "",
        "## Evaluation results",
        "",
        "| section | size | steps | optimizer | n | mean loss | median | best | worst | stdev | mean sec |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = sorted(aggs.values(), key=lambda x: (x["section"], x["size"], x["steps"], x["mean"]))
    for a in rows:
        lines.append(
            f"| {a['section']} | {a['size']} | {a['steps']} | `{a['optimizer']}` | "
            f"{a['n']} | {a['mean']:.5f} | {a['median']:.5f} | {a['min']:.5f} | "
            f"{a['max']:.5f} | {a['stdev']:.5f} | "
            f"{'—' if a['mean_seconds'] is None else f'{a[\"mean_seconds\"]:.1f}'} |"
        )

    lines += ["", "## Coverage", ""]
    cov = coverage(records)
    for cell, opts in cov.items():
        lines.append(f"- `{cell}`: {len(opts)} optimizers — " + ", ".join(f"`{o}`" for o in opts))

    if warnings:
        lines += ["", "## Read warnings", ""]
        lines.extend(f"- {w}" for w in warnings)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="ablation root to scan")
    parser.add_argument("--out", default=None, help="output directory; defaults to --root")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"root does not exist: {root}")
    out = Path(args.out).expanduser().resolve() if args.out else root
    out.mkdir(parents=True, exist_ok=True)

    records, warnings = load_records(root)
    aggs = aggregate(records)
    payload = {
        "schema_version": 1,
        "root": str(root),
        "state_files": [str(p) for p in walk_states(root)],
        "record_count": len(records),
        "aggregate_count": len(aggs),
        "records": records,
        "aggregates": aggs,
        "coverage": coverage(records),
        "warnings": warnings,
    }
    (out / "ablation_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (out / "ablation_summary.md").write_text(render_markdown(records, aggs, warnings))

    print(f"state files: {len(payload['state_files'])}")
    print(f"records:     {len(records)}")
    print(f"aggregates:  {len(aggs)}")
    print(f"json:        {out / 'ablation_summary.json'}")
    print(f"markdown:    {out / 'ablation_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
