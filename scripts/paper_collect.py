#!/usr/bin/env python3
"""Collect the clean ASTRO paper campaign into one plotting artifact.

Inputs are deliberately narrow:

* the canonical ``paper_campaign.py`` state, which contains the tuned 124M
  result and later frozen 355M / 2700 transfer cells;
* the frozen-config ``paper_mechanism.py`` state.

No historical ``paper_results.json`` value is imported, so the final figures
cannot silently mix the old confounded harness with the confirmatory campaign.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADLINE = ("adamw", "muon", "normuon", "adamuon_ref", "astro_v2")
MECHANISM = (
    "muon",
    "muon_m90",
    "astro_v2_gamma0",
    "astro_v2_nosplit",
    "astro_v2_blockwise",
    "astro_v2",
)


def load(path: Path | None) -> dict | None:
    return json.loads(path.read_text()) if path is not None and path.is_file() else None


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def sd(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def run_block(state: dict | None, size: str, steps: int, names, seeds) -> dict:
    if state is None:
        return {"complete": False, "size": size, "steps": steps, "optimizers": {}}
    out = {"complete": True, "size": size, "steps": steps, "optimizers": {}}
    for name in names:
        rows = []
        for seed in seeds:
            entry = state.get("runs", {}).get(f"{size}|{steps}|{name}|{seed}")
            if entry is not None:
                rows.append((seed, entry))
        values = [float(entry["value"]) for _, entry in rows]
        seconds = [float(entry["seconds"]) for _, entry in rows]
        out["optimizers"][name] = {
            "seeds": [seed for seed, _ in rows],
            "loss": values,
            "seconds": seconds,
            "mean": mean(values),
            "sample_sd": sd(values),
            "mean_seconds": mean(seconds),
            "config": rows[0][1].get("config") if rows else state.get("tuned", {}).get(name),
            "complete": len(rows) == len(seeds),
        }
        out["complete"] &= len(rows) == len(seeds)

    if "muon" in out["optimizers"] and out["optimizers"]["muon"]["loss"]:
        mrow = out["optimizers"]["muon"]
        m = {s: v for s, v in zip(mrow["seeds"], mrow["loss"])}
        for name, row in out["optimizers"].items():
            if name == "muon":
                continue
            other = {s: v for s, v in zip(row["seeds"], row["loss"])}
            common = sorted(set(m) & set(other))
            deltas = [other[s] - m[s] for s in common]
            row["paired_vs_muon"] = {
                "seeds": common,
                "delta": deltas,
                "mean_delta": mean(deltas),
                "wins": sum(delta < 0 for delta in deltas),
            }
    return out


def mechanism_block(state: dict | None) -> dict:
    if state is None:
        return {"complete": False, "optimizers": {}, "protocol": None}
    seeds = list(state.get("protocol", {}).get("seeds", [200, 201]))
    out = {"complete": True, "protocol": state.get("protocol"), "optimizers": {}}
    for name in MECHANISM:
        rows = []
        for seed in seeds:
            entry = state.get("runs", {}).get(f"{name}|{seed}")
            if entry is not None:
                rows.append((seed, entry))
        values = [float(entry["value"]) for _, entry in rows]
        seconds = [float(entry["seconds"]) for _, entry in rows]
        out["optimizers"][name] = {
            "seeds": [seed for seed, _ in rows],
            "loss": values,
            "seconds": seconds,
            "mean": mean(values),
            "sample_sd": sd(values),
            "mean_seconds": mean(seconds),
            "config": rows[0][1].get("config") if rows else None,
            "complete": len(rows) == len(seeds),
        }
        out["complete"] &= len(rows) == len(seeds)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, required=True,
                        help="paper_main/astro_lab_state.json")
    parser.add_argument("--mechanism", type=Path,
                        help="paper_mechanism/paper_mechanism_state.json")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "artifacts" / "paper_campaign.json")
    args = parser.parse_args()

    main_state = load(args.main)
    mech_state = load(args.mechanism)
    payload = {
        "schema_version": 2,
        "principle": (
            "Only confirmatory paper_campaign.py / paper_mechanism.py results are "
            "eligible. Historical astro_lab measurements are excluded."
        ),
        "paper_protocol": main_state.get("paper_protocol") if main_state else None,
        "main_124m_900": run_block(main_state, "124M", 900, HEADLINE, range(100, 105)),
        "mechanism_124m_900": mechanism_block(mech_state),
        "scale_355m_900": run_block(main_state, "355M", 900, ("muon", "astro_v2"), range(100, 102)),
        "horizon_124m_2700": run_block(main_state, "124M", 2700, ("muon", "astro_v2"), range(100, 102)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    for key in ("main_124m_900", "mechanism_124m_900", "scale_355m_900", "horizon_124m_2700"):
        print(f"  {key:24s} complete={payload[key]['complete']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
