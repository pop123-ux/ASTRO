#!/usr/bin/env python3
"""Collect the clean ASTRO paper campaign into one plotting artifact.

Inputs are deliberately narrow:

* the canonical ``paper_campaign.py`` headline state;
* the frozen-config ``paper_mechanism.py`` state;
* optional ``paper_transfer.py`` states for scale and long-horizon checks.

No historical ``paper_results.json`` value is imported, so final figures cannot
silently mix the old confounded harness with the confirmatory campaign.
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


def headline_block(state: dict | None) -> dict:
    if state is None:
        return {"complete": False, "optimizers": {}}
    seeds = list(range(100, 105))
    out = {"complete": True, "size": "124M", "steps": 900, "optimizers": {}}
    for name in HEADLINE:
        rows = []
        for seed in seeds:
            entry = state.get("runs", {}).get(f"124M|900|{name}|{seed}")
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
            "config": state.get("tuned", {}).get(name),
            "complete": len(rows) == 5,
        }
        out["complete"] &= len(rows) == 5

    if out["optimizers"].get("muon", {}).get("loss"):
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


def custom_block(state: dict | None, names) -> dict:
    if state is None:
        return {"complete": False, "protocol": None, "optimizers": {}}
    protocol = state.get("protocol", {})
    seeds = list(protocol.get("seeds", []))
    out = {"complete": True, "protocol": protocol, "optimizers": {}}
    for name in names:
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
            "complete": bool(seeds) and len(rows) == len(seeds),
        }
        out["complete"] &= bool(seeds) and len(rows) == len(seeds)

    if "muon" in out["optimizers"] and "astro_v2" in out["optimizers"]:
        mrow = out["optimizers"]["muon"]
        arow = out["optimizers"]["astro_v2"]
        m = {s: v for s, v in zip(mrow["seeds"], mrow["loss"])}
        a = {s: v for s, v in zip(arow["seeds"], arow["loss"])}
        common = sorted(set(m) & set(a))
        deltas = [a[s] - m[s] for s in common]
        arow["paired_vs_muon"] = {
            "seeds": common,
            "delta": deltas,
            "mean_delta": mean(deltas),
            "wins": sum(delta < 0 for delta in deltas),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, required=True,
                        help="paper_main/astro_lab_state.json")
    parser.add_argument("--mechanism", type=Path,
                        help="paper_mechanism/paper_mechanism_state.json")
    parser.add_argument("--scale", type=Path,
                        help="paper_scale_355m/paper_transfer_state.json")
    parser.add_argument("--horizon", type=Path,
                        help="paper_horizon_2700/paper_transfer_state.json")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "artifacts" / "paper_campaign.json")
    args = parser.parse_args()

    main_state = load(args.main)
    payload = {
        "schema_version": 3,
        "principle": (
            "Only confirmatory paper_campaign.py / paper_mechanism.py / "
            "paper_transfer.py results are eligible. Historical astro_lab "
            "measurements are excluded."
        ),
        "paper_protocol": main_state.get("paper_protocol") if main_state else None,
        "main_124m_900": headline_block(main_state),
        "mechanism_124m_900": custom_block(load(args.mechanism), MECHANISM),
        "scale_355m_900": custom_block(load(args.scale), ("muon", "astro_v2")),
        "horizon_124m_2700": custom_block(load(args.horizon), ("muon", "astro_v2")),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    for key in (
        "main_124m_900",
        "mechanism_124m_900",
        "scale_355m_900",
        "horizon_124m_2700",
    ):
        print(f"  {key:24s} complete={payload[key]['complete']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
