#!/usr/bin/env python3
"""Collect paper-campaign states into one plotting/paper artifact.

This script performs no training and never invents missing values. Stages that
have not been run remain explicitly incomplete. The resulting
``artifacts/paper_campaign.json`` is the only input used by the new paper-grade
figures, keeping historical/confounded measurements out of the final plots.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text())


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _sd(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def _main_block(state: dict | None, size: str = "124M", steps: int = 900) -> dict:
    if state is None:
        return {"complete": False, "optimizers": {}}
    # AdamW is the standard reference; Muon, NorMuon, and faithful AdaMuon are
    # the closest matrix-optimizer prior art to the ASTRO recipe.
    names = ["adamw", "muon", "normuon", "adamuon_ref", "astro_v2"]
    out: dict = {"complete": True, "optimizers": {}}
    for name in names:
        rows = []
        for seed in range(100, 105):
            key = f"{size}|{steps}|{name}|{seed}"
            if key in state.get("runs", {}):
                rows.append((seed, state["runs"][key]))
        values = [float(row[1]["value"]) for row in rows]
        seconds = [float(row[1]["seconds"]) for row in rows]
        out["optimizers"][name] = {
            "config": state.get("tuned", {}).get(name),
            "seeds": [row[0] for row in rows],
            "loss": values,
            "seconds": seconds,
            "mean": _mean(values),
            "sample_sd": _sd(values),
            "mean_seconds": _mean(seconds),
            "complete": len(rows) == 5,
        }
        out["complete"] &= len(rows) == 5

    muon = out["optimizers"]["muon"]
    m = {s: v for s, v in zip(muon["seeds"], muon["loss"])}
    for name in ("adamw", "normuon", "adamuon_ref", "astro_v2"):
        other = out["optimizers"][name]
        common = sorted(set(muon["seeds"]) & set(other["seeds"]))
        o = {s: v for s, v in zip(other["seeds"], other["loss"])}
        deltas = [o[s] - m[s] for s in common]
        other["paired_vs_muon"] = {
            "seeds": common,
            "delta": deltas,
            "mean_delta": _mean(deltas),
            "wins": sum(d < 0 for d in deltas),
        }
    return out


def _stage_block(state: dict | None) -> dict:
    if state is None:
        return {"complete": False, "protocol": None, "optimizers": {}}
    protocol = state.get("protocol", {})
    seeds = list(protocol.get("seeds", []))
    names = list(protocol.get("optimizers", []))
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
            "mean": _mean(values),
            "sample_sd": _sd(values),
            "mean_seconds": _mean(seconds),
            "config": rows[0][1].get("config") if rows else None,
            "complete": len(rows) == len(seeds) and bool(seeds),
        }
        out["complete"] &= len(rows) == len(seeds) and bool(seeds)
    return out


def _traces(directory: Path | None) -> dict:
    if directory is None or not directory.is_dir():
        return {}
    out: dict[str, dict[str, dict]] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        name = payload.get("optimizer")
        seed = payload.get("seed")
        if name is None or seed is None:
            continue
        out.setdefault(name, {})[str(seed)] = {
            "step": payload.get("train_trace", {}).get("step", []),
            "loss": payload.get("train_trace", {}).get("loss", []),
            "seconds": payload.get("train_trace", {}).get("seconds", []),
            "final_val_loss": payload.get("final_val_loss"),
            "peak_vram_gb": payload.get("peak_vram_gb"),
            "corpus_sha256": payload.get("corpus_sha256"),
            "paper_lab_sha256": payload.get("paper_lab_sha256"),
            "astro_lab_sha256": payload.get("astro_lab_sha256"),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, required=True,
                        help="main/astro_lab_state.json")
    parser.add_argument("--ablation", type=Path)
    parser.add_argument("--scale", type=Path)
    parser.add_argument("--horizon", type=Path)
    parser.add_argument("--main-traces", type=Path)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "artifacts" / "paper_campaign.json")
    args = parser.parse_args()

    payload = {
        "schema_version": 1,
        "principle": (
            "Only results produced by scripts/paper_lab.py or paper_stage.py belong "
            "in this artifact. Historical astro_lab results remain separate."
        ),
        "main_124m_900": _main_block(_load(args.main)),
        "ablation_124m_900": _stage_block(_load(args.ablation)),
        "scale_355m_900": _stage_block(_load(args.scale)),
        "horizon_124m_2700": _stage_block(_load(args.horizon)),
        "main_traces": _traces(args.main_traces),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(f"wrote {args.out}")
    for key in (
        "main_124m_900",
        "ablation_124m_900",
        "scale_355m_900",
        "horizon_124m_2700",
    ):
        print(f"  {key:24s} complete={payload[key]['complete']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
