#!/usr/bin/env python3
"""Audit the minimal ASTRO confirmatory campaign.

The audit checks the evidence the paper actually needs: five-seed headline
comparison, frozen causal controls, a two-seed 355M transfer, and a two-seed
2700-step durability check.  It never treats an inconvenient scientific result
as an error; ``--strict`` fails only for missing or provenance-inconsistent
measurements.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

HEADLINE = ("adamw", "muon", "normuon", "adamuon_ref", "astro_v2")
MECHANISM = (
    "muon",
    "muon_m90",
    "astro_v2_gamma0",
    "astro_v2_nosplit",
    "astro_v2_blockwise",
    "astro_v2",
)


def load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    payload.setdefault("runs", {})
    payload.setdefault("trials", {})
    payload.setdefault("tuned", {})
    return payload


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def sign_p(deltas: list[float]) -> float:
    if not deltas:
        return float("nan")
    n = len(deltas)
    wins = sum(delta < 0 for delta in deltas)
    tail = sum(math.comb(n, k) for k in range(min(wins, n - wins) + 1))
    return min(1.0, 2 * tail / 2**n)


def run_key(size: str, steps: int, opt: str, seed: int) -> str:
    return f"{size}|{steps}|{opt}|{seed}"


def seed_rows(state: dict, size: str, steps: int, opt: str, seeds: list[int]):
    rows, missing = [], []
    for seed in seeds:
        entry = state["runs"].get(run_key(size, steps, opt, seed))
        if entry is None:
            missing.append(seed)
        else:
            rows.append((seed, entry))
    return rows, missing


def config_key(config: dict) -> tuple:
    return tuple(sorted((key, round(float(value), 14)) for key, value in config.items()))


def add_frozen_pair_section(lines: list[str], incomplete: list[str], state: dict,
                            title: str, size: str, steps: int, seeds: list[int]) -> None:
    lines += ["", f"## {title}", "",
              "| optimizer | mean | sample sd | mean s/run | seeds |",
              "|---|---:|---:|---:|---:|"]
    values = {}
    for opt in ("muon", "astro_v2"):
        rows, missing = seed_rows(state, size, steps, opt, seeds)
        if missing:
            incomplete.append(f"{size}/{steps} {opt} missing seeds {missing}")
        vals = [float(entry["value"]) for _, entry in rows]
        secs = [float(entry["seconds"]) for _, entry in rows]
        values[opt] = {seed: float(entry["value"]) for seed, entry in rows}
        if vals:
            mu, sd = mean_sd(vals)
            lines.append(
                f"| `{opt}` | {mu:.5f} | {sd:.5f} | {statistics.fmean(secs):.1f} | "
                f"{len(vals)}/{len(seeds)} |"
            )
        else:
            lines.append(f"| `{opt}` | — | — | — | 0/{len(seeds)} |")

    common = sorted(set(values.get("muon", {})) & set(values.get("astro_v2", {})))
    if common:
        deltas = [values["astro_v2"][seed] - values["muon"][seed] for seed in common]
        lines += ["", f"ASTRO-v2 − Muon: mean `{statistics.fmean(deltas):+.5f}`, "
                  f"wins `{sum(d < 0 for d in deltas)}/{len(deltas)}`."]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-state", type=Path, required=True)
    parser.add_argument("--mechanism-state", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("paper_evidence_summary.md"))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    state = load(args.main_state)
    mech = json.loads(args.mechanism_state.read_text())
    incomplete: list[str] = []
    lines = ["# ASTRO paper evidence audit", ""]

    if not state.get("paper_protocol"):
        incomplete.append("main state has no paper_campaign protocol signature")
    main_digest = state.get("paper_protocol", {}).get("code_digest")
    mech_main_digest = mech.get("protocol", {}).get("main_protocol", {}).get("code_digest")
    if main_digest and mech_main_digest and main_digest != mech_main_digest:
        incomplete.append("mechanism campaign was not derived from the same paper-campaign code digest")

    # Headline five-seed experiment ---------------------------------------
    seeds = list(range(100, 105))
    lines += ["## 124M / 900-step held-out confirmation", "",
              "| optimizer | mean | sample sd | mean s/run | seeds |",
              "|---|---:|---:|---:|---:|"]
    by_seed = {}
    for opt in HEADLINE:
        rows, missing = seed_rows(state, "124M", 900, opt, seeds)
        if missing:
            incomplete.append(f"124M/900 {opt} missing seeds {missing}")
        values = [float(entry["value"]) for _, entry in rows]
        seconds = [float(entry["seconds"]) for _, entry in rows]
        by_seed[opt] = {seed: float(entry["value"]) for seed, entry in rows}
        configs = [entry.get("config", {}) for _, entry in rows]
        if configs and any(config_key(cfg) != config_key(configs[0]) for cfg in configs[1:]):
            incomplete.append(f"124M/900 {opt} held-out seeds used different configs")
        if values:
            mu, sd = mean_sd(values)
            lines.append(
                f"| `{opt}` | {mu:.5f} | {sd:.5f} | {statistics.fmean(seconds):.1f} | "
                f"{len(values)}/5 |"
            )
        else:
            lines.append(f"| `{opt}` | — | — | — | 0/5 |")

    if by_seed.get("muon"):
        lines += ["", "Paired against Muon:", "",
                  "| optimizer | mean paired Δ | worst Δ | wins | two-sided sign p |",
                  "|---|---:|---:|---:|---:|"]
        for opt in HEADLINE:
            if opt == "muon":
                continue
            common = sorted(set(by_seed["muon"]) & set(by_seed.get(opt, {})))
            if not common:
                continue
            deltas = [by_seed[opt][seed] - by_seed["muon"][seed] for seed in common]
            lines.append(
                f"| `{opt}` | {statistics.fmean(deltas):+.5f} | {max(deltas):+.5f} | "
                f"{sum(d < 0 for d in deltas)}/{len(deltas)} | {sign_p(deltas):.4f} |"
            )

    # Frozen mechanism campaign -------------------------------------------
    lines += ["", "## Frozen causal controls", "",
              "| contrast (A − B) | paired seeds | mean Δ | worst Δ | A wins |",
              "|---|---:|---:|---:|---:|"]
    mech_seeds = list(mech.get("protocol", {}).get("seeds", [200, 201]))
    for name in MECHANISM:
        missing = [seed for seed in mech_seeds if f"{name}|{seed}" not in mech.get("runs", {})]
        if missing:
            incomplete.append(f"mechanism {name} missing seeds {missing}")

    contrasts = [
        ("muon_m90", "muon", "matrix beta .90 vs .95"),
        ("astro_v2", "astro_v2_gamma0", "post-spectral row adaptation"),
        ("astro_v2", "astro_v2_nosplit", "semantic Q/K/V split"),
        ("astro_v2", "astro_v2_blockwise", "global vs blockwise redistribution"),
    ]
    for a, b, label in contrasts:
        deltas = []
        for seed in mech_seeds:
            aa = mech.get("runs", {}).get(f"{a}|{seed}")
            bb = mech.get("runs", {}).get(f"{b}|{seed}")
            if aa is not None and bb is not None:
                deltas.append(float(aa["value"]) - float(bb["value"]))
        if deltas:
            lines.append(
                f"| {label}: `{a}` − `{b}` | {len(deltas)} | "
                f"{statistics.fmean(deltas):+.5f} | {max(deltas):+.5f} | "
                f"{sum(delta < 0 for delta in deltas)}/{len(deltas)} |"
            )
        else:
            lines.append(f"| {label}: `{a}` − `{b}` | 0 | — | — | — |")

    lines += ["", "The `astro_v2` vs `astro_v2_blockwise` contrast isolates the "
              "candidate ASTRO-specific operation identified by the 2026 prior-art audit: "
              "one global redistribution/restoration after semantic blocks are independently "
              "polarised. Empirical value does not by itself prove exhaustive literature novelty."]

    # Two minimal robustness checks ---------------------------------------
    add_frozen_pair_section(lines, incomplete, state,
                            "355M / 900-step frozen scale transfer",
                            "355M", 900, [100, 101])
    add_frozen_pair_section(lines, incomplete, state,
                            "124M / 2700-step frozen durability check",
                            "124M", 2700, [100, 101])

    lines += ["", "## Completeness", ""]
    if incomplete:
        lines += [f"- INCOMPLETE: {item}" for item in incomplete]
    else:
        lines.append("All measurements required by the minimal paper campaign are complete.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")
    return 1 if args.strict and incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
