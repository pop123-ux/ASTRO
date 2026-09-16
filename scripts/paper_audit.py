#!/usr/bin/env python3
"""Audit the minimal ASTRO confirmatory campaign.

``--strict`` fails for missing/provenance-inconsistent evidence, never because a
scientific result is negative. The output is a claim-safe summary for writing.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

HEADLINE = ("adamw", "muon", "normuon", "adamuon_ref", "astro_v2")
MECHANISM = (
    "muon", "muon_m90", "astro_v2_gamma0", "astro_v2_nosplit",
    "astro_v2_blockwise", "astro_v2",
)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def sign_p(deltas: list[float]) -> float:
    if not deltas:
        return float("nan")
    n = len(deltas)
    wins = sum(delta < 0 for delta in deltas)
    tail = sum(math.comb(n, k) for k in range(min(wins, n - wins) + 1))
    return min(1.0, 2 * tail / 2**n)


def config_key(config: dict) -> tuple:
    return tuple(sorted((key, round(float(value), 14)) for key, value in config.items()))


def headline_rows(state: dict, opt: str, seeds: list[int]):
    rows, missing = [], []
    for seed in seeds:
        entry = state.get("runs", {}).get(f"124M|900|{opt}|{seed}")
        (missing if entry is None else rows).append(seed if entry is None else (seed, entry))
    return rows, missing


def custom_rows(state: dict, opt: str):
    seeds = list(state.get("protocol", {}).get("seeds", []))
    rows, missing = [], []
    for seed in seeds:
        entry = state.get("runs", {}).get(f"{opt}|{seed}")
        (missing if entry is None else rows).append(seed if entry is None else (seed, entry))
    return seeds, rows, missing


def validate_derived_protocol(label: str, state: dict, main_digest: str | None,
                              incomplete: list[str]) -> None:
    protocol = state.get("protocol", {})
    derived = protocol.get("main_protocol", {}).get("code_digest")
    if not protocol:
        incomplete.append(f"{label} state has no protocol")
    elif main_digest and derived != main_digest:
        incomplete.append(f"{label} was not derived from the headline campaign code digest")


def add_transfer(lines: list[str], incomplete: list[str], label: str,
                 state: dict, main_digest: str | None) -> None:
    validate_derived_protocol(label, state, main_digest, incomplete)
    lines += ["", f"## {label}", "",
              "| optimizer | mean | sample sd | mean s/run | seeds |",
              "|---|---:|---:|---:|---:|"]
    values_by = {}
    for opt in ("muon", "astro_v2"):
        seeds, rows, missing = custom_rows(state, opt)
        if missing:
            incomplete.append(f"{label} {opt} missing seeds {missing}")
        values = [float(entry["value"]) for _, entry in rows]
        seconds = [float(entry["seconds"]) for _, entry in rows]
        values_by[opt] = {seed: float(entry["value"]) for seed, entry in rows}
        if values:
            mu, sd = mean_sd(values)
            lines.append(
                f"| `{opt}` | {mu:.5f} | {sd:.5f} | {statistics.fmean(seconds):.1f} | "
                f"{len(values)}/{len(seeds)} |"
            )
        else:
            lines.append(f"| `{opt}` | — | — | — | 0/{len(seeds)} |")
    common = sorted(set(values_by.get("muon", {})) & set(values_by.get("astro_v2", {})))
    if common:
        deltas = [values_by["astro_v2"][s] - values_by["muon"][s] for s in common]
        lines += ["", f"ASTRO-v2 − Muon mean Δ: `{statistics.fmean(deltas):+.5f}`; "
                  f"wins `{sum(d < 0 for d in deltas)}/{len(deltas)}`."]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-state", type=Path, required=True)
    parser.add_argument("--mechanism-state", type=Path, required=True)
    parser.add_argument("--scale-state", type=Path, required=True)
    parser.add_argument("--horizon-state", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("paper_evidence_summary.md"))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    main_state = read(args.main_state)
    mech = read(args.mechanism_state)
    scale = read(args.scale_state)
    horizon = read(args.horizon_state)
    incomplete: list[str] = []
    lines = ["# ASTRO paper evidence audit", ""]

    main_protocol = main_state.get("paper_protocol")
    if not main_protocol:
        incomplete.append("headline state has no paper_campaign protocol signature")
    main_digest = main_protocol.get("code_digest") if main_protocol else None

    # Headline -------------------------------------------------------------
    seeds = list(range(100, 105))
    lines += ["## 124M / 900-step held-out confirmation", "",
              "| optimizer | mean | sample sd | mean s/run | seeds |",
              "|---|---:|---:|---:|---:|"]
    by_seed = {}
    for opt in HEADLINE:
        rows, missing = headline_rows(main_state, opt, seeds)
        if missing:
            incomplete.append(f"headline {opt} missing seeds {missing}")
        values = [float(entry["value"]) for _, entry in rows]
        seconds = [float(entry["seconds"]) for _, entry in rows]
        configs = [entry.get("config", {}) for _, entry in rows]
        by_seed[opt] = {seed: float(entry["value"]) for seed, entry in rows}
        if configs and any(config_key(cfg) != config_key(configs[0]) for cfg in configs[1:]):
            incomplete.append(f"headline {opt} held-out seeds used different configs")
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
            deltas = [by_seed[opt][s] - by_seed["muon"][s] for s in common]
            lines.append(
                f"| `{opt}` | {statistics.fmean(deltas):+.5f} | {max(deltas):+.5f} | "
                f"{sum(d < 0 for d in deltas)}/{len(deltas)} | {sign_p(deltas):.4f} |"
            )

    # Mechanism ------------------------------------------------------------
    validate_derived_protocol("mechanism", mech, main_digest, incomplete)
    mech_seeds = list(mech.get("protocol", {}).get("seeds", []))
    lines += ["", "## Frozen causal controls", "",
              "| contrast (A − B) | paired seeds | mean Δ | worst Δ | A wins |",
              "|---|---:|---:|---:|---:|"]
    for opt in MECHANISM:
        _, _, missing = custom_rows(mech, opt)
        if missing:
            incomplete.append(f"mechanism {opt} missing seeds {missing}")
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
                f"{sum(d < 0 for d in deltas)}/{len(deltas)} |"
            )
        else:
            lines.append(f"| {label}: `{a}` − `{b}` | 0 | — | — | — |")
    lines += ["", "`astro_v2` vs `astro_v2_blockwise` isolates the candidate "
              "ASTRO-specific operation identified by the prior-art audit. A positive "
              "result establishes empirical value, not exhaustive literature novelty."]

    add_transfer(lines, incomplete, "355M / 900-step frozen scale transfer", scale, main_digest)
    add_transfer(lines, incomplete, "124M / 2700-step frozen durability check", horizon, main_digest)

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
