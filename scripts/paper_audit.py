#!/usr/bin/env python3
"""Audit ASTRO paper-campaign state files and produce a claim-safe summary.

This script never trains. It checks completeness/provenance, pairs results by
explicit seed or identical configuration, computes descriptive statistics and
writes a Markdown summary suitable for the paper-writing stage.

It deliberately does NOT turn a positive result into a universal claim and does
not pretend experiments can prove literature novelty. The structural novelty
question is addressed by the prior-art audit plus the ASTRO-v2-vs-blockwise
control; this script reports that control without over-interpreting it.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    for section in ("runs", "trials", "tuned"):
        payload.setdefault(section, {})
    return payload


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def sign_p(deltas: list[float]) -> float:
    n = len(deltas)
    if not n:
        return float("nan")
    wins = sum(d < 0 for d in deltas)
    tail = sum(math.comb(n, k) for k in range(min(wins, n - wins) + 1))
    return min(1.0, 2 * tail / 2**n)


def run_key(size: str, steps: int, opt: str, seed: int) -> str:
    return f"{size}|{steps}|{opt}|{seed}"


def get_seed_values(state: dict, size: str, steps: int, opt: str, seeds: list[int]):
    values = []
    seconds = []
    configs = []
    missing = []
    for seed in seeds:
        entry = state["runs"].get(run_key(size, steps, opt, seed))
        if entry is None:
            missing.append(seed)
            continue
        values.append(float(entry["value"]))
        seconds.append(float(entry["seconds"]))
        configs.append(entry.get("config", {}))
    return values, seconds, configs, missing


def trial_rows(state: dict, size: str, steps: int, opt: str):
    out = []
    prefix = f"{size}|{steps}|{opt}|t"
    for key, entry in state["trials"].items():
        if key.startswith(prefix) and entry.get("value") is not None:
            out.append((entry["config"], float(entry["value"])))
    return out


def config_key(config: dict) -> tuple:
    return tuple(sorted((k, round(float(v), 14)) for k, v in config.items()))


def paired_trials(state: dict, size: str, steps: int, a: str, b: str):
    left = {config_key(c): v for c, v in trial_rows(state, size, steps, a)}
    right = {config_key(c): v for c, v in trial_rows(state, size, steps, b)}
    shared = sorted(set(left) & set(right))
    return [left[k] - right[k] for k in shared]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-state", type=Path, required=True)
    parser.add_argument("--mechanism-state", type=Path)
    parser.add_argument("--out", type=Path, default=Path("paper_evidence_summary.md"))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    state = load(args.main_state)
    seeds = list(range(100, 105))
    headline = ["adamw", "muon", "normuon", "adamuon_ref", "astro_v2"]
    lines = ["# ASTRO paper evidence audit", ""]
    incomplete = []

    lines += ["## 124M / 900-step held-out confirmation", "",
              "| optimizer | mean | sample sd | mean s/run | seeds |",
              "|---|---:|---:|---:|---:|"]
    values_by = {}
    times_by = {}
    for opt in headline:
        values, seconds, configs, missing = get_seed_values(state, "124M", 900, opt, seeds)
        if missing:
            incomplete.append(f"124M/900 {opt} missing seeds {missing}")
        if not values:
            lines.append(f"| `{opt}` | — | — | — | 0/5 |")
            continue
        mu, sd = mean_sd(values)
        values_by[opt] = values
        times_by[opt] = seconds
        lines.append(f"| `{opt}` | {mu:.5f} | {sd:.5f} | {statistics.fmean(seconds):.1f} | {len(values)}/5 |")
        if configs and any(config_key(c) != config_key(configs[0]) for c in configs[1:]):
            incomplete.append(f"124M/900 {opt} evaluation seeds used different configs")

    if "muon" in values_by:
        lines += ["", "Paired against Muon:", "",
                  "| optimizer | mean paired Δ | worst Δ | wins | two-sided sign p |",
                  "|---|---:|---:|---:|---:|"]
        for opt in headline:
            if opt == "muon" or opt not in values_by:
                continue
            n = min(len(values_by[opt]), len(values_by["muon"]))
            deltas = [values_by[opt][i] - values_by["muon"][i] for i in range(n)]
            lines.append(
                f"| `{opt}` | {statistics.fmean(deltas):+.5f} | {max(deltas):+.5f} | "
                f"{sum(d < 0 for d in deltas)}/{n} | {sign_p(deltas):.4f} |"
            )

    lines += ["", "## 355M / 900-step frozen transfer", "",
              "| optimizer | mean | sample sd | seeds |",
              "|---|---:|---:|---:|"]
    for opt in ("muon", "normuon", "adamuon_ref", "astro_v2"):
        vals, secs, configs, missing = get_seed_values(state, "355M", 900, opt, [100, 101])
        if missing:
            incomplete.append(f"355M/900 {opt} missing seeds {missing}")
        if vals:
            mu, sd = mean_sd(vals)
            lines.append(f"| `{opt}` | {mu:.5f} | {sd:.5f} | {len(vals)}/2 |")
        else:
            lines.append(f"| `{opt}` | — | — | 0/2 |")

    if args.mechanism_state:
        mech = load(args.mechanism_state)
        lines += ["", "## Mechanism / structural controls", ""]
        controls = [
            ("astro_v2", "astro_v2_gamma0", "post-spectral adaptation contribution"),
            ("astro_v2", "astro_v2_nosplit", "semantic split contribution"),
            ("astro_v2", "astro_v2_blockwise", "global-vs-blockwise redistribution contribution"),
            ("astro_muon_betas", "astro", "beta change with cautious WD on"),
            ("astro_v2", "astro_plain_wd", "beta change with plain WD"),
            ("astro_plain_wd", "astro", "plain-vs-cautious WD at beta1=.95"),
            ("astro_v2", "astro_muon_betas", "plain-vs-cautious WD at beta1=.90"),
        ]
        lines += ["| contrast (A − B) | shared configs | mean Δ | worst Δ | A wins |",
                  "|---|---:|---:|---:|---:|"]
        for a, b, label in controls:
            deltas = paired_trials(mech, "124M", 900, a, b)
            if len(deltas) < 3:
                incomplete.append(f"mechanism {a} vs {b}: only {len(deltas)}/3 shared configs")
            if deltas:
                lines.append(
                    f"| {label}: `{a}` − `{b}` | {len(deltas)} | "
                    f"{statistics.fmean(deltas):+.5f} | {max(deltas):+.5f} | "
                    f"{sum(d < 0 for d in deltas)}/{len(deltas)} |"
                )
            else:
                lines.append(f"| {label}: `{a}` − `{b}` | 0 | — | — | — |")

        lines += ["", "**Novelty gate.** The comparison `astro_v2` vs "
                  "`astro_v2_blockwise` is the experiment that isolates the candidate "
                  "new operation: global adaptive redistribution after independently "
                  "polarising semantic Q/K/V blocks, versus the prior-art-style choice "
                  "of normalising/restoring each block independently. This experiment "
                  "can establish empirical value of the distinction; literature novelty "
                  "still depends on the separate prior-art audit."]

    lines += ["", "## Completeness", ""]
    if incomplete:
        lines += [f"- INCOMPLETE: {item}" for item in incomplete]
    else:
        lines.append("All required campaign cells inspected by this audit are complete.")

    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")
    return 1 if args.strict and incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
