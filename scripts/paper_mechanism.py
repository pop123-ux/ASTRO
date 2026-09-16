#!/usr/bin/env python3
"""Minimal causal campaign for the ASTRO-v2 paper.

This script deliberately does *not* tune ablations.  It reads the configurations
selected by the clean headline campaign and changes one mechanism at a time.
That is the protocol needed to answer attribution questions without turning the
paper into an optimizer-zoo sweep.

Two configuration families are used:

* ``muon`` and ``muon_m90`` use the frozen Muon configuration;
* ASTRO-v2 and its three structural removals use the frozen ASTRO-v2
  configuration.

Required contrasts:

* ``muon_m90 - muon``: matrix-momentum beta effect;
* ``astro_v2 - astro_v2_gamma0``: post-spectral row adaptation;
* ``astro_v2 - astro_v2_nosplit``: semantic Q/K/V split;
* ``astro_v2 - astro_v2_blockwise``: global post-split redistribution versus
  the prior-art-style blockwise split+adapt pattern.

The last contrast is the candidate ASTRO-specific structural contribution.  A
benchmark can establish its empirical value, not exhaustive literature novelty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

from transformers import AutoTokenizer

import paper_campaign as campaign

NAMES = (
    "muon",
    "muon_m90",
    "astro_v2_gamma0",
    "astro_v2_nosplit",
    "astro_v2_blockwise",
    "astro_v2",
)


def config_key(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def build_with_beta_control(name: str, model, config: dict[str, float]):
    if name != "muon_m90":
        return _ORIGINAL_BUILD(name, model, config)
    scalar = config.get("scalar_lr_mult", 1.0)
    groups = campaign.paper_groups(model, "muon", config)
    return campaign.lab.Muon(
        groups,
        lr=config["lr"],
        adamw_lr=config["lr"] * scalar,
        momentum=0.90,
        weight_decay=config["weight_decay"],
    )


def report(path: Path, state: dict, seeds: list[int]) -> None:
    lines = ["# ASTRO mechanism campaign", ""]
    lines += [
        "All runs use configurations frozen before these seeds were observed.",
        "",
        "| condition | mean val loss | sample sd | per-seed | s/run |",
        "|---|---:|---:|---|---:|",
    ]
    means = {}
    for name in NAMES:
        entries = [state["runs"].get(f"{name}|{seed}") for seed in seeds]
        entries = [entry for entry in entries if entry is not None]
        if not entries:
            continue
        values = [float(entry["value"]) for entry in entries]
        seconds = [float(entry["seconds"]) for entry in entries]
        means[name] = statistics.fmean(values)
        sd = statistics.stdev(values) if len(values) > 1 else float("nan")
        lines.append(
            f"| `{name}` | {means[name]:.5f} | {sd:.5f} | "
            f"{', '.join(f'{v:.5f}' for v in values)} | {statistics.fmean(seconds):.0f} |"
        )

    contrasts = [
        ("muon_m90", "muon", "matrix beta .90 vs .95"),
        ("astro_v2", "astro_v2_gamma0", "post-spectral row adaptation"),
        ("astro_v2", "astro_v2_nosplit", "semantic Q/K/V split"),
        ("astro_v2", "astro_v2_blockwise", "global vs blockwise redistribution"),
    ]
    lines += ["", "## Intended contrasts", "",
              "| contrast | mean A-B |", "|---|---:|"]
    for a, b, label in contrasts:
        if a in means and b in means:
            lines.append(f"| {label}: `{a}` - `{b}` | {means[a] - means[b]:+.5f} |")
        else:
            lines.append(f"| {label}: `{a}` - `{b}` | pending |")
    path.write_text("\n".join(lines) + "\n")


_ORIGINAL_BUILD = campaign.paper_build_optimizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-state", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[200, 201])
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    main_state = json.loads(args.main_state.read_text())
    for source in ("muon", "astro_v2"):
        if source not in main_state.get("tuned", {}):
            raise SystemExit(
                f"headline campaign has no frozen {source!r} configuration yet"
            )

    muon_cfg = dict(main_state["tuned"]["muon"])
    astro_cfg = dict(main_state["tuned"]["astro_v2"])
    configs = {
        "muon": muon_cfg,
        "muon_m90": dict(muon_cfg),
        "astro_v2_gamma0": astro_cfg,
        "astro_v2_nosplit": dict(astro_cfg),
        "astro_v2_blockwise": dict(astro_cfg),
        "astro_v2": dict(astro_cfg),
    }

    # Register only this one beta control locally; it does not pollute the main
    # paper campaign search space.
    campaign.paper_build_optimizer = build_with_beta_control

    args.work_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.work_dir / "paper_mechanism_state.json"
    report_path = args.work_dir / "paper_mechanism_report.md"
    protocol = {
        "campaign_digest": campaign.campaign_digest(),
        "environment": campaign.current_environment(),
        "fineweb_revision": campaign.FINEWEB_REVISION,
        "shared_cache": str(campaign.SHARED_CACHE),
        "main_state": str(args.main_state),
        "main_protocol": main_state.get("paper_protocol"),
        "seeds": args.seeds,
        "size": "124M",
        "steps": 900,
        "muon_config_sha": config_key(muon_cfg),
        "astro_v2_config_sha": config_key(astro_cfg),
    }
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("protocol") != protocol:
            raise SystemExit(
                "mechanism workdir belongs to a different code/environment/config "
                "protocol. Use a new workdir rather than mixing experiments."
            )
    else:
        state = {"protocol": protocol, "runs": {}}

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokens = campaign.paper_load_tokens(
        tokenizer, campaign.TOTAL_TOKENS, campaign.SHARED_CACHE
    )
    data = (
        tokens[campaign.lab.VALIDATION_TOKENS :],
        tokens[: campaign.lab.VALIDATION_TOKENS],
    )

    began_session = time.perf_counter()
    completed = 0
    for seed in args.seeds:
        # Condition-major within each seed keeps interrupted sessions balanced.
        for name in NAMES:
            slot = f"{name}|{seed}"
            if slot in state["runs"]:
                print(f"[{slot}] already done -> {state['runs'][slot]['value']:.4f}")
                continue
            if args.stop_after is not None and completed >= args.stop_after:
                report(report_path, state, args.seeds)
                print("stop-after reached; rerun the same command to continue")
                return 0
            if args.max_minutes is not None:
                elapsed = (time.perf_counter() - began_session) / 60
                if elapsed >= args.max_minutes:
                    report(report_path, state, args.seeds)
                    print("time budget reached; rerun the same command to continue")
                    return 0

            config = configs[name]
            print(
                f"[124M 900st] {name} seed {seed} config={config_key(config)}",
                flush=True,
            )
            old_cwd = Path.cwd()
            try:
                os.chdir(args.work_dir)
                value, seconds = campaign.paper_train_once(
                    name,
                    config,
                    seed,
                    data=data,
                    size="124M",
                    steps=900,
                    seq=512,
                    vocab=len(tokenizer),
                    log_every=args.log_every,
                )
            finally:
                os.chdir(old_cwd)

            state["runs"][slot] = {
                "value": value,
                "seconds": seconds,
                "config": config,
                "config_sha": config_key(config),
            }
            state_path.write_text(json.dumps(state, indent=2))
            report(report_path, state, args.seeds)
            completed += 1
            print(f"      -> {value:.4f} ({seconds:.0f}s) saved", flush=True)

    report(report_path, state, args.seeds)
    print("\n" + report_path.read_text())
    print(f"wrote {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
