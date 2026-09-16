#!/usr/bin/env python3
"""Run a frozen Muon-vs-ASTRO-v2 transfer cell from the headline configs.

Use this for the two robustness questions that do not warrant retuning:

* larger model: 355M / 900 steps;
* longer horizon: 124M / 2700 steps.

The runner uses a separate state file per workdir, so those two cells can run in
parallel after the headline configurations are frozen.
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

NAMES = ("muon", "astro_v2")


def source_digest() -> str:
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def config_digest(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def write_report(path: Path, state: dict, seeds: list[int]) -> None:
    lines = ["# ASTRO frozen transfer", ""]
    lines += [
        f"cell: `{state['protocol']['size']} / {state['protocol']['steps']} steps`  ",
        "configs: frozen from the headline 124M/900 campaign",
        "",
        "| optimizer | mean val loss | sample sd | per-seed | s/run |",
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
    if all(name in means for name in NAMES):
        lines += ["", f"ASTRO-v2 − Muon mean Δ: `{means['astro_v2'] - means['muon']:+.5f}`"]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-state", type=Path, required=True)
    parser.add_argument("--size", choices=sorted(campaign.lab.SIZES), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    main_state = json.loads(args.main_state.read_text())
    main_protocol = main_state.get("paper_protocol")
    if not main_protocol:
        raise SystemExit("main state is not a paper_campaign state")
    if main_protocol.get("code_digest") != campaign.campaign_digest():
        raise SystemExit(
            "current paper_campaign code differs from the code that selected the "
            "headline configurations; checkout the recorded campaign commit/environment"
        )
    for name in NAMES:
        if name not in main_state.get("tuned", {}):
            raise SystemExit(f"headline campaign has no frozen {name!r} config")

    configs = {name: dict(main_state["tuned"][name]) for name in NAMES}
    protocol = {
        "runner_digest": source_digest(),
        "campaign_digest": campaign.campaign_digest(),
        "environment": campaign.current_environment(),
        "fineweb_revision": campaign.FINEWEB_REVISION,
        "shared_cache": str(campaign.SHARED_CACHE),
        "main_state": str(args.main_state),
        "main_protocol": main_protocol,
        "size": args.size,
        "steps": args.steps,
        "seeds": args.seeds,
        "config_sha": {name: config_digest(cfg) for name, cfg in configs.items()},
    }

    args.work_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.work_dir / "paper_transfer_state.json"
    report_path = args.work_dir / "paper_transfer_report.md"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("protocol") != protocol:
            raise SystemExit("transfer workdir belongs to a different protocol; use a new workdir")
    else:
        state = {"protocol": protocol, "runs": {}}

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokens = campaign.paper_load_tokens(tokenizer, campaign.TOTAL_TOKENS, campaign.SHARED_CACHE)
    data = (
        tokens[campaign.lab.VALIDATION_TOKENS :],
        tokens[: campaign.lab.VALIDATION_TOKENS],
    )

    session_start = time.perf_counter()
    completed = 0
    for seed in args.seeds:
        for name in NAMES:
            slot = f"{name}|{seed}"
            if slot in state["runs"]:
                print(f"[{slot}] already done -> {state['runs'][slot]['value']:.4f}")
                continue
            if args.stop_after is not None and completed >= args.stop_after:
                write_report(report_path, state, args.seeds)
                print("stop-after reached; rerun unchanged to continue")
                return 0
            if args.max_minutes is not None and (time.perf_counter() - session_start) / 60 >= args.max_minutes:
                write_report(report_path, state, args.seeds)
                print("time budget reached; rerun unchanged to continue")
                return 0

            cfg = configs[name]
            print(
                f"[{args.size} {args.steps}st] {name} seed {seed} "
                f"config={config_digest(cfg)}",
                flush=True,
            )
            old_cwd = Path.cwd()
            try:
                os.chdir(args.work_dir)
                value, seconds = campaign.paper_train_once(
                    name,
                    cfg,
                    seed,
                    data=data,
                    size=args.size,
                    steps=args.steps,
                    seq=512,
                    vocab=len(tokenizer),
                    log_every=args.log_every,
                )
            finally:
                os.chdir(old_cwd)

            state["runs"][slot] = {
                "value": value,
                "seconds": seconds,
                "config": cfg,
                "config_sha": config_digest(cfg),
            }
            state_path.write_text(json.dumps(state, indent=2))
            write_report(report_path, state, args.seeds)
            completed += 1
            print(f"      -> {value:.4f} ({seconds:.0f}s) saved", flush=True)

    write_report(report_path, state, args.seeds)
    print("\n" + report_path.read_text())
    print(f"wrote {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
