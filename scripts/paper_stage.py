#!/usr/bin/env python3
"""Run frozen paper-campaign stages without retuning.

The main 124M experiment selects each optimizer's configuration.  Every later
claim should either transfer those frozen configurations (``--config-policy
own``) or deliberately hold one configuration fixed across an ablation ladder
(``--config-policy shared:astro_v2``).  This runner makes that distinction
explicit and refuses to reuse a completed entry if its recorded configuration
changed.

Examples
--------
Causal ladder, two seeds::

    python paper_stage.py \
      --main-state /content/drive/MyDrive/astro/paper_campaign/main/astro_lab_state.json \
      --config-policy shared:astro_v2 \
      --optimizers muon muon_m90 astro_v2_gamma0_nosplit astro_v2_nosplit astro_v2 \
      --size 124M --steps 900 --seeds 200 201 \
      --work-dir /content/drive/MyDrive/astro/paper_campaign/ablation

Frozen 355M transfer::

    python paper_stage.py ... --config-policy own \
      --optimizers muon astro_v2 --size 355M --steps 900 --seeds 300 301
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from transformers import AutoTokenizer

import paper_lab


def config_digest(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def same_config(a: dict, b: dict) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def choose_configs(main_state: dict, policy: str, names: list[str]) -> dict[str, dict]:
    tuned = main_state.get("tuned", {})
    if policy == "own":
        missing = [name for name in names if name not in tuned]
        if missing:
            raise SystemExit(
                f"{missing[0]!r} has no frozen configuration in the main state"
            )
        return {name: dict(tuned[name]) for name in names}
    if policy.startswith("shared:"):
        source = policy.split(":", 1)[1]
        if source not in tuned:
            raise SystemExit(f"shared source {source!r} is not tuned in the main state")
        config = dict(tuned[source])
        return {name: dict(config) for name in names}
    raise SystemExit("--config-policy must be 'own' or 'shared:OPTIMIZER'")


def write_report(path: Path, state: dict, names: list[str], seeds: list[int]) -> None:
    lines = ["# ASTRO paper campaign stage", ""]
    lines += [
        f"size: `{state['protocol']['size']}`  ",
        f"steps: `{state['protocol']['steps']}`  ",
        f"config policy: `{state['protocol']['config_policy']}`  ",
        f"seeds: `{', '.join(map(str, seeds))}`",
        "",
        "| optimizer | mean val loss | sample sd | per-seed | s/run |",
        "|---|---:|---:|---|---:|",
    ]
    for name in names:
        entries = [state["runs"].get(f"{name}|{seed}") for seed in seeds]
        entries = [entry for entry in entries if entry is not None]
        if not entries:
            continue
        values = [float(entry["value"]) for entry in entries]
        seconds = [float(entry["seconds"]) for entry in entries]
        sd = statistics.stdev(values) if len(values) > 1 else float("nan")
        lines.append(
            f"| `{name}` | {statistics.fmean(values):.4f} | "
            f"{sd:.4f} | {', '.join(f'{v:.4f}' for v in values)} | "
            f"{statistics.fmean(seconds):.0f} |"
        )

    if state["protocol"]["config_policy"].startswith("shared:"):
        reference = names[0]
        ref = {
            seed: state["runs"].get(f"{reference}|{seed}") for seed in seeds
        }
        lines += ["", f"Paired deltas against `{reference}`:", ""]
        lines += ["| optimizer | mean paired Δ | wins |", "|---|---:|---:|"]
        for name in names[1:]:
            deltas = []
            for seed in seeds:
                a = state["runs"].get(f"{name}|{seed}")
                b = ref[seed]
                if a is not None and b is not None:
                    deltas.append(float(a["value"]) - float(b["value"]))
            if deltas:
                lines.append(
                    f"| `{name}` | {statistics.fmean(deltas):+.4f} | "
                    f"{sum(d < 0 for d in deltas)}/{len(deltas)} |"
                )

    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-state", type=Path, required=True)
    parser.add_argument("--config-policy", required=True)
    parser.add_argument("--optimizers", nargs="+", required=True)
    parser.add_argument("--size", choices=sorted(paper_lab.lab.SIZES), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--paper-corpus", type=Path, default=paper_lab.DEFAULT_CORPUS)
    parser.add_argument(
        "--paper-corpus-tokens", type=int, default=paper_lab.DEFAULT_CORPUS_TOKENS
    )
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    paper_lab.configure_paper_harness(args.paper_corpus, args.paper_corpus_tokens)
    for name in args.optimizers:
        if name not in paper_lab.lab.known_optimizers():
            raise SystemExit(f"unknown optimizer {name!r}")

    main_state = json.loads(args.main_state.read_text())
    configs = choose_configs(main_state, args.config_policy, args.optimizers)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.work_dir / "paper_stage_state.json"
    report_path = args.work_dir / "paper_stage_report.md"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"runs": {}}
    state.setdefault("runs", {})
    protocol = {
        "size": args.size,
        "steps": args.steps,
        "config_policy": args.config_policy,
        "optimizers": args.optimizers,
        "seeds": args.seeds,
        "main_state": str(args.main_state),
        "paper_corpus": str(args.paper_corpus),
    }
    if "protocol" in state and state["protocol"] != protocol:
        raise SystemExit(
            "existing paper_stage_state.json belongs to a different protocol; "
            "use a new work directory"
        )
    state["protocol"] = protocol

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    needed = (
        paper_lab.lab.SIZES[args.size]["batch"]
        * 512
        * (args.steps + 60)
        + paper_lab.lab.VALIDATION_TOKENS
    )
    tokens = paper_lab.fixed_load_tokens(tokenizer, needed, Path("ignored"))
    data = (
        tokens[paper_lab.lab.VALIDATION_TOKENS :],
        tokens[: paper_lab.lab.VALIDATION_TOKENS],
    )

    started = time.perf_counter()
    completed_this_session = 0

    for name in args.optimizers:
        config = configs[name]
        for seed in args.seeds:
            slot = f"{name}|{seed}"
            old = state["runs"].get(slot)
            if old is not None:
                if not same_config(old["config"], config):
                    raise SystemExit(
                        f"refusing stale result {slot}: stored config differs from frozen config"
                    )
                print(f"[{slot}] already done -> {old['value']:.4f}", flush=True)
                continue
            if args.stop_after is not None and completed_this_session >= args.stop_after:
                write_report(report_path, state, args.optimizers, args.seeds)
                print("stop-after reached; rerun the same command to continue")
                return 0
            if args.max_minutes is not None:
                elapsed = (time.perf_counter() - started) / 60
                if elapsed >= args.max_minutes:
                    write_report(report_path, state, args.optimizers, args.seeds)
                    print("time budget reached; rerun the same command to continue")
                    return 0

            print(
                f"[{args.size} {args.steps}st] {name} seed {seed} "
                f"config={config_digest(config)}",
                flush=True,
            )
            old_cwd = Path.cwd()
            try:
                # Traces land beside this stage's state file.
                import os

                os.chdir(args.work_dir)
                value, seconds = paper_lab.paper_train_once(
                    name,
                    config,
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
                "config": config,
                "config_sha": config_digest(config),
            }
            state_path.write_text(json.dumps(state, indent=2))
            write_report(report_path, state, args.optimizers, args.seeds)
            completed_this_session += 1
            print(f"      -> {value:.4f} ({seconds:.0f}s) saved", flush=True)

    write_report(report_path, state, args.optimizers, args.seeds)
    print("\n" + report_path.read_text())
    print(f"wrote {state_path} and {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
