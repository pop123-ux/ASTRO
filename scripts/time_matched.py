#!/usr/bin/env python3
"""Compare optimizers at matched *wall-clock* rather than matched steps.

    python scripts/time_matched.py --task gpt_finetune

A step-budget comparison flatters any optimizer whose step is expensive. ASTRO's
matrix path costs about 38% more per step than AdamW's on the GPT-2 benchmark, so
"lower loss after 120 steps" and "lower loss after 20 seconds" are different
claims, and only the second one matters to someone choosing an optimizer for a
fixed compute budget.

This script answers the second. It reads the tuned configuration each optimizer
won with, measures the per-step cost of each, then re-runs every optimizer with
its step count scaled so that all of them consume the same wall-clock as the
reference. The optimizer with the cheaper step gets *more* steps, which is
exactly the advantage a step-matched table hides.

Run it on an idle machine. Wall-clock measured while anything else is competing
for the same cores is not wall-clock, and the whole point of this script is that
the timing is real -- so it refuses to guess and simply reports what it measured,
including the per-step costs it derived the budgets from.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from astro.bench.protocol import bootstrap_ci  # noqa: E402
from astro.bench.registry import build_candidate_spaces, build_spaces  # noqa: E402
from astro.bench.tasks import TASKS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def measure_step_cost(task, space, config, *, seed: int = 0) -> tuple[float, float]:
    """Return ``(seconds, steps)`` for one run at the tuned configuration."""
    result = task(space.factory(config), seed)
    return result.seconds, float(result.steps)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="gpt_finetune", choices=sorted(TASKS))
    parser.add_argument(
        "--optimizers", nargs="+", default=["adamw", "muon", "astro"],
        help="the reference is the first name given",
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--artifacts", type=Path, default=ROOT / "artifacts" / "bench_llm" / "v2_finetune"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "bench_llm")
    args = parser.parse_args(argv)

    record_path = next(args.artifacts.glob("*_compare.json"), None)
    if record_path is None:
        raise SystemExit(
            f"no *_compare.json under {args.artifacts}; run the benchmark first so the "
            "tuned configurations exist"
        )
    tuned = dict(json.loads(record_path.read_text())["tuning"])
    # Candidate configurations live in their own artifacts, keyed by candidate
    # set. A candidate may appear in several rounds -- re-run after a harness fix,
    # for instance -- so the files are read oldest-first and later ones overwrite,
    # making the most recently measured configuration win. Alphabetical order with
    # setdefault silently picked a superseded config here, which produced a
    # wrong-configuration result that looked like a real regression.
    for extra in sorted(
        args.out.glob(f"{args.task}_candidate_*.json"), key=lambda f: f.stat().st_mtime
    ):
        payload = json.loads(extra.read_text())
        for name, config in payload.get("configs", {}).items():
            tuned[name] = {"best_config": config}

    task = TASKS[args.task]
    finetuning = args.task in {"finetune", "gpt_finetune"}
    # Candidates as well as the pre-registered field: a candidate that wins on
    # steps has to be checked on time like anything else, and it is the expensive
    # candidates that most need checking.
    spaces = {s.name: s for s in build_spaces(finetuning=finetuning)}
    spaces.update({s.name: s for s in build_candidate_spaces(finetuning=finetuning)})

    # 1. Per-step cost, measured once per optimizer on a shared seed.
    print("measuring per-step cost", flush=True)
    cost: dict[str, float] = {}
    for name in args.optimizers:
        seconds, steps = measure_step_cost(task, spaces[name], tuned[name]["best_config"])
        cost[name] = seconds / steps
        print(f"  {name:10s} {seconds:6.2f}s / {steps:.0f} steps = {cost[name]*1000:.1f} ms/step")

    reference = args.optimizers[0]
    budget = cost[reference] * _default_steps(task)
    print(f"\ntime budget: {budget:.2f}s (= {reference} at its natural step count)")

    # 2. Re-run each optimizer with the step count that fits the same budget.
    print("\nrunning at matched wall-clock", flush=True)
    results: dict[str, list[float]] = {}
    allotted: dict[str, int] = {}
    for name in args.optimizers:
        steps = max(1, int(round(budget / cost[name])))
        allotted[name] = steps
        values = []
        for seed in range(100, 100 + args.seeds):
            result = task(spaces[name].factory(tuned[name]["best_config"]), seed, steps=steps)
            values.append(result.final)
        results[name] = values
        low, high = bootstrap_ci(values)
        print(
            f"  {name:10s} {steps:4d} steps  loss {statistics.fmean(values):.4f} "
            f"[{low:.4f}, {high:.4f}]"
        )

    # 3. Paired deltas against the reference.
    print(f"\npaired against {reference}, at equal wall-clock:")
    lines = [
        "| optimizer | steps in budget | val loss | paired Δ [95% CI] | significant |",
        "|---|---|---|---|---|",
    ]
    for name in args.optimizers:
        mean = statistics.fmean(results[name])
        if name == reference:
            lines.append(f"| `{name}` *(reference)* | {allotted[name]} | {mean:.4f} | — | — |")
            continue
        deltas = [a - b for a, b in zip(results[name], results[reference], strict=True)]
        low, high = bootstrap_ci(deltas)
        significant = "**yes**" if (high < 0 or low > 0) else "no"
        lines.append(
            f"| `{name}` | {allotted[name]} | {mean:.4f} | "
            f"{statistics.fmean(deltas):+.4f} [{low:+.4f}, {high:+.4f}] | {significant} |"
        )
        print(f"  {name:10s} delta={statistics.fmean(deltas):+.4f} [{low:+.4f}, {high:+.4f}]")

    table = "\n".join(lines)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.task}_time_matched.md").write_text(table + "\n")
    (args.out / f"{args.task}_time_matched.json").write_text(
        json.dumps(
            {
                "task": args.task,
                "reference": reference,
                "seconds_per_step": cost,
                "steps_in_budget": allotted,
                "values": results,
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out / f'{args.task}_time_matched.md'}")
    return 0


def _default_steps(task) -> int:
    """The step count the task uses by default, read from its signature."""
    import inspect

    return int(inspect.signature(task).parameters["steps"].default)


if __name__ == "__main__":
    raise SystemExit(main())
