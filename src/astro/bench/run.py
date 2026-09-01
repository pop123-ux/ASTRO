"""Benchmark runner: tune every optimizer equally, evaluate across seeds, report.

    python -m astro.bench.run --task finetune --trials 16 --seeds 5

Writes a markdown table and a JSON record of every trial, so a result can be
re-checked without re-running.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from astro.bench.protocol import (
    EvaluationSummary,
    SearchSpace,
    evaluate,
    paired_comparison,
    tune,
)
from astro.bench.registry import build_ablation_spaces, build_spaces
from astro.bench.tasks import TASKS

__all__ = ["run_benchmark", "format_report", "main"]

#: Tasks that fine-tune pretrained weights; ASTRO's anchor only applies there.
FINETUNING_TASKS = {"finetune", "gpt_finetune"}


def run_benchmark(
    task_name: str,
    *,
    trials: int = 16,
    seeds: int = 5,
    ablation: bool = False,
    control: str = "adamw",
    optimizers: Sequence[str] | None = None,
    checkpoint: Path | None = None,
) -> dict[str, object]:
    """Tune, evaluate and compare every optimizer on one task."""
    if task_name not in TASKS:
        raise KeyError(f"unknown task {task_name!r}; available: {sorted(TASKS)}")
    task = TASKS[task_name]
    finetuning = task_name in FINETUNING_TASKS

    spaces: Sequence[SearchSpace] = (
        build_ablation_spaces(finetuning=finetuning)
        if ablation
        else build_spaces(finetuning=finetuning)
    )
    if optimizers:
        available = {space.name for space in spaces}
        unknown = [name for name in optimizers if name not in available]
        if unknown:
            raise KeyError(f"unknown optimizer(s) {unknown}; available: {sorted(available)}")
        # Narrowing the field does not weaken the protocol: tune() still enforces
        # an equal number of tuned hyperparameters across whoever remains.
        spaces = [space for space in spaces if space.name in optimizers]

    records = tune(task, spaces, trials=trials, seed=0)
    for name, record in records.items():
        print(f"  tuned {name}: {record.best_value:.4f} {record.best_config}", flush=True)

    summaries: dict[str, EvaluationSummary] = {}
    for space in spaces:
        record = records[space.name]
        summaries[space.name] = evaluate(
            task, space, record.best_config, seeds=range(100, 100 + seeds)
        )
        summary = summaries[space.name]
        print(
            f"  evaluated {space.name}: {summary.mean:.4f} +- {summary.stdev:.4f} "
            f"({summary.mean_seconds:.1f}s/run)",
            flush=True,
        )
        if checkpoint is not None:
            # Persist after every optimizer. A run of this length is routinely
            # interrupted, and writing only at the end means an interruption
            # costs the whole sweep -- which is exactly what happened once.
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(json.dumps({
                "task": task_name, "trials": trials, "seeds": seeds,
                "tuning": {n: asdict(r) for n, r in records.items()},
                "evaluation": {n: asdict(s) for n, s in summaries.items()},
            }, indent=2))

    reference = control if control in summaries else next(iter(summaries))
    comparisons = {
        name: paired_comparison(summary, summaries[reference])
        for name, summary in summaries.items()
        if name != reference
    }

    return {
        "task": task_name,
        "trials": trials,
        "seeds": seeds,
        "control": reference,
        "tuning": {name: asdict(record) for name, record in records.items()},
        "evaluation": {name: asdict(summary) for name, summary in summaries.items()},
        "comparisons": comparisons,
        "_summaries": summaries,
    }


def format_report(result: dict[str, object]) -> str:
    """Render a markdown report. Lower is better for every value shown."""
    summaries: dict[str, EvaluationSummary] = result["_summaries"]  # type: ignore[assignment]
    comparisons: dict[str, dict[str, float]] = result["comparisons"]  # type: ignore[assignment]
    control = str(result["control"])
    tuning: dict[str, dict] = result["tuning"]  # type: ignore[assignment]

    lines = [
        f"### `{result['task']}`",
        "",
        f"{result['trials']} tuning trials per optimizer (identical budget, "
        f"3 tuned hyperparameters each), best config re-run on {result['seeds']} seeds. "
        f"Control: `{control}`. Lower is better.",
        "",
        "| optimizer | final (mean ± sd) | 95% CI | vs control | paired Δ [CI] | sig | s/run |",
        "|---|---|---|---|---|---|---|",
    ]
    order = sorted(summaries, key=lambda n: summaries[n].mean)
    for name in order:
        summary = summaries[name]
        low, high = summary.interval()
        if name == control:
            relative, delta, significant = "--", "--", "--"
        else:
            comparison = comparisons[name]
            relative = f"{comparison['relative'] * 100:+.1f}%"
            delta = (
                f"{comparison['mean_delta']:+.4f} "
                f"[{comparison['ci_low']:+.4f}, {comparison['ci_high']:+.4f}]"
            )
            significant = "yes" if comparison["significant"] else "no"
        lines.append(
            f"| `{name}` | {summary.mean:.4f} ± {summary.stdev:.4f} | "
            f"[{low:.4f}, {high:.4f}] | {relative} | {delta} | {significant} | "
            f"{summary.mean_seconds:.2f} |"
        )

    lines += ["", "Tuning traces (best-so-far; a still-falling trace means under-tuned):", ""]
    for name in order:
        trace = tuning[name]["trace"]
        marks = [trace[i] for i in range(0, len(trace), max(1, len(trace) // 6))]
        lines.append(f"- `{name}`: " + " → ".join(f"{v:.3f}" for v in marks))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="finetune", choices=sorted(TASKS) + ["all"])
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--control", default="adamw")
    parser.add_argument(
        "--optimizers", nargs="+", default=None,
        help="restrict the field to these names (default: all)",
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/bench"))
    args = parser.parse_args(argv)

    names = sorted(TASKS) if args.task == "all" else [args.task]
    args.out.mkdir(parents=True, exist_ok=True)
    reports = []
    for name in names:
        control = args.control
        if args.ablation:
            control = "astro_full"
        suffix_now = "ablation" if args.ablation else "compare"
        result = run_benchmark(
            name, trials=args.trials, seeds=args.seeds, ablation=args.ablation,
            control=control, optimizers=args.optimizers,
            checkpoint=args.out / f"{name}_{suffix_now}.json",
        )
        report = format_report(result)
        reports.append(report)
        print(report + "\n", flush=True)

        suffix = "ablation" if args.ablation else "compare"
        payload = {k: v for k, v in result.items() if not k.startswith("_")}
        (args.out / f"{name}_{suffix}.json").write_text(json.dumps(payload, indent=2))
    (args.out / ("ablation.md" if args.ablation else "compare.md")).write_text(
        "\n\n".join(reports)
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
