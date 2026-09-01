#!/usr/bin/env python3
"""Re-evaluate the headline three-way comparison at a larger seed count.

    python scripts/headline.py --task gpt_finetune --seeds 10

AdamW and Muon are the comparisons this project is actually about; the wider
field exists to place them. This script re-runs *only* those, plus ASTRO, on more
seeds, and applies the tests that a five-seed table cannot support.

Why more seeds, specifically
---------------------------
An exact Wilcoxon signed-rank test on :math:`n` paired differences has a smallest
attainable two-sided p-value of :math:`2/2^{n}`. At five seeds that is
:math:`0.0625`: a *perfect sweep*, the treatment winning on every single seed, can
never clear :math:`p<0.05`, no matter how large the effect. Ten seeds brings the
floor to :math:`0.002`, which is what makes a rank-based significance claim
available at all. ``tests/test_stats.py`` pins this.

The tuned configurations are **not** recomputed. Tuning ran on seed 0 and is
recorded in the benchmark artifacts; re-tuning here would let the extra seeds
leak into model selection, which is the failure this protocol exists to prevent.
Only the evaluation is extended.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from astro.bench.protocol import bootstrap_ci, evaluate  # noqa: E402
from astro.bench.registry import build_spaces  # noqa: E402
from astro.bench.stats import holm_bonferroni, paired_test  # noqa: E402
from astro.bench.tasks import TASKS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ("adamw", "muon", "astro")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="gpt_finetune", choices=sorted(TASKS))
    parser.add_argument("--optimizers", nargs="+", default=list(DEFAULT))
    parser.add_argument("--control", default="adamw")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "bench_llm")
    args = parser.parse_args(argv)

    source = args.artifacts or (ROOT / "artifacts" / "bench_llm" / f"v2_{args.task.split('_')[1]}")
    record_path = next(source.glob("*_compare.json"), None)
    if record_path is None:
        raise SystemExit(f"no *_compare.json under {source}; run the benchmark first")
    tuned = json.loads(record_path.read_text())["tuning"]

    task = TASKS[args.task]
    finetuning = args.task in {"finetune", "gpt_finetune"}
    spaces = {s.name: s for s in build_spaces(finetuning=finetuning)}

    print(f"{args.task}: re-evaluating {args.optimizers} on {args.seeds} seeds", flush=True)
    summaries = {}
    for name in args.optimizers:
        config = tuned[name]["best_config"]
        summaries[name] = evaluate(
            task, spaces[name], config, seeds=range(200, 200 + args.seeds)
        )
        values = summaries[name].values
        low, high = bootstrap_ci(values)
        print(
            f"  {name:8s} {statistics.fmean(values):.4f} [{low:.4f}, {high:.4f}]  "
            f"({statistics.fmean(summaries[name].seconds):.1f}s/run)",
            flush=True,
        )

    control = args.control
    tests = {
        name: paired_test(summary.values, summaries[control].values)
        for name, summary in summaries.items()
        if name != control
    }
    adjusted = holm_bonferroni({name: test.p_value for name, test in tests.items()})

    lines = [
        f"**{args.task}**, {args.seeds} seeds, control `{control}`. "
        "Tuned configurations carried over from the 16-trial sweep; only the evaluation "
        "is extended, so the extra seeds cannot leak into model selection.",
        "",
        "| optimizer | val loss (mean ± sd) | 95% CI | paired Δ | Cohen's d | seeds won | "
        "exact Wilcoxon p | Holm-adj. p | s/run |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, summary in sorted(summaries.items(), key=lambda kv: statistics.fmean(kv[1].values)):
        values = summary.values
        mean = statistics.fmean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        low, high = bootstrap_ci(values)
        seconds = statistics.fmean(summary.seconds)
        if name == control:
            lines.append(
                f"| `{name}` *(control)* | {mean:.4f} ± {sd:.4f} | [{low:.4f}, {high:.4f}] "
                f"| — | — | — | — | — | {seconds:.1f} |"
            )
            continue
        test = tests[name]
        wins = sum(
            1 for t, c in zip(values, summaries[control].values, strict=True) if t < c
        )
        adj, reject = adjusted[name]
        mark = "**" if reject else ""
        lines.append(
            f"| `{name}` | {mean:.4f} ± {sd:.4f} | [{low:.4f}, {high:.4f}] "
            f"| {test.mean_delta:+.4f} | {test.effect_size:+.2f} ({test.magnitude}) "
            f"| {wins}/{len(values)} | {mark}{test.p_value:.4f}{mark} "
            f"| {mark}{adj:.4f}{mark} | {seconds:.1f} |"
        )

    table = "\n".join(lines)
    print("\n" + table)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.task}_headline.md").write_text(table + "\n")
    (args.out / f"{args.task}_headline.json").write_text(
        json.dumps(
            {
                "task": args.task,
                "seeds": args.seeds,
                # The actual seeds, not just how many. A paired test across two
                # artifacts is only valid when they share these, and a count
                # cannot establish that.
                "seed_list": {n: s.seeds for n, s in summaries.items()},
                "control": control,
                "values": {n: s.values for n, s in summaries.items()},
                "seconds": {n: s.seconds for n, s in summaries.items()},
                "configs": {n: tuned[n]["best_config"] for n in args.optimizers},
                "tests": {
                    n: {
                        "mean_delta": t.mean_delta,
                        "effect_size": t.effect_size,
                        "p_value": t.p_value,
                        "exact": t.exact,
                        "holm_adjusted": adjusted[n][0],
                        "reject": adjusted[n][1],
                    }
                    for n, t in tests.items()
                },
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out / f'{args.task}_headline.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
