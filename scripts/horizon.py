#!/usr/bin/env python3
"""Does the gap survive a longer training run, or is it an early-training effect?

    python scripts/horizon.py --task gpt_shakespeare_cosine --steps 400 800 1600

Wen et al. (2509.02046) report that optimizer advantages measured early shrink as
training continues, and that this is one of the commonest ways a headline number
fails to reproduce at a realistic budget. A single-horizon comparison cannot tell
a durable advantage from a faster start, so this script re-runs the tuned
configurations at several step counts and reports the gap at each.

What it does *not* do is re-tune at each horizon. The configurations come from a
sweep at the shortest horizon, so the longer runs are handicapped for both
optimizers -- the optimal rate generally falls as the budget grows. Read the
trend, not the absolute values, and treat a gap that shrinks toward zero as the
finding it is.

The cosine variant of a task is the right one to sweep: its schedule stretches
with the budget, so a longer run decays over all of its steps rather than
finishing early and coasting.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from astro.bench.registry import build_candidate_spaces, build_spaces  # noqa: E402
from astro.bench.stats import paired_test  # noqa: E402
from astro.bench.tasks import TASKS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="gpt_shakespeare_cosine", choices=sorted(TASKS))
    parser.add_argument("--optimizers", nargs="+", default=["adamw", "astro"],
                        help="the control is the first name given")
    parser.add_argument("--steps", type=int, nargs="+", default=[400, 800, 1600])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--artifacts", type=Path,
                        default=ROOT / "artifacts" / "bench_llm" / "shake_cos")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "bench_llm")
    args = parser.parse_args(argv)

    record_path = next(args.artifacts.glob("*_compare.json"), None)
    if record_path is None:
        raise SystemExit(f"no *_compare.json under {args.artifacts}; run the benchmark first")
    tuned = {name: record["best_config"]
             for name, record in json.loads(record_path.read_text())["tuning"].items()}

    task = TASKS[args.task]
    finetuning = args.task in {"finetune", "gpt_finetune"}
    spaces = {s.name: s for s in build_spaces(finetuning=finetuning)}
    spaces.update({s.name: s for s in build_candidate_spaces(finetuning=finetuning)})

    control = args.optimizers[0]
    values: dict[int, dict[str, list[float]]] = {}
    for steps in args.steps:
        values[steps] = {}
        for name in args.optimizers:
            runs = [
                task(spaces[name].factory(tuned[name]), seed, steps=steps).final
                for seed in range(100, 100 + args.seeds)
            ]
            values[steps][name] = runs
            print(f"{args.task} steps={steps:5d} {name:10s} "
                  f"{statistics.fmean(runs):.4f}", flush=True)

    lines = [f"| steps | {' | '.join(f'`{n}`' for n in args.optimizers)} | "
             f"paired Δ | Cohen's d | p |", "|---" * (len(args.optimizers) + 4) + "|"]
    for steps in args.steps:
        row = [str(steps)] + [f"{statistics.fmean(values[steps][n]):.4f}"
                              for n in args.optimizers]
        other = [n for n in args.optimizers if n != control]
        test = paired_test(values[steps][other[0]], values[steps][control])
        row += [f"{test.mean_delta:+.4f}", f"{test.effect_size:+.2f}", f"{test.p_value:.4f}"]
        lines.append("| " + " | ".join(row) + " |")
        print(f"  steps={steps:5d}  delta={test.mean_delta:+.4f}  "
              f"d={test.effect_size:+.2f}  p={test.p_value:.4f}", flush=True)

    table = "\n".join(lines)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.task}_horizon.md").write_text(table + "\n")
    (args.out / f"{args.task}_horizon.json").write_text(
        json.dumps({"task": args.task, "control": control, "seeds": args.seeds,
                    "values": {str(k): v for k, v in values.items()}}, indent=2)
    )
    print("\n" + table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
