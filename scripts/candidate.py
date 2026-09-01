#!/usr/bin/env python3
"""Tune and evaluate a candidate optimizer against an already-measured field.

    python scripts/candidate.py --task gpt_scratch --candidate astro_post \\
        --against normuon muon --seeds 10

Iterating on an algorithm and re-testing it on the same benchmark until something
clears significance is how false results are manufactured -- it is the mechanism
behind the five-seed result this project already had to retract. Two properties
of this script exist to make that harder:

1. **The candidate is tuned with the same budget the field got**, from the same
   RNG stream, so a win cannot come from a longer search.
2. **The comparison reuses the field's stored per-seed values**, so the baselines
   are not quietly re-run under different conditions, and the seeds are shared --
   which is what makes the paired test valid.

It reports the exact Wilcoxon test and Holm-corrected p-values, because the whole
point of running several candidate rounds is that the family of comparisons is
larger than one.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from astro.bench.protocol import bootstrap_ci, evaluate, tune  # noqa: E402
from astro.bench.registry import build_candidate_spaces  # noqa: E402
from astro.bench.stats import holm_bonferroni, paired_test  # noqa: E402
from astro.bench.tasks import TASKS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def resolve_paired_seeds(
    reference_seeds: dict[str, list[int]], against: list[str], wanted: int
) -> list[int]:
    """The seeds to evaluate candidates on, taken from the baselines themselves.

    A paired signed-rank test on values from different seeds is not a weaker
    test, it is a different and meaningless one, and nothing in the numbers
    reveals the mismatch. Rather than checking a convention that two scripts have
    to keep agreeing on -- ``astro.bench.run`` evaluates on 100+, ``headline.py``
    on 200+ -- the candidates are run on whatever the reference used, so the
    pairing holds by construction.
    """
    recorded = {name: reference_seeds.get(name, []) for name in against}
    missing = sorted(name for name, value in recorded.items() if not value)
    if missing:
        raise SystemExit(
            f"the stored values for {missing} do not record which seeds produced "
            "them, so nothing can be paired against them. Re-run the field with a "
            "version that records seeds."
        )
    distinct = {tuple(value[:wanted]) for value in recorded.values()}
    if len(distinct) > 1:
        detail = "; ".join(f"{n}={v[:wanted]}" for n, v in sorted(recorded.items()))
        raise SystemExit(f"the baselines were evaluated on different seeds: {detail}")
    seeds = list(distinct.pop())
    if len(seeds) < wanted:
        raise SystemExit(
            f"the reference has only {len(seeds)} seeds, fewer than the {wanted} requested"
        )
    return seeds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="gpt_scratch", choices=sorted(TASKS))
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--against", nargs="+", default=["normuon", "muon", "adamw"])
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--field", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "bench_llm")
    args = parser.parse_args(argv)

    finetuning = args.task in {"finetune", "gpt_finetune"}
    field_dir = args.field or (
        ROOT / "artifacts" / "bench_llm" / f"v2_{args.task.split('_')[1]}"
    )
    record_path = next(field_dir.glob("*_compare.json"), None)
    if record_path is None:
        raise SystemExit(f"no *_compare.json under {field_dir}; run the field comparison first")
    field = json.loads(record_path.read_text())

    task = TASKS[args.task]
    candidates = {s.name: s for s in build_candidate_spaces(finetuning=finetuning)}
    missing = [c for c in args.candidate if c not in candidates]
    if missing:
        raise SystemExit(f"unknown candidate(s) {missing}; available: {sorted(candidates)}")
    spaces = [candidates[c] for c in args.candidate]

    # Reference values must come from the *same seeds* these candidates ran on.
    # A paired signed-rank test on mismatched pairs is not a weaker test, it is a
    # different and meaningless one, and nothing in the numbers reveals the
    # mismatch. This script evaluates on seeds 200+, as does headline.py, while
    # astro.bench.run evaluates the field on 100+ -- so the field is only a
    # usable reference when it recorded seeds that actually match.
    headline_path = args.out / f"{args.task}_headline.json"
    reference: dict[str, list[float]] = {}
    reference_seconds: dict[str, list[float]] = {}
    reference_seeds: dict[str, list[int]] = {}
    if headline_path.exists():
        headline = json.loads(headline_path.read_text())
        reference.update(headline["values"])
        reference_seconds.update(headline["seconds"])
        stored = headline.get("seed_list") or {}
        # Older headline artifacts recorded only the seed *count*. headline.py
        # has always evaluated on 200+, so that range is recoverable for them --
        # but it is reconstructed explicitly here rather than assumed anywhere
        # downstream.
        legacy = list(range(200, 200 + int(headline.get("seeds", 0) or 0)))
        for name in headline["values"]:
            reference_seeds[name] = stored.get(name) or legacy
    for name in args.against:
        if name not in reference:
            entry = field["evaluation"][name]
            reference[name] = entry["values"]
            reference_seconds[name] = entry["seconds"]
            reference_seeds[name] = entry.get("seeds", [])

    # Run the candidates on whatever seeds the reference used, so the pairing
    # holds by construction rather than by a convention two scripts must share.
    seeds = resolve_paired_seeds(reference_seeds, list(args.against), args.seeds)
    print(f"pairing on the reference's seeds: {seeds[0]}-{seeds[-1]}", flush=True)

    # Same budget, same RNG stream as the field received.
    print(f"tuning {args.candidate} on {args.task} ({args.trials} trials)", flush=True)
    records = tune(task, spaces, trials=args.trials, seed=0)
    for name, record in records.items():
        print(f"  {name}: best={record.best_value:.4f} {record.best_config}", flush=True)

    print(f"\nevaluating on {args.seeds} seeds", flush=True)
    values: dict[str, list[float]] = {}
    seconds: dict[str, list[float]] = {}
    for space in spaces:
        summary = evaluate(task, space, records[space.name].best_config, seeds=seeds)
        values[space.name] = summary.values
        seconds[space.name] = summary.seconds
        low, high = bootstrap_ci(summary.values)
        print(
            f"  {space.name:22s} {statistics.fmean(summary.values):.4f} "
            f"[{low:.4f}, {high:.4f}]  ({statistics.fmean(summary.seconds):.1f}s/run)",
            flush=True,
        )

    lines = [
        f"**{args.task}** — candidate(s) {', '.join(args.candidate)} against "
        f"{', '.join(args.against)}. {args.trials} tuning trials (same budget as the field), "
        f"{args.seeds} evaluation seeds.",
        "",
        "| comparison | candidate | baseline | paired Δ | Cohen's d | seeds won | "
        "exact p | Holm-adj. p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    raw_p: dict[str, float] = {}
    tests = {}
    for candidate in args.candidate:
        for baseline in args.against:
            if baseline not in reference:
                continue
            n = min(len(values[candidate]), len(reference[baseline]))
            if n < 2:
                continue
            test = paired_test(values[candidate][:n], reference[baseline][:n])
            key = f"{candidate} vs {baseline}"
            tests[key] = (test, candidate, baseline, n)
            raw_p[key] = test.p_value
    adjusted = holm_bonferroni(raw_p)

    for key, (test, candidate, baseline, n) in tests.items():
        wins = sum(
            1
            for a, b in zip(values[candidate][:n], reference[baseline][:n], strict=True)
            if a < b
        )
        adj, reject = adjusted[key]
        mark = "**" if reject else ""
        lines.append(
            f"| {key} | {statistics.fmean(values[candidate][:n]):.4f} "
            f"| {statistics.fmean(reference[baseline][:n]):.4f} "
            f"| {test.mean_delta:+.4f} | {test.effect_size:+.2f} ({test.magnitude}) "
            f"| {wins}/{n} | {mark}{test.p_value:.4f}{mark} | {mark}{adj:.4f}{mark} |"
        )

    table = "\n".join(lines)
    print("\n" + table)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = "_".join(args.candidate)
    (args.out / f"{args.task}_candidate_{stem}.md").write_text(table + "\n")
    (args.out / f"{args.task}_candidate_{stem}.json").write_text(
        json.dumps(
            {
                "task": args.task,
                "candidates": args.candidate,
                "against": args.against,
                "trials": args.trials,
                "seeds": args.seeds,
                "configs": {n: r.best_config for n, r in records.items()},
                "values": values,
                "seconds": seconds,
                "reference_values": {k: reference[k] for k in args.against if k in reference},
                "tests": {
                    k: {
                        "mean_delta": t.mean_delta,
                        "effect_size": t.effect_size,
                        "p_value": t.p_value,
                        "holm_adjusted": adjusted[k][0],
                        "reject": adjusted[k][1],
                    }
                    for k, (t, _, _, _) in tests.items()
                },
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out / f'{args.task}_candidate_{stem}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
