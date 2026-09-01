#!/usr/bin/env python3
"""Generate the paper's results section from the benchmark artifacts.

    python scripts/make_results.py --inject

Reads the JSON written by ``astro.bench.run`` and renders Section 5 of
``docs/paper/paper.md``, replacing the ``<!--RESULTS-->`` marker.

The point is that no number in the paper is typed by hand. Transcription is a
real source of error in empirical papers, and it is the kind that survives
review because the reader has no way to check it. Here the paper is a function
of the artifacts, and re-running this script after a re-run of the benchmark
updates every figure at once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARKER = "<!--RESULTS-->"

#: The two comparisons the paper is actually about, in the order they are read.
HEADLINE = ("adamw", "muon", "astro")


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _fmt_delta(comparison: dict[str, float]) -> tuple[str, str, str]:
    relative = f"{comparison['relative'] * 100:+.2f}%"
    interval = (
        f"[{comparison['ci_low']:+.4f}, {comparison['ci_high']:+.4f}]"
    )
    return relative, interval, "**yes**" if comparison["significant"] else "no"


def table(result: dict, *, only: tuple[str, ...] | None = None) -> str:
    """Render one comparison as a markdown table, sorted by mean loss."""
    evaluation = result["evaluation"]
    comparisons = result["comparisons"]
    control = result["control"]

    names = [n for n in evaluation if only is None or n in only]
    names.sort(key=lambda n: sum(evaluation[n]["values"]) / len(evaluation[n]["values"]))

    lines = [
        "| optimizer | val loss (mean ± sd) | vs control | paired Δ [95% CI] | sig | s/run |",
        "|---|---|---|---|---|---|",
    ]
    for name in names:
        summary = evaluation[name]
        values = summary["values"]
        mean = sum(values) / len(values)
        sd = (sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)) ** 0.5
        seconds = summary["seconds"]
        mean_seconds = sum(seconds) / len(seconds) if seconds else float("nan")
        if name == control:
            relative, interval, significant = "—", "—", "—"
            label = f"`{name}` *(control)*"
        else:
            relative, interval, significant = _fmt_delta(comparisons[name])
            label = f"`{name}`"
        lines.append(
            f"| {label} | {mean:.4f} ± {sd:.4f} | {relative} | {interval} | "
            f"{significant} | {mean_seconds:.1f} |"
        )
    return "\n".join(lines)


def traces(result: dict, *, only: tuple[str, ...] | None = None) -> str:
    """Render tuning traces, which are how under-tuning is made visible."""
    lines = []
    for name, record in result["tuning"].items():
        if only is not None and name not in only:
            continue
        trace = record["trace"]
        marks = [trace[i] for i in range(0, len(trace), max(1, len(trace) // 5))]
        if marks[-1] != trace[-1]:
            marks.append(trace[-1])
        still_falling = len(trace) > 4 and trace[-1] < trace[-4]
        flag = "  ← still descending at the budget limit" if still_falling else ""
        lines.append("- `" + name + "`: " + " → ".join(f"{v:.3f}" for v in marks) + flag)
    return "\n".join(lines)


def best_configs(result: dict, *, only: tuple[str, ...] | None = None) -> str:
    lines = ["| optimizer | selected configuration |", "|---|---|"]
    for name, record in result["tuning"].items():
        if only is not None and name not in only:
            continue
        config = ", ".join(f"`{k}`={v:.4g}" for k, v in record["best_config"].items())
        lines.append(f"| `{name}` | {config} |")
    return "\n".join(lines)


def candidate_section(artifacts: Path) -> str:
    """Candidate rounds and the wall-clock comparison.

    Kept separate from the field tables because a variant proposed *after* seeing
    the field's results is a different kind of evidence from one specified
    beforehand, and a single table would blur that.
    """
    out: list[str] = []

    rounds = sorted(
        artifacts.glob("gpt_scratch_candidate_*.json"), key=lambda f: f.stat().st_mtime
    )
    if rounds:
        seen: dict[str, dict] = {}
        seeds = "?"
        for path in rounds:  # oldest first, so the newest measurement wins
            payload = json.loads(path.read_text())
            seeds = payload.get("seeds", seeds)
            for key, test in payload.get("tests", {}).items():
                seen[key] = test
        out += [
            "### 5.4 Candidate rounds (post-hoc, from scratch)",
            "",
            "Variants proposed after the field was measured, each tuned under the same "
            "budget and RNG stream the field received. A **negative** Delta favours the "
            f"candidate. Exact paired Wilcoxon, Holm-corrected, {seeds} seeds.",
            "",
            "| comparison | paired Delta | Cohen's d | Holm-adj. p |",
            "|---|---|---|---|",
        ]
        for key, test in sorted(seen.items(), key=lambda kv: kv[1]["mean_delta"]):
            mark = "**" if test.get("reject") else ""
            out.append(
                f"| {key} | {test['mean_delta']:+.4f} | {test['effect_size']:+.2f} "
                f"| {mark}{test['holm_adjusted']:.4f}{mark} |"
            )
        out.append("")

    matched = artifacts / "gpt_scratch_time_matched.json"
    if matched.exists():
        payload = json.loads(matched.read_text())
        cost, steps, values = (
            payload["seconds_per_step"], payload["steps_in_budget"], payload["values"]
        )
        out += [
            "### 5.5 Equal wall-clock, from scratch",
            "",
            "Every optimizer runs for the same number of seconds rather than the same "
            "number of steps, so a cheaper step converts directly into more of them. "
            "Measured on an idle machine; the reference is "
            f"`{payload['reference']}` at its natural step count.",
            "",
            "| optimizer | ms/step | steps in budget | val loss |",
            "|---|---|---|---|",
        ]
        for name in sorted(values, key=lambda n: sum(values[n]) / len(values[n])):
            mean = sum(values[name]) / len(values[name])
            out.append(f"| `{name}` | {cost[name] * 1000:.1f} | {steps[name]} | {mean:.4f} |")
        out.append("")
    return "\n".join(out)


def render(artifacts: Path) -> str:
    finetune = load(artifacts / "v2_finetune" / "gpt_finetune_compare.json")
    scratch = load(artifacts / "v2_scratch" / "gpt_scratch_compare.json")
    ablation = load(artifacts / "v2_ablation" / "gpt_finetune_ablation.json")

    out: list[str] = []

    if finetune:
        out += [
            "### 5.1 GPT-2 fine-tuning — the target regime",
            "",
            "AdamW-pretrained on WikiText-2, then fully fine-tuned on tinyshakespeare. "
            f"{finetune['trials']} tuning trials per optimizer at three hyperparameters each, "
            f"best configuration re-run on {finetune['seeds']} seeds. Lower is better; a "
            "negative Δ favours the row.",
            "",
            "**The two comparisons that matter:**",
            "",
            table(finetune, only=HEADLINE),
            "",
            "**Full field:**",
            "",
            table(finetune),
            "",
            "Selected configurations:",
            "",
            best_configs(finetune, only=HEADLINE),
            "",
            "Tuning traces (best-so-far):",
            "",
            traces(finetune),
            "",
        ]

    if scratch:
        out += [
            "### 5.2 GPT-2 from scratch — the held-out regime",
            "",
            "From random initialization on WikiText-2. This is the regime the "
            "Muon/SOAP literature is built on, where matrix methods are expected to win, and "
            "it is held out from the design decisions in Section 6.",
            "",
            table(scratch, only=HEADLINE),
            "",
            "**Full field:**",
            "",
            table(scratch),
            "",
            "Tuning traces (best-so-far):",
            "",
            traces(scratch),
            "",
        ]

    if ablation:
        out += [
            "### 5.3 Component ablation on GPT-2 fine-tuning",
            "",
            "Each variant reverts exactly one component, and each still tunes three "
            "hyperparameters, with the third knob following the anchor configuration so no "
            "variant wastes budget on an inert parameter. A **positive** Δ means the full "
            "method is better than the variant, i.e. the removed component was contributing.",
            "",
            table(ablation),
            "",
        ]

    extra = candidate_section(artifacts)
    if extra:
        out.append(extra)

    if not out:
        return (
            "*Benchmark artifacts not found. Run "
            "`python -m astro.bench.run --task gpt_finetune --trials 16 --seeds 5` "
            "and re-run `scripts/make_results.py`.*"
        )
    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts" / "bench_llm")
    parser.add_argument("--paper", type=Path, default=ROOT / "docs" / "paper" / "paper.md")
    parser.add_argument("--inject", action="store_true", help="write into the paper in place")
    args = parser.parse_args(argv)

    section = render(args.artifacts)
    if not args.inject:
        print(section)
        return 0

    text = args.paper.read_text()
    if MARKER not in text:
        raise SystemExit(
            f"marker {MARKER} not found in {args.paper}; it is replaced on first injection, "
            "so restore it before re-injecting"
        )
    args.paper.write_text(text.replace(MARKER, section))
    print(f"injected {len(section.splitlines())} lines into {args.paper}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
