"""Benchmark harness for optimizer comparisons.

Entry point: ``python -m astro.bench.run --task finetune``.
"""

from astro.bench.protocol import (
    EvaluationSummary,
    SearchSpace,
    TaskResult,
    bootstrap_ci,
    evaluate,
    paired_comparison,
    tune,
)
from astro.bench.registry import build_ablation_spaces, build_spaces
from astro.bench.tasks import TASKS

__all__ = [
    "EvaluationSummary",
    "SearchSpace",
    "TaskResult",
    "bootstrap_ci",
    "evaluate",
    "paired_comparison",
    "tune",
    "build_spaces",
    "build_ablation_spaces",
    "TASKS",
]
