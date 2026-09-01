"""Benchmark protocol: equal tuning budgets, multiple seeds, honest intervals.

Wen et al. (2025) benchmarked ten optimizers against AdamW and found the real
speedups were near 1.1x rather than the 2x commonly claimed. The gap was not
fraud; it was methodology. Two failure modes did most of the damage:

* **Unequal tuning.** The proposed method gets a careful sweep, the baseline gets
  the defaults from its own paper. Any method wins that way.
* **Single-seed reporting.** Run-to-run spread on small tasks routinely exceeds
  the effect being claimed.

This module makes both hard to do by accident:

* :class:`SearchSpace` requires every optimizer to declare the *same number* of
  tuned dimensions, and :func:`tune` gives every optimizer the *same trial
  count* drawn from the same RNG stream.
* :func:`evaluate` runs multiple seeds and returns a bootstrap interval;
  :func:`paired_comparison` compares two optimizers seed-by-seed rather than
  comparing means of independent runs.
* Results record wall-clock alongside step counts, so a per-step win that costs
  more time than it saves shows up as what it is.

Nothing here prevents a determined person from reporting a flattering subset. It
only removes the excuse that the protocol was ambiguous.
"""

from __future__ import annotations

import math
import random
import statistics
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import nn
from torch.optim import Optimizer

__all__ = [
    "TaskResult",
    "Task",
    "OptimizerFactory",
    "SearchSpace",
    "TuningRecord",
    "EvaluationSummary",
    "tune",
    "evaluate",
    "bootstrap_ci",
    "paired_comparison",
]


@dataclass
class TaskResult:
    """Outcome of one training run.

    Attributes
    ----------
    final:
        Primary objective, **lower is better** for every task in this suite so
        that comparisons never depend on remembering a direction.
    curve:
        Objective sampled during training, for plots and for steps-to-target.
    steps:
        Optimizer steps actually taken.
    seconds:
        Wall-clock spent in the training loop.
    reached:
        First step index at which the task's target was met, or ``None``.
    """

    final: float
    curve: list[float] = field(default_factory=list)
    steps: int = 0
    seconds: float = 0.0
    reached: int | None = None


#: A task trains a model with the supplied optimizer factory and returns a result.
Task = Callable[["OptimizerFactory", int], TaskResult]


class OptimizerFactory(Protocol):
    """Builds an optimizer for a model. Closes over one hyperparameter draw."""

    def __call__(self, model: nn.Module) -> Optimizer: ...


@dataclass(frozen=True)
class SearchSpace:
    """Log-uniform search ranges for one optimizer.

    Parameters
    ----------
    name:
        Optimizer label used in reports.
    build:
        ``(model, config) -> Optimizer``.
    ranges:
        ``{hyperparameter: (low, high)}``, sampled log-uniformly.
    fixed:
        Hyperparameters held at published defaults, not counted against budget.

    Raises
    ------
    ValueError
        If ``ranges`` is empty. Every optimizer must tune something, and
        :func:`tune` additionally checks that all competitors tune the same
        *number* of dimensions.
    """

    name: str
    build: Callable[[nn.Module, dict[str, float]], Optimizer]
    ranges: dict[str, tuple[float, float]]
    fixed: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.ranges:
            raise ValueError(f"{self.name}: must tune at least one hyperparameter")

    @property
    def dimension(self) -> int:
        return len(self.ranges)

    def sample(self, rng: random.Random) -> dict[str, float]:
        """Draw one log-uniform configuration."""
        return {
            key: math.exp(rng.uniform(math.log(low), math.log(high)))
            for key, (low, high) in self.ranges.items()
        }

    def factory(self, config: dict[str, float]) -> OptimizerFactory:
        merged = {**self.fixed, **config}
        return lambda model: self.build(model, merged)


@dataclass
class TuningRecord:
    """Result of a tuning sweep for one optimizer on one task."""

    name: str
    best_config: dict[str, float]
    best_value: float
    #: Best-so-far after each trial. A curve still descending at the end of the
    #: budget is direct evidence the optimizer was under-tuned.
    trace: list[float] = field(default_factory=list)
    trials: int = 0


@dataclass
class EvaluationSummary:
    """Multi-seed evaluation of one configuration."""

    name: str
    values: list[float]
    seconds: list[float]
    steps_to_target: list[int | None]
    config: dict[str, float] = field(default_factory=dict)
    #: The seeds these values came from, in order. Recorded because a paired
    #: test across two summaries is only valid when they share it, and nothing
    #: in the numbers themselves reveals a mismatch.
    seeds: list[int] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.values)

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.values) if len(self.values) > 1 else 0.0

    @property
    def mean_seconds(self) -> float:
        return statistics.fmean(self.seconds) if self.seconds else 0.0

    def interval(self, confidence: float = 0.95, resamples: int = 10_000) -> tuple[float, float]:
        return bootstrap_ci(self.values, confidence=confidence, resamples=resamples)


def tune(
    task: Task,
    spaces: Sequence[SearchSpace],
    *,
    trials: int,
    seed: int = 0,
    tuning_seed: int = 12345,
) -> dict[str, TuningRecord]:
    """Sweep every optimizer under an identical budget.

    Every optimizer gets ``trials`` draws from its own log-uniform space, using
    an RNG seeded identically per optimizer so that draw *k* is equally "lucky"
    across competitors. Tuning runs on a single fixed seed; the winning
    configuration is then re-run across seeds by :func:`evaluate`, which keeps
    tuning from silently selecting on seed noise.

    Raises
    ------
    ValueError
        If the optimizers do not all tune the same number of hyperparameters.
        This is the check that stops a three-knob method being compared against
        a one-knob baseline.
    """
    dimensions = {space.dimension for space in spaces}
    if len(dimensions) > 1:
        detail = ", ".join(f"{s.name}={s.dimension}" for s in spaces)
        raise ValueError(
            "every optimizer must tune the same number of hyperparameters "
            f"for the comparison to be fair; got {detail}"
        )

    records: dict[str, TuningRecord] = {}
    total = len(spaces) * trials
    done = 0
    started = time.perf_counter()
    for space in spaces:
        rng = random.Random(tuning_seed)
        best_value, best_config, trace = math.inf, {}, []
        for trial in range(trials):
            config = space.sample(rng)
            trial_started = time.perf_counter()
            try:
                value = task(space.factory(config), seed).final
            except (RuntimeError, ValueError, FloatingPointError):
                # Divergence is a legitimate outcome of a hyperparameter draw,
                # not a crash. Record it as such rather than aborting the sweep.
                value = math.inf
            if math.isnan(value):
                value = math.inf
            if value < best_value:
                best_value, best_config = value, config
            trace.append(best_value)

            # A sweep of this length is routinely left unattended, and silence
            # is indistinguishable from a hang. Report every trial with an ETA
            # derived from what has actually run, not from an estimate.
            done += 1
            elapsed = time.perf_counter() - started
            remaining = (total - done) * elapsed / done
            shown = "diverged" if math.isinf(value) else f"{value:.4f}"
            print(
                f"[tune {done:3d}/{total}] {space.name:22s} trial {trial + 1:2d} "
                f"{shown:>9s}  best {best_value:.4f}  "
                f"({time.perf_counter() - trial_started:.0f}s, eta {remaining / 60:.0f}m)",
                flush=True,
            )
        records[space.name] = TuningRecord(
            name=space.name,
            best_config=best_config,
            best_value=best_value,
            trace=trace,
            trials=trials,
        )
    return records


def evaluate(
    task: Task,
    space: SearchSpace,
    config: dict[str, float],
    *,
    seeds: Iterable[int],
) -> EvaluationSummary:
    """Re-run one configuration across seeds."""
    values, seconds, reached, used = [], [], [], []
    factory = space.factory(config)
    for index, seed in enumerate(seeds):
        started = time.perf_counter()
        result = task(factory, seed)
        print(
            f"[eval] {space.name:22s} seed {seed} ({index + 1}) "
            f"{result.final:.4f}  ({time.perf_counter() - started:.0f}s)",
            flush=True,
        )
        values.append(result.final)
        seconds.append(result.seconds or (time.perf_counter() - started))
        reached.append(result.reached)
        used.append(seed)
    return EvaluationSummary(
        name=space.name, values=values, seconds=seconds, steps_to_target=reached,
        config=config, seeds=used,
    )


def bootstrap_ci(
    values: Sequence[float], *, confidence: float = 0.95, resamples: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean.

    With three to five seeds this interval is wide. That is the point: it is an
    honest depiction of what a handful of runs can support, and a difference that
    does not survive it should not be reported as a difference.
    """
    if not values:
        return (math.nan, math.nan)
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choices(values, k=len(values))) for _ in range(resamples)
    )
    lower = means[int((1.0 - confidence) / 2 * resamples)]
    upper = means[min(resamples - 1, int((1.0 + confidence) / 2 * resamples))]
    return (lower, upper)


def paired_comparison(
    treatment: EvaluationSummary, control: EvaluationSummary, *, resamples: int = 10_000
) -> dict[str, float]:
    """Compare two optimizers seed-by-seed.

    Pairing matters: seeds contribute a shared component of variance (the data
    order, the initialisation), so differencing within a seed removes noise that
    an unpaired comparison of means would leave in.

    Returns
    -------
    dict
        ``mean_delta`` (treatment minus control; negative favours treatment),
        ``ci_low``/``ci_high`` for that delta, ``win_rate``, and ``relative``
        (fractional improvement over the control mean).
    """
    if len(treatment.values) != len(control.values):
        raise ValueError("paired comparison needs the same seeds on both sides")
    deltas = [t - c for t, c in zip(treatment.values, control.values, strict=True)]
    low, high = bootstrap_ci(deltas, resamples=resamples)
    control_mean = statistics.fmean(control.values)
    return {
        "mean_delta": statistics.fmean(deltas),
        "ci_low": low,
        "ci_high": high,
        "win_rate": sum(1 for d in deltas if d < 0) / len(deltas),
        "relative": (statistics.fmean(deltas) / control_mean) if control_mean else math.nan,
        "significant": float(high < 0.0 or low > 0.0),
    }


def seed_everything(seed: int) -> None:
    """Seed Python and torch so a task is reproducible across processes."""
    random.seed(seed)
    torch.manual_seed(seed)
