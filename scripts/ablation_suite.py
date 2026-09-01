#!/usr/bin/env python3
"""Launch a broad, auditable optimizer ablation campaign using ``astro_lab.py``.

The repository already contains the tested optimizer implementation and the
resumable benchmark engine. This file is intentionally only an ORCHESTRATOR:
it creates a large pre-specified campaign of shared configurations, ASTRO
component variants, scale/horizon checks, and stability checks, then executes
those jobs one-by-one. It never reimplements an optimizer.

Why this exists
---------------
A paper can be weakened by a baseline that received one convenient
configuration while the proposed method received extensive tuning. The suite
therefore has explicit *search strata* and equal trial budgets. The primary
comparison also includes ASTRO's named component variants so the final model is
not presented as an unexplained bundle.

The generated command manifest is a permanent provenance artifact. Every job
is resumable because ``astro_lab.py`` already checkpoints its state. Results
are stored under one directory per suite/cell, avoiding accidental state
collisions between independent Colab sessions.

Recommended use on three Colab instances
-----------------------------------------

    # Instance 1: deep baseline/configuration search
    python scripts/ablation_suite.py --suite primary --shard 0/3 \
        --work-root /content/drive/MyDrive/astro/ablation

    # Instance 2: component/mechanism attribution
    python scripts/ablation_suite.py --suite mechanisms --shard 1/3 \
        --work-root /content/drive/MyDrive/astro/ablation

    # Instance 3: scale + horizon + stability robustness
    python scripts/ablation_suite.py --suite robustness --shard 2/3 \
        --work-root /content/drive/MyDrive/astro/ablation

Start with ``--dry-run``. Use ``--max-jobs`` to cap a single four-hour session.
The launcher is deliberately conservative: it does not mark a job complete
until ``astro_lab.py`` exits successfully.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


BASELINES = ("muon", "normuon", "adamuon", "soap")
ASTRO_VARIANTS = (
    "astro",
    "astro_pinned",
    "astro_trust",
    "astro_cautious",
    "astro_converging",
    "astro_gamma25",
    "astro_gamma50",
    "astro_gamma0",
    "astro_equil",
    "astro_plain_wd",
    "astro_wd_rescaled",
    "astro_muon_betas",
    "astro_v2",
    "astro_v2_gamma0",
    "astro_nosplit",
    "astro_split100",
    "astro_split300",
)

# These are deliberately reviewable discrete points. The inner benchmark can
# still perform its own random/continuous tuning when ``--trials`` is used.
LR_POINTS = (0.003, 0.006, 0.012, 0.024, 0.048, 0.096)
WD_POINTS = (0.0, 0.003, 0.01, 0.03, 0.1)
SCALAR_POINTS = (0.25, 0.35, 0.4369, 0.55, 0.75)
CORE_PINS = ("lr=0.0144", "weight_decay=0.02", "scalar_lr_mult=0.4369")

SIZES = ("45M", "124M", "355M", "774M")
OPTIMIZERS = BASELINES + ASTRO_VARIANTS


@dataclass(frozen=True)
class Job:
    job_id: str
    suite: str
    optimizer: str
    size: str
    steps: int
    seeds: tuple[int, ...]
    trials: int
    pins: tuple[str, ...]
    purpose: str

    def command(self, astro_lab: Path, work_dir: Path) -> list[str]:
        cmd = [
            sys.executable, str(astro_lab),
            "--mode", "scaling",
            "--sizes", self.size,
            "--steps", str(self.steps),
            "--optimizers", self.optimizer,
            "--trials", str(self.trials),
            "--seeds", str(len(self.seeds)),
            "--work-dir", str(work_dir),
        ]
        for pin in self.pins:
            # astro_lab accepts repeated --pin KEY=VALUE arguments. Keeping the
            # flag/value as separate argv entries also prevents shell quoting
            # bugs when paths or future values contain special characters.
            cmd.extend(["--pin", pin])
        return cmd


def _id(*parts: object) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def make_job(suite: str, optimizer: str, size: str, steps: int,
             seeds: tuple[int, ...], trials: int, pins: tuple[str, ...],
             purpose: str) -> Job:
    return Job(
        job_id=_id(suite, optimizer, size, steps, seeds, trials, pins),
        suite=suite,
        optimizer=optimizer,
        size=size,
        steps=steps,
        seeds=seeds,
        trials=trials,
        pins=pins,
        purpose=purpose,
    )


def primary() -> list[Job]:
    """High-depth 124M configuration coverage.

    The first tier is intentionally dense: baseline optimizers and every ASTRO
    variant are exposed to the same LR/WD/scalar point set. This is not the final
    held-out evaluation; it establishes whether an apparent win is confined to
    a narrow part of parameter space.

    A second tier uses the inner tuner (12 trials) over the same 3-dimensional
    budget for the four strongest baselines and the best-known ASTRO recipes.
    """
    jobs: list[Job] = []
    dense = BASELINES + (
        "astro", "astro_muon_betas", "astro_v2", "astro_plain_wd",
        "astro_wd_rescaled", "astro_trust", "astro_converging",
        "astro_equil", "astro_gamma25", "astro_gamma50", "astro_gamma0",
        "astro_nosplit", "astro_split100", "astro_split300",
    )
    for opt in dense:
        for lr in LR_POINTS:
            for wd in WD_POINTS:
                for scalar in SCALAR_POINTS:
                    jobs.append(make_job(
                        "primary", opt, "124M", 900, (0,), 0,
                        (f"lr={lr}", f"weight_decay={wd}", f"scalar_lr_mult={scalar}"),
                        "dense LR/WD/scalar configuration point",
                    ))

    tuned = BASELINES + ("astro", "astro_muon_betas", "astro_v2", "astro_trust")
    for opt in tuned:
        jobs.append(make_job(
            "primary", opt, "124M", 900, (0,), 12, tuple(),
            "12-trial continuous tuning control",
        ))
    return jobs


def mechanisms() -> list[Job]:
    """ASTRO component isolation at multiple horizons and independent seeds."""
    variants = (
        "muon", "normuon", "adamuon", "astro", "astro_muon_betas", "astro_v2",
        "astro_cautious", "astro_plain_wd", "astro_wd_rescaled", "astro_converging",
        "astro_equil", "astro_trust", "astro_nosplit", "astro_split100", "astro_split300",
        "astro_gamma0", "astro_gamma25", "astro_gamma50", "astro_v2_gamma0",
    )
    jobs: list[Job] = []
    for steps in (300, 900, 2700):
        for opt in variants:
            jobs.append(make_job(
                "mechanisms", opt, "124M", steps, (100, 101, 102), 0,
                CORE_PINS, "one-factor component attribution over horizon",
            ))

    # Explicit LR transfer around the observed ASTRO-v2 point. This tests whether
    # a gain survives a factor-of-four misspecification instead of only the
    # selected point.
    for opt in ("muon", "normuon", "adamuon", "astro_v2", "astro_muon_betas"):
        for factor in (0.25, 0.5, 0.75, 1.0, 1.33, 2.0, 4.0):
            jobs.append(make_job(
                "mechanisms", opt, "124M", 900, (100, 101, 102), 0,
                (f"lr={0.0144 * factor:.8g}", "weight_decay=0.02", "scalar_lr_mult=0.4369"),
                "learning-rate sensitivity around common reference point",
            ))
    return jobs


def robustness() -> list[Job]:
    """Symmetric model-size and training-horizon transfer study."""
    opts = ("muon", "normuon", "adamuon", "soap", "astro", "astro_muon_betas", "astro_v2")
    jobs: list[Job] = []
    for size in SIZES:
        for steps in (300, 900, 2700):
            for opt in opts:
                # Keep the same nominal recipe for transfer. The primary suite
                # owns hyperparameter discovery; this suite asks whether that
                # discovered recipe extrapolates.
                jobs.append(make_job(
                    "robustness", opt, size, steps, (100, 101, 102), 0,
                    CORE_PINS, "scale x horizon transfer",
                ))
    return jobs


def stability() -> list[Job]:
    """Stress envelope to expose divergence, over-damping, and long-run drift."""
    opts = ("muon", "normuon", "adamuon", "astro", "astro_muon_betas", "astro_v2")
    jobs: list[Job] = []
    for opt in opts:
        for lr in (0.0015, 0.003, 0.006, 0.012, 0.024, 0.048, 0.096, 0.192):
            for wd in (0.0, 0.01, 0.1, 0.3):
                jobs.append(make_job(
                    "stability", opt, "124M", 2700, (100,), 0,
                    (f"lr={lr}", f"weight_decay={wd}", "scalar_lr_mult=0.4369"),
                    "long-horizon stability/divergence envelope",
                ))
    return jobs


def build_suite(name: str) -> list[Job]:
    builders = {
        "primary": primary,
        "mechanisms": mechanisms,
        "robustness": robustness,
        "stability": stability,
    }
    if name == "all":
        return [job for suite in builders.values() for job in suite()]
    return builders[name]()


def shard(jobs: list[Job], spec: str | None) -> list[Job]:
    if not spec:
        return jobs
    left, slash, right = spec.partition("/")
    if not slash:
        raise SystemExit("--shard must be PART/TOTAL, e.g. 0/3")
    part, total = int(left), int(right)
    if total <= 0 or part < 0 or part >= total:
        raise SystemExit("--shard requires 0 <= PART < TOTAL")
    return [job for i, job in enumerate(jobs) if i % total == part]


def rough_hours(jobs: list[Job], sec_per_step: float = 1.15) -> float:
    # Planning only. Polar iterations, model size, and data-loader overhead make
    # this deliberately an order-of-magnitude estimate, never a hard timeout.
    return sum(job.steps * sec_per_step for job in jobs) / 3600.0


def write_manifest(path: Path, jobs: list[Job], argv: list[str]) -> None:
    payload = {
        "schema_version": 1,
        "created_unix": time.time(),
        "argv": argv,
        "job_count": len(jobs),
        "jobs": [asdict(job) for job in jobs],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_log(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def completed_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("status") == "completed":
            done.add(str(rec.get("job_id")))
    return done


def run(args: argparse.Namespace) -> int:
    jobs = shard(build_suite(args.suite), args.shard)
    if args.max_jobs is not None:
        jobs = jobs[:args.max_jobs]
    root = Path(args.work_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / f"{args.suite}_{args.shard.replace('/', '_') if args.shard else 'all'}.json"
    audit = root / "ablation_audit.jsonl"
    write_manifest(manifest, jobs, sys.argv)

    astro_lab = Path(args.astro_lab).expanduser().resolve()
    if not astro_lab.exists():
        raise SystemExit(f"astro_lab.py not found: {astro_lab}")
    done = completed_ids(audit)
    print(f"jobs={len(jobs)} rough_hours={rough_hours(jobs):.1f}")
    print(f"manifest={manifest}")
    print(f"audit={audit}")

    for index, job in enumerate(jobs, 1):
        if job.job_id in done:
            print(f"[{index}/{len(jobs)}] SKIP {job.optimizer} {job.size} {job.steps} {job.job_id}")
            continue

        job_dir = root / job.suite / f"{job.size}_{job.optimizer}_{job.steps}"
        job_dir.mkdir(parents=True, exist_ok=True)
        cmd = job.command(astro_lab, job_dir)
        started = time.time()
        append_log(audit, {
            "job_id": job.job_id,
            "status": "started",
            "started_unix": started,
            "job": asdict(job),
            "command": cmd,
        })
        print("\n" + "=" * 96)
        print(f"[{index}/{len(jobs)}] {job.purpose}")
        print(f"optimizer={job.optimizer} size={job.size} steps={job.steps} seeds={job.seeds}")
        print(f"pins={job.pins or '(tuned by astro_lab)'}")
        print(shlex.join(cmd))
        print("=" * 96)
        try:
            result = subprocess.run(cmd, check=False, timeout=args.timeout_minutes * 60,
                                    env={**os.environ, "PYTHONUNBUFFERED": "1"})
            status = "completed" if result.returncode == 0 else "failed"
            code = result.returncode
        except subprocess.TimeoutExpired:
            status, code = "timeout", 124
        ended = time.time()
        append_log(audit, {
            "job_id": job.job_id,
            "status": status,
            "ended_unix": ended,
            "duration_seconds": ended - started,
            "returncode": code,
        })
        if status != "completed" and args.stop_on_error:
            return code
    print("\nCampaign shard finished. Re-run the same command to resume completed jobs.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=("primary", "mechanisms", "robustness", "stability", "all"), default="primary")
    parser.add_argument("--shard", help="PART/TOTAL, e.g. 0/3")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--astro-lab", default=str(Path(__file__).with_name("astro_lab.py")))
    parser.add_argument("--work-root", default="./artifacts/ablation")
    parser.add_argument("--timeout-minutes", type=int, default=180)
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    jobs = shard(build_suite(args.suite), args.shard)
    if args.max_jobs is not None:
        jobs = jobs[:args.max_jobs]
    if args.dry_run:
        print(f"jobs={len(jobs)} rough_hours={rough_hours(jobs):.1f}")
        for i, job in enumerate(jobs, 1):
            print(f"{i:5d} {job.job_id} {job.optimizer:20s} {job.size:5s} "
                  f"{job.steps:4d} {','.join(map(str, job.seeds)):12s} "
                  f"{' '.join(job.pins)}")
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
