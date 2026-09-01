#!/usr/bin/env python3
"""Orchestrate a broad, auditable ASTRO ablation campaign.

The repository already contains the tested optimizer implementations and the
resumable benchmark engine in ``scripts/astro_lab.py``. This file only defines
an experimental matrix and launches one benchmark cell at a time.

Suites
------
primary    : dense LR/WD/scalar sensitivity at 124M + 12-trial tuning controls
mechanisms : every baseline/ASTRO variant across 300/900/2700 and 3 seeds
robustness : 45M/124M/355M/774M x 300/900/2700 x 7 optimizers
stability  : long-horizon LR/WD stress envelope

Use ``--dry-run`` before GPU execution. Shard with ``--shard PART/TOTAL``
across Colab sessions. Each job is isolated to its own work directory and the
audit log is append-only, so interrupted sessions resume cleanly.
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

# Match the optimizer names currently exposed by astro_lab.py. Keep AdamW as the
# universal scalar baseline; the matrix-oriented comparison is Muon/NorMuon/
# AdaMuon versus ASTRO. SOAP is covered by the repository's other benchmark
# harnesses and is intentionally not invoked here unless it is explicitly wired
# into astro_lab's registry.
BASELINES = ("adamw", "muon", "normuon", "adamuon")
ASTRO = (
    "astro", "astro_pinned", "astro_trust", "astro_cautious",
    "astro_converging", "astro_gamma25", "astro_gamma50", "astro_gamma0",
    "astro_equil", "astro_plain_wd", "astro_wd_rescaled", "astro_muon_betas",
    "astro_v2", "astro_v2_gamma0", "astro_nosplit", "astro_split100",
    "astro_split300",
)
DEEP_ASTRO = (
    "astro", "astro_muon_betas", "astro_v2", "astro_plain_wd",
    "astro_wd_rescaled", "astro_trust", "astro_converging", "astro_equil",
    "astro_gamma0", "astro_gamma25", "astro_gamma50", "astro_nosplit",
    "astro_split100", "astro_split300",
)
CORE = ("lr=0.0144", "weight_decay=0.02", "scalar_lr_mult=0.4369")
LR = (0.003, 0.006, 0.012, 0.024, 0.048, 0.096)
WD = (0.0, 0.003, 0.01, 0.03, 0.1)
SCALAR = (0.25, 0.35, 0.4369, 0.55, 0.75)
SIZES = ("45M", "124M", "355M", "774M")


@dataclass(frozen=True)
class Job:
    job_id: str
    suite: str
    optimizer: str
    size: str
    steps: int
    seed: int
    trials: int
    config: tuple[str, ...]
    purpose: str

    def command(self, astro_lab: Path, work_dir: Path) -> list[str]:
        cmd = [
            sys.executable, str(astro_lab),
            "--mode", "scaling",
            "--sizes", self.size,
            "--steps", str(self.steps),
            "--optimizers", self.optimizer,
            "--seeds", str(self.seed),
            "--work-dir", str(work_dir),
        ]
        if self.trials:
            cmd.extend(["--trials", str(self.trials)])
        if self.config:
            cmd.append("--config")
            cmd.extend(self.config)
        return cmd


def jid(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12]


def make(suite: str, opt: str, size: str, steps: int, seed: int,
         trials: int, config: tuple[str, ...], purpose: str) -> Job:
    return Job(jid(suite, opt, size, steps, seed, trials, config), suite, opt,
               size, steps, seed, trials, config, purpose)


def build_primary() -> list[Job]:
    jobs: list[Job] = []
    for opt in BASELINES + DEEP_ASTRO:
        for lr in LR:
            for wd in WD:
                for scalar in SCALAR:
                    jobs.append(make(
                        "primary", opt, "124M", 900, 0, 0,
                        (f"lr={lr}", f"weight_decay={wd}",
                         f"scalar_lr_mult={scalar}"),
                        "dense shared LR/WD/scalar sensitivity",
                    ))
    for opt in BASELINES + ("astro", "astro_muon_betas", "astro_v2", "astro_trust"):
        jobs.append(make("primary", opt, "124M", 900, 0, 12, tuple(),
                          "12-trial optimizer-specific tuning control"))
    return jobs


def build_mechanisms() -> list[Job]:
    jobs: list[Job] = []
    for steps in (300, 900, 2700):
        for seed in (100, 101, 102):
            for opt in BASELINES + ASTRO:
                jobs.append(make("mechanisms", opt, "124M", steps, seed, 0,
                                  CORE, "component attribution across horizon"))
    for opt in ("muon", "normuon", "adamuon", "astro_v2", "astro_muon_betas"):
        for factor in (0.25, 0.5, 0.75, 1.0, 1.33, 2.0, 4.0):
            for seed in (100, 101, 102):
                jobs.append(make(
                    "mechanisms", opt, "124M", 900, seed, 0,
                    (f"lr={0.0144 * factor:.8g}",
                     "weight_decay=0.02", "scalar_lr_mult=0.4369"),
                    "factor-of-four LR sensitivity",
                ))
    return jobs


def build_robustness() -> list[Job]:
    jobs: list[Job] = []
    opts = BASELINES + ("astro", "astro_muon_betas", "astro_v2")
    for size in SIZES:
        for steps in (300, 900, 2700):
            for seed in (100, 101, 102):
                for opt in opts:
                    jobs.append(make("robustness", opt, size, steps, seed, 0,
                                      CORE, "frozen-config scale x horizon transfer"))
    return jobs


def build_stability() -> list[Job]:
    jobs: list[Job] = []
    opts = BASELINES + ("astro", "astro_muon_betas", "astro_v2")
    for opt in opts:
        for lr in (0.0015, 0.003, 0.006, 0.012, 0.024, 0.048, 0.096, 0.192):
            for wd in (0.0, 0.01, 0.1, 0.3):
                jobs.append(make(
                    "stability", opt, "124M", 2700, 100, 0,
                    (f"lr={lr}", f"weight_decay={wd}",
                     "scalar_lr_mult=0.4369"),
                    "long-horizon stability envelope",
                ))
    return jobs


def build_suite(name: str) -> list[Job]:
    builders = {
        "primary": build_primary,
        "mechanisms": build_mechanisms,
        "robustness": build_robustness,
        "stability": build_stability,
    }
    if name == "all":
        return [job for fn in builders.values() for job in fn()]
    return builders[name]()


def apply_shard(jobs: list[Job], spec: str | None) -> list[Job]:
    if not spec:
        return jobs
    try:
        part_s, total_s = spec.split("/", 1)
        part, total = int(part_s), int(total_s)
    except ValueError as exc:
        raise SystemExit("--shard must be PART/TOTAL, e.g. 0/3") from exc
    if total <= 0 or not 0 <= part < total:
        raise SystemExit("--shard requires 0 <= PART < TOTAL")
    return [job for i, job in enumerate(jobs) if i % total == part]


def read_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status") == "completed":
            done.add(str(record.get("job_id")))
    return done


def log(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def estimate(jobs: list[Job], sec_per_step: float = 1.2) -> float:
    return sum(job.steps * sec_per_step for job in jobs) / 3600.0


def run(args: argparse.Namespace) -> int:
    jobs = apply_shard(build_suite(args.suite), args.shard)
    if args.max_jobs is not None:
        jobs = jobs[:args.max_jobs]
    root = Path(args.work_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / f"{args.suite}_{args.shard.replace('/', '_') if args.shard else 'all'}.json"
    audit = root / "ablation_audit.jsonl"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "argv": sys.argv,
        "job_count": len(jobs),
        "jobs": [asdict(job) for job in jobs],
    }, indent=2, sort_keys=True) + "\n")

    astro_lab = Path(args.astro_lab).expanduser().resolve()
    if not astro_lab.exists():
        raise SystemExit(f"astro_lab.py not found: {astro_lab}")
    done = read_done(audit)
    print(f"suite={args.suite} jobs={len(jobs)} rough_hours={estimate(jobs):.1f}")
    print(f"manifest={manifest}")
    print(f"audit={audit}")

    for index, job in enumerate(jobs, 1):
        if job.job_id in done:
            print(f"[{index}/{len(jobs)}] SKIP {job.job_id}")
            continue
        work = root / job.suite / (
            f"{job.size}_{job.optimizer}_{job.steps}_s{job.seed}_{job.job_id}"
        )
        work.mkdir(parents=True, exist_ok=True)
        cmd = job.command(astro_lab, work)
        start = time.time()
        log(audit, {"job_id": job.job_id, "status": "started",
                    "started_unix": start, "job": asdict(job), "command": cmd})
        print("\n" + "=" * 92)
        print(f"[{index}/{len(jobs)}] {job.purpose}")
        print(f"optimizer={job.optimizer} size={job.size} steps={job.steps} seed={job.seed}")
        print(f"config={job.config or '(astro_lab tuner)'}")
        print(shlex.join(cmd))
        print("=" * 92)
        try:
            proc = subprocess.run(
                cmd, check=False, timeout=args.timeout_minutes * 60,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            status = "completed" if proc.returncode == 0 else "failed"
            code = proc.returncode
        except subprocess.TimeoutExpired:
            status, code = "timeout", 124
        end = time.time()
        log(audit, {"job_id": job.job_id, "status": status,
                    "ended_unix": end, "duration_seconds": end - start,
                    "returncode": code})
        if status != "completed" and args.stop_on_error:
            return code
    print("\nShard finished; rerun the same command to resume completed jobs.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=("primary", "mechanisms", "robustness", "stability", "all"), default="primary")
    parser.add_argument("--shard")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--astro-lab", default=str(Path(__file__).with_name("astro_lab.py")))
    parser.add_argument("--work-root", default="./artifacts/ablation")
    parser.add_argument("--timeout-minutes", type=int, default=180)
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    jobs = apply_shard(build_suite(args.suite), args.shard)
    if args.max_jobs is not None:
        jobs = jobs[:args.max_jobs]
    if args.dry_run:
        print(f"suite={args.suite} jobs={len(jobs)} rough_hours={estimate(jobs):.1f}")
        for i, job in enumerate(jobs, 1):
            print(f"{i:4d} {job.job_id} {job.optimizer:20s} {job.size:5s} "
                  f"{job.steps:4d} seed={job.seed:3d} trials={job.trials:2d} "
                  + " ".join(job.config))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
