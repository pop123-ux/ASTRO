#!/usr/bin/env python3
"""Upload locally recorded paper trajectories to Weights & Biases.

Matplotlib + committed JSON remain the source of record for the paper.  W&B is
an optional experiment browser: useful for overlaying runs, inspecting seeds,
and sharing dashboards, but the manuscript must still rebuild without network
access or a W&B account.

Example:
    pip install wandb
    wandb login
    python scripts/paper/wandb_sync.py \
        --project astro-paper --input artifacts/trajectories.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="astro-paper")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--input", type=Path, default=Path("artifacts/trajectories.json"))
    parser.add_argument("--group", default="paper-trajectories")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    metadata = payload.get("metadata", {})

    if args.dry_run:
        for optimizer, runs in payload.get("runs", {}).items():
            for run in runs:
                print(f"{optimizer}: seed={run.get('seed')} points={len(run.get('step', []))}")
        return 0

    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb is optional; install it with `pip install wandb`") from exc

    for optimizer, runs in payload.get("runs", {}).items():
        for record in runs:
            seed = record.get("seed")
            config = dict(record.get("config", {}))
            config.update({
                "optimizer": optimizer,
                "seed": seed,
                "size": metadata.get("size"),
                "steps": metadata.get("steps"),
                "seq": metadata.get("seq"),
                "astro_lab_fingerprint": metadata.get("astro_lab_fingerprint"),
            })
            run = wandb.init(
                project=args.project,
                entity=args.entity,
                group=args.group,
                name=f"{metadata.get('size', 'model')}-{optimizer}-seed{seed}",
                job_type="paper-trajectory",
                config=config,
                tags=["paper", "trajectory", optimizer],
                reinit=True,
            )
            for i, step in enumerate(record["step"]):
                row = {
                    "global_step": step,
                    "val/loss": record["val_loss"][i],
                    "train/seconds": record["train_seconds"][i],
                }
                train_loss = record.get("last_train_loss", [None] * len(record["step"]))[i]
                if train_loss is not None:
                    row["train/loss_last"] = train_loss
                run.log(row, step=int(step))
            run.summary["final_val_loss"] = record["val_loss"][-1]
            run.summary["training_seconds"] = record["train_seconds"][-1]
            run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
