#!/usr/bin/env python3
"""Record validation-loss trajectories for frozen optimizer configurations.

This is deliberately separate from the headline benchmark.  Periodic validation
would distort benchmark wall-clock, so the main ``astro_lab.py`` remains clean;
this script subtracts validation-probe time from the reported training clock and
exists only to generate learning-dynamics figures.

Example
-------
python scripts/paper/run_trajectory.py \
  --size 124M --steps 900 --seed 100 --eval-every 75 \
  --optimizers muon normuon adamuon astro_muon_betas astro_v2 \
  --config lr=0.010205978810672643 weight_decay=0.001059717889298923 \
           scalar_lr_mult=0.4369 \
  --cache /content/drive/MyDrive/astro/paper_trajectory_cache.pt \
  --out artifacts/trajectories.json

For optimizer-specific frozen configurations, pass ``--config-json`` pointing to
JSON of the form ``{"muon": {"lr": ...}, "astro_v2": {"lr": ...}}``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
LAB_PATH = ROOT / "scripts" / "astro_lab.py"


def load_lab():
    spec = importlib.util.spec_from_file_location("astro_lab_paper", LAB_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {LAB_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lab = load_lab()


def parse_config(items: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"expected KEY=VALUE, got {item!r}")
        out[key] = float(value)
    out.setdefault("weight_decay", 0.01)
    out.setdefault("scalar_lr_mult", 0.1)
    return out


def batches(source: torch.Tensor, generator: torch.Generator, *, batch: int,
            seq: int, count: int, device: str):
    for _ in range(count):
        start = torch.randint(0, source.numel() - seq - 1, (batch,), generator=generator)
        yield torch.stack([source[s:s + seq] for s in start]).to(device)


def evaluate(model, validation: torch.Tensor, *, batch: int, seq: int,
             device: str, count: int) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    generator = torch.Generator().manual_seed(777)
    with torch.no_grad():
        for x in batches(validation, generator, batch=batch, seq=seq,
                         count=count, device=device):
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                total += float(model(x, labels=x).loss)
    if was_training:
        model.train()
    return total / count


def train_trajectory(name: str, config: dict[str, float], *, data, size: str,
                     steps: int, seq: int, vocab: int, seed: int,
                     eval_every: int, val_batches: int) -> dict:
    from transformers import GPT2Config, GPT2LMHeadModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("trajectory runs are intended for CUDA; CPU would be impractically slow")

    shape = dict(lab.SIZES[size])
    batch = shape.pop("batch")
    train, validation = data

    torch.manual_seed(seed)
    model = GPT2LMHeadModel(GPT2Config(n_positions=seq, vocab_size=vocab, **shape)).to(device)
    if size in ("355M", "774M"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.train()
    optimizer = lab.build_optimizer(name, model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    generator = torch.Generator().manual_seed(seed + 4242)
    warmup = max(1, steps // 10)

    record = {
        "seed": seed,
        "config": config,
        "step": [],
        "val_loss": [],
        "train_seconds": [],
        "last_train_loss": [],
    }

    started = time.perf_counter()
    probe_seconds = 0.0

    def probe(step: int, train_loss: float | None) -> None:
        nonlocal probe_seconds
        probe_started = time.perf_counter()
        val = evaluate(model, validation, batch=batch, seq=seq, device=device,
                       count=val_batches)
        probe_seconds += time.perf_counter() - probe_started
        train_seconds = time.perf_counter() - started - probe_seconds
        record["step"].append(step)
        record["val_loss"].append(val)
        record["train_seconds"].append(train_seconds)
        record["last_train_loss"].append(train_loss)
        print(f"[{name}] step {step:4d}/{steps}: val={val:.4f}, "
              f"training_time={train_seconds:.1f}s", flush=True)

    probe(0, None)

    source = batches(train, generator, batch=batch, seq=seq, count=steps, device=device)
    last_loss = None
    for step_index, x in enumerate(source, start=1):
        zero_index = step_index - 1
        factor = ((zero_index + 1) / (warmup + 1) if zero_index < warmup else
                  0.1 + 0.45 * (1 + math.cos(math.pi * (zero_index - warmup) /
                                             max(1, steps - warmup))))
        for group in optimizer.param_groups:
            group.setdefault("base_lr", group["lr"])
            group["lr"] = group["base_lr"] * factor

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(x, labels=x).loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        last_loss = float(loss.detach())

        if step_index % eval_every == 0 or step_index == steps:
            probe(step_index, last_loss)

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", default="124M", choices=sorted(lab.SIZES))
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--optimizers", nargs="+",
                        default=["muon", "normuon", "adamuon", "astro_v2"])
    parser.add_argument("--config", nargs="+", metavar="K=V",
                        default=["lr=0.010205978810672643",
                                 "weight_decay=0.001059717889298923",
                                 "scalar_lr_mult=0.4369"])
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--eval-every", type=int, default=75)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts" / "fineweb_cache.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "trajectories.json")
    args = parser.parse_args()

    unknown = [name for name in args.optimizers if name not in lab.known_optimizers()]
    if unknown:
        raise SystemExit(f"unknown optimizer {unknown[0]!r}")

    shared = parse_config(args.config)
    specific = json.loads(args.config_json.read_text()) if args.config_json else {}

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    batch = lab.SIZES[args.size]["batch"]
    needed = batch * args.seq * (args.steps + 60) + lab.VALIDATION_TOKENS
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    tokens = lab.load_tokens(tokenizer, needed, args.cache)
    data = (tokens[lab.VALIDATION_TOKENS:], tokens[:lab.VALIDATION_TOKENS])

    fingerprint = hashlib.sha256(LAB_PATH.read_bytes()).hexdigest()[:12]
    payload = {"metadata": {}, "runs": {}}
    if args.out.exists():
        payload = json.loads(args.out.read_text())
        payload.setdefault("metadata", {})
        payload.setdefault("runs", {})

    payload["metadata"].update({
        "astro_lab_fingerprint": fingerprint,
        "size": args.size,
        "steps": args.steps,
        "seq": args.seq,
        "validation_tokens": lab.VALIDATION_TOKENS,
        "eval_every": args.eval_every,
        "val_batches_per_probe": args.val_batches,
        "note": "train_seconds excludes periodic validation-probe time",
    })

    for name in args.optimizers:
        config = dict(shared)
        config.update(specific.get(name, {}))
        print(f"\n=== {name}: {config} ===", flush=True)
        record = train_trajectory(name, config, data=data, size=args.size,
                                  steps=args.steps, seq=args.seq,
                                  vocab=tokenizer.vocab_size, seed=args.seed,
                                  eval_every=args.eval_every,
                                  val_batches=args.val_batches)
        existing = payload["runs"].setdefault(name, [])
        existing = [run for run in existing if run.get("seed") != args.seed]
        existing.append(record)
        payload["runs"][name] = existing
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"saved {args.out}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
