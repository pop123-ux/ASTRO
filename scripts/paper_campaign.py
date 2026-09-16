#!/usr/bin/env python3
"""Paper-grade ASTRO benchmark wrapper.

This file intentionally leaves scripts/astro_lab.py untouched so historical
results retain their exact provenance. It patches only the parts needed for
the confirmatory paper campaign:

1. schedules Muon's/AdaMuon's auxiliary AdamW learning rate with exactly the
   same warmup/cosine factor as the matrix learning rate;
2. pins FineWeb-Edu to v1.0.0 and uses one shared fixed token cache independent
   of the requested step budget;
3. adds a single-GPU, algorithm-faithful AdaMuon reference baseline following
   Algorithm 1 / the authors' public implementation;
4. adds ASTRO-v2-NoSplit and ASTRO-v2-Blockwise controls. The latter performs
   NorMuon-style adaptation and Frobenius restoration independently inside
   each Q/K/V block, while ASTRO-v2 restores one global norm after the split
   blocks are recombined. This is the key control for ASTRO's candidate novel
   contribution: global post-spectral adaptive redistribution across semantic
   blocks after blockwise spectral geometry.

Run exactly like astro_lab.py, for example:

    python scripts/paper_campaign.py --mode scaling --sizes 124M --steps 900 \
      --optimizers muon normuon adamuon_ref astro_v2 --trials 5 --seeds 5 \
      --work-dir /content/drive/MyDrive/astro/paper_main --stop-after 5

The script prints a campaign fingerprint derived from BOTH this wrapper and the
underlying astro_lab.py. Record it with every paper result.
"""

from __future__ import annotations

import gc
import hashlib
import math
import os
import time
from pathlib import Path

import torch

import astro_lab as lab


FINEWEB_REVISION = "v1.0.0"
TRAIN_TOKENS = 12_000_000
TOTAL_TOKENS = lab.VALIDATION_TOKENS + TRAIN_TOKENS
SHARED_CACHE = Path(
    os.environ.get(
        "ASTRO_PAPER_TOKEN_CACHE",
        "/content/drive/MyDrive/astro/paper_shared/fineweb_edu_v1.0.0_12p4m.pt",
    )
)


class PublishedAdaMuon(torch.optim.Optimizer):
    """AdaMuon reference semantics plus an auxiliary AdamW path.

    The authors' distributed reference stores the NS communication buffer in
    bfloat16. T4 (sm_75) has no native BF16 tensor-core path, so this campaign
    uses astro_lab's float32 Newton-Schulz for all Muon-family optimizers.
    That is a numerical implementation accommodation, not an algorithm change.
    """

    def __init__(
        self,
        groups,
        *,
        lr: float,
        adamw_lr: float,
        weight_decay: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-8,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
    ) -> None:
        super().__init__(
            groups,
            dict(
                lr=lr,
                adamw_lr=adamw_lr,
                weight_decay=weight_decay,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                eps=eps,
                betas=adamw_betas,
            ),
        )

    @torch.no_grad()
    def step(self):  # type: ignore[override]
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                operator = grad.T if group.get("transposed") else grad
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(operator)
                    state["variance"] = torch.zeros_like(operator)
                state["step"] += 1

                if group["spectral"]:
                    momentum = state["momentum"].mul_(group["momentum"]).add_(operator)
                    update = (
                        operator.add(momentum, alpha=group["momentum"])
                        if group["nesterov"]
                        else momentum
                    )
                    update = lab.newton_schulz(
                        torch.sign(update), steps=group["ns_steps"]
                    )

                    variance = state["variance"]
                    beta = group["momentum"]
                    variance.mul_(beta).addcmul_(update, update, value=1.0 - beta)
                    update = update / variance.sqrt().add_(group["eps"])

                    rms_scale = (
                        0.2
                        * math.sqrt(update.size(0) * update.size(1))
                        / update.norm().clamp_min(group["eps"])
                    )
                    update = (update * rms_scale).to(param.dtype)
                    if group.get("transposed"):
                        update = update.T

                    if group["weight_decay"]:
                        param.mul_(1.0 - group["lr"] * group["weight_decay"])
                    param.add_(update, alpha=-group["lr"])
                else:
                    beta1, beta2 = group["betas"]
                    step = state["step"]
                    momentum = state["momentum"].mul_(beta1).add_(grad, alpha=1 - beta1)
                    variance = state["variance"]
                    variance.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    update = (momentum / (1 - beta1**step)) / (
                        (variance / (1 - beta2**step)).sqrt().add_(group["eps"])
                    )
                    if group["weight_decay"]:
                        param.mul_(1.0 - group["adamw_lr"] * group["weight_decay"])
                    param.add_(update, alpha=-group["adamw_lr"])


class AstroV2Blockwise(lab.Astro):
    """ASTRO-v2 with per-block rather than global post-spectral redistribution."""

    @torch.no_grad()
    def step(self):  # type: ignore[override]
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                operator = grad.T if group.get("transposed") else grad
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(operator)
                    if group["spectral"]:
                        state["variance"] = torch.zeros(operator.size(0), device=param.device)
                    else:
                        state["variance"] = torch.zeros_like(operator)
                state["step"] += 1
                step = state["step"]

                momentum = state["momentum"].mul_(beta1).add_(operator, alpha=1 - beta1)

                if group["spectral"]:
                    lookahead = operator.lerp(momentum, beta1)
                    sizes = group.get("blocks") or (operator.size(0),)
                    limit = group["split_steps"]
                    if limit is not None and step > limit:
                        sizes = (operator.size(0),)

                    variance = state["variance"]
                    offset = 0
                    pieces = []
                    for chunk in lookahead.split(list(sizes), 0):
                        filtered = lab.polar_iterate(
                            chunk, group["ns_steps"], group["converging"]
                        )
                        rows = filtered.size(0)
                        row_var = variance[offset : offset + rows]
                        row_var.mul_(beta2).add_(
                            filtered.pow(2).mean(dim=1), alpha=1 - beta2
                        )
                        moment = row_var / (1 - beta2**step)
                        denominator = moment.pow(
                            0.5 * group["variance_power"]
                        ).add_(group["eps"])
                        scaled = filtered / denominator.unsqueeze(1)

                        redistributed = scaled * (
                            filtered.norm() / scaled.norm().clamp_min(1e-12)
                        )
                        redistributed = redistributed * max(
                            1.0, rows / filtered.size(1)
                        ) ** 0.5
                        pieces.append(redistributed)
                        offset += rows
                    direction = torch.cat(pieces, dim=0).to(param.dtype)
                else:
                    variance = state["variance"]
                    variance.mul_(beta2).addcmul_(operator, operator, value=1 - beta2)
                    denominator = (variance / (1 - beta2**step)).sqrt().add_(group["eps"])
                    direction = (momentum / (1 - beta1**step)) / denominator

                if group.get("transposed"):
                    direction = direction.T
                lab.apply_weight_decay(
                    param,
                    direction,
                    group["lr"] * group["weight_decay"],
                    cautious=group["cautious_wd"],
                    rescale=group["cautious_wd_rescale"],
                )
                param.add_(direction, alpha=-group["lr"])


ORIGINAL_BUILD_OPTIMIZER = lab.build_optimizer
ORIGINAL_SPACE_FOR = lab.space_for

lab.BASELINES = tuple(dict.fromkeys((*lab.BASELINES, "adamuon_ref")))
lab.VARIANTS.update(
    {
        "astro_v2_nosplit": {
            "betas": (0.9, 0.95),
            "cautious_wd": False,
            "split": False,
        },
        "astro_v2_blockwise": {
            "betas": (0.9, 0.95),
            "cautious_wd": False,
        },
    }
)


def paper_space_for(name: str) -> dict[str, tuple[float, float]]:
    if name == "adamuon_ref":
        return {
            "lr": lab.ADAM_LR,
            "weight_decay": (1e-3, 3e-1),
            "momentum": (0.90, 0.99),
        }
    return ORIGINAL_SPACE_FOR(name)


def paper_build_optimizer(name: str, model, config: dict[str, float]):
    if name == "adamuon_ref":
        groups = lab.build_groups(model, "muon", model.config)
        return PublishedAdaMuon(
            groups,
            lr=config["lr"],
            adamw_lr=config["lr"],
            weight_decay=config["weight_decay"],
            momentum=config.get("momentum", 0.95),
        )
    if name == "astro_v2_blockwise":
        groups = lab.build_groups(model, "astro", model.config)
        return AstroV2Blockwise(
            groups,
            lr=config["lr"],
            scalar_lr_mult=config.get("scalar_lr_mult", 1.0),
            weight_decay=config["weight_decay"],
            betas=(0.9, 0.95),
            cautious_wd=False,
        )
    return ORIGINAL_BUILD_OPTIMIZER(name, model, config)


def paper_load_tokens(tokenizer, needed: int, cache: Path) -> torch.Tensor:
    del needed, cache
    SHARED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if SHARED_CACHE.exists():
        tokens = torch.load(SHARED_CACHE, map_location="cpu")
        if tokens.numel() != TOTAL_TOKENS:
            raise RuntimeError(
                f"shared token cache has {tokens.numel():,} tokens, expected "
                f"exactly {TOTAL_TOKENS:,}; remove it deliberately before rebuilding"
            )
        print(
            f"paper corpus cache: {tokens.numel():,} tokens "
            f"(FineWeb-Edu {FINEWEB_REVISION})",
            flush=True,
        )
        return tokens

    from datasets import load_dataset

    print(
        f"streaming pinned FineWeb-Edu {FINEWEB_REVISION} for {TOTAL_TOKENS:,} tokens",
        flush=True,
    )
    stream = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
        revision=FINEWEB_REVISION,
    )
    tokenizer.model_max_length = 10**30
    collected: list[int] = []
    for record in stream:
        collected.extend(tokenizer(record["text"], add_special_tokens=False).input_ids)
        if len(collected) >= TOTAL_TOKENS:
            break
    tokens = torch.tensor(collected[:TOTAL_TOKENS], dtype=torch.long)
    torch.save(tokens, SHARED_CACHE)
    return tokens


def schedule_optimizer_lrs(optimizer, factor: float) -> None:
    """Apply one schedule factor to every LR field consumed by an update."""
    for group in optimizer.param_groups:
        for field in ("lr", "adamw_lr"):
            if field not in group:
                continue
            base = f"paper_base_{field}"
            group.setdefault(base, group[field])
            group[field] = group[base] * factor


def paper_train_once(
    name: str,
    config: dict[str, float],
    seed: int,
    *,
    data,
    size: str,
    steps: int,
    seq: int,
    vocab: int,
    log_every: int,
) -> tuple[float, float]:
    from transformers import GPT2Config, GPT2LMHeadModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    shape = dict(lab.SIZES[size])
    batch = shape.pop("batch")
    train, validation = data

    torch.manual_seed(seed)
    model = GPT2LMHeadModel(
        GPT2Config(n_positions=seq, vocab_size=vocab, **shape)
    ).to(device)
    model.train()
    if size in ("355M", "774M"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    optimizer = paper_build_optimizer(name, model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    generator = torch.Generator().manual_seed(seed + 4242)
    warmup = max(1, steps // 10)
    started = time.perf_counter()

    def batches(source, gen, count):
        for _ in range(count):
            start = torch.randint(
                0, source.numel() - seq - 1, (batch,), generator=gen
            )
            yield torch.stack([source[s : s + seq] for s in start]).to(device)

    for step, x in enumerate(batches(train, generator, steps)):
        factor = (
            (step + 1) / (warmup + 1)
            if step < warmup
            else 0.1
            + 0.45
            * (
                1
                + math.cos(
                    math.pi
                    * (step - warmup)
                    / max(1, steps - warmup)
                )
            )
        )
        schedule_optimizer_lrs(optimizer, factor)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = model(x, labels=x).loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % log_every == 0:
            elapsed = time.perf_counter() - started
            rate = (step + 1) / max(elapsed, 1e-9)
            print(
                f"      step {step:5d}/{steps} train {float(loss.detach()):.4f} "
                f"({rate:.2f} it/s, eta "
                f"{(steps - step) / max(rate, 1e-9) / 60:.0f}m)",
                flush=True,
            )

    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x in batches(validation, torch.Generator().manual_seed(777), 20):
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                total += float(model(x, labels=x).loss)
            count += 1
    value = total / max(1, count)

    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return value, time.perf_counter() - started


def paper_announce(args) -> None:
    wrapper = Path(__file__).resolve().read_bytes()
    base = Path(lab.__file__).resolve().read_bytes()
    digest = hashlib.sha256(wrapper + b"\0" + base).hexdigest()
    print(
        f"paper_campaign {digest[:12]} "
        f"(wrapper {len(wrapper):,} + astro_lab {len(base):,} bytes)",
        flush=True,
    )
    print(
        f"protocol FineWeb-Edu@{FINEWEB_REVISION}; shared cache {SHARED_CACHE}; "
        f"total tokens {TOTAL_TOKENS:,}",
        flush=True,
    )
    if args.config:
        return
    unknown = set(args.pin) - {
        k for n in args.optimizers for k in paper_space_for(n)
    }
    if unknown:
        raise SystemExit(
            f"--pin {sorted(unknown)[0]!r} is not tuned by any requested optimizer"
        )
    print("search space per optimizer (pinned values marked *):", flush=True)
    for name in args.optimizers:
        terms = []
        for key, (lo, hi) in paper_space_for(name).items():
            terms.append(
                f"{key}=*{args.pin[key]:g}"
                if key in args.pin
                else f"{key}=[{lo:g},{hi:g}]"
            )
        print(f"  {name:20s} " + "  ".join(terms), flush=True)


lab.space_for = paper_space_for
lab.build_optimizer = paper_build_optimizer
lab.load_tokens = paper_load_tokens
lab.train_once = paper_train_once
lab.announce = paper_announce


def main() -> int:
    return lab.main()


if __name__ == "__main__":
    raise SystemExit(main())
