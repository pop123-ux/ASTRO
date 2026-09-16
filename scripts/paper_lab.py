#!/usr/bin/env python3
"""Paper-grade ASTRO experiment harness.

This is a thin, auditable layer over ``astro_lab.py``.  Historical experiments
remain reproducible with the old harness; paper-grade experiments use this file
so protocol fixes cannot silently rewrite the provenance of earlier results.

The paper harness changes only things required for a defensible optimizer
comparison:

* one fixed FineWeb-Edu token pool is shared by every run and every workdir;
* the same warmup/cosine factor schedules *both* Muon's matrix LR and its
  auxiliary AdamW LR (the old harness accidentally left ``adamw_lr`` constant);
* biases/norm gains are no-decay for every optimizer, while matrices and
  embedding/head tables use the declared weight decay;
* ``adamuon_ref`` follows the official AdaMuon update: sign-transformed
  Nesterov momentum, Newton--Schulz, elementwise second moment, and the
  published 0.2 RMS-aligned rescaling; its auxiliary parameters use AdamW;
* two minimal controls are exposed for causal attribution:
  ``muon_m90`` and the ASTRO-v2 no-split / gamma-zero variants;
* every completed run writes a lightweight trajectory/provenance JSON at no
  additional forward-pass cost.

Run directly exactly like ``astro_lab.py``.  Two wrapper-only options are
accepted before the ordinary lab arguments::

    python paper_lab.py \
      --paper-corpus /content/drive/MyDrive/astro/paper_campaign/shared/fineweb_fixed.pt \
      --paper-corpus-tokens 12000000 \
      --mode scaling --sizes 124M --steps 900 ...

The token file itself is the dataset snapshot for this campaign.  Its SHA256 is
printed and recorded in every trajectory, so later Hugging Face revisions cannot
silently change the training ground.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

import astro_lab as lab

HERE = Path(__file__).resolve()
BASE_HARNESS = Path(lab.__file__).resolve()
DEFAULT_CORPUS = Path(
    os.environ.get(
        "ASTRO_PAPER_CORPUS",
        "/content/drive/MyDrive/astro/paper_campaign/shared/fineweb_fixed.pt",
    )
)
DEFAULT_CORPUS_TOKENS = 12_000_000

_PAPER_CORPUS = DEFAULT_CORPUS
_PAPER_CORPUS_TOKENS = DEFAULT_CORPUS_TOKENS
_CORPUS_SHA256: str | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_source(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_load_tokens(tokenizer, needed: int, _cache: Path) -> torch.Tensor:
    """Return one immutable token pool for the whole paper campaign.

    ``astro_lab`` used to size the corpus from ``--steps``.  That meant a
    900-step and a 940-step run sampled from different index ranges even with
    the same RNG seed.  Here every run samples from the same fixed pool.
    """
    global _CORPUS_SHA256

    path = _PAPER_CORPUS
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = path.with_suffix(path.suffix + ".json")

    if path.exists():
        tokens = torch.load(path, map_location="cpu")
        if tokens.numel() != _PAPER_CORPUS_TOKENS:
            raise RuntimeError(
                f"paper corpus has {tokens.numel():,} tokens, expected "
                f"{_PAPER_CORPUS_TOKENS:,}; use a new corpus path rather than "
                "silently changing an existing campaign"
            )
    else:
        from datasets import load_dataset

        print(
            f"streaming one fixed FineWeb-Edu paper corpus: {_PAPER_CORPUS_TOKENS:,} tokens",
            flush=True,
        )
        stream = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        collected: list[int] = []
        for record in stream:
            collected.extend(tokenizer(record["text"]).input_ids)
            if len(collected) >= _PAPER_CORPUS_TOKENS:
                break
        tokens = torch.tensor(
            collected[:_PAPER_CORPUS_TOKENS], dtype=torch.long
        )
        torch.save(tokens, path)

    if needed > tokens.numel():
        raise RuntimeError(
            f"this run asks for {needed:,} tokens but the fixed paper corpus has "
            f"{tokens.numel():,}; create a new campaign/corpus rather than changing "
            "the pool in-place"
        )

    if metadata.exists():
        info = json.loads(metadata.read_text())
        _CORPUS_SHA256 = info.get("sha256")
    if not _CORPUS_SHA256:
        _CORPUS_SHA256 = _sha256_file(path)
        metadata.write_text(
            json.dumps(
                {
                    "tokens": int(tokens.numel()),
                    "dtype": str(tokens.dtype),
                    "sha256": _CORPUS_SHA256,
                    "dataset": "HuggingFaceFW/fineweb-edu/sample-10BT",
                    "tokenizer": "gpt2",
                },
                indent=2,
            )
        )

    print(
        f"fixed paper corpus: {tokens.numel():,} tokens  sha256={_CORPUS_SHA256[:16]}",
        flush=True,
    )
    # Intentionally return the entire fixed pool.  The training sampler must see
    # the same index range at 900, 940, 2700, and across separate workdirs.
    return tokens


def paper_groups(model, optimizer_name: str, config, weight_decay: float) -> list[dict]:
    """Common parameter routing and decay policy for every paper optimizer."""
    try:
        from transformers.pytorch_utils import Conv1D
    except ImportError:  # pragma: no cover
        Conv1D = ()  # type: ignore[assignment]

    table_ids: set[int] = set()
    transposed_ids: set[int] = set()
    fused_ids: set[int] = set()

    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            table_ids.add(id(module.weight))
    output = getattr(model, "lm_head", None)
    if output is not None:
        table_ids.add(id(output.weight))

    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2:
            continue
        if Conv1D and isinstance(module, Conv1D):
            transposed_ids.add(id(weight))
        if name.endswith("c_attn"):
            fused_ids.add(id(weight))

    width = getattr(config, "n_embd", None)
    nodecay: list[torch.nn.Parameter] = []
    tables: list[torch.nn.Parameter] = []
    fused: list[torch.nn.Parameter] = []
    matrices: dict[bool, list[torch.nn.Parameter]] = {}

    for param in model.parameters():
        if param.ndim < 2:
            nodecay.append(param)
        elif id(param) in table_ids:
            tables.append(param)
        elif id(param) in fused_ids and width is not None:
            fused.append(param)
        else:
            matrices.setdefault(id(param) in transposed_ids, []).append(param)

    groups: list[dict] = []
    if nodecay:
        groups.append(
            {
                "params": nodecay,
                "spectral": False,
                "transposed": False,
                "weight_decay": 0.0,
            }
        )
    if tables:
        groups.append(
            {
                "params": tables,
                "spectral": False,
                "transposed": False,
                "weight_decay": weight_decay,
            }
        )
    for transposed, params in matrices.items():
        groups.append(
            {
                "params": params,
                "spectral": True,
                "transposed": transposed,
                "weight_decay": weight_decay,
            }
        )
    if fused:
        groups.append(
            {
                "params": fused,
                "spectral": True,
                "transposed": all(id(p) in transposed_ids for p in fused),
                "blocks": (width, width, width) if optimizer_name == "astro" else None,
                "weight_decay": weight_decay,
            }
        )
    return groups


class AdaMuonReference(torch.optim.Optimizer):
    """Single-GPU faithful AdaMuon update plus AdamW on auxiliary parameters.

    The matrix path follows the authors' official implementation.  Newton--
    Schulz itself stays float32 because a Tesla T4 has no native BF16 tensor
    cores; this changes numerical precision, not the optimizer rule.
    """

    def __init__(
        self,
        groups,
        lr: float = 6e-4,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        ns_steps: int = 5,
    ) -> None:
        super().__init__(
            groups,
            dict(
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                betas=betas,
                eps=eps,
                ns_steps=ns_steps,
            ),
        )

    @torch.no_grad()
    def step(self):  # type: ignore[override]
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]
                state["step"] = state.get("step", 0) + 1
                step = state["step"]

                if group["spectral"]:
                    operator = grad.T if group.get("transposed") else grad
                    if "momentum" not in state:
                        state["momentum"] = torch.zeros_like(operator)
                    buf = state["momentum"]
                    buf.mul_(group["momentum"]).add_(operator)
                    nesterov = operator.add(buf, alpha=group["momentum"])

                    # Published AdaMuon: sign-stabilize *before* orthogonalizing.
                    direction = lab.newton_schulz(
                        torch.sign(nesterov), steps=group["ns_steps"]
                    )
                    if "variance" not in state:
                        state["variance"] = torch.zeros_like(direction)
                    variance = state["variance"]
                    variance.mul_(group["momentum"]).addcmul_(
                        direction, direction, value=1.0 - group["momentum"]
                    )
                    direction = direction.div(variance.sqrt().add_(group["eps"]))

                    # Official RMS-aligned rescaling: RMS(update) = 0.2.
                    scale = 0.2 * math.sqrt(direction.numel()) / (
                        direction.norm() + group["eps"]
                    )
                    direction.mul_(scale)
                    if group.get("transposed"):
                        direction = direction.T
                    if group["weight_decay"]:
                        param.mul_(1.0 - group["lr"] * group["weight_decay"])
                    param.add_(direction.to(param.dtype), alpha=-group["lr"])
                    continue

                # Official setup uses AdamW for embeddings/head/norms/biases.
                beta1, beta2 = group["betas"]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(grad)
                    state["variance"] = torch.zeros_like(grad)
                momentum = state["momentum"].mul_(beta1).add_(grad, alpha=1 - beta1)
                variance = state["variance"]
                variance.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                update = (momentum / (1 - beta1**step)) / (
                    (variance / (1 - beta2**step)).sqrt().add_(group["eps"])
                )
                if group["weight_decay"]:
                    param.mul_(1.0 - group["lr"] * group["weight_decay"])
                param.add_(update, alpha=-group["lr"])


def _paper_space_for(name: str) -> dict[str, tuple[float, float]]:
    # AdaMuon's published RMS alignment is designed to reuse Adam-scale LRs.
    if name == "adamuon_ref":
        return {"lr": lab.ADAM_LR, "weight_decay": (1e-3, 3e-1)}
    if name == "muon_m90":
        return {
            "lr": lab.MUON_LR,
            "weight_decay": (1e-3, 3e-1),
            "scalar_lr_mult": lab.SCALAR_MULT,
        }
    return _ORIGINAL_SPACE_FOR(name)


def _paper_build_optimizer(name: str, model, config: dict[str, float]):
    if name not in lab.known_optimizers():
        raise SystemExit(f"unknown paper optimizer {name!r}")

    weight_decay = config.get("weight_decay", 0.01)
    scalar = config.get("scalar_lr_mult", 1.0)

    if name == "adamw":
        # Keep the common no-decay policy even for the pure AdamW baseline.
        groups = paper_groups(model, "muon", model.config, weight_decay)
        for group in groups:
            group.pop("spectral", None)
            group.pop("transposed", None)
            group.pop("blocks", None)
        return torch.optim.AdamW(
            groups,
            lr=config["lr"],
            betas=(0.9, config.get("beta2", 0.95)),
        )

    if name == "adamuon_ref":
        groups = paper_groups(model, "muon", model.config, weight_decay)
        return AdaMuonReference(
            groups,
            lr=config["lr"],
            weight_decay=weight_decay,
            betas=(0.9, config.get("beta2", 0.95)),
        )

    if name in {"muon", "muon_m90", "normuon", "adamuon"}:
        groups = paper_groups(model, "muon", model.config, weight_decay)
        momentum = 0.90 if name == "muon_m90" else 0.95
        if name in {"muon", "muon_m90"}:
            return lab.Muon(
                groups,
                lr=config["lr"],
                adamw_lr=config["lr"] * scalar,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        if name == "normuon":
            return lab.NorMuon(
                groups,
                lr=config["lr"],
                adamw_lr=config["lr"] * scalar,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        # Historical local AdaMuon remains available only for reproduction.
        return lab.AdaMuon(
            groups,
            lr=config["lr"],
            adamw_lr=config["lr"] * scalar,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    overrides = dict(lab.VARIANTS.get(name, {}))
    split = overrides.pop("split", True)
    groups = paper_groups(
        model, "astro" if split else "muon", model.config, weight_decay
    )
    return lab.Astro(
        groups,
        lr=config["lr"],
        scalar_lr_mult=scalar,
        weight_decay=weight_decay,
        **overrides,
    )


def _schedule_group(group: dict, factor: float) -> None:
    group.setdefault("paper_base_lr", group["lr"])
    group["lr"] = group["paper_base_lr"] * factor
    # Critical fairness fix: Muon/NorMuon/local-AdaMuon auxiliary parameters use
    # adamw_lr internally.  They must receive the same schedule shape.
    if "adamw_lr" in group:
        group.setdefault("paper_base_adamw_lr", group["adamw_lr"])
        group["adamw_lr"] = group["paper_base_adamw_lr"] * factor


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
    """The common paper training loop; optimizer choice changes only the update."""
    from transformers import GPT2Config, GPT2LMHeadModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    shape = dict(lab.SIZES[size])
    batch = shape.pop("batch")
    train, validation = data

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = GPT2LMHeadModel(
        GPT2Config(n_positions=seq, vocab_size=vocab, **shape)
    ).to(device)
    model.train()
    if size in ("355M", "774M"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    optimizer = _paper_build_optimizer(name, model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    generator = torch.Generator().manual_seed(seed + 4242)
    warmup = max(1, steps // 10)
    started = time.perf_counter()

    trace_steps: list[int] = []
    trace_loss: list[float] = []
    trace_seconds: list[float] = []

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
                    math.pi * (step - warmup) / max(1, steps - warmup)
                )
            )
        )
        for group in optimizer.param_groups:
            _schedule_group(group, factor)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = model(x, labels=x).loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % log_every == 0 or step == steps - 1:
            elapsed = time.perf_counter() - started
            loss_value = float(loss.detach())
            trace_steps.append(step)
            trace_loss.append(loss_value)
            trace_seconds.append(elapsed)
            rate = (step + 1) / max(elapsed, 1e-9)
            print(
                f"      step {step:5d}/{steps} train {loss_value:.4f} "
                f"({rate:.2f} it/s, eta {(steps-step)/max(rate,1e-9)/60:.0f}m)",
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
    seconds = time.perf_counter() - started
    peak_vram = (
        torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
    )

    trace_dir = Path("paper_traces")
    trace_dir.mkdir(parents=True, exist_ok=True)
    config_digest = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:10]
    trace_path = trace_dir / (
        f"{size}_{steps}_{name}_seed{seed}_{config_digest}.json"
    )
    trace_path.write_text(
        json.dumps(
            {
                "optimizer": name,
                "size": size,
                "steps": steps,
                "seed": seed,
                "config": config,
                "final_val_loss": value,
                "seconds": seconds,
                "peak_vram_gb": peak_vram,
                "train_trace": {
                    "step": trace_steps,
                    "loss": trace_loss,
                    "seconds": trace_seconds,
                },
                "paper_lab_sha256": _sha256_source(HERE),
                "astro_lab_sha256": _sha256_source(BASE_HARNESS),
                "corpus_sha256": _CORPUS_SHA256,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            },
            indent=2,
        )
    )

    del model, optimizer
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return value, seconds


_ORIGINAL_SPACE_FOR = lab.space_for


def configure_paper_harness(
    corpus: Path | str = DEFAULT_CORPUS,
    corpus_tokens: int = DEFAULT_CORPUS_TOKENS,
) -> None:
    """Patch the legacy lab in-memory with the paper protocol."""
    global _PAPER_CORPUS, _PAPER_CORPUS_TOKENS
    _PAPER_CORPUS = Path(corpus)
    _PAPER_CORPUS_TOKENS = int(corpus_tokens)

    # Keep historical names, add only the controls/reference needed by the paper.
    if "adamuon_ref" not in lab.BASELINES:
        lab.BASELINES = tuple(lab.BASELINES) + ("adamuon_ref", "muon_m90")
    lab.VARIANTS.update(
        {
            "astro_v2_nosplit": {
                "betas": (0.9, 0.95),
                "cautious_wd": False,
                "split": False,
            },
            "astro_v2_gamma0_nosplit": {
                "betas": (0.9, 0.95),
                "cautious_wd": False,
                "variance_power": 0.0,
                "split": False,
            },
        }
    )

    lab.space_for = _paper_space_for
    lab.build_optimizer = _paper_build_optimizer
    lab.load_tokens = fixed_load_tokens
    lab.train_once = paper_train_once


def _wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--paper-corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument(
        "--paper-corpus-tokens", type=int, default=DEFAULT_CORPUS_TOKENS
    )
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    wrapper, remaining = _wrapper_args(argv)
    configure_paper_harness(wrapper.paper_corpus, wrapper.paper_corpus_tokens)

    print(
        f"paper_lab {_sha256_source(HERE)[:12]} ({HERE.stat().st_size:,} bytes)",
        flush=True,
    )
    print(
        f"base astro_lab {_sha256_source(BASE_HARNESS)[:12]} "
        f"({BASE_HARNESS.stat().st_size:,} bytes)",
        flush=True,
    )

    old_argv = sys.argv
    try:
        sys.argv = [str(HERE), *remaining]
        return lab.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
