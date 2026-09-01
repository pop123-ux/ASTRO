#!/usr/bin/env python3
"""Run the language-model optimizer comparison at real GPT-2 scale, on a GPU.

    python scripts/bench_gpu_llm.py --task gpt_finetune --size gpt2-small
    python scripts/bench_gpu_llm.py --task gpt_scratch  --size gpt2-medium --trials 16

The CPU suite (``python -m astro.bench.run --task gpt_finetune``) trains a
4-layer, 128-wide GPT-2 -- 806K non-embedding parameters. That is enough to show
the *mechanism*, and nothing more. Optimizer rankings are known to change with
scale; that is the central finding of Wen et al. (arXiv:2509.02046), who measured
matrix-optimizer speedups shrinking from ~30% to ~10% between small models and
1.2B. A result at 806K parameters is therefore evidence about 806K parameters.

This script runs the **same protocol** -- equal tuning budgets, multiple seeds,
paired statistics (Algorithm 4 in ``docs/paper/paper.md``) -- at sizes where the
comparison is worth making.

Tokenisation
------------
``--tokenizer bpe`` uses GPT-2's real BPE vocabulary through ``tiktoken``, which
is what the published numbers are computed against. It is the default here and
*not* on CPU, where a 50257-row embedding would dominate a small model and push
most of the parameter count onto the scalar path. ``--tokenizer char`` reproduces
the CPU configuration for a controlled comparison across scales.

Sizes follow nanoGPT: ``gpt2-small`` is the 124M configuration (12 layers, 12
heads, 768 wide, 1024 context).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from astro.bench.corpora import load_corpus  # noqa: E402
from astro.bench.gpt import GPT, GPTConfig  # noqa: E402
from astro.bench.protocol import (  # noqa: E402
    OptimizerFactory,
    TaskResult,
    evaluate,
    paired_comparison,
    tune,
)
from astro.bench.registry import build_ablation_spaces, build_spaces  # noqa: E402
from astro.bench.run import format_report  # noqa: E402

#: nanoGPT's GPT-2 configurations, plus two smaller ones for quick checks.
SIZES: dict[str, dict[str, int]] = {
    "tiny": dict(n_layer=4, n_head=4, n_embd=256, block_size=256),
    "small-ish": dict(n_layer=6, n_head=6, n_embd=384, block_size=256),
    "gpt2-small": dict(n_layer=12, n_head=12, n_embd=768, block_size=1024),
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024, block_size=1024),
}


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device found. This script is for the GPU numbers; use "
            "`python -m astro.bench.run --task gpt_finetune` for the CPU suite."
        )
    return torch.device("cuda")


def _tokenised(
    corpus_name: str, tokenizer: str, max_chars: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return ``(train, val, vocab_size)`` under the chosen tokenizer."""
    if tokenizer == "char":
        corpus = load_corpus(corpus_name, max_chars=max_chars)
        return corpus.train, corpus.val, corpus.vocab_size

    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "--tokenizer bpe needs tiktoken: pip install tiktoken\n"
            "(or pass --tokenizer char to reproduce the CPU configuration)"
        ) from exc

    from astro.bench.corpora import CORPUS_SOURCES, data_root

    filename = {
        "wikitext2": CORPUS_SOURCES["wikitext2-train"][1],
        "shakespeare": CORPUS_SOURCES["shakespeare"][1],
    }[corpus_name]
    text = (data_root() / filename).read_text(encoding="utf-8", errors="replace")[:max_chars]
    encoding = tiktoken.get_encoding("gpt2")
    tokens = torch.tensor(encoding.encode_ordinary(text), dtype=torch.long)
    split = int(len(tokens) * 0.9)
    # 50304 rather than 50257: nanoGPT pads to a multiple of 64 for tensor cores.
    return tokens[:split], tokens[split:], 50304


def _batch(
    data: torch.Tensor, batch: int, block: int, generator: torch.Generator, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, data.numel() - block - 1, (batch,), generator=generator)
    x = torch.stack([data[s : s + block] for s in starts])
    y = torch.stack([data[s + 1 : s + 1 + block] for s in starts])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def _validate(model: GPT, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    model.eval()
    total = 0.0
    for x, y in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            total += float(model(x, y)[1])
    model.train()
    return total / len(batches)


def make_task(
    *,
    finetune: bool,
    size: str,
    tokenizer: str,
    steps: int,
    batch: int,
    grad_accum: int,
    pretrain_steps: int,
    train_tokens: int | None,
    max_chars: int,
    target: float,
):
    """Build a task closure over one configuration.

    The AdamW-pretrained checkpoint for the fine-tuning task is computed once per
    seed and cached in-process, so every optimizer fine-tunes from bit-identical
    weights -- the same discipline as the CPU suite.
    """
    device = _device()
    shape = SIZES[size]
    pretrain_cache: dict[int, dict[str, torch.Tensor]] = {}

    train_corpus = "shakespeare" if finetune else "wikitext2"
    train_data, val_data, vocab = _tokenised(train_corpus, tokenizer, max_chars)
    if finetune:
        pre_train, _, _ = _tokenised("wikitext2", tokenizer, max_chars)

    def build(seed: int) -> GPT:
        torch.manual_seed(seed)
        return GPT(GPTConfig(vocab_size=vocab, **shape)).to(device)

    def pretrain(seed: int) -> dict[str, torch.Tensor]:
        if seed in pretrain_cache:
            return pretrain_cache[seed]
        model = build(seed)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=6e-4, weight_decay=1e-1, betas=(0.9, 0.95)
        )
        generator = torch.Generator().manual_seed(seed + 31337)
        model.train()
        for _ in range(pretrain_steps):
            x, y = _batch(pre_train, batch, shape["block_size"], generator, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, y)[1]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        pretrain_cache[seed] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        return pretrain_cache[seed]

    def task(factory: OptimizerFactory, seed: int) -> TaskResult:
        model = build(seed)
        if finetune:
            model.load_state_dict(pretrain(seed))
        optimizer = factory(model)

        generator = torch.Generator().manual_seed(777)
        pool = train_data if train_tokens is None else train_data[:train_tokens]
        val_batches = [
            _batch(val_data, batch, shape["block_size"], generator, device) for _ in range(8)
        ]
        generator = torch.Generator().manual_seed(seed + 4242)

        model.train()
        curve: list[float] = []
        reached: int | None = None
        torch.cuda.synchronize()
        started = time.perf_counter()
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(grad_accum):
                x, y = _batch(pool, batch, shape["block_size"], generator, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(x, y)[1] / grad_accum
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % max(1, steps // 20) == 0 or step == steps - 1:
                value = _validate(model, val_batches)
                curve.append(value)
                if reached is None and value <= target:
                    reached = step
        torch.cuda.synchronize()
        return TaskResult(
            final=curve[-1],
            curve=curve,
            steps=steps,
            seconds=time.perf_counter() - started,
            reached=reached,
        )

    return task


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="gpt_finetune", choices=["gpt_finetune", "gpt_scratch"])
    parser.add_argument("--size", default="small-ish", choices=sorted(SIZES))
    parser.add_argument("--tokenizer", default="bpe", choices=["bpe", "char"])
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--pretrain-steps", type=int, default=3000)
    parser.add_argument(
        "--train-tokens",
        type=int,
        default=200_000,
        help="fine-tuning pool size; the low-data regime is where mismatch bites",
    )
    parser.add_argument("--max-chars", type=int, default=20_000_000)
    parser.add_argument("--target", type=float, default=3.0)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--control", default="adamw")
    parser.add_argument("--out", type=Path, default=Path("artifacts/bench_gpu_llm"))
    args = parser.parse_args(argv)

    finetune = args.task == "gpt_finetune"
    task = make_task(
        finetune=finetune,
        size=args.size,
        tokenizer=args.tokenizer,
        steps=args.steps,
        batch=args.batch,
        grad_accum=args.grad_accum,
        pretrain_steps=args.pretrain_steps,
        train_tokens=args.train_tokens if finetune else None,
        max_chars=args.max_chars,
        target=args.target,
    )

    spaces = (
        build_ablation_spaces(finetuning=finetune)
        if args.ablation
        else build_spaces(finetuning=finetune)
    )
    records = tune(task, spaces, trials=args.trials, seed=0)
    summaries = {
        space.name: evaluate(
            task, space, records[space.name].best_config, seeds=range(100, 100 + args.seeds)
        )
        for space in spaces
    }

    control = args.control if args.control in summaries else next(iter(summaries))
    if args.ablation:
        control = "astro_full" if "astro_full" in summaries else control
    result = {
        "task": f"{args.task}[{args.size},{args.tokenizer}]",
        "trials": args.trials,
        "seeds": args.seeds,
        "control": control,
        "tuning": {name: vars(record) for name, record in records.items()},
        "evaluation": {name: vars(summary) for name, summary in summaries.items()},
        "comparisons": {
            name: paired_comparison(summary, summaries[control])
            for name, summary in summaries.items()
            if name != control
        },
        "_summaries": summaries,
    }

    print(format_report(result), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in result.items() if not k.startswith("_")}
    suffix = "ablation" if args.ablation else "compare"
    (args.out / f"{args.task}_{args.size}_{suffix}.json").write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
