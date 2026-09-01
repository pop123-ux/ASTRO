"""Language-model benchmark tasks: GPT-2 from scratch, and GPT-2 fine-tuned.

Both tasks train :class:`astro.bench.gpt.GPT` -- nanoGPT's implementation of the
GPT-2 architecture -- on real text (:mod:`astro.bench.corpora`).

``gpt_scratch``
    Pretrain from random initialisation on WikiText-2. This is the regime the
    Muon/SOAP/Hyperball literature is built on and where matrix-structured
    optimizers are expected to win. Including it is a control: a method that
    only ever wins on the task it was designed for has not been shown to work.

``gpt_finetune``
    Pretrain on WikiText-2 **with AdamW**, then fully fine-tune on
    tinyshakespeare. This reproduces, on a real language model, the setting where
    Qu et al. (ICML 2026) find Muon *loses* to Adam: the pretrained weights carry
    Adam's implicit bias, and a spectral optimizer's larger, more isotropic
    updates disrupt them. It is the regime ASTRO's anchored trust region targets.

Sizing
------
``n_layer=4, n_head=4, n_embd=128, block_size=128`` -- 806K non-embedding
parameters, about 74 ms per optimizer step on four CPU cores. GPT-2 (124M) is
the same architecture at ``n_layer=12, n_head=12, n_embd=768, block_size=1024``.
The reduction is a compute constraint and nothing else; every structural choice
is nanoGPT's. Conclusions therefore transfer only as far as scale-sensitivity
allows, and Wen et al. (2025) specifically found optimizer advantages *shrink*
with scale -- so a win here is evidence, not proof, and is reported as such.

Validation
----------
Loss is measured on a fixed, held-out tail of the corpus, over validation
batches drawn from a generator seeded independently of the run seed. Every
optimizer therefore sees the identical validation set, and run-to-run spread
reflects training noise rather than evaluation noise.
"""

from __future__ import annotations

import math
import time
from functools import lru_cache

import torch

from astro.bench.corpora import get_corpus
from astro.bench.gpt import GPT, GPTConfig
from astro.bench.protocol import OptimizerFactory, TaskResult

__all__ = [
    "gpt_scratch_task",
    "gpt_scratch_mid_task",
    "gpt_shakespeare_task",
    "gpt_shakespeare_cosine_task",
    "gpt_bpe_task",
    "gpt_bpe_cosine_task",
    "gpt_finetune_task",
    "build_gpt",
    "BENCH_GPT",
    "MID_GPT",
]

#: The benchmark model size. GPT-2 architecture, reduced to a CPU budget.
BENCH_GPT = dict(n_layer=4, n_head=4, n_embd=128, block_size=128)

#: A 6x larger configuration, for checking whether a result holds as the model
#: grows. 4.76M non-embedding parameters against BENCH_GPT's 0.81M, at roughly
#: 0.6 s/step against 0.2 -- the largest size at which the full protocol (tuning
#: sweep plus multi-seed evaluation, for several optimizers) completes in hours
#: rather than weeks on four CPU cores. GPT-2 small itself measures 34.4 s/step
#: here, which puts one protocol run at about sixteen days.
MID_GPT = dict(n_layer=6, n_head=8, n_embd=256, block_size=256)


def build_gpt(vocab_size: int, *, seed: int, **overrides: int) -> GPT:
    """Construct the benchmark GPT with a deterministic initialisation."""
    config = GPTConfig(vocab_size=vocab_size, **{**BENCH_GPT, **overrides})
    torch.manual_seed(seed)
    return GPT(config)


@torch.no_grad()
def _validation_loss(model: GPT, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    """Mean cross-entropy over a fixed set of validation batches."""
    was_training = model.training
    model.eval()
    total = 0.0
    for x, y in batches:
        _, loss = model(x, y)
        total += float(loss)
    if was_training:
        model.train()
    return total / len(batches)


def _validation_batches(
    corpus_name: str, *, count: int, batch: int, block: int, seed: int = 777
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Draw a fixed validation set, identical for every optimizer and seed."""
    corpus = get_corpus(corpus_name)
    generator = torch.Generator().manual_seed(seed)
    return [corpus.batch(batch, block, generator, split="val") for _ in range(count)]


def _cosine_factor(step: int, steps: int, warmup: int, floor: float) -> float:
    """nanoGPT's learning-rate schedule, as a multiplier on each group's base LR.

    Linear warmup over ``warmup`` steps, then cosine decay to ``floor`` times the
    base rate. Expressed as a multiplier rather than an absolute rate because the
    matrix optimizers carry several parameter groups at different base rates, and
    the schedule has to preserve the ratios between them.
    """
    if step < warmup:
        return (step + 1) / (warmup + 1)
    ratio = (step - warmup) / max(1, steps - warmup)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return floor + coefficient * (1.0 - floor)


def _train(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    corpus_name: str,
    *,
    steps: int,
    batch: int,
    block: int,
    seed: int,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    eval_every: int,
    target: float,
    grad_clip: float = 1.0,
    train_tokens: int | None = None,
    schedule: str = "none",
    warmup_fraction: float = 0.1,
    lr_floor: float = 0.1,
) -> TaskResult:
    """Shared training loop for both language-model tasks.

    ``grad_clip`` is nanoGPT's default of 1.0 and applies to every optimizer
    equally. It matters for a fair comparison: without it a badly scaled draw
    from one optimizer's search space diverges and scores ``inf``, which flatters
    optimizers whose search space happens to be better centred.

    ``train_tokens`` restricts sampling to the first N tokens of the training
    split. Fine-tuning uses it to create the low-data regime in which Qu et al.
    locate the optimizer-mismatch effect; validation always uses the full
    held-out tail regardless.

    ``schedule="cosine"`` applies nanoGPT's warmup-then-cosine-decay rate to
    every optimizer equally. Equal treatment is not the same as neutral
    treatment: a constant rate penalises optimizers whose stability depends on
    decay more than it penalises ones that normalise their own update size, so
    the schedule is a variable to report under, not a detail to fix by fiat.
    Both settings are therefore kept and both are measured.
    """
    corpus = get_corpus(corpus_name)
    data = corpus.train if train_tokens is None else corpus.train[:train_tokens]
    high = data.numel() - block - 1
    if high <= 0:
        raise ValueError(f"train_tokens={train_tokens} is smaller than block_size={block}")

    generator = torch.Generator().manual_seed(seed + 4242)
    model.train()

    if schedule not in {"none", "cosine"}:
        raise ValueError(f"unknown schedule {schedule!r}")
    base_rates = [group["lr"] for group in optimizer.param_groups]
    warmup = max(1, int(round(warmup_fraction * steps)))

    curve: list[float] = []
    reached: int | None = None
    started = time.perf_counter()
    for step in range(steps):
        if schedule == "cosine":
            factor = _cosine_factor(step, steps, warmup, lr_floor)
            for group, base in zip(optimizer.param_groups, base_rates, strict=True):
                group["lr"] = base * factor

        starts = torch.randint(0, high, (batch,), generator=generator)
        x = torch.stack([data[s : s + block] for s in starts])
        y = torch.stack([data[s + 1 : s + 1 + block] for s in starts])
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step % eval_every == 0 or step == steps - 1:
            value = _validation_loss(model, val_batches)
            curve.append(value)
            if reached is None and value <= target:
                reached = step
    return TaskResult(
        final=curve[-1],
        curve=curve,
        steps=steps,
        seconds=time.perf_counter() - started,
        reached=reached,
    )


def gpt_scratch_task(
    factory: OptimizerFactory,
    seed: int,
    *,
    steps: int = 400,
    batch: int = 16,
    eval_every: int = 25,
    val_count: int = 6,
    target: float = 2.0,
    shape: dict[str, int] | None = None,
    corpus_name: str = "wikitext2",
    schedule: str = "none",
) -> TaskResult:
    """Train GPT-2 from scratch on ``corpus_name``, character level.

    The control task: from-scratch pretraining is where matrix-structured
    optimizers are documented to beat AdamW, so it checks that the routing and
    spectral machinery work at all before the fine-tuning claim is tested.
    """
    shape = shape or BENCH_GPT
    corpus = get_corpus(corpus_name)
    block = shape["block_size"]
    val_batches = _validation_batches(corpus_name, count=val_count, batch=batch, block=block)

    model = build_gpt(corpus.vocab_size, seed=seed, **shape)
    return _train(
        model,
        factory(model),
        corpus_name,
        steps=steps,
        batch=batch,
        block=block,
        seed=seed,
        val_batches=val_batches,
        eval_every=eval_every,
        target=target,
        schedule=schedule,
    )


@lru_cache(maxsize=8)
def _adamw_pretrained_gpt(
    seed: int, steps: int, batch: int, lr: float
) -> dict[str, torch.Tensor]:
    """Pretrain the benchmark GPT on WikiText-2 with AdamW and cache the weights.

    Cached because every optimizer under test must fine-tune from *bit-identical*
    weights; otherwise the comparison measures pretraining noise. AdamW
    specifically, because the mismatch being reproduced is between Adam's
    implicit bias in the pretrained weights and a different optimizer's updates
    during fine-tuning -- pretraining with anything else would not pose the
    question.
    """
    corpus = get_corpus("wikitext2")
    block = BENCH_GPT["block_size"]
    model = build_gpt(corpus.vocab_size, seed=seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-1, betas=(0.9, 0.95)
    )
    generator = torch.Generator().manual_seed(seed + 31337)
    model.train()
    for _ in range(steps):
        x, y = corpus.batch(batch, block, generator)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def gpt_finetune_task(
    factory: OptimizerFactory,
    seed: int,
    *,
    pretrain_steps: int = 1500,
    pretrain_lr: float = 3e-3,
    steps: int = 120,
    batch: int = 16,
    eval_every: int = 10,
    val_count: int = 6,
    train_tokens: int = 40_000,
    target: float = 1.6,
) -> TaskResult:
    """Fully fine-tune an AdamW-pretrained GPT-2 on tinyshakespeare.

    The flagship task. WikiText-2 (encyclopedic prose) to tinyshakespeare (Early
    Modern verse and dialogue) is a large stylistic shift over a shared
    alphabet, so the pretrained representation is useful but not sufficient --
    which is the condition under which destroying it has a measurable cost.

    Calibration matters here and was measured, not assumed. Under matched AdamW
    at ``lr=1e-3``, this configuration reaches **2.146** validation loss from the
    pretrained checkpoint against **2.622** from a random initialisation -- a
    0.48-nat gap. Without such a gap the task cannot detect feature disruption at
    all, because an optimizer could destroy every pretrained feature and still
    score well by relearning from the fine-tuning corpus. Widening it is what the
    40k-token pool buys: at the full 1M-token corpus and 150 steps the same gap
    is only 0.23 nats, because there is enough data to relearn from scratch.
    120 steps at batch 16 x block 128 is roughly six epochs over the pool, which
    is the low-data regime where Qu et al. find update strength does the damage.
    """
    corpus = get_corpus("shakespeare")
    block = BENCH_GPT["block_size"]
    val_batches = _validation_batches("shakespeare", count=val_count, batch=batch, block=block)

    state = _adamw_pretrained_gpt(seed, pretrain_steps, batch, pretrain_lr)
    model = build_gpt(corpus.vocab_size, seed=seed)
    model.load_state_dict(state)

    return _train(
        model,
        factory(model),
        "shakespeare",
        steps=steps,
        batch=batch,
        block=block,
        seed=seed,
        val_batches=val_batches,
        eval_every=eval_every,
        target=target,
        train_tokens=train_tokens,
    )


def gpt_scratch_mid_task(
    factory: OptimizerFactory, seed: int, *, batch: int = 8, **kwargs: object
) -> TaskResult:
    """``gpt_scratch`` at :data:`MID_GPT` -- 4.76M non-embedding parameters.

    A scale check, not a second headline. If a result obtained at 806K parameters
    is an artefact of the model being tiny, it should weaken or reverse here; the
    literature's own finding is that optimizer advantages shrink as models grow
    [Wen et al. 2025], so holding at 6x is evidence and growing would be
    surprising.

    Batch is halved to 8 so that a step stays affordable; the comparison is
    internally consistent because every optimizer runs under the identical
    setting.
    """
    return gpt_scratch_task(factory, seed, batch=batch, shape=MID_GPT, **kwargs)  # type: ignore[arg-type]


def gpt_shakespeare_task(
    factory: OptimizerFactory,
    seed: int,
    *,
    steps: int = 400,
    batch: int = 16,
    target: float = 1.6,
    **kwargs: object,
) -> TaskResult:
    """Train a small GPT-2 from scratch on tinyshakespeare.

    A deliberately small, deliberately easy setting: 806K non-embedding
    parameters on a 1M-character corpus, which is the configuration nanoGPT ships
    as its introductory example. It is the cheapest task in the suite that is
    still a real language model on real text, which makes it the one where a
    large seed count is affordable -- and seed count is what the exact
    signed-rank test needs, since with :math:`n` seeds the smallest attainable
    two-sided p-value is :math:`2/2^{n}`.

    Validation is on the held-out tail of the same corpus, so this measures
    generalisation within a distribution rather than transfer across one.
    """
    return gpt_scratch_task(
        factory, seed, steps=steps, batch=batch, target=target,
        corpus_name="shakespeare", **kwargs,  # type: ignore[arg-type]
    )


def gpt_bpe_task(
    factory: OptimizerFactory,
    seed: int,
    *,
    steps: int = 400,
    batch: int = 16,
    target: float = 5.0,
    **kwargs: object,
) -> TaskResult:
    """WikiText-2 from scratch under a locally-trained byte-level BPE vocabulary.

    The character-level tasks put 3.5% of the model's parameters on the
    elementwise path against GPT-2 small's 31.7%, because the embedding and the
    tied head scale with the vocabulary and nothing else does. A matrix
    optimizer is responsible for the *other* path, so that ratio decides how much
    of the model it can affect at all -- and a benchmark that gets it wrong by
    10x cannot predict which optimizer wins at a scale where it is right.

    ``BPE_VOCAB_SIZE`` is picked to reproduce GPT-2 small's ratio at this width:
    the measured split here is 32.4% against 31.7%. Everything else -- model,
    step budget, batch, schedule -- is unchanged from ``gpt_scratch``, so the
    difference between the two tasks is tokenisation alone.

    The target is 5.0 rather than 2.0: a 2816-token vocabulary has a much higher
    entropy floor per token than a 97-character one, so the losses are not
    comparable across the two tasks and were never meant to be.
    """
    return gpt_scratch_task(
        factory, seed, steps=steps, batch=batch, target=target,
        corpus_name="wikitext2-bpe", **kwargs,  # type: ignore[arg-type]
    )


def gpt_bpe_cosine_task(
    factory: OptimizerFactory, seed: int, *, steps: int = 400, **kwargs: object
) -> TaskResult:
    """:func:`gpt_bpe_task` under nanoGPT's warmup-plus-cosine schedule."""
    return gpt_bpe_task(factory, seed, steps=steps, schedule="cosine", **kwargs)  # type: ignore[arg-type]


def gpt_shakespeare_cosine_task(
    factory: OptimizerFactory, seed: int, *, steps: int = 400, **kwargs: object
) -> TaskResult:
    """:func:`gpt_shakespeare_task` under nanoGPT's warmup-plus-cosine schedule.

    A constant learning rate is the harsher setting for AdamW, whose stability
    late in training depends on decay, than for an optimizer that normalises its
    own update size -- so a win measured without a schedule could be a win over a
    handicap rather than over AdamW. This task removes that objection by giving
    every optimizer the schedule nanoGPT ships with, and the claim is reported
    under both.

    ``steps`` is declared explicitly rather than left to ``**kwargs`` because
    ``scripts/time_matched.py`` reads the default off this signature to size its
    wall-clock budget. Under a stretched budget the schedule stretches with it,
    which is the intended behaviour: an optimizer given more steps should decay
    over all of them, not finish decaying early and then coast.
    """
    return gpt_shakespeare_task(factory, seed, steps=steps, schedule="cosine", **kwargs)  # type: ignore[arg-type]
