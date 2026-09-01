"""Benchmark tasks. Every one runs on CPU in seconds and downloads nothing.

The suite is built around a single question: *does the proposed optimizer help in
the regime this repository actually trains in, and does it stop helping
elsewhere?* So it deliberately includes tasks ASTRO is not expected to win.

``quadratic``
    Least squares with a controllable input covariance. Muon's update is the
    steepest-descent direction under the spectral norm *when the layer inputs are
    isotropic*; Newton-Muon shows the correct direction is otherwise
    ``msgn(G (ZZ^T)^-1)``. Sweeping the conditioning of ``ZZ^T`` measures how much
    that assumption costs, against an analytically known optimum.

``mlp`` and ``convnet``
    From-scratch training. The convnet carries the shape zoo that makes routing
    matter: stem, depthwise, dense conv, linear, norms.

``finetune_mismatch``
    The flagship. Pretrain with AdamW, then fully fine-tune on a shifted
    distribution. This is the setting where Qu et al. (ICML 2026) find Muon
    *loses* to Adam. The pretrained checkpoint is computed once and cached, so
    every optimizer fine-tunes from bit-identical weights.

``char_transformer``
    From-scratch attention on a procedurally generated formal language. Guards
    against a method that merely trades the pretraining regime for the
    fine-tuning one.

``gpt_scratch`` and ``gpt_finetune``
    The same two questions asked of a real language model: nanoGPT's GPT-2 on
    WikiText-2 and tinyshakespeare. Defined in :mod:`astro.bench.llm`; they are
    the only tasks needing a corpus on disk, fetched once by
    ``scripts/fetch_llm_data.py``.
"""

from __future__ import annotations

import math
import time
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F

from astro.bench.llm import (
    gpt_bpe_cosine_task,
    gpt_bpe_task,
    gpt_finetune_task,
    gpt_scratch_mid_task,
    gpt_scratch_task,
    gpt_shakespeare_cosine_task,
    gpt_shakespeare_task,
)
from astro.bench.protocol import OptimizerFactory, TaskResult

__all__ = [
    "gpt_finetune_task",
    "gpt_scratch_task",
    "gpt_scratch_mid_task",
    "gpt_shakespeare_task",
    "gpt_shakespeare_cosine_task",
    "gpt_bpe_task",
    "gpt_bpe_cosine_task",
    "quadratic_task",
    "mlp_task",
    "convnet_task",
    "finetune_mismatch_task",
    "char_transformer_task",
    "TASKS",
]


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def _blob_images(
    n: int,
    classes: int,
    size: int,
    generator: torch.Generator,
    *,
    scale: float = 1.0,
    contrast: float = 1.0,
    noise: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render images whose class is encoded in the layout of Gaussian blobs.

    Structured enough that convolutions beat a linear model, and cheap enough to
    regenerate per step. ``scale``, ``contrast`` and ``noise`` parameterise a
    domain shift without changing the label semantics.
    """
    labels = torch.randint(0, classes, (n,), generator=generator)
    coords = torch.linspace(-1.0, 1.0, size)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")

    # Fixed per-class blob layout: deterministic in the class index, so the label
    # is genuinely recoverable from the image.
    layout_gen = torch.Generator().manual_seed(1234)
    centres = torch.rand(classes, 3, 2, generator=layout_gen) * 1.4 - 0.7
    signs = torch.where(torch.rand(classes, 3, generator=layout_gen) > 0.5, 1.0, -1.0)

    images = torch.zeros(n, 3, size, size)
    jitter = (torch.rand(n, 3, 2, generator=generator) - 0.5) * 0.25
    for blob in range(3):
        cx = centres[labels, blob, 0] + jitter[:, blob, 0]
        cy = centres[labels, blob, 1] + jitter[:, blob, 1]
        width = 0.16 * scale
        field = torch.exp(
            -(
                (grid_x[None] - cx[:, None, None]) ** 2
                + (grid_y[None] - cy[:, None, None]) ** 2
            )
            / (2 * width**2)
        )
        images[:, blob] += contrast * signs[labels, blob][:, None, None] * field

    images += noise * torch.randn(images.shape, generator=generator)
    return images, labels


class _SmallCNN(nn.Module):
    """Stem, depthwise, dense conv, linear, head: the routing-relevant shapes."""

    def __init__(self, classes: int, width: int = 24) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, width, 3, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(4, width)
        self.depthwise = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.pointwise = nn.Conv2d(width, width * 2, 1)
        self.norm2 = nn.GroupNorm(4, width * 2)
        self.dense = nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1)
        self.proj = nn.Linear(width * 2, width * 2)
        self.head = nn.Linear(width * 2, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.norm1(self.stem(x)))
        x = F.gelu(self.pointwise(self.depthwise(x)))
        x = F.gelu(self.norm2(x))
        x = F.gelu(self.dense(x)).mean(dim=(2, 3))
        return self.head(F.gelu(self.proj(x)))


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def quadratic_task(
    factory: OptimizerFactory,
    seed: int,
    *,
    rows: int = 64,
    cols: int = 64,
    samples: int = 256,
    condition: float = 100.0,
    steps: int = 120,
    target: float = 1e-3,
) -> TaskResult:
    """Least squares ``min_W 0.5 ||WZ - W*Z||^2`` with input condition number ``condition``.

    The optimum is known exactly, so ``final`` is true suboptimality rather than a
    proxy. ``condition = 1`` gives the isotropic inputs Muon's derivation assumes;
    larger values measure the cost of that assumption being wrong.
    """
    generator = torch.Generator().manual_seed(seed)
    spectrum = torch.logspace(0, -math.log10(condition), cols, base=10.0)
    basis, _ = torch.linalg.qr(torch.randn(cols, cols, generator=generator))
    inputs = (basis * spectrum.sqrt()) @ torch.randn(cols, samples, generator=generator)

    target_weight = torch.randn(rows, cols, generator=generator) / math.sqrt(cols)
    targets = target_weight @ inputs

    weight = nn.Parameter(torch.zeros(rows, cols))
    model = nn.Module()
    model.weight = weight  # type: ignore[assignment]
    optimizer = factory(model)

    curve, reached = [], None
    started = time.perf_counter()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = 0.5 * ((weight @ inputs - targets) ** 2).mean()
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        curve.append(value)
        if reached is None and value <= target:
            reached = step
    return TaskResult(
        final=curve[-1], curve=curve, steps=steps, seconds=time.perf_counter() - started,
        reached=reached,
    )


def mlp_task(
    factory: OptimizerFactory,
    seed: int,
    *,
    width: int = 96,
    depth: int = 3,
    classes: int = 6,
    features: int = 64,
    samples: int = 512,
    steps: int = 150,
    batch: int = 64,
    target: float = 0.35,
) -> TaskResult:
    """From-scratch MLP on an anisotropic synthetic classification problem."""
    generator = torch.Generator().manual_seed(seed)
    scale = torch.logspace(0, -1.5, features, base=10.0)
    x = torch.randn(samples, features, generator=generator) * scale
    teacher = torch.randn(features, classes, generator=generator)
    y = (x @ teacher + 0.3 * torch.randn(samples, classes, generator=generator)).argmax(dim=1)
    x_val = torch.randn(samples, features, generator=generator) * scale
    y_val = (x_val @ teacher).argmax(dim=1)

    torch.manual_seed(seed)
    layers: list[nn.Module] = [nn.Linear(features, width), nn.GELU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.LayerNorm(width), nn.GELU()]
    layers.append(nn.Linear(width, classes))
    model = nn.Sequential(*layers)
    optimizer = factory(model)

    curve, reached = [], None
    started = time.perf_counter()
    for step in range(steps):
        index = torch.randint(0, samples, (batch,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(x[index]), y[index]).backward()
        optimizer.step()
        if step % 5 == 0 or step == steps - 1:
            with torch.no_grad():
                value = float(F.cross_entropy(model(x_val), y_val))
            curve.append(value)
            if reached is None and value <= target:
                reached = step
    return TaskResult(
        final=curve[-1], curve=curve, steps=steps, seconds=time.perf_counter() - started,
        reached=reached,
    )


def convnet_task(
    factory: OptimizerFactory,
    seed: int,
    *,
    classes: int = 10,
    size: int = 24,
    train_n: int = 256,
    val_n: int = 256,
    steps: int = 120,
    batch: int = 48,
    target: float = 1.2,
) -> TaskResult:
    """From-scratch CNN carrying the stem / depthwise / dense-conv shape zoo.

    Noise and class count are set so a well-tuned AdamW lands well short of zero
    loss. A task the baseline solves outright cannot rank anything.
    """
    generator = torch.Generator().manual_seed(seed)
    x, y = _blob_images(train_n, classes, size, generator, noise=0.6)
    x_val, y_val = _blob_images(val_n, classes, size, generator, noise=0.6)

    torch.manual_seed(seed)
    model = _SmallCNN(classes)
    optimizer = factory(model)

    curve, reached = [], None
    started = time.perf_counter()
    for step in range(steps):
        index = torch.randint(0, train_n, (batch,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(x[index]), y[index]).backward()
        optimizer.step()
        if step % 5 == 0 or step == steps - 1:
            with torch.no_grad():
                value = float(F.cross_entropy(model(x_val), y_val))
            curve.append(value)
            if reached is None and value <= target:
                reached = step
    return TaskResult(
        final=curve[-1], curve=curve, steps=steps, seconds=time.perf_counter() - started,
        reached=reached,
    )


@lru_cache(maxsize=8)
def _pretrained_cnn(seed: int, classes: int, size: int, steps: int) -> dict[str, torch.Tensor]:
    """Pretrain a CNN with AdamW and cache the weights.

    Cached because every optimizer under test must fine-tune from *bit-identical*
    pretrained weights -- otherwise the comparison measures pretraining noise.
    AdamW specifically, because reproducing the Adam-pretrained/Muon-fine-tuned
    mismatch is the entire point of the task.
    """
    generator = torch.Generator().manual_seed(seed)
    x, y = _blob_images(768, classes, size, generator, contrast=1.0, noise=0.25)

    torch.manual_seed(seed)
    model = _SmallCNN(classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    for _ in range(steps):
        index = torch.randint(0, x.size(0), (64,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(x[index]), y[index]).backward()
        optimizer.step()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def finetune_mismatch_task(
    factory: OptimizerFactory,
    seed: int,
    *,
    classes: int = 10,
    size: int = 24,
    pretrain_steps: int = 400,
    train_n: int = 64,
    val_n: int = 256,
    steps: int = 60,
    batch: int = 16,
    target: float = 1.2,
) -> TaskResult:
    """Fully fine-tune an AdamW-pretrained CNN on a shifted, small dataset.

    The shift changes blob scale, contrast and noise but not the label rule, so
    the pretrained features remain useful and destroying them is a real cost.

    Difficulty is calibrated so that the pretrained initialisation genuinely
    matters: under AdamW this configuration reaches ~1.30 validation loss from
    the pretrained checkpoint versus ~2.35 from a random initialisation. Without
    that gap the task cannot detect feature disruption at all -- an optimizer
    could destroy every pretrained feature and still score well by relearning
    from the fine-tuning set. The 64-example, 60-step, batch-16 budget is the
    regime where Qu et al. find update strength does the damage, and is close to
    what the target workload trains in.
    """
    state = _pretrained_cnn(seed, classes, size, pretrain_steps)

    generator = torch.Generator().manual_seed(seed + 9000)
    shift = dict(scale=1.45, contrast=0.5, noise=0.7)
    x, y = _blob_images(train_n, classes, size, generator, **shift)
    x_val, y_val = _blob_images(val_n, classes, size, generator, **shift)

    torch.manual_seed(seed)
    model = _SmallCNN(classes)
    model.load_state_dict(state)
    optimizer = factory(model)

    curve, reached = [], None
    started = time.perf_counter()
    for step in range(steps):
        index = torch.randint(0, train_n, (batch,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(x[index]), y[index]).backward()
        optimizer.step()
        if step % 3 == 0 or step == steps - 1:
            with torch.no_grad():
                value = float(F.cross_entropy(model(x_val), y_val))
            curve.append(value)
            if reached is None and value <= target:
                reached = step
    return TaskResult(
        final=curve[-1], curve=curve, steps=steps, seconds=time.perf_counter() - started,
        reached=reached,
    )


class _TinyTransformer(nn.Module):
    """Minimal pre-norm decoder: embeddings, attention projections, MLP, head."""

    def __init__(self, vocab: int, width: int = 64, heads: int = 4, layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.position = nn.Parameter(torch.zeros(1, 64, width))
        self.blocks = nn.ModuleList(
            nn.ModuleDict(
                {
                    "norm1": nn.LayerNorm(width),
                    "attn": nn.MultiheadAttention(width, heads, batch_first=True),
                    "norm2": nn.LayerNorm(width),
                    "fc1": nn.Linear(width, width * 2),
                    "fc2": nn.Linear(width * 2, width),
                }
            )
            for _ in range(layers)
        )
        self.norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.size(1)
        x = self.embed(tokens) + self.position[:, :length]
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
        for block in self.blocks:
            h = block["norm1"](x)
            x = x + block["attn"](h, h, h, attn_mask=mask, need_weights=False)[0]
            h = block["norm2"](x)
            x = x + block["fc2"](F.gelu(block["fc1"](h)))
        return self.classifier(self.norm(x))


def char_transformer_task(
    factory: OptimizerFactory,
    seed: int,
    *,
    vocab: int = 24,
    length: int = 48,
    samples: int = 256,
    steps: int = 120,
    batch: int = 16,
    target: float = 1.5,
) -> TaskResult:
    """Next-token prediction on a procedurally generated regular language.

    Sequences are drawn from a fixed random Markov chain, so the entropy floor is
    known to be positive and a model cannot trivially reach zero loss.
    """
    generator = torch.Generator().manual_seed(seed)
    transitions = torch.rand(vocab, vocab, generator=generator).pow(4.0)
    transitions = transitions / transitions.sum(dim=1, keepdim=True)

    tokens = torch.zeros(samples, length + 1, dtype=torch.long)
    tokens[:, 0] = torch.randint(0, vocab, (samples,), generator=generator)
    for position in range(length):
        probabilities = transitions[tokens[:, position]]
        tokens[:, position + 1] = torch.multinomial(
            probabilities, 1, generator=generator
        ).squeeze(1)

    torch.manual_seed(seed)
    model = _TinyTransformer(vocab)
    optimizer = factory(model)

    curve, reached = [], None
    started = time.perf_counter()
    for step in range(steps):
        index = torch.randint(0, samples, (batch,), generator=generator)
        chunk = tokens[index]
        optimizer.zero_grad(set_to_none=True)
        logits = model(chunk[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, vocab), chunk[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        if step % 5 == 0 or step == steps - 1:
            value = float(loss.detach())
            curve.append(value)
            if reached is None and value <= target:
                reached = step
    return TaskResult(
        final=curve[-1], curve=curve, steps=steps, seconds=time.perf_counter() - started,
        reached=reached,
    )


#: Task registry consumed by ``astro.bench.run``.
#:
#: The ``gpt_*`` tasks train nanoGPT's GPT-2 on real text and are imported
#: lazily-ish here rather than defined above, because they are the only tasks
#: that need a corpus on disk (``scripts/fetch_llm_data.py``). Everything else in
#: this module is self-contained and downloads nothing.
TASKS = {
    "quadratic": quadratic_task,
    "mlp": mlp_task,
    "convnet": convnet_task,
    "finetune": finetune_mismatch_task,
    "transformer": char_transformer_task,
    "gpt_scratch": gpt_scratch_task,
    "gpt_scratch_mid": gpt_scratch_mid_task,
    "gpt_shakespeare": gpt_shakespeare_task,
    "gpt_shakespeare_cosine": gpt_shakespeare_cosine_task,
    "gpt_bpe": gpt_bpe_task,
    "gpt_bpe_cosine": gpt_bpe_cosine_task,
    "gpt_finetune": gpt_finetune_task,
}
