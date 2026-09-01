#!/usr/bin/env python3
"""The decisive comparison, sized for a Colab T4: ASTRO against Muon, NorMuon, AdamW.

    !pip -q install transformers datasets
    !python colab_bench.py --steps 300 --trials 8 --seeds 3

Everything in this project so far was measured on four CPU cores at 806K
parameters with a 97-character vocabulary. That setting cannot answer the
question it was built for, for a reason that is structural rather than a matter
of degree: embeddings and the tied head scale with the vocabulary and take the
elementwise path, everything else takes the spectral path and does not. At
vocab 97 and width 128 that puts 3.5% of the model on the elementwise path;
GPT-2 small puts 31.7% there. A matrix optimizer is responsible for the other
path, so the CPU benchmark hands it 96% of the model where the real target hands
it 68%.

This script runs the same protocol at the scale and tokenisation the optimizer
literature actually uses: nanoGPT-shaped GPT-2 124M, GPT-2 BPE, FineWeb-Edu.

Protocol
--------
Enforced rather than intended, because the standard way to manufacture an
optimizer result is to sweep the proposal harder than the baseline:

* every optimizer tunes the **same number** of hyperparameters, with the same
  trial count drawn from the same RNG stream, and the script refuses to run if
  the counts differ;
* learning-rate ranges follow each optimizer's **update scale** -- a Muon-scaled
  update needs a range about 10x higher than an Adam-scaled one, and pairing the
  wrong range with the wrong scale is a silent handicap this project has already
  shipped twice;
* tuning happens on one seed, evaluation on disjoint seeds;
* results are paired seed-by-seed and reported with an exact sign test, wall
  clock alongside loss, and no selection after the fact.

Interruptions
-------------
A Colab session drops. Every trial and every evaluation is appended to
``astro_bench_state.json`` as it completes, and re-running the identical command
resumes from that file rather than repeating work. Delete it to start over.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import statistics
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Vendored optimizers
#
# Copied verbatim from scripts/colab_probe.py so this file runs alone. Splitting
# them cost a user a ModuleNotFoundError when Colab renamed the second upload to
# "colab_probe (1).py", and a script that runs on someone else's machine once
# should not have an import that can fail. tests/test_colab_bench.py asserts
# this copy still produces the same update as astro.optimizer, so drift is
# caught rather than shipped.
# ---------------------------------------------------------------------------

#: Solved per-step coefficients for a *converging* polar iteration.
#:
#: Muon's quintic is one polynomial applied k times and cannot reach sigma = 1
#: at any budget: its fixed points solve 2.4445 - 4.7750 s^2 + 2.0315 s^4 = 0,
#: at 0.868 and 1.264. A different polynomial per step has no such obstruction.
#: Greedily solved, so a shorter schedule is a prefix of a longer one. The last
#: two entries land near (1.875, -1.25, 0.375), independently reproducing the
#: asymptote the Polar Express publishes.
POLAR_SCHEDULE: tuple[tuple[float, float, float], ...] = (
    (5.741408, -17.016317, 12.623472),
    (4.240444, -6.859093, 2.787935),
    (4.186216, -6.613335, 2.669455),
    (3.958440, -5.645446, 2.206946),
    (2.621392, -2.503740, 0.833594),
    (1.889525, -1.266059, 0.376621),
    (1.777582, -1.055164, 0.277581),
)


def polar_iterate(matrix: torch.Tensor, steps: int = 5, converging: bool = False):
    """Polar factor by Newton-Schulz, optionally with the converging schedule.

    Always float32: a T4 is sm_75 with no bfloat16, and float16 here degrades
    exactly the small singular values the measurement is about.
    """
    x = matrix.float()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    schedule = (
        POLAR_SCHEDULE[:steps] if converging else ((3.4445, -4.7750, 2.0315),) * steps
    )
    for a, b, c in schedule:
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.T if transposed else x

def newton_schulz(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Muon's quintic iteration for the polar factor ``msgn(A) = U V^T``.

    Coefficients are Jordan's (3.4445, -4.7750, 2.0315). Always computed in
    float32: on a T4 there is no bfloat16, and float16 here degrades the small
    singular values silently, which is precisely the regime the measurement is
    about.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix.float()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        gram = x @ x.T
        correction = b * gram + c * (gram @ gram)
        x = a * x + correction @ x
    return x.T if transposed else x

def cautious_mask(update: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """Liang et al.'s mask: zero the coordinates that disagree with the gradient."""
    mask = (update * grad > 0).to(update.dtype)
    return update * mask * (mask.numel() / mask.sum().clamp_min(1e-12))


def apply_weight_decay(param, update, rate, *, cautious):
    """Cautious weight decay (Chen et al.): decay only where the step already
    carries the weight toward zero."""
    if rate == 0.0:
        return
    if not cautious:
        param.mul_(1.0 - rate)
        return
    param.add_(torch.where(update * param > 0, param, torch.zeros_like(param)), alpha=-rate)

class Astro(torch.optim.Optimizer):
    """ASTRO's shipped recipe, inlined so the script runs from a bare Colab.

    Faithful to ``astro.optimizer.Astro``'s defaults at every setting that
    matters here: Nesterov momentum at 0.95, the neuron-wise second moment
    applied *after* orthogonalisation and rescaled to preserve the update norm,
    Muon's aspect-ratio update scale applied per block, the cautious mask, the
    QKV split, and cautious weight decay.

    An earlier version of this file inlined the *previous* defaults --
    pre-orthogonalisation variance and Adam-style scaling -- which is the
    configuration the 124M run measured and which we already knew was 0.041
    worse on the CPU benchmark. Keeping this in step with the library is the
    whole point of the class, so it is asserted against it in
    ``tests/test_colab_probe.py``.
    """

    def __init__(self, groups, lr=1e-3, scalar_lr_mult=0.1, betas=(0.95, 0.95),
                 eps=1e-8, weight_decay=0.01, variance_power=1.0,
                 post_normalize=False, cautious_wd=True, converging=False,
                 equilibrate=False, ns_steps=5, cautious=True):
        # The scalar path needs its own rate. A Muon-scaled spectral update is
        # 4.4x smaller in Frobenius norm than an Adam-like one at 768 width, and
        # the gap grows with width, so running embeddings and the tied head at
        # the spectral rate overshoots them badly. Muon's reference carries a
        # separate adamw_lr for exactly this reason; an earlier revision of this
        # file gave every group one rate, which handed the baseline an advantage
        # over 31.7% of a GPT-2's parameters.
        for group in groups:
            group.setdefault("lr", lr if group.get("spectral") else lr * scalar_lr_mult)
        super().__init__(groups, dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            variance_power=variance_power, post_normalize=post_normalize,
            cautious_wd=cautious_wd, converging=converging,
            equilibrate=equilibrate, ns_steps=ns_steps, cautious=cautious,
        ))

    @torch.no_grad()
    def step(self):  # type: ignore[override]
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                # HuggingFace GPT-2 stores every projection as a ``Conv1D``,
                # whose weight is (in, out) -- the transpose of the operator. Row
                # statistics, the RMS scale and the QKV split are all defined on
                # the operator, so the whole spectral path runs on a transposed
                # view and the result is transposed back before the write.
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
                    # Nesterov: the momentum the buffer *would* have after
                    # absorbing this gradient, which is Muon's default.
                    lookahead = operator.lerp(momentum, beta1)
                    if group["equilibrate"]:
                        # MuonEq's R variant: equalise the momentum's row norms
                        # before the polar step, rescaled to preserve its norm.
                        norms = lookahead.norm(dim=1, keepdim=True).clamp_min(group["eps"])
                        balanced = lookahead / norms
                        lookahead = balanced * (
                            lookahead.norm() / balanced.norm().clamp_min(group["eps"])
                        )
                    sizes = group.get("blocks") or (operator.size(0),)
                    pieces = []
                    for chunk in lookahead.split(list(sizes), 0):
                        filtered = polar_iterate(
                            chunk, group["ns_steps"], group["converging"]
                        )
                        if group["post_normalize"]:
                            target = math.sqrt(min(chunk.size(0), chunk.size(1)))
                            filtered = filtered * (target / filtered.norm().clamp_min(1e-12))
                        pieces.append(filtered)
                    direction = torch.cat(pieces, dim=0)

                    # Second moment of the *orthogonalised* update, not of the
                    # gradient: the polar step exists to remove the gradient's
                    # ill-conditioning, so accumulating on the raw gradient
                    # measures the wrong quantity.
                    variance = state["variance"]
                    variance.mul_(beta2).add_(direction.pow(2).mean(dim=1), alpha=1 - beta2)
                    moment = variance / (1 - beta2**step)
                    denominator = moment.pow(0.5 * group["variance_power"]).add_(group["eps"])
                    scaled = direction / denominator.unsqueeze(1)
                    # Norm-preserving: this sets the distribution of step length
                    # across neurons, not the total step size.
                    direction = scaled * (direction.norm() / scaled.norm().clamp_min(1e-12))

                    # Muon's aspect-ratio scale, per block: under grouped-query
                    # attention the blocks have different row counts.
                    offset, blocks = 0, []
                    for rows in sizes:
                        piece = direction[offset : offset + rows]
                        blocks.append(piece * max(1.0, rows / direction.size(1)) ** 0.5)
                        offset += rows
                    direction = torch.cat(blocks, dim=0).to(param.dtype)
                else:
                    variance = state["variance"]
                    variance.mul_(beta2).addcmul_(operator, operator, value=1 - beta2)
                    denominator = (variance / (1 - beta2**step)).sqrt().add_(group["eps"])
                    direction = (momentum / (1 - beta1**step)) / denominator

                if group.get("transposed"):
                    direction = direction.T
                if group["cautious"]:
                    direction = cautious_mask(direction, grad)
                apply_weight_decay(
                    param, direction, group["lr"] * group["weight_decay"],
                    cautious=group["cautious_wd"],
                )
                param.add_(direction, alpha=-group["lr"])

class Muon(torch.optim.Optimizer):
    """Muon with Nesterov momentum and the ``max(1, m/n)^0.5`` scale, AdamW elsewhere."""

    def __init__(self, groups, lr=0.02, adamw_lr=3e-4, momentum=0.95,
                 betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01):
        super().__init__(groups, dict(lr=lr, adamw_lr=adamw_lr, momentum=momentum,
                                      betas=betas, eps=eps, weight_decay=weight_decay))

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
                    if not group["spectral"]:
                        state["variance"] = torch.zeros_like(operator)
                state["step"] += 1

                if group["spectral"]:
                    buffer = state["momentum"].mul_(group["momentum"]).add_(operator)
                    nesterov = operator.add(buffer, alpha=group["momentum"])
                    direction = newton_schulz(nesterov).to(param.dtype)
                    direction = direction * math.sqrt(
                        max(1.0, operator.size(0) / operator.size(1))
                    )
                    if group.get("transposed"):
                        direction = direction.T
                    param.mul_(1 - group["lr"] * group["weight_decay"])
                    param.add_(direction, alpha=-group["lr"])
                else:
                    beta1, beta2 = group["betas"]
                    step = state["step"]
                    momentum = state["momentum"].mul_(beta1).add_(grad, alpha=1 - beta1)
                    variance = state["variance"]
                    variance.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    update = (momentum / (1 - beta1**step)) / (
                        (variance / (1 - beta2**step)).sqrt().add_(group["eps"])
                    )
                    param.mul_(1 - group["adamw_lr"] * group["weight_decay"])
                    param.add_(update, alpha=-group["adamw_lr"])

class NorMuon(Muon):
    """Muon plus a neuron-wise second moment applied *after* orthogonalisation.

    Li et al. (arXiv:2510.05491). Muon equalises singular values, which is a
    statement about directions and says nothing about how far any individual
    output neuron moves; NorMuon adds that. The moment is accumulated on the
    orthogonalised update rather than on the gradient, because the polar step
    exists precisely to remove the gradient's conditioning, and the rescale is
    norm-preserving so the rule changes the distribution of step length across
    neurons without changing the total step.
    """

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
                    if group["spectral"]:
                        state["row"] = torch.zeros(operator.size(0), device=param.device)
                    else:
                        state["variance"] = torch.zeros_like(operator)
                state["step"] += 1
                step = state["step"]

                if group["spectral"]:
                    buffer = state["momentum"].mul_(group["momentum"]).add_(operator)
                    nesterov = operator.add(buffer, alpha=group["momentum"])
                    direction = newton_schulz(nesterov)

                    beta2 = group["betas"][1]
                    row = state["row"]
                    row.mul_(beta2).add_(direction.pow(2).mean(dim=1), alpha=1 - beta2)
                    denominator = (row / (1 - beta2**step)).sqrt().add_(group["eps"])
                    scaled = direction / denominator.unsqueeze(1)
                    direction = scaled * (direction.norm() / scaled.norm().clamp_min(1e-12))

                    direction = direction.to(param.dtype) * max(
                        1.0, operator.size(0) / operator.size(1)
                    ) ** 0.5
                    if group.get("transposed"):
                        direction = direction.T
                    param.mul_(1 - group["lr"] * group["weight_decay"])
                    param.add_(direction, alpha=-group["lr"])
                else:
                    beta1, beta2 = group["betas"]
                    momentum = state["momentum"].mul_(beta1).add_(operator, alpha=1 - beta1)
                    variance = state["variance"]
                    variance.mul_(beta2).addcmul_(operator, operator, value=1 - beta2)
                    update = (momentum / (1 - beta1**step)) / (
                        (variance / (1 - beta2**step)).sqrt().add_(group["eps"])
                    )
                    if group.get("transposed"):
                        update = update.T
                    param.mul_(1 - group["adamw_lr"] * group["weight_decay"])
                    param.add_(update, alpha=-group["adamw_lr"])

def build_groups(model, optimizer_name: str, config) -> list[dict]:
    """Route parameters by module identity, not by name.

    Embeddings, the tied head, norms and biases take the elementwise path; every
    genuine 2-D operator takes the spectral one. Two details matter and both are
    resolved structurally, because the name-based shortcut is wrong on GPT-2:

    ``transposed``
        A HuggingFace ``Conv1D`` stores its weight as (in, out). Every GPT-2
        projection is a ``Conv1D``, so the operator is the transpose and the
        spectral path has to know it.
    ``blocks``
        The fused ``c_attn`` carries Q, K and V in contiguous row blocks of the
        *operator*, which the optimizer splits before orthogonalising.

    Weight tying makes ``wte.weight`` and ``lm_head.weight`` one tensor, so the
    head is excluded by identity rather than by matching the string ``lm_head``,
    which ``named_parameters`` never reports for a tied model.
    """
    try:
        from transformers.pytorch_utils import Conv1D
    except ImportError:  # pragma: no cover - older/newer layouts
        Conv1D = ()  # type: ignore[assignment]

    excluded, transposed_ids, fused_ids = set(), set(), set()
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            excluded.add(id(module.weight))
    output = getattr(model, "lm_head", None)
    if output is not None:
        excluded.add(id(output.weight))

    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2:
            continue
        if Conv1D and isinstance(module, Conv1D):
            transposed_ids.add(id(weight))
        if name.endswith("c_attn"):
            fused_ids.add(id(weight))

    width = getattr(config, "n_embd", None)
    buckets: dict[tuple[bool, bool], list[torch.nn.Parameter]] = {}
    scalar: list[torch.nn.Parameter] = []
    fused: list[torch.nn.Parameter] = []
    for param in model.parameters():
        if param.ndim < 2 or id(param) in excluded:
            scalar.append(param)
        elif id(param) in fused_ids and width is not None:
            fused.append(param)
        else:
            buckets.setdefault((True, id(param) in transposed_ids), []).append(param)

    groups: list[dict] = [{"params": scalar, "spectral": False, "transposed": False}]
    for (spectral, transposed), params in buckets.items():
        groups.append({"params": params, "spectral": spectral, "transposed": transposed})
    if fused:
        # ASTRO splits; Muon deliberately does not, since not splitting is the
        # behaviour under test.
        groups.append({
            "params": fused, "spectral": True,
            "transposed": all(id(p) in transposed_ids for p in fused),
            "blocks": (width, width, width) if optimizer_name == "astro" else None,
        })
    return groups

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

STATE = Path("astro_bench_state.json")


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

#: An Adam-scaled update and a Muon-scaled one want ranges an order of magnitude
#: apart; the scale each optimizer uses decides which it gets.
ADAM_LR = (1e-4, 3e-3)
MUON_LR = (2e-3, 1e-1)

#: Every optimizer tunes exactly these three. The count is the fairness
#: constraint; the ranges differ because the update scales do.
SPACES: dict[str, dict[str, tuple[float, float]]] = {
    # AdamW has no separate scalar path, so its third knob is beta2 rather than
    # a degenerate multiplier. Counting a fixed value as a tuned dimension would
    # satisfy the equal-budget rule on paper while giving the baseline two real
    # knobs against everyone else's three -- the precise bias the rule exists to
    # prevent, wearing the rule as a disguise.
    "adamw": {"lr": ADAM_LR, "weight_decay": (1e-3, 3e-1), "beta2": (0.9, 0.999)},
    "muon": {"lr": MUON_LR, "weight_decay": (1e-3, 3e-1), "scalar_lr_mult": (0.02, 0.5)},
    "normuon": {"lr": MUON_LR, "weight_decay": (1e-3, 3e-1), "scalar_lr_mult": (0.02, 0.5)},
    "astro": {"lr": MUON_LR, "weight_decay": (1e-3, 3e-1), "scalar_lr_mult": (0.02, 0.5)},
}

#: Round-4 components, each the shipped recipe plus exactly one change so the
#: control isolates it. ``astro`` itself is the control.
VARIANTS: dict[str, dict[str, object]] = {
    "astro": {},
    # -- subtractive: which of ASTRO's additions over NorMuon costs the 0.104 --
    # At 124M, ASTRO trails NorMuon by 0.1044 while NorMuon's own gain over Muon
    # is 0.0051. The damage is therefore not the shared machinery but one of the
    # three things ASTRO adds, and it is twenty times larger than the gain the
    # shared machinery provides. These three variants remove them one at a time.
    "astro_nocautious": {"cautious": False},
    "astro_nosplit": {"split": False},
    "astro_plain_wd": {"cautious_wd": False},
    # -- additive: the round-4 components ----------------------------------
    "astro_converging": {"converging": True},
    "astro_gamma50": {"variance_power": 0.5},
    "astro_pinned": {"post_normalize": True},
    "astro_equil": {"equilibrate": True},
}
for _name, _overrides in list(VARIANTS.items()):
    if _name != "astro":
        SPACES[_name] = dict(SPACES["astro"])


def sample(space: dict[str, tuple[float, float]], rng: random.Random) -> dict[str, float]:
    """One log-uniform draw. A degenerate range returns its single value."""
    return {
        key: low if low == high else math.exp(rng.uniform(math.log(low), math.log(high)))
        for key, (low, high) in space.items()
    }


def build(name: str, model, config: dict[str, float], steps: int):
    overrides = dict(VARIANTS.get(name, {}))
    # ``split`` is not an optimizer argument -- it decides whether the fused QKV
    # projection is handed over as one block or three, which happens in routing.
    split = overrides.pop("split", True)
    routing = name if name in ("adamw", "muon", "normuon") else ("astro" if split else "muon")
    groups = build_groups(model, routing, model.config)
    scalar = config.get("scalar_lr_mult", 1.0)
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=config["lr"],
            betas=(0.9, config.get("beta2", 0.95)),
            weight_decay=config["weight_decay"],
        )
    if name == "muon":
        return Muon(groups, lr=config["lr"], adamw_lr=config["lr"] * scalar,
                    weight_decay=config["weight_decay"])
    if name == "normuon":
        return NorMuon(groups, lr=config["lr"], adamw_lr=config["lr"] * scalar,
                       weight_decay=config["weight_decay"])
    return Astro(groups, lr=config["lr"], scalar_lr_mult=scalar,
                 weight_decay=config["weight_decay"], **overrides)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


def load_tokens(tokenizer, needed: int) -> torch.Tensor:
    from datasets import load_dataset

    print(f"streaming FineWeb-Edu for {needed:,} tokens", flush=True)
    stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
    collected: list[int] = []
    for record in stream:
        collected.extend(tokenizer(record["text"]).input_ids)
        if len(collected) >= needed:
            break
    return torch.tensor(collected[:needed], dtype=torch.long)


def train_once(name: str, config: dict[str, float], seed: int, *, data, shape,
               steps: int, batch: int, seq: int, vocab: int) -> tuple[float, float]:
    """One run. Returns ``(validation loss, seconds)``."""
    from transformers import GPT2Config, GPT2LMHeadModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    train, validation = data

    torch.manual_seed(seed)
    model = GPT2LMHeadModel(
        GPT2Config(n_positions=seq, vocab_size=vocab, **shape)
    ).to(device)
    model.train()
    optimizer = build(name, model, config, steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    generator = torch.Generator().manual_seed(seed + 4242)
    warmup = max(1, steps // 10)
    started = time.perf_counter()

    def batches(source, gen, count):
        # HuggingFace shifts labels internally, so the window is unshifted and
        # ``labels=x``. A pre-shifted target would train two-tokens-ahead.
        for _ in range(count):
            start = torch.randint(0, source.numel() - seq - 1, (batch,), generator=gen)
            yield torch.stack([source[s : s + seq] for s in start]).to(device)

    for step, x in enumerate(batches(train, generator, steps)):
        factor = ((step + 1) / (warmup + 1) if step < warmup else
                  0.1 + 0.45 * (1 + math.cos(math.pi * (step - warmup) /
                                             max(1, steps - warmup))))
        for group in optimizer.param_groups:
            group.setdefault("base_lr", group["lr"])
            group["lr"] = group["base_lr"] * factor
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = model(x, labels=x).loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step % 50 == 0:
            print(f"      step {step:4d}/{steps} train {float(loss.detach()):.4f}", flush=True)

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
    torch.cuda.empty_cache()
    return value, time.perf_counter() - started


# ---------------------------------------------------------------------------
# Resumable state
# ---------------------------------------------------------------------------


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"trials": {}, "evals": {}}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizers", nargs="+",
                        default=["adamw", "muon", "normuon", "astro"])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument(
        "--reuse-config", metavar="FROM", default=None,
        help="skip tuning and take FROM's tuned configuration from the state "
             "file. This is the right move for an ablation: the point is to "
             "isolate a component at fixed hyperparameters, not to re-tune "
             "around it. It also removes the tuning sweep, which is most of the "
             "cost -- three variants on three seeds is 9 runs against 44.",
    )
    parser.add_argument(
        "--config", nargs="+", metavar="K=V", default=None,
        help="hyperparameters to evaluate every listed optimizer at, e.g. "
             "--config lr=0.0144 weight_decay=0.01 scalar_lr_mult=0.1. Use this "
             "when the state file is gone -- a Colab session wipes /content, and "
             "the report records only the tuned learning rate. Include the "
             "control in --optimizers and the ablation is self-contained: every "
             "variant runs at one configuration, so the comparison between them "
             "needs nothing from the lost run.",
    )
    parser.add_argument(
        "--baseline", nargs="+", metavar="NAME=LOSS", default=None,
        help="known losses from an earlier run, shown for context, e.g. "
             "--baseline muon=6.6574 normuon=6.6523. Comparable only if the "
             "task, steps, batch and seeds are identical, and even then only "
             "down to about 0.005: GPU training is not bit-reproducible across "
             "sessions, measured at up to 0.0021 on a re-run of one "
             "configuration with identical seeds.",
    )
    parser.add_argument(
        "--max-minutes", type=float, default=None,
        help="stop starting new runs after this long and report what finished. "
             "A free Colab session is time-limited, and a run killed mid-flight "
             "reports nothing; this one always leaves a table behind.",
    )
    args = parser.parse_args()

    unknown = [n for n in args.optimizers if n not in SPACES]
    if unknown:
        raise SystemExit(f"unknown optimizer(s) {unknown}; available {sorted(SPACES)}")
    counts = {n: len(SPACES[n]) for n in args.optimizers}
    if len(set(counts.values())) > 1:
        raise SystemExit(f"unequal tuning budgets, which would rig the result: {counts}")

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability()
        print(f"GPU {name} sm_{major}{minor} "
              f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB", flush=True)
    else:
        print("no CUDA device; this will be extremely slow", flush=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    needed = args.batch * args.seq * (args.steps + 60)
    tokens = load_tokens(tokenizer, needed)
    split = int(0.95 * tokens.numel())
    data = (tokens[:split], tokens[split:])
    shape = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
    print(f"corpus {tokens.numel():,} tokens; model {shape}; "
          f"{args.steps} steps x batch {args.batch} x seq {args.seq}", flush=True)

    baselines: dict[str, float] = {}
    for item in args.baseline or []:
        key, _, value = item.partition("=")
        baselines[key] = float(value)

    state = load_state()
    runner = dict(data=data, shape=shape, steps=args.steps, batch=args.batch,
                  seq=args.seq, vocab=len(tokenizer))
    started = time.perf_counter()

    def out_of_time() -> bool:
        if args.max_minutes is None:
            return False
        return (time.perf_counter() - started) / 60.0 >= args.max_minutes

    if args.config:
        config = {}
        for item in args.config:
            key, _, value = item.partition("=")
            if not value:
                raise SystemExit(f"--config expects K=V, got {item!r}")
            config[key] = float(value)
        for key, fallback in (("lr", None), ("weight_decay", 0.01),
                              ("scalar_lr_mult", 0.1), ("beta2", 0.95)):
            if key not in config:
                if fallback is None:
                    raise SystemExit(f"--config must include {key}")
                config[key] = fallback
                print(f"--config: {key} not given, using {fallback}", flush=True)
        best = {name: {"config": config, "value": float("nan")} for name in args.optimizers}
        planned = sum(
            1 for name in args.optimizers for s in range(100, 100 + args.seeds)
            if f"{name}|{s}" not in state["evals"]
        )
        print(f"evaluating every optimizer at {config}", flush=True)
        print(f"{planned} runs to go", flush=True)
        if len(args.optimizers) > 1:
            print("NOTE: one configuration for all of them, so the comparison "
                  "between these variants is internally valid regardless of "
                  "what it was tuned on.", flush=True)
    elif args.reuse_config:
        source = args.reuse_config
        entries = [(v["value"], v["config"]) for k, v in state["trials"].items()
                   if k.split("|")[0] == source]
        if not entries:
            raise SystemExit(
                f"no tuned configuration for {source!r} in {STATE}. Run the field "
                f"comparison first, or pick a name that appears in it."
            )
        value, config = min(entries, key=lambda pair: pair[0])
        best = {name: {"config": config, "value": value} for name in args.optimizers}
        per_run = statistics.fmean(
            [v["seconds"] for v in state["evals"].values() if v["seconds"] > 0]
            or [args.steps * 1.2]
        )
        planned = sum(
            1 for name in args.optimizers for s in range(100, 100 + args.seeds)
            if f"{name}|{s}" not in state["evals"]
        )
        print(f"reusing {source}'s tuned configuration: {config}", flush=True)
        print(f"{planned} runs to go at about {per_run:.0f}s each "
              f"-> roughly {planned * per_run / 60:.0f} minutes", flush=True)
        print("NOTE: a variant evaluated at the control's configuration is a "
              "clean isolation of the component, but the configuration was "
              "tuned for the control. A variant that wins anyway is strong "
              "evidence; one that loses may just be mis-tuned.", flush=True)
    else:
        best = tune_all(args, state, runner, started, out_of_time)
        if best is None:
            return 1

    # -- evaluation on disjoint seeds ---------------------------------------
    for name in args.optimizers:
        for seed in range(100, 100 + args.seeds):
            key = f"{name}|{seed}"
            if key in state["evals"]:
                continue
            if out_of_time():
                print(f"\nstopping at the {args.max_minutes:.0f}-minute budget; "
                      f"re-run the same command to continue", flush=True)
                report(args.optimizers, best, state, args.seeds, baselines)
                return 0
            print(f"[eval] {name} seed {seed}", flush=True)
            value, seconds = train_once(name, best[name]["config"], seed, **runner)
            state["evals"][key] = {"value": value, "seconds": seconds}
            save_state(state)
            print(f"      -> {value:.4f} ({seconds:.0f}s)", flush=True)

    report(args.optimizers, best, state, args.seeds, baselines)
    return 0


def tune_all(args, state, runner, started, out_of_time):
    """Equal-budget sweep for every optimizer. Returns the winners, or None."""
    total = len(args.optimizers) * args.trials
    done = sum(1 for k in state["trials"] if k.split("|")[0] in args.optimizers)

    for name in args.optimizers:
        rng = random.Random(12345)  # same stream for everyone
        for trial in range(args.trials):
            config = sample(SPACES[name], rng)
            key = f"{name}|{trial}"
            if key in state["trials"]:
                continue
            if out_of_time():
                print(f"\nstopping at the {args.max_minutes:.0f}-minute budget "
                      "during tuning; re-run the same command to continue",
                      flush=True)
                return None
            shown = " ".join(f"{k}={v:.4g}" for k, v in sorted(config.items()))
            print(f"[tune {done + 1}/{total}] {name} trial {trial + 1}  {shown}",
                  flush=True)
            try:
                value, seconds = train_once(name, config, 0, **runner)
            except (RuntimeError, ValueError) as error:
                print(f"      diverged: {type(error).__name__}", flush=True)
                value, seconds = float("inf"), 0.0
            if math.isnan(value):
                value = float("inf")
            state["trials"][key] = {"config": config, "value": value, "seconds": seconds}
            save_state(state)
            done += 1
            elapsed = time.perf_counter() - started
            print(f"      -> {value:.4f} ({seconds:.0f}s)  "
                  f"eta {(total - done) * elapsed / max(1, done) / 60:.0f}m", flush=True)

    best: dict[str, dict] = {}
    for name in args.optimizers:
        entries = [(v["value"], v["config"]) for k, v in state["trials"].items()
                   if k.split("|")[0] == name]
        value, config = min(entries, key=lambda pair: pair[0])
        best[name] = {"config": config, "value": value}
        print(f"tuned {name}: {value:.4f} {config}", flush=True)
    return best


def report(optimizers: list[str], best: dict, state: dict, seeds: int,
           baselines: dict[str, float] | None = None) -> None:

    values = {
        name: [state["evals"][f"{name}|{s}"]["value"] for s in range(100, 100 + seeds)
               if f"{name}|{s}" in state["evals"]]
        for name in optimizers
    }
    seconds = {
        name: [state["evals"][f"{name}|{s}"]["seconds"] for s in range(100, 100 + seeds)
               if f"{name}|{s}" in state["evals"]]
        for name in optimizers
    }

    # A budget-limited or interrupted run leaves some optimizers unevaluated;
    # reporting must still produce a table rather than raising on an empty mean.
    measured = [name for name in optimizers if values[name]]
    missing = [name for name in optimizers if not values[name]]
    if missing:
        print(f"\nno evaluation seeds yet for: {', '.join(missing)}", flush=True)
    if not measured:
        print("nothing evaluated yet; re-run to continue", flush=True)
        return

    shown = best[measured[0]]["config"]
    lines = ["", f"Configuration: `{shown}`", "",
             "| optimizer | val loss | per-seed | s/run |",
             "|---|---|---|---|"]
    for name in sorted(measured, key=lambda n: statistics.fmean(values[n])):
        lines.append(
            f"| `{name}` | {statistics.fmean(values[name]):.4f} | "
            f"{', '.join(f'{v:.4f}' for v in values[name])} | "
            f"{statistics.fmean(seconds[name]):.0f} |"
        )

    reference = "muon" if "muon" in measured else measured[0]
    lines += ["", f"Paired against `{reference}`, same seeds:", "",
              "| optimizer | paired Δ | seeds won | exact sign p |", "|---|---|---|---|"]
    for name in measured:
        if name == reference:
            continue
        pairs = list(zip(values[name], values[reference], strict=False))
        deltas = [a - b for a, b in pairs]
        wins = sum(1 for d in deltas if d < 0)
        n = len(deltas)
        # Exact two-sided sign test: with n seeds the smallest attainable
        # p-value is 2/2^n, so 3 seeds can never reach 0.05. Reported anyway so
        # the limit is visible rather than implied.
        tail = sum(math.comb(n, k) for k in range(min(wins, n - wins) + 1))
        p = min(1.0, 2 * tail / 2**n) if n else float("nan")
        lines.append(f"| `{name}` | {statistics.fmean(deltas):+.4f} | {wins}/{n} | {p:.4f} |")

    if baselines:
        lines += ["", "From an earlier run at the same task, steps, batch and "
                      "seeds. GPU training is NOT bit-reproducible across "
                      "sessions -- cuDNN autotuning and non-deterministic "
                      "reductions in the backward -- and re-running one "
                      "configuration on identical seeds moved it by up to "
                      "0.0021. Treat a cross-session gap smaller than about "
                      "0.005 as noise; larger gaps are informative but carry "
                      "that floor:", "",
                  "| optimizer | val loss |", "|---|---|"]
        for name, value in sorted(baselines.items(), key=lambda kv: kv[1]):
            lines.append(f"| `{name}` *(earlier)* | {value:.4f} |")

    lines += ["", f"With {seeds} seeds the smallest attainable two-sided p is "
                  f"{2 / 2**seeds:.4f}; treat these as directional."]
    table = "\n".join(lines)
    print(table)
    Path("astro_bench_report.md").write_text(table + "\n")
    print("\nwrote astro_bench_report.md and astro_bench_state.json")


if __name__ == "__main__":
    raise SystemExit(main())
