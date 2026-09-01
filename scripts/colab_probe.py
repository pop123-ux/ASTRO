#!/usr/bin/env python3
"""GPU probe for the fused-projection defect, sized for a Colab Tesla T4.

    !pip -q install torch transformers datasets
    !python colab_probe.py --part A          # ~5 min, the important one
    !python colab_probe.py --part B          # ~2 min, cost of the fix
    !python colab_probe.py --part C          # ~40 min, GPT-2 124M training

Three independent parts. Each prints a markdown table and appends a JSON record
to ``astro_probe_results.json`` as soon as it finishes, so a Colab disconnect
costs you the running part and nothing earlier.

Part A is the one that matters
------------------------------
The claim under test is that jointly orthogonalising a *fused* QKV projection
hands almost all of the update to V, because Q and K reach the loss through the
softmax Jacobian and receive far smaller gradients. On our 806K-parameter
benchmark models we measure 85/9/5 (V/K/Q). Whether that holds on real
pretrained checkpoints is a question about linear algebra plus one backward
pass -- it needs no training run, and it is the part of the result that does not
depend on the scale we could afford on CPU.

Two layouts, and why the distinction is not pedantic
----------------------------------------------------
``GPT2`` stores the fused projection as ``[Q | K | V]`` in contiguous row
blocks. ``GPT-NeoX`` (Pythia) does not: it reshapes to
``(num_heads, 3 * head_dim)`` and slices *within each head*, so the layout is
``[q0 k0 v0 | q1 k1 v1 | ...]`` and a contiguous three-way split cuts across all
three operators. A splitter that assumes contiguity is silently wrong on every
GPT-NeoX model, which is exactly the class of error this paper is about. Both
layouts are handled and the script reports which one it used.

Llama-style checkpoints (Qwen, SmolLM2, TinyLlama) store ``q_proj``/``k_proj``/
``v_proj`` separately, so the defect cannot arise from the checkpoint itself.
It still arises in training: Megatron-LM, NeMo and most large-scale frameworks
fuse QKV for throughput, and a matrix optimizer sees whatever layout the
framework hands it. For those models the script measures the gradient asymmetry
directly and reports what fusing them *would* do.

Tesla T4 notes
--------------
T4 is ``sm_75``: no bfloat16, and no FlashAttention-2. The Newton-Schulz
iteration is numerically delicate and is therefore run in float32 regardless of
the model's storage dtype -- running it in float16 on a T4 is the most likely
way to get a wrong answer that still looks plausible.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch

RESULTS = Path("astro_probe_results.json")

# A fixed public-domain passage, so Part A runs even where the datasets hub is
# unreachable. The Q/K/V gradient asymmetry is a property of attention, not of
# the text, so the exact sample matters far less than it being real language.
SAMPLE_TEXT = """
The Congress shall have Power To lay and collect Taxes, Duties, Imposts and
Excises, to pay the Debts and provide for the common Defence and general Welfare
of the United States; but all Duties, Imposts and Excises shall be uniform
throughout the United States. To borrow Money on the credit of the United
States; To regulate Commerce with foreign Nations, and among the several States,
and with the Indian Tribes; To establish an uniform Rule of Naturalization, and
uniform Laws on the subject of Bankruptcies throughout the United States; To
coin Money, regulate the Value thereof, and of foreign Coin, and fix the
Standard of Weights and Measures; To provide for the Punishment of counterfeiting
the Securities and current Coin of the United States; To establish Post Offices
and post Roads; To promote the Progress of Science and useful Arts, by securing
for limited Times to Authors and Inventors the exclusive Right to their
respective Writings and Discoveries; To constitute Tribunals inferior to the
supreme Court; To define and punish Piracies and Felonies committed on the high
Seas, and Offences against the Law of Nations; To declare War, grant Letters of
Marque and Reprisal, and make Rules concerning Captures on Land and Water; To
raise and support Armies, but no Appropriation of Money to that Use shall be for
a longer Term than two Years; To provide and maintain a Navy; To make Rules for
the Government and Regulation of the land and naval Forces; To provide for
calling forth the Militia to execute the Laws of the Union, suppress
Insurrections and repel Invasions; To provide for organizing, arming, and
disciplining, the Militia, and for governing such Part of them as may be
employed in the Service of the United States, reserving to the States
respectively, the Appointment of the Officers, and the Authority of training the
Militia according to the discipline prescribed by Congress.
""".strip()


# ---------------------------------------------------------------------------
# Spectral primitives
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


def row_statistics(update: torch.Tensor) -> dict[str, float]:
    """Squared row norms of an update, summarised the way the paper reports them."""
    squared = update.float().pow(2).sum(dim=1)
    rows = squared.numel()
    mean = squared.mean()
    participation = (squared.sum() ** 2) / (rows * squared.pow(2).sum() + 1e-30)
    return {
        "rows": int(rows),
        "cv": float(squared.std(unbiased=False) / (mean + 1e-30)),
        "participation": float(participation),
        "mass": float(squared.sum()),
    }


# ---------------------------------------------------------------------------
# Locating and splitting fused QKV projections
# ---------------------------------------------------------------------------


@dataclass
class Projection:
    """One attention projection, normalised to operator form ``(out, in)``."""

    name: str
    layer: int
    grad: torch.Tensor
    layout: str  # "contiguous" | "interleaved" | "separate"
    heads: int
    kv_heads: int
    head_dim: int


def qkv_row_blocks(projection: Projection) -> dict[str, torch.Tensor]:
    """Row indices belonging to Q, K and V, honouring the framework's layout.

    ``contiguous`` is GPT-2's ``[Q | K | V]``. ``interleaved`` is GPT-NeoX's
    ``[q0 k0 v0 | q1 k1 v1 | ...]``, where a contiguous three-way split would
    cut across all three operators and produce a meaningless measurement.
    """
    rows = projection.grad.size(0)
    device = projection.grad.device
    if projection.layout == "contiguous":
        query = projection.heads * projection.head_dim
        kv = projection.kv_heads * projection.head_dim
        assert query + 2 * kv == rows, f"{query} + 2*{kv} != {rows}"
        index = torch.arange(rows, device=device)
        return {"Q": index[:query], "K": index[query : query + kv],
                "V": index[query + kv :]}

    if projection.layout == "interleaved":
        head_dim, heads = projection.head_dim, projection.heads
        assert heads * 3 * head_dim == rows, f"{heads}*3*{head_dim} != {rows}"
        index = torch.arange(rows, device=device).view(heads, 3, head_dim)
        return {"Q": index[:, 0].reshape(-1), "K": index[:, 1].reshape(-1),
                "V": index[:, 2].reshape(-1)}

    raise ValueError(f"no row blocks for layout {projection.layout!r}")


def measure_projection(projection: Projection, ns_steps: int = 5) -> dict[str, object]:
    """Update-mass share and gradient norms for one fused projection.

    ``fused`` orthogonalises the whole tensor at once, as Muon does today.
    ``split`` orthogonalises Q, K and V separately, which is the proposed fix.
    Both mass shares are reported so the restoration is visible rather than
    asserted.
    """
    grad = projection.grad
    blocks = qkv_row_blocks(projection)

    fused = newton_schulz(grad, ns_steps)
    fused_squared = fused.float().pow(2).sum(dim=1)
    total = fused_squared.sum() + 1e-30
    fused_share = {k: float(fused_squared[idx].sum() / total) for k, idx in blocks.items()}

    split_squared = torch.zeros_like(fused_squared)
    for index in blocks.values():
        piece = newton_schulz(grad[index], ns_steps)
        split_squared[index] = piece.float().pow(2).sum(dim=1)
    split_total = split_squared.sum() + 1e-30
    split_share = {k: float(split_squared[idx].sum() / split_total) for k, idx in blocks.items()}

    grad_norm = {k: float(grad[idx].norm()) for k, idx in blocks.items()}
    return {
        "name": projection.name,
        "layer": projection.layer,
        "shape": list(grad.shape),
        "layout": projection.layout,
        "fused_share": fused_share,
        "split_share": split_share,
        "grad_norm": grad_norm,
        "grad_ratio_v_over_q": grad_norm["V"] / (grad_norm["Q"] + 1e-30),
        "fused_stats": row_statistics(fused),
    }


def collect_projections(model: torch.nn.Module, family: str) -> list[Projection]:
    """Find every attention projection and normalise it to ``(out, in)``.

    GPT-2 stores ``c_attn`` as a ``Conv1D`` whose weight is ``(in, out)`` -- the
    transpose of the operator. Measuring it untransposed swaps rows for columns
    and inverts the entire analysis, so the orientation is asserted rather than
    assumed.
    """
    found: list[Projection] = []
    config = model.config

    if family == "gpt2":
        heads = config.n_head
        head_dim = config.n_embd // heads
        for layer, block in enumerate(model.transformer.h):
            weight = block.attn.c_attn.weight  # Conv1D: (in, out)
            assert weight.shape == (config.n_embd, 3 * config.n_embd), weight.shape
            found.append(Projection(
                name=f"h.{layer}.attn.c_attn", layer=layer,
                grad=weight.grad.T, layout="contiguous",
                heads=heads, kv_heads=heads, head_dim=head_dim,
            ))
        return found

    if family == "neox":
        heads = config.num_attention_heads
        head_dim = config.hidden_size // heads
        for layer, block in enumerate(model.gpt_neox.layers):
            linear = block.attention.query_key_value  # Linear: (out, in)
            found.append(Projection(
                name=f"layers.{layer}.attention.query_key_value", layer=layer,
                grad=linear.weight.grad, layout="interleaved",
                heads=heads, kv_heads=heads, head_dim=head_dim,
            ))
        return found

    if family == "llama":
        heads = config.num_attention_heads
        kv_heads = getattr(config, "num_key_value_heads", heads)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // heads
        for layer, block in enumerate(model.model.layers):
            attention = block.self_attn
            grads = [attention.q_proj.weight.grad, attention.k_proj.weight.grad,
                     attention.v_proj.weight.grad]
            if any(g is None for g in grads):
                continue
            # Separate in the checkpoint; fused by Megatron-style training. The
            # concatenation is what a fused framework would hand the optimizer.
            found.append(Projection(
                name=f"layers.{layer}.self_attn.qkv[simulated fusion]", layer=layer,
                grad=torch.cat(grads, dim=0), layout="contiguous",
                heads=heads, kv_heads=kv_heads, head_dim=head_dim,
            ))
        return found

    raise ValueError(f"unknown family {family!r}")


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------

#: Ungated checkpoints that fit a T4 with room for gradients. Family selects the
#: module walk and, critically, the fused-row layout.
MODELS = [
    ("gpt2", "gpt2", "fp32"),
    ("gpt2-medium", "gpt2", "fp32"),
    ("gpt2-large", "gpt2", "fp16"),
    ("EleutherAI/pythia-410m", "neox", "fp32"),
    ("EleutherAI/pythia-1.4b", "neox", "fp16"),
    ("HuggingFaceTB/SmolLM2-360M", "llama", "fp32"),
    ("Qwen/Qwen2.5-0.5B", "llama", "fp32"),
    ("TinyLlama/TinyLlama_v1.1", "llama", "fp16"),
]


def silence_dropout(model: torch.nn.Module) -> int:
    """Set every dropout probability to zero, and report how many were changed.

    This matters more than it looks. HuggingFace ships GPT-2 with ``p = 0.1`` on
    attention, embeddings and residuals, and a dropout-perturbed gradient does
    not just add noise to this measurement -- it *biases* it. Measured on a
    4-layer GPT-2, V's share of the fused update reads 0.78 with dropout active
    against 0.61 without: a systematic overstatement of the very effect the
    paper claims. Zero dropout is also the conservative direction and makes the
    measurement deterministic, so it is the default here.
    """
    changed = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) and module.p != 0.0:
            module.p = 0.0
            changed += 1
    return changed


def backward_on_batches(model, ids: torch.Tensor, batches: int,
                        scale: float = 1.0) -> float:
    """Accumulate gradients over several windows, returning the mean loss.

    A single short window is a high-variance estimate of the gradient. Averaging
    a handful of disjoint windows costs almost nothing and tightens the estimate;
    with dropout off we measure a per-window spread of about 0.03 in V's share,
    so this is a refinement rather than a correction.

    ``scale`` multiplies the loss before the backward and is divided out of the
    gradients afterwards. A billion-parameter backward in pure float16
    underflows the smaller gradients to zero and can overflow the larger ones to
    infinity -- TinyLlama returned NaN for every share without it. The
    measurement is scale-invariant, so this changes nothing except the dynamic
    range the intermediate values occupy.
    """
    model.zero_grad(set_to_none=True)
    width = max(8, ids.size(1) // batches)
    total, used = 0.0, 0
    for index in range(batches):
        window = ids[:, index * width : (index + 1) * width]
        if window.size(1) < 8:
            break
        loss = model(window, labels=window).loss / batches
        (loss * scale).backward()
        total += float(loss.detach()) * batches
        used += 1
    if scale != 1.0:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad = parameter.grad.float() / scale
    return total / max(1, used)


def run_part_a(names: list[str] | None, seq_len: int, ns_steps: int,
               batches: int = 4, control: bool = True) -> dict[str, object]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    selected = [m for m in MODELS if names is None or m[0] in names]
    records: list[dict[str, object]] = []

    for name, family, dtype_name in selected:
        print(f"\n=== {name} ({family}, {dtype_name}) ===", flush=True)
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
            dtype = torch.float16 if dtype_name == "fp16" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device)
            model.train()
            model.gradient_checkpointing_disable()
            silenced = silence_dropout(model)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            ids = tokenizer(SAMPLE_TEXT, return_tensors="pt",
                            truncation=True, max_length=seq_len).input_ids.to(device)
            if ids.size(1) < 8:
                raise RuntimeError("tokenised sample too short")

            scale = 2.0**12 if dtype == torch.float16 else 1.0
            loss = backward_on_batches(model, ids, batches, scale)
            print(f"  loss {loss:.4f}  tokens {ids.size(1)}  windows {batches}  "
                  f"dropout zeroed on {silenced} modules"
                  + (f"  loss scale {scale:g}" if scale != 1.0 else ""), flush=True)

            projections = collect_projections(model, family)
            if not all(torch.isfinite(p.grad).all() for p in projections):
                raise RuntimeError(
                    "non-finite gradients: the backward overflowed in this dtype. "
                    "Re-run this model in fp32, or raise the loss scale."
                )
            measured = [measure_projection(p, ns_steps) for p in projections]
            for entry in measured:
                entry["model"] = name

            def mean_of(section: str, key: str, rows=measured) -> float:
                return sum(e[section][key] for e in rows) / len(rows)

            def mean_field(key: str, rows=measured) -> float:
                return sum(e[key] for e in rows) / len(rows)

            def mean_stat(key: str, rows=measured) -> float:
                return sum(e["fused_stats"][key] for e in rows) / len(rows)

            summary = {
                "model": name, "family": family, "layers": len(measured),
                "layout": measured[0]["layout"],
                "fused_share": {k: mean_of("fused_share", k) for k in "QKV"},
                "split_share": {k: mean_of("split_share", k) for k in "QKV"},
                "grad_norm": {k: mean_of("grad_norm", k) for k in "QKV"},
                "grad_ratio_v_over_q": mean_field("grad_ratio_v_over_q"),
                "participation_fused": mean_stat("participation"),
                "per_layer": measured,
            }
            fused, split = summary["fused_share"], summary["split_share"]
            print(f"  fused  Q {fused['Q']:.3f}  K {fused['K']:.3f}  V {fused['V']:.3f}")
            print(f"  split  Q {split['Q']:.3f}  K {split['K']:.3f}  V {split['V']:.3f}")
            print(f"  grad |V|/|Q| = {summary['grad_ratio_v_over_q']:.1f}x   "
                  f"participation {summary['participation_fused']:.3f}")
            records.append(summary)

            if control:
                # Same architecture, random weights. If the defect appears here
                # too it is a property of the attention parameterisation; if it
                # appears only on the trained checkpoint it is a property of the
                # solution. A reviewer will ask, and the control is nearly free.
                del model
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    from transformers import AutoConfig
                    from transformers import AutoModelForCausalLM as Auto

                    torch.manual_seed(0)
                    fresh = Auto.from_config(AutoConfig.from_pretrained(name)).to(device)
                    fresh.train()
                    silence_dropout(fresh)
                    backward_on_batches(fresh, ids, batches)
                    fresh_measured = [measure_projection(p, ns_steps)
                                      for p in collect_projections(fresh, family)]
                    share = {k: sum(e["fused_share"][k] for e in fresh_measured)
                                / len(fresh_measured) for k in "QKV"}
                    summary["random_init_fused_share"] = share
                    print(f"  random-init control  Q {share['Q']:.3f}  "
                          f"K {share['K']:.3f}  V {share['V']:.3f}")
                    del fresh
                except Exception as error:  # noqa: BLE001
                    print(f"  control skipped: {type(error).__name__}: {error}")
            else:
                del model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as error:  # noqa: BLE001 - one bad model must not kill the sweep
            print(f"  SKIPPED: {type(error).__name__}: {error}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()

    print("\n" + format_part_a(records))
    return {"part": "A", "records": records}


def format_part_a(records: list[dict[str, object]]) -> str:
    lines = [
        "| model | layers | layout | fused Q/K/V | split Q/K/V | random-init Q/K/V "
        "| grad |V|/|Q| | participation |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        f, s = r["fused_share"], r["split_share"]  # type: ignore[index]
        raw = r.get("random_init_fused_share")
        control = (f"{raw['Q']:.3f} / {raw['K']:.3f} / {raw['V']:.3f}"  # type: ignore[index]
                   if raw else "--")
        lines.append(
            f"| `{r['model']}` | {r['layers']} | {r['layout']} | "
            f"{f['Q']:.3f} / {f['K']:.3f} / {f['V']:.3f} | "
            f"{s['Q']:.3f} / {s['K']:.3f} / {s['V']:.3f} | "
            f"{control} | "
            f"{r['grad_ratio_v_over_q']:.1f}x | {r['participation_fused']:.3f} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------


def run_part_b(ns_steps: int, repeats: int, widths: tuple[int, ...]) -> dict[str, object]:
    """Is splitting the projection cheaper than not splitting it?

    Newton-Schulz is cubic in the row count, so three orthogonalisations of
    ``(d, d)`` should beat one of ``(3d, d)``. The claim in the paper is that the
    fix costs less than the defect; this measures it on the GPU rather than
    inferring it from an operation count.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for width in widths:
        grad = torch.randn(3 * width, width, device=device, dtype=torch.float32)

        def timed(function) -> float:
            if device == "cuda":
                torch.cuda.synchronize()
            for _ in range(3):  # warm up kernels and autotune
                function()
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(repeats):
                function()
            if device == "cuda":
                torch.cuda.synchronize()
            return (time.perf_counter() - start) / repeats * 1000.0

        pieces = grad.split(width, dim=0)
        fused_ms = timed(lambda g=grad: newton_schulz(g, ns_steps))
        split_ms = timed(lambda ps=pieces: [newton_schulz(p, ns_steps) for p in ps])
        rows.append({"width": width, "fused_ms": fused_ms, "split_ms": split_ms,
                     "speedup": fused_ms / split_ms})
        print(f"  d={width:5d}  fused {fused_ms:7.2f} ms   split {split_ms:7.2f} ms   "
              f"{fused_ms / split_ms:.2f}x", flush=True)
        del grad
        torch.cuda.empty_cache()

    table = ["| hidden size | fused (3d x d) | split (3 x d x d) | speedup |", "|---|---|---|---|"]
    for r in rows:
        table.append(f"| {r['width']} | {r['fused_ms']:.2f} ms | {r['split_ms']:.2f} ms | "
                     f"{r['speedup']:.2f}x |")
    print("\n" + "\n".join(table))
    return {"part": "B", "records": rows}


# ---------------------------------------------------------------------------
# Part C -- optimizers, inlined so this file has no dependency on the repo
# ---------------------------------------------------------------------------


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


def run_part_c(steps: int, batch: int, seq_len: int, seeds: int, lr_overrides: dict,
               shape: dict[str, int]) -> dict:
    """GPT-2 124M from scratch on FineWeb-Edu, BPE tokenised, with a tuned LR.

    This is the architecture, scale and tokenisation the optimizer literature
    uses, and the axis on which the CPU results are weakest.

    Each optimizer gets an equal-sized learning-rate sweep on seed 0 before the
    multi-seed evaluation on disjoint seeds, because an untuned three-way
    comparison is the exact failure mode Wen et al. document and would be
    worthless whichever way it came out. The sweep is nonetheless coarse -- three
    rates, half the step budget -- and a few hundred steps at batch 8 is a tiny
    fraction of the Chinchilla-optimal token budget for 124M. This is a smoke
    test at a standard scale, not a publishable comparison, and it says so in
    its own output.
    """
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokens = load_corpus_tokens(tokenizer, needed=batch * seq_len * (steps + 40))
    split = int(0.95 * tokens.numel())
    train, validation = tokens[:split], tokens[split:]
    print(f"corpus: {tokens.numel():,} tokens ({train.numel():,} train)", flush=True)

    config = GPT2Config(n_positions=seq_len, vocab_size=len(tokenizer), **shape)
    print(f"model: {shape}", flush=True)

    def batches(source, generator, count):
        # HuggingFace shifts labels inside the model, so the batch is a single
        # unshifted window and ``labels=x``. Passing a pre-shifted target here
        # would shift twice and silently train two-tokens-ahead prediction.
        for _ in range(count):
            start = torch.randint(0, source.numel() - seq_len - 1, (batch,), generator=generator)
            yield torch.stack([source[s : s + seq_len] for s in start]).to(device)

    def train_once(name: str, lr: float, seed: int, budget: int, quiet: bool = False) -> float:
        torch.manual_seed(seed)
        model = GPT2LMHeadModel(config).to(device)
        model.train()
        groups = build_groups(model, name, config)
        if name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                          betas=(0.9, 0.95), weight_decay=0.01)
        elif name == "muon":
            optimizer = Muon(groups, lr=lr, adamw_lr=lr_overrides.get("muon_aux", 3e-4))
        else:
            optimizer = Astro(groups, lr=lr)

        # fp16 AMP is a GPU path; on CPU the same loop runs in fp32 so the
        # script stays testable without a device.
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        generator = torch.Generator().manual_seed(seed + 4242)
        warmup = max(1, budget // 10)
        for step, x in enumerate(batches(train, generator, budget)):
            factor = ((step + 1) / (warmup + 1) if step < warmup else
                      0.1 + 0.45 * (1 + math.cos(math.pi * (step - warmup) /
                                                 max(1, budget - warmup))))
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
            if not quiet and step % 50 == 0:
                print(f"    {name} lr={lr:.2e} seed{seed} step {step:4d} "
                      f"train {float(loss.detach()):.4f}", flush=True)

        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            evaluation = torch.Generator().manual_seed(777)
            for x in batches(validation, evaluation, 20):
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    total += float(model(x, labels=x).loss)
                count += 1
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        return total / max(1, count)

    # Equal tuning budget per optimizer. Wen et al. attribute most published
    # optimizer speedups to a baseline that was not swept as hard as the
    # proposal, so an untuned three-way comparison here would be worthless no
    # matter which way it came out. Each optimizer gets the same number of
    # learning rates over its own appropriate decade, tuned on seed 0 and
    # evaluated on disjoint seeds. This is a coarse sweep, not a real one.
    grids = {
        "adamw": [3e-4, 6e-4, 1.2e-3],
        "muon": [1e-2, 2e-2, 4e-2],
        # Widened downward after a 124M run selected the bottom edge (5e-3) and
        # diverged at 1e-2. An edge selection is not a tuned result.
        "astro": [1e-3, 2.5e-3, 5e-3],
    }
    tuning: dict[str, dict] = {}
    for name, grid in grids.items():
        if name in lr_overrides:
            tuning[name] = {"lr": lr_overrides[name], "trace": None}
            print(f"tuning {name}: fixed at {lr_overrides[name]:.2e} by --lr", flush=True)
            continue
        budget = max(50, steps // 2)
        trace = {}
        for lr in grid:
            trace[lr] = train_once(name, lr, seed=0, budget=budget, quiet=True)
            print(f"tuning {name}: lr={lr:.2e} -> {trace[lr]:.4f}", flush=True)
        best = min(trace, key=trace.get)  # type: ignore[arg-type]
        tuning[name] = {"lr": best, "trace": trace}
        if best in (grid[0], grid[-1]):
            print(f"  WARNING: {name} selected an edge of its grid ({best:.2e}); "
                  "the optimum may lie outside it", flush=True)

    results: dict[str, list[float]] = {}
    for name in ("adamw", "muon", "astro"):
        lr = tuning[name]["lr"]
        results[name] = []
        for seed in range(100, 100 + seeds):
            started = time.perf_counter()
            value = train_once(name, lr, seed, steps)
            results[name].append(value)
            print(f"{name} seed{seed}: val {value:.4f}  lr {lr:.2e}  "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)

    lines = ["| optimizer | tuned lr | val loss (mean) | per-seed |", "|---|---|---|---|"]
    for name, values in results.items():
        mean = sum(values) / len(values)
        lines.append(f"| `{name}` | {tuning[name]['lr']:.2e} | {mean:.4f} | "
                     f"{', '.join(f'{v:.4f}' for v in values)} |")
    print("\n" + "\n".join(lines))
    print("\nNOTE: 3 learning rates and a few hundred steps is far below the "
          "sweep and token budget the literature uses. Treat this as a smoke "
          "test at 124M, not as a tuned comparison.")
    return {"part": "C", "steps": steps, "batch": batch, "seq_len": seq_len,
            "shape": shape, "tuning": {k: v for k, v in tuning.items()},
            "results": results}


def load_corpus_tokens(tokenizer, needed: int) -> torch.Tensor:
    """FineWeb-Edu if the hub is reachable, WikiText-103 next, the sample last."""
    for loader in (_fineweb, _wikitext):
        try:
            tokens = loader(tokenizer, needed)
            if tokens.numel() >= needed // 4:
                return tokens
        except Exception as error:  # noqa: BLE001
            print(f"  corpus source unavailable ({type(error).__name__}), trying next",
                  flush=True)
    print("  falling back to the built-in sample; Part C results will be weak", flush=True)
    return torch.tensor(tokenizer(SAMPLE_TEXT * 400).input_ids, dtype=torch.long)


def _fineweb(tokenizer, needed: int) -> torch.Tensor:
    from datasets import load_dataset

    stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
    collected: list[int] = []
    for record in stream:
        collected.extend(tokenizer(record["text"]).input_ids)
        if len(collected) >= needed:
            break
    return torch.tensor(collected[:needed], dtype=torch.long)


def _wikitext(tokenizer, needed: int) -> torch.Tensor:
    from datasets import load_dataset

    data = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    collected: list[int] = []
    for record in data:
        if not record["text"].strip():
            continue
        collected.extend(tokenizer(record["text"]).input_ids)
        if len(collected) >= needed:
            break
    return torch.tensor(collected[:needed], dtype=torch.long)


# ---------------------------------------------------------------------------


def report_environment() -> dict[str, object]:
    info: dict[str, object] = {"torch": torch.__version__, "cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        info |= {
            "gpu": torch.cuda.get_device_name(0),
            "capability": f"sm_{major}{minor}",
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
            "bf16": torch.cuda.is_bf16_supported(),
        }
        print(f"GPU {info['gpu']}  {info['capability']}  {info['vram_gb']} GB  "
              f"bf16={info['bf16']}")
        if not info["bf16"]:
            print("  (no bf16 -- Newton-Schulz runs in fp32, as intended on T4)")
    else:
        print("No CUDA device; Part A still works on CPU but will be slow.")
    return info


def save(payload: dict) -> None:
    existing = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    existing.append(payload)
    RESULTS.write_text(json.dumps(existing, indent=2))
    print(f"\nappended to {RESULTS} ({RESULTS.stat().st_size / 1024:.0f} KB)")
    print("Send me that file, or just paste the markdown tables above.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", default="A", choices=["A", "B", "C"])
    parser.add_argument("--models", nargs="+", default=None,
                        help="Part A: restrict to these checkpoints")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batches", type=int, default=4,
                        help="Part A: windows to average the gradient over")
    parser.add_argument("--no-control", action="store_true",
                        help="Part A: skip the random-init control")
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10, help="Part B timing repeats")
    parser.add_argument("--widths", type=int, nargs="+",
                        default=[768, 1024, 1600, 2048, 4096],
                        help="Part B: hidden sizes to time")
    parser.add_argument("--steps", type=int, default=400, help="Part C training steps")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--n-layer", type=int, default=12, help="Part C: GPT-2 124M by default")
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--lr", nargs="*", default=[],
                        help="Part C overrides, e.g. --lr astro=0.008 adamw=6e-4")
    args = parser.parse_args()

    environment = report_environment()
    overrides = {}
    for item in args.lr:
        key, _, value = item.partition("=")
        overrides[key] = float(value)

    if args.part == "A":
        payload = run_part_a(args.models, args.seq_len, args.ns_steps,
                             args.batches, not args.no_control)
    elif args.part == "B":
        payload = run_part_b(args.ns_steps, args.repeats, tuple(args.widths))
    else:
        shape = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
        payload = run_part_c(args.steps, args.batch, args.seq_len, args.seeds, overrides,
                             shape)

    payload["environment"] = environment
    save(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
