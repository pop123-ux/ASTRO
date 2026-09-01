r"""Curvature probe: why one optimizer beats another, in the field's own terms.

Wang et al., *Why Muon Outperforms Adam: A Curvature Perspective* (2606.04662),
explain Muon's advantage over Adam with a second-order expansion of the
one-step loss decrease,

    L(W) - L(W - Z)  ~=  <G, Z>  -  (1/2) <Z, H[Z]>
                          \_____/     \___________/
                        first-order   curvature penalty

and then factor the penalty as ``(1/2) ||Z||_F^2 * S_F(W; Z)`` where

    S_F(W; Z) = <Z, H[Z]> / ||Z||_F^2

is the Normalized Directional Sharpness (NDS). Their finding is that Muon and
Adam take comparably sized steps, so Muon's smaller penalty comes from a
*direction* that meets less curvature.

This script runs that same decomposition over ASTRO, Muon, NorMuon and AdamW.
If ASTRO beats Muon, this says whether it does so for the same reason Muon
beats Adam -- and it is a falsifiable prediction rather than a description:
should ASTRO's advantage show up as a *larger* update norm at equal NDS, the
curvature story does not explain it and we report that instead.

Two things make the comparison fair, both taken from the reference:

* **Matched validation loss, not matched step.** A better optimizer is at a
  lower loss by step t, where the landscape is genuinely different, so a
  step-matched curvature comparison is confounded. Trajectories are therefore
  logged with their validation loss and aligned afterwards by interpolation.
* **Probes in fp32.** A Hessian-vector product through fp16 activations is
  numerically worthless, so the probe runs the forward twice in fp32 even when
  training uses AMP.

Output is JSON; ``scripts/figures/fig_curvature.py`` draws it.

Usage (Colab, after cloning the repo):

    python scripts/measure_curvature.py --optimizers muon astro --steps 600 \
        --probe-every 50 --seeds 100 --out curvature.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import astro_lab  # noqa: E402  the optimizers, model sizes and corpus loader

# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------


def flat_dot(left, right) -> float:
    return float(sum((a * b).sum() for a, b in zip(left, right, strict=True)))


def hessian_vector_product(model, batch, vector, params):
    """H[v] by double backprop, in fp32.

    ``torch.autograd.grad`` of ``<grad, v>`` with respect to the parameters is
    the Hessian applied to ``v``; no Hessian is ever formed. The graph is built
    fresh here rather than reused from the training step, because the optimizer
    has already written to the parameters in place by the time we are called
    and the training graph no longer refers to the point we want.
    """
    model.zero_grad(set_to_none=True)
    loss = model(batch, labels=batch).loss
    grads = torch.autograd.grad(loss, params, create_graph=True)
    pairing = sum((g * v).sum() for g, v in zip(grads, vector, strict=True))
    product = torch.autograd.grad(pairing, params)
    return [tensor.detach() for tensor in product], float(loss.detach()), \
        [tensor.detach() for tensor in grads]


@torch.no_grad()
def assign(params, values) -> None:
    for param, value in zip(params, values, strict=True):
        param.copy_(value)


def probe(model, optimizer, batch, probe_batch, params, step_fn, step):
    """One measured optimizer step.

    Returns the pieces of the second-order decomposition, and the update itself
    so a caller wanting the layer split does not pay for a second step.

    The order matters: the real step happens first so that training is
    untouched by the measurement, then the parameters are rewound to take the
    curvature at the point the step was taken from, then restored.

    Every quantity except ``Z`` is evaluated on ``probe_batch``. ``Z`` is the
    true update, computed by the optimizer from the full training batch. The
    decomposition is exact for whatever batch the loss is evaluated on, so a
    smaller probe batch costs precision, not validity.
    """
    # Two full-model copies live at once, ~0.5 GB each at 124M, and the one we
    # stop needing is released before the Hessian-vector product, which is
    # where memory peaks.
    before = [param.detach().clone() for param in params]
    loss_before = step_fn(batch, step)                 # the real training step
    update = [b - param.detach() for b, param in zip(before, params, strict=True)]  # Z

    assign(params, before)                             # rewind to W_before
    del before
    curvature, probe_loss_before, grads = hessian_vector_product(
        model, probe_batch, update, params)

    with torch.no_grad():                              # W_after = W_before - Z
        for param, delta in zip(params, update, strict=True):
            param.sub_(delta)
        probe_loss_after = float(model(probe_batch, labels=probe_batch).loss)

    model.zero_grad(set_to_none=True)

    quadratic = flat_dot(update, curvature)
    norm_squared = float(sum((z * z).sum() for z in update))
    first_order = flat_dot(grads, update)
    del curvature, grads

    return {
        "train_loss": loss_before,
        "probe_loss": probe_loss_before,
        "realized": probe_loss_before - probe_loss_after,
        "first_order": first_order,
        "curvature_penalty": 0.5 * quadratic,
        "predicted": first_order - 0.5 * quadratic,
        "update_norm_sq": norm_squared,
        "nds": quadratic / norm_squared if norm_squared > 0 else float("nan"),
    }, update


# ---------------------------------------------------------------------------
# Within- vs cross-layer split of the same quantity
# ---------------------------------------------------------------------------


def layer_decomposition(model, probe_batch, update, params, names):
    """Split NDS into within-layer and cross-layer Hessian contributions.

    ``S_within`` restricts each layer's update to its own diagonal Hessian
    block; the remainder is the cross-layer interaction. Computed as a second
    pass with the update zeroed outside one layer at a time, so the cost is
    linear in the number of layers -- expensive, hence off unless asked for.
    """
    groups: dict[int, list[int]] = {}
    for index, name in enumerate(names):
        parts = [piece for piece in name.split(".") if piece.isdigit()]
        groups.setdefault(int(parts[0]) if parts else -1, []).append(index)

    within = 0.0
    for indices in groups.values():
        masked = [torch.zeros_like(tensor) for tensor in update]
        for index in indices:
            masked[index] = update[index]
        product, _, _ = hessian_vector_product(model, probe_batch, masked, params)
        within += flat_dot(masked, product)
    return within, len(groups)


# ---------------------------------------------------------------------------
# Training loop with probes
# ---------------------------------------------------------------------------


def run(name: str, config: dict[str, float], seed: int, *, data, size: str,
        steps: int, seq: int, vocab: int, probe_every: int, probe_batch_size: int,
        layers: bool) -> list[dict]:
    from transformers import GPT2Config, GPT2LMHeadModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    shape = dict(astro_lab.SIZES[size])
    batch_size = shape.pop("batch")
    train, validation = data

    torch.manual_seed(seed)
    model = GPT2LMHeadModel(GPT2Config(n_positions=seq, vocab_size=vocab, **shape))
    model.to(device)
    model.train()
    # Dropout would make the two forwards in a probe disagree for reasons that
    # have nothing to do with curvature.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0

    optimizer = astro_lab.build_optimizer(name, model, config)
    params = [p for p in model.parameters() if p.requires_grad]
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    generator = torch.Generator().manual_seed(seed + 4242)
    warmup = max(1, steps // 10)
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    def sample(source, size_):
        start = torch.randint(0, source.numel() - seq - 1, (size_,),
                              generator=generator)
        return torch.stack([source[i:i + seq] for i in start]).to(device)

    def step_fn(batch, step):
        factor = min(1.0, (step + 1) / warmup)
        for group, base in zip(optimizer.param_groups, base_lrs, strict=True):
            group["lr"] = base * factor
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch, labels=batch).loss
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    @torch.no_grad()
    def validate() -> float:
        model.eval()
        total, batches = 0.0, 8
        checker = torch.Generator().manual_seed(1234)
        for _ in range(batches):
            start = torch.randint(0, validation.numel() - seq - 1, (batch_size,),
                                  generator=checker)
            chunk = torch.stack([validation[i:i + seq] for i in start]).to(device)
            total += float(model(chunk, labels=chunk).loss)
        model.train()
        return total / batches

    records: list[dict] = []
    started = time.time()
    for step in range(steps):
        batch = sample(train, batch_size)
        if step % probe_every == 0 or step == steps - 1:
            measurement, update = probe(model, optimizer, batch,
                                        batch[:probe_batch_size], params,
                                        step_fn, step)
            measurement.update(step=step, optimizer=name, seed=seed,
                               val_loss=validate(),
                               elapsed=round(time.time() - started, 1))
            if layers:
                within, count = layer_decomposition(
                    model, batch[:probe_batch_size], update, params, names)
                measurement["nds_within"] = within / measurement["update_norm_sq"]
                measurement["nds_cross"] = measurement["nds"] - measurement["nds_within"]
                measurement["layer_groups"] = count
            del update
            records.append(measurement)
            print(f"  [{name} seed {seed}] step {step:4d}  "
                  f"val {measurement['val_loss']:.4f}  "
                  f"NDS {measurement['nds']:.3e}  "
                  f"|Z|^2 {measurement['update_norm_sq']:.3e}  "
                  f"I1 {measurement['first_order']:.4f}  "
                  f"I2 {measurement['curvature_penalty']:.4f}", flush=True)
        else:
            step_fn(batch, step)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--optimizers", nargs="+",
                        default=["adamw", "muon", "normuon", "astro"])
    parser.add_argument("--size", default="124M", choices=list(astro_lab.SIZES))
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--probe-every", type=int, default=50)
    parser.add_argument("--probe-batch", type=int, default=2,
                        help="sequences used for the Hessian-vector product; "
                             "smaller than the training batch because the "
                             "double backward doubles activation memory")
    parser.add_argument("--seeds", type=int, nargs="+", default=[100])
    parser.add_argument("--layers", action="store_true",
                        help="also split NDS into within- and cross-layer parts")
    parser.add_argument("--out", type=Path, default=Path("curvature.json"))
    parser.add_argument("--config", nargs="*", default=[],
                        help="NAME:KEY=VALUE, e.g. astro:lr=0.02")
    args = parser.parse_args()

    overrides: dict[str, dict[str, float]] = {}
    for item in args.config:
        target, assignment = item.split(":", 1)
        key, value = assignment.split("=", 1)
        overrides.setdefault(target, {})[key] = float(value)

    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    needed = args.steps * astro_lab.SIZES[args.size]["batch"] * args.seq * 2
    tokens = astro_lab.load_tokens(tokenizer, needed + 2_000_000,
                                   Path("fineweb_cache.pt"))
    split = int(tokens.numel() * 0.9)
    data = (tokens[:split], tokens[split:])

    if not torch.cuda.is_available():
        print("WARNING: no GPU. This will be extremely slow at 124M.", flush=True)

    everything: list[dict] = []
    for name in args.optimizers:
        # Midpoint of each tuned range unless overridden: the curvature
        # comparison is about direction, and it is reported at whatever
        # configuration is stated here rather than implying a tuned one.
        config = {key: (low * high) ** 0.5
                  for key, (low, high) in astro_lab.space_for(name).items()}
        config.update(overrides.get(name, {}))
        print(f"\n=== {name}  {config} ===", flush=True)
        for seed in args.seeds:
            everything += run(name, config, seed, data=data, size=args.size,
                              steps=args.steps, seq=args.seq,
                              vocab=len(tokenizer), probe_every=args.probe_every,
                              probe_batch_size=args.probe_batch, layers=args.layers)
            args.out.write_text(json.dumps(
                {"records": everything, "size": args.size, "steps": args.steps,
                 "seq": args.seq, "probe_batch": args.probe_batch,
                 "configs": {n: {k: (low * high) ** 0.5 for k, (low, high)
                                 in astro_lab.space_for(n).items()}
                             | overrides.get(n, {}) for n in args.optimizers}},
                indent=2))
            print(f"  wrote {args.out} ({len(everything)} probes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
