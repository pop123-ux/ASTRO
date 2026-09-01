"""Does Muon's non-convergence actually move the step size during training?

``fig_quintic.py`` shows on synthetic matrices that Muon's iteration lands
singular values in a band around its fixed points, so the update's Frobenius
norm depends on the conditioning of the momentum it was handed. That is a
property of a polynomial, and it would be a curiosity if real momentum matrices
were all similarly conditioned.

This script checks whether they are. It trains normally and, every few steps,
records for each spectral parameter:

  * ``ratio``  -- ||Z||_F / sqrt(min(m, n)), which is 1 if the iteration
                  reached the polar factor and drifts if it did not
  * ``sigma_max`` / ``sigma_min`` of the filtered update, showing where in the
    band the values landed
  * the condition number of the momentum that produced it

The prediction is specific and falsifiable: Muon's ratio should move over
training and across layers, and ASTRO's converging schedule should sit at 1.
If Muon's ratio turns out to be flat at 1 in practice, the non-convergence is
real but irrelevant, and we say so.

Output is JSON; ``scripts/figures/fig_drift.py`` draws it.

    python scripts/measure_drift.py --optimizers muon astro --steps 300 \
        --every 10 --out drift.json
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

import astro_lab  # noqa: E402


def spectral_report(update: torch.Tensor, momentum: torch.Tensor | None) -> dict:
    """Where the filtered singular values landed, and what went in."""
    operator = update if update.size(0) >= update.size(1) else update.T
    values = torch.linalg.svdvals(operator.float())
    report = {
        "ratio": float(torch.linalg.norm(update) / min(update.shape) ** 0.5),
        "sigma_max": float(values.max()),
        "sigma_min": float(values.min()),
        "sigma_mean": float(values.mean()),
        "rows": update.size(0), "cols": update.size(1),
    }
    if momentum is not None and momentum.numel():
        source = momentum if momentum.size(0) >= momentum.size(1) else momentum.T
        incoming = torch.linalg.svdvals(source.float())
        floor = float(incoming.min())
        report["momentum_condition"] = (
            float(incoming.max() / floor) if floor > 1e-12 else float("inf"))
    return report


def spectral_parameters(optimizer) -> list[tuple[str, torch.nn.Parameter]]:
    """The parameters that actually take the spectral path.

    Read off the optimizer's own param groups rather than guessed from shapes,
    so a routing change cannot silently make this measure the wrong tensors.
    """
    found = []
    for index, group in enumerate(optimizer.param_groups):
        if not group.get("spectral", False):
            continue
        for position, param in enumerate(group["params"]):
            found.append((f"group{index}.param{position}", param))
    return found


def momentum_of(optimizer, param) -> torch.Tensor | None:
    state = optimizer.state.get(param, {})
    for key in ("momentum", "moment", "exp_avg", "buf"):
        if key in state and torch.is_tensor(state[key]):
            return state[key]
    return None


def run(name: str, config: dict, seed: int, *, data, size: str, steps: int,
        seq: int, vocab: int, every: int, track: int) -> list[dict]:
    from transformers import GPT2Config, GPT2LMHeadModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    shape = dict(astro_lab.SIZES[size])
    batch_size = shape.pop("batch")
    train, _validation = data

    torch.manual_seed(seed)
    model = GPT2LMHeadModel(GPT2Config(n_positions=seq, vocab_size=vocab, **shape))
    model.to(device)
    model.train()
    optimizer = astro_lab.build_optimizer(name, model, config)
    tracked = spectral_parameters(optimizer)
    if not tracked:
        raise SystemExit(f"{name} has no spectral parameter group to measure")
    # Evenly spaced through the depth rather than the first few, so the report
    # is not dominated by the embedding-adjacent layers.
    stride = max(1, len(tracked) // track)
    tracked = tracked[::stride][:track]
    print(f"  tracking {len(tracked)} of {len(spectral_parameters(optimizer))} "
          f"spectral tensors", flush=True)

    generator = torch.Generator().manual_seed(seed + 4242)
    warmup = max(1, steps // 10)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    records: list[dict] = []
    started = time.time()

    for step in range(steps):
        start = torch.randint(0, train.numel() - seq - 1, (batch_size,),
                              generator=generator)
        batch = torch.stack([train[i:i + seq] for i in start]).to(device)

        factor = min(1.0, (step + 1) / warmup)
        for group, base in zip(optimizer.param_groups, base_lrs, strict=True):
            group["lr"] = base * factor

        optimizer.zero_grad(set_to_none=True)
        loss = model(batch, labels=batch).loss
        loss.backward()

        measure = step % every == 0
        before = ([param.detach().clone() for _, param in tracked]
                  if measure else None)
        optimizer.step()

        if measure:
            for (label, param), snapshot in zip(tracked, before, strict=True):
                # Undo the learning rate so the ratio reports the filter's
                # output rather than the step size chosen for it.
                delta = (snapshot - param.detach())
                lr = next(group["lr"] for group in optimizer.param_groups
                          if any(param is other for other in group["params"]))
                if lr > 0:
                    delta = delta / lr
                report = spectral_report(delta, momentum_of(optimizer, param))
                report.update(step=step, optimizer=name, seed=seed, tensor=label,
                              loss=float(loss.detach()),
                              elapsed=round(time.time() - started, 1))
                records.append(report)
            ratios = [r["ratio"] for r in records[-len(tracked):]]
            print(f"  [{name}] step {step:4d}  loss {float(loss.detach()):.4f}  "
                  f"ratio {min(ratios):.4f}-{max(ratios):.4f}  "
                  f"spread {max(ratios) - min(ratios):.4f}", flush=True)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--optimizers", nargs="+", default=["muon", "astro"])
    parser.add_argument("--size", default="124M", choices=list(astro_lab.SIZES))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--track", type=int, default=12,
                        help="spectral tensors to follow, spread through depth")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("drift.json"))
    args = parser.parse_args()

    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    needed = args.steps * astro_lab.SIZES[args.size]["batch"] * args.seq * 2
    tokens = astro_lab.load_tokens(tokenizer, needed + 1_000_000,
                                   Path("fineweb_cache.pt"))
    split = int(tokens.numel() * 0.9)
    data = (tokens[:split], tokens[split:])

    everything: list[dict] = []
    for name in args.optimizers:
        config = {key: (low * high) ** 0.5
                  for key, (low, high) in astro_lab.space_for(name).items()}
        print(f"\n=== {name}  {config} ===", flush=True)
        everything += run(name, config, args.seed, data=data, size=args.size,
                          steps=args.steps, seq=args.seq, vocab=len(tokenizer),
                          every=args.every, track=args.track)
        args.out.write_text(json.dumps({"records": everything, "size": args.size,
                                        "steps": args.steps}, indent=2))
        print(f"  wrote {args.out} ({len(everything)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
