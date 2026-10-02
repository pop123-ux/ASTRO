#!/usr/bin/env python3
"""Cheap mathematical gate for the frozen ORBIT-v0 mechanism.

This is not a literature search. It checks that the implementation has the exact
structural properties the paper would claim before GPU time is spent:

1. isotropic Q/K statistics make RoPE rotations irrelevant;
2. anisotropic statistics make relative-position rotations change the metric;
3. the full 2x2 rule is not reducible to a diagonal rescaling;
4. joint Frobenius restoration prevents a larger total Q/K step from explaining
   the mechanism.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from orbit import Orbit, OrbitGPT, OrbitGPTConfig


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def run_gate() -> dict:
    torch.manual_seed(123)
    model = OrbitGPT(
        OrbitGPTConfig(vocab_size=127, block_size=32, n_layer=1, n_head=4, n_embd=64)
    )
    attn = model.blocks[0].attn

    eye = torch.eye(2).view(1, 1, 2, 2).repeat(attn.n_head, attn.n_freq, 1, 1)
    with torch.no_grad():
        attn.orbit_q_cov.copy_(eye * 2.0)
        attn.orbit_k_cov.copy_(eye * 3.0)
    iso_rope_q, iso_rope_k = attn.orbit_metrics((1, 7, 31), rotate=True)
    iso_plain_q, iso_plain_k = attn.orbit_metrics((1,), rotate=False)
    isotropic_error = max(
        relative_error(iso_rope_q, iso_plain_q), relative_error(iso_rope_k, iso_plain_k)
    )

    anis_q = torch.tensor([[4.0, 1.1], [1.1, 0.7]])
    anis_k = torch.tensor([[0.9, -0.6], [-0.6, 3.2]])
    with torch.no_grad():
        attn.orbit_q_cov.copy_(
            anis_q.view(1, 1, 2, 2).repeat(attn.n_head, attn.n_freq, 1, 1)
        )
        attn.orbit_k_cov.copy_(
            anis_k.view(1, 1, 2, 2).repeat(attn.n_head, attn.n_freq, 1, 1)
        )
    rope_q, rope_k = attn.orbit_metrics((1, 2, 4, 8, 16, 32), rotate=True)
    plain_q, plain_k = attn.orbit_metrics((1,), rotate=False)
    rope_effect = max(relative_error(rope_q, plain_q), relative_error(rope_k, plain_k))
    diag_q = torch.diag_embed(torch.diagonal(rope_q, dim1=-2, dim2=-1))
    phase_coupling = relative_error(rope_q, diag_q)

    x = torch.randint(0, 127, (2, 16))
    model(x, labels=x).loss.backward()
    optimizer = Orbit(model, lr=0.01, adamw_lr=3e-4, functional_power=1.0)
    pair = model.orbit_qk_pairs()[0]
    uq = optimizer._spectral_candidate(pair["q"], optimizer.param_groups[0])
    uk = optimizer._spectral_candidate(pair["k"], optimizer.param_groups[0])
    before = torch.sqrt(uq.float().square().sum() + uk.float().square().sum())
    q2, k2 = optimizer._precondition_pair(pair, uq.clone(), uk.clone())
    after = torch.sqrt(q2.float().square().sum() + k2.float().square().sum())
    norm_error = float((after - before).abs() / before.clamp_min(1e-12))
    direction_change = max(relative_error(q2, uq), relative_error(k2, uk))

    return {
        "isotropic_rotation_error": isotropic_error,
        "anisotropic_rope_effect": rope_effect,
        "off_diagonal_phase_coupling": phase_coupling,
        "joint_norm_restoration_error": norm_error,
        "update_direction_change": direction_change,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path)
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()
    result = run_gate()
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.strict:
        checks = {
            "isotropic collapse": result["isotropic_rotation_error"] < 1e-5,
            "RoPE effect exists": result["anisotropic_rope_effect"] > 1e-3,
            "phase coupling exists": result["off_diagonal_phase_coupling"] > 1e-3,
            "norm restoration": result["joint_norm_restoration_error"] < 1e-5,
            "direction changes": result["update_direction_change"] > 1e-3,
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise SystemExit("ORBIT mechanism gate failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
