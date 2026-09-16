#!/usr/bin/env python3
"""Zero-GPU structural gate before the ASTRO paper campaign.

This script does not claim literature novelty.  It checks the narrower code-level
statement on which the current paper framing depends:

1. ``astro_v2_blockwise`` must match the update obtained by treating Q, K and V
   as three separate NorMuon parameters.  That is the public split-NorMuon
   prior-art pattern documented by implementations such as Muonium/Dion.
2. full ASTRO-v2 must differ from that blockwise pattern only because it
   concatenates independently polarised blocks and performs one *global*
   row-adaptive redistribution / Frobenius restoration before per-block aspect
   scaling.

If (1) fails, the control is not a trustworthy prior-art proxy and GPU runs must
not start.  If (2) is numerically zero, the proposed ASTRO distinction has no
algorithmic effect and the paper should not frame it as a new optimizer rule.

The script runs on CPU in seconds and writes ``artifacts/novelty_gate.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import paper_campaign as campaign  # noqa: E402


def block_norms(update: torch.Tensor, rows: int) -> list[float]:
    return [float(chunk.norm()) for chunk in update.split(rows, dim=0)]


def main() -> int:
    torch.manual_seed(20260916)
    rows = cols = 16
    blocks = 3
    lr = 0.01

    start_parts = [torch.randn(rows, cols) for _ in range(blocks)]
    fused_start = torch.cat(start_parts, dim=0)

    full = torch.nn.Parameter(fused_start.clone())
    blockwise = torch.nn.Parameter(fused_start.clone())
    separate = [torch.nn.Parameter(part.clone()) for part in start_parts]

    common = dict(
        lr=lr,
        scalar_lr_mult=1.0,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        cautious_wd=False,
    )
    full_opt = campaign.lab.Astro(
        [{
            "params": [full],
            "spectral": True,
            "transposed": False,
            "blocks": (rows, rows, rows),
        }],
        **common,
    )
    block_opt = campaign.AstroV2Blockwise(
        [{
            "params": [blockwise],
            "spectral": True,
            "transposed": False,
            "blocks": (rows, rows, rows),
        }],
        **common,
    )
    split_normuon = campaign.lab.NorMuon(
        [{"params": separate, "spectral": True, "transposed": False}],
        lr=lr,
        adamw_lr=lr,
        momentum=0.90,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    records = []
    for step in range(1, 6):
        grads = [torch.randn(rows, cols) for _ in range(blocks)]
        fused_grad = torch.cat(grads, dim=0)
        full.grad = fused_grad.clone()
        blockwise.grad = fused_grad.clone()
        for param, grad in zip(separate, grads):
            param.grad = grad.clone()

        before_full = full.detach().clone()
        before_block = blockwise.detach().clone()
        before_sep = torch.cat([p.detach().clone() for p in separate], dim=0)

        full_opt.step()
        block_opt.step()
        split_normuon.step()

        sep_after = torch.cat([p.detach() for p in separate], dim=0)
        full_update = (before_full - full.detach()) / lr
        block_update = (before_block - blockwise.detach()) / lr
        sep_update = (before_sep - sep_after) / lr

        block_prior_max = float((block_update - sep_update).abs().max())
        full_block_max = float((full_update - block_update).abs().max())
        records.append(
            {
                "step": step,
                "blockwise_vs_separate_normuon_max_abs": block_prior_max,
                "full_vs_blockwise_max_abs": full_block_max,
                "full_qkv_update_norms": block_norms(full_update, rows),
                "blockwise_qkv_update_norms": block_norms(block_update, rows),
            }
        )

    prior_error = max(r["blockwise_vs_separate_normuon_max_abs"] for r in records)
    structural_gap = max(r["full_vs_blockwise_max_abs"] for r in records)

    payload = {
        "claim": (
            "Blockwise control reproduces three separate NorMuon parameters; "
            "full ASTRO-v2 differs through global post-recombination redistribution/restoration."
        ),
        "steps": records,
        "max_blockwise_vs_separate_normuon": prior_error,
        "max_full_vs_blockwise": structural_gap,
        "blockwise_matches_prior_art_proxy": prior_error < 5e-5,
        "full_rule_is_numerically_distinct": structural_gap > 1e-6,
    }

    out = ROOT / "artifacts" / "novelty_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    print("ASTRO structural novelty gate")
    print("=" * 34)
    print(f"blockwise vs separate NorMuon max |Δ| : {prior_error:.3e}")
    print(f"full ASTRO-v2 vs blockwise max |Δ|   : {structural_gap:.3e}")
    print(f"wrote {out}")

    if not payload["blockwise_matches_prior_art_proxy"]:
        print("FAIL: blockwise control does not reproduce split-NorMuon semantics")
        return 1
    if not payload["full_rule_is_numerically_distinct"]:
        print("FAIL: full global rule is numerically indistinguishable from blockwise control")
        return 2
    print("PASS: the planned GPU control isolates a real structural distinction.")
    print("NOTE: this is a code-level distinction, not proof of literature priority.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
