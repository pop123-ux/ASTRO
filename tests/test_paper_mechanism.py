"""Fast checks for the minimal frozen-config mechanism campaign."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import paper_campaign as campaign  # noqa: E402
import paper_mechanism as mechanism  # noqa: E402


def test_mechanism_campaign_contains_only_required_conditions() -> None:
    assert mechanism.NAMES == (
        "muon",
        "muon_m90",
        "astro_v2_gamma0",
        "astro_v2_nosplit",
        "astro_v2_blockwise",
        "astro_v2",
    )


def test_muon_m90_changes_only_matrix_momentum() -> None:
    from transformers import GPT2Config, GPT2LMHeadModel

    model = GPT2LMHeadModel(
        GPT2Config(n_layer=1, n_head=2, n_embd=32, n_positions=16, vocab_size=64)
    )
    cfg = {"lr": 0.01, "weight_decay": 0.01, "scalar_lr_mult": 0.2}
    opt = mechanism.build_with_beta_control("muon_m90", model, cfg)
    assert all(group["momentum"] == pytest.approx(0.90) for group in opt.param_groups)
    assert all(group["betas"] == (0.9, 0.95) for group in opt.param_groups)
    assert all(group["adamw_lr"] == pytest.approx(0.002) for group in opt.param_groups)


def test_candidate_global_control_exists_and_matches_single_block() -> None:
    # The canonical paper campaign already owns the deeper numerical equivalence
    # test. This check keeps the mechanism runner tied to that registered control.
    assert "astro_v2_blockwise" in campaign.lab.known_optimizers()
    assert campaign.lab.VARIANTS["astro_v2_blockwise"]["betas"] == (0.9, 0.95)
    assert campaign.lab.VARIANTS["astro_v2_blockwise"]["cautious_wd"] is False
