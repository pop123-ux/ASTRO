"""Paper-campaign fairness and reference-baseline tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import paper_campaign as campaign  # noqa: E402


def test_auxiliary_lr_receives_the_same_schedule_factor() -> None:
    class Dummy:
        param_groups = [{"lr": 0.02, "adamw_lr": 0.003}]

    opt = Dummy()
    campaign.schedule_optimizer_lrs(opt, 0.1)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.002)
    assert opt.param_groups[0]["adamw_lr"] == pytest.approx(0.0003)

    campaign.schedule_optimizer_lrs(opt, 0.5)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.01)
    assert opt.param_groups[0]["adamw_lr"] == pytest.approx(0.0015)


def test_published_adamuon_signs_before_newton_schulz(monkeypatch) -> None:
    seen = {}

    def capture(x, steps=5):
        seen["x"] = x.detach().clone()
        return x.float()

    monkeypatch.setattr(campaign.lab, "newton_schulz", capture)
    weight = torch.nn.Parameter(torch.zeros(4, 4))
    weight.grad = torch.tensor(
        [[-3.0, -0.1, 2.0, 0.0], [1.0, -2.0, 4.0, -5.0],
         [0.2, 0.3, -0.4, 0.5], [-7.0, 8.0, 9.0, -10.0]]
    )
    opt = campaign.PublishedAdaMuon(
        [{"params": [weight], "spectral": True, "transposed": False}],
        lr=0.01,
        adamw_lr=0.01,
        weight_decay=0.0,
        momentum=0.0,
        nesterov=False,
    )
    opt.step()
    assert "x" in seen
    assert set(seen["x"].unique().tolist()) <= {-1.0, 0.0, 1.0}


def test_published_adamuon_uses_adam_scaled_lr_search() -> None:
    space = campaign.paper_space_for("adamuon_ref")
    assert space["lr"] == campaign.lab.ADAM_LR
    assert set(space) == {"lr", "weight_decay", "momentum"}


def test_paper_only_structural_controls_are_registered() -> None:
    names = set(campaign.lab.known_optimizers())
    assert "astro_v2_nosplit" in names
    assert "astro_v2_blockwise" in names
    assert "adamuon_ref" in names


def test_blockwise_control_matches_v2_on_a_single_block() -> None:
    torch.manual_seed(7)
    start = torch.randn(16, 16)
    grad = torch.randn(16, 16)

    def run(cls):
        p = torch.nn.Parameter(start.clone())
        p.grad = grad.clone()
        opt = cls(
            [{"params": [p], "spectral": True, "transposed": False, "blocks": (16,)}],
            lr=0.01,
            scalar_lr_mult=1.0,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            cautious_wd=False,
        )
        opt.step()
        return p.detach()

    global_v2 = run(campaign.lab.Astro)
    blockwise = run(campaign.AstroV2Blockwise)
    assert torch.allclose(global_v2, blockwise, atol=1e-6, rtol=1e-5)


def test_corpus_protocol_is_fixed_and_revision_pinned() -> None:
    assert campaign.FINEWEB_REVISION == "v1.0.0"
    assert campaign.TRAIN_TOKENS == 12_000_000
    assert campaign.TOTAL_TOKENS == campaign.TRAIN_TOKENS + campaign.lab.VALIDATION_TOKENS
