"""Paper-campaign fairness and reference-baseline tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import paper_campaign as campaign  # noqa: E402


def test_auxiliary_lr_receives_the_same_schedule_factor_without_compounding() -> None:
    class Dummy:
        param_groups = [{"lr": 0.02, "adamw_lr": 0.003}]

    opt = Dummy()
    campaign.schedule_optimizer_lrs(opt, 0.1)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.002)
    assert opt.param_groups[0]["adamw_lr"] == pytest.approx(0.0003)

    campaign.schedule_optimizer_lrs(opt, 0.5)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.01)
    assert opt.param_groups[0]["adamw_lr"] == pytest.approx(0.0015)


def test_common_decay_routing_matches_standard_gpt_policy() -> None:
    from transformers import GPT2Config, GPT2LMHeadModel

    model = GPT2LMHeadModel(
        GPT2Config(n_layer=2, n_head=2, n_embd=64, n_positions=32, vocab_size=128)
    )
    groups = campaign.paper_groups(model, "astro", {"weight_decay": 0.1})
    by_id = {
        id(param): group
        for group in groups
        for param in group["params"]
    }

    # Biases and LayerNorm vectors do not decay.
    for name, param in model.named_parameters():
        if param.ndim < 2:
            assert by_id[id(param)]["weight_decay"] == pytest.approx(0.0), name

    # Tied token embedding / LM head stays on AdamW but does decay.
    embedding = model.transformer.wte.weight
    assert by_id[id(embedding)]["spectral"] is False
    assert by_id[id(embedding)]["weight_decay"] == pytest.approx(0.1)

    # Hidden operators decay and use the matrix path.
    hidden = model.transformer.h[0].mlp.c_fc.weight
    assert by_id[id(hidden)]["spectral"] is True
    assert by_id[id(hidden)]["weight_decay"] == pytest.approx(0.1)


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
        adamw_lr=0.001,
        weight_decay=0.0,
        momentum=0.0,
        nesterov=False,
    )
    opt.step()
    assert "x" in seen
    assert set(seen["x"].unique().tolist()) <= {-1.0, 0.0, 1.0}


def test_published_adamuon_uses_its_natural_lr_and_common_auxiliary_knob() -> None:
    space = campaign.paper_space_for("adamuon_ref")
    assert space["lr"] == campaign.lab.ADAM_LR
    assert set(space) == {"lr", "weight_decay", "scalar_lr_mult"}
    assert len(space) == len(campaign.paper_space_for("muon"))


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


def test_blockwise_control_matches_three_separate_normuon_parameters() -> None:
    """Tie the novelty control to the actual split-NorMuon prior-art semantics.

    Muonium/Dion-style ``split_sizes`` are documented to make a fused QKV tensor
    receive the same update as separate Q/K/V parameters.  Here we verify that
    ``AstroV2Blockwise`` does exactly that when beta1=.90: each block is
    orthogonalized, row-normalized, norm-restored, and aspect-scaled under its
    own magnitude budget.  Therefore any measured difference between full
    ASTRO-v2 and this control is specifically the *global* post-recombination
    redistribution/restoration, not QKV splitting or NorMuon adaptation itself.
    """
    torch.manual_seed(17)
    q0, k0, v0 = (torch.randn(8, 8) for _ in range(3))
    fused0 = torch.cat([q0, k0, v0], dim=0)

    fused = torch.nn.Parameter(fused0.clone())
    fused_opt = campaign.AstroV2Blockwise(
        [{
            "params": [fused],
            "spectral": True,
            "transposed": False,
            "blocks": (8, 8, 8),
        }],
        lr=0.01,
        scalar_lr_mult=1.0,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        cautious_wd=False,
    )

    separate = [torch.nn.Parameter(x.clone()) for x in (q0, k0, v0)]
    split_opt = campaign.lab.NorMuon(
        [{"params": separate, "spectral": True, "transposed": False}],
        lr=0.01,
        adamw_lr=0.01,
        momentum=0.90,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    for _ in range(3):
        grads = [torch.randn_like(q0), torch.randn_like(k0), torch.randn_like(v0)]
        fused.grad = torch.cat(grads, dim=0)
        for param, grad in zip(separate, grads):
            param.grad = grad.clone()
        fused_opt.step()
        split_opt.step()

    reference = torch.cat([p.detach() for p in separate], dim=0)
    assert torch.allclose(fused.detach(), reference, atol=3e-5, rtol=3e-5)


def test_corpus_protocol_is_fixed_and_revision_pinned() -> None:
    assert campaign.FINEWEB_REVISION == "v1.0.0"
    assert campaign.TRAIN_TOKENS == 12_000_000
    assert campaign.TOTAL_TOKENS == campaign.TRAIN_TOKENS + campaign.lab.VALIDATION_TOKENS


def test_protocol_record_binds_code_data_and_environment() -> None:
    record = campaign.protocol_record()
    assert len(record["code_digest"]) == 64
    assert record["fineweb_revision"] == "v1.0.0"
    assert record["training_token_pool"] == 12_000_000
    assert {"python", "torch", "torch_cuda", "transformers", "datasets"} <= set(
        record["environment"]
    )
