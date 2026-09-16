"""Fast invariants for the paper-grade harness.

These tests do not claim optimizer superiority; they prove that the comparison
code implements the protocol the paper says it implements.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import paper_lab  # noqa: E402


def test_scheduler_scales_muon_auxiliary_lr_with_matrix_lr() -> None:
    group = {"lr": 0.02, "adamw_lr": 0.001}
    paper_lab._schedule_group(group, 0.25)
    assert group["lr"] == pytest.approx(0.005)
    assert group["adamw_lr"] == pytest.approx(0.00025)
    paper_lab._schedule_group(group, 0.10)
    assert group["lr"] == pytest.approx(0.002)
    assert group["adamw_lr"] == pytest.approx(0.0001)


def test_paper_variants_expose_only_the_needed_causal_controls() -> None:
    paper_lab.configure_paper_harness("/tmp/not-read.pt", 12_000_000)
    assert paper_lab.lab.VARIANTS["astro_v2_nosplit"] == {
        "betas": (0.9, 0.95), "cautious_wd": False, "split": False
    }
    assert paper_lab.lab.VARIANTS["astro_v2_gamma0_nosplit"] == {
        "betas": (0.9, 0.95), "cautious_wd": False,
        "variance_power": 0.0, "split": False,
    }
    assert "muon_m90" in paper_lab.lab.known_optimizers()
    assert "adamuon_ref" in paper_lab.lab.known_optimizers()


def test_adamuon_reference_sign_stabilizes_before_newton_schulz(monkeypatch) -> None:
    seen = {}

    def fake_ns(matrix, steps=5):
        seen["input"] = matrix.detach().clone()
        return matrix.float()

    monkeypatch.setattr(paper_lab.lab, "newton_schulz", fake_ns)
    p = torch.nn.Parameter(torch.zeros(4, 4))
    p.grad = torch.tensor([
        [1.0, -2.0, 3.0, -4.0],
        [-1.0, 2.0, -3.0, 4.0],
        [0.2, -0.4, 0.6, -0.8],
        [-0.2, 0.4, -0.6, 0.8],
    ])
    opt = paper_lab.AdaMuonReference(
        [{"params": [p], "spectral": True, "transposed": False}],
        lr=1e-3, weight_decay=0.0,
    )
    opt.step()
    assert torch.equal(seen["input"].abs(), torch.ones_like(seen["input"]))


def test_common_routing_gives_vectors_zero_weight_decay() -> None:
    transformers = pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    model = GPT2LMHeadModel(
        GPT2Config(n_layer=1, n_head=2, n_embd=32, n_positions=16, vocab_size=64)
    )
    groups = paper_lab.paper_groups(model, "muon", model.config, 0.1)
    vector_groups = [g for g in groups if not g["spectral"] and g["weight_decay"] == 0.0]
    assert vector_groups
    assert all(p.ndim < 2 for g in vector_groups for p in g["params"])
