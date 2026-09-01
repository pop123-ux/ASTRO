"""The T4 benchmark's vendored optimizers must still be the library's.

``colab_bench.py`` carries its own copy of the optimizers so that it runs from a
single upload -- splitting it across two files cost a user a
``ModuleNotFoundError`` when Colab renamed the second one to
``colab_probe (1).py``. A vendored copy that drifts is worse than no copy,
because its numbers still arrive under the library's name, so the copy is pinned
here against ``astro.optimizer``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import colab_bench  # noqa: E402

pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402


def _config() -> GPT2Config:
    return GPT2Config(n_layer=2, n_head=2, n_embd=64, n_positions=32, vocab_size=128)


def test_it_runs_from_one_file() -> None:
    """No import of colab_probe, and every symbol the benchmark needs present."""
    source = (ROOT / "scripts" / "colab_bench.py").read_text()
    assert "from colab_probe import" not in source
    assert "import colab_probe" not in source
    for name in ("Astro", "Muon", "NorMuon", "build_groups", "polar_iterate"):
        assert hasattr(colab_bench, name), name


def test_vendored_astro_matches_the_library_on_one_matrix() -> None:
    from astro.optimizer import astro_matrix_update
    from astro.polar import muon_filter

    torch.manual_seed(0)
    start = torch.randn(48, 16)
    weight = torch.nn.Parameter(start.clone())
    weight.grad = torch.randn(48, 16)
    gradient = weight.grad.clone()

    optimizer = colab_bench.Astro(
        [{"params": [weight], "spectral": True, "transposed": False, "blocks": None}],
        lr=1.0, weight_decay=0.0,
    )
    optimizer.step()

    reference = astro_matrix_update(
        gradient, torch.zeros(48, 16), None, muon_filter(5),
        variance_post=torch.zeros(48), beta1=0.95, beta2=0.95, eps=1e-8, step=1,
        variance_axis="row", variance_placement="post", rms_match=True,
        normalize_direction=False, cautious=True, nesterov=True,
        update_scale="muon", blocks=(48,),
    )
    assert torch.allclose(weight.detach(), start - reference, atol=1e-5)


def test_vendored_converging_schedule_matches_the_library() -> None:
    from astro.polar import polar_filter

    torch.manual_seed(0)
    matrix = torch.randn(96, 32)
    assert torch.allclose(
        colab_bench.polar_iterate(matrix, 7, converging=True),
        polar_filter(7, 1e-3)(matrix),
        atol=1e-5,
    )


def test_every_optimizer_descends_without_nan() -> None:
    config = _config()
    ids = torch.randint(0, 128, (2, 32))
    draw = {"lr": 2e-2, "weight_decay": 0.01, "scalar_lr_mult": 0.1, "beta2": 0.95}

    for name in colab_bench.SPACES:
        torch.manual_seed(0)
        model = GPT2LMHeadModel(config)
        model.train()
        optimizer = colab_bench.build(
            name, model, {**draw, "lr": 3e-3 if name == "adamw" else 2e-2}, 10
        )
        losses = []
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            loss = model(ids, labels=ids).loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        assert all(value == value for value in losses), f"{name} produced NaN"
        assert losses[-1] < losses[0], f"{name} did not descend"


def test_budgets_are_equal_and_every_range_is_real() -> None:
    counts = {name: len(space) for name, space in colab_bench.SPACES.items()}
    assert len(set(counts.values())) == 1, counts
    for name, space in colab_bench.SPACES.items():
        for key, (low, high) in space.items():
            assert low < high, f"{name}.{key} is a fixed value posing as a tuned range"


def test_muon_scaled_optimizers_get_a_higher_learning_rate_range() -> None:
    adam_high = colab_bench.SPACES["adamw"]["lr"][1]
    for name in ("muon", "normuon", "astro"):
        assert colab_bench.SPACES[name]["lr"][1] > adam_high * 5, name
