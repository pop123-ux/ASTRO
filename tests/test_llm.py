"""Tests for the GPT-2 language-model benchmark.

The value of this task rests on three things being true, none of which is
visible from a loss number: the model is really the GPT-2 architecture, the
parameters are routed correctly *despite weight tying*, and the fine-tuning task
can actually detect the effect it claims to measure.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from astro.bench.gpt import GPT, GPTConfig
from astro.optimizer import Astro
from astro.routing import ParamKind, classify_module

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _tiny() -> GPT:
    return GPT(GPTConfig(n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=97))


# ---------------------------------------------------------------------------
# Architecture fidelity
# ---------------------------------------------------------------------------


def test_weights_are_tied_like_gpt2() -> None:
    """``wte.weight`` and ``lm_head.weight`` must be one tensor, not two equal ones."""
    model = _tiny()
    assert model.transformer.wte.weight is model.lm_head.weight


def test_tying_hides_the_head_name_from_named_parameters() -> None:
    """The premise of the routing fix: ``lm_head.weight`` is never reported."""
    names = {name for name, _ in _tiny().named_parameters()}
    assert "lm_head.weight" not in names
    assert "transformer.wte.weight" in names


def test_mlp_expands_four_times_and_attention_fuses_qkv() -> None:
    model = _tiny()
    block = model.transformer.h[0]
    assert block.mlp.c_fc.weight.shape == (4 * 64, 64)
    assert block.attn.c_attn.weight.shape == (3 * 64, 64)


def test_residual_projections_get_the_scaled_initialisation() -> None:
    """GPT-2 scales residual-projection init by 1/sqrt(2L); check the std is in range."""
    config = GPTConfig(n_layer=4, n_head=4, n_embd=128, block_size=32, vocab_size=97)
    torch.manual_seed(0)
    model = GPT(config)
    expected = 0.02 / (2 * config.n_layer) ** 0.5
    observed = float(model.transformer.h[0].mlp.c_proj.weight.std())
    assert observed == pytest.approx(expected, rel=0.15)


def test_forward_returns_logits_and_loss() -> None:
    model = _tiny()
    idx = torch.randint(0, 97, (2, 16))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 16, 97)
    assert loss is not None and loss.ndim == 0
    assert model(idx)[1] is None


def test_sequences_longer_than_the_block_are_rejected() -> None:
    model = _tiny()
    with pytest.raises(AssertionError):
        model(torch.randint(0, 97, (1, 64)))


# ---------------------------------------------------------------------------
# Routing under weight tying -- the bug this model exposed
# ---------------------------------------------------------------------------


def test_tied_embedding_is_not_orthogonalised() -> None:
    """Shape alone makes ``wte`` (vocab, n_embd) look like a dense operator.

    It is a lookup table whose rows are indexed by token, and it is also the
    output head. Muon's guidance excludes both. Before structural detection this
    tensor went to the spectral path.
    """
    specs = classify_module(_tiny())
    assert specs["transformer.wte.weight"].kind is ParamKind.TABLE
    assert not specs["transformer.wte.weight"].is_spectral


def test_position_embedding_is_not_orthogonalised() -> None:
    specs = classify_module(_tiny())
    assert specs["transformer.wpe.weight"].kind is ParamKind.TABLE


def test_final_block_projection_stays_on_the_spectral_path() -> None:
    """The 'last 2-D parameter is the head' heuristic misfires under tying.

    With ``lm_head.weight`` deduplicated away, the last 2-D parameter is the
    final block's MLP projection -- a genuine hidden operator that must be
    orthogonalised.
    """
    specs = classify_module(_tiny())
    assert specs["transformer.h.1.mlp.c_proj.weight"].kind is ParamKind.MATRIX
    assert specs["transformer.h.1.mlp.c_proj.weight"].is_spectral


def test_every_transformer_operator_takes_the_spectral_path() -> None:
    """Two blocks x (c_attn, c_proj, c_fc, c_proj) = eight matrices."""
    specs = classify_module(_tiny())
    spectral = {name for name, spec in specs.items() if spec.is_spectral}
    assert len(spectral) == 8
    assert all(".h." in name for name in spectral)


def test_untied_head_is_still_excluded() -> None:
    """Structural head detection must not depend on tying being present."""

    class Untied(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(97, 64)
            self.hidden = nn.Linear(64, 64)
            self.out = nn.Linear(64, 97)

    specs = classify_module(Untied())
    assert specs["embed.weight"].kind is ParamKind.TABLE
    assert specs["out.weight"].kind is ParamKind.TABLE
    assert specs["hidden.weight"].kind is ParamKind.MATRIX


# ---------------------------------------------------------------------------
# The optimizer actually runs on it
# ---------------------------------------------------------------------------


def test_astro_trains_gpt_and_reduces_loss() -> None:
    torch.manual_seed(0)
    model = _tiny()
    optimizer = Astro.from_model(model, lr=1e-3)
    idx = torch.randint(0, 97, (4, 32))

    first = float(model(idx, idx)[1])
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        model(idx, idx)[1].backward()
        optimizer.step()
    assert float(model(idx, idx)[1]) < first


def test_astro_splits_gpt_into_both_paths() -> None:
    """A single-path optimizer would silently make the comparison meaningless."""
    optimizer = Astro.from_model(_tiny(), lr=1e-3)
    spectral = [g for g in optimizer.param_groups if g["spectral"]]
    scalar = [g for g in optimizer.param_groups if not g["spectral"]]
    assert len(spectral[0]["params"]) == 8
    assert scalar and len(scalar[0]["params"]) > 0


# ---------------------------------------------------------------------------
# The learning-rate schedule
# ---------------------------------------------------------------------------


def test_cosine_schedule_warms_up_then_decays_to_the_floor() -> None:
    from astro.bench.llm import _cosine_factor

    factors = [_cosine_factor(step, 400, 40, 0.1) for step in range(400)]
    assert factors[0] == pytest.approx(1 / 41)
    assert max(factors) == pytest.approx(1.0) and factors.index(max(factors)) == 40
    assert factors[-1] == pytest.approx(0.1, abs=2e-3)
    assert factors[:41] == sorted(factors[:41])
    assert factors[41:] == sorted(factors[41:], reverse=True)


def test_schedule_preserves_the_ratio_between_parameter_groups() -> None:
    """The matrix path and the scalar path run at different rates by design.

    A schedule that set an absolute rate would collapse ``scalar_lr_mult`` to 1
    and quietly change the optimizer being measured.
    """
    from astro.bench.llm import _cosine_factor

    optimizer = Astro.from_model(_tiny(), lr=1e-2, scalar_lr_mult=0.3)
    base = [group["lr"] for group in optimizer.param_groups]
    assert base == [1e-2, 3e-3]

    for step in (0, 40, 399):
        factor = _cosine_factor(step, 400, 40, 0.1)
        scaled = [rate * factor for rate in base]
        assert scaled[1] / scaled[0] == pytest.approx(0.3)


def test_unknown_schedule_is_rejected() -> None:
    from astro.bench.llm import gpt_shakespeare_task

    with pytest.raises(ValueError, match="unknown schedule"):
        gpt_shakespeare_task(
            lambda m: torch.optim.AdamW(m.parameters(), lr=1e-3), 0,
            steps=1, schedule="linear",
        )
