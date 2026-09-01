"""Tests for parameter routing.

Misrouting is silent: orthogonalising a depthwise kernel or a classifier head
raises nothing, it just costs accuracy. These tests are the only thing standing
between a refactor and that failure mode.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from astro.routing import (
    ParamKind,
    classify_module,
    classify_parameter,
    matrix_view,
)


class _ConvNeXtish(nn.Module):
    """The shape zoo that matters: stem, depthwise, pointwise, norms, head."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 32, 4, stride=4)
        self.norm = nn.GroupNorm(1, 32)
        self.depthwise = nn.Conv2d(32, 32, 7, padding=3, groups=32)
        self.pointwise = nn.Conv2d(32, 128, 1)
        self.reduce = nn.Conv2d(128, 32, 1)
        self.gamma = nn.Parameter(torch.ones(32))
        self.proj = nn.Linear(32, 64)
        self.head = nn.Linear(64, 10)


def test_convnext_shapes_route_as_intended() -> None:
    specs = classify_module(_ConvNeXtish())
    assert specs["stem.weight"].kind is ParamKind.STEM
    assert specs["depthwise.weight"].kind is ParamKind.DEPTHWISE
    assert specs["pointwise.weight"].kind is ParamKind.CONV
    assert specs["reduce.weight"].kind is ParamKind.CONV
    assert specs["proj.weight"].kind is ParamKind.MATRIX
    assert specs["head.weight"].kind is ParamKind.TABLE
    assert specs["gamma"].kind is ParamKind.VECTOR
    assert specs["norm.weight"].kind is ParamKind.VECTOR


def test_only_genuine_operators_take_the_spectral_path() -> None:
    specs = classify_module(_ConvNeXtish())
    spectral = {name for name, spec in specs.items() if spec.is_spectral}
    assert spectral == {"pointwise.weight", "reduce.weight", "proj.weight"}


def test_every_parameter_records_why_it_was_routed() -> None:
    """Routing decisions must be inspectable, since they fail silently."""
    for spec in classify_module(_ConvNeXtish()).values():
        assert spec.reason


def test_biases_and_gains_are_never_spectral() -> None:
    for name in ("stem.bias", "norm.bias", "proj.bias", "head.bias"):
        assert not classify_module(_ConvNeXtish())[name].is_spectral


def test_thin_matrices_fall_back_to_the_scalar_path() -> None:
    """Orthogonalising a 4x512 matrix sets four singular values and costs a pass."""
    spec = classify_parameter("block.weight", torch.zeros(4, 512), min_dim=8)
    assert spec.kind is ParamKind.VECTOR
    assert classify_parameter("block.weight", torch.zeros(64, 512)).kind is ParamKind.MATRIX


def test_embedding_names_are_detected() -> None:
    assert classify_parameter("tok_embed.weight", torch.zeros(1000, 64)).kind is ParamKind.TABLE


def test_fc_is_not_assumed_to_be_a_head() -> None:
    """``fc`` names a hidden layer at least as often as an output layer."""
    assert classify_parameter("blocks.0.fc1.weight", torch.zeros(64, 32)).kind is ParamKind.MATRIX


def test_single_matrix_model_keeps_its_matrix() -> None:
    """Head detection must not empty the spectral path on a one-matrix model."""
    model = nn.Linear(32, 32, bias=False)
    assert classify_module(model)["weight"].is_spectral


def test_head_detection_can_be_disabled() -> None:
    specs = classify_module(_ConvNeXtish(), detect_head=False)
    assert specs["head.weight"].kind is ParamKind.MATRIX


def test_conv1d_style_tensors_are_not_orthogonalised_as_matrices() -> None:
    """NVIDIA reported NaNs orthogonalising Mamba conv1d filters."""
    assert classify_parameter("mixer.conv1d.weight", torch.zeros(256, 1, 4)).kind is (
        ParamKind.DEPTHWISE
    )


def test_matrix_view_folds_convolutions_the_standard_way() -> None:
    kernel = torch.zeros(64, 32, 3, 3)
    assert matrix_view(kernel).shape == (64, 32 * 3 * 3)
    assert matrix_view(torch.zeros(8, 16)).shape == (8, 16)


# -- fused attention projections ---------------------------------------------


def test_fused_qkv_is_detected_by_name_and_by_shape() -> None:
    from astro.routing import fused_block_count

    assert fused_block_count("transformer.h.0.attn.c_attn.weight", torch.zeros(384, 128)) == 3
    assert fused_block_count("blocks.0.attn.qkv.weight", torch.zeros(768, 256)) == 3
    assert fused_block_count("layers.0.self_attn.in_proj_weight", torch.zeros(192, 64)) == 3
    assert fused_block_count("h.0.attn.query_key_value.weight", torch.zeros(96, 32)) == 3


def test_ordinary_matrices_are_not_split() -> None:
    from astro.routing import fused_block_count

    assert fused_block_count("mlp.c_fc.weight", torch.zeros(512, 128)) == 1
    assert fused_block_count("mlp.c_proj.weight", torch.zeros(128, 512)) == 1
    assert fused_block_count("attn.c_proj.weight", torch.zeros(128, 128)) == 1
    assert fused_block_count("norm.weight", torch.zeros(128)) == 1


def test_splitting_rebalances_the_update_across_q_k_and_v() -> None:
    """The measurement this whole feature exists for.

    Q and K reach the loss through the softmax Jacobian while V enters linearly,
    so V's gradient is far larger. Orthogonalising the fused tensor as one matrix
    equalises singular values but leaves V holding almost all the row-norm mass,
    which defeats the property Muon exists to provide.
    """
    from astro.optimizer import astro_matrix_update
    from astro.polar import muon_filter

    torch.manual_seed(0)
    grad = torch.randn(384, 128)
    grad[:256] *= 0.02  # Q and K get much smaller gradients than V
    common = dict(
        spectral_filter=muon_filter(5), beta1=0.9, beta2=0.95, eps=1e-8, step=1,
        variance_axis="none", variance_placement="pre", rms_match=True,
        normalize_direction=False, cautious=False,
    )

    def mass(update: torch.Tensor) -> list[float]:
        r2 = update.pow(2).sum(dim=1)
        return [float(r2[i * 128 : (i + 1) * 128].sum() / r2.sum()) for i in range(3)]

    fused = mass(astro_matrix_update(grad.clone(), torch.zeros(384, 128), None, blocks=1, **common))
    split = mass(astro_matrix_update(grad.clone(), torch.zeros(384, 128), None, blocks=3, **common))

    # Fused: V swamps Q and K.
    assert fused[2] > 0.9
    assert fused[0] < 0.05 and fused[1] < 0.05
    # Split: each operator gets a comparable share.
    assert all(0.28 < share < 0.40 for share in split)


def test_splitting_preserves_overall_update_scale() -> None:
    """Rebalancing must not smuggle in a learning-rate change."""
    from astro.optimizer import astro_matrix_update
    from astro.polar import muon_filter

    torch.manual_seed(0)
    grad = torch.randn(384, 128)
    common = dict(
        spectral_filter=muon_filter(5), beta1=0.9, beta2=0.95, eps=1e-8, step=1,
        variance_axis="none", variance_placement="pre", rms_match=True,
        normalize_direction=False, cautious=False,
    )
    fused = astro_matrix_update(grad.clone(), torch.zeros(384, 128), None, blocks=1, **common)
    split = astro_matrix_update(grad.clone(), torch.zeros(384, 128), None, blocks=3, **common)
    assert float(split.norm()) == pytest.approx(float(fused.norm()), rel=0.1)


def test_astro_splits_gpt_attention_and_nothing_else() -> None:
    from astro import Astro
    from astro.bench.gpt import GPT, GPTConfig

    torch.manual_seed(0)
    model = GPT(GPTConfig(n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=97))
    optimizer = Astro.from_model(model, lr=1e-3, split_fused=True)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()

    names = {id(p): n for n, p in model.named_parameters()}
    split = {
        names[id(p)]
        for group in optimizer.param_groups
        for p in group["params"]
        if len(optimizer.state[p].get("blocks", (1,))) > 1
    }
    assert split == {
        "transformer.h.0.attn.c_attn.weight",
        "transformer.h.1.attn.c_attn.weight",
    }


def test_grouped_query_attention_splits_on_the_real_boundaries() -> None:
    """GQA stacks blocks of *unequal* height, so a count is not enough.

    LLaMA-2/3, Mistral, Qwen and Gemma give K and V fewer heads than Q, making the
    fused projection ``(d + 2 d_kv, d)`` with ``d_kv < d``. Splitting that into
    equal thirds would cut across the Q/K/V boundaries -- each block would mix two
    operators, which is worse than not splitting at all.
    """
    from astro import Astro
    from astro.bench.gpt import GPT, GPTConfig

    torch.manual_seed(0)
    model = GPT(
        GPTConfig(n_layer=1, n_head=8, n_embd=256, block_size=32, vocab_size=97, n_kv_head=2)
    )
    weight = model.transformer.h[0].attn.c_attn.weight
    assert tuple(weight.shape) == (256 + 2 * 64, 256)

    optimizer = Astro.from_model(model, lr=1e-3, split_fused=True)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    assert tuple(optimizer.state[weight]["blocks"]) == (256, 64, 64)


def test_multi_query_attention_is_not_missed() -> None:
    """MQA gives ``m = d + 2*head_dim``, which need not be divisible by three."""
    from astro.routing import fused_block_sizes

    # d=256, one kv head of width 32: m = 256 + 64 = 320, and 320 % 3 != 0.
    assert fused_block_sizes("c_attn.weight", torch.zeros(320, 256)) == (256, 32, 32)


def test_real_llama_shapes_split_correctly() -> None:
    from astro.routing import fused_block_sizes

    # LLaMA-3-8B: d=4096, 32 query heads, 8 kv heads -> d_kv = 1024.
    assert fused_block_sizes("qkv.weight", torch.zeros(4096 + 2 * 1024, 4096)) == (
        4096, 1024, 1024,
    )
    # LLaMA-2-7B uses full MHA: d_kv = d.
    assert fused_block_sizes("qkv.weight", torch.zeros(3 * 4096, 4096)) == (4096, 4096, 4096)
