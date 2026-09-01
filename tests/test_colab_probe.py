"""Tests for the Colab GPU probe, runnable without a GPU or the model hub.

The probe runs on someone else's machine, once, and its output goes into a
paper. That makes it exactly the kind of code where a silent error is expensive:
nobody re-derives the number, and a wrong one is indistinguishable from a right
one. Everything here therefore checks a property that would be wrong *quietly*
rather than loudly -- orientation, row layout, label shifting, routing coverage.

The hub is not reachable in CI, so model *weights* are never downloaded;
``transformers`` builds the same architectures from config, which exercises the
identical module walk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from colab_probe import (  # noqa: E402
    Astro,
    Muon,
    Projection,
    backward_on_batches,
    build_groups,
    collect_projections,
    measure_projection,
    newton_schulz,
    qkv_row_blocks,
    row_statistics,
    silence_dropout,
)

transformers = pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402


def _tiny_config() -> GPT2Config:
    return GPT2Config(n_layer=2, n_head=4, n_embd=128, n_positions=64, vocab_size=256)


def _tiny_model() -> GPT2LMHeadModel:
    torch.manual_seed(0)
    return GPT2LMHeadModel(_tiny_config())


# ---------------------------------------------------------------------------
# Spectral primitives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(96, 32), (32, 96), (3 * 128, 128)])
def test_newton_schulz_maps_singular_values_toward_one(shape) -> None:
    """Muon's quintic does not converge to exactly 1; it converges to its own
    fixed point with sigma in roughly [0.68, 1.14]. Pinning the real behaviour
    stops a future 'fix' from silently changing what is being measured.

    The shapes here are the ones actually measured: a fused QKV projection is
    (3d, d), which is well conditioned. See the square case below for why that
    distinction matters.
    """
    torch.manual_seed(0)
    values = torch.linalg.svdvals(newton_schulz(torch.randn(*shape), 5))
    assert 0.6 < float(values.min()) and float(values.max()) < 1.2


def test_newton_schulz_leaves_the_small_singular_tail_behind_on_square_input() -> None:
    """A square Gaussian has a smallest singular value near zero, and five
    quintic steps do not lift it: we measure ~0.04 against a bulk in
    [0.68, 1.16]. This is Muon's known behaviour, not a defect in this code, and
    it is why the measured shapes are aspect-ratio 3 rather than square. A test
    that asserted uniformity here would be asserting something false.
    """
    torch.manual_seed(0)
    values = torch.linalg.svdvals(newton_schulz(torch.randn(64, 64), 5))
    assert float(values.min()) < 0.2
    assert float(values.sort().values[1]) > 0.6
    assert float(values.max()) < 1.2


def test_newton_schulz_runs_in_float32_from_half_input() -> None:
    """A T4 has no bfloat16; the iteration must not inherit float16 precision."""
    out = newton_schulz(torch.randn(64, 32, dtype=torch.float16), 5)
    assert out.dtype == torch.float32


def test_wide_matrices_have_uniform_rows() -> None:
    """Proposition: for m <= n the polar factor has orthonormal rows, so the
    participation ratio is 1 and any row normalisation is inert."""
    torch.manual_seed(0)
    wide = newton_schulz(torch.randn(64, 256), 5)
    assert row_statistics(wide)["participation"] == pytest.approx(1.0, abs=0.02)


# ---------------------------------------------------------------------------
# Row layouts -- the silent-error surface
# ---------------------------------------------------------------------------


def _projection(rows: int, cols: int, layout: str, heads: int, kv: int, dim: int) -> Projection:
    grad = torch.randn(rows, cols)
    return Projection("t", 0, grad, layout, heads, kv, dim)


def test_contiguous_layout_splits_multi_head_attention() -> None:
    blocks = qkv_row_blocks(_projection(3 * 768, 768, "contiguous", 12, 12, 64))
    assert [len(blocks[k]) for k in "QKV"] == [768, 768, 768]
    assert blocks["Q"][0] == 0 and blocks["K"][0] == 768 and blocks["V"][0] == 1536


def test_contiguous_layout_handles_grouped_query_attention() -> None:
    """GQA makes the fused tensor (d + 2*d_kv, d); an equal three-way split of it
    would cut across Q and K."""
    blocks = qkv_row_blocks(_projection(4096 + 2048, 4096, "contiguous", 32, 8, 128))
    assert [len(blocks[k]) for k in "QKV"] == [4096, 1024, 1024]


def test_interleaved_layout_matches_gpt_neox() -> None:
    """GPT-NeoX reshapes to (heads, 3*head_dim) and slices within each head, so
    Q's rows are strided. A contiguous split recovers only half of them."""
    heads, dim = 4, 8
    blocks = qkv_row_blocks(_projection(heads * 3 * dim, 32, "interleaved", heads, heads, dim))
    assert [len(blocks[k]) for k in "QKV"] == [32, 32, 32]
    assert blocks["Q"][:8].tolist() == list(range(0, 8))
    assert blocks["K"][:8].tolist() == list(range(8, 16))
    assert blocks["V"][:8].tolist() == list(range(16, 24))
    assert blocks["Q"][8:16].tolist() == list(range(24, 32))

    naive = torch.arange(heads * 3 * dim).split(heads * dim)[0]
    overlap = len(set(blocks["Q"].tolist()) & set(naive.tolist()))
    assert overlap == 16, "a contiguous split must be visibly wrong here"


def test_unknown_layout_is_rejected() -> None:
    with pytest.raises(ValueError, match="no row blocks"):
        qkv_row_blocks(_projection(96, 32, "separate", 4, 4, 8))


# ---------------------------------------------------------------------------
# The measurement itself
# ---------------------------------------------------------------------------


def test_measurement_detects_planted_v_dominance_and_the_split_restores_parity() -> None:
    torch.manual_seed(0)
    width = 256
    grad = torch.randn(3 * width, width)
    grad[:width] *= 0.02          # Q, starved as the softmax Jacobian makes it
    grad[width : 2 * width] *= 0.05  # K
    out = measure_projection(_projection_from(grad, width))

    assert out["fused_share"]["V"] > 0.8
    assert all(out["split_share"][k] == pytest.approx(1 / 3, abs=0.05) for k in "QKV")
    assert sum(out["fused_share"].values()) == pytest.approx(1.0, abs=1e-4)
    assert sum(out["split_share"].values()) == pytest.approx(1.0, abs=1e-4)


def _projection_from(grad: torch.Tensor, width: int) -> Projection:
    return Projection("synthetic", 0, grad, "contiguous", 4, 4, width // 4)


def test_gpt2_projections_are_transposed_into_operator_form() -> None:
    """HuggingFace Conv1D stores (in, out). Measuring it untransposed swaps rows
    for columns and inverts the entire analysis."""
    model = _tiny_model()
    ids = torch.randint(0, 256, (2, 32))
    model(ids, labels=ids).loss.backward()

    projections = collect_projections(model, "gpt2")
    assert len(projections) == 2
    assert projections[0].grad.shape == (3 * 128, 128)


# ---------------------------------------------------------------------------
# Dropout: a bias, not just noise
# ---------------------------------------------------------------------------


def test_gpt2_ships_with_dropout_on() -> None:
    """If this ever becomes 0.0 upstream, the guard below stops mattering and we
    want the test to say so rather than pass vacuously."""
    config = _tiny_config()
    assert (config.attn_pdrop, config.resid_pdrop, config.embd_pdrop) == (0.1, 0.1, 0.1)


def test_silence_dropout_zeroes_every_module_and_reports_the_count() -> None:
    model = _tiny_model()
    assert any(m.p > 0 for m in model.modules() if isinstance(m, torch.nn.Dropout))
    changed = silence_dropout(model)
    assert changed > 0
    assert all(m.p == 0.0 for m in model.modules() if isinstance(m, torch.nn.Dropout))
    assert silence_dropout(model) == 0


def test_silencing_dropout_makes_the_measurement_deterministic() -> None:
    """With dropout live the same input gives a different answer every call, and
    the difference is a systematic overstatement of V's share rather than noise
    that averages out."""
    model = _tiny_model()
    ids = torch.randint(0, 256, (2, 64))

    def v_share() -> float:
        model.zero_grad(set_to_none=True)
        model(ids, labels=ids).loss.backward()
        return measure_projection(collect_projections(model, "gpt2")[0])["fused_share"]["V"]

    model.train()
    noisy = [v_share() for _ in range(3)]
    assert max(noisy) - min(noisy) > 1e-6, "dropout should perturb the measurement"

    silence_dropout(model)
    clean = [v_share() for _ in range(3)]
    assert max(clean) - min(clean) == pytest.approx(0.0, abs=1e-9)


def test_backward_on_batches_accumulates_over_windows() -> None:
    model = _tiny_model()
    silence_dropout(model)
    ids = torch.randint(0, 256, (2, 64))
    loss = backward_on_batches(model, ids, batches=4)
    assert 0.0 < loss < 20.0
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


# ---------------------------------------------------------------------------
# Routing and the optimizers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["astro", "muon"])
def test_routing_partitions_every_parameter_exactly_once(name) -> None:
    model = _tiny_model()
    groups = build_groups(model, name, _tiny_config())
    routed = [p for group in groups for p in group["params"]]
    assert len(routed) == sum(1 for _ in model.parameters())
    assert len({id(p) for p in routed}) == len(routed)


def test_conv1d_parameters_are_marked_transposed() -> None:
    """Every GPT-2 projection is a Conv1D, so the whole spectral path runs on a
    transposed view -- not just the fused one."""
    model = _tiny_model()
    for group in build_groups(model, "astro", _tiny_config()):
        if group["spectral"]:
            assert group["transposed"], "GPT-2 spectral params are all Conv1D"


def test_only_astro_splits_the_fused_projection() -> None:
    config = _tiny_config()
    model = _tiny_model()
    astro_blocks = [g.get("blocks") for g in build_groups(model, "astro", config)]
    muon_blocks = [g.get("blocks") for g in build_groups(model, "muon", config)]
    assert (config.n_embd,) * 3 in astro_blocks
    assert all(b is None for b in muon_blocks)


def test_tied_head_and_embeddings_stay_off_the_spectral_path() -> None:
    """Weight tying makes wte.weight and lm_head.weight one tensor, which
    ``named_parameters`` reports only under the embedding name -- so a
    name-based exclusion misses it."""
    model = _tiny_model()
    groups = build_groups(model, "astro", _tiny_config())
    spectral = {id(p) for g in groups if g["spectral"] for p in g["params"]}
    assert id(model.transformer.wte.weight) not in spectral
    assert id(model.transformer.wpe.weight) not in spectral
    assert id(model.lm_head.weight) not in spectral


@pytest.mark.parametrize("name,factory,lr", [
    ("astro", Astro, 3e-3),
    ("muon", Muon, 2e-2),
])
def test_optimizers_reduce_loss_without_nan(name, factory, lr) -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    silence_dropout(model)
    optimizer = factory(build_groups(model, name, _tiny_config()), lr=lr)
    ids = torch.randint(0, 256, (2, 32))

    losses = []
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        loss = model(ids, labels=ids).loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert all(value == value for value in losses), "NaN in the loss trace"
    assert losses[-1] < losses[0]


def test_astro_state_matches_operator_orientation() -> None:
    """The momentum buffer lives in operator space; if it were allocated in
    storage space the very first step would raise on the transposed views."""
    model = _tiny_model()
    optimizer = Astro(build_groups(model, "astro", _tiny_config()), lr=1e-3)
    ids = torch.randint(0, 256, (2, 32))
    model(ids, labels=ids).loss.backward()
    optimizer.step()

    weight = model.transformer.h[0].attn.c_attn.weight
    assert optimizer.state[weight]["momentum"].shape == weight.T.shape


# ---------------------------------------------------------------------------
# The inlined optimizer must be the library's, not a lookalike
# ---------------------------------------------------------------------------


def test_inlined_astro_matches_the_library_on_one_matrix() -> None:
    """Part C's conclusions are about ASTRO, so the inlined copy has to be it.

    An earlier revision of the probe inlined the *previous* defaults, which is
    how a 124M run came to measure a configuration already known to be worse.
    This pins the two implementations together on the update they produce.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from colab_probe import Astro as InlinedAstro

    from astro.optimizer import astro_matrix_update
    from astro.polar import muon_filter

    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(48, 16))
    grad = torch.randn(48, 16)
    weight.grad = grad.clone()

    inlined = InlinedAstro(
        [{"params": [weight], "spectral": True, "transposed": False, "blocks": None}],
        lr=1.0, weight_decay=0.0,
    )
    inlined.step()
    produced = -(weight.detach() - torch.zeros(48, 16))  # w started at w0, lr=1

    reference = astro_matrix_update(
        grad, torch.zeros(48, 16), None, muon_filter(5),
        variance_post=torch.zeros(48), beta1=0.95, beta2=0.95, eps=1e-8, step=1,
        variance_axis="row", variance_placement="post", rms_match=True,
        normalize_direction=False, cautious=True, nesterov=True,
        update_scale="muon", blocks=(48,),
    )
    torch.manual_seed(0)
    start = torch.randn(48, 16)
    expected = start - reference
    assert torch.allclose(weight.detach(), expected, atol=1e-5), (
        f"max |inlined - library| = {float((weight.detach() - expected).abs().max()):.2e}"
    )
    assert produced is not None


def test_inlined_astro_uses_post_placement_and_muon_scale() -> None:
    """Guard the two settings whose earlier values cost a 124M run."""
    import inspect

    from colab_probe import Astro as InlinedAstro

    source = inspect.getsource(InlinedAstro)
    assert "direction.pow(2).mean" in source, "second moment must be on the orthogonalised update"
    assert "max(1.0, rows / direction.size(1)) ** 0.5" in source, "Muon's aspect-ratio scale"
    assert inspect.signature(InlinedAstro.__init__).parameters["betas"].default == (0.95, 0.95)


# ---------------------------------------------------------------------------
# NorMuon, the converging schedule, and the benchmark's fairness constraints
# ---------------------------------------------------------------------------


def test_converging_schedule_reaches_one_where_muon_cannot() -> None:
    from colab_probe import polar_iterate

    torch.manual_seed(0)
    matrix = torch.randn(128, 64)

    muon = torch.linalg.svdvals(polar_iterate(matrix, 7, converging=False))
    assert float(muon.min()) < 0.75, "Muon's quintic cannot reach 1 at any budget"

    solved = torch.linalg.svdvals(polar_iterate(matrix, 7, converging=True))
    assert float(solved.min()) == pytest.approx(1.0, abs=1e-3)
    assert float(solved.max()) == pytest.approx(1.0, abs=1e-3)


def test_normuon_applies_its_moment_after_orthogonalisation() -> None:
    """Accumulating on the gradient instead would measure the conditioning the
    polar step exists to remove."""
    import inspect

    from colab_probe import NorMuon

    source = inspect.getsource(NorMuon)
    assert "direction.pow(2).mean" in source
    assert "newton_schulz(nesterov)" in source


def test_normuon_differs_from_muon_but_still_descends() -> None:
    from colab_probe import Muon, NorMuon

    ids = torch.randint(0, 256, (2, 32))

    def final(factory) -> tuple[float, torch.Tensor]:
        torch.manual_seed(0)
        model = _tiny_model()
        silence_dropout(model)
        optimizer = factory(build_groups(model, "muon", _tiny_config()))
        for _ in range(15):
            optimizer.zero_grad(set_to_none=True)
            loss = model(ids, labels=ids).loss
            loss.backward()
            optimizer.step()
        return float(loss.detach()), model.transformer.h[0].attn.c_attn.weight.detach().clone()

    muon_loss, muon_weight = final(lambda g: Muon(g, lr=2e-2))
    nor_loss, nor_weight = final(lambda g: NorMuon(g, lr=2e-2))
    assert not torch.allclose(muon_weight, nor_weight, atol=1e-6)
    assert muon_loss == muon_loss and nor_loss == nor_loss


def test_benchmark_gives_every_optimizer_the_same_number_of_real_knobs() -> None:
    """A fixed-value range would satisfy the equal-budget rule on paper while
    handing the baseline fewer real knobs -- the bias the rule exists to stop."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from colab_bench import SPACES

    counts = {name: len(space) for name, space in SPACES.items()}
    assert len(set(counts.values())) == 1, counts
    for name, space in SPACES.items():
        for key, (low, high) in space.items():
            assert low < high, f"{name}.{key} is a fixed value posing as a tuned range"


def test_benchmark_gives_muon_scaled_optimizers_a_higher_learning_rate_range() -> None:
    """Pairing a Muon-scaled update with Adam's range is a silent handicap this
    project shipped twice."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from colab_bench import SPACES

    adam_high = SPACES["adamw"]["lr"][1]
    for name in ("muon", "normuon", "astro"):
        assert SPACES[name]["lr"][1] > adam_high * 5, name
