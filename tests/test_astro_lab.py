"""The scaling lab: manifest honesty, fairness, and no drift from the library.

``astro_lab.py`` vendors the optimizers so it runs from a single upload. A
vendored copy that drifts is worse than none, because its numbers still arrive
under the library's name -- so it is pinned here. The component manifest is
pinned too: it is the file that answers "what is ASTRO", and a flag renamed in
the optimizer must break the manifest rather than survive in prose.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import astro_lab  # noqa: E402

pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

# ---------------------------------------------------------------------------
# The manifest describes the real optimizer
# ---------------------------------------------------------------------------


def test_every_documented_flag_exists_on_the_optimizer() -> None:
    """A renamed or deleted component must break this, not linger in the docs."""
    from astro.optimizer import Astro

    parameters = inspect.signature(Astro.__init__).parameters
    for component in astro_lab.COMPONENTS:
        assert component.flag in parameters, f"{component.flag} is not an Astro argument"


def test_documented_defaults_match_the_optimizer() -> None:
    from astro.optimizer import Astro

    parameters = inspect.signature(Astro.__init__).parameters
    for component in astro_lab.COMPONENTS:
        assert parameters[component.flag].default == component.default, (
            f"{component.flag}: manifest says {component.default!r}, "
            f"Astro says {parameters[component.flag].default!r}"
        )


def test_the_manifest_admits_what_is_unmeasured() -> None:
    """The manifest's value is that it distinguishes evidence from implementation.
    If everything were marked 'kept' it would be marketing."""
    statuses = {c.status for c in astro_lab.COMPONENTS}
    assert statuses <= {"kept", "off", "unmeasured"}
    assert sum(1 for c in astro_lab.COMPONENTS if c.status == "unmeasured") >= 1
    assert sum(1 for c in astro_lab.COMPONENTS if c.status == "kept") >= 1


def test_the_scale_reversal_is_recorded_against_the_component_it_happened_to() -> None:
    cautious = next(c for c in astro_lab.COMPONENTS if c.flag == "cautious")
    assert cautious.default is False
    assert cautious.status == "off"
    assert "1.17M" in cautious.evidence and "124M" in cautious.evidence


# ---------------------------------------------------------------------------
# No drift from the library
# ---------------------------------------------------------------------------


def test_it_runs_from_one_file() -> None:
    source = (ROOT / "scripts" / "astro_lab.py").read_text()
    assert "from colab_probe import" not in source
    assert "from colab_bench import" not in source


def test_vendored_astro_matches_the_library() -> None:
    from astro.optimizer import astro_matrix_update
    from astro.polar import muon_filter

    torch.manual_seed(0)
    start = torch.randn(48, 16)
    weight = torch.nn.Parameter(start.clone())
    weight.grad = torch.randn(48, 16)
    gradient = weight.grad.clone()

    optimizer = astro_lab.Astro(
        [{"params": [weight], "spectral": True, "transposed": False, "blocks": None}],
        lr=1.0, weight_decay=0.0, cautious=True,
    )
    optimizer.step()

    reference = astro_matrix_update(
        gradient, torch.zeros(48, 16), None, muon_filter(5),
        variance_post=torch.zeros(48), beta1=0.95, beta2=0.95, eps=1e-8, step=1,
        variance_axis="row", variance_placement="post", rms_match=True,
        normalize_direction=False, cautious=True, nesterov=True,
        update_scale="muon", blocks=(48,), post_normalize=False,
    )
    assert torch.allclose(weight.detach(), start - reference, atol=1e-5)


def test_vendored_trust_scale_matches_the_library() -> None:
    """The angular-learning-rate scale has to agree too, or the sweep measures
    a different optimizer than the one the paper describes."""
    from astro.optimizer import astro_matrix_update
    from astro.polar import muon_filter

    torch.manual_seed(0)
    start = torch.randn(48, 16)
    weight = torch.nn.Parameter(start.clone())
    weight.grad = torch.randn(48, 16)
    gradient = weight.grad.clone()

    astro_lab.Astro(
        [{"params": [weight], "spectral": True, "transposed": False, "blocks": None}],
        lr=1.0, weight_decay=0.0, cautious=False, update_scale="trust",
    ).step()

    reference = astro_matrix_update(
        gradient, torch.zeros(48, 16), None, muon_filter(5),
        variance_post=torch.zeros(48), beta1=0.95, beta2=0.95, eps=1e-8, step=1,
        variance_axis="row", variance_placement="post", rms_match=True,
        normalize_direction=False, cautious=False, nesterov=True,
        update_scale="trust", blocks=(48,), post_normalize=False, weight=start,
    )
    assert torch.allclose(weight.detach(), start - reference, atol=1e-5)


def test_the_split_schedule_relaxes_to_one_block() -> None:
    """After ``split_steps`` the update must equal the unsplit one, or the
    schedule is not doing what its name says."""
    torch.manual_seed(0)
    start = torch.randn(48, 16)

    def run(split_steps, steps):
        torch.manual_seed(0)
        weight = torch.nn.Parameter(start.clone())
        optimizer = astro_lab.Astro(
            [{"params": [weight], "spectral": True, "transposed": False,
              "blocks": (16, 16, 16)}],
            lr=0.01, weight_decay=0.0, cautious=False, split_steps=split_steps)
        generator = torch.Generator().manual_seed(7)
        for _ in range(steps):
            weight.grad = torch.randn(48, 16, generator=generator)
            optimizer.step()
        return weight.detach().clone()

    # One step: the schedule has not fired yet, so splitting is still active.
    assert torch.allclose(run(1, 1), run(None, 1), atol=1e-6)
    # Past the limit the two diverge from the always-split run.
    assert not torch.allclose(run(1, 6), run(None, 6), atol=1e-4)


def test_vendored_converging_schedule_matches_the_library() -> None:
    from astro.polar import polar_filter

    torch.manual_seed(0)
    matrix = torch.randn(96, 32)
    assert torch.allclose(
        astro_lab.polar_iterate(matrix, 7, converging=True),
        polar_filter(7, 1e-3)(matrix),
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Fairness and sizing
# ---------------------------------------------------------------------------


def test_every_optimizer_tunes_the_same_number_of_real_knobs() -> None:
    counts = {name: len(astro_lab.space_for(name))
              for name in ("adamw", "muon", "normuon", "astro")}
    assert len(set(counts.values())) == 1, counts
    for name in counts:
        for key, (low, high) in astro_lab.space_for(name).items():
            assert low < high, f"{name}.{key} is a fixed value posing as a tuned range"


def test_muon_scaled_optimizers_get_the_higher_learning_rate_range() -> None:
    adam_high = astro_lab.space_for("adamw")["lr"][1]
    for name in ("muon", "normuon", "astro"):
        assert astro_lab.space_for(name)["lr"][1] > adam_high * 5, name


def test_parameter_count_tracks_the_shape_not_a_lookup_table() -> None:
    """The Chinchilla ratio in the report depends on this, and a table keyed by
    size name silently breaks for any shape added later."""
    counted = astro_lab.parameter_count("124M", 50257, 1024)
    assert 110e6 < counted < 140e6, counted
    assert (astro_lab.parameter_count("355M", 50257, 1024)
            > astro_lab.parameter_count("124M", 50257, 1024))


def test_batch_shrinks_as_the_model_grows() -> None:
    batches = [astro_lab.SIZES[s]["batch"] for s in ("45M", "124M", "355M", "774M")]
    assert batches == sorted(batches, reverse=True)


@pytest.mark.parametrize("name", ["adamw", "muon", "normuon", "astro",
                                  "astro_pinned", "astro_trust",
                                  "astro_cautious", "astro_converging",
                                  "astro_gamma0", "astro_gamma25",
                                  "astro_gamma50", "astro_nosplit",
                                  "astro_split100", "astro_equil",
                                  "astro_plain_wd", "astro_v2",
                                  "astro_v2_gamma0", "adamuon"])
def test_every_optimizer_and_variant_builds_and_descends(name) -> None:
    config = GPT2Config(n_layer=2, n_head=2, n_embd=64, n_positions=32, vocab_size=128)
    ids = torch.randint(0, 128, (2, 32))
    # Learning rate from the variant's own registered range, because the trust
    # scale measures the step in a different unit and a shared value would make
    # this test assert that a mis-scaled optimizer still descends.
    low, high = astro_lab.space_for(name)["lr"]
    draw = {"lr": (low * high) ** 0.5, "weight_decay": 0.01,
            "scalar_lr_mult": 0.1, "beta2": 0.95}

    torch.manual_seed(0)
    model = GPT2LMHeadModel(config)
    model.train()
    optimizer = astro_lab.build_optimizer(name, model, draw)
    losses = []
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = model(ids, labels=ids).loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert all(value == value for value in losses), f"{name} produced NaN"
    assert losses[-1] < losses[0], f"{name} did not descend"


def test_sign_test_reports_its_own_floor() -> None:
    assert astro_lab.sign_test([-1.0, -1.0])[1] == pytest.approx(0.5)
    assert astro_lab.sign_test([-1.0] * 3)[1] == pytest.approx(0.25)
    assert astro_lab.sign_test([-1.0, 1.0])[1] == pytest.approx(1.0)
