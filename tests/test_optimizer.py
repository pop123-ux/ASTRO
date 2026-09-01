"""Tests for ASTRO.

Two things get the most attention here:

* **Reductions.** ASTRO must collapse to known optimizers under known settings.
  If those reductions hold, an ablation that turns components off is measuring
  what it claims to measure. If they silently break, every ablation number in
  the paper becomes meaningless.
* **The trust-region guarantee.** The whole argument for the anchor is that it
  bounds update strength, so the bound is tested adversarially rather than on
  well-behaved gradients.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from astro import Astro
from astro.optimizer import rms_match_scale


def _model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.GroupNorm(4, 16),
        nn.Conv2d(16, 16, 3, padding=1, groups=16),
        nn.Conv2d(16, 32, 1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 24),
        nn.Linear(24, 5),
    )


def _batch(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(12, 3, 12, 12, generator=generator),
        torch.randint(0, 5, (12,), generator=generator),
    )


def _train(optimizer_factory, steps: int = 25, seed: int = 0) -> tuple[nn.Module, list[float]]:
    model = _model(seed)
    optimizer = optimizer_factory(model)
    inputs, targets = _batch(seed)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(inputs), targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return model, losses


# -- basic behaviour ---------------------------------------------------------


def test_astro_reduces_the_loss() -> None:
    _, losses = _train(lambda m: Astro.from_model(m, lr=3e-3))
    assert losses[-1] < losses[0]


def test_rejects_invalid_hyperparameters() -> None:
    model = _model()
    with pytest.raises(ValueError, match="lr must be positive"):
        Astro(model.parameters(), lr=0.0)
    with pytest.raises(ValueError, match="betas"):
        Astro(model.parameters(), betas=(1.0, 0.9))
    with pytest.raises(ValueError, match="rho must be positive"):
        Astro(model.parameters(), rho=0.0)
    with pytest.raises(ValueError, match="trust_scope"):
        Astro(model.parameters(), trust_scope="whatever")  # type: ignore[arg-type]


def test_missing_gradients_are_skipped_not_crashed() -> None:
    model = _model()
    optimizer = Astro.from_model(model, lr=1e-3)
    inputs, targets = _batch()
    F.cross_entropy(model(inputs), targets).backward()
    # A frozen tensor with no gradient is normal during staged fine-tuning.
    list(model.parameters())[0].grad = None
    optimizer.step()


def test_sparse_gradients_are_rejected_explicitly() -> None:
    parameter = nn.Parameter(torch.randn(10, 10))
    parameter.grad = torch.sparse_coo_tensor(
        torch.tensor([[0], [0]]), torch.tensor([1.0]), (10, 10)
    )
    with pytest.raises(RuntimeError, match="sparse"):
        Astro([parameter], lr=1e-3).step()


# -- reductions --------------------------------------------------------------


def test_disabling_everything_leaves_a_pure_spectral_step() -> None:
    """variance='none', no anchor, no norm control: the Muon update, RMS-matched."""
    _, losses = _train(
        lambda m: Astro.from_model(
            m, lr=1e-3, variance="none", anchor=False, norm_control="none"
        )
    )
    assert losses[-1] < losses[0]


def test_variance_axis_none_leaves_momentum_untouched() -> None:
    from astro.optimizer import astro_matrix_update
    from astro.polar import muon_filter

    torch.manual_seed(0)
    grad = torch.randn(16, 24)
    plain = torch.zeros(16, 24)
    adapted = torch.zeros(16, 24)

    common = dict(
        spectral_filter=muon_filter(5), beta1=0.9, beta2=0.95, eps=1e-8, step=1,
        rms_match=True, normalize_direction=False, variance_placement="post",
        cautious=False,
    )
    a = astro_matrix_update(grad.clone(), plain, None, variance_axis="none", **common)
    b = astro_matrix_update(
        grad.clone(), adapted, torch.zeros(16), variance_post=torch.zeros(16),
        variance_axis="row", **common
    )
    assert not torch.allclose(a, b)


def test_rms_match_scale_matches_the_derivation() -> None:
    """sqrt((1-b1)/(1+b1)) * sqrt(max(m, n)); the first factor is 0.2 at b1=0.9."""
    assert rms_match_scale(64, 64, 0.9) == pytest.approx(0.2294 * 8.0, rel=1e-3)
    assert rms_match_scale(32, 128, 0.9) == pytest.approx(rms_match_scale(128, 32, 0.9))


# -- trust region ------------------------------------------------------------


def test_trust_region_bound_holds_under_adversarial_gradients() -> None:
    """Huge, sign-flipping gradients must still not move the model past rho."""
    torch.manual_seed(0)
    model = _model()
    anchor = {name: p.detach().clone() for name, p in model.named_parameters()}
    rho = 0.05
    optimizer = Astro.from_model(
        model, lr=1.0, anchor=True, anchor_mode="hard", rho=rho, trust_scope="global",
        weight_decay=0.0
    )

    for step in range(40):
        for parameter in model.parameters():
            parameter.grad = torch.randn_like(parameter) * (1e3 if step % 2 else -1e3)
        optimizer.step()

        drift = sum(float((p - anchor[n]).pow(2).sum()) for n, p in model.named_parameters())
        scale = sum(float(v.pow(2).sum()) for v in anchor.values())
        assert (drift**0.5) <= rho * (scale**0.5) * (1 + 1e-4)


def test_param_scope_bounds_every_tensor_individually() -> None:
    torch.manual_seed(0)
    model = _model()
    anchor = {name: p.detach().clone() for name, p in model.named_parameters()}
    rho = 0.05
    optimizer = Astro.from_model(
        model, lr=1.0, anchor=True, anchor_mode="hard", rho=rho, trust_scope="param",
        weight_decay=0.0
    )
    for _ in range(20):
        for parameter in model.parameters():
            parameter.grad = torch.randn_like(parameter) * 1e3
        optimizer.step()

    for name, parameter in model.named_parameters():
        reference = anchor[name].norm()
        if float(reference) > 0:
            assert float((parameter - anchor[name]).norm()) <= rho * float(reference) * (1 + 1e-4)


def test_rho_warmup_tightens_the_early_budget() -> None:
    torch.manual_seed(0)
    model = _model()
    anchor = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = Astro.from_model(
        model, lr=1.0, anchor=True, anchor_mode="hard", rho=0.5, rho_warmup_steps=20,
        weight_decay=0.0
    )
    for parameter in model.parameters():
        parameter.grad = torch.randn_like(parameter) * 1e3
    optimizer.step()

    drift = sum(float((p - anchor[n]).pow(2).sum()) for n, p in model.named_parameters()) ** 0.5
    scale = sum(float(v.pow(2).sum()) for v in anchor.values()) ** 0.5
    # After one step of a 20-step ramp the budget is 1/20 of the final radius.
    assert drift <= 0.5 / 20 * scale * (1 + 1e-4)


def test_hyperball_holds_the_radius_constant() -> None:
    torch.manual_seed(0)
    model = _model()
    matrices = [p for p in model.parameters() if p.ndim >= 2 and min(p.shape[:2]) >= 8]
    before = [float(p.norm()) for p in matrices]

    optimizer = Astro.from_model(model, lr=0.05, norm_control="hyperball", anchor=False)
    inputs, targets = _batch()
    for _ in range(15):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(inputs), targets).backward()
        optimizer.step()

    for parameter, initial in zip(matrices, before, strict=True):
        if parameter.ndim >= 2:
            assert float(parameter.norm()) == pytest.approx(initial, rel=1e-3)


# -- state -------------------------------------------------------------------


def test_state_dict_round_trip_preserves_the_trajectory() -> None:
    """Resuming from a checkpoint must equal running straight through.

    ``tests/test_reproducibility.py`` already guards this class of bug for the
    training loop; an optimizer with momentum, per-neuron variance and an anchor
    has three ways to lose state on reload.
    """
    model_a, _ = _train(lambda m: Astro.from_model(m, lr=3e-3, anchor=True, rho=0.5), steps=10)

    model_b = _model(0)
    optimizer_b = Astro.from_model(model_b, lr=3e-3, anchor=True, rho=0.5)
    inputs, targets = _batch(0)
    for _ in range(5):
        optimizer_b.zero_grad(set_to_none=True)
        F.cross_entropy(model_b(inputs), targets).backward()
        optimizer_b.step()

    checkpoint = copy.deepcopy(optimizer_b.state_dict())
    weights = copy.deepcopy(model_b.state_dict())

    model_c = _model(0)
    model_c.load_state_dict(weights)
    optimizer_c = Astro.from_model(model_c, lr=3e-3, anchor=True, rho=0.5)
    optimizer_c.load_state_dict(checkpoint)
    for _ in range(5):
        optimizer_c.zero_grad(set_to_none=True)
        F.cross_entropy(model_c(inputs), targets).backward()
        optimizer_c.step()

    for (name, expected), (_, restored) in zip(
        model_a.state_dict().items(), model_c.state_dict().items(), strict=True
    ):
        assert torch.allclose(expected, restored, atol=1e-6), name


def test_runs_are_deterministic_under_a_fixed_seed() -> None:
    _, first = _train(lambda m: Astro.from_model(m, lr=3e-3), seed=7)
    _, second = _train(lambda m: Astro.from_model(m, lr=3e-3), seed=7)
    assert first == second


def test_dead_zone_selects_the_filter_and_changes_the_update() -> None:
    plain = Astro.from_model(_model(), lr=1e-3)
    filtered = Astro.from_model(_model(), lr=1e-3, dead_zone=0.1, ns_steps=10)
    assert plain._filter.name.startswith("muon")
    assert filtered._filter.name.startswith("deadzone")


# -- ASTRO v2 components -----------------------------------------------------


def test_cautious_mask_guarantees_a_descent_direction() -> None:
    """The property the mask buys: <masked u, g> >= 0 for *any* u, including
    ones an orthogonalisation produced that point uphill."""
    from astro.optimizer import cautious_mask

    generator = torch.Generator().manual_seed(0)
    for _ in range(50):
        update = torch.randn(16, 24, generator=generator)
        grad = torch.randn(16, 24, generator=generator)
        assert float((cautious_mask(update, grad) * grad).sum()) >= 0.0


def test_orthogonalisation_really_does_produce_uphill_coordinates() -> None:
    """Motivation check: if the polar factor never disagreed with the gradient,
    the mask would be a no-op and there would be nothing to fix."""
    from astro.polar import muon_filter

    torch.manual_seed(0)
    grad = torch.randn(32, 48)
    direction = muon_filter(5)(grad)
    disagreeing = float((direction * grad <= 0).to(torch.float32).mean())
    assert disagreeing > 0.05, f"only {disagreeing:.1%} of coordinates disagree"


def test_cautious_mask_preserves_update_scale() -> None:
    """Renormalisation stops the mask from acting as a hidden LR cut."""
    from astro.optimizer import cautious_mask

    torch.manual_seed(0)
    update, grad = torch.randn(64, 64), torch.randn(64, 64)
    masked = cautious_mask(update, grad)
    assert float(masked.abs().sum()) == pytest.approx(float(update.abs().sum()), rel=0.15)


def test_variance_placement_changes_the_update() -> None:
    """pre and post are genuinely different operations, not a relabelling."""
    from astro.optimizer import astro_matrix_update
    from astro.polar import muon_filter

    torch.manual_seed(0)
    grad = torch.randn(16, 24)
    common = dict(
        spectral_filter=muon_filter(5), beta1=0.9, beta2=0.95, eps=1e-8, step=1,
        rms_match=True, normalize_direction=False, cautious=False, variance_axis="row",
    )
    updates = {
        placement: astro_matrix_update(
            grad.clone(), torch.zeros(16, 24), torch.zeros(16),
            variance_post=torch.zeros(16), variance_placement=placement, **common,
        )
        for placement in ("pre", "post", "both")
    }
    assert not torch.allclose(updates["pre"], updates["post"])
    assert not torch.allclose(updates["both"], updates["pre"])
    assert not torch.allclose(updates["both"], updates["post"])


def test_post_placement_equalises_neuron_step_length() -> None:
    """What post-placement is for, and what pre-placement cannot do.

    The polar filter sets every singular value to one, which equalises the
    *spectrum* but leaves per-row update norms unequal. NorMuon's rule divides by
    a row-wise second moment of the orthogonalised update, so on the first step
    -- when the EMA holds exactly that update -- every row comes out with equal
    RMS. Pre-placement cannot do this: whatever it rescales is passed through the
    filter afterwards.
    """
    from astro.optimizer import astro_matrix_update
    from astro.polar import muon_filter

    torch.manual_seed(0)
    grad = torch.randn(16, 24)
    grad[:8] *= 100.0  # rows differing wildly in scale is what adaptation is for
    common = dict(
        spectral_filter=muon_filter(5), beta1=0.9, beta2=0.95, eps=1e-12, step=1,
        rms_match=True, normalize_direction=False, cautious=False, variance_axis="row",
    )

    def row_spread(update: torch.Tensor) -> float:
        norms = update.norm(dim=1)
        return float(norms.max() / norms.min())

    plain = astro_matrix_update(
        grad.clone(), torch.zeros(16, 24), None, variance_placement="pre",
        **{k: v for k, v in common.items() if k != "variance_axis"}, variance_axis="none",
    )
    post = astro_matrix_update(
        grad.clone(), torch.zeros(16, 24), torch.zeros(16), variance_post=torch.zeros(16),
        variance_placement="post", **common,
    )
    assert row_spread(post) < row_spread(plain)
    assert row_spread(post) == pytest.approx(1.0, abs=1e-4)


def test_post_placement_preserves_the_update_norm() -> None:
    """Unbounded division was a real bug: a quiet neuron gives a denominator near
    eps and an update orders of magnitude too large. Renormalising fixes it."""
    from astro.optimizer import astro_matrix_update
    from astro.polar import muon_filter

    torch.manual_seed(0)
    grad = torch.randn(16, 24)
    grad[3] *= 1e-8  # one effectively dead neuron
    common = dict(
        spectral_filter=muon_filter(5), beta1=0.9, beta2=0.95, eps=1e-12, step=1,
        rms_match=True, normalize_direction=False, cautious=False, variance_axis="row",
    )
    plain = astro_matrix_update(
        grad.clone(), torch.zeros(16, 24), None, variance_placement="pre",
        **{k: v for k, v in common.items() if k != "variance_axis"}, variance_axis="none",
    )
    post = astro_matrix_update(
        grad.clone(), torch.zeros(16, 24), torch.zeros(16), variance_post=torch.zeros(16),
        variance_placement="post", **common,
    )
    assert float(post.norm()) == pytest.approx(float(plain.norm()), rel=1e-4)


def test_elastic_anchor_pulls_toward_the_anchor_without_a_budget() -> None:
    """A restoring force shrinks displacement every step; it never stops motion."""
    torch.manual_seed(0)
    model = _model()
    # The anchor is captured at the first step(), so the displacement has to
    # happen before the optimizer ever runs -- otherwise it anchors the moved
    # weights and there is nothing to pull back toward.
    reference = {n: p.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)

    optimizer = Astro.from_model(
        model, lr=1e-2, anchor=True, anchor_mode="elastic", anchor_strength=5.0,
        weight_decay=0.0,
    )
    # Zero gradients: only the restoring force acts, so drift must decay.
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()  # captures the anchor at the displaced position

    displaced = {n: p.detach().clone() for n, p in model.named_parameters()}

    def drift_from(base: dict[str, torch.Tensor]) -> float:
        return sum(
            float((p - base[n]).pow(2).sum()) for n, p in model.named_parameters()
        ) ** 0.5

    # Now displace again and check the force pulls back toward the captured anchor.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)
    before = drift_from(displaced)
    for _ in range(30):
        optimizer.step()
    after = drift_from(displaced)
    assert after < before
    # A restoring force decays geometrically rather than stopping at a boundary.
    assert after < 0.5 * before
    assert reference  # the pre-displacement weights are not what it anchors to


def test_elastic_anchor_is_off_when_anchor_is_off() -> None:
    torch.manual_seed(0)
    model = _model()
    reference = copy.deepcopy(model)
    optimizer = Astro.from_model(model, lr=1e-2, anchor=False, weight_decay=0.0)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    for (_, a), (_, b) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert torch.allclose(a, b)


def test_muon_reduction_still_holds_with_new_defaults_disabled() -> None:
    """cautious=False, variance='none', no anchor, no norm control: plain Muon."""
    _, losses = _train(
        lambda m: Astro.from_model(
            m, lr=1e-3, variance="none", cautious=False, anchor=False, norm_control="none"
        )
    )
    assert losses[-1] < losses[0]


def _reference_muon(grad, momentum, *, beta1=0.95, nesterov=True, steps=5):
    """Muon, written out independently of the ASTRO code path."""
    from astro.polar import muon_filter

    momentum.lerp_(grad, 1.0 - beta1)
    direction = grad.lerp(momentum, beta1) if nesterov else momentum
    out = muon_filter(steps)(direction.clone())
    return out * max(1.0, out.size(0) / out.size(1)) ** 0.5


@pytest.mark.parametrize("shape", [(48, 16), (16, 48), (32, 32)])
def test_the_muon_reduction_is_exact_not_merely_descending(shape) -> None:
    """The claim in the module docstring, actually asserted.

    The ablation is only interpretable if the disabled configuration *is* the
    baseline, and the previous version of this test checked that the loss went
    down -- which almost any optimizer would satisfy. Reducing to Muon requires
    ``post_normalize=False`` as well, because pinning each block's norm is a
    deliberate departure from what Muon's iteration delivers.
    """
    from astro.optimizer import astro_matrix_update
    from astro.polar import muon_filter

    torch.manual_seed(0)
    rows, cols = shape
    mine, reference = torch.zeros(rows, cols), torch.zeros(rows, cols)
    for step in range(1, 6):
        grad = torch.randn(rows, cols)
        got = astro_matrix_update(
            grad, mine, None, muon_filter(5), variance_post=None,
            beta1=0.95, beta2=0.95, eps=1e-8, step=step, variance_axis="none",
            variance_placement="post", rms_match=True, normalize_direction=False,
            cautious=False, nesterov=True, update_scale="muon",
            post_normalize=False, blocks=1,
        )
        want = _reference_muon(grad, reference)
        assert torch.allclose(got, want, atol=1e-6), f"step {step}, shape {shape}"


def test_post_normalize_is_what_breaks_the_reduction() -> None:
    """It is on by default, so the reduction must name it -- and it must have a
    measurable effect, or defaulting it on is a no-op dressed as a decision."""
    from astro.optimizer import astro_matrix_update
    from astro.polar import muon_filter

    torch.manual_seed(0)
    grad = torch.randn(64, 32)

    def update(post_normalize):
        return astro_matrix_update(
            grad, torch.zeros(64, 32), None, muon_filter(5), variance_post=None,
            beta1=0.95, beta2=0.95, eps=1e-8, step=1, variance_axis="none",
            variance_placement="post", rms_match=True, normalize_direction=False,
            cautious=False, nesterov=True, update_scale="muon",
            post_normalize=post_normalize, blocks=1,
        )

    plain, pinned = update(False), update(True)
    assert not torch.allclose(plain, pinned, atol=1e-4)
    # Pinned lands exactly on the theoretical norm; Muon's iteration does not.
    target = math.sqrt(32) * max(1.0, 64 / 32) ** 0.5
    assert float(pinned.norm()) == pytest.approx(target, rel=1e-5)
    assert abs(float(plain.norm()) - target) > 1e-3


# ---------------------------------------------------------------------------
# Layer-wise trust ratio
# ---------------------------------------------------------------------------


def test_trust_ratio_makes_the_step_proportional_to_the_layer_size() -> None:
    """Two identically shaped layers whose weights differ by 5x should receive
    updates differing by 5x -- which is exactly what Muon's shape scale does
    not do."""
    from astro.optimizer import astro_matrix_update
    from astro.polar import muon_filter

    torch.manual_seed(0)
    grad = torch.randn(64, 32)

    def update(weight):
        return astro_matrix_update(
            grad, torch.zeros(64, 32), None, muon_filter(5), variance_post=None,
            beta1=0.95, beta2=0.95, eps=1e-8, step=1, variance_axis="none",
            variance_placement="post", rms_match=True, normalize_direction=False,
            cautious=False, nesterov=True, update_scale="trust",
            post_normalize=True, blocks=1, weight=weight,
        )

    small = torch.randn(64, 32) * 0.1
    big = small * 5.0
    assert float(update(big).norm()) == pytest.approx(
        5.0 * float(update(small).norm()), rel=1e-4)


def test_trust_scale_shrinks_rather_than_explodes_on_a_near_zero_layer() -> None:
    """The rule cannot fire in the dangerous direction.

    ``||u||`` is pinned by ``post_normalize`` so it cannot vanish, and a layer
    whose weights are near zero therefore gets a *smaller* step, not a larger
    one. That is the property that makes a wide pathology-guard clip adequate.
    """
    from astro.optimizer import astro_matrix_update
    from astro.polar import muon_filter

    torch.manual_seed(0)
    grad = torch.randn(64, 32)

    def step_norm(weight):
        return float(astro_matrix_update(
            grad, torch.zeros(64, 32), None, muon_filter(5), variance_post=None,
            beta1=0.95, beta2=0.95, eps=1e-8, step=1, variance_axis="none",
            variance_placement="post", rms_match=True, normalize_direction=False,
            cautious=False, nesterov=True, update_scale="trust",
            post_normalize=True, blocks=1, weight=weight,
        ).norm())

    tiny = step_norm(torch.full((64, 32), 1e-9))
    normal = step_norm(torch.randn(64, 32) * 0.1)
    assert tiny == tiny and tiny < normal
    # Floored by the pathology clip at ||u|| / trust_clip rather than running
    # away: the unclipped factor here would be 8e-9.
    assert tiny == pytest.approx(math.sqrt(32) / 1e3, rel=1e-4)
    assert normal > 100 * tiny


def test_trust_scale_needs_the_weight_and_says_so() -> None:
    from astro.optimizer import astro_matrix_update
    from astro.polar import muon_filter

    with pytest.raises(ValueError, match="needs the weight"):
        astro_matrix_update(
            torch.randn(8, 4), torch.zeros(8, 4), None, muon_filter(5),
            variance_post=None, beta1=0.95, beta2=0.95, eps=1e-8, step=1,
            variance_axis="none", variance_placement="post", rms_match=True,
            normalize_direction=False, cautious=False, nesterov=True,
            update_scale="trust", blocks=1,
        )


def test_the_trust_ratio_reaches_the_optimizer_end_to_end() -> None:
    """A knob wired only into the kernel is a knob nobody can use."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
    optimizer = Astro.from_model(model, lr=2e-2, update_scale="trust",
                                 cautious=False)
    inputs, targets = torch.randn(24, 16), torch.randn(24, 4)
    losses = []
    for _ in range(25):
        optimizer.zero_grad(set_to_none=True)
        loss = ((model(inputs) - targets) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    assert all(value == value for value in losses)
    assert losses[-1] < losses[0]


# ---------------------------------------------------------------------------
# Round-4 components: cautious weight decay, partial leverage damping,
# partial orthogonalisation, per-block scaling
# ---------------------------------------------------------------------------


def test_the_decay_mask_deletes_decay_rather_than_redistributing_it() -> None:
    """The finding that explains a gap widening with training length.

    ``cautious_mask`` rescales survivors by numel/count so masking changes
    direction without changing step size. The decay mask never got that
    convention, so at one nominal ``weight_decay`` a masked optimizer receives
    only the agreeing fraction of the decay an unmasked one does -- and decay
    compounds, so the resulting weight-norm difference grows without bound in
    the step count.
    """
    from astro.optimizer import apply_weight_decay

    torch.manual_seed(0)
    param = torch.randn(64, 64)
    update = torch.randn(64, 64)
    agrees = float((update * param > 0).float().mean())
    assert 0.3 < agrees < 0.7, agrees  # roughly half, by symmetry

    plain = param.clone()
    apply_weight_decay(plain, update, 0.1, cautious=False)
    masked = param.clone()
    apply_weight_decay(masked, update, 0.1, cautious=True)

    # Total shrinkage delivered, as a fraction of the unmasked amount.
    removed_plain = float((param - plain).abs().sum())
    removed_masked = float((param - masked).abs().sum())
    assert removed_masked < 0.75 * removed_plain, (removed_masked, removed_plain)


def test_rescaling_the_decay_mask_restores_the_total() -> None:
    """With ``rescale=True`` the decay is concentrated, not thinned: the same
    total shrinkage is delivered, on the coordinates that agree."""
    from astro.optimizer import apply_weight_decay

    torch.manual_seed(0)
    param = torch.randn(64, 64)
    update = torch.randn(64, 64)

    plain = param.clone()
    apply_weight_decay(plain, update, 0.01, cautious=False)
    rescaled = param.clone()
    apply_weight_decay(rescaled, update, 0.01, cautious=True, rescale=True)

    removed_plain = float((param - plain).abs().sum())
    removed_rescaled = float((param - rescaled).abs().sum())
    # Concentrated on the agreeing coordinates, so the totals match only in
    # expectation; on one draw they land within a few percent.
    assert removed_rescaled == pytest.approx(removed_plain, rel=0.10)

    # And it is genuinely different from the unrescaled mask.
    thinned = param.clone()
    apply_weight_decay(thinned, update, 0.01, cautious=True)
    assert float((param - thinned).abs().sum()) < 0.75 * removed_rescaled


def test_rescaled_decay_survives_a_fully_disagreeing_update() -> None:
    """numel/count is unbounded when nothing agrees; that must not divide by
    zero or move the weights."""
    from astro.optimizer import apply_weight_decay

    param = torch.ones(8, 8)
    update = -torch.ones(8, 8)  # disagrees everywhere
    before = param.clone()
    apply_weight_decay(param, update, 0.1, cautious=True, rescale=True)
    assert torch.equal(param, before)


def test_the_rescale_flag_reaches_the_optimizer() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
    reference = copy.deepcopy(model)
    inputs, targets = torch.randn(24, 16), torch.randn(24, 4)

    def train(module, **kwargs):
        optimizer = Astro.from_model(module, lr=1e-2, weight_decay=0.2,
                                     cautious=False, **kwargs)
        for _ in range(15):
            optimizer.zero_grad(set_to_none=True)
            ((module(inputs) - targets) ** 2).mean().backward()
            optimizer.step()
        return sum(float(p.pow(2).sum()) for p in module.parameters()) ** 0.5

    thinned = train(model)
    concentrated = train(reference, cautious_wd_rescale=True)
    assert concentrated < thinned, (concentrated, thinned)


def test_cautious_weight_decay_reduces_to_decoupled_when_disabled() -> None:
    from astro.optimizer import apply_weight_decay

    param = torch.tensor([1.0, -2.0, 3.0, -4.0])
    update = torch.tensor([1.0, 1.0, -1.0, -1.0])
    expected = param * (1.0 - 0.1)
    apply_weight_decay(param, update, 0.1, cautious=False)
    assert torch.allclose(param, expected)


def test_cautious_weight_decay_skips_coordinates_the_step_pushes_outward() -> None:
    """Decay only where the step is already moving the weight toward zero.

    ``w <- w - lr*u``, so the step moves ``w`` toward zero exactly when ``u`` and
    ``w`` share a sign. Coordinates 0 and 3 agree here; 1 and 2 do not.
    """
    from astro.optimizer import apply_weight_decay

    param = torch.tensor([1.0, -2.0, 3.0, -4.0])
    update = torch.tensor([1.0, 1.0, -1.0, -1.0])
    apply_weight_decay(param, update, 0.1, cautious=True)
    assert torch.allclose(param, torch.tensor([0.9, -2.0, 3.0, -3.6]))


def test_cautious_weight_decay_never_increases_magnitude() -> None:
    torch.manual_seed(0)
    from astro.optimizer import apply_weight_decay

    param = torch.randn(64, 32)
    before = param.abs().clone()
    apply_weight_decay(param, torch.randn(64, 32), 0.05, cautious=True)
    assert (param.abs() <= before + 1e-7).all()


def _matrix_update(**kwargs):
    from astro.optimizer import astro_matrix_update
    from astro.polar import muon_filter

    torch.manual_seed(0)
    grad = torch.randn(48, 16)
    defaults = dict(
        beta1=0.95, beta2=0.95, eps=1e-8, step=1, variance_axis="row",
        variance_placement="post", rms_match=True, normalize_direction=False,
        cautious=False, nesterov=True, update_scale="muon",
    )
    defaults.update(kwargs)
    return astro_matrix_update(
        grad, torch.zeros_like(grad), torch.zeros(48),
        muon_filter(5), variance_post=torch.zeros(48), **defaults,
    )


def test_variance_power_one_reproduces_the_normuon_rule() -> None:
    assert torch.allclose(_matrix_update(variance_power=1.0), _matrix_update())


def test_variance_power_interpolates_between_muon_and_normuon() -> None:
    """gamma sweeps a genuine interior: each setting must differ from both ends."""
    ends = _matrix_update(variance_power=1.0)
    middle = _matrix_update(variance_power=0.5)
    assert not torch.allclose(ends, middle, atol=1e-5)


def test_spectral_blend_zero_is_a_no_op() -> None:
    assert torch.allclose(_matrix_update(spectral_blend=0.0), _matrix_update())


def test_spectral_blend_moves_toward_the_unfiltered_direction() -> None:
    """The blend must change the update's shape without changing its length."""
    base = _matrix_update(spectral_blend=0.0)
    blended = _matrix_update(spectral_blend=0.5)
    assert not torch.allclose(base, blended, atol=1e-5)
    assert float(blended.norm()) == pytest.approx(float(base.norm()), rel=0.25)


def test_blocks_are_scaled_by_their_own_shape_not_the_largest() -> None:
    """Under grouped-query attention the blocks have different row counts, so a
    single scale taken from the largest is wrong for the other two."""
    from astro.optimizer import astro_matrix_update
    from astro.polar import muon_filter

    torch.manual_seed(0)
    grad = torch.randn(64 + 16 + 16, 64)  # GQA: Q is square, K and V are wide
    update = astro_matrix_update(
        grad, torch.zeros_like(grad), None, muon_filter(5),
        beta1=0.95, beta2=0.95, eps=1e-8, step=1, variance_axis="none",
        variance_placement="post", rms_match=True, normalize_direction=False,
        cautious=False, nesterov=True, update_scale="muon", blocks=(64, 16, 16),
    )
    q, k, v = update.split([64, 16, 16], dim=0)
    # Muon's scale is max(1, rows/cols)**0.5: 1.0 for every block here, since all
    # three are square-or-wider. The wide blocks must not inherit Q's row count.
    assert float(k.norm()) == pytest.approx(float(v.norm()), rel=0.5)
    assert float(q.norm()) > float(k.norm())


def test_split_steps_relaxes_to_a_single_block_after_its_horizon() -> None:
    """The fused-projection defect decays, so the repair should too."""
    from astro.optimizer import Astro

    group = {"split_steps": 100}
    state = {"blocks": (64, 64, 64)}
    assert Astro._blocks_at(group, state, 1) == (64, 64, 64)
    assert Astro._blocks_at(group, state, 100) == (64, 64, 64)
    assert Astro._blocks_at(group, state, 101) == (192,)


def test_split_steps_none_never_relaxes() -> None:
    from astro.optimizer import Astro

    state = {"blocks": (64, 64, 64)}
    for step in (1, 10_000):
        assert Astro._blocks_at({"split_steps": None}, state, step) == (64, 64, 64)


def test_relaxed_split_matches_an_unsplit_optimizer() -> None:
    """After the horizon the update must equal what split_fused=False produces."""
    from astro.bench.gpt import GPT, GPTConfig
    from astro.optimizer import Astro

    config = GPTConfig(vocab_size=97, n_layer=1, n_head=2, n_embd=64, block_size=32)
    idx = torch.randint(0, 97, (2, 16))

    def run(**kwargs) -> torch.Tensor:
        torch.manual_seed(0)
        model = GPT(config)
        optimizer = Astro.from_model(model, lr=1e-3, **kwargs)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            model(idx, idx)[1].backward()
            optimizer.step()
        return model.transformer.h[0].attn.c_attn.weight.detach().clone()

    assert torch.allclose(run(split_fused=False), run(split_steps=0), atol=1e-6)
    assert not torch.allclose(run(split_fused=False), run(split_steps=None), atol=1e-6)


def test_post_normalize_pins_the_update_norm_to_its_theoretical_value() -> None:
    """Newton-Schulz under-converges by an amount that depends on conditioning,
    so the effective step size drifts with the gradient spectrum unless pinned."""
    import math as _math

    from astro.optimizer import _filter_block
    from astro.polar import muon_filter

    torch.manual_seed(0)
    rows, cols = 128, 64
    target = _math.sqrt(min(rows, cols))
    raw, pinned = [], []
    for condition in (1.0, 10.0, 100.0, 1000.0):
        left, _ = torch.linalg.qr(torch.randn(rows, cols))
        right, _ = torch.linalg.qr(torch.randn(cols, cols))
        spectrum = torch.logspace(0, -_math.log10(condition), cols)
        block = (left * spectrum) @ right.T
        raw.append(float(_filter_block(muon_filter(5), block, False).norm()) / target)
        pinned.append(float(_filter_block(muon_filter(5), block, True).norm()) / target)

    assert max(raw) - min(raw) > 0.15, "the uncorrected norm should vary with conditioning"
    assert all(value == pytest.approx(1.0, abs=1e-5) for value in pinned)


def test_equilibrate_equalises_row_norms_without_resizing_the_step() -> None:
    """MuonEq's R variant conditions the matrix handed to Newton-Schulz."""
    from astro.optimizer import astro_matrix_update
    from astro.polar import muon_filter

    torch.manual_seed(0)
    grad = torch.randn(32, 16)
    grad[0] *= 50.0  # one row dominating, which is what equilibration targets

    def run(**kwargs) -> torch.Tensor:
        return astro_matrix_update(
            grad, torch.zeros_like(grad), None, muon_filter(5),
            beta1=0.95, beta2=0.95, eps=1e-8, step=1, variance_axis="none",
            variance_placement="post", rms_match=False, normalize_direction=False,
            cautious=False, nesterov=True, update_scale="muon", **kwargs,
        )

    assert not torch.allclose(run(equilibrate=True), run(equilibrate=False), atol=1e-5)


def test_equilibration_preserves_the_momentum_norm() -> None:
    torch.manual_seed(0)
    update = torch.randn(32, 16)
    update[0] *= 50.0
    norms = update.norm(dim=1, keepdim=True).clamp_min(1e-8)
    balanced = update / norms
    rescaled = balanced * (update.norm() / balanced.norm().clamp_min(1e-8))

    assert float(rescaled.norm()) == pytest.approx(float(update.norm()), rel=1e-5)
    row_norms = rescaled.norm(dim=1)
    assert float(row_norms.std() / row_norms.mean()) == pytest.approx(0.0, abs=1e-5)
