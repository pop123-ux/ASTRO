"""Search spaces for every optimizer in the comparison.

**Every optimizer tunes exactly three hyperparameters.** That constraint is the
whole point of this module and :func:`~astro.bench.protocol.tune` enforces
it. The commonest way to manufacture an optimizer win is to sweep the proposed
method over three knobs and the baseline over one, so the number is fixed here
once and applies to everyone.

Everything else is pinned to the value its own authors published, listed in
``fixed`` so the choice is visible rather than buried in a default argument.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import Optimizer

from astro.baselines import SOAP, AdEMAMix, CautiousAdamW, Muon, NorMuon
from astro.bench.protocol import SearchSpace
from astro.optimizer import Astro

__all__ = [
    "build_spaces",
    "build_ablation_spaces",
    "build_candidate_spaces",
    "BASELINE_NAMES",
    "TUNED_DIMENSIONS",
]

#: Number of hyperparameters every optimizer is allowed to tune.
TUNED_DIMENSIONS = 3

BASELINE_NAMES = ("adamw", "sgd", "muon", "normuon", "soap", "ademamix", "cautious")


def _adamw(model: nn.Module, config: dict[str, float]) -> Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        betas=(0.9, config["beta2"]),
    )


def _sgd(model: nn.Module, config: dict[str, float]) -> Optimizer:
    return torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        nesterov=True,
    )


def _muon(model: nn.Module, config: dict[str, float]) -> Optimizer:
    return Muon.from_model(
        model,
        lr=config["lr"],
        adamw_lr=config["adamw_lr"],
        weight_decay=config["weight_decay"],
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
    )


def _normuon(model: nn.Module, config: dict[str, float]) -> Optimizer:
    return NorMuon.from_model(
        model,
        lr=config["lr"],
        adamw_lr=config["adamw_lr"],
        weight_decay=config["weight_decay"],
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
    )


def _soap(model: nn.Module, config: dict[str, float]) -> Optimizer:
    return SOAP.from_model(
        model,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        betas=(0.95, config["beta2"]),
        shampoo_beta=0.95,
        precondition_frequency=10,
    )


def _ademamix(model: nn.Module, config: dict[str, float]) -> Optimizer:
    return AdEMAMix(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        alpha=config["alpha"],
        betas=(0.9, 0.999, 0.9999),
        beta3_warmup=50,
        alpha_warmup=50,
    )


def _cautious(model: nn.Module, config: dict[str, float]) -> Optimizer:
    return CautiousAdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        betas=(0.9, config["beta2"]),
    )


def _astro_builder(**overrides: object):
    def build(model: nn.Module, config: dict[str, float]) -> Optimizer:
        kwargs = dict(
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 0.0),
            betas=(0.9, 0.95),
            variance="row",
            norm_control="wd",
            anchor=False,
            ns_steps=5,
            trust_scope="global",
        )
        kwargs.update(overrides)  # type: ignore[arg-type]
        if "rho" in config:
            kwargs["rho"] = config["rho"]
        if "anchor_strength" in config:
            kwargs["anchor_strength"] = config["anchor_strength"]
        if "beta2" in config:
            kwargs["betas"] = (0.9, config["beta2"])
        # Muon tunes a separate learning rate for its AdamW path, and on the
        # from-scratch convnet it picks a 3.6x ratio between the two. Forcing
        # ASTRO to share one rate across both paths was a handicap of this
        # harness rather than a property of the algorithm, so it gets the same
        # two-rate structure -- still three tuned knobs in total.
        scalar_mult = config.get("scalar_lr_mult", 1.0)
        return Astro.from_model(  # type: ignore[arg-type]
            model, scalar_lr_mult=scalar_mult, **kwargs
        )

    return build


#: The configuration round 3 selected, written out in full. It is also the
#: shipped default now, but round-4 variants state it explicitly so the
#: experiment does not change meaning if a default moves under it.
_ROUND3_WINNER: dict[str, object] = {
    "split_fused": True,
    "variance_placement": "post",
    "nesterov": True,
    "betas": (0.95, 0.95),
    "update_scale": "muon",
}


def build_spaces(*, finetuning: bool) -> list[SearchSpace]:
    """Search spaces for the comparison.

    Parameters
    ----------
    finetuning:
        When true, ASTRO enables its anchored trust region and tunes ``rho`` as
        its third hyperparameter. When false there are no pretrained weights to
        protect, the anchor is disabled, and the third knob is ``beta2`` instead
        -- so the budget stays at three either way.
    """
    spaces = [
        SearchSpace(
            name="adamw",
            build=_adamw,
            ranges={"lr": (1e-4, 3e-2), "weight_decay": (1e-4, 3e-1), "beta2": (0.9, 0.999)},
        ),
        SearchSpace(
            name="sgd",
            build=_sgd,
            ranges={"lr": (1e-3, 1.0), "momentum": (0.5, 0.99), "weight_decay": (1e-5, 1e-1)},
        ),
        SearchSpace(
            name="muon",
            build=_muon,
            ranges={"lr": (1e-3, 3e-1), "adamw_lr": (1e-4, 3e-2), "weight_decay": (1e-5, 1e-1)},
            fixed={"momentum": 0.95},
        ),
        SearchSpace(
            name="normuon",
            build=_normuon,
            ranges={"lr": (1e-3, 3e-1), "adamw_lr": (1e-4, 3e-2), "weight_decay": (1e-5, 1e-1)},
            fixed={"momentum": 0.95},
        ),
        SearchSpace(
            name="soap",
            build=_soap,
            ranges={"lr": (1e-4, 3e-2), "weight_decay": (1e-5, 3e-1), "beta2": (0.9, 0.999)},
            fixed={"shampoo_beta": 0.95, "precondition_frequency": 10},
        ),
        SearchSpace(
            name="ademamix",
            build=_ademamix,
            ranges={"lr": (1e-4, 3e-2), "weight_decay": (1e-5, 1e-1), "alpha": (1.0, 15.0)},
            fixed={"beta3": 0.9999},
        ),
        SearchSpace(
            name="cautious",
            build=_cautious,
            ranges={"lr": (1e-4, 3e-2), "weight_decay": (1e-5, 1e-1), "beta2": (0.9, 0.999)},
        ),
    ]

    if finetuning:
        # Fine-tuning spends its third knob on the anchor's restoring strength,
        # so weight decay is pinned rather than swept. Declared in ``fixed`` so
        # the choice is visible instead of buried in a default argument.
        #
        # The range reaches down to 1e-4, which is effectively no anchor at all:
        # the tuner is free to switch the component off, and on the earlier hard
        # variant it did exactly that (selecting rho = 0.77 of 1.0). Leaving that
        # escape open is what makes a positive result mean something.
        astro_ranges = {
            "lr": (1e-4, 3e-2),
            "scalar_lr_mult": (0.1, 10.0),
            "anchor_strength": (1e-4, 3.0),
        }
        astro_overrides: dict[str, object] = {"anchor": True, "anchor_mode": "elastic"}
        astro_fixed = {"beta1": 0.9, "weight_decay": 1e-2}
    else:
        # ASTRO's shipped recipe sizes its update like Muon's, so it gets Muon's
        # learning-rate range. Pairing a Muon-scaled update with Adam's range is
        # the round-2 harness bug, and it would have returned silently here the
        # moment the default changed -- so the recipe is written out rather than
        # inherited, and the range follows from it.
        astro_ranges = {
            "lr": (1e-3, 3e-1),
            "scalar_lr_mult": (0.02, 3.0),
            "weight_decay": (1e-5, 1e-1),
        }
        astro_overrides = {"anchor": False, **_ROUND3_WINNER}
        astro_fixed = {"beta1": 0.95, "beta2": 0.95}

    spaces.append(
        SearchSpace(
            name="astro",
            build=_astro_builder(**astro_overrides),
            ranges=astro_ranges,
            fixed=astro_fixed,
        )
    )
    return spaces


def build_ablation_spaces(*, finetuning: bool) -> list[SearchSpace]:
    """ASTRO variants with one component disabled each.

    Each keeps three tuned dimensions so it is directly comparable to the full
    method under the same budget. ``astro_no_variance`` with the anchor off is
    Muon plus RMS matching, which is why it doubles as a reduction test.
    """
    two = {"lr": (1e-4, 3e-2), "scalar_lr_mult": (0.1, 10.0)}
    base: dict[str, object] = {"anchor": finetuning, "anchor_mode": "elastic"}

    variants: dict[str, dict[str, object]] = {
        "astro_full": {},
        # -- components carried over from v1 --
        "astro_no_variance": {"variance": "none"},
        "astro_no_anchor": {"anchor": False},
        "astro_no_rms": {"rms_match": False},
        "astro_hyperball": {"norm_control": "hyperball"},
        # The dead-zone filter suppresses the small-singular-value tail instead of
        # lifting it to one. It costs 10 iterations rather than 5, so it has to
        # earn that on quality; the ablation is where it does or does not.
        "astro_deadzone": {"dead_zone": 0.1, "ns_steps": 10},
        # -- components introduced in v2, each reverting to what preceded it --
        "astro_no_cautious": {"cautious": False},
        "astro_variance_pre": {"variance_placement": "pre"},
        "astro_variance_both": {"variance_placement": "both"},
        "astro_anchor_hard": {"anchor_mode": "hard", "rho_warmup_steps": 10},
    }

    spaces = []
    for name, overrides in variants.items():
        merged = {**base, **overrides}
        local = dict(two)
        # Every variant must tune exactly three dimensions, and the third has to
        # be one that actually does something for that variant -- tuning an inert
        # knob wastes the budget and quietly handicaps the variant against the
        # others. So the third knob follows the anchor configuration.
        if not merged.get("anchor", False):
            local["weight_decay"] = (1e-5, 1e-1)
        elif merged.get("anchor_mode") == "hard":
            local["rho"] = (2e-3, 1.0)
        else:
            local["anchor_strength"] = (1e-4, 3.0)
        spaces.append(SearchSpace(name=name, build=_astro_builder(**merged), ranges=local))
    return spaces


def build_candidate_spaces(*, finetuning: bool) -> list[SearchSpace]:
    """ASTRO configurations proposed *after* the field was measured.

    These are candidate improvements, kept separate from :func:`build_spaces` so
    that the headline comparison stays exactly what it was when it was run. A
    candidate tested after seeing the field's results is a different kind of
    evidence from one specified beforehand, and mixing them in one table would
    hide that distinction.

    Each still tunes three hyperparameters, from the same ranges and the same RNG
    stream the field received, so nothing here wins by searching harder.

    The motivating observation: on ``gpt_scratch`` NorMuon (post-orthogonalisation
    neuron normalisation, no cautious mask) beat both Muon and ASTRO, while on
    ``gpt_finetune`` the ablation showed cautious masking worth 0.7%. Those two
    ingredients have never been combined -- ``astro_post`` is that combination.
    """
    third = ("weight_decay", (1e-5, 1e-1)) if not finetuning else ("anchor_strength", (1e-4, 3.0))

    # The learning-rate range must follow the *update scale*, not the optimizer's
    # name. ASTRO's default RMS matching sizes its update like an Adam update, so
    # Adam's range is the right one; ``update_scale="muon"`` sizes it like a Muon
    # update, and Muon's range is. Round 2 got this wrong: the NorMuon replica ran
    # Muon's update rule while searching Adam's range, whose ceiling (0.03) sits
    # barely above the 0.0198 NorMuon itself tuned to, with log-uniform sampling
    # putting most of its mass far below. The resulting 0.034 shortfall was a
    # handicap, not a property of the algorithm.
    #
    # This is Wen et al.'s failure mode in a form the "equal number of tuned
    # hyperparameters" rule does not catch: equal counts, incomparable ranges.
    adam_lr = (1e-4, 3e-2)
    muon_lr = (1e-3, 3e-1)

    # ``_astro_builder`` pins every recipe field explicitly rather than
    # inheriting it, and the range is chosen from the *pinned* value. Reading a
    # class default here would silently reopen the round-2 bug the moment that
    # default changed -- which it since has.
    def ranges_for(overrides: dict[str, object]) -> dict[str, tuple[float, float]]:
        scale = overrides.get("update_scale", "adam_rms")
        if scale not in ("adam_rms", "muon"):
            raise ValueError(f"unknown update_scale {scale!r}")
        return {
            "lr": muon_lr if scale == "muon" else adam_lr,
            # A Muon-scaled update is 2-9x smaller in Frobenius norm than an
            # Adam-like one at the same rate, and the gap grows with width, so
            # the scalar path wants a *smaller* multiplier -- Muon's own
            # convention is adamw_lr = lr/10. The old range (0.1, 10.0) put that
            # recommendation on its lower boundary, allowing 10x too high and
            # nothing below the value theory points at. Same family of bug as
            # the round-2 range error: a limit that silently excludes the answer.
            "scalar_lr_mult": (0.02, 3.0),
            third[0]: third[1],
        }

    base: dict[str, object] = {"anchor": finetuning, "anchor_mode": "elastic"}

    variants: dict[str, dict[str, object]] = {
        # -- round 1: NorMuon's placement plus ASTRO's cautious mask ---------
        "astro_post": {"variance_placement": "post"},
        "astro_post_nocautious": {"variance_placement": "post", "cautious": False},
        "astro_both": {"variance_placement": "both"},
        # -- round 2 -----------------------------------------------------------
        # Round 1 showed post-placement alone does not recover NorMuon's margin.
        # astro_post_nocautious differs from NorMuon in exactly two ways -- the
        # momentum rule and the update scale -- and trails it by 0.055, so those
        # two are what the margin is made of. Each is isolated, then combined.
        "astro_nesterov": {"nesterov": True, "betas": (0.95, 0.95)},
        "astro_muonscale": {"update_scale": "muon"},
        "astro_nesterov_muonscale": {
            "nesterov": True, "betas": (0.95, 0.95), "update_scale": "muon",
        },
        # NorMuon's full recipe plus the cautious mask: the combination the
        # evidence points at, and the one nobody has run.
        "astro_normuon_cautious": {
            "variance_placement": "post", "nesterov": True, "betas": (0.95, 0.95),
            "update_scale": "muon",
        },
        # The same without the mask, which should reproduce NorMuon closely. If
        # it does not, the remaining gap is an implementation difference rather
        # than an algorithmic one, and that is worth knowing before claiming a win.
        "astro_normuon_replica": {
            "variance_placement": "post", "nesterov": True, "betas": (0.95, 0.95),
            "update_scale": "muon", "cautious": False,
        },
        # -- round 3: split the fused QKV projection ---------------------------
        # Measured on nanoGPT's GPT-2: orthogonalising the fused (3d, d) c_attn
        # tensor as one matrix puts 85% of the polar factor's squared row-norm
        # mass on V, against 9% for K and 5% for Q, because Q and K reach the
        # loss through the softmax Jacobian while V enters linearly and so has a
        # ~70x larger gradient. Splitting into three blocks and orthogonalising
        # each restores 33/33/33. This is a cause; NorMuon's row normalisation
        # is a partial treatment of the symptom, so the two should overlap.
        "astro_split": {"split_fused": True},
        "astro_nosplit": {"split_fused": False},
        "astro_split_post": {"split_fused": True, "variance_placement": "post"},
        "astro_split_normuon": {
            "split_fused": True, "variance_placement": "post", "nesterov": True,
            "betas": (0.95, 0.95), "update_scale": "muon",
        },
        # -- round 4: partial leverage damping and partial orthogonalisation ---
        # Round 3's winner is now the shipped default, so every variant below is
        # that recipe plus exactly one new knob, and ``astro_v2`` is the control
        # that isolates the knob from the recipe.
        #
        # ``variance_power`` (gamma) is a family of dual maps, not a tuning knob.
        # Muon's update is steepest descent under the spectral norm; NorMuon and
        # Muown push toward a row-wise (l-infinity) geometry in which every
        # neuron takes an equal step. Proposition 1 says the quantity separating
        # them is the leverage profile, and gamma is a continuous interpolation
        # between the two induced norms: 0 is Muon, 1 is NorMuon. Both endpoints
        # are published and neither paper searches between them. Sampled at five
        # points because a curve with an interior minimum is a claim about
        # geometry, while two points are a coin flip.
        #
        # ``spectral_blend`` (alpha) mixes the unfiltered direction back in.
        # Ma et al.'s river-valley analysis finds Muon converges more slowly than
        # gradient descent near the optimum because a constant-magnitude update
        # cannot take a small step; the blend restores that ability at the cost
        # of some spectral normalisation.
        "astro_v2": {**_ROUND3_WINNER},
        "astro_gamma00": {**_ROUND3_WINNER, "variance_power": 0.0},
        "astro_gamma25": {**_ROUND3_WINNER, "variance_power": 0.25},
        "astro_gamma50": {**_ROUND3_WINNER, "variance_power": 0.5},
        "astro_gamma75": {**_ROUND3_WINNER, "variance_power": 0.75},
        "astro_blend20": {**_ROUND3_WINNER, "spectral_blend": 0.2},
        "astro_blend40": {**_ROUND3_WINNER, "spectral_blend": 0.4},
        # Newton-Schulz stops at its own fixed point, not at the polar factor, so
        # the update's Frobenius norm lands somewhere in 0.69-0.92 of
        # sqrt(min(m,n)) depending on the conditioning of what it was handed.
        # That is a 34% drift in effective step size driven by a quantity nobody
        # tracks. Pinning it costs one norm per block.
        "astro_pinned": {**_ROUND3_WINNER, "post_normalize": True},
        # Cautious weight decay is on by default, so the ablation needs the
        # variant with it removed rather than added.
        "astro_plain_wd": {**_ROUND3_WINNER, "cautious_wd": False},
        # The defect the split repairs is an initialisation-time phenomenon;
        # 120 of 400 steps is the first 30% of training.
        "astro_split120": {**_ROUND3_WINNER, "split_steps": 120},
        # The two most promising single knobs, combined.
        "astro_pinned_gamma50": {
            **_ROUND3_WINNER, "post_normalize": True, "variance_power": 0.5,
        },
        # MuonEq's R variant: instantaneous row equilibration of the momentum
        # before the polar step, which its authors report as consistently
        # beating Muon. Distinct from our pre-placement, which is an EMA of the
        # gradient rather than the momentum's current row norms.
        "astro_equil": {**_ROUND3_WINNER, "equilibrate": True},
        # A converging polar iteration. Muon's quintic has fixed points at 0.868
        # and 1.264 and so cannot reach sigma = 1 at any budget; solved per-step
        # coefficients can. Same cost.
        "astro_converging": {**_ROUND3_WINNER, "converging": True},
        # Two extra steps reach the polar factor essentially exactly. Whether
        # that *helps* is a separate question: Muon's under-convergence may be
        # acting as useful damping, and 7 steps costs 40% more than 5.
        "astro_converging7": {**_ROUND3_WINNER, "converging": True, "ns_steps": 7},
    }
    return [
        SearchSpace(
            name=name,
            build=_astro_builder(**{**base, **overrides}),
            ranges=ranges_for(overrides),
        )
        for name, overrides in variants.items()
    ]
