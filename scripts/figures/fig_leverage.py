"""Figure 1: the row norms of a Muon update are leverage scores.

Runs on CPU in seconds and needs no training run, no checkpoint and no GPU.
That is the point of it: the claim is an identity, so the figure that supports
it is a verification rather than an experiment, and a reader can re-run it.

Three panels:
  (a) squared row norms of the polar factor against leverage scores, on y = x
  (b) the identity's first corollary -- on wide matrices every row norm is
      exactly 1, so row normalisation has nothing to act on
  (c) its second corollary -- a fused QKV projection routes update mass to
      whichever block has the largest gradient, and splitting removes it
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from style import (
    COLORS,
    annotate_value,
    figure,
    grid,
    label_panels,
    reference_line,
    save,
    write_data,
)


def polar_factor(matrix: torch.Tensor) -> torch.Tensor:
    """UV^T by exact SVD.

    The figure is about the identity, not about Newton-Schulz, so it uses the
    exact polar factor. fig_quintic.py measures what the iteration does to it.
    """
    u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    return u @ vh


def leverage_scores(matrix: torch.Tensor) -> torch.Tensor:
    """Leverage of each row: the squared norm of its row in the left factor."""
    u, _, _ = torch.linalg.svd(matrix, full_matrices=False)
    return (u ** 2).sum(dim=1)


# ---------------------------------------------------------------------------
# (a) the identity
# ---------------------------------------------------------------------------


def measure_identity(rows: int = 256, cols: int = 64, seed: int = 0):
    """Squared row norms of UV^T against leverage scores, on one tall matrix.

    The matrix is given a decaying spectrum and a non-uniform row scaling so
    the leverage profile is genuinely spread out; on an isotropic Gaussian
    every score sits near n/m and the panel would show nothing.
    """
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(rows, cols, generator=generator, dtype=torch.float64)
    spectrum = torch.logspace(0, -2, cols, dtype=torch.float64)
    row_scale = torch.exp(torch.linspace(0.0, 2.5, rows, dtype=torch.float64))
    matrix = row_scale[:, None] * base * spectrum[None, :]

    factor = polar_factor(matrix)
    measured = (factor ** 2).sum(dim=1)
    predicted = leverage_scores(matrix)
    return {
        "measured": measured.numpy(),
        "predicted": predicted.numpy(),
        "max_abs_error": float((measured - predicted).abs().max()),
        "sum_measured": float(measured.sum()),
        "cols": cols,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# (b) inert on wide matrices
# ---------------------------------------------------------------------------


def measure_aspect(width: int = 128, ratios=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
                   seed: int = 0):
    """Spread of the update's row norms as the matrix goes wide to tall.

    For m <= n the left factor is square orthogonal, so every row of UV^T has
    norm exactly 1 and the spread is 0 to machine precision. That is the
    statement that row normalisation is inert, drawn rather than asserted.
    """
    generator = torch.Generator().manual_seed(seed)
    spread, concentration = [], []
    for ratio in ratios:
        rows = max(2, int(round(width * ratio)))
        base = torch.randn(rows, width, generator=generator, dtype=torch.float64)
        scale = torch.exp(torch.linspace(0.0, 2.5, rows, dtype=torch.float64))
        norms = (polar_factor(scale[:, None] * base) ** 2).sum(dim=1)
        spread.append(float(norms.max() - norms.min()))
        # Fraction of the total update mass carried by the heaviest tenth of
        # rows: 0.1 when uniform, higher when leverage localises.
        top = max(1, rows // 10)
        concentration.append(float(norms.sort(descending=True).values[:top].sum()
                                   / norms.sum()))
    return {"ratios": list(ratios), "spread": spread,
            "concentration": concentration, "width": width}


# ---------------------------------------------------------------------------
# (c) the fused-projection defect
# ---------------------------------------------------------------------------


def measure_fusion(width: int = 128, imbalances=None, seed: int = 0):
    """Update-mass share of Q, K and V as their gradient scales diverge.

    ``imbalance`` is the factor by which the Q and K gradient blocks are
    smaller than V's -- the situation at initialisation, where Q and K reach
    the loss through the softmax Jacobian and V enters linearly.
    """
    if imbalances is None:
        imbalances = np.logspace(0, -2, 13)
    generator = torch.Generator().manual_seed(seed)
    blocks = [torch.randn(width, width, generator=generator, dtype=torch.float64)
              for _ in range(3)]

    fused_shares, split_shares = [], []
    for imbalance in imbalances:
        scaled = [blocks[0] * imbalance, blocks[1] * imbalance, blocks[2]]

        fused = polar_factor(torch.cat(scaled, dim=0))
        parts = [(part ** 2).sum() for part in fused.split(width, dim=0)]
        total = sum(parts)
        fused_shares.append([float(part / total) for part in parts])

        split = [polar_factor(block) for block in scaled]
        parts = [(part ** 2).sum() for part in split]
        total = sum(parts)
        split_shares.append([float(part / total) for part in parts])

    return {"imbalances": [float(value) for value in imbalances],
            "fused": fused_shares, "split": split_shares, "width": width}


# ---------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------


def build(seed: int = 0) -> None:
    identity = measure_identity(seed=seed)
    aspect = measure_aspect(seed=seed)
    fusion = measure_fusion(seed=seed)

    fig, axes = figure(3, height=2.15)
    left, middle, right = axes

    # (a) --------------------------------------------------------------
    left.scatter(identity["predicted"], identity["measured"], s=4,
                 color=COLORS["muon"], alpha=0.55, linewidths=0, zorder=3)
    span = [float(identity["predicted"].min()), float(identity["predicted"].max())]
    left.plot(span, span, color=COLORS["reference"], linestyle="--",
              linewidth=0.8, zorder=2, label="$y = x$")
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel(r"leverage score $\|u_i\|^2$")
    left.set_ylabel(r"update row mass $\|p_i\|^2$")
    left.legend(loc="upper left")
    grid(left)
    left.text(0.97, 0.06,
              f"max error {identity['max_abs_error']:.1e}\n"
              rf"$\sum_i\|p_i\|^2 = {identity['sum_measured']:.4f}$ ($n={identity['cols']}$)",
              transform=left.transAxes, ha="right", va="bottom", fontsize=6.5)

    # (b) --------------------------------------------------------------
    ratios = np.asarray(aspect["ratios"])
    spread = np.asarray(aspect["spread"])
    floor = 1e-17
    middle.plot(ratios, np.maximum(spread, floor), color=COLORS["normuon"],
                marker="o", zorder=3)
    middle.set_xscale("log", base=2)
    middle.set_yscale("log")
    middle.set_ylim(floor / 3, 10)
    middle.axvspan(ratios.min() / 1.5, 1.0, color=COLORS["normuon"],
                   alpha=0.08, zorder=0)
    middle.axvline(1.0, color=COLORS["reference"], linestyle="--", linewidth=0.8)
    middle.set_xlabel(r"aspect ratio $m/n$")
    middle.set_ylabel(r"spread of $\|p_i\|^2$")
    middle.annotate("wide: row norms\nexactly 1", (0.4, 1e-13),
                    fontsize=6.5, ha="center", color=COLORS["normuon"])
    middle.annotate("tall: leverage\nspreads", (4.0, 1e-3), fontsize=6.5,
                    ha="center", color=COLORS["reference"])
    grid(middle)

    # (c) --------------------------------------------------------------
    imbalance = np.asarray(fusion["imbalances"])
    fused = np.asarray(fusion["fused"])
    split = np.asarray(fusion["split"])
    right.plot(imbalance, fused[:, 2], color=COLORS["muon"], marker="s",
               label="fused, $V$", zorder=3)
    right.plot(imbalance, fused[:, 0], color=COLORS["muon"], marker="s",
               linestyle=":", label="fused, $Q$", zorder=3)
    right.plot(imbalance, split[:, 2], color=COLORS["astro"], marker="D",
               label="split, $V$", zorder=3)
    right.set_xscale("log")
    right.invert_xaxis()
    reference_line(right, 1 / 3, "uniform $1/3$")
    right.set_ylim(0, 1.02)
    right.set_xlabel(r"$Q,K$ gradient scale relative to $V$")
    right.set_ylabel("share of update mass")
    right.legend(loc="center left")
    grid(right)
    annotate_value(right, imbalance[-1], fused[-1, 2],
                   f"{fused[-1, 2]:.2f}", dx=-5, dy=-10, ha="right",
                   color=COLORS["muon"])
    annotate_value(right, imbalance[-1], split[-1, 2],
                   f"{split[-1, 2]:.2f}", dx=-5, dy=-11, ha="right",
                   color=COLORS["astro"])

    label_panels(axes)
    fig.tight_layout(pad=0.4, w_pad=1.4)
    path = save(fig, "fig1_leverage")

    write_data("fig1_leverage", {
        "identity": {key: value for key, value in identity.items()
                     if key not in {"measured", "predicted"}},
        "aspect": aspect,
        "fusion": fusion,
        "seed": seed,
    })

    print(f"  wrote {path}")
    print(f"    identity holds to {identity['max_abs_error']:.2e}; "
          f"row masses sum to {identity['sum_measured']:.6f} "
          f"(n = {identity['cols']})")
    print(f"    wide-matrix spread: {aspect['spread'][0]:.2e} at m/n = "
          f"{aspect['ratios'][0]}")
    print(f"    fused V share at {imbalance[-1]:.2g}x imbalance: "
          f"{fused[-1, 2]:.4f}  (split: {split[-1, 2]:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    build(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
