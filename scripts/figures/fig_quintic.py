"""Figure 2: Muon's quintic cannot reach the polar factor at any budget.

CPU, seconds, no training. The claim is about a fixed polynomial, so the figure
is a verification of an algebraic fact plus its measurable consequence for the
size of the step an optimizer actually takes.

  (a) the scalar map with its fixed points, and the orbit that gets stuck there
  (b) the singular-value band after 5, 8 and 12 iterations -- more steps do not
      narrow it, which is what "cannot converge" means operationally
  (c) the consequence: the update's Frobenius norm drifts with the conditioning
      of the matrix handed over, and the solved schedule removes the drift
"""

from __future__ import annotations

import argparse
import sys
from itertools import cycle, pairwise
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from style import (  # noqa: E402
    COLORS,
    annotate_value,
    figure,
    grid,
    label_panels,
    reference_line,
    save,
    write_data,
)

from astro.polar import MUON_QUINTIC, muon_filter, polar_filter  # noqa: E402

ASTRO_SHADES = ["#f4a3a3", "#e05c5c", COLORS["astro"]]


def quintic(sigma: np.ndarray, coefficients=MUON_QUINTIC) -> np.ndarray:
    a, b, c = coefficients
    return a * sigma + b * sigma ** 3 + c * sigma ** 5


def fixed_points(coefficients=MUON_QUINTIC) -> list[float]:
    """Solve p(s) = s for s > 0, i.e. a - 1 + b s^2 + c s^4 = 0."""
    a, b, c = coefficients
    roots = np.roots([c, 0.0, b, 0.0, a - 1.0])
    real = sorted({float(root.real) for root in roots
                   if abs(root.imag) < 1e-9 and root.real > 1e-9})
    return real


def orbit(start: float, steps: int, coefficients=MUON_QUINTIC) -> list[float]:
    value, trail = start, [start]
    for _ in range(steps):
        value = float(quintic(np.asarray(value), coefficients))
        trail.append(value)
    return trail


# ---------------------------------------------------------------------------
# (b) the band does not narrow
# ---------------------------------------------------------------------------


def measure_band(rows: int = 256, cols: int = 128, budgets=(5, 8, 12),
                 condition: float = 100.0, seed: int = 0):
    """Singular values of the filtered matrix at several iteration counts."""
    generator = torch.Generator().manual_seed(seed)
    left = torch.linalg.qr(torch.randn(rows, cols, generator=generator,
                                       dtype=torch.float64))[0]
    right = torch.linalg.qr(torch.randn(cols, cols, generator=generator,
                                        dtype=torch.float64))[0]
    spectrum = torch.logspace(0, -np.log10(condition), cols, dtype=torch.float64)
    matrix = left @ torch.diag(spectrum) @ right.T

    out = {}
    for steps in budgets:
        filtered = muon_filter(steps)(matrix.clone())
        values = torch.linalg.svdvals(filtered).numpy()
        out[steps] = values
    return out, spectrum.numpy()


# ---------------------------------------------------------------------------
# (c) the drift this causes, and its removal
# ---------------------------------------------------------------------------


def measure_drift(rows: int = 128, cols: int = 64,
                  conditions=(1.0, 10.0, 100.0, 1000.0), seed: int = 0):
    """Update norm as a fraction of its theoretical value sqrt(min(m, n)).

    A spectral update is supposed to have Frobenius norm sqrt(min(m, n)); the
    optimizer's step size is calibrated on that assumption. What the iteration
    actually delivers depends on the conditioning of the momentum it was
    handed, which no one tracks and which moves during training.
    """
    generator = torch.Generator().manual_seed(seed)
    left = torch.linalg.qr(torch.randn(rows, cols, generator=generator,
                                       dtype=torch.float64))[0]
    right = torch.linalg.qr(torch.randn(cols, cols, generator=generator,
                                        dtype=torch.float64))[0]
    target = np.sqrt(min(rows, cols))

    filters = {"muon5": muon_filter(5), "polar5": polar_filter(5),
               "polar6": polar_filter(6), "polar7": polar_filter(7)}
    out = {name: [] for name in filters}
    for condition in conditions:
        spectrum = torch.logspace(0, -np.log10(condition), cols, dtype=torch.float64)
        matrix = left @ torch.diag(spectrum) @ right.T
        for name, spectral in filters.items():
            norm = float(torch.linalg.norm(spectral(matrix.clone())))
            out[name].append(norm / target)
    return out, list(conditions)


# ---------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------


def build(seed: int = 0) -> None:
    points = fixed_points()
    bands, spectrum = measure_band(seed=seed)
    drift, conditions = measure_drift(seed=seed)

    fig, axes = figure(3, height=2.15)
    left, middle, right = axes

    # (a) --------------------------------------------------------------
    sigma = np.linspace(0, 1.45, 400)
    left.plot(sigma, quintic(sigma), color=COLORS["muon"], zorder=3,
              label=r"$p(\sigma)$")
    left.plot(sigma, sigma, color=COLORS["reference"], linestyle="--",
              linewidth=0.8, zorder=2, label=r"$\sigma$")

    trail = orbit(0.05, 14)
    walk_x, walk_y = [], []
    for before, after in pairwise(trail):
        walk_x += [before, before]
        walk_y += [before, after]
    left.plot(walk_x, walk_y, color="#777777", linewidth=0.55, zorder=2)

    for point in points:
        left.plot([point], [point], marker="o", markersize=4.5,
                  color=COLORS["astro"], zorder=5)
        left.annotate(f"{point:.3f}", (point, point), textcoords="offset points",
                      xytext=(4, -9), fontsize=6.5, color=COLORS["astro"])
    left.axvline(1.0, color=COLORS["normuon"], linestyle=":", linewidth=0.9)
    left.annotate(r"target $\sigma=1$", (1.0, 0.12), rotation=90, fontsize=6.5,
                  color=COLORS["normuon"], ha="right", va="bottom")
    left.set_xlim(0, 1.45)
    left.set_ylim(0, 1.45)
    left.set_xlabel(r"singular value $\sigma$")
    left.set_ylabel(r"$p(\sigma)$")
    left.legend(loc="upper left")
    grid(left)

    # (b) --------------------------------------------------------------
    # cycle, not strict: the budget list is a parameter, the palette is not
    for shade, (steps, values) in zip(cycle(ASTRO_SHADES), sorted(bands.items()),
                                      strict=False):
        middle.plot(np.sort(values)[::-1], color=shade, linewidth=1.2,
                    label=f"{steps} steps", zorder=3)
    for point in points:
        middle.axhline(point, color=COLORS["reference"], linestyle=":",
                       linewidth=0.7, zorder=1)
    middle.axhline(1.0, color=COLORS["normuon"], linestyle="--", linewidth=0.9,
                   zorder=2)
    middle.annotate(f"fixed points\n{points[0]:.3f}, {points[-1]:.3f}",
                    (0.97, 0.5), xycoords="axes fraction", fontsize=6.5,
                    ha="right", va="center", color=COLORS["reference"])
    middle.set_xlabel("singular value index")
    middle.set_ylabel(r"filtered $\sigma$")
    # Zoomed onto the band: the whole claim is that these three curves lie on
    # top of each other between the fixed points, which a 0-1.45 axis hides.
    middle.set_ylim(0.55, 1.35)
    middle.legend(loc="lower left", title="Muon quintic", title_fontsize=6.5)
    grid(middle)

    # (c) --------------------------------------------------------------
    positions = np.arange(len(conditions))
    styles = {"muon5": (COLORS["muon"], "s", "-"),
              "polar5": (ASTRO_SHADES[0], "D", "--"),
              "polar6": (ASTRO_SHADES[1], "D", "-."),
              "polar7": (COLORS["astro"], "D", "-")}
    for name, (color, marker, dash) in styles.items():
        right.plot(positions, drift[name], color=color, marker=marker,
                   linestyle=dash, label=name, zorder=3)
    reference_line(right, 1.0, "theoretical")
    right.set_xticks(positions)
    right.set_xticklabels([f"$10^{int(np.log10(value))}$" for value in conditions])
    right.set_xlabel("condition number of the momentum")
    right.set_ylabel(r"$\|Z\|_F \,/\, \sqrt{\min(m,n)}$")
    right.set_ylim(0.6, 1.12)
    right.legend(loc="lower center", ncol=2, columnspacing=0.9)
    grid(right)
    swing = max(drift["muon5"]) - min(drift["muon5"])
    annotate_value(right, positions[1], max(drift["muon5"]),
                   f"{swing:.0%} swing", dx=0, dy=9, ha="center",
                   color=COLORS["muon"])

    label_panels(axes)
    fig.tight_layout(pad=0.4, w_pad=1.4)
    path = save(fig, "fig2_quintic")

    write_data("fig2_quintic", {
        "fixed_points": points,
        "quintic": list(MUON_QUINTIC),
        "orbit_from_0.05": trail,
        "band": {str(steps): values.tolist() for steps, values in bands.items()},
        "band_input_spectrum": spectrum.tolist(),
        "drift": drift,
        "conditions": conditions,
        "muon5_swing": swing,
        "seed": seed,
    })

    print(f"  wrote {path}")
    print(f"    fixed points: {', '.join(f'{p:.6f}' for p in points)}")
    print(f"    orbit from 0.05 after 14 steps: {trail[-1]:.6f} (target 1.0)")
    for steps in sorted(bands):
        values = bands[steps]
        print(f"    {steps:2d} steps -> sigma in [{values.min():.4f}, "
              f"{values.max():.4f}]")
    print(f"    muon5 norm ratio swings {swing:.1%} over condition 1..1000; "
          f"polar7 swings {max(drift['polar7']) - min(drift['polar7']):.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    build(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
