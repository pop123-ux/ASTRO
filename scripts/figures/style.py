"""Shared publication style for every ASTRO paper figure.

The paper treats figures as evidence, not decoration.  This module gives every
plot one visual language and keeps the aesthetics close to modern optimizer
papers: restrained serif typography, light grids, thin axes, direct labels,
consistent optimizer colours, and vector PDF output.

Use ``column_figure`` for a one-column figure and ``figure`` for a full-width
row.  Never hand-style a paper plot outside this module.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "artifacts" / "figures"

# Approximate widths for the two-column paper template in docs/paper/paper.tex.
TEXT_WIDTH = 7.05
COLUMN_WIDTH = 3.38

# Colour-blind-friendly, print-safe palette.  A colour means the same optimizer
# everywhere in the paper.  ASTRO-v2 is intentionally the strongest accent.
COLORS = {
    "adamw": "#4C78A8",
    "adam": "#4C78A8",
    "muon": "#F58518",
    "normuon": "#54A24B",
    "adamuon": "#72B7B2",
    "soap": "#B279A2",
    "astro": "#E45756",
    "astro_v1": "#E45756",
    "astro_muon_betas": "#9D5CBB",
    "astro_v2": "#7A3E9D",
    "astro_v2_gamma0": "#C58AE6",
    "astro_cautious": "#FF9DA6",
    "reference": "#4D4D4D",
    "pending": "#B8B8B8",
}

MARKERS = {
    "adamw": "o",
    "adam": "o",
    "muon": "s",
    "normuon": "^",
    "adamuon": "v",
    "soap": "P",
    "astro": "D",
    "astro_v1": "D",
    "astro_muon_betas": "X",
    "astro_v2": "D",
    "astro_v2_gamma0": "h",
    "astro_cautious": "X",
}

LABELS = {
    "adamw": "AdamW",
    "adam": "Adam",
    "muon": "Muon",
    "normuon": "NorMuon",
    "adamuon": "AdaMuon",
    "soap": "SOAP",
    "astro": "ASTRO (earlier recipe)",
    "astro_v1": "ASTRO (earlier recipe)",
    "astro_muon_betas": "ASTRO-MB",
    "astro_v2": "ASTRO-v2",
    "astro_v2_gamma0": r"ASTRO-v2 ($\gamma=0$)",
    "astro_cautious": "ASTRO + mask",
}


def use_paper_style() -> None:
    """Apply the global house style used by every paper plot."""
    mpl.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 350,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.025,
        "savefig.transparent": False,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8.2,
        "axes.labelsize": 8.2,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7.0,
        "xtick.labelsize": 7.1,
        "ytick.labelsize": 7.1,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#3C3C3C",
        "axes.labelcolor": "#222222",
        "text.color": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.45,
        "grid.alpha": 0.75,
        "lines.linewidth": 1.55,
        "lines.markersize": 4.0,
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.45,
        "legend.columnspacing": 0.9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def figure(panels: int = 1, height: float = 2.25, width: float | None = None,
           **kwargs: Any) -> tuple[plt.Figure, Any]:
    """Create a full-width row of paper axes."""
    use_paper_style()
    width = TEXT_WIDTH if width is None else width
    fig, axes = plt.subplots(1, panels, figsize=(width, height), **kwargs)
    return fig, axes


def column_figure(height: float = 2.2, **kwargs: Any) -> tuple[plt.Figure, plt.Axes]:
    """Create a single-column paper figure."""
    use_paper_style()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, height), **kwargs)
    return fig, ax


def label_panels(axes: Iterable[plt.Axes], start: str = "a") -> None:
    """Place compact (a), (b), ... labels in the upper-left of each panel."""
    for index, ax in enumerate(axes):
        ax.text(0.0, 1.025, f"({chr(ord(start) + index)})",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8.2, fontweight="bold")


def grid(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, linestyle="-", zorder=0)
    ax.set_axisbelow(True)


def annotate_value(ax: plt.Axes, x: float, y: float, text: str,
                   dx: float = 0.0, dy: float = 7.0, **kwargs: Any) -> None:
    """Place a quantitative annotation without obscuring the data."""
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                ha=kwargs.pop("ha", "center"), fontsize=kwargs.pop("fontsize", 6.8),
                **kwargs)


def reference_line(ax: plt.Axes, value: float = 0.0, text: str | None = None,
                   axis: str = "y") -> None:
    """Draw a quiet no-difference reference line."""
    draw = ax.axhline if axis == "y" else ax.axvline
    draw(value, color=COLORS["reference"], linestyle="--", linewidth=0.8, zorder=1)
    if text:
        if axis == "y":
            ax.annotate(text, (0.99, value), xycoords=("axes fraction", "data"),
                        textcoords="offset points", xytext=(0, 3), ha="right",
                        fontsize=6.4, color=COLORS["reference"])
        else:
            ax.annotate(text, (value, 0.98), xycoords=("data", "axes fraction"),
                        textcoords="offset points", xytext=(3, 0), va="top",
                        fontsize=6.4, color=COLORS["reference"], rotation=90)


def shade_between(ax: plt.Axes, x: Sequence[float], lower: Sequence[float],
                  upper: Sequence[float], color: str, alpha: float = 0.13) -> None:
    ax.fill_between(x, lower, upper, color=color, alpha=alpha, linewidth=0, zorder=1)


def series(ax: plt.Axes, x, y, name: str, *, band=None, **kwargs: Any):
    """Plot one optimizer with its globally assigned visual identity."""
    line = ax.plot(x, y,
                   color=COLORS.get(name, "#333333"),
                   marker=MARKERS.get(name, "o"),
                   label=kwargs.pop("label", LABELS.get(name, name)),
                   zorder=3, **kwargs)
    if band is not None:
        shade_between(ax, x, band[0], band[1], COLORS.get(name, "#333333"))
    return line


def direct_label(ax: plt.Axes, x: float, y: float, name: str, *, dx: float = 5,
                 dy: float = 0, **kwargs: Any) -> None:
    """Direct-label a curve; preferred over legends when the panel has room."""
    ax.annotate(LABELS.get(name, name), (x, y), textcoords="offset points",
                xytext=(dx, dy), ha="left", va="center", fontsize=6.8,
                color=COLORS.get(name, "#333333"), **kwargs)


def save(fig: plt.Figure, name: str, *, also_pdf: bool = True,
         also_svg: bool = False) -> Path:
    """Write a high-resolution PNG and a vector PDF for LaTeX."""
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{name}.png"
    fig.savefig(png, facecolor="white")
    if also_pdf:
        fig.savefig(OUT / f"{name}.pdf", facecolor="white")
    if also_svg:
        fig.savefig(OUT / f"{name}.svg", facecolor="white")
    plt.close(fig)
    return png


# ---------------------------------------------------------------------------
# Data plumbing
# ---------------------------------------------------------------------------


def write_data(name: str, payload: dict[str, Any]) -> Path:
    """Write the exact numbers behind a figure next to the image."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_plain))
    return path


def _plain(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialise {type(value)}")


def read_data(name: str, *, search: Sequence[Path] = ()) -> dict[str, Any] | None:
    """Load a JSON measurement from the common paper/result locations."""
    candidates = [Path(name), OUT / name, OUT / f"{name}.json",
                  ROOT / name, ROOT / "artifacts" / name]
    candidates += [Path(directory) / name for directory in search]
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text())
    return None


def missing(name: str, produced_by: str) -> None:
    """Print an actionable message when a figure has no evidence yet."""
    print(f"  SKIP {name}: no measurement file found.")
    print(f"       produce it with: {produced_by}")


# ---------------------------------------------------------------------------
# Alignment at matched loss
# ---------------------------------------------------------------------------


def align_on_loss(reference_loss, reference_value, target_loss, target_value,
                  grid_points: int = 40):
    """Interpolate two trajectories onto their shared validation-loss range."""
    reference_loss = np.asarray(reference_loss, dtype=float)
    target_loss = np.asarray(target_loss, dtype=float)
    reference_value = np.asarray(reference_value, dtype=float)
    target_value = np.asarray(target_value, dtype=float)

    low = max(reference_loss.min(), target_loss.min())
    high = min(reference_loss.max(), target_loss.max())
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError(
            f"the two trajectories do not overlap in loss: "
            f"reference [{reference_loss.min():.4f}, {reference_loss.max():.4f}], "
            f"target [{target_loss.min():.4f}, {target_loss.max():.4f}]"
        )
    axis = np.linspace(low, high, grid_points)

    def resample(loss, value):
        order = np.argsort(loss)
        return np.interp(axis, loss[order], value[order])

    return axis, resample(reference_loss, reference_value), resample(target_loss, target_value)
