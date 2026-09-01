"""Shared publication style for every figure in the paper.

One module so that a reader flipping between figures sees one visual language:
the same colour means the same optimizer everywhere, panel labels are placed
identically, and the annotation helpers are the ones the figures actually need
rather than whatever each script reinvented.

Sizes assume a single-column article at 1in margins, so the usable text width
is 6.5in. A three-panel row is ``figure(3)``; a single plot is ``figure(1)``.
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

TEXT_WIDTH = 6.5  # inches, \textwidth at 1in margins on US letter

# One colour per optimizer, used in every figure. Adam blue and Muon orange
# follow the convention of the curvature literature we compare against, so a
# reader coming from those papers reads our panels without a key.
COLORS = {
    "adamw": "#1f77b4",
    "adam": "#1f77b4",
    "muon": "#ff7f0e",
    "normuon": "#2ca02c",
    "adamuon": "#17becf",
    "astro": "#d62728",
    "astro_cautious": "#9467bd",
    "soap": "#8c564b",
    "reference": "#444444",
}

MARKERS = {
    "adamw": "o", "adam": "o", "muon": "s", "normuon": "^",
    "adamuon": "v", "astro": "D", "astro_cautious": "X", "soap": "P",
}

LABELS = {
    "adamw": "AdamW", "adam": "Adam", "muon": "Muon", "normuon": "NorMuon",
    "adamuon": "AdaMuon", "astro": "ASTRO", "astro_cautious": "ASTRO + mask",
    "soap": "SOAP",
}


def use_paper_style() -> None:
    """Serif, thin spines, no top/right box -- the house style of the field."""
    mpl.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "lines.linewidth": 1.4,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })


def figure(panels: int = 1, height: float = 2.1, width: float | None = None,
           **kwargs: Any) -> tuple[plt.Figure, Any]:
    """A row of ``panels`` axes spanning the text width."""
    use_paper_style()
    width = TEXT_WIDTH if width is None else width
    fig, axes = plt.subplots(1, panels, figsize=(width, height), **kwargs)
    return fig, axes


def label_panels(axes: Iterable[plt.Axes], start: str = "a") -> None:
    """(a) (b) (c) beneath each panel, as the reference figures place them."""
    for index, ax in enumerate(axes):
        ax.set_title(f"({chr(ord(start) + index)})", loc="left",
                     fontsize=8, fontweight="bold", pad=4)


def grid(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, linestyle=":", zorder=0)
    ax.set_axisbelow(True)


def annotate_value(ax: plt.Axes, x: float, y: float, text: str,
                   dx: float = 0.0, dy: float = 8.0, **kwargs: Any) -> None:
    """A number placed on the plot itself.

    Every quantitative claim in a caption should be readable off the figure
    without the caption, which is why the reference figures label their bars
    and mark their averages inline.
    """
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                ha=kwargs.pop("ha", "center"), fontsize=kwargs.pop("fontsize", 7),
                **kwargs)


def reference_line(ax: plt.Axes, value: float = 1.0, text: str | None = None,
                   axis: str = "y") -> None:
    """The 'no difference' line, drawn so a ratio panel reads at a glance."""
    draw = ax.axhline if axis == "y" else ax.axvline
    draw(value, color=COLORS["reference"], linestyle="--", linewidth=0.8, zorder=1)
    if text:
        if axis == "y":
            ax.annotate(text, (0.99, value), xycoords=("axes fraction", "data"),
                        textcoords="offset points", xytext=(0, 3),
                        ha="right", fontsize=6.5, color=COLORS["reference"])
        else:
            ax.annotate(text, (value, 0.98), xycoords=("data", "axes fraction"),
                        textcoords="offset points", xytext=(3, 0),
                        va="top", fontsize=6.5, color=COLORS["reference"],
                        rotation=90)


def shade_between(ax: plt.Axes, x: Sequence[float], lower: Sequence[float],
                  upper: Sequence[float], color: str, alpha: float = 0.15) -> None:
    ax.fill_between(x, lower, upper, color=color, alpha=alpha, linewidth=0, zorder=1)


def series(ax: plt.Axes, x, y, name: str, *, band=None, **kwargs: Any):
    """Plot one optimizer with its assigned colour, marker and label."""
    line = ax.plot(x, y, color=COLORS.get(name, "#333333"),
                   marker=MARKERS.get(name, "o"),
                   label=kwargs.pop("label", LABELS.get(name, name)),
                   zorder=3, **kwargs)
    if band is not None:
        shade_between(ax, x, band[0], band[1], COLORS.get(name, "#333333"))
    return line


def save(fig: plt.Figure, name: str, *, also_pdf: bool = True) -> Path:
    """Write to artifacts/figures. PDF too, because that is what LaTeX wants."""
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{name}.png"
    fig.savefig(png)
    if also_pdf:
        fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    return png


# ---------------------------------------------------------------------------
# Data plumbing
# ---------------------------------------------------------------------------


def write_data(name: str, payload: dict[str, Any]) -> Path:
    """Every figure writes the numbers it drew.

    A figure whose underlying numbers are not on disk cannot be checked by a
    reader or regenerated after the plotting code changes, so this is not
    optional bookkeeping -- it is what makes the figure evidence.
    """
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
    """Load a measurement JSON, searching the usual drop points.

    Measurement happens on a GPU elsewhere and plotting happens here, so the
    loader looks where an uploaded file plausibly landed rather than demanding
    one path.
    """
    candidates = [Path(name), OUT / name, OUT / f"{name}.json",
                  ROOT / name, ROOT / "artifacts" / name]
    candidates += [Path(directory) / name for directory in search]
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text())
    return None


def missing(name: str, produced_by: str) -> None:
    """Say precisely what is absent and which command creates it."""
    print(f"  SKIP {name}: no measurement file found.")
    print(f"       produce it with: {produced_by}")


# ---------------------------------------------------------------------------
# Alignment at matched loss
# ---------------------------------------------------------------------------


def align_on_loss(reference_loss, reference_value, target_loss, target_value,
                  grid_points: int = 40):
    """Interpolate two trajectories onto a shared validation-loss axis.

    Comparing optimizers at matched *step* flatters whichever one is ahead: a
    better optimizer is measured at a lower loss, where the landscape is
    different. The curvature literature therefore compares at matched
    validation loss, interpolating between checkpoints because two runs rarely
    hit the same loss at a recorded step. We do the same, and this is the
    function that does it.

    Both trajectories are assumed to be monotonically decreasing in loss; the
    returned grid covers only the overlap, so nothing is extrapolated.
    """
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
