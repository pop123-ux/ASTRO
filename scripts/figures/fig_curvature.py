"""Figure 3: the second-order decomposition, ASTRO against Muon and AdamW.

Reads the JSON written by ``scripts/measure_curvature.py``. Panels follow the
decomposition rather than the convenience of plotting:

  (a) predicted vs realized one-step loss decrease -- establishes that the
      second-order model describes what actually happens before anything is
      concluded from its terms
  (b) the two terms: first-order gain, and curvature penalty
  (c) the penalty factored into update norm and NDS, as ratios against Muon,
      which is the panel that says *which* of the two carries the difference

Everything is plotted against matched validation loss, not against step. See
``style.align_on_loss`` for why that is not a cosmetic choice.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
from style import (
    COLORS,
    LABELS,
    align_on_loss,
    figure,
    grid,
    label_panels,
    missing,
    read_data,
    reference_line,
    save,
    write_data,
)

PRODUCED_BY = ("python scripts/measure_curvature.py --optimizers adamw muon "
               "normuon astro --steps 600 --probe-every 50 --out curvature.json")


def group(records: list[dict]) -> dict[str, dict[str, np.ndarray]]:
    """Average repeated seeds at each step, then sort by validation loss."""
    buckets: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in records:
        buckets[(record["optimizer"], record["step"])].append(record)

    fields = ("val_loss", "realized", "predicted", "first_order",
              "curvature_penalty", "update_norm_sq", "nds")
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (name, step), group_records in sorted(buckets.items()):
        for field in fields:
            values = [record[field] for record in group_records if field in record]
            if values:
                out[name][field].append(float(np.mean(values)))
        out[name]["step"].append(step)
        out[name]["seeds"].append(len(group_records))

    finished: dict[str, dict[str, np.ndarray]] = {}
    for name, series_map in out.items():
        order = np.argsort(np.asarray(series_map["val_loss"]))
        finished[name] = {key: np.asarray(value)[order]
                          for key, value in series_map.items()}
    return finished


def build(path: str = "curvature.json") -> None:
    payload = read_data(path)
    if payload is None:
        missing("fig3_curvature", PRODUCED_BY)
        return
    curves = group(payload["records"])
    if "muon" not in curves:
        print("  SKIP fig3_curvature: no Muon trajectory to use as reference.")
        return

    fig, axes = figure(3, height=2.2)
    left, middle, right = axes
    ordered = [name for name in ("adamw", "muon", "normuon", "astro")
               if name in curves]

    # (a) does the second-order model describe reality at all? ----------
    for name in ordered:
        data = curves[name]
        color = COLORS.get(name, "#333")
        left.plot(data["val_loss"], data["realized"], color=color,
                  marker="o", markersize=3, label=f"{LABELS.get(name, name)} (true)")
        left.plot(data["val_loss"], data["predicted"], color=color,
                  linestyle=":", marker="x", markersize=3, alpha=0.8,
                  label=f"{LABELS.get(name, name)} (pred)")
    left.set_yscale("log")
    left.set_xlabel("validation loss")
    left.set_ylabel("one-step loss decrease")
    left.legend(loc="best", ncol=2, fontsize=5.5, columnspacing=0.8)
    grid(left)

    # (b) the two terms -------------------------------------------------
    for name in ordered:
        data = curves[name]
        color = COLORS.get(name, "#333")
        middle.plot(data["val_loss"], data["first_order"], color=color,
                    marker="o", markersize=3, label=LABELS.get(name, name))
        middle.plot(data["val_loss"], data["curvature_penalty"], color=color,
                    linestyle="--", marker="s", markersize=3, alpha=0.85)
    middle.set_yscale("log")
    middle.set_xlabel("validation loss")
    middle.set_ylabel("term value")
    # The two families are decades apart, so the middle of the panel is empty
    # and the corners are not.
    middle.annotate("solid: $\\langle G,Z\\rangle$\ndashed: curvature penalty",
                    (0.5, 0.5), xycoords="axes fraction", ha="center",
                    va="center", fontsize=6, color="#555555")
    middle.legend(loc="upper left", fontsize=6)
    grid(middle)

    # (c) which factor carries the gap ----------------------------------
    summary: dict[str, dict[str, float]] = {}
    reference = curves["muon"]
    for name in ordered:
        if name == "muon":
            continue
        data = curves[name]
        color = COLORS.get(name, "#333")
        try:
            axis, muon_nds, other_nds = align_on_loss(
                reference["val_loss"], reference["nds"],
                data["val_loss"], data["nds"])
            _, muon_norm, other_norm = align_on_loss(
                reference["val_loss"], reference["update_norm_sq"],
                data["val_loss"], data["update_norm_sq"])
        except ValueError as error:
            print(f"  note: {name} not comparable to muon -- {error}")
            continue
        nds_ratio = other_nds / muon_nds
        norm_ratio = other_norm / muon_norm
        right.plot(axis, nds_ratio, color=color, linestyle="-",
                   marker="o", markersize=2.5,
                   label=f"{LABELS.get(name, name)}: NDS")
        right.plot(axis, norm_ratio, color=color, linestyle=":",
                   marker="s", markersize=2.5, alpha=0.75,
                   label=f"{LABELS.get(name, name)}: $\\|Z\\|^2$")
        summary[name] = {"mean_nds_ratio": float(np.mean(nds_ratio)),
                         "mean_norm_ratio": float(np.mean(norm_ratio))}
    reference_line(right, 1.0, "= Muon")
    right.set_xlabel("validation loss")
    right.set_ylabel("ratio to Muon")
    right.legend(loc="best", fontsize=5.5, ncol=1)
    grid(right)

    label_panels(axes)
    fig.tight_layout(pad=0.4, w_pad=1.4)
    saved = save(fig, "fig3_curvature")
    write_data("fig3_curvature", {"ratios_vs_muon": summary,
                                  "configs": payload.get("configs"),
                                  "size": payload.get("size")})

    print(f"  wrote {saved}")
    for name, values in summary.items():
        verdict = ("direction" if values["mean_nds_ratio"] < 0.98
                   else "step size" if values["mean_norm_ratio"] < 0.98
                   else "neither")
        print(f"    {name} vs muon: NDS x{values['mean_nds_ratio']:.3f}, "
              f"|Z|^2 x{values['mean_norm_ratio']:.3f}  -> {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="curvature.json")
    args = parser.parse_args()
    build(args.data)


if __name__ == "__main__":
    main()
