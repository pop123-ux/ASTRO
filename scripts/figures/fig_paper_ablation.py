"""Paper figure: minimal causal ladder for the ASTRO-v2 recipe.

The ladder answers exactly the attribution questions that matter:
1) does changing Muon's matrix momentum from .95 to .90 matter?
2) does an ASTRO implementation with both novel mechanisms disabled match the
   beta-aligned Muon control?
3) what changes when post-polar row redistribution is enabled?
4) what changes when Q/K/V are treated as separate operators?
"""

from __future__ import annotations

import numpy as np

from paper_style import COLORS, LABELS, column_figure, grid, missing, read_data, save, write_data


def build(source: str = "paper_campaign.json") -> None:
    payload = read_data(source)
    if payload is None:
        missing("fig_paper_ablation", "python scripts/paper_collect.py ...")
        return
    block = payload.get("ablation_124m_900", {})
    if not block.get("complete"):
        missing("fig_paper_ablation", "finish the two-seed causal ablation stage")
        return

    names = [
        "muon",
        "muon_m90",
        "astro_v2_gamma0_nosplit",
        "astro_v2_nosplit",
        "astro_v2",
    ]
    labels = [
        r"Muon\n$\beta=.95$",
        r"Muon\n$\beta=.90$",
        r"parity ctrl.\n$\gamma=0$, fused",
        r"+ row adapt.\nfused",
        r"+ QKV split\nASTRO-v2",
    ]

    fig, ax = column_figure(height=2.78)
    x = np.arange(len(names), dtype=float)
    means = []
    payload_out = {}

    for xi, name in zip(x, names):
        values = np.asarray(block["optimizers"][name]["loss"], dtype=float)
        mean = float(values.mean())
        means.append(mean)
        jitter = np.linspace(-0.055, 0.055, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(np.full(len(values), xi) + jitter, values, s=20,
                   color=COLORS[name], alpha=0.70, linewidths=0, zorder=3)
        ax.scatter(xi, mean, marker="D" if name == "astro_v2" else "o",
                   s=36 if name == "astro_v2" else 29,
                   color=COLORS[name], edgecolor="white", linewidth=0.5, zorder=4)
        payload_out[name] = {"loss": values.tolist(), "mean": mean}

    # Quiet connectors make the sequential interventions explicit without
    # pretending the five points form a continuous scalar hyperparameter.
    ax.plot(x, means, color="#D1D1D1", linewidth=1.0, zorder=1)
    for i in range(1, len(names)):
        delta = means[i] - means[i - 1]
        y = (means[i] + means[i - 1]) / 2
        ax.annotate(f"{delta:+.3f}", ((x[i] + x[i - 1]) / 2, y),
                    textcoords="offset points", xytext=(0, 8), ha="center",
                    fontsize=5.6, color="#666666")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.7)
    ax.set_ylabel("validation loss ↓")
    grid(ax, axis="y")
    ax.set_title("Causal recipe ladder · identical frozen hyperparameters",
                 loc="left", pad=7, fontsize=7.5, fontweight="bold")
    ax.text(0.02, 0.035, "two held-out seeds per intervention",
            transform=ax.transAxes, fontsize=5.7, color="#666666")

    all_values = np.concatenate([
        np.asarray(block["optimizers"][name]["loss"], dtype=float) for name in names
    ])
    pad = max(0.02, 0.14 * float(all_values.max() - all_values.min()))
    ax.set_ylim(float(all_values.min() - pad), float(all_values.max() + pad))
    fig.tight_layout(pad=0.4)
    saved = save(fig, "fig_paper_ablation")
    write_data("fig_paper_ablation", {
        "ladder": payload_out,
        "sequential_deltas": [means[i] - means[i - 1] for i in range(1, len(means))],
        "note": "all variants use the frozen ASTRO-v2 hyperparameter configuration",
    })
    print(f"  wrote {saved}")


if __name__ == "__main__":
    build()
