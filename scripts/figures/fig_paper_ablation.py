"""Paper figure: minimal causal controls for the ASTRO-v2 recipe.

Panel (a) removes the matrix-momentum confound and checks implementation parity.
Panel (b) is the smallest factorial that can support a claim about the *combined*
effect of post-polar row redistribution and operator-aware QKV splitting.  Both
components have close prior art; the interaction/composition is therefore more
important scientifically than presenting either component as independently new.
"""

from __future__ import annotations

import numpy as np

from paper_style import COLORS, figure, grid, label_panels, missing, read_data, save, write_data


def _values(block: dict, name: str) -> np.ndarray:
    return np.asarray(block["optimizers"][name]["loss"], dtype=float)


def build(source: str = "paper_campaign.json") -> None:
    payload = read_data(source)
    if payload is None:
        missing("fig_paper_ablation", "python scripts/paper_collect.py ...")
        return
    block = payload.get("ablation_124m_900", {})
    required = [
        "muon",
        "muon_m90",
        "astro_v2_gamma0_nosplit",
        "astro_v2_gamma0",
        "astro_v2_nosplit",
        "astro_v2",
    ]
    if not block.get("complete") or any(name not in block.get("optimizers", {}) for name in required):
        missing("fig_paper_ablation", "finish the six-condition, two-seed causal ablation stage")
        return

    fig, axes = figure(panels=2, height=2.48)
    ax_beta, ax_factor = axes
    output = {}

    # (a) Beta + parity control --------------------------------------------
    beta_names = ["muon", "muon_m90", "astro_v2_gamma0_nosplit"]
    beta_labels = [r"Muon\n$\beta=.95$", r"Muon\n$\beta=.90$", r"ASTRO ctrl.\n$\gamma=0$, fused"]
    beta_x = np.arange(3, dtype=float)
    beta_means = []
    for xi, name in zip(beta_x, beta_names):
        values = _values(block, name)
        mean = float(values.mean())
        beta_means.append(mean)
        jitter = np.linspace(-0.045, 0.045, len(values)) if len(values) > 1 else np.array([0.0])
        ax_beta.scatter(np.full(len(values), xi) + jitter, values, s=18,
                        color=COLORS[name], alpha=0.65, linewidths=0, zorder=3)
        ax_beta.scatter(xi, mean, s=32, color=COLORS[name], edgecolor="white",
                        linewidth=0.45, zorder=4)
        output[name] = {"loss": values.tolist(), "mean": mean}
    ax_beta.plot(beta_x, beta_means, color="#D2D2D2", linewidth=0.9, zorder=1)
    ax_beta.set_xticks(beta_x)
    ax_beta.set_xticklabels(beta_labels, fontsize=5.7)
    ax_beta.set_ylabel("validation loss ↓")
    ax_beta.set_title("beta control + parity check", loc="left", pad=6,
                      fontsize=7.2, fontweight="bold")
    grid(ax_beta, axis="y")

    # (b) 2x2 mechanism factorial -----------------------------------------
    # x=0: row redistribution disabled (gamma=0); x=1: enabled (gamma=1).
    # line 1: fused QKV; line 2: split Q/K/V.
    factorial = {
        "fused QKV": ["astro_v2_gamma0_nosplit", "astro_v2_nosplit"],
        "split Q/K/V": ["astro_v2_gamma0", "astro_v2"],
    }
    line_colors = {"fused QKV": COLORS["astro_v2_nosplit"],
                   "split Q/K/V": COLORS["astro_v2"]}
    fx = np.asarray([0.0, 1.0])
    factorial_means = {}
    for label, names in factorial.items():
        means = []
        for xi, name in zip(fx, names):
            values = _values(block, name)
            mean = float(values.mean())
            means.append(mean)
            jitter = np.linspace(-0.025, 0.025, len(values)) if len(values) > 1 else np.array([0.0])
            ax_factor.scatter(np.full(len(values), xi) + jitter, values, s=17,
                              color=line_colors[label], alpha=0.55, linewidths=0, zorder=3)
            output[name] = {"loss": values.tolist(), "mean": mean}
        ax_factor.plot(fx, means, marker="D" if label == "split Q/K/V" else "o",
                       markersize=4.2, color=line_colors[label], linewidth=1.35,
                       label=label, zorder=4)
        factorial_means[label] = means

    ax_factor.set_xticks(fx)
    ax_factor.set_xticklabels([r"row adapt. off\n$\gamma=0$", r"row adapt. on\n$\gamma=1$"],
                              fontsize=5.8)
    ax_factor.set_title("row adaptation × operator split", loc="left", pad=6,
                        fontsize=7.2, fontweight="bold")
    ax_factor.legend(loc="upper right", fontsize=5.8, handlelength=1.0)
    grid(ax_factor, axis="y")
    ax_factor.set_ylabel("validation loss ↓")
    label_panels(axes)

    all_values = np.concatenate([_values(block, name) for name in required])
    pad = max(0.02, 0.13 * float(all_values.max() - all_values.min()))
    lo, hi = float(all_values.min() - pad), float(all_values.max() + pad)
    for ax in axes:
        ax.set_ylim(lo, hi)

    fig.tight_layout(pad=0.5, w_pad=1.0)
    saved = save(fig, "fig_paper_ablation")
    write_data("fig_paper_ablation", {
        "conditions": output,
        "beta_95_to_90": beta_means[1] - beta_means[0],
        "parity_control_minus_muon_beta90": beta_means[2] - beta_means[1],
        "factorial_means": factorial_means,
        "row_effect_fused": factorial_means["fused QKV"][1] - factorial_means["fused QKV"][0],
        "row_effect_split": factorial_means["split Q/K/V"][1] - factorial_means["split Q/K/V"][0],
        "split_effect_gamma0": factorial_means["split Q/K/V"][0] - factorial_means["fused QKV"][0],
        "split_effect_gamma1": factorial_means["split Q/K/V"][1] - factorial_means["fused QKV"][1],
        "note": "all six conditions use the frozen ASTRO-v2 hyperparameter configuration",
    })
    print(f"  wrote {saved}")


if __name__ == "__main__":
    build()
