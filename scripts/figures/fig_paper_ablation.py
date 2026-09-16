"""Paper figure: the four causal contrasts required for ASTRO-v2.

The mechanism campaign does not retune ablations.  Muon's beta control uses the
frozen Muon configuration; ASTRO removals use the frozen ASTRO-v2 configuration.
This keeps the figure focused on attribution rather than building an optimizer
zoo.
"""

from __future__ import annotations

import numpy as np

from paper_style import COLORS, figure, grid, label_panels, missing, read_data, save, write_data


def _paired(block: dict, a: str, b: str) -> tuple[list[int], np.ndarray, np.ndarray]:
    ra, rb = block["optimizers"][a], block["optimizers"][b]
    av = {s: v for s, v in zip(ra["seeds"], ra["loss"])}
    bv = {s: v for s, v in zip(rb["seeds"], rb["loss"])}
    seeds = sorted(set(av) & set(bv))
    return seeds, np.asarray([av[s] for s in seeds]), np.asarray([bv[s] for s in seeds])


def build(source: str = "paper_campaign.json") -> None:
    payload = read_data(source)
    if payload is None:
        missing("fig_paper_ablation", "python scripts/paper_collect.py ...")
        return
    block = payload.get("mechanism_124m_900", {})
    if not block.get("complete"):
        missing("fig_paper_ablation", "finish scripts/paper_mechanism.py")
        return

    contrasts = [
        ("muon_m90", "muon", r"matrix $\beta_1$\n.90 vs .95"),
        ("astro_v2", "astro_v2_gamma0", "row adaptation\non vs off"),
        ("astro_v2", "astro_v2_nosplit", "QKV split\non vs fused"),
        ("astro_v2", "astro_v2_blockwise", "global vs blockwise\nredistribution"),
    ]

    fig, axes = figure(panels=2, height=2.55)
    ax_loss, ax_delta = axes
    x = np.arange(len(contrasts), dtype=float)
    delta_means = []
    output = {}

    # Left: paired endpoint values. Each grey segment is one held-out seed.
    for xi, (a, b, label) in zip(x, contrasts):
        seeds, va, vb = _paired(block, a, b)
        for aa, bb in zip(va, vb):
            ax_loss.plot([xi - 0.12, xi + 0.12], [bb, aa], color="#D0D0D0",
                         linewidth=0.8, zorder=1)
            ax_loss.scatter(xi - 0.12, bb, s=17, color=COLORS.get(b, "#777777"),
                            alpha=0.65, linewidths=0, zorder=3)
            ax_loss.scatter(xi + 0.12, aa, s=19, color=COLORS.get(a, "#333333"),
                            alpha=0.78, linewidths=0, zorder=4)
        deltas = va - vb
        delta_means.append(float(deltas.mean()))
        output[label] = {
            "A": a,
            "B": b,
            "seeds": seeds,
            "A_loss": va.tolist(),
            "B_loss": vb.tolist(),
            "A_minus_B": deltas.tolist(),
            "mean_delta": float(deltas.mean()),
        }

    ax_loss.set_xticks(x)
    ax_loss.set_xticklabels([row[2] for row in contrasts], fontsize=5.5)
    ax_loss.set_ylabel("validation loss ↓")
    ax_loss.set_title("matched held-out seeds", loc="left", pad=6,
                      fontsize=7.2, fontweight="bold")
    grid(ax_loss, axis="y")

    # Right: mean causal deltas. Negative means the first-named intervention is
    # better under the frozen configuration.
    ax_delta.axhline(0, color="#555555", linestyle="--", linewidth=0.8, zorder=1)
    bars = ax_delta.bar(x, delta_means, width=0.58,
                        color=[COLORS["muon"], COLORS["astro_v2_gamma0"],
                               COLORS["astro_v2_nosplit"], COLORS["astro_v2"]],
                        alpha=0.82, zorder=3)
    for xi, value in zip(x, delta_means):
        ax_delta.annotate(f"{value:+.3f}", (xi, value), textcoords="offset points",
                          xytext=(0, 5 if value >= 0 else -9), ha="center",
                          fontsize=5.8, color="#444444")
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels(["beta", "row", "split", "global"], fontsize=5.8)
    ax_delta.set_ylabel("mean A − B loss ↓")
    ax_delta.set_title("isolated contribution", loc="left", pad=6,
                       fontsize=7.2, fontweight="bold")
    grid(ax_delta, axis="y")
    ax_delta.text(0.02, 0.035, "negative favors the intervention",
                  transform=ax_delta.transAxes, fontsize=5.6, color="#666666")
    label_panels(axes)

    fig.tight_layout(pad=0.5, w_pad=1.0)
    saved = save(fig, "fig_paper_ablation")
    write_data("fig_paper_ablation", output)
    print(f"  wrote {saved}")


if __name__ == "__main__":
    build()
