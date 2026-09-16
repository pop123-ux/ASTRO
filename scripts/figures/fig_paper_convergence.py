"""Paper figure: training-loss convergence versus steps and wall-clock.

No extra evaluation passes are introduced for this plot.  ``paper_lab.py`` logs
training loss and elapsed time at the normal console logging points, and this
figure averages those matched points over the five held-out seeds.
"""

from __future__ import annotations

import numpy as np

from paper_style import COLORS, LABELS, figure, grid, label_panels, missing, read_data, save, write_data


def _mean_trace(traces: dict[str, dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    rows = []
    for seed, row in sorted(traces.items(), key=lambda item: int(item[0])):
        step = np.asarray(row.get("step", []), dtype=float)
        loss = np.asarray(row.get("loss", []), dtype=float)
        seconds = np.asarray(row.get("seconds", []), dtype=float)
        if len(step) and len(step) == len(loss) == len(seconds):
            rows.append((step, loss, seconds))
    if not rows:
        return None
    common = rows[0][0]
    if not all(np.array_equal(step, common) for step, _, _ in rows):
        return None
    loss = np.mean(np.stack([row[1] for row in rows]), axis=0)
    seconds = np.mean(np.stack([row[2] for row in rows]), axis=0)
    return common, loss, seconds


def build(source: str = "paper_campaign.json") -> None:
    payload = read_data(source)
    if payload is None:
        missing("fig_paper_convergence", "python scripts/paper_collect.py --main-traces ...")
        return
    traces = payload.get("main_traces", {})
    names = ["muon", "adamuon_ref", "astro_v2"]
    averaged = {name: _mean_trace(traces.get(name, {})) for name in names}
    if any(averaged[name] is None for name in names):
        missing("fig_paper_convergence", "finish the clean main run and collect its paper_traces directory")
        return

    fig, axes = figure(panels=2, height=2.38)
    ax_step, ax_time = axes
    output = {}
    for name in names:
        step, loss, seconds = averaged[name]  # type: ignore[misc]
        ax_step.plot(step, loss, color=COLORS[name], linewidth=1.45,
                     label=LABELS[name])
        ax_time.plot(seconds / 60.0, loss, color=COLORS[name], linewidth=1.45,
                     label=LABELS[name])
        output[name] = {
            "step": step.tolist(),
            "mean_train_loss": loss.tolist(),
            "mean_minutes": (seconds / 60.0).tolist(),
        }

    for ax in axes:
        grid(ax)
        ax.set_ylabel("mean training loss ↓")
    ax_step.set_xlabel("training step")
    ax_time.set_xlabel("wall-clock minutes")
    ax_step.set_title("matched optimizer steps", loc="left", pad=6,
                      fontsize=7.3, fontweight="bold")
    ax_time.set_title("measured wall-clock", loc="left", pad=6,
                      fontsize=7.3, fontweight="bold")
    ax_time.legend(loc="upper right", fontsize=6.0, handlelength=1.1)
    label_panels(axes)

    # The first few logged points dominate the y-range and hide the regime the
    # paper actually discusses.  Keep all points but cap only if the common
    # starting loss is far above the late-training range.
    late = np.concatenate([averaged[name][1][-6:] for name in names])  # type: ignore[index]
    upper = max(float(np.max(late)) + 0.8, float(np.min([averaged[n][1][0] for n in names])))
    for ax in axes:
        ax.set_ylim(bottom=min(float(np.min(averaged[n][1])) for n in names) - 0.15,
                    top=upper)

    fig.tight_layout(pad=0.5, w_pad=1.1)
    saved = save(fig, "fig_paper_convergence")
    write_data("fig_paper_convergence", output)
    print(f"  wrote {saved}")


if __name__ == "__main__":
    build()
