"""Paper figure: validation-loss trajectories through training.

Reads ``artifacts/trajectories.json`` produced by
``scripts/paper/run_trajectory.py``.  Panel (a) compares validation loss against
training step; panel (b) compares the same trajectories against measured
training wall-clock with validation-probe overhead removed.

This figure should be regenerated only from frozen configurations.  It is a
visualization of learning dynamics, not another tuning surface.
"""

from __future__ import annotations

import numpy as np

from style import COLORS, LABELS, figure, grid, label_panels, missing, read_data, save, write_data

PREFERRED_ORDER = ["adamw", "muon", "normuon", "adamuon",
                   "astro_muon_betas", "astro_v2"]


def _mean_on_common_axis(runs: list[dict], x_key: str):
    if len(runs) == 1:
        return (np.asarray(runs[0][x_key], dtype=float),
                np.asarray(runs[0]["val_loss"], dtype=float), None)

    # Runs produced by our trajectory script probe at the same step numbers.
    # Wall-clock differs, so for time we interpolate onto the overlap.
    if x_key == "step":
        x = np.asarray(runs[0]["step"], dtype=float)
        ys = np.asarray([run["val_loss"] for run in runs], dtype=float)
        return x, ys.mean(axis=0), (ys.min(axis=0), ys.max(axis=0))

    low = max(min(run[x_key]) for run in runs)
    high = min(max(run[x_key]) for run in runs)
    x = np.linspace(low, high, 80)
    ys = []
    for run in runs:
        source_x = np.asarray(run[x_key], dtype=float)
        source_y = np.asarray(run["val_loss"], dtype=float)
        ys.append(np.interp(x, source_x, source_y))
    ys = np.asarray(ys)
    return x, ys.mean(axis=0), (ys.min(axis=0), ys.max(axis=0))


def build(source: str = "trajectories.json") -> None:
    payload = read_data(source)
    if payload is None or not payload.get("runs"):
        missing("fig_training_trajectories",
                "python scripts/paper/run_trajectory.py --help")
        return

    available = [name for name in PREFERRED_ORDER if name in payload["runs"]]
    available += [name for name in payload["runs"] if name not in available]

    fig, axes = figure(2, height=2.45)
    step_ax, time_ax = axes
    rendered = {}

    for name in available:
        runs = payload["runs"][name]
        color = COLORS.get(name, "#333333")
        label = LABELS.get(name, name)

        x, mean, band = _mean_on_common_axis(runs, "step")
        step_ax.plot(x, mean, color=color, linewidth=1.6, label=label, zorder=3)
        if band is not None:
            step_ax.fill_between(x, band[0], band[1], color=color, alpha=0.10, linewidth=0)

        x_t, mean_t, band_t = _mean_on_common_axis(runs, "train_seconds")
        time_ax.plot(x_t, mean_t, color=color, linewidth=1.6, label=label, zorder=3)
        if band_t is not None:
            time_ax.fill_between(x_t, band_t[0], band_t[1], color=color, alpha=0.10,
                                 linewidth=0)

        rendered[name] = {
            "seeds": [run.get("seed") for run in runs],
            "final_mean_val_loss": float(np.mean([run["val_loss"][-1] for run in runs])),
        }

    step_ax.set_xlabel("training step")
    step_ax.set_ylabel("validation loss ↓")
    step_ax.set_title("Learning dynamics", fontsize=8.2)
    grid(step_ax)

    time_ax.set_xlabel("training wall-clock (s), probe time excluded")
    time_ax.set_ylabel("validation loss ↓")
    time_ax.set_title("Compute-normalized dynamics", fontsize=8.2)
    grid(time_ax)
    time_ax.legend(loc="best", fontsize=6.2, ncol=2)

    label_panels(axes)
    fig.tight_layout(pad=0.45, w_pad=1.5)
    saved = save(fig, "fig_training_trajectories")
    write_data("fig_training_trajectories", {
        "source_metadata": payload.get("metadata", {}),
        "rendered": rendered,
    })
    print(f"  wrote {saved}")


if __name__ == "__main__":
    build()
