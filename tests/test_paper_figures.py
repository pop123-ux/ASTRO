"""Tests for the current ASTRO-v2 paper-facing figure layer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "figures"))

import fig_efficiency_v2  # noqa: E402
import fig_horizon_v2  # noqa: E402
import fig_optimizer_journey  # noqa: E402
import fig_training_trajectories  # noqa: E402

FIGURES = ROOT / "artifacts" / "figures"


def paper_results() -> dict:
    return json.loads((ROOT / "artifacts" / "paper_results.json").read_text())


def test_targeted_mean_delta_is_reproducible_from_raw_configs() -> None:
    block = paper_results()["targeted_mechanism_124m_900"]
    for name, reported in block["mean_delta_vs_muon"].items():
        deltas = [cfg["loss"][name] - cfg["loss"]["muon"] for cfg in block["configs"]]
        assert float(np.mean(deltas)) == pytest.approx(reported, abs=1e-4)


def test_shared_config_deltas_match_losses() -> None:
    block = paper_results()["shared_config_124m_900"]
    muon = block["loss"]["muon"]
    for name, delta in block["delta_vs_muon"].items():
        assert block["loss"][name] - muon == pytest.approx(delta, abs=1e-6)


@pytest.mark.parametrize("builder,stem", [
    (fig_optimizer_journey.build, "fig_optimizer_journey"),
    (fig_horizon_v2.build, "fig_horizon_v2"),
    (fig_efficiency_v2.build, "fig_efficiency_v2"),
])
def test_current_paper_figures_build(builder, stem) -> None:
    builder()
    assert (FIGURES / f"{stem}.png").is_file()
    assert (FIGURES / f"{stem}.pdf").is_file()
    assert (FIGURES / f"{stem}.json").is_file()


def test_horizon_keeps_incomplete_cells_explicit() -> None:
    block = paper_results()["horizon_124m_v2"]["steps"]
    assert block["300"]["complete"] is True
    assert block["600"]["complete"] is True
    assert block["900"]["complete"] is False
    assert block["2700"]["complete"] is False
    assert len(block["900"]["deltas_vs_muon"]) == 2
    assert block["2700"]["deltas_vs_muon"] == []


def test_trajectory_plotter_refuses_to_invent_missing_data(tmp_path, capsys) -> None:
    fig_training_trajectories.build(str(tmp_path / "not-there.json"))
    out = capsys.readouterr().out
    assert "SKIP" in out
    assert "run_trajectory.py" in out
