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
import fig_paired_900  # noqa: E402
import fig_scale_status  # noqa: E402
import fig_seed_replication  # noqa: E402
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


def test_complete_900_horizon_recomputes_from_rows() -> None:
    block = paper_results()["horizon_124m_v2"]["steps"]["900"]
    rows = block["loss_by_config"]
    assert block["complete"] is True
    assert len(rows) == 5
    deltas = [row["astro_v2"] - row["muon"] for row in rows]
    # Three rows are available only to the four decimals printed by the
    # terminal report, so use a tolerance consistent with that provenance.
    assert deltas == pytest.approx(block["deltas_vs_muon"], abs=1e-4)
    assert float(np.mean(block["deltas_vs_muon"])) == pytest.approx(block["mean_delta"], abs=1e-8)
    assert sum(delta < 0 for delta in block["deltas_vs_muon"]) == 3


@pytest.mark.parametrize("builder,stem", [
    (fig_optimizer_journey.build, "fig_optimizer_journey"),
    (fig_paired_900.build, "fig_paired_900"),
    (fig_horizon_v2.build, "fig_horizon_v2"),
    (fig_efficiency_v2.build, "fig_efficiency_v2"),
    (fig_seed_replication.build, "fig_seed_replication"),
    (fig_scale_status.build, "fig_scale_status"),
])
def test_current_paper_figures_build(builder, stem) -> None:
    builder()
    assert (FIGURES / f"{stem}.png").is_file()
    assert (FIGURES / f"{stem}.pdf").is_file()
    assert (FIGURES / f"{stem}.json").is_file()


def test_horizon_keeps_only_2700_incomplete() -> None:
    block = paper_results()["horizon_124m_v2"]["steps"]
    assert block["300"]["complete"] is True
    assert block["600"]["complete"] is True
    assert block["900"]["complete"] is True
    assert block["2700"]["complete"] is False
    assert len(block["900"]["deltas_vs_muon"]) == 5
    assert block["2700"]["deltas_vs_muon"] == []


def test_replication_does_not_invent_astro_seeds() -> None:
    block = paper_results()["replication_124m_900"]
    assert len(block["muon"]["loss"]) == 7
    assert block["astro_v2"]["loss"] == []
    assert block["complete"] is False


def test_trajectory_plotter_refuses_to_invent_missing_data(tmp_path, capsys) -> None:
    fig_training_trajectories.build(str(tmp_path / "not-there.json"))
    out = capsys.readouterr().out
    assert "SKIP" in out
    assert "run_trajectory.py" in out
