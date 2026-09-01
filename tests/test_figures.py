"""The figure suite: it draws, it does not lie, and it fails loudly.

Two of these figures are drawn from measurements that take a GPU session to
produce. Their plotting code must therefore be exercised *here*, against
synthetic data of the right shape, so that a broken plotter is found before
someone spends an hour of a free-tier T4 on the run that feeds it.

The rest of the suite checks the properties that make a figure evidence rather
than decoration: the identity panels really do reproduce the identity, a
missing measurement is reported instead of drawn around, and every figure
writes the numbers it plotted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
sys.path.insert(0, str(ROOT / "src"))

import fig_curvature  # noqa: E402
import fig_drift  # noqa: E402
import fig_horizon  # noqa: E402
import fig_inversion  # noqa: E402
import fig_leverage  # noqa: E402
import fig_quintic  # noqa: E402
import fig_results  # noqa: E402
import style  # noqa: E402

FIGURES = ROOT / "artifacts" / "figures"


# ---------------------------------------------------------------------------
# The claims the identity panels make are actually identities
# ---------------------------------------------------------------------------


def test_row_mass_equals_leverage_to_machine_precision() -> None:
    result = fig_leverage.measure_identity(rows=96, cols=24)
    assert result["max_abs_error"] < 1e-12, result["max_abs_error"]
    assert result["sum_measured"] == pytest.approx(24.0, abs=1e-9)


def test_row_normalisation_is_inert_on_wide_matrices() -> None:
    """The first corollary: for m <= n every row norm is exactly 1."""
    result = fig_leverage.measure_aspect(width=64, ratios=(0.25, 0.5, 1.0, 4.0))
    wide = result["spread"][:3]
    assert max(wide) < 1e-12, wide
    assert result["spread"][-1] > 1e-3, "tall matrices must show real spread"


def test_fusion_starves_the_small_blocks_and_splitting_does_not() -> None:
    result = fig_leverage.measure_fusion(width=48, imbalances=(1.0, 0.01))
    assert result["fused"][-1][2] > 0.9, result["fused"][-1]
    for share in result["split"][-1]:
        assert share == pytest.approx(1 / 3, abs=1e-9)


def test_the_quintic_fixed_points_are_where_the_paper_says() -> None:
    points = fig_quintic.fixed_points()
    assert points[0] == pytest.approx(0.868, abs=5e-4)
    assert points[-1] == pytest.approx(1.264, abs=5e-4)


def test_more_iterations_do_not_narrow_the_band() -> None:
    """This is what 'cannot converge at any budget' means operationally."""
    bands, _ = fig_quintic.measure_band(rows=96, cols=48, budgets=(5, 8, 12))
    widths = {steps: values.max() - values.min() for steps, values in bands.items()}
    assert widths[12] > 0.3, widths
    assert widths[12] == pytest.approx(widths[8], abs=0.02), widths


@pytest.mark.parametrize("rows,cols", [(64, 32), (128, 64), (256, 64)])
def test_the_solved_schedule_removes_the_drift_the_quintic_has(rows, cols) -> None:
    """The magnitude of the drift depends on the shape -- 0.146 at 64x32 and
    0.233 at 128x64 -- so the claim under test is the relative one: Muon's
    step size moves with conditioning by an order of magnitude more than the
    solved schedule's does."""
    drift, _ = fig_quintic.measure_drift(rows=rows, cols=cols)
    muon_swing = max(drift["muon5"]) - min(drift["muon5"])
    polar_swing = max(drift["polar7"]) - min(drift["polar7"])
    assert muon_swing > 0.10, (rows, cols, muon_swing)
    assert polar_swing < 0.02, (rows, cols, polar_swing)
    assert muon_swing > 8 * polar_swing, (muon_swing, polar_swing)


# ---------------------------------------------------------------------------
# Statistics reported on the figures
# ---------------------------------------------------------------------------


def test_the_sign_test_reports_its_own_floor() -> None:
    """3/3 cannot beat p = 0.25, and the figure must not imply otherwise."""
    assert fig_results.sign_test([-1.0, -1.0, -1.0])[1] == pytest.approx(0.25)
    assert fig_results.sign_test([-1.0] * 8)[1] == pytest.approx(2 / 256)
    assert fig_results.sign_test([-1.0, 1.0])[1] == pytest.approx(1.0)


def test_matched_loss_alignment_refuses_non_overlapping_runs() -> None:
    """Interpolating outside the overlap would invent comparisons."""
    with pytest.raises(ValueError, match="do not overlap"):
        style.align_on_loss([5.0, 4.0], [1.0, 2.0], [3.0, 2.0], [1.0, 2.0])


def test_matched_loss_alignment_interpolates_within_the_overlap() -> None:
    axis, first, second = style.align_on_loss(
        [4.0, 3.0, 2.0], [10.0, 20.0, 30.0],
        [3.5, 2.5, 1.5], [100.0, 200.0, 300.0], grid_points=5)
    assert axis.min() == pytest.approx(2.0)
    assert axis.max() == pytest.approx(3.5)
    assert len(first) == len(second) == 5


# ---------------------------------------------------------------------------
# The GPU-fed plotters run, without a GPU
# ---------------------------------------------------------------------------


def _curvature_records() -> dict:
    """Shaped like measure_curvature.py's output, with a planted NDS gap."""
    records = []
    for name, nds, offset in (("muon", 4.0e-5, 0.0), ("astro", 2.5e-5, -0.05)):
        for index, step in enumerate([0, 50, 100, 150]):
            norm = 90.0 + index
            records.append({
                "optimizer": name, "seed": 100, "step": step,
                "val_loss": 7.0 - 0.25 * index + offset,
                "realized": 0.05 - 0.004 * index,
                "predicted": 0.048 - 0.004 * index,
                "first_order": 0.09 - 0.003 * index,
                "curvature_penalty": 0.5 * nds * norm,
                "update_norm_sq": norm,
                "nds": nds,
            })
    return {"records": records, "size": "124M", "configs": {}}


def _drift_records() -> dict:
    records = []
    for name, base in (("muon", 0.86), ("astro", 1.0)):
        for step in (0, 10, 20):
            for tensor in range(3):
                records.append({
                    "optimizer": name, "seed": 100, "step": step,
                    "tensor": f"group0.param{tensor}",
                    "ratio": base + 0.02 * tensor,
                    "sigma_max": 1.2, "sigma_min": 0.7, "sigma_mean": 0.95,
                    "rows": 768, "cols": 768, "loss": 7.0 - 0.1 * step,
                    "momentum_condition": 10.0 ** (1 + tensor),
                })
    return {"records": records, "size": "124M", "steps": 30}


@pytest.mark.parametrize("module,payload,stem", [
    (fig_curvature, _curvature_records(), "fig3_curvature"),
    (fig_drift, _drift_records(), "fig7_drift"),
])
def test_gpu_fed_plotters_draw_from_synthetic_measurements(
        module, payload, stem, tmp_path) -> None:
    path = tmp_path / "measurement.json"
    path.write_text(json.dumps(payload))
    target = FIGURES / f"{stem}.png"
    before = target.stat().st_mtime if target.exists() else None

    module.build(str(path))

    assert target.exists(), f"{stem} was not written"
    if before is not None:
        assert target.stat().st_mtime > before


def test_the_curvature_figure_identifies_which_factor_carries_the_gap() -> None:
    """The planted data has equal update norms and a 1.6x NDS gap, so the
    figure must attribute the difference to direction, not step size."""
    path = FIGURES / "fig3_curvature.json"
    summary = json.loads(path.read_text())["ratios_vs_muon"]["astro"]
    assert summary["mean_norm_ratio"] == pytest.approx(1.0, abs=0.01)
    assert summary["mean_nds_ratio"] < 0.8


# ---------------------------------------------------------------------------
# Absent measurements are reported, not drawn around
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", [fig_curvature, fig_drift])
def test_a_missing_measurement_is_announced_and_not_faked(module, capsys) -> None:
    module.build("definitely-not-a-real-file.json")
    printed = capsys.readouterr().out
    assert "SKIP" in printed
    assert "produce it with" in printed


def test_the_horizon_figure_says_when_the_scaling_run_is_absent(capsys) -> None:
    fig_horizon.build("measured.json", "no-such-scaling-state.json")
    assert "NOT MEASURED" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Every figure leaves its numbers behind
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stem", ["fig1_leverage", "fig2_quintic",
                                  "fig4_results", "fig5_inversion",
                                  "fig6_horizon"])
def test_each_figure_writes_the_numbers_it_drew(stem) -> None:
    """A figure whose numbers are not on disk cannot be checked by a reader."""
    for builder, name in ((fig_leverage.build, "fig1_leverage"),
                          (fig_quintic.build, "fig2_quintic"),
                          (fig_results.build, "fig4_results"),
                          (fig_inversion.build, "fig5_inversion"),
                          (fig_horizon.build, "fig6_horizon")):
        if name == stem:
            builder()
    assert (FIGURES / f"{stem}.json").is_file()
    assert (FIGURES / f"{stem}.png").is_file()
    assert (FIGURES / f"{stem}.pdf").is_file()


def test_measured_numbers_match_the_round4_writeup() -> None:
    """artifacts/measured.json is the figures' source; it must agree with the
    prose that quotes the same runs."""
    measured = json.loads((ROOT / "artifacts" / "measured.json").read_text())
    block = measured["gpt2_124m_fineweb"]["losses"]
    text = (ROOT / "docs" / "ROUND4.md").read_text()
    for name, values in block.items():
        for value in values:
            assert f"{value:.4f}" in text, (
                f"{name} seed value {value} is in measured.json but not in ROUND4.md")
