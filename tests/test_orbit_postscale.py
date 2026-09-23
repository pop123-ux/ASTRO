import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import orbit_campaign as base
import orbit_postscale as post
import orbit_postscale_merge as merge


def write_legacy_best(tmp_path: Path) -> None:
    merged = tmp_path / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    best = {
        "muon": {"config": {"lr": 0.07, "scalar_lr_mult": 0.05, "weight_decay": 0.2}},
        "orbit": {"config": {"lr": 0.02, "scalar_lr_mult": 0.15, "weight_decay": 0.07}},
        "astro_v2": {"config": {"lr": 0.03, "scalar_lr_mult": 0.1, "weight_decay": 0.02}},
    }
    (merged / "best_configs.json").write_text(json.dumps(best))


def write_matched_best(tmp_path: Path) -> None:
    merged = tmp_path / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    best = {
        "muon": {"config": {"lr": 0.04, "scalar_lr_mult": 0.1, "weight_decay": 0.05}},
        "orbit": {"config": {"lr": 0.03, "scalar_lr_mult": 0.1, "weight_decay": 0.05}},
    }
    (merged / "matched_best_configs.json").write_text(json.dumps(best))


def test_matched_tune_uses_same_ten_candidates_for_both_optimizers(tmp_path):
    tasks = post.make_tasks("matched_tune", work_dir=tmp_path)
    assert len(tasks) == 20
    assert {task["optimizer"] for task in tasks} == {"muon", "orbit"}
    assert {task["trial"] for task in tasks} == set(range(10))

    for trial in range(10):
        pair = [task for task in tasks if task["trial"] == trial]
        assert len(pair) == 2
        assert pair[0]["config"] == pair[1]["config"]
        assert pair[0]["config_id"] == pair[1]["config_id"] == f"shared-{trial:02d}"


def test_xconfig_is_complete_two_by_two_cross_on_five_seeds(tmp_path):
    write_legacy_best(tmp_path)
    tasks = post.make_tasks("xconfig", work_dir=tmp_path)
    assert len(tasks) == 20

    labels = {
        "muon_at_muon_config",
        "muon_at_orbit_config",
        "orbit_at_muon_config",
        "orbit_at_orbit_config",
    }
    assert {task["analysis_label"] for task in tasks} == labels
    for label in labels:
        rows = [task for task in tasks if task["analysis_label"] == label]
        assert {task["seed"] for task in rows} == {100, 101, 102, 103, 104}


def test_matched_confirm_and_extended_ablation_counts(tmp_path):
    write_matched_best(tmp_path)

    confirm = post.make_tasks("matched_confirm", work_dir=tmp_path)
    assert len(confirm) == 20
    assert {task["seed"] for task in confirm} == set(range(500, 510))
    assert {task["optimizer"] for task in confirm} == {"muon", "orbit"}

    ablation = post.make_tasks("ablation_ext", work_dir=tmp_path)
    assert len(ablation) == 40
    assert {task["seed"] for task in ablation} == set(range(600, 610))
    assert {task["optimizer"] for task in ablation} == set(post.EXTENDED_ABLATIONS)
    configs = {json.dumps(task["config"], sort_keys=True) for task in ablation}
    assert len(configs) == 1


def test_astro_followups_only_run_missing_strong_baseline(tmp_path):
    write_legacy_best(tmp_path)

    horizon = post.make_tasks("astro_horizon", work_dir=tmp_path)
    scale = post.make_tasks("astro_scale", work_dir=tmp_path)

    assert len(horizon) == 2
    assert len(scale) == 2
    assert {task["optimizer"] for task in horizon + scale} == {"astro_v2"}
    assert {task["seed"] for task in horizon} == {400, 401}
    assert {task["seed"] for task in scale} == {300, 301}


def test_paired_delta_reports_seedwise_wins():
    rows = []
    for seed, orbit_loss, muon_loss in [
        (1, 1.0, 1.2),
        (2, 1.1, 1.3),
        (3, 0.9, 1.0),
    ]:
        rows.append({"analysis_label": "orbit", "optimizer": "orbit", "seed": seed, "val_loss": orbit_loss})
        rows.append({"analysis_label": "muon", "optimizer": "muon", "seed": seed, "val_loss": muon_loss})

    result = merge.paired_delta(rows, "orbit", "muon")
    assert result["n"] == 3
    assert result["a_wins"] == 3
    assert result["b_wins"] == 0
    assert result["mean_delta"] < 0


def test_core_digest_does_not_include_postscale_script():
    assert len(base.code_digest()) == 64
    assert len(post.postscale_digest()) == 64
