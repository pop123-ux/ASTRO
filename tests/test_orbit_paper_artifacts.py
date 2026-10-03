"""Regression tests for the release-facing ORBIT paper contract."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import orbit_build_paper as build_paper  # noqa: E402
import orbit_paper_artifacts as paper  # noqa: E402


ENV = {
    "python": "3.13.15",
    "torch": "2.11.0+cu128",
    "cuda": "12.8",
    "transformers": "4.57.6",
    "datasets": "4.8.5",
}
CFG = {"lr": 0.018, "scalar_lr_mult": 0.49, "weight_decay": 0.006}


def row(phase: str, optimizer: str, seed: int, idx: int, *, label=None, trial=None):
    item = {
        "phase": phase,
        "optimizer": optimizer,
        "seed": seed,
        "task_id": f"{phase}-{optimizer}-{label or 'default'}-{seed}-{idx}",
        "status": "ok",
        "code_digest": paper.CORE_DIGEST,
        "environment": ENV,
        "val_loss": 5.0 + 0.001 * idx + (0.004 if optimizer == "muon" else 0.0),
        "seconds": 100.0 + idx,
        "peak_cuda_bytes": 4 * 1024**3,
        "config": dict(CFG),
        "curve": [],
    }
    if label is not None:
        item["analysis_label"] = label
    if trial is not None:
        item["trial"] = trial
        item["config_id"] = f"shared-{trial:02d}"
    return item


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x) + "\n" for x in rows))


def build_fixture(tmp_path: Path) -> None:
    merged = tmp_path / "merged"
    merged.mkdir(parents=True, exist_ok=True)

    confirm = []
    for opt in ("adamw", "adamuon_ref", "muon", "normuon", "astro_v2", "orbit"):
        for i, seed in enumerate(range(100, 105)):
            confirm.append(row("confirm", opt, seed, i))
    write_jsonl(merged / "confirm.jsonl", confirm)

    xconfig = []
    for opt, label in (
        ("muon", "muon_at_muon_config"),
        ("muon", "muon_at_orbit_config"),
        ("orbit", "orbit_at_muon_config"),
        ("orbit", "orbit_at_orbit_config"),
    ):
        for i, seed in enumerate(range(100, 105)):
            xconfig.append(row("xconfig", opt, seed, i, label=label))
    write_jsonl(merged / "xconfig.jsonl", xconfig)

    matched_tune = []
    for trial in range(10):
        for opt in ("muon", "orbit"):
            matched_tune.append(row("matched_tune", opt, 0, trial, trial=trial))
    write_jsonl(merged / "matched_tune.jsonl", matched_tune)

    matched_confirm = []
    for opt in ("muon", "orbit"):
        for i, seed in enumerate(range(500, 510)):
            matched_confirm.append(row("matched_confirm", opt, seed, i))
    write_jsonl(merged / "matched_confirm.jsonl", matched_confirm)

    ablation = []
    for opt in ("orbit", "orbit_identity", "orbit_norope", "orbit_diag"):
        for i, seed in enumerate(range(600, 610)):
            ablation.append(row("ablation_ext", opt, seed, i))
    write_jsonl(merged / "ablation_ext.jsonl", ablation)

    for phase, seeds in (("horizon_with_astro", (400, 401)), ("scale_with_astro", (300, 301))):
        rows = []
        for opt in ("muon", "normuon", "astro_v2", "orbit"):
            for i, seed in enumerate(seeds):
                rows.append(row(phase, opt, seed, i))
        write_jsonl(merged / f"{phase}.jsonl", rows)

    best = {
        "muon": {"config": dict(CFG), "config_id": "shared-04", "code_digest": paper.CORE_DIGEST},
        "orbit": {"config": dict(CFG), "config_id": "shared-04", "code_digest": paper.CORE_DIGEST},
    }
    (merged / "matched_best_configs.json").write_text(json.dumps(best))


def test_strict_freeze_accepts_complete_paper_fixture(tmp_path):
    build_fixture(tmp_path)
    results, manifest = paper.build(tmp_path)
    assert manifest["status"] == "paper_ready"
    assert results["matched_confirmation"]["orbit_vs_muon"]["n"] == 10
    assert results["mechanism_ablation"]["orbit_vs_identity"]["n"] == 10
    assert results["long_horizon_transfer"]["orbit_vs_astro_v2"]["n"] == 2


def test_strict_freeze_rejects_wrong_core_digest(tmp_path):
    build_fixture(tmp_path)
    path = tmp_path / "merged" / "matched_confirm.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows[0]["code_digest"] = "not-the-paper-implementation"
    write_jsonl(path, rows)
    try:
        paper.build(tmp_path)
    except (SystemExit, ValueError) as exc:
        assert "digest" in str(exc).lower()
    else:
        raise AssertionError("strict evidence freeze accepted a mismatched implementation")


def test_manuscript_states_forward_and_optimizer_rotations_separately():
    main = (ROOT / "docs" / "orbit" / "paper" / "main.tex").read_text()
    methods = (ROOT / "docs" / "orbit" / "paper" / "methods.tex").read_text()
    text = main + "\n" + methods
    normalized_main = " ".join(main.split())
    normalized_methods = " ".join(methods.split())

    assert r"R_f(-\Delta)" in text
    assert r"R_f(+\Delta)" in text
    assert "optimizer design choice" in normalized_main
    assert "not an exact score-side pullback" in normalized_main
    assert r"R_f(+\Delta)S_{K,f}R_f(+\Delta)^\top" in methods
    assert r"R_f(+\Delta)^\top S_{Q,f}R_f(+\Delta)" in methods
    assert "exact causal-score pullback" in normalized_methods


def test_manuscript_presentation_and_github_links():
    main = (ROOT / "docs" / "orbit" / "paper" / "main.tex").read_text()
    methods = (ROOT / "docs" / "orbit" / "paper" / "methods.tex").read_text()

    assert r"\usepackage{fontawesome5}" in main
    assert r"\href{https://github.com/pop123-ux/ORBIT}{\faGithub}" in main
    assert r"\href{https://github.com/pop123-ux}{\faGithub\;pop123-ux}" in main
    assert "Independent Researcher" in main
    assert "alexandrupp55@gmail.com" in main
    assert "Toward Function-Aware Optimization" in main
    assert "ORBIT update rule" in methods
    assert r"\hrule height 0.8pt" in methods
    assert "355M" not in main.split(r"\section{Discussion}")[0]

    paper_text = main + "\n" + methods
    assert not re.search(r"\b(?:you|your|we|our)\b", paper_text, flags=re.IGNORECASE)


def test_primary_protocol_is_shared_grid_with_coincident_winner():
    main = (ROOT / "docs" / "orbit" / "paper" / "main.tex").read_text()
    methods = (ROOT / "docs" / "orbit" / "paper" / "methods.tex").read_text()
    normalized = " ".join((main + " " + methods).split())
    assert "both independently select the same configuration" in normalized
    assert "independently attain their lowest tuning loss at the same candidate" in normalized
    assert "same 10 candidates for both" in normalized
    assert "Muon alone is tuned" not in normalized


def test_generated_macros_and_table_build(tmp_path):
    build_fixture(tmp_path)
    results, _ = paper.build(tmp_path)
    generated = tmp_path / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    build_paper.write_macros(results, generated)
    build_paper.write_matched_table(results, generated)
    macros = (generated / "macros.tex").read_text()
    table = (generated / "table_matched.tex").read_text()
    assert r"\renewcommand{\MatchedMuonLoss}{" in macros
    assert "same configuration was the lowest-loss shared-grid candidate for both optimizers" in table
    assert "selected shared recipe" in table
