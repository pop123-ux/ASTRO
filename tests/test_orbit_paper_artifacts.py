"""Tests for the strict ORBIT paper evidence-freeze layer."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import orbit_paper_artifacts as paper  # noqa: E402
import orbit_build_paper as build_paper  # noqa: E402


ENV = {
    "python": "3.13.15",
    "torch": "2.11.0+cu128",
    "cuda": "12.8",
    "transformers": "4.57.6",
    "datasets": "4.8.5",
}
CFG = {"lr": 0.018, "scalar_lr_mult": 0.49, "weight_decay": 0.006}


def row(phase: str, optimizer: str, seed: int, idx: int, *, label=None, trial=None):
    base = {
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
        base["analysis_label"] = label
    if trial is not None:
        base["trial"] = trial
        base["config_id"] = f"shared-{trial:02d}"
    return base


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
    cells = (
        ("muon", "muon_at_muon_config"),
        ("muon", "muon_at_orbit_config"),
        ("orbit", "orbit_at_muon_config"),
        ("orbit", "orbit_at_orbit_config"),
    )
    for opt, label in cells:
        for i, seed in enumerate(range(100, 105)):
            xconfig.append(row("xconfig", opt, seed, i, label=label))
    write_jsonl(merged / "xconfig.jsonl", xconfig)

    matched_tune = []
    for trial in range(10):
        for opt in ("muon", "orbit"):
            matched_tune.append(
                row("matched_tune", opt, 0, trial, trial=trial)
            )
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

    horizon = []
    scale = []
    for opt in ("muon", "normuon", "astro_v2", "orbit"):
        for i, seed in enumerate((400, 401)):
            horizon.append(row("horizon_with_astro", opt, seed, i))
        for i, seed in enumerate((300, 301)):
            scale.append(row("scale_with_astro", opt, seed, i))
    write_jsonl(merged / "horizon_with_astro.jsonl", horizon)
    write_jsonl(merged / "scale_with_astro.jsonl", scale)

    best = {
        "muon": {
            "config": dict(CFG),
            "config_id": "shared-04",
            "code_digest": paper.CORE_DIGEST,
        },
        "orbit": {
            "config": dict(CFG),
            "config_id": "shared-04",
            "code_digest": paper.CORE_DIGEST,
        },
    }
    (merged / "matched_best_configs.json").write_text(json.dumps(best))


def test_strict_freeze_accepts_complete_exact_fixture(tmp_path):
    build_fixture(tmp_path)
    results, manifest = paper.build(tmp_path)
    assert manifest["status"] == "paper_ready"
    assert results["matched_confirmation"]["orbit_vs_muon"]["n"] == 10
    assert results["mechanism_ablation"]["orbit_vs_identity"]["n"] == 10
    assert results["long_horizon_transfer"]["orbit_vs_astro_v2"]["n"] == 2
    assert results["scale_transfer"]["orbit_vs_astro_v2"]["n"] == 2
    assert (tmp_path / "paper_artifacts" / "manifest.json").is_file()
    assert (tmp_path / "paper_artifacts" / "paper_results.json").is_file()


def test_strict_freeze_refuses_missing_ablation(tmp_path):
    build_fixture(tmp_path)
    (tmp_path / "merged" / "ablation_ext.jsonl").unlink()
    try:
        paper.build(tmp_path)
    except SystemExit as exc:
        assert "ablation_ext.jsonl" in str(exc)
    else:
        raise AssertionError("strict evidence freeze accepted a missing ablation phase")


def test_paper_facing_astro_name_is_clean():
    paper_paths = [
        ROOT / "docs" / "orbit" / "paper" / "main.tex",
        ROOT / "docs" / "orbit" / "paper" / "methods.tex",
    ]
    forbidden = ("ASTRO-v2", "astro_v2", r"astro\_v2")
    for path in paper_paths:
        text = path.read_text()
        for token in forbidden:
            assert token not in text, f"internal ASTRO label leaked into {path}: {token}"

    # Plot/build code may use the frozen internal identifier to read historical
    # artifacts, but it must never render the public label as ASTRO-v2.
    for path in (
        ROOT / "scripts" / "orbit_plot.py",
        ROOT / "scripts" / "orbit_build_paper.py",
    ):
        assert "ASTRO-v2" not in path.read_text()



def test_paper_method_contains_orbit_algorithm_and_no_direct_address():
    methods = (ROOT / "docs" / "orbit" / "paper" / "methods.tex").read_text()
    assert r"\begin{algorithm}" in methods
    assert r"\label{alg:orbit}" in methods
    assert "ORBIT update for a RoPE query--key projection pair" in methods

    paper_text = "\n".join(
        (ROOT / "docs" / "orbit" / "paper" / name).read_text()
        for name in ("main.tex", "methods.tex", "astro_provenance.tex")
    )
    assert not re.search(r"\b(?:you|your|we|our)\b", paper_text, flags=re.IGNORECASE)



def test_paper_ends_without_appendix_scaffolding():
    main = (ROOT / "docs" / "orbit" / "paper" / "main.tex").read_text()
    assert r"\appendix" not in main
    assert "astro_provenance.tex" not in main
    assert r"\section{Reproducibility}" not in main
    assert r"\section{Matched Training Dynamics}" not in main
    assert r"\section{Optimizer Diagnostics}" not in main
    assert r"\section{Broad Confirmation Visualization}" not in main
    methods = (ROOT / "docs" / "orbit" / "paper" / "methods.tex").read_text()
    assert r"\subsection{Compute and reproducibility}" not in methods
    assert "reproducibility artifacts" in methods
    assert "automated consistency checks" in methods



def test_paper_avoids_internal_identifiers():
    paper_text = "\n".join(
        (ROOT / "docs" / "orbit" / "paper" / name).read_text()
        for name in ("main.tex", "methods.tex")
    )
    forbidden = (
        "HuggingFaceFW/",
        "shared-04",
        "de8b994a734276871770c6c67648117d3613d0954f3ae90d7e7b69246308c200",
        "environment fingerprint",
        "JSONL",
    )
    for token in forbidden:
        assert token not in paper_text, f"internal identifier leaked into manuscript: {token}"



def test_manuscript_uses_publication_facing_names_only():
    main = (ROOT / "docs" / "orbit" / "paper" / "main.tex").read_text()
    methods = (ROOT / "docs" / "orbit" / "paper" / "methods.tex").read_text()
    paper_text = main + "\n" + methods

    assert "Independent Researcher" in main
    assert "alexandrupp55@gmail.com" in main
    assert "FineWeb-Edu" in paper_text

    forbidden = (
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        "shared-04",
        "de8b994a734276871770c6c67648117d3613d0954f3ae90d7e7b69246308c200",
        "astro_v2",
        r"astro\_v2",
        "ASTRO-v2",
    )
    for token in forbidden:
        assert token not in paper_text, f"implementation-facing token leaked into paper: {token}"


def test_generated_macros_define_then_override(tmp_path):
    build_fixture(tmp_path)
    results, _ = paper.build(tmp_path)
    generated = tmp_path / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    build_paper.write_macros(results, generated)
    text = (generated / "macros.tex").read_text()
    assert r"\newcommand{\MatchedMuonLoss}{--}" in text
    assert r"\renewcommand{\MatchedMuonLoss}{" in text
    assert r"\newcommand{\AblIdentityDelta}{--}" in text
    assert r"\renewcommand{\AblIdentityDelta}{" in text
    assert r"\\n\\newcommand" not in text
    assert len(text.splitlines()) > 10
    assert text.splitlines()[0] == "% AUTO-GENERATED. Do not edit."
