#!/usr/bin/env python3
"""Fail closed unless the audited ORBIT paper evidence is release-ready."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import orbit_campaign as campaign


REQUIRED_RESULTS = {
    "matched_confirmation",
    "cross_configuration_isolation",
    "mechanism_ablation",
    "long_horizon_transfer",
    "scale_transfer",
    "broad_independently_tuned_context",
}

EXPECTED_SOURCE_ROWS = {
    "confirm": 30,
    "xconfig": 20,
    "matched_tune": 10,
    "matched_confirm": 20,
    "ablation_ext": 40,
    "horizon_with_astro": 8,
    "scale_with_astro": 8,
}


def fail(message: str) -> None:
    raise SystemExit("RELEASE BLOCKED: " + message)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--paper-dir", type=Path, default=Path("docs/orbit/paper"))
    args = p.parse_args()

    artifacts = args.work_dir / "paper_artifacts"
    manifest_path = artifacts / "manifest.json"
    results_path = artifacts / "paper_results.json"
    best_path = args.work_dir / "merged" / "matched_best_configs.json"

    for path in (manifest_path, results_path, best_path):
        if not path.exists():
            fail(f"missing {path}")

    manifest = json.loads(manifest_path.read_text())
    results = json.loads(results_path.read_text())
    best = json.loads(best_path.read_text())
    digest = campaign.code_digest()

    if manifest.get("status") != "paper_ready":
        fail(f"manifest status is {manifest.get('status')!r}, expected 'paper_ready'")
    if manifest.get("warnings"):
        fail("paper manifest contains warnings")
    if manifest.get("core_digest") != digest:
        fail("paper manifest is not bound to the checked-out audited implementation")
    if results.get("core_digest") != digest:
        fail("paper_results.json is not bound to the checked-out audited implementation")

    if best.get("selection_rule") != "muon_winner_from_shared_grid":
        fail("primary recipe was not selected by the Muon-only rule")
    for optimizer in ("muon", "orbit"):
        record = best.get(optimizer, {})
        if record.get("selected_by") != "muon":
            fail(f"{optimizer} primary recipe provenance is not Muon-selected")
        if record.get("code_digest") != digest:
            fail(f"{optimizer} primary recipe was frozen under another implementation")
    if best["muon"].get("config") != best["orbit"].get("config"):
        fail("Muon and ORBIT do not share one frozen primary configuration")
    if best["muon"].get("config_id") != best["orbit"].get("config_id"):
        fail("Muon and ORBIT do not share one primary config ID")

    sources = manifest.get("sources", {})
    for phase, expected_rows in EXPECTED_SOURCE_ROWS.items():
        source = sources.get(phase)
        if not source:
            fail(f"missing source manifest for {phase}")
        if int(source.get("rows", -1)) != expected_rows:
            fail(f"{phase} has {source.get('rows')} rows, expected {expected_rows}")
        if source.get("code_digests") != [digest]:
            fail(f"{phase} contains evidence from another implementation digest")

    missing_results = sorted(REQUIRED_RESULTS - set(results))
    if missing_results:
        fail("missing result blocks: " + ", ".join(missing_results))

    primary = results["matched_confirmation"]
    if int(primary["orbit_vs_muon"]["n"]) != 10:
        fail("primary paired comparison is not n=10")
    for key in ("orbit_vs_identity", "orbit_vs_norope", "orbit_vs_diag"):
        if int(results["mechanism_ablation"][key]["n"]) != 10:
            fail(f"mechanism ablation {key} is not n=10")

    pdf = args.paper_dir / "main.pdf"
    if not pdf.exists() or pdf.stat().st_size == 0:
        fail(f"compiled paper missing at {pdf}")

    print("ORBIT RELEASE CHECK: PASS")
    print(f"implementation digest: {digest}")
    print(f"primary config: {best['muon'].get('config_id')}")
    print(f"paper: {pdf}")


if __name__ == "__main__":
    main()
