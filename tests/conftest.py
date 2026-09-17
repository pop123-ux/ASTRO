"""Shared pytest isolation for the mixed historical/current research harnesses.

The confirmatory ``paper_campaign`` deliberately monkey-patches the historical
``astro_lab`` module when imported.  Pytest imports every test module during
collection, so without isolation those patches leak into tests whose purpose is
to exercise the historical lab itself.  The production scripts are left
untouched here because their byte fingerprint is already attached to an active
confirmatory campaign.
"""

from __future__ import annotations

import importlib

import pytest
import torch


@pytest.fixture(autouse=True)
def isolate_test_protocols(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """Keep tests from changing the protocol they think they are exercising."""
    module_name = getattr(request.module, "__name__", "")

    if module_name.endswith("test_sweep_plumbing"):
        # test_sweep_plumbing predates paper_campaign and explicitly tests the
        # base astro_lab CLI/state machine.  paper_campaign is imported earlier
        # during collection and patches that module object in-place, so reload
        # the frozen base implementation before the sweep fixture stubs training.
        import astro_lab

        importlib.reload(astro_lab)

    if request.node.name == "test_unknown_schedule_is_rejected":
        # This unit test is about argument validation, not corpus availability.
        # CI intentionally does not download benchmark corpora, so provide the
        # minimum in-memory corpus needed to reach the schedule guard.
        from astro.bench import llm

        class _DummyCorpus:
            vocab_size = 32
            train = torch.zeros(512, dtype=torch.long)

        monkeypatch.setattr(llm, "get_corpus", lambda _name: _DummyCorpus())
        monkeypatch.setattr(llm, "_validation_batches", lambda *args, **kwargs: [])

    yield
