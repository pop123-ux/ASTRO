.PHONY: test figures paper paper-primary

PYTHON ?= python

test:
	$(PYTHON) -m pytest

figures:
	$(PYTHON) scripts/figures/make_all.py

paper-primary:
	$(PYTHON) scripts/figures/make_all.py --only optimizer_journey horizon_v2 efficiency_v2 training_trajectories leverage quintic

paper:
	$(PYTHON) scripts/paper/build.py
