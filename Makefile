.PHONY: test figures paper paper-primary orbit-test orbit-gate orbit-plots orbit-paper orbit-merge

PYTHON ?= python
ORBIT_WORK ?= /tmp/orbit-paper

test:
	$(PYTHON) -m pytest

figures:
	$(PYTHON) scripts/figures/make_all.py

paper-primary:
	$(PYTHON) scripts/figures/make_all.py --only optimizer_journey horizon_v2 efficiency_v2 training_trajectories leverage quintic

paper:
	$(PYTHON) scripts/paper/build.py

orbit-test:
	$(PYTHON) -m pytest tests/test_orbit.py

orbit-gate:
	$(PYTHON) scripts/orbit_novelty_gate.py --strict

orbit-merge:
	$(PYTHON) scripts/orbit_merge.py --work-dir $(ORBIT_WORK) --phase all

orbit-plots:
	$(PYTHON) scripts/orbit_plot.py --work-dir $(ORBIT_WORK)

orbit-paper:
	$(PYTHON) scripts/orbit_build_paper.py --work-dir $(ORBIT_WORK)
