# ORBIT

**Branch:** `orbit-paper-campaign`

ORBIT is the third, deliberately isolated optimizer research track in this repository. It is
not an ASTRO rename and it does not modify ASTRO's historical result files.

The current frozen candidate is a **RoPE-aware functional Query--Key preconditioner**. It
keeps a normal Transformer inference graph and uses only tiny training-time 2x2 covariance
statistics for each head/frequency pair. ORBIT applies those statistics to Muon candidate
Q/K directions, then restores the original joint Q+K Frobenius update norm.

Start here:

- Method and equations: [`docs/orbit/METHOD.md`](docs/orbit/METHOD.md)
- Prior-art / novelty boundary: [`docs/orbit/PRIOR_ART_GATE.md`](docs/orbit/PRIOR_ART_GATE.md)
- Exact four-Colab commands: [`docs/orbit/COLAB_4WAY.md`](docs/orbit/COLAB_4WAY.md)
- Frozen experiment manifest: [`configs/orbit_protocol.json`](configs/orbit_protocol.json)
- Optimizer: [`src/orbit/optimizer.py`](src/orbit/optimizer.py)
- RoPE GPT research model: [`src/orbit/model.py`](src/orbit/model.py)
- Campaign runner: [`scripts/orbit_campaign.py`](scripts/orbit_campaign.py)
- Shard merger/config freezer: [`scripts/orbit_merge.py`](scripts/orbit_merge.py)
- Plotting: [`scripts/orbit_plot.py`](scripts/orbit_plot.py)
- Mechanism gate: [`scripts/orbit_novelty_gate.py`](scripts/orbit_novelty_gate.py)
- LaTeX builder: [`scripts/orbit_build_paper.py`](scripts/orbit_build_paper.py)
- Manuscript: [`docs/orbit/paper/main.tex`](docs/orbit/paper/main.tex)

## First local check

```bash
pip install -e . "transformers>=4.56,<5" "datasets>=3,<5" matplotlib pytest
python -m pytest tests/test_orbit.py
python scripts/orbit_novelty_gate.py --strict
```

## Research discipline

The branch is designed to make a failed hypothesis cheap. Discovery happens before held-out
confirmation; confirmation and ablations happen before 2700-step or 355M runs. Result merging
refuses mixed code fingerprints and mixed package environments. The paper skeleton contains
no positive numerical claim until generated artifacts exist.

The strongest novelty statement is intentionally **not** committed as a fact. The exact
frozen equation must survive the final search checklist in `PRIOR_ART_GATE.md` immediately
before public release.
