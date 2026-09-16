# ASTRO paper run order

This is the concise execution order. The longer rationale and claim rules live in
`docs/PAPER_PROOF_CAMPAIGN.md`; the novelty boundary lives in
`docs/PRIOR_ART_2026_AUDIT.md`.

The sequence below is intentionally smaller than a typical optimizer-zoo sweep.
It keeps only experiments that answer a distinct paper question.

## What recent optimizer papers make necessary here

For ASTRO's current claim, the minimum evidence is:

1. **fair direct baselines** under one trainer and one data stream;
2. **held-out-seed confirmation** after hyperparameters are frozen;
3. **a faithful AdaMuon baseline**, not the old local approximation;
4. **causal controls for the parts that differ from prior art**;
5. **one longer-horizon check**, because 900 steps is an early-training regime;
6. runtime disclosure;
7. a current prior-art audit.

A second model scale is useful, but it is not required unless the paper claims
scale transfer or broad scalability. A separate time-matched campaign is not
required unless the paper claims wall-clock superiority.

---

## 0. Before GPU experiments: code / novelty gate

```python
%cd /content/ASTRO
!git fetch origin
!git checkout paper-proof-campaign
!git pull origin paper-proof-campaign
!pip -q install transformers datasets matplotlib numpy pytest

!pytest -q tests/test_paper_campaign.py tests/test_paper_mechanism.py
!python scripts/novelty_gate.py
!python scripts/prepare_paper_data.py
!git rev-parse HEAD
```

Do not start the GPU campaign if `novelty_gate.py` fails. It verifies that the
`astro_v2_blockwise` control reproduces split-NorMuon semantics and that full
ASTRO-v2 is numerically distinct from that prior-art-style blockwise rule.

This is a **structural code check**, not proof that the literature contains no
identical method.

---

## A. Mandatory headline experiment

Question: under a fairness-controlled GPT-2 124M / FineWeb-Edu protocol, how do
the relevant optimizers compare after equal tuning effort?

```python
%cd /content/ASTRO

!python scripts/paper_campaign.py \
  --mode scaling \
  --sizes 124M \
  --steps 900 \
  --optimizers \
    adamw \
    muon \
    normuon \
    adamuon_ref \
    astro_v2 \
  --trials 5 \
  --seeds 5 \
  --work-dir /content/drive/MyDrive/astro/paper_main \
  --max-minutes 180 \
  --stop-after 5
```

Repeat the **identical command** until no runs remain. Tuning uses seed 0;
confirmation uses held-out seeds 100--104. Once those seeds are complete, the
configs are frozen permanently for the rest of the paper.

### Plot immediately after A

```python
%cd /content/ASTRO
!python scripts/paper_plot.py --stage main
```

Output: `artifacts/figures/fig_paper_main.{pdf,png,json}`.

---

## B. Mandatory mechanism / novelty-control experiment

Question: what part of the full ASTRO-v2 recipe adds value, and does the proposed
global post-recombination redistribution improve over the public split-NorMuon
pattern?

No ablation is retuned.

```python
%cd /content/ASTRO

!python scripts/paper_mechanism.py \
  --main-state /content/drive/MyDrive/astro/paper_main/astro_lab_state.json \
  --seeds 200 201 \
  --work-dir /content/drive/MyDrive/astro/paper_mechanism \
  --max-minutes 180 \
  --stop-after 6
```

Repeat unchanged until complete.

The required contrasts are:

```text
muon_m90 - muon                    matrix-beta confound
astro_v2 - astro_v2_gamma0         post-spectral row adaptation
astro_v2 - astro_v2_nosplit        semantic Q/K/V splitting
astro_v2 - astro_v2_blockwise      global vs split-NorMuon-style blockwise restore
```

The final contrast is the only one that can support the current narrow
ASTRO-specific structural contribution. QKV splitting and NorMuon-style row
adaptation themselves are already prior art.

### Plot immediately after B

```python
%cd /content/ASTRO
!python scripts/paper_plot.py --stage mechanism
```

Output: `artifacts/figures/fig_paper_ablation.{pdf,png,json}`.

---

## C. Mandatory durability experiment

Question: does the frozen ASTRO-v2 result obviously collapse when training is
extended 3x?

Only Muon and ASTRO-v2 are needed. This is a durability check, not a new
leaderboard.

```python
%cd /content/ASTRO

!python scripts/paper_transfer.py \
  --main-state /content/drive/MyDrive/astro/paper_main/astro_lab_state.json \
  --size 124M \
  --steps 2700 \
  --seeds 400 401 \
  --work-dir /content/drive/MyDrive/astro/paper_horizon_2700 \
  --max-minutes 180 \
  --stop-after 2
```

Repeat unchanged until all four runs are complete.

### Plot immediately after C

```python
%cd /content/ASTRO
!python scripts/paper_plot.py --stage horizon
```

Output: `artifacts/figures/fig_paper_horizon.{pdf,png,json}`.

At this point there is enough experimental material to begin the full paper
draft **if A, B, and C support a coherent claim**.

---

## D. Optional scale-transfer experiment

Run this only if the paper will claim that the frozen recipe transfers beyond
124M, or if compute is available and you want stronger evidence before
submission.

```python
%cd /content/ASTRO

!python scripts/paper_transfer.py \
  --main-state /content/drive/MyDrive/astro/paper_main/astro_lab_state.json \
  --size 355M \
  --steps 900 \
  --seeds 300 301 \
  --work-dir /content/drive/MyDrive/astro/paper_scale_355m \
  --max-minutes 180 \
  --stop-after 2
```

### Plot after D

```python
%cd /content/ASTRO
!python scripts/paper_plot.py --stage scale
```

Output: `artifacts/figures/fig_paper_scale.{pdf,png,json}`.

---

## Final audit before writing claims

For the mandatory A--C campaign:

```python
%cd /content/ASTRO

!python scripts/paper_audit.py \
  --main-state /content/drive/MyDrive/astro/paper_main/astro_lab_state.json \
  --mechanism-state /content/drive/MyDrive/astro/paper_mechanism/paper_mechanism_state.json \
  --horizon-state /content/drive/MyDrive/astro/paper_horizon_2700/paper_transfer_state.json \
  --out /content/drive/MyDrive/astro/paper_evidence_summary.md \
  --strict

!python scripts/paper_collect.py \
  --main /content/drive/MyDrive/astro/paper_main/astro_lab_state.json \
  --mechanism /content/drive/MyDrive/astro/paper_mechanism/paper_mechanism_state.json \
  --horizon /content/drive/MyDrive/astro/paper_horizon_2700/paper_transfer_state.json \
  --out artifacts/paper_campaign.json

!python scripts/figures/make_all.py --only \
  paper_main paper_ablation paper_horizon
```

If Stage D was also run, use:

```python
%cd /content/ASTRO

!python scripts/paper_audit.py \
  --main-state /content/drive/MyDrive/astro/paper_main/astro_lab_state.json \
  --mechanism-state /content/drive/MyDrive/astro/paper_mechanism/paper_mechanism_state.json \
  --horizon-state /content/drive/MyDrive/astro/paper_horizon_2700/paper_transfer_state.json \
  --scale-state /content/drive/MyDrive/astro/paper_scale_355m/paper_transfer_state.json \
  --require-scale \
  --out /content/drive/MyDrive/astro/paper_evidence_summary.md \
  --strict

!python scripts/paper_plot.py --stage all
```

---

## Do not run by default

Do not spend GPU time on these unless the manuscript creates a specific need:

```text
new 300/600-step grids
more than five headline seeds
another broad optimizer zoo
separate equal-wall-clock runs
774M scaling
new hyperparameter searches for every ablation
modded-nanoGPT replication before the first complete draft
```

Those are follow-ups, not prerequisites for a properly scoped first ASTRO paper.
