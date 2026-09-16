# ASTRO paper-proof campaign

This is the **minimal confirmatory sequence** for the first ASTRO paper. It is
aligned with what recent optimizer papers need to establish: fair tuning,
held-out seeds, strong direct baselines, causal controls, a longer-horizon check,
a scale-transfer check, runtime disclosure, and reproducible figures.

It intentionally does **not** schedule a giant optimizer zoo, duplicate
300/600-step endpoint sweeps, or a separate wall-clock-matched campaign unless
the final paper explicitly claims wall-clock superiority.

Read `docs/PRIOR_ART_2026_AUDIT.md` before writing novelty claims. QKV splitting
and post-polar row adaptation are prior art. The narrower ASTRO candidate is the
**global adaptive redistribution/restoration after independently polarising
semantic fused blocks**, tested directly against `astro_v2_blockwise`.

> Important: once Stage A starts, do not modify `scripts/astro_lab.py` or
> `scripts/paper_campaign.py` until the campaign is complete. The campaign state
> deliberately refuses to mix different code or package environments.

---

## Stage 0 — checkout, fairness tests, and the shared corpus

Run once in a fresh Colab environment before spending GPU time:

```python
%cd /content/ASTRO
!git fetch origin
!git checkout paper-proof-campaign
!git pull origin paper-proof-campaign
!pip -q install transformers datasets matplotlib pytest

!pytest -q tests/test_paper_campaign.py tests/test_paper_mechanism.py
!git rev-parse HEAD
```

Prepare the exact shared token pool once:

```python
%cd /content/ASTRO
!python scripts/prepare_paper_data.py
```

Record the printed SHA-256, campaign code digest, package versions, and git
commit. Every later experiment reads this same pinned FineWeb-Edu token cache.
Do not delete/rebuild the cache midway through the campaign.

---

# Stage A — headline 124M / 900-step comparison

This is the paper's main optimizer experiment.

**Optimizers:** AdamW, Muon, NorMuon, faithful published-rule AdaMuon, ASTRO-v2.  
**Tuning:** 5 trials per optimizer, seed 0.  
**Confirmation:** 5 held-out seeds, 100–104.  
**Protocol:** same GPT-2 architecture, initialization procedure, minibatch RNG,
fixed token pool, fixed validation slice, FP16 policy, clipping, LR schedule
shape, and standard GPT decay routing.

Each optimizer gets its own best configuration from the same tuning budget.
Run this exact cell repeatedly; finished trials/seeds are persisted to Drive.

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

Keep rerunning the same command until it reports no work remaining.

Final raw readout:

```python
!cat /content/drive/MyDrive/astro/paper_main/astro_lab_report.md
!python -m json.tool /content/drive/MyDrive/astro/paper_main/astro_lab_state.json
```

### Plot now: headline result + runtime

As soon as Stage A is complete:

```python
%cd /content/ASTRO

!python scripts/paper_collect.py \
  --main /content/drive/MyDrive/astro/paper_main/astro_lab_state.json \
  --out artifacts/paper_campaign.json

!python scripts/figures/make_all.py --only paper_main
```

Produces:

```text
artifacts/figures/fig_paper_main.pdf
artifacts/figures/fig_paper_main.png
artifacts/figures/fig_paper_main.json
```

The figure shows the five held-out validation losses and the measured runtime
for each baseline. Runtime is disclosed as cost; this is **not** a separate
claim of wall-clock superiority.

**Freeze point:** after seeds 100–104 have been evaluated, never retune ASTRO or
any baseline because of those results. All remaining experiments use frozen
Stage-A configurations.

---

# Stage B — minimal causal / structural controls

This is the only required ablation campaign. It does not retune any ablation.

Muon controls use Stage A's frozen Muon configuration. ASTRO controls use Stage
A's frozen ASTRO-v2 configuration. Two new held-out seeds (200, 201) are used.

The six conditions are:

```text
muon                  published/default matrix beta1=.95
muon_m90              same Muon recipe, matrix beta1=.90
astro_v2_gamma0       ASTRO-v2 without post-spectral row adaptation
astro_v2_nosplit      ASTRO-v2 without semantic Q/K/V splitting
astro_v2_blockwise    split + adapt, but restore each Q/K/V block independently
astro_v2              full global post-split redistribution/restoration
```

Those six runs answer exactly four paper questions:

```text
muon_m90 - muon                    -> beta confound
astro_v2 - astro_v2_gamma0         -> row-adaptation contribution
astro_v2 - astro_v2_nosplit        -> semantic-split contribution
astro_v2 - astro_v2_blockwise      -> candidate ASTRO-specific global redistribution
```

Run:

```python
%cd /content/ASTRO

!python scripts/paper_mechanism.py \
  --main-state /content/drive/MyDrive/astro/paper_main/astro_lab_state.json \
  --seeds 200 201 \
  --work-dir /content/drive/MyDrive/astro/paper_mechanism \
  --max-minutes 180 \
  --stop-after 6
```

There are 12 total runs. If needed, rerun the same cell; completed conditions are
skipped.

Raw readout:

```python
!cat /content/drive/MyDrive/astro/paper_mechanism/paper_mechanism_report.md
!python -m json.tool /content/drive/MyDrive/astro/paper_mechanism/paper_mechanism_state.json
```

### Plot now: mechanism attribution

```python
%cd /content/ASTRO

!python scripts/paper_collect.py \
  --main /content/drive/MyDrive/astro/paper_main/astro_lab_state.json \
  --mechanism /content/drive/MyDrive/astro/paper_mechanism/paper_mechanism_state.json \
  --out artifacts/paper_campaign.json

!python scripts/figures/make_all.py --only paper_ablation
```

Produces `fig_paper_ablation.{pdf,png,json}`.

A positive `astro_v2` vs `astro_v2_blockwise` result establishes empirical value
for the narrow operation identified in the prior-art audit. It does **not** by
itself prove that no earlier publication implements the same operation.

---

# Stage C — long-horizon durability

Why this is required: the headline experiment is intentionally compute-limited.
Recent optimizer work is rightly skeptical of improvements that exist only very
early in training. Instead of wasting GPU time on new 300- and 600-step endpoint
sweeps, we use the 900-step headline plus one **3× longer** endpoint.

Only Muon and ASTRO-v2 are needed. Their Stage-A configurations remain frozen.
Two paired seeds are enough for this durability check; it is not presented as a
standalone significance experiment.

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

Rerun unchanged until all four Muon/ASTRO runs are complete.

### Plot now: durability

```python
%cd /content/ASTRO

!python scripts/paper_collect.py \
  --main /content/drive/MyDrive/astro/paper_main/astro_lab_state.json \
  --mechanism /content/drive/MyDrive/astro/paper_mechanism/paper_mechanism_state.json \
  --horizon /content/drive/MyDrive/astro/paper_horizon_2700/paper_transfer_state.json \
  --out artifacts/paper_campaign.json

!python scripts/figures/make_all.py --only paper_horizon
```

Produces `fig_paper_horizon.{pdf,png,json}`.

---

# Stage D — frozen 355M scale transfer

This asks whether the 124M-selected recipes transfer to a model about 3× larger.
Do **not** retune at 355M; otherwise this becomes another expensive leaderboard
rather than a transfer test.

Only Muon and ASTRO-v2 are required because the scale claim concerns the proposed
recipe relative to its direct parent. The full baseline table has already been
established at 124M.

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

Rerun unchanged until all four runs are complete.

### Plot now: scale transfer

```python
%cd /content/ASTRO

!python scripts/paper_collect.py \
  --main /content/drive/MyDrive/astro/paper_main/astro_lab_state.json \
  --mechanism /content/drive/MyDrive/astro/paper_mechanism/paper_mechanism_state.json \
  --horizon /content/drive/MyDrive/astro/paper_horizon_2700/paper_transfer_state.json \
  --scale /content/drive/MyDrive/astro/paper_scale_355m/paper_transfer_state.json \
  --out artifacts/paper_campaign.json

!python scripts/figures/make_all.py --only paper_scale
```

Produces `fig_paper_scale.{pdf,png,json}`.

Stages B, C, and D may run in separate Colab sessions **after Stage A has frozen
the selected configurations**, because each uses a different persistent state
file while reading the same immutable corpus cache.

---

# Final evidence audit and all paper plots

After A–D are complete:

```python
%cd /content/ASTRO

!python scripts/paper_audit.py \
  --main-state /content/drive/MyDrive/astro/paper_main/astro_lab_state.json \
  --mechanism-state /content/drive/MyDrive/astro/paper_mechanism/paper_mechanism_state.json \
  --scale-state /content/drive/MyDrive/astro/paper_scale_355m/paper_transfer_state.json \
  --horizon-state /content/drive/MyDrive/astro/paper_horizon_2700/paper_transfer_state.json \
  --out /content/drive/MyDrive/astro/paper_evidence_summary.md \
  --strict
```

Then rebuild the single clean plotting artifact and every final paper figure:

```python
%cd /content/ASTRO

!python scripts/paper_collect.py \
  --main /content/drive/MyDrive/astro/paper_main/astro_lab_state.json \
  --mechanism /content/drive/MyDrive/astro/paper_mechanism/paper_mechanism_state.json \
  --scale /content/drive/MyDrive/astro/paper_scale_355m/paper_transfer_state.json \
  --horizon /content/drive/MyDrive/astro/paper_horizon_2700/paper_transfer_state.json \
  --out artifacts/paper_campaign.json

!python scripts/figures/make_all.py --paper
```

Expected final confirmatory figures:

```text
fig_paper_main.pdf       # held-out baseline comparison + runtime
fig_paper_ablation.pdf   # beta / row / split / global-vs-blockwise attribution
fig_paper_horizon.pdf    # 900 -> 2700 durability
fig_paper_scale.pdf      # 124M -> 355M frozen transfer
```

Every plot also has a JSON sidecar containing the exact plotted values.
Historical/confounded results remain available for the research record but are
not read by these final figures.

---

# Experiments deliberately not required before writing

## No new 300/600-step endpoint grid

It repeats a question already covered by the 900-step result and the 2700-step
durability check. The historical 300/600 evidence can remain background, not
headline confirmation.

## No separate time-matched campaign by default

The headline state records actual seconds/run and `fig_paper_main` reports that
cost. A dedicated equal-wall-clock experiment is required **only** if the title,
abstract, or conclusion claims ASTRO is faster in wall-clock training. If the
paper claims lower validation loss at equal optimizer steps while disclosing the
runtime overhead, no additional time-match run is necessary.

## No larger optimizer zoo

AdamW + Muon + NorMuon + faithful AdaMuon are the directly relevant comparison
set. Add another optimizer only if it tests a specific reviewer-relevant
hypothesis, not merely to lengthen the table.

## Independent trainer validation is strengthening evidence, not a blocker

A modded-nanoGPT-derived replication remains a valuable follow-up after the
first complete paper draft. It is especially useful before a stronger venue or
if reviewers question Hugging Face GPT-2-specific behaviour. Do not delay the
first paper draft solely to run a second harness if A–D already support a
properly scoped claim.

---

# What constitutes a paper-supporting outcome

The experiments do not need every cell to favor ASTRO. The paper is supported
when the results permit a truthful, specific contribution such as:

> Under one fairness-controlled GPT training protocol, ASTRO-v2 improves over
> direct matrix-optimizer baselines; controlled removals localize the gain to a
> particular composition of semantic spectral splitting and global adaptive
> redistribution; the frozen recipe remains competitive at a longer horizon and
> a larger model size.

If the blockwise control ties ASTRO-v2, do **not** claim the global
redistribution as a novel performance contribution. If the headline comparison
collapses after the fairness fixes, do **not** add tests until something becomes
positive; narrow the paper to the mechanistic result or redesign the optimizer.

Before submission, repeat the literature searches listed in
`docs/PRIOR_ART_2026_AUDIT.md` and record the date. Experiments establish
empirical behaviour; literature review establishes the defensible novelty
boundary.
