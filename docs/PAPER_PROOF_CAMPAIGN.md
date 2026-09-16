# ASTRO paper-proof campaign

This is the confirmatory experiment sequence to run **after** the historical `astro_lab.py`
results. Historical results remain useful research evidence, but the headline paper tables should
come from `scripts/paper_campaign.py`, which fixes the auxiliary-LR schedule confound, pins the
corpus revision/cache, adds a faithful AdaMuon reference baseline, and adds the structural control
needed for the ASTRO novelty claim.

The campaign is designed for repeated Google Colab T4 sessions. Every GPU command uses a persistent
Drive workdir. Re-run an interrupted command unchanged; completed trials/seeds are skipped.

## What this campaign can establish

If the measurements support it, the campaign can establish:

1. ASTRO-v2 outperforms tuned AdamW/Muon/NorMuon/AdaMuon baselines under one declared protocol;
2. the result survives held-out seeds;
3. the result transfers from 124M to 355M without retuning;
4. the result survives an approximate equal-wall-clock Muon control;
5. beta/weight-decay recipe effects are separated from architecture effects;
6. semantic splitting, row adaptation, and **global-vs-blockwise post-split redistribution** are isolated;
7. the candidate novel operation has empirical value beyond the known NorMuon + split-QKV pattern.

Experiments cannot prove exhaustive literature novelty. Use `docs/PRIOR_ART_2026_AUDIT.md` together
with the `astro_v2_blockwise` control, and update the prior-art search immediately before submission.

---

## 0. Checkout and smoke-test the paper branch

```bash
%cd /content/ASTRO
!git fetch origin paper-proof-campaign
!git checkout paper-proof-campaign
!git pull origin paper-proof-campaign
!pip -q install transformers datasets pytest

!python -m pytest tests/test_paper_campaign.py -q
!git rev-parse HEAD
```

Do not start expensive runs if the tests fail.

A campaign run should begin with a line like:

```text
paper_campaign <12-hex fingerprint> (...)
protocol FineWeb-Edu@v1.0.0; shared cache ...
```

Record that fingerprint. Do not mix campaign fingerprints in one paper table without explicitly
re-running the affected cells.

---

# REQUIRED RUNS

## 1. Headline 124M/900 tuning + held-out confirmation

This is the central paper experiment.

Optimizers:

```text
AdamW
Muon
NorMuon
published/reference AdaMuon
ASTRO-v2
```

Each optimizer gets **5 tuning trials on seed 0** and then **5 held-out seeds 100–104**.
The search spaces may use different units where the algorithms require them, but each optimizer
receives the same number of real tuning knobs and the same number of trials.

Run this exact cell repeatedly until it reports no work remaining:

```bash
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

The command is intentionally one persistent protocol. Early sessions fill the 25 tuning trials;
later sessions evaluate the selected configurations on seeds 100–104.

Final readout:

```bash
!cat /content/drive/MyDrive/astro/paper_main/astro_lab_report.md
!python -m json.tool /content/drive/MyDrive/astro/paper_main/astro_lab_state.json
```

**Paper gate:** do not proceed to a superiority claim unless ASTRO-v2's held-out result survives this
fairness-fixed rerun. If it does not, rewrite the paper around the mechanism findings instead of
trying more seeds until significance appears.

---

## 2. Mechanism + structural-novelty campaign

This is the causal experiment. It separates recipe tuning from the candidate architectural
contribution.

Variants:

```text
astro                 beta1=.95, cautious WD on
astro_plain_wd        beta1=.95, cautious WD off
astro_muon_betas      beta1=.90, cautious WD on
astro_v2              beta1=.90, cautious WD off
astro_v2_gamma0       remove post-spectral row adaptation
astro_v2_nosplit      remove semantic QKV splitting
astro_v2_blockwise    split QKV, but adapt/restore each block independently
```

The key novelty contrast is:

```text
astro_v2  vs  astro_v2_blockwise
```

Both use the same beta, scalar path, QKV split, row statistic, weight decay and schedule. The only
intended difference is whether post-spectral adaptive redistribution restores one **global** fused
operator norm (ASTRO-v2) or restores each semantic block independently (prior-art-style control).

Run the exact cell repeatedly until complete:

```bash
%cd /content/ASTRO

!python scripts/paper_campaign.py \
  --mode scaling \
  --sizes 124M \
  --steps 900 \
  --optimizers \
    astro \
    astro_plain_wd \
    astro_muon_betas \
    astro_v2 \
    astro_v2_gamma0 \
    astro_v2_nosplit \
    astro_v2_blockwise \
  --trials 3 \
  --seeds 0 \
  --pin scalar_lr_mult=0.4369 \
  --work-dir /content/drive/MyDrive/astro/paper_mechanism \
  --max-minutes 180 \
  --stop-after 5
```

Final readout:

```bash
!cat /content/drive/MyDrive/astro/paper_mechanism/astro_lab_report.md
!python -m json.tool /content/drive/MyDrive/astro/paper_mechanism/astro_lab_state.json
```

Interpretation order:

```text
ASTRO-MB - ASTRO                  -> beta effect with cautious WD
ASTRO-v2 - ASTRO-Plain-WD        -> beta effect with plain WD
ASTRO-Plain-WD - ASTRO           -> WD effect at beta1=.95
ASTRO-v2 - ASTRO-MB              -> WD effect at beta1=.90
ASTRO-v2 - ASTRO-v2-gamma0       -> row-adaptation contribution
ASTRO-v2 - ASTRO-v2-NoSplit      -> semantic-split contribution
ASTRO-v2 - ASTRO-v2-Blockwise    -> global redistribution contribution
```

Do not attribute the full ASTRO-v2-vs-Muon gap to a component unless the corresponding isolated
contrast supports it.

---

## 3. Frozen 355M scale transfer

Do **not** retune at 355M. This experiment asks whether the 124M-selected configurations transfer.
It reuses `paper_main`, so the selected configurations are loaded directly from state rather than
copied by hand.

Run repeatedly until complete:

```bash
%cd /content/ASTRO

!python scripts/paper_campaign.py \
  --mode scaling \
  --sizes 355M \
  --steps 900 \
  --optimizers \
    adamw \
    muon \
    normuon \
    adamuon_ref \
    astro_v2 \
  --trials 0 \
  --seeds 2 \
  --work-dir /content/drive/MyDrive/astro/paper_main \
  --max-minutes 180 \
  --stop-after 4
```

This is a **frozen-recipe transfer test**, not a tuned 355M ranking.

---

## 4. Approximate equal-wall-clock Muon control

First compute Muon's step count from the *corrected* 124M/900 held-out timings. Never reuse the old
940-step estimate from the historical harness.

```bash
%cd /content/ASTRO

!python - <<'PY'
import json
from pathlib import Path
from statistics import fmean

p = Path('/content/drive/MyDrive/astro/paper_main/astro_lab_state.json')
s = json.loads(p.read_text())

muon = [
    v['seconds'] for k, v in s['runs'].items()
    if k.startswith('124M|900|muon|')
]
astro = [
    v['seconds'] for k, v in s['runs'].items()
    if k.startswith('124M|900|astro_v2|')
]
assert len(muon) == 5 and len(astro) == 5, (len(muon), len(astro))
steps = round(900 * fmean(astro) / fmean(muon))
print('Muon mean s/run:', fmean(muon))
print('ASTRO-v2 mean s/run:', fmean(astro))
print('time-matched Muon steps:', steps)
Path('/content/drive/MyDrive/astro/paper_matched_muon_steps.txt').write_text(str(steps))
PY
```

Then launch Muon using its selected corrected configuration automatically:

```bash
%cd /content/ASTRO

!python - <<'PY'
import json
import subprocess
import sys
from pathlib import Path

state = json.loads(Path('/content/drive/MyDrive/astro/paper_main/astro_lab_state.json').read_text())
cfg = state['tuned']['muon']
steps = Path('/content/drive/MyDrive/astro/paper_matched_muon_steps.txt').read_text().strip()

cmd = [
    sys.executable, 'scripts/paper_campaign.py',
    '--mode', 'scaling',
    '--sizes', '124M',
    '--steps', steps,
    '--optimizers', 'muon',
    '--config', *[f'{k}={v}' for k, v in cfg.items()],
    '--seeds', '5',
    '--work-dir', '/content/drive/MyDrive/astro/paper_time_match',
    '--max-minutes', '180',
]
print(' '.join(cmd))
subprocess.run(cmd, check=True)
PY
```

Because `paper_campaign.py` uses one fixed shared corpus, changing the step budget no longer changes
the training-token population. The LR schedule still legitimately depends on the total horizon.
Report the **actual measured seconds**, not merely the theoretical matched step count.

---

## 5. Paper evidence audit

After Runs 1–4 and the 355M transfer are complete:

```bash
%cd /content/ASTRO

!python scripts/paper_audit.py \
  --main-state /content/drive/MyDrive/astro/paper_main/astro_lab_state.json \
  --mechanism-state /content/drive/MyDrive/astro/paper_mechanism/astro_lab_state.json \
  --out /content/drive/MyDrive/astro/paper_evidence_summary.md \
  --strict
```

Then:

```bash
!cat /content/drive/MyDrive/astro/paper_evidence_summary.md
```

`--strict` exits non-zero if required seeds/cells are missing. It does not fail just because an
experiment gives an inconvenient scientific result.

---

# CONDITIONAL RUNS

Do these only if the corrected 124M experiment remains supportive.

## 6. 124M / 2700-step long-horizon check

Use the configurations already selected in `paper_main`; do not retune after seeing the held-out
results.

```bash
%cd /content/ASTRO

!python scripts/paper_campaign.py \
  --mode scaling \
  --sizes 124M \
  --steps 2700 \
  --optimizers muon astro_v2 \
  --trials 0 \
  --seeds 2 \
  --work-dir /content/drive/MyDrive/astro/paper_main \
  --max-minutes 180 \
  --stop-after 2
```

Re-run unchanged until both optimizers have both seeds. Treat two seeds as a long-horizon transfer
check, not as a significance experiment.

---

## 7. Independent trainer validation

The final paper is stronger if the frozen ASTRO-v2 recipe is reproduced in a second training harness
(e.g. the modded-nanoGPT-derived T4 protocol described in `docs/MODDED_NANOGPT_VALIDATION.md`).
Do this **after** the corrected main campaign, not before. The adapter must use the same fixed
optimizer recipe and must not retune ASTRO on the final validation seeds.

Minimum independent comparison:

```text
Muon
NorMuon
ASTRO-v2
5 held-out seeds if affordable; otherwise 3 with an explicit limitation
```

This independent harness is a robustness check against implementation-specific conclusions. It is
not required to claim that one experiment ran, but it materially improves a first optimizer paper.

---

# STOP CONDITIONS

Stop expanding the benchmark and write the paper when all of the following are true:

```text
[ ] paper-campaign fairness tests pass
[ ] 124M/900 tuning complete for all headline baselines
[ ] 5 held-out seeds complete for all headline baselines
[ ] ASTRO-v2 vs blockwise structural control complete
[ ] beta x WD attribution complete
[ ] gamma0 and no-split controls complete
[ ] 355M frozen transfer complete
[ ] time-matched Muon control complete
[ ] paper_audit.py --strict passes
[ ] prior-art audit updated immediately before submission
```

A 2700-step or second-harness result is valuable but should not become an excuse to keep adding
experiments indefinitely if the central claims are already supported and scoped correctly.
