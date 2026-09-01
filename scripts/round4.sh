#!/usr/bin/env bash
# Round 4, run unattended and in dependency order.
#
#   bash scripts/round4.sh gpt_bpe
#
# The subword task is the one to trust. The character-level tasks put 3.5% of
# parameters on the elementwise path against GPT-2 small's 31.7%, so they cannot
# measure the scalar path at all and over-reward anything that helps the
# spectral one. gpt_bpe reproduces the real split at 32.2%.
#
# Everything writes into artifacts/bench_llm/<task>_r4/ and appends to one log,
# so a disconnect loses the running stage and nothing earlier.
set -uo pipefail

TASK="${1:-gpt_bpe}"
TRIALS="${TRIALS:-16}"
SEEDS="${SEEDS:-10}"
OUT="artifacts/bench_llm/${TASK}_r4"
LOG="artifacts/bench_llm/${TASK}_r4.log"
mkdir -p "$OUT"

say() { printf '\n=== %s === %s\n' "$1" "$(date -u +%H:%M:%S)" | tee -a "$LOG"; }

say "field: adamw muon normuon astro on $TASK"
python -m astro.bench.run --task "$TASK" \
    --optimizers adamw muon normuon astro \
    --trials "$TRIALS" --seeds "$SEEDS" --out "$OUT" 2>&1 | tee -a "$LOG"

# Candidates are compared against the three published baselines rather than
# against `astro`, because `astro` is now the round-3 recipe and the question is
# whether a component beats the state of the art, not whether it beats us.
say "candidates: one component each, against the published field"
python scripts/candidate.py --task "$TASK" \
    --candidate astro_v2 astro_pinned astro_gamma50 astro_equil astro_plain_wd \
    --against muon normuon adamw \
    --trials "$TRIALS" --seeds "$SEEDS" --field "$OUT" 2>&1 | tee -a "$LOG"

say "candidates: the gamma curve"
# Five points between the spectral-norm dual map (0) and the row-wise one (1).
python scripts/candidate.py --task "$TASK" \
    --candidate astro_gamma00 astro_gamma25 astro_gamma75 \
    --against muon normuon adamw \
    --trials "$TRIALS" --seeds "$SEEDS" --field "$OUT" 2>&1 | tee -a "$LOG"

say "candidates: remaining knobs"
python scripts/candidate.py --task "$TASK" \
    --candidate astro_blend20 astro_blend40 astro_split120 \
    --against muon normuon adamw \
    --trials "$TRIALS" --seeds "$SEEDS" --field "$OUT" 2>&1 | tee -a "$LOG"

say "done"
