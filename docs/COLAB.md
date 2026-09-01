# Running the GPU probe on a Colab T4

`scripts/colab_probe.py` measures what four CPU cores could not. It is three
independent parts; run them in the order below and paste the markdown tables
back, or send `astro_probe_results.json`, which accumulates every part.

Set the runtime to **T4 GPU** first: *Runtime → Change runtime type → T4 GPU*.

## Cell 1 — setup and a two-minute sanity check

```python
!pip -q install transformers datasets
!python colab_probe.py --part B --repeats 20
```

Part B first because it needs no downloads, so it confirms the GPU works before
anything expensive starts. It times one Newton–Schulz on a fused `(3d, d)`
projection against three on `(d, d)` blocks — the claim that the fix costs less
than the defect. The iteration is cubic in the row count, so splitting should
win at real widths; if it does not on a T4, that is itself a finding and the
paper's cost claim gets qualified.

## Cell 2 — the measurement that matters

```python
!python colab_probe.py --part A --seq-len 512
```

Roughly five minutes plus download time. Loads eight ungated checkpoints one at
a time, accumulates a gradient over four windows, and reports how Muon's update
mass splits across Q, K and V.

A slow download or an OOM skips that model and continues. For a subset:

```python
!python colab_probe.py --part A --models gpt2 gpt2-medium EleutherAI/pythia-410m
```

**What to look for.** On our 806K-parameter models the fused share is roughly
`Q 0.05 / K 0.09 / V 0.85`. If real checkpoints land anywhere near that, the
defect is a property of transformers rather than of our benchmark. **If they
come out near `0.33 / 0.33 / 0.33` the claim is wrong at scale and the paper has
to say so** — that outcome is worth exactly as much as the confirming one, so
send the table either way.

Columns worth reading:

- **`layout`** — `contiguous` for GPT-2, `interleaved` for Pythia. GPT-NeoX
  slices Q/K/V *within each head*; a naive contiguous three-way split recovers
  only half of Q's rows, which the tests pin explicitly.
- **`grad |V|/|Q|`** — the root cause. Q and K reach the loss through the
  softmax Jacobian and should show much smaller gradients than V.
- **`participation`** — 1.0 means update mass is spread evenly across rows;
  lower means it has collapsed onto a few.
- **`random-init Q/K/V`** — the same architecture with random weights. If the
  defect appears here too it is a property of the attention parameterisation; if
  only on the trained checkpoint, a property of the solution. A reviewer will
  ask, so the control runs by default (`--no-control` to skip).

Llama-style rows (Qwen, SmolLM2, TinyLlama) keep Q/K/V separate, so their
numbers describe what a *fused training framework* — Megatron-LM, NeMo — would
produce from those gradients, not what the checkpoint itself does. They are
labelled `[simulated fusion]`.

**Dropout is forced to zero**, and the script prints how many modules it
silenced. This is not cosmetic: HuggingFace ships GPT-2 with `p = 0.1`, and with
dropout live V's share reads 0.78 against 0.61 without — a systematic
overstatement of the exact effect being claimed, not noise that averages out.
Zero is both the deterministic and the conservative choice.

## Cell 3 — optional, the scale gap

```python
!python colab_probe.py --part C --steps 400 --batch 8 --seq-len 512 --seeds 2
```

Trains GPT-2 124M from scratch on FineWeb-Edu with the GPT-2 BPE tokenizer: the
architecture, scale and tokenisation the optimizer literature actually uses.
Budget roughly 40–60 minutes on a T4, including the sweep.

Each optimizer gets an **equal three-rate learning-rate sweep** on seed 0 before
being evaluated on disjoint seeds, because an untuned comparison is the failure
Wen et al. attribute most published optimizer speedups to. The script warns when
an optimizer selects the edge of its grid, which means the optimum probably lies
outside it and the number should not be trusted.

**This is still a smoke test, not a result**, and the script says so in its own
output. Three rates is a coarse sweep, and 400 steps at batch 8 is roughly
0.0007× the Chinchilla-optimal token budget for 124M against the 1×–16× the
literature uses. The paper will describe it as a smoke test at a standard scale.

If it OOMs: `--batch 4`, or `--seq-len 256`, or shrink the model with
`--n-layer 6 --n-embd 512 --n-head 8`.

Colab disconnects cost only the running part — each writes its JSON on
completion.

## What each part changes in the paper

| part | what it settles | weight |
|---|---|---|
| A | whether the fused-projection defect is real on production checkpoints, 124M–1.4B, across two fusion layouts, with a random-init control | **high** |
| B | the hardware cost of splitting, against the operation count — **settled: splitting is more expensive**, 1.29× predicted and 1.33× measured, correcting an earlier claim that it was cheaper | medium |
| C | a first ASTRO measurement at 124M with BPE tokenisation | low — smoke test |

Part A is the one that carries weight. It is a statement about linear algebra
plus one gradient, so unlike every training result in this project it does not
inherit the 806K-parameter scale caveat.

## Verification

`tests/test_colab_probe.py` (24 tests, no GPU or hub needed) pins the properties
that would otherwise fail silently: operator orientation for HuggingFace's
`Conv1D`, both fused row layouts, label shifting, routing coverage, dropout
determinism, and the Newton–Schulz fixed point. Run `pytest
tests/test_colab_probe.py` to confirm before trusting a number out of this
script.
