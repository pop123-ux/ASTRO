"""Real text corpora for the language-model benchmark, and their tokenisation.

Two corpora, both fetched once by ``scripts/fetch_llm_data.py`` and cached on
disk:

``wikitext-2``
    The standard small language-modelling benchmark: cleaned Wikipedia prose.
    Used as the *pretraining* distribution.

``tinyshakespeare``
    The corpus Karpathy's char-rnn, minGPT and nanoGPT all ship with. Early
    Modern English, in verse and dialogue, with speaker headings. Used as the
    *fine-tuning* distribution.

The pair is chosen so the domain shift is large in style and vocabulary
statistics while the alphabet stays shared, which is what makes transferring a
pretrained embedding table meaningful.

Character level, not BPE
------------------------
GPT-2's BPE vocabulary is 50257. At the model widths that fit on a CPU budget an
embedding table that size would hold an order of magnitude more parameters than
the transformer blocks, and it is routed to the scalar path by
:mod:`astro.routing` -- so the benchmark would mostly be measuring AdamW on an
embedding table, not the matrix path under test. Character level keeps the
spectral path dominant, which is the thing being compared. It is the same choice
nanoGPT's ``shakespeare_char`` configuration makes.

A **shared vocabulary** is built from the union of both corpora. Without it the
embedding table could not transfer from pretraining to fine-tuning and the
fine-tuning task would be measuring re-initialisation instead of adaptation.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

__all__ = [
    "BPE_VOCAB_SIZE",
    "get_corpus",
    "load_bpe_corpus",
    "CORPUS_SOURCES",
    "MIN_CHAR_COUNT",
    "REPLACEMENT",
    "Corpus",
    "data_root",
    "load_corpus",
    "shared_vocab",
]

#: ``name -> (url, filename)``. Fetched by ``scripts/fetch_llm_data.py``.
CORPUS_SOURCES: dict[str, tuple[str, str]] = {
    "wikitext2-train": (
        "https://raw.githubusercontent.com/pytorch/examples/main/"
        "word_language_model/data/wikitext-2/train.txt",
        "wikitext2_train.txt",
    ),
    "wikitext2-valid": (
        "https://raw.githubusercontent.com/pytorch/examples/main/"
        "word_language_model/data/wikitext-2/valid.txt",
        "wikitext2_valid.txt",
    ),
    "shakespeare": (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
        "data/tinyshakespeare/input.txt",
        "tinyshakespeare.txt",
    ),
}


def data_root() -> Path:
    """Directory holding the cached corpora.

    Overridable with ``ASTRO_DATA_DIR`` so the benchmark can run against a
    read-only or pre-seeded mount.
    """
    override = os.environ.get("ASTRO_DATA_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "data"


@dataclass(frozen=True)
class Corpus:
    """A tokenised corpus split into train and validation halves."""

    name: str
    train: torch.Tensor
    val: torch.Tensor
    vocab_size: int

    def batch(
        self, batch_size: int, block_size: int, generator: torch.Generator, *, split: str = "train"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of ``(input, target)`` windows, nanoGPT style."""
        data = self.train if split == "train" else self.val
        high = data.numel() - block_size - 1
        if high <= 0:
            raise ValueError(f"{self.name}: split {split!r} shorter than block_size")
        starts = torch.randint(0, high, (batch_size,), generator=generator)
        x = torch.stack([data[s : s + block_size] for s in starts])
        y = torch.stack([data[s + 1 : s + 1 + block_size] for s in starts])
        return x, y


def _read(filename: str) -> str:
    path = data_root() / filename
    if not path.exists():
        raise FileNotFoundError(
            f"corpus file {path} is missing. Fetch the corpora once with:\n"
            f"    python scripts/fetch_llm_data.py\n"
            f"(or point ASTRO_DATA_DIR at a directory that already has them)"
        )
    return path.read_text(encoding="utf-8", errors="replace")


#: Characters rarer than this across the union of corpora collapse to REPLACEMENT.
MIN_CHAR_COUNT = 100
#: Stand-in for characters below the frequency floor.
REPLACEMENT = "�"


@lru_cache(maxsize=1)
def shared_vocab() -> tuple[str, ...]:
    """Sorted character vocabulary spanning every corpus.

    Shared across corpora so a model pretrained on one can be fine-tuned on the
    other without resizing -- see the module docstring.

    Characters occurring fewer than :data:`MIN_CHAR_COUNT` times collapse to
    :data:`REPLACEMENT`. WikiText-2 contains a long tail of CJK and symbol
    characters -- 283 distinct characters in total, of which 187 account for
    0.02% of the text. Keeping them would trained-nothing rows in the embedding
    table and widen the softmax for no signal; folding them to one symbol keeps
    the vocabulary at 96 with 99.98% coverage. Mapping rather than deleting
    preserves sequence length and the local context around a rare character.
    """
    counts: Counter[str] = Counter()
    for filename in {name for _, name in CORPUS_SOURCES.values()}:
        counts.update(_read(filename))
    kept = {c for c, n in counts.items() if n >= MIN_CHAR_COUNT}
    kept.add(REPLACEMENT)
    return tuple(sorted(kept))


def _encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    unknown = stoi[REPLACEMENT]
    return torch.tensor([stoi.get(c, unknown) for c in text], dtype=torch.long)


@lru_cache(maxsize=4)
def load_corpus(name: str, *, max_chars: int = 2_000_000, val_fraction: float = 0.1) -> Corpus:
    """Load and tokenise a corpus under the shared vocabulary.

    Parameters
    ----------
    name:
        ``"wikitext2"`` or ``"shakespeare"``.
    max_chars:
        Truncation bound. WikiText-2 is 10.8 MB; the benchmark reads far fewer
        tokens than that, and truncating keeps tokenisation off the critical
        path of every run.
    val_fraction:
        Tail fraction held out. Contiguous rather than random, so validation
        windows never overlap a training window.

    Results are cached: the tasks call this once per process, not once per run.
    """
    vocab = shared_vocab()
    stoi = {c: i for i, c in enumerate(vocab)}

    if name == "wikitext2":
        text = _read(CORPUS_SOURCES["wikitext2-train"][1])[:max_chars]
        data = _encode(text, stoi)
        split = int(len(data) * (1.0 - val_fraction))
        train, val = data[:split], data[split:]
    elif name == "shakespeare":
        text = _read(CORPUS_SOURCES["shakespeare"][1])[:max_chars]
        data = _encode(text, stoi)
        split = int(len(data) * (1.0 - val_fraction))
        train, val = data[:split], data[split:]
    else:
        raise KeyError(f"unknown corpus {name!r}; available: 'wikitext2', 'shakespeare'")

    return Corpus(name=name, train=train, val=val, vocab_size=len(vocab))

#: Subword vocabulary size for the BPE corpora. Chosen so the benchmark model's
#: parameter split matches GPT-2 small's rather than being an artifact of
#: character-level tokenisation -- see :func:`load_bpe_corpus`.
BPE_VOCAB_SIZE = 2816

#: Corpus name to the key it is stored under in CORPUS_SOURCES.
_SOURCE_KEY = {"wikitext2": "wikitext2-train", "shakespeare": "shakespeare"}


@lru_cache(maxsize=2)
def _bpe_tokenizer(name: str, vocab_size: int):
    """Train a byte-level BPE on the corpus itself. Nothing is downloaded.

    ``tokenizers`` trains this in a few seconds and the result is cached for the
    process, so it stays off the critical path of a run.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[],
        show_progress=False,
    )
    text = _read(CORPUS_SOURCES[_SOURCE_KEY[name]][1])
    tokenizer.train_from_iterator([text], trainer=trainer)
    return tokenizer


@lru_cache(maxsize=4)
def load_bpe_corpus(
    name: str, *, vocab_size: int = BPE_VOCAB_SIZE, max_chars: int = 2_000_000,
    val_fraction: float = 0.1,
) -> Corpus:
    """Load a corpus under a locally-trained byte-level BPE vocabulary.

    Why this exists
    ---------------
    Character-level tokenisation makes the benchmark unrepresentative in a way
    that is invisible until a result fails to transfer. Embeddings and the tied
    head take the elementwise path, every other operator takes the spectral one,
    so the split between the two paths decides how much of the model a matrix
    optimizer is even responsible for. At ``vocab = 97, d = 128`` that split is
    3.5% scalar; GPT-2 small at ``vocab = 50257, d = 768`` is 31.7%. The
    character-level benchmark therefore cannot measure the scalar path at all,
    and any tuning of ``scalar_lr_mult`` on it is fitting noise.

    ``BPE_VOCAB_SIZE = 2816`` is chosen to reproduce GPT-2 small's ratio at
    ``d = 128``: with 806K non-embedding parameters, a vocabulary of about 2829
    puts 31% on the scalar path. This is the one number in the benchmark tuned
    to match the target rather than to be convenient.
    """
    tokenizer = _bpe_tokenizer(name, vocab_size)
    text = _read(CORPUS_SOURCES[_SOURCE_KEY[name]][1])[:max_chars]
    ids = tokenizer.encode(text).ids
    data = torch.tensor(ids, dtype=torch.long)
    split = int(len(data) * (1.0 - val_fraction))
    return Corpus(
        name=f"{name}-bpe", train=data[:split], val=data[split:],
        vocab_size=tokenizer.get_vocab_size(),
    )


def get_corpus(name: str) -> Corpus:
    """Resolve a corpus name, dispatching on a ``-bpe`` suffix.

    ``"wikitext2"`` is character level; ``"wikitext2-bpe"`` is the same text
    under a locally-trained byte-level BPE. One entry point so a task can switch
    tokenisation by name and nothing else changes.
    """
    if name.endswith("-bpe"):
        return load_bpe_corpus(name[: -len("-bpe")])
    return load_corpus(name)
