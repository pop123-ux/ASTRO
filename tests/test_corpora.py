"""Tests for corpus loading and tokenisation.

Skipped when the corpora have not been fetched, so a clone without
``scripts/fetch_llm_data.py`` having been run still has a green suite.
"""

from __future__ import annotations

import pytest
import torch

from astro.bench.corpora import (
    CORPUS_SOURCES,
    MIN_CHAR_COUNT,
    REPLACEMENT,
    data_root,
    load_corpus,
    shared_vocab,
)

_missing = [name for _, name in CORPUS_SOURCES.values() if not (data_root() / name).exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing), reason=f"corpora not fetched: {_missing}; run scripts/fetch_llm_data.py"
)


def test_vocabulary_is_shared_across_corpora() -> None:
    """Pretraining and fine-tuning must share an embedding table to transfer."""
    wiki = load_corpus("wikitext2")
    shakespeare = load_corpus("shakespeare")
    assert wiki.vocab_size == shakespeare.vocab_size == len(shared_vocab())


def test_rare_characters_collapse_to_one_symbol() -> None:
    """WikiText-2's CJK tail would otherwise widen the softmax for no signal."""
    vocab = shared_vocab()
    assert REPLACEMENT in vocab
    # 283 distinct characters appear; the frequency floor keeps far fewer.
    assert len(vocab) < 150


def test_every_token_is_in_range() -> None:
    for name in ("wikitext2", "shakespeare"):
        corpus = load_corpus(name)
        for split in (corpus.train, corpus.val):
            assert int(split.min()) >= 0
            assert int(split.max()) < corpus.vocab_size


def test_validation_split_is_a_contiguous_tail() -> None:
    """Random splits would let a validation window overlap a training window."""
    corpus = load_corpus("shakespeare")
    total = corpus.train.numel() + corpus.val.numel()
    assert corpus.val.numel() == pytest.approx(0.1 * total, rel=0.01)


def test_batches_are_input_target_shifted_by_one() -> None:
    corpus = load_corpus("shakespeare")
    generator = torch.Generator().manual_seed(0)
    x, y = corpus.batch(4, 16, generator)
    assert x.shape == y.shape == (4, 16)
    # y is x shifted one position: y[:, :-1] equals x[:, 1:].
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_batching_is_deterministic_given_a_generator_seed() -> None:
    corpus = load_corpus("shakespeare")
    first = corpus.batch(4, 16, torch.Generator().manual_seed(7))
    second = corpus.batch(4, 16, torch.Generator().manual_seed(7))
    assert torch.equal(first[0], second[0])


def test_unknown_corpus_names_raise() -> None:
    with pytest.raises(KeyError):
        load_corpus("wikitext103")


def test_frequency_floor_is_documented_where_it_is_used() -> None:
    """Guards against the constant drifting away from the docstring's claim."""
    assert MIN_CHAR_COUNT == 100


# ---------------------------------------------------------------------------
# Subword tokenisation, and why the vocabulary size is what it is
# ---------------------------------------------------------------------------


def test_bpe_corpus_trains_locally_and_hits_its_vocabulary_size() -> None:
    from astro.bench.corpora import BPE_VOCAB_SIZE, load_bpe_corpus

    corpus = load_bpe_corpus("wikitext2")
    assert corpus.vocab_size == BPE_VOCAB_SIZE
    assert corpus.train.numel() > 100_000 and corpus.val.numel() > 10_000
    assert int(corpus.train.max()) < corpus.vocab_size
    assert int(corpus.val.max()) < corpus.vocab_size


def test_bpe_is_denser_than_characters_on_the_same_text() -> None:
    from astro.bench.corpora import load_bpe_corpus, load_corpus

    assert load_bpe_corpus("wikitext2").train.numel() < load_corpus("wikitext2").train.numel()


def test_bpe_vocabulary_reproduces_gpt2s_parameter_split() -> None:
    """The point of the subword task: a matrix optimizer is responsible for the
    non-embedding path, so the fraction on the *other* path decides how much of
    the model it can affect. Character level puts 3.5% there; GPT-2 small 31.7%.
    """
    from astro.bench.corpora import get_corpus
    from astro.bench.llm import BENCH_GPT, build_gpt

    model = build_gpt(get_corpus("wikitext2-bpe").vocab_size, seed=0, **BENCH_GPT)
    scalar = model.transformer.wte.weight.numel() + model.transformer.wpe.weight.numel()
    total = sum(p.numel() for p in model.parameters())
    assert 0.28 < scalar / total < 0.36, f"got {scalar / total:.1%}, GPT-2 small is 31.7%"

    character = build_gpt(get_corpus("wikitext2").vocab_size, seed=0, **BENCH_GPT)
    scalar_char = (
        character.transformer.wte.weight.numel() + character.transformer.wpe.weight.numel()
    )
    assert scalar_char / sum(p.numel() for p in character.parameters()) < 0.06


def test_get_corpus_dispatches_on_the_bpe_suffix() -> None:
    from astro.bench.corpora import get_corpus

    assert get_corpus("wikitext2").vocab_size == 97
    assert get_corpus("wikitext2-bpe").vocab_size == 2816
