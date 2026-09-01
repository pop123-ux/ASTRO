#!/usr/bin/env python3
"""Fetch the language-model benchmark corpora once and cache them on disk.

    python scripts/fetch_llm_data.py

Downloads WikiText-2 (pytorch/examples) and tinyshakespeare (karpathy/char-rnn)
into ``astro/data/``. Both are small, public, and served from raw.githubusercontent.

The benchmark deliberately does *not* download anything at run time: a task that
touches the network cannot be trusted to be reproducible, and a corpus that
changes underneath a stored result invalidates it silently.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astro.bench.corpora import CORPUS_SOURCES, data_root  # noqa: E402


def fetch(url: str, destination: Path, *, force: bool = False, timeout: int = 120) -> bool:
    """Download ``url`` to ``destination`` unless it is already there.

    Returns True if a download happened.
    """
    if destination.exists() and not force:
        print(f"  have  {destination.name} ({destination.stat().st_size:,} bytes)")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"  get   {destination.name} <- {url}")
    # Written to a temporary name first so an interrupted download cannot leave a
    # truncated file that later looks like a valid cache entry.
    staging = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        staging.write_bytes(response.read())
    staging.replace(destination)
    print(f"        {destination.stat().st_size:,} bytes")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args(argv)

    root = data_root()
    print(f"corpus cache: {root}")
    for name, (url, filename) in CORPUS_SOURCES.items():
        print(f"[{name}]")
        fetch(url, root / filename, force=args.force)

    # Building the vocabulary reads every file, so it doubles as a check that
    # each cached file is present and decodable before any benchmark runs.
    from astro.bench.corpora import load_corpus, shared_vocab

    vocab = shared_vocab()
    print(f"\nshared vocabulary: {len(vocab)} characters")
    for corpus_name in ("wikitext2", "shakespeare"):
        corpus = load_corpus(corpus_name)
        print(
            f"  {corpus_name:14s} train={corpus.train.numel():,} tokens  "
            f"val={corpus.val.numel():,} tokens"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
