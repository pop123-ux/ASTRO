#!/usr/bin/env python3
"""Prepare and checksum the shared paper-campaign token corpus once.

Run this before opening multiple Colab sessions. It prevents two independent
campaign workdirs from racing to create the same Drive cache and records a
SHA-256 checksum for the exact token-cache file used by every experiment.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from transformers import AutoTokenizer

import paper_campaign as campaign


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokens = campaign.paper_load_tokens(
        tokenizer,
        campaign.TOTAL_TOKENS,
        campaign.SHARED_CACHE,
    )
    assert tokens.numel() == campaign.TOTAL_TOKENS

    path = campaign.SHARED_CACHE
    checksum = sha256_file(path)
    sidecar = Path(str(path) + ".sha256")
    if sidecar.exists():
        expected = sidecar.read_text().strip()
        if expected != checksum:
            raise SystemExit(
                f"CORPUS CHECKSUM MISMATCH: recorded {expected}, actual {checksum}. "
                "Do not train until the cache provenance is resolved."
            )
    else:
        sidecar.write_text(checksum + "\n")

    print(f"paper corpus: {path}")
    print(f"tokens: {tokens.numel():,}")
    print(f"sha256: {checksum}")
    print(f"revision: FineWeb-Edu@{campaign.FINEWEB_REVISION}")
    print(f"campaign code digest: {campaign.campaign_digest()}")
    print("environment:", campaign.current_environment())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
