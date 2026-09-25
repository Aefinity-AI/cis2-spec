"""Digest helpers for the eval harness.

Reuses the sha256-of-file-bytes convention already established in this
repository (see EXPECTED_DIGESTS.md: weights_sha256 / config_sha256 /
tokenizer_sha256, and scripts/fetch_weights.sh). This module does NOT
reimplement CIS-2's fp32 bit-exact witness digest (CIS2_REF, computed by
cis2-verify/ over the actual arithmetic trace) -- that is a much larger,
architecture-specific undertaking scoped to the reference kernel, not to an
eval harness. The digests here are a lighter-weight "receipt" in the same
spirit: sha256 over exact file bytes, composed with a documented, fixed
encoding so the composition itself is reproducible.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compose_digest(parts: Iterable[tuple[str, str]]) -> str:
    """Deterministically combine named (label, hex_digest) pairs into one
    digest. Encoding: sha256 of the UTF-8 bytes of
    "label1=digest1\\nlabel2=digest2\\n..." with parts sorted by label, so
    the composition is order-independent and exactly reproducible from the
    inputs alone."""
    lines = [f"{label}={digest}" for label, digest in sorted(parts)]
    blob = ("\n".join(lines) + "\n").encode("utf-8")
    return sha256_bytes(blob)


def model_digest(model_dir: Path) -> tuple[str, dict]:
    """sha256 of each present checkpoint file (weights, config, tokenizer),
    composed into one digest via compose_digest. Returns (digest, detail)
    where detail maps filename -> sha256 for the receipt."""
    candidates = [
        "model.safetensors",
        "pytorch_model.bin",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ]
    detail = {}
    for name in candidates:
        p = model_dir / name
        if p.is_file():
            detail[name] = sha256_file(p)
    if not detail:
        raise FileNotFoundError(f"no known checkpoint files found under {model_dir}")
    digest = compose_digest(detail.items())
    return digest, detail


def prompt_set_digest(path: Path) -> str:
    """sha256 of the exact bytes of the frozen prompt-set file."""
    return sha256_file(path)


def outputs_digest(records: list[dict]) -> str:
    """sha256 of the raw model outputs, serialized deterministically
    (sorted keys, sorted by record id) so the digest depends only on the
    generated content, not on dict/JSON ordering incidental to this run."""
    ordered = sorted(records, key=lambda r: r["id"])
    blob = json.dumps(ordered, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return sha256_bytes(blob)
