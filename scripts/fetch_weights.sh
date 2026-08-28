#!/usr/bin/env bash
# Fetches the pinned SmolLM2-135M weights referenced by
# docs/CIS2_SPEC_v0.2.md §2.1, verifying sha256 before use.
#
# Usage: scripts/fetch_weights.sh [dest_dir]
# dest_dir defaults to ./weights (gitignored; never committed).
set -euo pipefail

DEST="${1:-weights}"
mkdir -p "$DEST"

BASE_URL="https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main"

fetch_and_check() {
  local name="$1" expected="$2"
  echo "Fetching $name ..."
  curl -sSf -L --retry 6 --retry-delay 10 --retry-all-errors \
    -o "$DEST/$name" "$BASE_URL/$name"
  local got
  got=$(sha256sum "$DEST/$name" | awk '{print $1}')
  echo "  expected=$expected"
  echo "  got=     $got"
  if [ "$got" != "$expected" ]; then
    echo "sha256 MISMATCH for $name" >&2
    exit 1
  fi
}

# sha256 values from EXPECTED_DIGESTS.md / spec §2.1
fetch_and_check "model.safetensors" "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"
fetch_and_check "config.json"        "1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843"
fetch_and_check "tokenizer.json"     "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"

echo "All weight artifacts fetched and sha256-verified into $DEST/"
