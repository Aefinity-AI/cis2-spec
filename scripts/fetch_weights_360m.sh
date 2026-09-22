#!/usr/bin/env bash
# Fetches the pinned SmolLM2-360M weights for the informative second-model
# vector in EXPECTED_DIGESTS.md ("Informative: SmolLM2-360M"), verifying
# sha256 before use. Same-family, same-tokenizer sibling of the primary
# SmolLM2-135M §13.1 test vector fetched by fetch_weights.sh.
#
# Usage: scripts/fetch_weights_360m.sh [dest_dir]
# dest_dir defaults to ./weights360m (gitignored; never committed).
set -euo pipefail

DEST="${1:-weights360m}"
mkdir -p "$DEST"

BASE_URL="https://huggingface.co/HuggingFaceTB/SmolLM2-360M/resolve/main"

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

# sha256 values from EXPECTED_DIGESTS.md ("Informative: SmolLM2-360M")
fetch_and_check "model.safetensors" "7aaff6661428bed033abba9522bec81938678642cca3181fe752b6ca9e1e540f"
fetch_and_check "config.json"        "34f7801487078de7e434e19162c497e5cc6ff397080e40e8586627cb68a5168a"
fetch_and_check "tokenizer.json"     "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"

echo "All weight artifacts fetched and sha256-verified into $DEST/"
echo "Reproduce: cargo run --release --bin cis2_ref -- --weights-dir $DEST"
