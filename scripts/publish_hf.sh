#!/usr/bin/env bash
# Publish this repository's conformance payload to Hugging Face as a dataset.
#
#   HF_TOKEN=hf_xxx scripts/publish_hf.sh --dry-run   # assemble + list, upload nothing
#   HF_TOKEN=hf_xxx scripts/publish_hf.sh             # create/update the dataset repo
#
# HF_TOKEN must be a **write**-scoped token from
# https://huggingface.co/settings/tokens. Override the destination with
# HF_REPO_ID=owner/name. huggingface_hub is installed into a throwaway uv venv;
# nothing is added to this repo's toolchain.
#
# The payload is assembled fresh from the working tree on every run, so the
# published dataset can never drift from the spec it claims to carry.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ID="${HF_REPO_ID:-aefinityAIINC/cis2-conformance}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

mkdir -p "$STAGE/vectors"
cp "$ROOT/docs/HF_DATASET_CARD.md"          "$STAGE/README.md"
cp "$ROOT/LICENSE"                          "$STAGE/LICENSE"
cp "$ROOT/docs/CIS2_SPEC_v0.3b.md"          "$STAGE/"
cp "$ROOT/EXPECTED_DIGESTS.md"              "$STAGE/"
cp "$ROOT/docs/GPU_RESULT.md"               "$STAGE/"
cp "$ROOT/tests/conformance/PROTOCOL.md"    "$STAGE/"
cp "$ROOT/tests/conformance/README.md"      "$STAGE/vectors/README.md"
cp "$ROOT/tests/conformance/vectors/"*      "$STAGE/vectors/"

# The card quotes the normative digest; refuse to publish if they disagree.
DIGEST=d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df
grep -q "$DIGEST" "$STAGE/README.md"          || { echo "card does not carry the v0.3b digest" >&2; exit 1; }
grep -q "$DIGEST" "$STAGE/EXPECTED_DIGESTS.md" || { echo "EXPECTED_DIGESTS.md does not carry the v0.3b digest" >&2; exit 1; }

echo "payload:"; (cd "$STAGE" && find . -type f | sed 's|^\./|  |' | sort)
echo "target:  https://huggingface.co/datasets/$REPO_ID"
[ "$DRY" = 1 ] && { echo "dry run — nothing uploaded"; exit 0; }
[ -n "${HF_TOKEN:-}" ] || { echo "set HF_TOKEN (write-scoped) first" >&2; exit 1; }

VENV="${TMPDIR:-/tmp}/cis2-hf-venv"
if [ ! -x "$VENV/bin/python" ]; then
  uv venv "$VENV" >/dev/null
  VIRTUAL_ENV="$VENV" uv pip install -q huggingface_hub
fi

REPO_ID="$REPO_ID" STAGE="$STAGE" "$VENV/bin/python" - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = os.environ["REPO_ID"]
print("authenticated as:", api.whoami().get("name"))
api.create_repo(repo_id=repo, repo_type="dataset", private=False, exist_ok=True)
api.upload_folder(
    folder_path=os.environ["STAGE"],
    repo_id=repo,
    repo_type="dataset",
    commit_message="CIS-2 v0.3b: spec, op-level conformance vectors, expected digests, GPU result",
)
print("published: https://huggingface.co/datasets/%s" % repo)
PY
