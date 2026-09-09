#!/usr/bin/env bash
# Publish this repository's conformance payload to Hugging Face as a dataset.
#
#   HF_TOKEN=hf_xxx scripts/publish_hf.sh --dry-run   # assemble + list, upload nothing
#   HF_TOKEN=hf_xxx scripts/publish_hf.sh             # create/update the dataset repo
#   scripts/publish_hf.sh --bucket                    # push to the S3 bucket instead
#
# HF_TOKEN must be a **write**-scoped token from
# https://huggingface.co/settings/tokens. Override the destination with
# HF_REPO_ID=owner/name.
#
# --bucket takes a different credential: the S3-compatible storage keys from
# the Hugging Face storage settings (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY,
# endpoint https://s3.hf.co/<namespace>). Those keys can create and fill a
# bucket repo but cannot create a dataset repo, so the two modes are not
# interchangeable: the bucket is a public file store, the dataset is the
# indexed, card-rendering page. Override with HF_BUCKET / HF_NAMESPACE.
#
# Python deps go into a throwaway uv venv; nothing is added to this repo's
# toolchain.
#
# The payload is assembled fresh from the working tree on every run, so the
# published dataset can never drift from the spec it claims to carry.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ID="${HF_REPO_ID:-aefinityAIINC/cis2-conformance}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
DRY=0; MODE=dataset
case "${1:-}" in
  --dry-run) DRY=1 ;;
  --bucket)  MODE=bucket ;;
  "")        ;;
  *) echo "usage: $0 [--dry-run|--bucket]" >&2; exit 2 ;;
esac
NAMESPACE="${HF_NAMESPACE:-${REPO_ID%%/*}}"
BUCKET="${HF_BUCKET:-${REPO_ID##*/}}"

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
if [ "$MODE" = bucket ]; then
  echo "target:  https://huggingface.co/buckets/$NAMESPACE/$BUCKET"
else
  echo "target:  https://huggingface.co/datasets/$REPO_ID"
fi
[ "$DRY" = 1 ] && { echo "dry run — nothing uploaded"; exit 0; }

VENV="${TMPDIR:-/tmp}/cis2-hf-venv"
[ -x "$VENV/bin/python" ] || uv venv "$VENV" >/dev/null

if [ "$MODE" = bucket ]; then
  [ -n "${AWS_ACCESS_KEY_ID:-}" ] && [ -n "${AWS_SECRET_ACCESS_KEY:-}" ] || {
    echo "set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (HF storage keys) first" >&2; exit 1; }
  "$VENV/bin/python" -c "import boto3" 2>/dev/null || \
    VIRTUAL_ENV="$VENV" uv pip install -q boto3
  NAMESPACE="$NAMESPACE" BUCKET="$BUCKET" STAGE="$STAGE" "$VENV/bin/python" - <<'BUCKET_PY'
import mimetypes, os, pathlib
import boto3
from botocore.config import Config

ns, bucket, stage = os.environ["NAMESPACE"], os.environ["BUCKET"], os.environ["STAGE"]
s3 = boto3.client(
    "s3",
    endpoint_url="https://s3.hf.co/%s" % ns,
    region_name="us-east-1",
    config=Config(
        s3={"addressing_style": "path"},
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    ),
)
s3.create_bucket(Bucket=bucket)          # idempotent for a bucket that exists

TYPES = {".md": "text/markdown; charset=utf-8"}
root = pathlib.Path(stage)
for f in sorted(p for p in root.rglob("*") if p.is_file()):
    key = str(f.relative_to(root))
    ctype = TYPES.get(f.suffix) or mimetypes.guess_type(key)[0] or "text/plain; charset=utf-8"
    s3.upload_file(str(f), bucket, key, ExtraArgs={"ContentType": ctype})
    print("  uploaded", key)

listed = s3.list_objects_v2(Bucket=bucket).get("Contents", [])
print("bucket now holds %d objects" % len(listed))
print("published: https://huggingface.co/buckets/%s/%s" % (ns, bucket))
BUCKET_PY
  exit 0
fi

[ -n "${HF_TOKEN:-}" ] || { echo "set HF_TOKEN (write-scoped) first" >&2; exit 1; }
"$VENV/bin/python" -c "import huggingface_hub" 2>/dev/null || \
  VIRTUAL_ENV="$VENV" uv pip install -q huggingface_hub

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
