#!/usr/bin/env bash
# Runs both bundled verifiers against the bundled receipts and checks the
# output against the pinned EXPECTED_DIGESTS.txt. Fails loudly (non-zero
# exit, clear message) on any mismatch. No network access required.
set -u
cd "$(dirname "$0")"

fail=0

echo "=== CIS-1 (cis-verify) ==="
out1=$(./bin/cis-verify receipts/cis1_witness.receipt \
        model/MODEL.SAF model/EMBED.BIN model/VOCAB.BIN 2>&1)
echo "$out1"
exp1=$(grep '^CIS1_RESULT=' EXPECTED_DIGESTS.txt | cut -d= -f2-)
if ! echo "$out1" | grep -qF "$exp1"; then
  echo "FAIL: cis-verify output does not contain expected result '$exp1'"
  fail=1
fi

echo
echo "=== CIS-2 (agent_trace) ==="
out2=$(./bin/agent_trace verify model/MODEL.SAF model/EMBED.BIN model/VOCAB.BIN \
        receipts/cis2_agent_trace.receipt --table receipts/chain.tsv 2>&1)
echo "$out2"
exp_chain=$(grep '^CIS2_TRACE_CHAIN=' EXPECTED_DIGESTS.txt | cut -d= -f2-)
exp2=$(grep '^CIS2_RESULT=' EXPECTED_DIGESTS.txt | cut -d= -f2-)
if ! echo "$out2" | grep -qF "$exp_chain"; then
  echo "FAIL: agent_trace trace-chain does not match pinned value '$exp_chain'"
  fail=1
fi
if ! echo "$out2" | grep -qF "$exp2"; then
  echo "FAIL: agent_trace output does not contain expected result '$exp2'"
  fail=1
fi

echo
echo "=== sha256 manifest ==="
if command -v sha256sum >/dev/null 2>&1; then
  if sha256sum -c SHA256SUMS.txt --quiet 2>/dev/null; then
    echo "sha256 manifest OK"
  else
    echo "FAIL: sha256 manifest mismatch (run: sha256sum -c SHA256SUMS.txt)"
    fail=1
  fi
else
  echo "sha256sum not found, skipping manifest check"
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "RESULT: ALL CHECKS PASSED"
  exit 0
else
  echo "RESULT: ONE OR MORE CHECKS FAILED"
  exit 1
fi
