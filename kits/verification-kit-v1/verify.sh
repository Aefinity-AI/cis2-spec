#!/bin/sh
# Runs both bundled verifiers against the bundled receipts and checks the
# output against the pinned EXPECTED_DIGESTS.txt. Fails loudly (non-zero
# exit, clear message) on any mismatch. No network access required.
# POSIX sh (not bash-specific) so it also runs under a plain `sh` on
# minimal/Android-toybox environments.
set -u
cd "$(dirname "$0")"

# pick the binaries built for this machine's architecture
ARCH=$(uname -m)
BINDIR="bin/$ARCH"
if [ ! -x "$BINDIR/cis-verify" ] || [ ! -x "$BINDIR/agent_trace" ]; then
  echo "FAIL: no verifier binaries for this architecture ($ARCH) — looked in $BINDIR"
  echo "available: $(ls bin 2>/dev/null)"
  exit 1
fi

fail=0

echo "=== CIS-1 (cis-verify, $ARCH) ==="
out1=$("$BINDIR/cis-verify" receipts/cis1_witness.receipt \
        model/MODEL.SAF model/EMBED.BIN model/VOCAB.BIN 2>&1)
echo "$out1"
exp1=$(grep '^CIS1_RESULT=' EXPECTED_DIGESTS.txt | cut -d= -f2-)
if ! echo "$out1" | grep -qF "$exp1"; then
  echo "FAIL: cis-verify output does not contain expected result '$exp1'"
  fail=1
fi

echo
echo "=== CIS-2 (agent_trace, $ARCH) ==="
out2=$("$BINDIR/agent_trace" verify model/MODEL.SAF model/EMBED.BIN model/VOCAB.BIN \
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
  # -q/--quiet is not portable (toybox's sha256sum on Android lacks it);
  # capture full -c output and just check the exit code + grep for FAILED.
  sha_out=$(sha256sum -c SHA256SUMS.txt 2>&1)
  sha_rc=$?
  if [ "$sha_rc" -eq 0 ] && ! echo "$sha_out" | grep -q FAILED; then
    echo "sha256 manifest OK"
  else
    echo "$sha_out"
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
