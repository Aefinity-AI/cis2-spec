#!/usr/bin/env bash
# Makes 4 tampered copies of the bundled receipts, one distinct tamper
# class each, and shows all 4 correctly FAIL verification. No network
# access required. Exits 0 only if all 4 tampers were caught (i.e. the
# verifiers did their job); exits 1 if any tamper was NOT caught, which
# would mean a verifier bug.
#
# This kit's bundled receipts have no gateway-issued capability token or
# session/counter field (those are properties of the live gateway socket
# protocol, not of a CIS-1/CIS-2 receipt file, and the gateway binary is
# out of scope for this offline kit). The 4 tampers below are chosen to
# cover the closest available attack surface on the two receipt formats
# that *are* bundled:
#   1. flip a byte in an integrity digest field (CIS-1 witness `chain`)
#   2. edit a recorded argument field (CIS-2 step `ctx=`)
#   3. truncate the chain (drop a CIS-2 step line)
#   4. replay the same trace data under a different claimed identity
#      (CIS-2 `host` line — format 3+ folds host/commit into the trace
#      genesis specifically so this fails; this is the closest receipt-
#      level analog of "replay under a different session")
set -u
cd "$(dirname "$0")"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fail=0

check_fails() {
  local label="$1"; shift
  if "$@" >"$tmp/out.txt" 2>&1; then
    echo "FAIL (BUG): tamper '$label' was accepted — verifier did not catch it"
    cat "$tmp/out.txt"
    fail=1
  else
    echo "OK: tamper '$label' correctly rejected:"
    sed 's/^/    /' "$tmp/out.txt"
  fi
  echo
}

echo "--- Tamper 1: flip a byte in the CIS-1 witness chain digest ---"
sed 's/aee25b770bd7b22eea2ea8476bbd949881d58a98d6dc3085c7cc94d322b1961b$/aee25b770bd7b22eea2ea8476bbd949881d58a98d6dc3085c7cc94d322b1961a/' \
  receipts/cis1_witness.receipt > "$tmp/t1.receipt"
check_fails "flipped chain digest byte" \
  ./bin/cis-verify "$tmp/t1.receipt" model/MODEL.SAF model/EMBED.BIN model/VOCAB.BIN

echo "--- Tamper 2: edit a recorded argument (step 0 ctx=) ---"
sed 's/ctx=ce60b59df7fb416f06c6a34d5432d9ce0acd76def6db096fababf2e0fa180fe0 q=ce60b59df7fb416f06c6a34d5432d9ce0acd76def6db096fababf2e0fa180fe0/ctx=ce60b59df7fb416f06c6a34d5432d9ce0acd76def6db096fababf2e0fa180fe1 q=ce60b59df7fb416f06c6a34d5432d9ce0acd76def6db096fababf2e0fa180fe0/' \
  receipts/cis2_agent_trace.receipt > "$tmp/t2.receipt"
check_fails "edited ctx= argument" \
  ./bin/agent_trace verify model/MODEL.SAF model/EMBED.BIN model/VOCAB.BIN "$tmp/t2.receipt" --table receipts/chain.tsv

echo "--- Tamper 3: truncate the chain (drop the last recorded step) ---"
head -n -2 receipts/cis2_agent_trace.receipt > "$tmp/t3.receipt"
tail -n1 receipts/cis2_agent_trace.receipt >> "$tmp/t3.receipt"
check_fails "dropped step + kept old trace-chain" \
  ./bin/agent_trace verify model/MODEL.SAF model/EMBED.BIN model/VOCAB.BIN "$tmp/t3.receipt" --table receipts/chain.tsv

echo "--- Tamper 4: replay identical trace data under a different claimed identity ---"
sed 's/^host .*$/host some-other-machine/' \
  receipts/cis2_agent_trace.receipt > "$tmp/t4.receipt"
check_fails "relabelled host line" \
  ./bin/agent_trace verify model/MODEL.SAF model/EMBED.BIN model/VOCAB.BIN "$tmp/t4.receipt" --table receipts/chain.tsv

if [ "$fail" -eq 0 ]; then
  echo "RESULT: all 4 tampers correctly rejected"
  exit 0
else
  echo "RESULT: one or more tampers were NOT rejected — investigate"
  exit 1
fi
