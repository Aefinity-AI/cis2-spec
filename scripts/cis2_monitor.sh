#!/usr/bin/env bash
# Minimal CIS-2 continuous-validation monitor
# Wraps ./selfcheck.sh or scripts/self_check.sh for scheduled / on-demand integrity checks.
# See docs/public/Continuous_Validation_Monitor_Sketch.md
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${CIS2_MONITOR_LOG_DIR:-$REPO_ROOT/monitor_logs}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="$LOG_DIR/cis2_monitor_$TIMESTAMP.log"
SUMMARY_FILE="$LOG_DIR/cis2_monitor_summary.log"

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

echo "== CIS-2 monitor run at $TIMESTAMP ==" | tee "$LOG_FILE"

if [[ -x ./selfcheck.sh ]]; then
  CHECK_CMD="./selfcheck.sh"
elif [[ -x scripts/self_check.sh ]]; then
  CHECK_CMD="scripts/self_check.sh"
else
  echo "ERROR: neither ./selfcheck.sh nor scripts/self_check.sh found" | tee -a "$LOG_FILE"
  exit 2
fi

set +e
$CHECK_CMD >> "$LOG_FILE" 2>&1
STATUS=$?
set -e

if grep -q "PASS: all" "$LOG_FILE" 2>/dev/null || grep -q "PASS: all digests match" "$LOG_FILE" 2>/dev/null; then
  RESULT="PASS"
else
  RESULT="FAIL"
  STATUS=1
fi

# Witness digest as printed by the self-check ("CIS2_VERIFY3 digest=<64 hex> ...").
DIGEST_HINT=$(grep -oE 'CIS2_VERIFY3 digest=[a-f0-9]{64}' "$LOG_FILE" 2>/dev/null | tail -1 | cut -d= -f2 | cut -c1-16 || true)
DIGEST_HINT="${DIGEST_HINT:-unknown}"

echo "$TIMESTAMP  $RESULT  digest_hint=$DIGEST_HINT  log=$LOG_FILE" | tee -a "$SUMMARY_FILE"
echo "== RESULT: $RESULT ==" | tee -a "$LOG_FILE"

exit $STATUS
