#!/usr/bin/env bash
# Local reproduction check: builds verify3/ (C clean-room) with the system
# compiler, runs it against the pinned SmolLM2-135M test vector, and
# compares every digest against EXPECTED_DIGESTS.md / docs/CIS2_SPEC_v0.2.md
# §13.1. No timing numbers are printed or recorded.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CC="${CC:-gcc}"

echo "== CIS-2 v0.2 self-check (verify3/, CC=$CC) =="

echo "-- fetching weights --"
"$ROOT_DIR/scripts/fetch_weights.sh" "$ROOT_DIR/weights"

echo "-- building verify3 --"
make -C "$ROOT_DIR/verify3" clean >/dev/null 2>&1 || true
make -C "$ROOT_DIR/verify3" CC="$CC"

echo "-- FMA gate (objdump) --"
if command -v objdump >/dev/null 2>&1; then
  objdump -d "$ROOT_DIR/verify3/cis2_verify3" > /tmp/cis2_spec_disasm.txt
  if grep -Ei 'vfmadd|vfnmadd|vfmsub|vfnmsub|fmadd|fmsub|fmla|fmls' /tmp/cis2_spec_disasm.txt; then
    echo "FMA-family instruction found in disassembly — spec §1.4 violation" >&2
    exit 1
  fi
  echo "FMA gate: PASS (0 matches)"
else
  echo "objdump not found, skipping FMA gate"
fi

echo "-- running verify3 --"
OUT="$(cd "$ROOT_DIR/verify3" && ./cis2_verify3 ../weights/model.safetensors ../weights/config.json ../weights/tokenizer.json)"
echo "$OUT"

# Portable field extraction: prefer GNU grep -P (PCRE lookbehind), fall
# back to grep -E + sed for macOS/BSD grep (which lacks -P).
extract_field() {
  local field="$1"
  if printf '' | grep -P '' >/dev/null 2>&1; then
    echo "$OUT" | grep -oP "(?<=^CIS2_VERIFY3 ${field}=)[0-9a-f]+"
  else
    echo "$OUT" | grep -E "^CIS2_VERIFY3 ${field}=" | sed -E "s/^CIS2_VERIFY3 ${field}=([0-9a-f]+).*/\1/"
  fi
}

witness=$(extract_field digest)
argmax=$(extract_field argmax_digest)
table=$(extract_field table_digest)
invfreq=$(extract_field inv_freq_table_digest)

exp_witness=a0c563ef804f50413b7fb6619ae4afe9b51b1ffa7655e944221393e85d6261da
exp_argmax=0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff
exp_table=465d358ccd63721256dbd2abbc77ad5de755adf3230635f95b12b1727dfa1ea3
exp_invfreq=da9f6dcfde0425588815509e874515cdcd3d6b8818b6d0136590052e7bbf6f12

echo
echo "witness digest: got=$witness expected=$exp_witness"
echo "argmax digest:  got=$argmax expected=$exp_argmax"
echo "table digest:   got=$table expected=$exp_table"
echo "invfreq digest: got=$invfreq expected=$exp_invfreq"

fail=0
[ "$witness" = "$exp_witness" ] || { echo "MISMATCH: witness digest (CIS2_REF)"; fail=1; }
[ "$argmax"  = "$exp_argmax" ]  || { echo "MISMATCH: argmax digest"; fail=1; }
[ "$table"   = "$exp_table" ]   || { echo "MISMATCH: table digest"; fail=1; }
[ "$invfreq" = "$exp_invfreq" ] || { echo "MISMATCH: inv_freq_table digest"; fail=1; }

if [ "$fail" -eq 0 ]; then
  echo
  echo "PASS: all digests match the pinned CIS-2 v0.2 test vector."
else
  echo
  echo "FAIL: one or more digests did not match. See MISMATCH lines above." >&2
fi
exit $fail
