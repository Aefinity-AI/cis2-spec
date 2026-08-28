#!/usr/bin/env bash
# CIS-2 §3.2 mechanical FMA gate — codifies the two checks used to confirm
# zero FMA contraction in the reference implementation:
#   1. grep gate: no `.mul_add(` call sites anywhere in src/ (the only
#      construct that lowers to LLVM's fused `llvm.fma` intrinsic in Rust —
#      ordinary `a*b + c` never auto-fuses, see src/math.rs module docs and
#      the E14a design memo §2 "Compiler flags" row).
#   2. disassembly gate: build the release binary and confirm no
#      vfmadd*/vfnmadd*/vfmsub*/vfnmsub*/fmadd/fmsub instruction appears
#      anywhere in .text.
#
# Exit 0 = both gates pass. Exit 1 = a gate failed (prints the offending
# lines/instructions). Exit 2 = a required tool (objdump) is missing.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== gate 1: grep for .mul_add( in src/ =="
HITS_SRC=$(grep -rn '\.mul_add(' src/ || true)
if [ -n "$HITS_SRC" ]; then
  echo "FAIL: found .mul_add( call sites (forbidden by CIS-2 §3.2):" >&2
  echo "$HITS_SRC" >&2
  exit 1
fi
echo "PASS: no .mul_add( call sites in src/"

echo "== gate 2: disassemble release binary, search for FMA instructions =="
cargo build --release
BIN=target/release/cis2_ref
if ! command -v objdump >/dev/null; then
  echo "objdump not found; cannot run disassembly gate" >&2
  exit 2
fi
HITS_ASM=$(objdump -d "$BIN" | grep -icE "vfmadd|vfnmadd|vfmsub|vfnmsub|fmadd|fmsub" || true)
if [ "$HITS_ASM" -ne 0 ]; then
  echo "FAIL: found $HITS_ASM FMA instruction(s) in $BIN — §3.2 violated" >&2
  objdump -d "$BIN" | grep -iE "vfmadd|vfnmadd|vfmsub|vfnmsub|fmadd|fmsub" >&2
  exit 1
fi
echo "PASS: no FMA instructions found in $BIN (0 matches for vfmadd/vfnmadd/vfmsub/vfnmsub/fmadd/fmsub)"

echo "== both FMA gates PASS =="
