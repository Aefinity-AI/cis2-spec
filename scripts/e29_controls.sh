#!/usr/bin/env bash
# E29 controls. An invariance result needs two of them, or it says nothing:
#
#   (a) execution -- that the vectorized loop whose invariance we claim is
#       actually on the executed path, and not behind a length check the
#       real shapes never satisfy;
#   (b) sensitivity -- that the dump moves at all, so eight runs agreeing on
#       51,750,154 bytes is a measurement rather than a tautology.
#
# Usage: e29_controls.sh <native-binary> <artifact-dir> <workdir>
# See docs/E29_OPTIMIZER_INVARIANCE.md sections 4 and 5.
set -euo pipefail

BIN=${1:?usage: e29_controls.sh <native-binary> <artifact-dir> <workdir>}
ART=${2:?}
WORK=${3:?}
mkdir -p "$WORK"

echo "=== reference dump ==="
"$BIN" "$ART" "$WORK/ref.bin" | grep -E 'bytes|witness|argmax'

echo
echo "=== control (a): does the AVX2 vector loop execute? ==="
# The four consecutive 8-wide squarings of rmsnorm's sum-of-squares map.
# Overwritten in a *copy* with ud2;nop;nop -- same length, invalid opcode.
python3 - "$BIN" "$WORK/ud2_bin" <<'PY'
import shutil, sys
src, dst = sys.argv[1], sys.argv[2]
shutil.copy(src, dst)
pat = bytes.fromhex("c5fc59c0" "c5f459c9" "c5ec59d2" "c5e459db")
b = open(dst, "rb").read()
n = b.count(pat)
if n != 1:
    sys.exit(f"expected exactly one rmsnorm vmulps block, found {n} "
             "(non-AVX2 build, or a different rustc/LLVM laid it out differently)")
off = b.find(pat)
with open(dst, "r+b") as f:
    f.seek(off); f.write(bytes.fromhex("0f0b9090"))
print(f"patched vmulps -> ud2 at file offset {off:#x}")
PY
chmod +x "$WORK/ud2_bin"
set +e
"$WORK/ud2_bin" "$ART" "$WORK/ud2.bin" >/dev/null 2>&1
rc=$?
set -e
echo "exit=$rc  (132 = 128+SIGILL: the instruction executed; 0 = it did not)"
[ "$rc" -eq 132 ] || echo "NOTE: not 132 -- the vector path did not execute in this build"

echo
echo "=== control (b): does one flipped weight bit move the dump? ==="
rm -rf "$WORK/flip" && mkdir -p "$WORK/flip"
cp "$ART"/config.json "$ART"/tokenizer.json "$WORK/flip/"
cp "$ART"/model.safetensors "$WORK/flip/"
python3 - "$WORK/flip/model.safetensors" <<'PY'
import sys
p = sys.argv[1]
off = 200_000_000          # well past the safetensors header, inside tensor data
with open(p, "r+b") as f:
    f.seek(off); b = f.read(1)
    f.seek(off); f.write(bytes([b[0] ^ 0x01]))
print(f"flipped one mantissa bit at offset {off}: {b[0]:#x} -> {b[0]^1:#x}")
PY
"$BIN" "$WORK/flip" "$WORK/flip.bin" | grep -E 'witness|argmax'
echo "-- argmax-digest above must MATCH the reference; witness-digest must NOT --"
python3 "$(dirname "$0")/diff_dumps.py" "$WORK/ref.bin" "$WORK/flip.bin" --max-report 4
