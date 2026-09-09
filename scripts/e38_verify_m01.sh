#!/usr/bin/env bash
# E38 adversarial verification of the Qwen 256-token band hit.
#   (1) reproducibility of cell 10
#   (2) where in the decode it first fires (128, 192)
#   (3) M01: does the digest MOVE when 1.3's FTZ/DAZ pin is gutted?
set -uo pipefail
SRC=~/projects/cis2-spec/cis2-verify
CEN=$SRC/target/release/examples/silu_reach
Q=~/qwen05b
IDS=12522,5193,264,882
P="Once upon a time"

echo "===== (1) reproducibility: pinned build, gen=256, second run"
"$CEN" "$Q" "$P" 256 "$IDS" 2>&1 | grep -E "LOW guard|SUBNORMAL BAND|softmax arg range|witness-digest|argmax-digest"

for G in 128 192; do
  echo "===== (2) pinned build, gen=$G"
  "$CEN" "$Q" "$P" "$G" "$IDS" 2>&1 | grep -E "LOW guard|SUBNORMAL BAND|softmax arg range|softmax exp args|witness-digest"
done

echo "===== (3) building M01 mutant (1.3 pin gutted; FTZ/DAZ explicitly CLEARED)"
rm -rf ~/e38-mut && cp -r "$SRC" ~/e38-mut && rm -rf ~/e38-mut/target
python3 - <<'PY'
import pathlib, re
p = pathlib.Path.home()/ "e38-mut/src/fpenv.rs"
s = p.read_text()
old = """pub fn pin_and_selftest() -> Result<(), FpEnvError> {
    let cur = arch::read_control();
    arch::write_control(cur | arch::REQUIRED);"""
new = """pub fn pin_and_selftest() -> Result<(), FpEnvError> {
    // M01 MUTANT (E38): 1.3's pin is gutted. FTZ/DAZ are explicitly CLEARED
    // and the self-test below is skipped, so denormals are computed and kept.
    let cur = arch::read_control();
    arch::write_control(cur & !arch::REQUIRED);
    return Ok(());
    #[allow(unreachable_code)]
    {
    let cur = arch::read_control();
    arch::write_control(cur | arch::REQUIRED);"""
assert old in s
s = s.replace(old, new, 1)
old2 = """    Ok(())
}

/// Re-assert the readback without re-writing"""
new2 = """    Ok(())
    }
}

/// Re-assert the readback without re-writing"""
assert old2 in s
s = s.replace(old2, new2, 1)
p.write_text(s)
print("mutant patched")
PY
cd ~/e38-mut && cargo build --release --offline --features census --example silu_reach 2>&1 | tail -2
MUT=~/e38-mut/target/release/examples/silu_reach
echo "===== (3a) MUTANT gen=256 (expect control= line to show the pin absent)"
"$MUT" "$Q" "$P" 256 "$IDS" 2>&1 | grep -E "fp-env|LOW guard|SUBNORMAL BAND|softmax arg range|witness-digest|argmax-digest"
echo "===== (3b) MUTANT gen=16 control cell (no band hit expected -> digests must MATCH pinned)"
"$MUT" "$Q" "$P" 16 "$IDS" 2>&1 | grep -E "SUBNORMAL BAND|witness-digest|argmax-digest"
echo "===== (3c) PINNED gen=16 for comparison"
"$CEN" "$Q" "$P" 16 "$IDS" 2>&1 | grep -E "SUBNORMAL BAND|witness-digest|argmax-digest"
echo "== E38-VERIFY DONE"
