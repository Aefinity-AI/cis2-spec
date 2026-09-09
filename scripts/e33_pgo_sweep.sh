#!/usr/bin/env bash
# E33: PGO -- the one codegen flag E29 sec 6 / spec 13.5 still listed as untested.
# Three profiles (including one trained on a DIFFERENT prompt than the one verified),
# two target-cpu settings, plus a thin-LTO+native+PGO cell.
set -uo pipefail
cd ~/projects/cis2-spec/cis2-verify
W=~/projects/cis2-spec/weights
TD=~/e33-target
PD=~/e33-prof
LLVMBIN=~/.rustup/toolchains/1.98.0-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/bin
mkdir -p $PD

cell () { # $1=label $2=cpu $3=extra rustflags $4=lto
  local lbl="$1" cpu="$2" extra="$3" lto="${4:-false}"
  local RF="$extra"; [ "$cpu" = native ] && RF="$RF -C target-cpu=native"
  CARGO_PROFILE_RELEASE_LTO=$lto \
  RUSTFLAGS="$RF" \
    cargo build --release --offline --features layerdump --example layer_dump \
    --target-dir $TD >/tmp/e33build.txt 2>&1 || { echo "$lbl $cpu BUILD FAIL"; tail -3 /tmp/e33build.txt; return; }
  local B=$TD/release/examples/layer_dump
  local BS=$(sha256sum $B|cut -d' ' -f1)
  objdump -d $B > /tmp/e33d.txt
  local Y=$(grep -c '%ymm' /tmp/e33d.txt || true)
  local F=$(grep -cEi 'vfmadd|vfnmadd|vfmsub|vfnmsub' /tmp/e33d.txt || true)
  LLVM_PROFILE_FILE="$PD/${lbl}-%p.profraw" $B "$W" ~/e33-cell.bin "Once upon a time" 16 >/tmp/e33r.txt 2>&1
  local DS=$(sha256sum ~/e33-cell.bin|cut -d' ' -f1)
  local w=$(awk '/witness-digest/{print substr($3,1,8)}' /tmp/e33r.txt)
  local a=$(awk '/argmax-digest/{print substr($3,1,8)}' /tmp/e33r.txt)
  local bytes=$(stat -c%s ~/e33-cell.bin)
  printf "%-18s %-8s %-64s %5s %4s %s w=%s a=%s bytes=%s\n" "$lbl" "$cpu" "$BS" "$Y" "$F" "$DS" "$w" "$a" "$bytes"
  rm -f ~/e33-cell.bin
}

printf "%-18s %-8s %-64s %5s %4s %s\n" cell tgtcpu binary_sha256 ymm fma dump_sha256

# --- stage 1: instrumented builds. These also produce the profiles.
rm -rf $PD/raw-once $PD/raw-math; mkdir -p $PD/raw-once $PD/raw-math
for cpu in generic native; do
  RF="-C profile-generate=$PD/raw-once"; [ "$cpu" = native ] && RF="$RF -C target-cpu=native"
  RUSTFLAGS="$RF" cargo build --release --offline --features layerdump --example layer_dump \
    --target-dir $TD >/tmp/e33build.txt 2>&1 || { echo "gen $cpu BUILD FAIL"; tail -5 /tmp/e33build.txt; continue; }
  B=$TD/release/examples/layer_dump
  BS=$(sha256sum $B|cut -d' ' -f1)
  objdump -d $B > /tmp/e33d.txt
  Y=$(grep -c '%ymm' /tmp/e33d.txt || true); F=$(grep -cEi 'vfmadd|vfnmadd|vfmsub|vfnmsub' /tmp/e33d.txt || true)
  LLVM_PROFILE_FILE="$PD/raw-once/gen-$cpu-%p.profraw" \
    $B "$W" ~/e33-cell.bin "Once upon a time" 16 >/tmp/e33r.txt 2>&1
  DS=$(sha256sum ~/e33-cell.bin|cut -d' ' -f1)
  w=$(awk '/witness-digest/{print substr($3,1,8)}' /tmp/e33r.txt)
  a=$(awk '/argmax-digest/{print substr($3,1,8)}' /tmp/e33r.txt)
  bytes=$(stat -c%s ~/e33-cell.bin)
  printf "%-18s %-8s %-64s %5s %4s %s w=%s a=%s bytes=%s\n" "profile-generate" "$cpu" "$BS" "$Y" "$F" "$DS" "$w" "$a" "$bytes"
  rm -f ~/e33-cell.bin
  # second run of the SAME instrumented native binary on a different prompt -> the cross profile
  if [ "$cpu" = native ]; then
    LLVM_PROFILE_FILE="$PD/raw-math/genmath-%p.profraw" \
      $B "$W" ~/e33-cell.bin "1234567890 + 9876543210 =" 16 >/tmp/e33rm.txt 2>&1
    DSM=$(sha256sum ~/e33-cell.bin|cut -d' ' -f1)
    wm=$(awk '/witness-digest/{print substr($3,1,8)}' /tmp/e33rm.txt)
    printf "%-18s %-8s %-64s %5s %4s %s w=%s (prompt=math, trains cross profile)\n" "profile-generate" "$cpu" "$BS" "$Y" "$F" "$DSM" "$wm"
    rm -f ~/e33-cell.bin
  fi
done

$LLVMBIN/llvm-profdata merge -output=$PD/once.profdata $PD/raw-once/*.profraw 2>&1 | head -3
$LLVMBIN/llvm-profdata merge -output=$PD/math.profdata $PD/raw-math/*.profraw 2>&1 | head -3
echo "profiles: once=$(stat -c%s $PD/once.profdata 2>/dev/null) math=$(stat -c%s $PD/math.profdata 2>/dev/null) bytes"

# --- stage 2: profile-use
for cpu in generic native; do
  cell "pgo:once"  "$cpu" "-C profile-use=$PD/once.profdata"
done
for cpu in generic native; do
  cell "pgo:cross" "$cpu" "-C profile-use=$PD/math.profdata"
done
cell "pgo:once+thinlto" native "-C profile-use=$PD/once.profdata" thin
cell "pgo:cross+fatlto" native "-C profile-use=$PD/math.profdata" fat
