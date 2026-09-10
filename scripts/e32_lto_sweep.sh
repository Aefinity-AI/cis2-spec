#!/usr/bin/env bash
# E32: the codegen flags E31 listed as untested -- LTO and codegen-units=1.
set -uo pipefail
cd ~/projects/cis2-spec/cis2-verify
W=~/projects/cis2-spec/weights
TD=~/e32-target
printf "%-22s %-8s %-64s %5s %4s %s\n" flags tgtcpu binary_sha256 ymm fma dump_sha256
for cfg in "lto=fat" "lto=thin" "cgu=1" "lto=fat,cgu=1"; do
  for cpu in generic native; do
    env_lto=""; env_cgu=""
    case "$cfg" in *lto=fat*) env_lto=fat;; *lto=thin*) env_lto=thin;; esac
    case "$cfg" in *cgu=1*) env_cgu=1;; esac
    RF=""; [ "$cpu" = native ] && RF="-C target-cpu=native"
    CARGO_PROFILE_RELEASE_LTO=${env_lto:-false} \
    CARGO_PROFILE_RELEASE_CODEGEN_UNITS=${env_cgu:-16} \
    RUSTFLAGS="$RF" \
      cargo build --release --offline --features layerdump --example layer_dump \
      --target-dir $TD >/dev/null 2>&1 || { echo "$cfg $cpu BUILD FAIL"; continue; }
    B=$TD/release/examples/layer_dump
    BS=$(sha256sum $B|cut -d' ' -f1)
    objdump -d $B > /tmp/e32d.txt
    Y=$(grep -c '%ymm' /tmp/e32d.txt || true)
    F=$(grep -cEi 'vfmadd|vfnmadd|vfmsub|vfnmsub' /tmp/e32d.txt || true)
    $B "$W" ~/e32-cell.bin "Once upon a time" 16 >/tmp/e32r.txt 2>&1
    DS=$(sha256sum ~/e32-cell.bin|cut -d' ' -f1)
    w=$(awk '/witness-digest/{print substr($3,1,8)}' /tmp/e32r.txt)
    printf "%-22s %-8s %-64s %5s %4s %s w=%s\n" "$cfg" "$cpu" "$BS" "$Y" "$F" "$DS" "$w"
    rm -f ~/e32-cell.bin
  done
done
