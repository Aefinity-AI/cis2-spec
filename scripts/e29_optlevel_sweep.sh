#!/usr/bin/env bash
# E29 extension: is the every-intermediate identity stable across opt-level too?
# Spec 13.4 covers opt-level 0..3,s at the digest level; 13.5 covered only
# --release x {generic,native}. This closes that stated scope limit.
set -uo pipefail
cd ~/projects/cis2-spec/cis2-verify
TD=~/e29-target
OUT=~/e29-cell.bin
printf '%-8s %-8s %-64s %6s %5s %-24s %s\n' optlvl tgtcpu binary_sha256 ymm fma matvec_fp dump_sha256
for lvl in 0 1 2 3 s; do
  for cpu in generic native; do
    if [ "$cpu" = native ]; then RF="-C target-cpu=native"; else RF=""; fi
    CARGO_PROFILE_RELEASE_OPT_LEVEL=$lvl RUSTFLAGS="$RF" \
      cargo build --release --offline --features layerdump --example layer_dump \
      --target-dir $TD >/dev/null 2>&1 || { echo "BUILD FAIL lvl=$lvl cpu=$cpu"; continue; }
    B=$TD/release/examples/layer_dump
    BS=$(sha256sum $B | cut -d' ' -f1)
    YMM=$(objdump -d $B | grep -c '%ymm')
    FMA=$(objdump -d $B | grep -cE '\bv?f(n?m)(add|sub)[0-9a-z]*')
    MV=$(objdump -d --demangle $B | awk '/^[0-9a-f]+ <cis2_verify::ops::matvec>:/{f=1;next} f&&/^[0-9a-f]+ </{f=0} f' \
         | grep -oE '\bv?(mul|add)[ps][sd]\b' | sort | uniq -c | awk '{printf "%s%s ",$1,$2}')
    LOG=$($B ~/projects/cis2-spec/weights $OUT 2>&1)
    W=$(echo "$LOG" | awk '/witness-digest/{print substr($3,1,8)}')
    A=$(echo "$LOG" | awk '/argmax-digest/{print substr($3,1,8)}')
    DS=$(sha256sum $OUT | cut -d' ' -f1)
    printf '%-8s %-8s %-64s %6s %5s %-24s %s w=%s a=%s\n' "$lvl" "$cpu" "$BS" "$YMM" "$FMA" "${MV:-none}" "$DS" "$W" "$A"
  done
done
rm -f $OUT
