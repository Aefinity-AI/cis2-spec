#!/usr/bin/env bash
# E35: close E22 follow-up item 1 -- does a real pinned decode ever put a
# softmax exp argument inside the UNGUARDED subnormal band [-88.0,
# -87.33654022216797)?  Inside that band exp_pinned computes a subnormal that
# 1.3's FTZ flushes to zero; below -88.0 the 6.2 LOW guard returns 0.0 and FTZ
# is irrelevant.  E29's counter measured only the guarded side.
set -uo pipefail
W=~/projects/cis2-spec/weights
CEN=~/projects/cis2-spec/cis2-verify/target/release/examples/silu_reach
while IFS='|' read -r G P; do
  [ -z "${G:-}" ] && continue
  echo "### gen_toks=$G prompt=<$P>"
  $CEN "$W" "$P" "$G" 2>&1 | grep -E "REACH" | sed 's/^/  /'
done < ~/e30-prompts.txt
