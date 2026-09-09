#!/usr/bin/env bash
# E30: does E29's every-intermediate invariance, and E27's reach census,
# hold across prompts and context lengths -- or were both one-input results?
set -uo pipefail
W=~/projects/cis2-spec/weights
CEN=~/projects/cis2-spec/cis2-verify/target/release/examples/silu_reach
while IFS='|' read -r G P; do
  [ -z "${G:-}" ] && continue
  echo "### gen_toks=$G prompt=<$P>"
  # 14.1 reach census
  $CEN "$W" "$P" "$G" 2>&1 | grep -E "REACH|IN 6.2|arg <|softmax" | sed 's/^/  /'
  # every-intermediate invariance across two codegen targets
  for b in generic native; do
    L=$(~/e30-bin-$b "$W" ~/e30-cell.bin "$P" "$G" 2>&1)
    echo "  $b dump=$(sha256sum ~/e30-cell.bin|cut -c1-16) bytes=$(echo "$L"|awk '/bytes=/{split($2,a,"=");print a[2]}')" \
         "w=$(echo "$L"|awk '/witness-digest/{print substr($3,1,8)}')" \
         "a=$(echo "$L"|awk '/argmax-digest/{print substr($3,1,8)}')" \
         "toks=$(echo "$L"|awk '/generated-token-ids/{print $2}'|cut -c1-40)"
  done
  rm -f ~/e30-cell.bin
done < ~/e30-prompts.txt
