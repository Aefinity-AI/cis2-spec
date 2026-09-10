#!/usr/bin/env bash
set -uo pipefail
W=~/e28-src/weights
[ -x /tmp/tnative/release/examples/layer_dump ] || { PATH=$HOME/.cargo/bin:$PATH; cd ~/e28-src/cis2-verify && RUSTFLAGS="-C target-cpu=native" cargo build --release --offline --features layerdump --example layer_dump --target-dir /tmp/tnative >/dev/null 2>&1; }
while IFS="|" read -r G P; do
  [ -z "${G:-}" ] && continue
  echo "### gen_toks=$G prompt=<$P>"
  for b in "generic:$HOME/e30-bin-generic" "sse-native:/tmp/tnative/release/examples/layer_dump"; do
    n=${b%%:*}; x=${b#*:}
    L=$($x "$W" ~/e30-cell.bin "$P" "$G" 2>&1)
    echo "  $n dump=$(sha256sum ~/e30-cell.bin|cut -c1-16) w=$(echo "$L"|awk "/witness-digest/{print substr(\$3,1,8)}") a=$(echo "$L"|awk "/argmax-digest/{print substr(\$3,1,8)}")"
  done
  rm -f ~/e30-cell.bin
done < ~/e30-prompts.txt
