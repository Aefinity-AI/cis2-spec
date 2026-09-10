#!/usr/bin/env bash
# E37: E36 widened E28's oracle comparison from one prompt to six -- and then
# named its own remaining limit: "Still one model. All six cells are
# SmolLM2-135M. E28's Qwen-0.5B cell is not re-run here, so 'six cells' means
# six inputs, not six architectures."  This runs the same six cells on
# Qwen2.5-0.5B.
#
# E28's Qwen figure (worst rel_l2 9.923e-05) is a per-cell maximum from one
# prompt, exactly the kind of number E36 showed must not be quoted as a bound.
# This checks the sibling of the correction E36 made.
#
# Qwen2.5-0.5B's tokenizer.json is refused by spec 3.1.3/3.1.4 (14.8 /
# erratum E-3), so prompt token ids are SUPPLIED on the command line from the
# oracle's own tokenizer. Such a run exercises spec 4-11 only and is not a
# conformance run -- the same footing E28's Qwen cell stood on.
set -uo pipefail
R=~/projects/cis2-spec
W=~/qwen05b
LD=$R/cis2-verify/target/release/examples/layer_dump
PY=~/venvs/torch/bin/python
V=~/e37-verifier.bin
O=~/e37-oracle.bin

cell=0
while IFS='|' read -r G P IDS; do
  [ -z "${G:-}" ] && continue
  cell=$((cell+1))
  echo "### cell=$cell gen_toks=$G prompt=<$P>"
  free -m | awk '/^Mem:/{printf "  E37 mem_avail_mb=%s\n",$7}'

  L=$("$LD" "$W" "$V" "$P" "$G" "$IDS" 2>&1) || { echo "  FAIL layer_dump"; continue; }
  echo "$L" | grep -E "LAYERDUMP (fp-env|tokenization|bytes|witness|argmax)" | sed 's/^/  /'
  PIDS=$(echo "$L" | sed -n 's/.*prompt-token-ids=\[\(.*\)\]/\1/p' | tr -d ' ')
  GIDS=$(echo "$L" | sed -n 's/.*generated-token-ids=\[\(.*\)\]/\1/p' | tr -d ' ')
  # The verifier never feeds the last generated token, so the oracle steps
  # prompt + generated[:-1] -- one forward per verifier forward, teacher forced.
  FEED=$(echo "$PIDS,$GIDS" | awk -F, '{for(i=1;i<NF;i++)printf "%s%s",$i,(i<NF-1?",":"")}')
  NFEED=$(echo "$FEED" | tr ',' '\n' | wc -l)
  echo "  E37 supplied_ids=$IDS"
  echo "  E37 prompt_ids=$PIDS"
  echo "  E37 generated_ids=$GIDS"
  echo "  E37 oracle_feed_len=$NFEED"
  # The verifier must have used the ids it was handed, not re-derived them.
  [ "$PIDS" = "$IDS" ] && echo "  E37 SUPPLIED-ID CHECK PASS" \
                       || echo "  E37 SUPPLIED-ID CHECK MISMATCH dump=$PIDS supplied=$IDS"

  OARG=$("$PY" "$R/scripts/oracle_layers.py" "$W" "$O" "$FEED" 2>/dev/null) || { echo "  FAIL oracle"; rm -f "$V" "$O"; continue; }
  echo "  E37 oracle_argmax_per_step=$OARG"
  echo "$OARG" | tr ',' '\n' | tail -n "$(echo "$GIDS" | tr ',' '\n' | wc -l)" | paste -sd, - | \
    awk -v want="$GIDS" '{print ($0==want)?"  E37 TOKEN CHECK PASS (oracle reproduces every generated id)":"  E37 TOKEN CHECK MISMATCH got="$0" want="want}'

  "$PY" "$R/scripts/compare_layers.py" "$V" "$O" 2>&1 | sed 's/^/  /'
  rm -f "$V" "$O"
done < ~/e37-cells.txt
