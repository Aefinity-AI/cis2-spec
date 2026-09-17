#!/usr/bin/env bash
# E36: E28 closed 14.6 on ONE prompt at 19 positions. Its own scope note says
# so: "Not exhaustive over inputs. One prompt, 19 positions, two models. A
# compensating pair that only appears at a context length or an activation
# pattern outside this run is not excluded."  This runs the same
# every-intermediate oracle comparison over E30's six cells, one at a time,
# deleting each pair of dumps before the next so the disk footprint stays flat.
set -uo pipefail
R=~/projects/cis2-spec
W=$R/weights
LD=$R/cis2-verify/target/release/examples/layer_dump
PY=~/venvs/torch/bin/python
V=~/e36-verifier.bin
O=~/e36-oracle.bin

cell=0
while IFS='|' read -r G P; do
  [ -z "${G:-}" ] && continue
  cell=$((cell+1))
  echo "### cell=$cell gen_toks=$G prompt=<$P>"

  L=$("$LD" "$W" "$V" "$P" "$G" 2>&1) || { echo "  FAIL layer_dump"; continue; }
  echo "$L" | grep -E "LAYERDUMP (fp-env|bytes|witness|argmax)" | sed 's/^/  /'
  PIDS=$(echo "$L" | sed -n 's/.*prompt-token-ids=\[\(.*\)\]/\1/p' | tr -d ' ')
  GIDS=$(echo "$L" | sed -n 's/.*generated-token-ids=\[\(.*\)\]/\1/p' | tr -d ' ')
  # The verifier never feeds the last generated token, so the oracle steps
  # prompt + generated[:-1] -- one forward per verifier forward, teacher forced.
  FEED=$(echo "$PIDS,$GIDS" | awk -F, '{for(i=1;i<NF;i++)printf "%s%s",$i,(i<NF-1?",":"")}')
  NFEED=$(echo "$FEED" | tr ',' '\n' | wc -l)
  echo "  E36 prompt_ids=$PIDS"
  echo "  E36 generated_ids=$GIDS"
  echo "  E36 oracle_feed_len=$NFEED"

  OARG=$("$PY" "$R/scripts/oracle_layers.py" "$W" "$O" "$FEED" 2>/dev/null) || { echo "  FAIL oracle"; rm -f "$V" "$O"; continue; }
  echo "  E36 oracle_argmax_per_step=$OARG"
  # Independent token check: the oracle's argmax after each of the last G-1
  # fed positions must reproduce the verifier's generated ids 1..G-1, and its
  # argmax at the end of the prompt must reproduce generated id 0.
  echo "$OARG" | tr ',' '\n' | tail -n "$(echo "$GIDS" | tr ',' '\n' | wc -l)" | paste -sd, - | \
    awk -v want="$GIDS" '{print ($0==want)?"  E36 TOKEN CHECK PASS (oracle reproduces every generated id)":"  E36 TOKEN CHECK MISMATCH got="$0" want="want}'

  "$PY" "$R/scripts/compare_layers.py" "$V" "$O" 2>&1 | sed 's/^/  /'
  rm -f "$V" "$O"
done < ~/e30-prompts.txt
