#!/usr/bin/env bash
# E38: close spec 14.1(a)'s three remaining scope limits -- "one model, six
# ASCII prompts and at most 64 generated tokens" -- for the 6.2 clip-band /
# subnormal-band reach census.
#
# Every cell SUPPLIES its prompt token ids (silu_reach's 4th argument), so:
#   * a non-ASCII prompt does not depend on 3.1.4's still-open Unicode
#     question, and
#   * a Qwen2.5-0.5B cell does not depend on 3.1.3/3.1.4 refusing that
#     checkpoint's tokenizer.json (14.8 / erratum E-3).
# Such a run exercises 4-11 only and is NOT a conformance run.  Cell 1 on
# SmolLM2 is additionally run through 3 as a control: the supplied-id path must
# reproduce the spec-3 path's digests exactly, or the other cells mean nothing.
set -uo pipefail
CEN=~/projects/cis2-spec/cis2-verify/target/release/examples/silu_reach

run_cells() {  # $1 = model tag, $2 = artifact dir, $3 = cells file
  local TAG=$1 W=$2 F=$3 n=0
  echo "##### model=$TAG dir=$W"
  while IFS='|' read -r G P IDS WHY; do
    [ -z "${G:-}" ] && continue
    n=$((n+1))
    echo "### $TAG cell=$n gen_toks=$G why=<$WHY> prompt=<$P>"
    echo "  E38 supplied_ids=$IDS"
    free -m | awk '/^Mem:/{printf "  E38 mem_avail_mb=%s\n",$7}'
    OUT=$("$CEN" "$W" "$P" "$G" "$IDS" 2>&1) || { echo "  FAIL silu_reach"; continue; }
    echo "$OUT" | grep -E "^REACH" | sed 's/^/  /'
    # The census must have used the ids it was handed, not re-derived them.
    U=$(echo "$OUT" | sed -n 's/.*prompt_token_ids=\[\([0-9, ]*\)\].*/\1/p' | tr -d ' ')
    [ "$U" = "$IDS" ] && echo "  E38 SUPPLIED-ID CHECK PASS" \
                      || echo "  E38 SUPPLIED-ID CHECK MISMATCH used=$U supplied=$IDS"
  done < "$F"
}

echo "== E38 control: cell 1 on SmolLM2 through spec-3 (ids derived), for digest comparison"
"$CEN" ~/projects/cis2-spec/weights "Once upon a time" 16 2>&1 | grep -E "^REACH" | sed 's/^/  /'

run_cells smollm2 ~/projects/cis2-spec/weights ~/e38-cells-smollm2.txt
run_cells qwen    ~/qwen05b                    ~/e38-cells-qwen.txt
echo "== E38 DONE"
