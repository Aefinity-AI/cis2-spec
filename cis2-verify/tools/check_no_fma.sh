#!/usr/bin/env bash
# CIS-2 v0.3b 1.4 forbids fused multiply-add anywhere in the computation:
# an FMA keeps the product at full width, so a*b+c rounds once instead of
# twice and the result differs from the spec's two-rounding sequence.
#
# The source-level rule (never call f32::mul_add) is not enough on its own,
# because a compiler is free to contract a*b+c into an FMA by itself. This
# script checks the property where it actually has to hold: in the emitted
# machine code.
#
# Usage: tools/check_no_fma.sh [path-to-binary]
set -euo pipefail

BIN="${1:-target/release/cis2-verify}"

if [ ! -f "$BIN" ]; then
    echo "check_no_fma: no binary at $BIN (run: cargo build --release)" >&2
    exit 2
fi

if ! command -v objdump >/dev/null 2>&1; then
    echo "check_no_fma: objdump not found; install binutils" >&2
    exit 2
fi

ARCH="$(uname -m)"
case "$ARCH" in
    x86_64)
        # AVX2/AVX-512 scalar and packed FMA, and the 4-operand FMA4 forms.
        PATTERN='\b(vfmadd|vfmsub|vfnmadd|vfnmsub|vfmaddsub|vfmsubadd)[0-9a-z]*'
        ;;
    aarch64)
        # fmadd/fmsub/fnmadd/fnmsub are scalar; fmla/fmls are the vector forms.
        # `fmul` and `fadd` are fine -- they round separately, which is what
        # the spec asks for.
        PATTERN='\b(fmadd|fmsub|fnmadd|fnmsub|fmla|fmls)\b'
        ;;
    *)
        echo "check_no_fma: no FMA mnemonic list for $ARCH; refusing to report a pass" >&2
        exit 2
        ;;
esac

HITS="$(objdump -d --no-show-raw-insn "$BIN" 2>/dev/null \
        | grep -Eio "$PATTERN" | sort | uniq -c | sort -rn || true)"

if [ -n "$HITS" ]; then
    echo "check_no_fma: FAIL -- $ARCH FMA instructions present in $BIN"
    echo "$HITS"
    echo
    echo "CIS-2 v0.3b 1.4: the decode must not use fused multiply-add."
    exit 1
fi

echo "check_no_fma: PASS -- 0 FMA instructions in $BIN ($ARCH)"
