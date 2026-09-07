# E15m pre-registration — CIS-2 spec v0.3: f64-staged RoPE range reduction

**Status: pre-registered before implementation.** Written to close H2's one
stated finding (`docs/H2_TRANSCENDENTAL_RIGOR.md`, §6.7 of
`docs/CIS2_SPEC_v0.2.md`): `sin_pinned`/`cos_pinned`'s 2π range reduction,
done entirely in `f32`, suffers catastrophic cancellation growing with the
RoPE angle magnitude (`pos · inv_freq[i]`), reaching ~1e-4 relative error by
position 19 and ~0.58 relative (unusable) toward position 8192.

## Mechanism chosen

**Exact-widen-to-f64, reduce-in-f64, correctly-rounded-narrow-back**, NOT
Payne-Hanek. Justification for why the simpler route suffices here (and is
preferable — fewer new pinned constants, smaller attack surface for a
clean-room divergence):

- Payne-Hanek arbitrary-precision reduction exists to handle arguments where
  `x` itself is many orders of magnitude larger than 2π (e.g. `x ~ 1e30`),
  where even a 52-bit-mantissa f64 constant for 2π isn't enough bits of
  precision relative to `x` to keep the reduced remainder accurate. That is
  not this domain: the RoPE angle is `pos · inv_freq[i]`, `pos ∈ [0, 8192)`
  (`max_position_embeddings`, CIS2_SPEC_v0.2.md §2.2) and `inv_freq[i] ≤ 1.0`
  (§7.1: `inv_freq[i] = theta^(-2i/head_dim) ≤ 1` for `theta ≥ 1`, `i ≥ 0`).
  So `|x| < 8192` always, for every pinned model in scope. An f64 has 52
  mantissa bits; representing `x` up to `8192 = 2^13` leaves 52-13=39 bits of
  fractional precision, and the constant `2π` is itself representable to
  full f64 precision (2.2e-16 relative). Reducing modulo 2π in f64 for
  `|x| < 8192` therefore leaves a remainder accurate to a few ULP of f64,
  i.e. an absolute error on the order of `2π · 2^-52 ≈ 1.4e-15` — utterly
  negligible next to `f32`'s own ULP (~1.2e-7 relative, i.e. absolute error
  up to ~4e-7 for `r` near π). Narrowing that f64 remainder to f32 with a
  single correctly-rounded cast is therefore accurate to within 0.5 ULP of
  f32 — the f64 stage contributes no meaningful error of its own.
- Payne-Hanek's added complexity (multi-word fixed-point 2/π table, extended
  precision integer part extraction) buys nothing extra in a domain where
  the input never exceeds `2^13` and IEEE-754 f64 already carries 39 bits of
  margin past that. Using it here would be over-engineering relative to the
  actual finding and would introduce a large new pinned table purely to
  handle a case (`|x| ≫ 2^40`) that cannot occur in this spec's decode path
  (`pos < max_position_embeddings` is a hard architectural bound, not just
  an empirical one for the current pinned test vectors).

## Determinism justification (why this stays cross-ISA bit-identical)

Every operation in the new reduction is one of:
1. `x as f64` — exact widening cast, f32→f64, zero rounding error by
   construction (every f32 value is exactly representable in f64).
2. f64 `/`, `*`, `-` — IEEE-754-mandatory correctly-rounded operations
   (RNE), identical bits on any conformant ISA, same argument as
   CIS2_SPEC_v0.2.md §1.6 already makes for f32 `/`/`sqrt`.
3. f64 `.round()` — Rust's documented semantics are "round half away from
   zero", a pure function of the input's bit pattern with no
   ISA-dependent instruction selection when compiled without fast-math
   (same "no reassociation, no fast-math" constraint as §1.5); this is
   already relied on nowhere else in the spec as an f32 op but is being
   newly introduced here at f64 — flagged explicitly as new normative
   surface, see spec text below.
4. `r as f32` (f64→f32 narrowing) — single RNE rounding, IEEE-754-mandatory
   correctly-rounded narrowing per the standard's conversion rules.
5. **No FMA**: nothing above is a candidate for fused multiply-add (there is
   no `a*b+c` shape at all in the new reduction — it is `x/C`, `.round()`,
   then `x - k*C` as two separate ops, matching the existing `reduce_2pi`'s
   structure exactly, just staged in f64 instead of f32).

None of this depends on x87 80-bit intermediate precision (Rust/LLVM f64 on
x86-64 uses SSE2 scalar ops by default, never x87, for any target with SSE2
— which is every target this spec's compiler matrix already builds for);
aarch64 f64 arithmetic is likewise IEEE-754-binary64-conformant scalar FP.
FTZ/DAZ (§1.3) is pinned for the decode path already; f64 intermediate
values here never come remotely close to subnormal range (`x < 8192`,
`2π ≈ 6.28`), so FTZ/DAZ has no observable effect on this reduction either
way — stated for completeness, not because it matters here.

## What changes / what doesn't

- **Changes**: `reduce_2pi` (renamed `reduce_2pi_f64` in the pinned
  algorithm text) now stages through f64; `TWO_PI_HI`/`TWO_PI_LO` (the two
  f32 half-constants) are replaced by one `TWO_PI_F64` f64 constant.
  `table_digest()` changes (new constant, new byte layout) — **CIS2_REF and
  `table_digest` both change**; `inv_freq_table_digest` does **not** change
  (RoPE's `inv_freq` construction, §7.1, uses only `ln_pinned`/`exp_pinned`,
  untouched here).
- **Unchanged**: the degree-10 `SIN_COEF`/`COS_COEF` Taylor polynomials
  (§6.3's polynomial-on-reduced-argument step is unaffected — H2's root
  cause was reduction, not polynomial degree, and the spec's own text
  already said so); `exp_pinned`, `ln_pinned`, `rsqrt_cr`; RoPE's rotation
  math (§7.4); everything outside §6.3/§6.6.

## Gates (pre-registered, to be run after implementation)

1. **Determinism**: new `CIS2_REF` digest identical x86_64 vs aarch64 across
   the existing 20-cell compiler matrix (`e15d-compiler-invariance.yml`,
   dispatched on this branch with updated `TARGET_DIGEST`), plus
   `cis2_ref`/`verify2`/`verify3` cross-implementation digest match after a
   spec-text-only update to all three (verify2/verify3 updated by reading
   the new spec §6.3 text, not by copying `src/math.rs`, logged in each
   crate's own `CLEANROOM_LOG.md`).
2. **Accuracy**: max relative sin/cos error vs a high-precision (mpmath)
   oracle ≤ 2 ULP of correctly-rounded f32, for the RoPE angle domain
   `pos ∈ [0, 8192)`, `inv_freq[i]` the actual SmolLM2-135M table
   (`theta=100000`, `head_dim=64`). New harness:
   `scripts/h2m_accuracy_harness.py`.
3. **Oracle-correctness**: SmolLM2-135M and Qwen2.5-0.5B greedy-decode
   argmax matches PyTorch/`transformers` fp32 at 16/128/512 **and 2048**
   generated tokens; max logit relative diff reported at each length
   (extends `scripts/oracle_compare.py` / `scripts/oracle_compare_qwen.py`,
   which already parameterize `N_GEN`/`CIS2_ORACLE_N_GEN` from the H1
   512-token work).

## Risks / things that could still fail this gate

- If the true remaining error at pos≈8192 is still >2 ULP even after the
  f64 stage (e.g. because the *polynomial*, not just the reduction, has
  its own f32-accumulated error that only becomes visible once the
  reduction error stops dominating) — this pre-registration commits to
  reporting that honestly rather than silently narrowing the domain
  claim.
- `round()`'s "ties away from zero" semantics vs the old f32 `.round()`
  (same method name, now called on an f64) could in principle pick a
  different `k` right at a half-integer boundary of `x/2π` — this changes
  which branch of `[-π,π]` a given `x` reduces into but not the
  correctness of the reduction; not expected to be observable given `2π`
  is irrational relative to any f32 `x`'s exact value (no exact tie exists
  for a finite non-adversarial `x`), stated for completeness.
