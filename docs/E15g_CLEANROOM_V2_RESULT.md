# E15g — clean-room attempt #2 result (spec-text-only, `verify2/`)

**Status: MATCH. Full bit-for-bit conformance on first CI run, no fixes needed.**

## Method

`verify2/` (crate `cis2-verify2`) was implemented reading **only**
`docs/CIS2_SPEC_v0.1.md` (body, Appendix A citations read as informative
provenance text only — not followed to `src/`, Appendix B). `src/`,
`tests/`, and `verify/` (the prior clean-room attempt, E15e) were never
opened, grepped, catted, or diffed. Full read log: `verify2/CLEANROOM_LOG.md`.
Infrastructure files (`Cargo.toml`, `.gitignore`, the two existing CI
workflow YAMLs) were read for build/CI scaffolding only, not for
algorithm content — logged in the same file. No `rust-toolchain.toml`
exists in this repo (checked); `e15g-cleanroom-v2.yml` follows the
`e15c-cross-isa.yml` pattern (plain rustup install, no pinned version)
since there was nothing to reuse.

## CI run

Run: reference-repository CI run 33192338105 (private CI, not reproducible externally)
(push-triggered on `cm/e15g-cleanroom-v2`, both matrix legs `completed
success`, first attempt — 0 of the allotted 3 fix attempts used).

| digest | spec target (§13.1) | x86_64 result | aarch64 result |
|---|---|---|---|
| `CIS2_REF` (witness) | `ba88708bf4159a4057d4eef727820ebc5ce1744ae334bfe0cc0776e03509dd0f` | **MATCH** | **MATCH** |
| `argmax_digest` | `0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff` | **MATCH** | **MATCH** |
| `table_digest` (§6.5, informative) | `0bf9257bc56c4588cb4003e1a105b60169e152222ec734811d0a58e66e1cdeca` | **MATCH** | **MATCH** |
| `inv_freq_table_digest` (§7.2) | `da9f6dcfde0425588815509e874515cdcd3d6b8818b6d0136590052e7bbf6f12` | **MATCH** | **MATCH** |
| `generated_token_ids` (16) | `[28, 665, 436, 253, 1838, 8180, 3365, 14176, 30, 2306, 4161, 281, 253, 2066, 2291, 351]` | **MATCH** | **MATCH** |

No-FMA gates: source grep for `.mul_add(` — PASS (0 hits) on both legs.
Disassembly grep for `vfmadd\|vfnmadd\|vfmsub\|vfnmsub\|fmadd\|fmsub` on
the release binary — PASS (0 hits) on both legs (also re-confirmed
locally on x86_64 via `objdump -d target/release/cis2-verify2` before
pushing).

`cargo test --release` (unit tests: `rsqrt(64.0) == 0.125` exact bits,
`dot_seq` order-sensitivity regression on `[1e8, 1.0, -1e8]` → `0.0`,
bf16 widening exactness) passed on both legs. Same-process
determinism check (§12.4, both runs inside `main()`) asserted internally
and did not trip (process would have panicked otherwise, and it didn't).

## First divergent quantity

**None.** There was no divergence to localize — all four digests and
all 16 generated token ids matched on both ISAs on the first attempt.

## Spec gaps found

**None found insufficient.** Every numeric primitive needed (bf16
widening, `dot_seq`/`sum_seq` reduction order, `rsqrt`, the pinned
`exp`/`sin`/`cos` polynomials and coefficient bit patterns, `ldexp_exact`,
SiLU, RoPE `inv_freq` construction restricted to `rope_theta=100000.0`,
RMSNorm's pinned multiply association, GQA causal attention, SwiGLU MLP,
argmax tie-breaking, and the exact three-digest/witness-chain byte
encoding including the hex-ASCII-not-raw-bytes artifact-hash detail)
was stated explicitly enough in `docs/CIS2_SPEC_v0.1.md` §1–§13 to
implement without consulting `src/`. No entry was added to a
`verify2/SPEC_GAPS_v0.1.md` because none was needed — the spec's own §14
pre-declared gaps (theta-generality, table digests not bound into
`CIS2_REF`) are documentation of known scope limits, not new findings
from this attempt.

One minor, non-blocking deviation from the task instructions: no
`rust-toolchain.toml` exists in this repo, so the CI workflow could not
"use the repo's rust-toolchain.toml" as instructed — it falls back to
the `e15c-cross-isa.yml` convention (plain `rustup.rs` install, floating
stable toolchain, no explicit version pin) instead. This did not affect
digest reproduction on this run but is worth noting if a future CI run's
default `stable` toolchain version drifts.

## Files read (see `verify2/CLEANROOM_LOG.md` for the full annotated list)

`docs/CIS2_SPEC_v0.1.md`; `Cargo.toml` (root); `.gitignore` (root);
`.github/workflows/e15c-cross-isa.yml`; `.github/workflows/e15e-cleanroom.yml`.
Never read: anything under `src/`, `tests/`, `verify/`.

## Conclusion

This is the strongest possible outcome for a spec-conformance test:
CIS-2 v0.1 as written was sufficient, on its own, for an independent
from-spec-text implementation to reproduce all pinned digests
bit-for-bit on both x86-64 and aarch64, first try. This corroborates
that the earlier clean-room gaps (`verify/SPEC_GAPS.md`, E15e) were
genuinely closed by the v0.1 spec revision, at least for this one
(model, prompt, decode-length) tuple.
