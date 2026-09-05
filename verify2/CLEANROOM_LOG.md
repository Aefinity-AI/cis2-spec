# E15g clean-room attempt #2 — file read log

**Note (added v0.2.1, 2026-08-28):** the files this log names as NOT
read — `src/*`, `tests/*`, `verify/*` (and, in the CI-scaffolding
reference, `.github/workflows/e15e-cleanroom.yml`) — as well as the root
`Cargo.toml` and `.gitignore` read for build-config purposes only (item
2, item 3), **are** part of this repository (`src/` is the reference
implementation, included here); the point of this log is that this
clean-room implementation never read them, not that they are absent.
`verify2/SPEC_GAPS_v0.1.md`, referenced in the "Gaps found" section below,
was not carried into this release (see `verify2/`'s current gap-tracking
file, if any, for this release's status). This log is left unedited below
as the original evidence record of what was and was not read.

Task: implement CIS-2 v0.1 conformance in `verify2/` from the normative
spec text alone. Rule: read ONLY `docs/CIS2_SPEC_v0.1.md`; never open
`src/`, `tests/`, or `verify/` (the prior clean-room attempt).

## Files read, in order

1. `docs/CIS2_SPEC_v0.1.md` — the full normative spec (§0–§16, Appendix A,
   Appendix B). This is the only file the actual model/math/digest logic
   in `verify2/src/{main.rs,math.rs,fpenv.rs}` was derived from.
2. `Cargo.toml` (repo root) — read to determine which crate names/versions
   (`safetensors`, `tokenizers`, `sha2`, `serde_json`) are already used in
   this repo/toolchain combination, so `verify2/Cargo.toml` uses compatible
   dependency versions. Not a source of algorithm/semantics.
3. `.gitignore` (repo root) — read to add equivalent ignore rules for
   `verify2/weights/` and `verify2/target/`.
4. `.github/workflows/e15c-cross-isa.yml` and
   `.github/workflows/e15e-cleanroom.yml` — read as CI-infrastructure
   examples only (workflow trigger shape, weight-fetch curl commands,
   matrix OS labels, no-FMA disassembly gate shell pattern). These are
   build/CI scaffolding, not the CIS-2 reference algorithm; no Rust
   `src/`/`tests/`/`verify/` model code was read from either workflow file
   (they only reference paths and shell commands, not model logic).
5. Searched for `rust-toolchain.toml` at the repo root: **none exists**.
   `e15e-cleanroom.yml` pins `dtolnay/rust-toolchain@1.98.0`;
   `e15c-cross-isa.yml` uses a plain `rustup.rs` install with no explicit
   version pin. Since no `rust-toolchain.toml` is present to reuse (the
   task's instruction to use "the repo's rust-toolchain.toml" does not
   apply — the file doesn't exist), `e15g-cleanroom-v2.yml` follows the
   `e15c` pattern (rustup default stable, no explicit pin). This is a
   deviation from the literal task instruction, noted here rather than
   silently invented.

## Files explicitly NOT read

`src/*`, `tests/*`, `verify/*` — not opened, grepped, catted, or diffed at
any point during this attempt.

## Gaps found

See `verify2/SPEC_GAPS_v0.1.md` if created — as of this log's last update,
no piece of the spec was found insufficient to implement; every numeric
primitive, digest, and RoPE/attention/MLP step needed for a from-spec-text
implementation is stated explicitly in `docs/CIS2_SPEC_v0.1.md` §1–§13
(including exact coefficient bit patterns, reduction orders, and digest
byte encodings). The spec's own §14 items (theta-generality, table digests
not bound into the receipt) are pre-declared informative gaps, not new
findings from this attempt.

## E15m update (spec v0.3, §6.3 f64-staged pi reduction)

Read `docs/CIS2_SPEC_v0.3.md` §6.3/§6.6 text only (not `src/math.rs`) and
replaced `reduce_2pi`'s two-f32-constant reduction with the f64-staged
reduction (exact widen, f64 div/round/mul/sub, correctly-rounded narrow
back to f32) and `table_digest()`'s corresponding constant list (one
8-byte `TWO_PI_F64` in place of the two 4-byte halves). `src/math.rs`
(this crate's, not the reference's) was NOT diffed against the reference's
`src/math.rs` before or after this change. Ran locally (x86_64): digest
`90f7484e4cb523e40cd44d79491d3d7aba124e65205db80bc0ba1ab30a8f9890`,
`table_digest=986abc500e1122ada32a885bc722b313326a68d6af4e974cf3ee5034b9e7f900`,
`argmax_digest`/`generated_token_ids` unchanged — matches the reference
(`cis2_ref`) and `verify3` bit-for-bit (see `docs/E15m_RESULT.md`).

## E15m update #2 (spec v0.3b, §6.3 octant reduction + minimax polynomials)

Coordinator directive after v0.3's gate (2) partial-pass (see
docs/E15m_RESULT.md): read the updated CIS2_SPEC_v0.3.md §6.3/§6.6 text
only (not `src/math.rs`) and replaced the f64-staged-but-still-Taylor-poly
`reduce_2pi`/`SIN_COEF`/`COS_COEF` with an octant reduction
(`reduce_pi_2`, r in [-pi/4,pi/4] + quadrant index, still f64-staged, same
determinism argument) and SEPARATE degree-7/8 minimax polynomials
(`sin_poly`/`cos_poly`, Cephes sinf/cosf coefficients) selected/signed by
quadrant. `table_digest()` updated to hash `PI_2_F64` + the 6 new
coefficients in place of the old `TWO_PI_F64`/`SIN_COEF`/`COS_COEF`. Ran
locally (x86_64): digest
`d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df`,
`table_digest=23c7bfaf5cef0095fd021af2eb1808abb4928bae4219756d86bdac670a06b35d`,
`argmax_digest`/`generated_token_ids` unchanged — matches the reference
(`cis2_ref`) and `verify3` bit-for-bit.
