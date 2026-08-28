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
