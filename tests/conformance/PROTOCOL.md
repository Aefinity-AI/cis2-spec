# `cis2-conformance` stdin/stdout protocol

This documents the exact wire contract a third-party binary must implement
to be checked by `cis2-conformance` (source: `src/bin/cis2_conformance.rs`
in this repo — reading that source is NOT required to implement a
conforming binary; this document is the full contract). No knowledge of
this repo's Rust code, `src/math.rs`, or `src/main.rs` is required either;
only this document and `tests/conformance/README.md` (which documents the
per-`op` field layout of each vector file).

## Running the checker

```
cargo run --release --bin cis2-conformance -- /path/to/your-binary
```

or, against an already-built `cis2-conformance`:

```
cis2-conformance /path/to/your-binary
```

This runs **every** `tests/conformance/vectors/*.txt` / `*.expected` pair
against `your-binary`, invoking `your-binary` once per vector, and prints a
PASS/FAIL line per vector plus a final summary. Exit code is `0` if every
vector passed, `1` otherwise (including if `your-binary` fails to launch,
times out, or produces unparseable output on any vector).

## Per-vector invocation

For each `<name>.txt` file under `tests/conformance/vectors/`:

1. `cis2-conformance` spawns `your-binary` with **one argument**: the
   vector's `op=` value (e.g. `your-binary rmsnorm`, `your-binary
   attention_block`). This lets an implementation dispatch without having
   to inspect the input first, but is redundant with the `op=` line inside
   the input itself (both are provided; a conforming binary MUST accept
   either identically-valued source, or ignore argv and parse `op=` from
   stdin — both are conformant, since they are guaranteed to agree).
2. `cis2-conformance` writes the **entire raw byte content** of
   `<name>.txt` to `your-binary`'s stdin, unmodified (same `key=value`
   text format documented in `tests/conformance/README.md`), then closes
   stdin (EOF).
3. `your-binary` MUST:
   - Parse the `key=value` lines per `tests/conformance/README.md`'s
     general format and the specific field layout documented there for
     that vector's `op=` value.
   - Compute the primitive named by `op=`, per the cited `spec_ref=`
     section of `docs/CIS2_SPEC_v0.2.md`, exactly (bit-for-bit — no
     tolerance, no "close enough": every operation in the cited section
     must be reproduced in the exact order and association specified,
     including strict left-to-right reduction (§5.1/§5.3), no
     fused-multiply-add, and the pinned transcendental routes (§6)).
   - Write to stdout, before exiting, **exactly one line** of the form:
     ```
     out_sha256=<64 lowercase hex chars>
     ```
     computed as SHA-256 over the concatenation of each output element's
     4 raw IEEE-754 binary32 bytes, **little-endian**, in index order —
     the same convention `tests/conformance/README.md` and
     `src/math.rs::table_digest` use (`out_sha256` in each `.expected`
     file is exactly this digest).
   - Optionally also write a second line `out_bits=<comma-separated
     0x-prefixed 32-bit hex>` (same convention as `.expected`'s
     `out_bits`) — `cis2-conformance` will report it in diagnostic output
     on a mismatch if present, but only `out_sha256` is checked for
     pass/fail.
   - Exit with status `0` on success. Any nonzero exit status is treated
     as a FAIL for that vector regardless of stdout content.
4. `your-binary` MUST NOT read any file from disk, make any network call,
   or depend on any state outside its own stdin — every input the
   computation needs is in the `.txt` content on stdin. This is what makes
   the suite runnable against an arbitrary black-box binary with no shared
   filesystem layout assumptions.
5. `cis2-conformance` compares `your-binary`'s `out_sha256=` line (trimmed,
   case-insensitively) against the pinned `out_sha256=` field of
   `<name>.expected`. Match is PASS; anything else (missing line, mismatch,
   nonzero exit, timeout, unparseable stdout) is FAIL, with a diagnostic
   printed to `cis2-conformance`'s own stdout.

## Example: one full stdin blob

For `rmsnorm_v1`, `your-binary rmsnorm` receives on stdin exactly the byte
content of `tests/conformance/vectors/rmsnorm_v1.txt` (see that file), and
must write to stdout:

```
out_sha256=f712eda3e8b49d3d84639db584c134fe265f7994a21dede9f401d7be9bc5e60c
```

(optionally preceded or followed by an `out_bits=...` line), then exit 0.

## Timeouts

`cis2-conformance` allows each vector invocation up to 10 seconds
wall-clock before treating it as a FAIL (`timeout`). Every vector in this
suite is small (hand-sized inputs, no model weights), so a conforming
implementation should return in well under a second; the timeout exists
only to keep a hung or blocking candidate binary from stalling the whole
run.

## What this does NOT cover

This protocol checks individual normative primitives in isolation (one
`RMSNorm` call, one RoPE table, one `exp_pinned` evaluation, one attention
block, one matvec) — it is deliberately narrower than the full
end-to-end §13.1 witness digest in `EXPECTED_DIGESTS.md`, which requires
loading the actual SmolLM2-135M weights and running a full forward pass.
Passing every vector here is necessary but not sufficient evidence of
full-model conformance.
