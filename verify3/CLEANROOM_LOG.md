# E15j clean-room log (C11 implementation, verify3/)

**Note (added v0.2.1, 2026-08-28):** several files this log names below —
`docs/E15d_v0.2_DIGESTS.md` (read, item 2), and `src/`, `verify/`,
`verify2/`, `tests/`, `docs/CIS2_SPEC_v0.1.md`, `docs/E15b_*.md`,
`docs/E15e_*.md`, `docs/E15g_*.md`, `docs/E15h_*.md` (named as NOT read,
item 3), plus the `e15c-cross-isa.yml` CI workflow and `SPEC_GAPS_v0.1.md`
referenced elsewhere in this repository's history — **are** part of this
repository (`src/` is the reference implementation, included here) with
the sole exception of `docs/paper/*` and the private `e15c-cross-isa.yml`
workflow itself, which remain absent from this public release. Everything
else is named here only because it was the actual read/forbidden set this
clean-room pass was run against at the time; this log is left otherwise
unedited below as the original evidence record of what was and was not
read.

Files read by this clean-room implementer, in order, with justification.
Per task rules: ONLY `docs/CIS2_SPEC_v0.2.md` and `docs/E15d_v0.2_DIGESTS.md`
were read from this repo. No file under `src/`, `verify/`, `verify2/`,
`tests/`, any `docs/E15*.md` other than the two listed, or `docs/paper/*`
was opened.

1. `docs/CIS2_SPEC_v0.2.md` (full, 1238 lines) — the normative spec this
   implementation targets. Read in three chunks (1-400, 400-800, 800-1238).
2. `docs/E15d_v0.2_DIGESTS.md` (full, 69 lines) — expected digest values
   for the compiler-invariance / cross-ISA matrix (informative context,
   confirms `CIS2_REF = a0c563ef80...` and `inv_freq_table_digest =
   da9f6dcfde...` are the correct v0.2 pinned values to target).
3. `docs/CIS2_SPEC_v0.1.md`, `docs/E15b_*.md`, `docs/E15e_*.md`,
   `docs/E15g_*.md`, `docs/E15h_*.md`, `docs/paper/*`, `src/`, `verify/`,
   `verify2/`, `tests/` — NOT read (forbidden by task rules).
4. `config.json` present in the local `weights/` dir (from a prior
   `.gitignore`'d fetch) was listed by `ls`/`find` while checking what
   weight artifacts exist locally, but its *contents* were never opened by
   this implementer — architecture parameters used below are taken
   verbatim from spec §2.2's table, not from a local file read. No
   `tokenizer.json` or `model.safetensors` exists locally in this
   worktree; those are fetched by the CI workflow only.

## Notes on non-spec knowledge required

The spec (§3.1) says the tokenizer is "the `tokenizer.json` file shipped
with the checkpoint... loaded as a complete, self-contained HuggingFace
`tokenizers`-format tokenizer (BPE-family)" but does not itself define the
`tokenizers` library's on-disk JSON schema or the GPT2/ByteLevel
pre-tokenization regex/byte-to-unicode mapping. This implementer used
general pretrained knowledge of the public HuggingFace `tokenizers`
library's BPE JSON schema (`model.vocab`, `model.merges`,
`pre_tokenizer.type == "ByteLevel"`, the standard GPT-2 byte<->unicode
table, and the standard BPE merge-by-rank algorithm) to write
`verify3/tokenizer.c` from scratch — this is public, widely-documented
tokenizer-library format knowledge, not anything read from this repo's
forbidden files. The pre-tokenizer regex splitter implemented here is a
hand-written equivalent sufficient for ASCII text (adequate for the
pinned prompt `"Once upon a time"`, spec §3.2); it is not a full Unicode
`\p{L}`/`\p{N}` implementation. See `verify3/SPEC_GAPS_v0.2.md` if this
turns out to be insufficient for the pinned prompt in practice.

SHA-256: vendored a public-domain implementation (Brad Conte's
`crypto-algorithms`, sha256, public domain / no rights reserved) into
`verify3/sha256.c`/`verify3/sha256.h`, adapted for this codebase's naming.

JSON: hand-written minimal recursive-descent JSON reader
(`verify3/json.c`/`verify3/json.h`), not vendored from any third party —
written from scratch for this task, handles objects/arrays/strings
(with `\"`, `\\`, `\/`, `\n` etc. and `\uXXXX` escapes)/numbers/
true/false/null, sufficient to parse both `model.safetensors`' JSON
header and `tokenizer.json`.

## E15m update (spec v0.3, §6.3 f64-staged pi reduction)

Read `docs/CIS2_SPEC_v0.3.md` §6.3/§6.6 text only and rewrote
`two_pi_reduce()` in `verify3/mathpin.c` to stage the range reduction
through `double` (f64) per the spec text: exact widening cast, `round()`
of the f64 quotient, f64 multiply/subtract, single narrowing cast back to
`float`. Added `cis2_f64_from_bits()` to `mathpin.h` (same pattern as the
existing `cis2_f32_from_bits()`/`cis2_f32_bits()` helpers, `memcpy`-based
bit reinterpretation, no library call). Updated `cis2_feed_table_digest()`
to hash the 8 LE bytes of the new `TWO_PI_F64` constant in place of the
two old 4-byte halves. Updated `main.c`'s pinned expected-digest literals
(`PINNED_TABLE_DIGEST_HEX`, `PINNED_WITNESS_DIGEST_HEX`) to the new spec
v0.3 values. Built with the existing `-ffp-contract=off -fno-fast-math
-mno-fma` flags (unchanged Makefile) -- no new compiler flags needed for
the f64 stage. Ran locally (x86_64): `conformance=PASS`, digest
`90f7484e4cb523e40cd44d79491d3d7aba124e65205db80bc0ba1ab30a8f9890`,
matches `cis2_ref` and `verify2` bit-for-bit.

## E15m update #2 (spec v0.3b, §6.3 octant reduction + minimax polynomials)

Coordinator directive after v0.3's gate (2) partial-pass (see
docs/E15m_RESULT.md): read the updated CIS2_SPEC_v0.3.md §6.3/§6.6 text
only and replaced `two_pi_reduce()`/`SIN_COEF_BITS`/`COS_COEF_BITS` with
`reduce_pi_2()` (r in [-pi/4,pi/4] + `long long` quadrant index k, still
f64-staged) and separate `sin_poly()`/`cos_poly()` minimax polynomials
(Cephes sinf/cosf coefficients, pinned as `#define` hex literals), selected
and signed via a `quadrant()`/`switch` per the spec's quadrant identity
table. Updated `cis2_feed_table_digest()` and `main.c`'s pinned
`PINNED_TABLE_DIGEST_HEX`/`PINNED_WITNESS_DIGEST_HEX` literals to the new
v0.3b values. Ran locally (x86_64): `conformance=PASS`, digest
`d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df`,
matches `cis2_ref` and `verify2` bit-for-bit.
