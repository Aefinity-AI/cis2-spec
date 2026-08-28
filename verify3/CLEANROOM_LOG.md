# E15j clean-room log (C11 implementation, verify3/)

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
