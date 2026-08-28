# Changelog

This repository tracks the CIS-2 specification and its independent
clean-room verifiers. The reference implementation the spec was audited
against lives in a separate repository (see README "Scope"); this
changelog covers the spec text and verifiers published here.

## v0.2 (2026-08-28)

Three normative changes relative to v0.1, all affecting the pinned test
vector's `CIS2_REF` witness digest:

1. **Table digests folded into `CIS2_REF`.** The pinned transcendental
   coefficient-table digest (§6.6) and the RoPE `inv_freq` table digest
   (§7.2) are now inputs to the witness chain (§12.1 items 4-5), seeded
   after the three artifact hashes (weights, tokenizer, config).
2. **RoPE `inv_freq` is theta-general.** A new pinned `ln_pinned`
   polynomial (§6.5) replaces a bare literal that only worked for one
   `rope_theta` value, so the construction now supports any
   `rope_theta` read from `config.json`.
3. **Witness header uses raw 32-byte digests, not hex-ASCII strings.**
   Closes an encoding ambiguity in v0.1's witness chain (§12.1 items 1-3).

**§12.1 witness-order correction (post-v0.2, folded into this document):**
an earlier draft of §12.1 stated the witness chain's item order as
table-digest-then-inv_freq-then-artifact-hashes; the reference
implementation this spec was audited against actually feeds the three
artifact hashes first, then `table_digest`, then `inv_freq_table_digest`.
This mismatch was caught by an independent clean-room implementation built
strictly from the spec text: it reproduced every other pinned value
(`table_digest`, `inv_freq_table_digest`, `argmax_digest`, all 16
generated token ids) bit-for-bit but computed a different `CIS2_REF`
purely because it faithfully followed the incorrect prose. §12.1 in this
document states the corrected order; see docs/CIS2_SPEC_v0.2.md §12.1's
inline note and §16 for detail. No pinned digest value changed as a result
of this correction — only the prose describing the order was wrong.

Net effect on test vectors vs. v0.1: `CIS2_REF` changed
(`ba88708bf4...` -> `a0c563ef80...`); `table_digest` changed
(new `ln` coefficients folded in); `inv_freq_table_digest`, `argmax_digest`,
and `generated_token_ids` unchanged.

## v0.2.1 (2026-08-28)

Doc-only, no digest change. §3.1 rewritten from a one-paragraph pointer
into a self-contained byte-level BPE tokenizer specification (byte<->unicode
map, GPT-2 ByteLevel pre-tokenizer regex plus digit pre-split, rank-ordered
merge algorithm, special-token handling, and a worked example). No §13
test-vector value changed.

## v0.1 (2026-08-28)

First draft. Established the normative scope: fp32 transformer decode of
`HuggingFaceTB/SmolLM2-135M` on the pinned prompt, digest and receipt
format, floating-point environment (FTZ/DAZ, no FMA), reduction order,
transcendental polynomials, RoPE, RMSNorm, attention, MLP, and argmax.
