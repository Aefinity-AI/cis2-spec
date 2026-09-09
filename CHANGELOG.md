# Changelog

This repository tracks the CIS-2 specification and its independent
clean-room verifiers. The reference implementation the spec was audited
against lives in a separate repository (see README "Scope"); this
changelog covers the spec text and verifiers published here.

## Errata against v0.3b

Numbered corrections to the published v0.3b text. **No erratum here moves a
pinned digest**; every value in section 13 is unchanged.

### E-1 (2026-09-09) --- section 14.5 was wrong: the RMSNorm multiply order *is* bit-level necessary

**What v0.3b said**, carried unchanged from v0.1:

> 14.5. **RMSNorm multiply order (section 8) is pinned but its bit-level
> necessity is unconfirmed.** `(x[i]*inv)*weight[i]` vs. `x[i]*(inv*weight[i])`
> are not provably identical for arbitrary fp32 operands under rounding, but no
> divergence between the two orders has actually been observed on either model
> tested.

**Why that is wrong.** A divergence had in fact been observed before v0.3b was
tagged --- E22's M09 mutation is exactly this reassociation, and it moved the
witness digest --- and section 14.5 was simply never updated. Measured since
(`docs/E25_RMSNORM_ASSOCIATION.md`): the two associations differ by one ULP on
**231,014 of the 667,584** RMSNorm elements a section 13.1 decode produces,
about 35%, and substituting the other association changes the witness digest
from `d82743059d...` to `570c0bbb0d...`. The Rust clean-room verifier and the C
reference implementation agree bit-for-bit on that non-conforming digest as
well as on the conforming one.

**What it should say:**

> 14.5. **RMSNorm multiply order (section 8) is pinned, and its bit-level
> necessity is measured.** `(x[i]*inv)*weight[i]` and `x[i]*(inv*weight[i])`
> differ by one ULP on about 35% of the operand triples an actual decode
> produces --- 231,014 of the 667,584 RMSNorm elements in the section 13.1
> vector. Substituting the other association changes the section 13.1 witness
> digest from `d82743059d...` to `570c0bbb0d...`. It does **not** change the
> argmax digest or the generated token ids for this vector, so the violation is
> invisible to any check that hashes only the model's outputs --- one of the
> reasons section 12.1 hashes the full logit vector. See
> `docs/E25_RMSNORM_ASSOCIATION.md`.

The corrected text is applied in place in `docs/CIS2_SPEC_v0.3b.md` with an
E-1 marker. Section 8 itself is unchanged; only the limitations note was wrong.

## Repository releases

Version numbers above name the *specification* document. The section
below records what the repository shipped alongside it.

### 2026-09-09 --- release `v0.3b` (first tagged release)

The v0.3b spec text is unchanged; every pinned digest in section 13 is
unchanged. What is new in the repository since the spec was frozen:

- **Op-level conformance vectors** (`tests/conformance/`): five pinned
  input/expected pairs -- `matvec_v1`, `rmsnorm_v1`, `rope_v1`,
  `exp_pinned_v1`, `attention_block_v1` -- with a written protocol
  (`PROTOCOL.md`), so an implementer can localize a divergence to one
  operation instead of bisecting a whole decode. Previously the only
  conformance surface was the end-to-end `CIS2_REF` digest: pass or fail,
  no diagnosis.
- **Dual-ISA CI** (`.github/workflows/verify.yml`): every job now runs
  natively on both `ubuntu-24.04` (x86_64) and `ubuntu-24.04-arm`
  (aarch64) and asserts `uname -m` matches, so the cross-ISA claim is
  re-checked by a third party on every push rather than asserted from a
  local log. Jobs: `reference`, `conformance`, `verify2`, and `verify3`
  built with both gcc and clang -- 10 jobs, all green on `main`.
- **A GPU leg** (`docs/GPU_RESULT.md`): a CUDA port written from the
  v0.3b spec text reproduces the normative witness digest
  `d82743059d...` bit-for-bit on an NVIDIA Tesla P100-PCIE-16GB (sm_60,
  CUDA 12.8), with a byte-identical per-step trace against the CPU
  reference. This is a first-party result, not an independent
  replication; the CUDA source is not published. See that document's
  Scope section for what it does and does not establish.
- **A Hugging Face dataset card** (`docs/HF_DATASET_CARD.md`) and
  `scripts/publish_hf.sh`, which assembles the conformance payload from
  the working tree and refuses to upload if the v0.3b digest is missing
  from either the card or `EXPECTED_DIGESTS.md`.

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

## v0.3 (2026-08-29, PARTIAL -- superseded by v0.3b)

Attempted to close CIS-2's one remaining stated technical limitation
(v0.2 section 6.7): `sin_pinned`/`cos_pinned`'s range reduction is
staged through f64 instead of two f32 half-constants, fixing
catastrophic-cancellation accuracy loss that grew with RoPE position (up
to 1104 ULP by position 19 in v0.2; ~0.58 relative, unusable, toward
position 8192).

**What changed:** section 6.3 (reduction only, not the degree-10 Taylor
polynomials); section 6.6 (`table_digest` now hashes one 8-byte
`TWO_PI_F64` constant in place of v0.2's two 4-byte halves).

**Net effect on test vectors:** `CIS2_REF` changed (`a0c563ef80...` ->
`90f7484e4c...`); `table_digest` changed (`465d358ccd...` ->
`986abc500e...`); `inv_freq_table_digest`, `argmax_digest`,
`generated_token_ids` unchanged from v0.2.

**PARTIAL result:** the reduction itself became accurate, but the fixed
degree-10 Taylor polynomial's own truncation error (worst near `|r| ~
pi` and at sin/cos zero crossings) remained dominant -- up to 2813 ULP,
missing the pre-registered <=2 ULP accuracy bar. Superseded by v0.3b.

## v0.3b (2026-08-29)

Closes CIS-2's one remaining stated technical limitation, meeting the
accuracy bar v0.3 missed. Refines the mechanism: section 6.3 now reduces
to an octant (`r` in `[-pi/4, pi/4]`, quadrant index `k mod 4`, still
f64-staged, same Cody-Waite pattern as v0.3 but mod `pi/2` instead of mod
`2*pi`), then evaluates `sin(r)`/`cos(r)` with separate degree-7/8
Cephes `sinf`/`cosf` minimax polynomials (not Taylor truncations),
selected and signed by quadrant.

**What changed:** section 6.3 (reduction period and polynomial, both);
section 6.6 (`table_digest` now hashes one 8-byte `PI_2_F64` constant
plus 6 pinned f32 minimax coefficients in place of v0.3's `TWO_PI_F64`
plus two 11-entry Taylor tables).

**What did not change:** `inv_freq_table_digest`, `argmax_digest`,
`generated_token_ids` -- unchanged from v0.1/v0.2/v0.3.

**Net effect on test vectors:** `CIS2_REF` changed (v0.3's
`90f7484e4c...` -> `d82743059d...`); `table_digest` changed (v0.3's
`986abc500e...` -> `23c7bfaf5c...`).

An accuracy harness (mpmath oracle, RoPE position domain up to
`max_position_embeddings=8192`) confirms max 1.5 ULP for both sin and
cos, meeting the pre-registered <=2 ULP bar. `verify2` (Rust) and
`verify3` (C) clean-rooms were both updated a second time from the
v0.3b spec text only (not from the reference implementation) and
reproduce `d82743059d...` bit-for-bit locally on x86_64; see each
crate's `CLEANROOM_LOG.md`. See `docs/CIS2_SPEC_v0.3b.md` (supersedes
`docs/CIS2_SPEC_v0.2.md` as the current normative document, which is
kept for history) and `docs/E15m_RESULT.md`/`docs/E15m_PREREG.md` for
full detail.

## v0.1 (2026-08-28)

First draft. Established the normative scope: fp32 transformer decode of
`HuggingFaceTB/SmolLM2-135M` on the pinned prompt, digest and receipt
format, floating-point environment (FTZ/DAZ, no FMA), reduction order,
transcendental polynomials, RoPE, RMSNorm, attention, MLP, and argmax.
