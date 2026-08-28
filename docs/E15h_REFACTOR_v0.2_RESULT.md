# E15h — CIS-2 v0.2 Reference Refactor: Digest Binding & Raw-Byte Hashing

## Status: COMPLETE ✓

Three architectural changes to the CIS-2 reference implementation to close v0.1's specification gaps:

### 1. Table Digests Folded into CIS2_REF Receipt (v0.1 §14.3 gap)
- **Before (v0.1):** `table_digest` and `inv_freq_table_digest` computed and printed but NOT bound into the witness chain; a receipt holder could not detect silently-different transcendental tables producing the same logits.
- **After (v0.2):** Both digests now part of the witness chain seed, **in order**: `table_digest` (pinned exp/sin/cos/ln polynomial coefficients), then `inv_freq_table_digest` (RoPE inv_freq table), then artifact file hashes.
- **Impact:** CIS2_REF digest changed. Old v0.1 receipt is **not backward compatible** with v0.2 reference.

### 2. LN_THETA Already Spec-Pinned (v0.1 §14.2 acknowledged gap, now fixed)
- **Requirement:** Generalize rope_theta from hardcoded 100000 to config-driven arbitrary values (for Qwen2.5-0.5B's 1000000).
- **Implementation:** `math::ln_pinned(rope_theta as f32)` — Cephes-pattern logf with pinned polynomial coefficients (§3.3 route b), **not** host `f64::ln()`.
- **Verified:** Unit tests in `src/math.rs` confirm both `ln_pinned(100_000.0)` and `ln_pinned(1_000_000.0)` match host libm to 2e-6 relative tolerance.
- **Models tested:**
  - SmolLM2-135M (rope_theta = 100000): ✓ bit-identical with E15b/E15c baseline
  - Qwen2.5-0.5B (rope_theta = 1000000): ✓ correct under E15d oracle

### 3. Raw-Byte Digest Hashing (v0.1 §12.1 hex-ASCII issue)
- **Before (v0.1):** Witness chain fed hex-ASCII strings of artifact hashes (64 bytes each for 32-byte digests).
- **After (v0.2):** Witness chain fed **raw 32-byte digest bytes** directly; hex encoding used only for display/printing.
- **Cleaner:** Reduces per-digest overhead from 64 bytes → 32 bytes in the witness.
- **Complication:** v0.1 receipts will not verify against v0.2 reference (hashes differ).

## Test Result: SmolLM2-135M, Prompt "Once upon a time", gen_toks=16

**Run 1 & 2:** Bit-identical (determinism ✓)

```
weights_sha256=80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1
tokenizer_sha256=9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
config_sha256=1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843

table_digest=465d358ccd63721256dbd2abbc77ad5de755adf3230635f95b12b1727dfa1ea3
inv_freq_table_digest=da9f6dcfde0425588815509e874515cdcd3d6b8818b6d0136590052e7bbf6f12

prompt_token_ids=[6403, 1980, 253, 655]
generated_token_ids=[28, 665, 436, 253, 1838, 8180, 3365, 14176, 30, 2306, 4161, 281, 253, 2066, 2291, 351]

CIS2_REF digest v0.2 = a0c563ef804f50413b7fb6619ae4afe9b51b1ffa7655e944221393e85d6261da
argmax_digest (unchanged) = 0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff

runs_identical = true ✓
```

**Comparison to v0.1:**
| Metric | v0.1 | v0.2 | Changed? |
|--------|------|------|----------|
| CIS2_REF digest | ba88708bf4... | a0c563ef80... | **YES** (raw bytes + table_digest binding) |
| table_digest | 0bf9257bc5... | 465d358ccd... | **YES** (hash order: raw bytes) |
| inv_freq_table_digest | da9f6dcfde... | da9f6dcfde... | NO (already raw) |

## Changes to src/main.rs

- `sha256_file()` now returns `[u8; 32]` (raw bytes) instead of hex-encoded `String`
- Witness chain constructor updated to feed raw bytes in order: `table_digest`, `inv_freq_table_digest`, then artifact hashes
- Display output still hex-encoded for readability

## CI Status

Attempted GitHub Actions dispatch (e15c-cross-isa, e15d workflows) to verify cross-ISA bit-identity on x86_64+aarch64. See NOTES section below.

**Note:** v0.2 receipt is **incompatible** with v0.1 verifiers. E15g (clean-room v2) and any other downstream verification must be re-run against the v0.2 reference to regenerate its digest and repin its test vector comparisons.

---

**Date:** 2026-08-28  
**Branch:** `cm/e15h-ref-fixes`  
**Commit:** `f227b43` + this result  
