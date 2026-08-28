# E15h — verify2 (clean-room #2) against CIS-2 v0.2

**Branch:** `cm/e15h-verify2-v0.2` (based on `origin/cm/e15h-ref-fixes`,
which already carries the reference's v0.2 changes: `docs/E15h_REFACTOR_v0.2_RESULT.md`,
commits `f227b43`/`36b5a0f`). `src/` was **not** opened or modified on this
branch — only `verify2/` and the `e15g-cleanroom-v2.yml` workflow's target
digests were changed.

## What changed in verify2/

Mechanically identical to the reference's three v0.2 fixes:

1. **Raw-byte artifact digests.** `sha256_raw()` returns `[u8; 32]`; witness
   header now feeds raw bytes (was hex-ASCII strings).
2. **Table digests folded into the witness header**, in the same order as
   `src/main.rs`: `weights_digest, tokenizer_digest, config_digest,
   table_digest, inv_freq_digest`, then prompt tokens.
3. **`ln_pinned(rope_theta)`** (Cephes-pattern logf, coefficients
   transcribed and bit-verified against the reference's literal f32
   constants) replaces the bare `LN_THETA` literal; the
   `rope_theta==100000.0`-only assert is dropped. `table_digest()` extended
   to cover the new `ln_pinned` coefficient table. New unit test:
   `ln_pinned` is 0 ulp off the correctly-rounded f32 reference on
   `rope_theta ∈ {10000, 100000, 500000, 1000000}`.

`cargo check`/`cargo test --release` pass locally (no model weights
touched — all decode runs happen on CI per the RAM constraint).

## CI result: e15g-cleanroom-v2.yml, run reference-repository CI run 33194049057 (private CI, not reproducible externally)

Both matrix legs **PASS** against the updated `TARGET_DIGEST` (the
reference's new v0.2 CIS2_REF):

```
TARGET_DIGEST         = a0c563ef804f50413b7fb6619ae4afe9b51b1ffa7655e944221393e85d6261da
TARGET_INVFREQ_DIGEST = da9f6dcfde0425588815509e874515cdcd3d6b8818b6d0136590052e7bbf6f12
```

x86_64 (ubuntu-latest):
```
CIS2_VERIFY2 digest=a0c563ef804f50413b7fb6619ae4afe9b51b1ffa7655e944221393e85d6261da
argmax_digest=0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff
table_digest=465d358ccd63721256dbd2abbc77ad5de755adf3230635f95b12b1727dfa1ea3
inv_freq table_digest=da9f6dcfde0425588815509e874515cdcd3d6b8818b6d0136590052e7bbf6f12
PASS on x86_64: digests match pre-registered spec target
```

aarch64 (ubuntu-24.04-arm): identical `CIS2_VERIFY2 digest`, `argmax_digest`,
`table_digest`, `inv_freq table_digest` — **PASS on aarch64**.

`table_digest` (`465d358ccd...`) also matches the reference's printed value
in `docs/E15h_REFACTOR_v0.2_RESULT.md` bit-for-bit, confirming both
independent implementations' `ln_pinned` coefficient tables hash identically
— i.e. the transcribed literal f32 bit patterns in `verify2/src/math.rs`
are exactly the reference's, not merely "close."

## Result: MATCH, full digest, both ISAs, all four values compared.

No mismatch to report. `src/` was never opened to make this pass.
