# v0.4 open items

Tracking issues for prose-only open items called out in
`docs/CIS2_SPEC_v0.3b.md`.

- [#16](https://github.com/Aefinity-AI/cis2-spec/issues/16) — §15's
  conformance bar has carried a single (model, prompt, length) test vector
  since v0.1, unlike CIS-1's three-tier op-goldens/selftest/token-digest
  split; a future version should factor out op-level unit vectors
  (`exp_pinned`/`sin_pinned`/`cos_pinned`/`ln_pinned`/`rsqrt`/`dot_seq`) as
  their own tier.

## Note on the §6.7 `max_position_embeddings = 8192` limitation

v0.2 §6.7 originally flagged an open accuracy concern for RoPE angles
approaching `max_position_embeddings = 8192` (up to ~0.58 relative error,
unusable, per `docs/H2_TRANSCENDENTAL_RIGOR.md`). As of **v0.3b** this is
explicitly closed, not open: octant range reduction plus separate
degree-7/8 minimax polynomials reach **≤2 ULP** vs. a correctly-rounded
fp32 oracle across the full RoPE position domain (`pos ∈ [0, 8192)`) — see
`docs/CIS2_SPEC_v0.3b.md` lines 12-24 and 1011-1022 ("This closes the 'if
decode length were ever extended toward `max_position_embeddings = 8192`'
caveat v0.2 §6.7 flagged for future work"). No tracking issue was filed for
this item since the spec text itself already records it as resolved; filing
one would misstate current status. If a reviewer finds a residual gap here,
open a fresh issue citing the specific unresolved claim.
