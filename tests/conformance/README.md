# CIS-2 third-party conformance suite (E21, slice 1: RMSNorm only)

This directory lets someone with an independent CIS-2 implementation
check individual normative primitives against pinned vectors, without
building or reading this repo's `src/`, `verify2/`, or `verify3/`. It is
narrower and more granular than the full end-to-end §13.1 witness digest
in `EXPECTED_DIGESTS.md` (which requires the SmolLM2-135M weights and a
full forward pass): each vector here exercises exactly one primitive with
a small, hand-sized input.

**Status: second slice.** RMSNorm (§8) and RoPE table (§7) vectors exist so
far. See "What's missing" at the bottom.

## Protocol

Each vector is a pair of files under `vectors/`:

- `<name>.txt` — the input. `key=value` lines. Vector-valued fields
  (suffixed `_bits`) are comma-separated `0x`-prefixed 32-bit hex
  strings, each the raw IEEE-754 binary32 bit pattern of one element (not
  a decimal literal — this avoids any ambiguity from a language's
  string-to-float parser rounding differently than another's). Scalar
  fields follow the same `_bits` convention.
- `<name>.expected` — the pinned output: `out_bits` (same comma-separated
  hex-bit-pattern convention, one element per output value, in index
  order) and `out_sha256` (lowercase hex SHA-256 of the concatenation of
  each output element's 4 raw bytes, **little-endian**, in index order —
  the same "feed_f32_le" convention `src/math.rs::table_digest` already
  uses in this repo).

A conforming implementation reproduces `out_bits` exactly (bit-for-bit,
not "close") and therefore also `out_sha256`. The digest is provided as a
convenience for a one-line pass/fail check; the bit patterns are the
normative artifact.

There is currently no single `cis2-conformance <your-binary>` CLI wrapper
yet (planned — see "What's missing"). For now, an implementer:

1. Parses `<name>.txt`.
2. Runs the algorithm cited by `spec_ref=` in that file, on that input,
   in fp32, following every rule in the cited section (reduction order,
   no FMA, association order, etc. — see `docs/CIS2_SPEC_v0.2.md` §§1–3
   for the general rules that apply throughout, not just the cited
   section).
3. Compares the output bit patterns against `<name>.expected`'s
   `out_bits`.

## Vectors

### `rmsnorm_v1` (§8 RMSNorm)

- Spec reference: `docs/CIS2_SPEC_v0.2.md` §8 (RMSNorm), which in turn
  depends on §5.3 (strict left-to-right sequential sum) and §6.1 (`rsqrt`
  = `1.0f32 / x.sqrt()`, both IEEE-754-mandatory correctly-rounded ops).
- `n=8` (deliberately small and hand-checkable; the real model uses
  `hidden=576`, but the algorithm is length-independent).
- Algorithm (verbatim from spec §8):
  ```
  for i in 0..n: sq[i] = x[i] * x[i]
  ss   = sum_seq(sq)              # §5.3, strict left-to-right
  mean = ss / (n as f32)
  inv  = rsqrt(mean + eps)        # §6.1: 1.0 / (mean+eps).sqrt()
  for i in 0..n:
      scaled = x[i] * inv
      out[i] = scaled * weight[i]     # THIS association, not x[i]*(inv*weight[i])
  ```
- `eps_bits=0x3727C5AC` is CIS-2's pinned `EPS_F32` (§2.3: `1.0e-5f64` cast
  to f32), used everywhere in the spec, not a value specific to this
  vector.
- Input `x` and `gamma` were chosen to be exactly representable in f32
  (halves/quarters of small integers), so there is no ambiguity in how
  `<name>.txt` itself was produced, only in how the RMSNorm arithmetic is
  carried out.

**How this vector was derived** (so it is provably correct, not just
asserted): `tests/conformance_rmsnorm.rs` includes `src/math.rs` by path
(`sum_seq`, `rsqrt_cr` — the same functions `src/main.rs::rmsnorm` calls
for the full-model §13.1 witness digest) and reimplements the §8 loop
above verbatim, computes the SHA-256 digest per the convention above, and
asserts both the bit patterns and the digest match
`vectors/rmsnorm_v1.expected`. Run it with:

```
cargo test --test conformance_rmsnorm
```

This was run once while authoring the vector (see `LOG.md`/commit
message for the exact `cargo test` output) to derive `rmsnorm_v1.expected`
from the reference implementation; the test now exists as a standing
self-check that the fixture stays correct across any future edit to
`src/math.rs`.

### `rope_v1` (§7 RoPE table: `inv_freq` + per-position cos/sin)

- Spec reference: `docs/CIS2_SPEC_v0.2.md` §7 (RoPE), specifically §7.1
  (`inv_freq` construction via the general, theta-general
  `ln_pinned(rope_theta)` / `exp_pinned` route — not v0.1's hardcoded
  literal), §7.2 (`inv_freq_table_digest`), and §7.3 (per-position
  cos/sin table). §7.4 (rotate-half application to a query/key head) is
  NOT covered by this vector — it is pure elementwise arithmetic given a
  cos/sin table and is deferred to a future attention-block vector (see
  "What's missing").
- `head_dim=8` (so `half = head_dim/2 = 4`; deliberately small and
  hand-checkable, same rationale as `rmsnorm_v1`'s `n=8` — the real model
  uses `head_dim=64`, but the construction is length-independent).
- `rope_theta=10000.0` (`0x461C4000`) — deliberately **not** one of the
  two model-pinned values (`100000.0` SmolLM2, `1000000.0` Qwen2.5-0.5B)
  cited in spec §7's "which `rope_theta` values are conformant" note, to
  exercise the theta-general `ln_pinned` path at an arbitrary value rather
  than only the two values the spec's own worked examples use.
- `positions=0,3` — two hand-sized sequence positions (§7.3's `pos`,
  cast to f32 exactly), to exercise both the degenerate `pos=0` case
  (`angle=0` for every `i`, so `cos_v` is all `1.0` and `sin_v` is all
  `0.0` bit-exactly — a useful sanity check that range reduction doesn't
  perturb the zero case) and a nonzero case.
- `out_bits` layout (20 values, index order): `inv_freq[0..4)`, then
  `cos_v[0..4)`/`sin_v[0..4)` at `pos=0`, then `cos_v[0..4)`/`sin_v[0..4)`
  at `pos=3` — see the header comment in `rope_v1.expected` for the exact
  slice boundaries.
- Two digests are pinned: `out_sha256` (this suite's general convention,
  full output, same as `rmsnorm_v1`) and `inv_freq_table_digest` (raw
  SHA-256 over only the `inv_freq` values, index order — the same
  construction as spec §7.2 and `src/main.rs`'s own
  `inv_freq_table_digest` variable, so this vector can also be checked
  directly against that specific spec definition).

**How this vector was derived**: `tests/conformance_rope.rs` includes
`src/math.rs` by path (`ln_pinned`, `exp_pinned`, `cos_pinned`,
`sin_pinned` — the same functions `src/main.rs` calls for the full-model
inv_freq/cos/sin construction) and reimplements §7.1/§7.2/§7.3 verbatim,
then asserts both the per-element bit patterns and both digests match
`vectors/rope_v1.expected`. Run it with:

```
cargo test --test conformance_rope
```

This was run once while authoring the vector to derive `rope_v1.expected`
from the reference implementation; the test now exists as a standing
self-check that the fixture stays correct across any future edit to
`src/math.rs`. (An `#[ignore]`d `print_bits_for_generation` test in the
same file is not part of the pinned suite; it exists only to
regenerate the fixture if the reference math ever changes — run with
`cargo test --test conformance_rope print_bits -- --ignored --nocapture`.)

## What's missing (next pass, E21 continues)

This is a second slice of a larger task. Not yet done:

- RoPE §7.4 rotation vector (apply a pinned cos/sin table to a hand-sized
  query/key head slice — pure elementwise arithmetic, not covered by
  `rope_v1` above).
- `exp_pinned`/softmax table vector (§3.3(b), §10).
- One full attention block vector (§9: GQA score/softmax/V-mix on a
  hand-sized `n_heads`/`head_dim`, not the full 576-wide model).
- One `matvec` vector (§5.2).
- The actual `cis2-conformance <your-binary>` CLI: a small wrapper that
  feeds a vector's `.txt` to a candidate binary's stdin and diffs its
  stdout against `.expected`, so an outside implementer doesn't have to
  hand-roll the parsing step above.
- A dual-ISA (x86_64/aarch64) CI job that runs this suite against the
  in-repo reference implementation on both ISAs, the same way
  `scripts/self_check.sh` / CI already does for the full §13.1 vector.
