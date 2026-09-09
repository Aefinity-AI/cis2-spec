# CIS-2 third-party conformance suite (E21: RMSNorm, RoPE table, exp_pinned, attention block, matvec + CLI protocol)

This directory lets someone with an independent CIS-2 implementation
check individual normative primitives against pinned vectors, without
building or reading this repo's `src/`, `verify2/`, or `verify3/`. It is
narrower and more granular than the full end-to-end §13.1 witness digest
in `EXPECTED_DIGESTS.md` (which requires the SmolLM2-135M weights and a
full forward pass): each vector here exercises exactly one primitive with
a small, hand-sized input.

**Status.** RMSNorm (§8), RoPE table (§7), exp_pinned (§6.2 pinned exp()
polynomial), one attention block (§9.2–§9.4), and one matvec (§5.2)
vector exist, plus a `cis2-conformance` CLI wrapper implementing the
stdin/stdout protocol documented in `PROTOCOL.md`. See "What's missing"
at the bottom for what's still outstanding.

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

A `cis2-conformance <your-binary>` CLI wrapper exists (`src/bin/cis2_conformance.rs`,
run via `cargo run --release --bin cis2-conformance -- /path/to/your-binary`)
that drives every vector below against an arbitrary black-box binary over
stdin/stdout and reports PASS/FAIL per vector — see `PROTOCOL.md` for the
exact wire contract. An implementer may use that wrapper directly, or
hand-roll the same steps against these files without it:

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

### `exp_pinned_v1` (§6.2 pinned exp(x) polynomial)

- Spec reference: `docs/CIS2_SPEC_v0.2.md` §6.2 (the pinned exp(x) route
  (b) fallback described normatively in `src/math.rs::exp_pinned`'s module
  doc comment) — the same function used by §7.1's RoPE `inv_freq`
  construction (already exercised indirectly by `rope_v1` above), §9's
  softmax, and SiLU's gate.
- `n=8` (deliberately small and hand-sized, same rationale as
  `rmsnorm_v1`'s `n=8` / `rope_v1`'s `head_dim=8`).
- Inputs span §6.2's documented accuracy domain `x in [-40,40]` (softmax
  post-max-sub args <= 0, SiLU gate args, RoPE inv_freq exponents): the two
  domain endpoints (`-40.0`, `40.0`), mid-range values on each side
  (`-10.0`, `-1.0`, `10.0`), the two special values `0.0` and `1.0`
  (`exp(0)=1` exactly is a useful bit-exact sanity check that range
  reduction doesn't perturb the zero case), and `0.5` for a non-integer,
  non-zero small positive value.
- `out_bits` layout: 8 values, index order matching `x_bits`.
- Only the general `out_sha256` convention is pinned here (no
  table-specific digest, unlike `rope_v1`'s `inv_freq_table_digest`) —
  `src/math.rs::table_digest()` already covers the exp/sin/cos/ln
  coefficient tables as a whole (see `EXPECTED_DIGESTS.md`); this vector
  is about the *function's output* on specific inputs, not the
  coefficients themselves.

**How this vector was derived**: `tests/conformance_exp_pinned.rs`
includes `src/math.rs` by path (`exp_pinned` — the same function
`src/main.rs`/`src/math.rs::silu_pinned`/`src/math.rs::softmax_seq` call)
and evaluates it directly on each `x_bits` input, computes the SHA-256
digest per the convention above, and asserts both the bit patterns and
the digest match `vectors/exp_pinned_v1.expected`. Run it with:

```
cargo test --test conformance_exp_pinned
```

This was run once while authoring the vector (see `LOG.md`/commit
message for the exact `cargo test` output) to derive
`exp_pinned_v1.expected` from the reference implementation; the test now
exists as a standing self-check that the fixture stays correct across any
future edit to `src/math.rs`. (An `#[ignore]`d `print_bits_for_generation`
test in the same file is not part of the pinned suite; it exists only to
regenerate the fixture if the reference math ever changes — run with
`cargo test --test conformance_exp_pinned print_bits -- --ignored --nocapture`.)

### `attention_block_v1` (§9.2 Score, §9.3 Softmax, §9.4 V-mix)

- Spec reference: `docs/CIS2_SPEC_v0.2.md` §9.2 (score: `dot_seq(q_head,
  k_j) * rsqrt(head_dim)`, the multiply by scale applied *after* the dot,
  as a separate op), §9.3 (softmax, in place), §9.4 (V-mix: for each
  output dim, a strict left-to-right accumulation of `scores[j] *
  v[j][d]`). RoPE (§7, already covered by `rope_v1` above) is assumed
  already applied to the `q_head`/`k_bits` inputs, per §9.1's ordering
  (RoPE happens before scoring) — this vector starts from already-rotated
  q/k, as the spec's own §9.2 step does.
- `head_dim=8`, `seq_len=3` (deliberately small and hand-sized, same
  rationale as the other vectors above; the real model uses `head_dim=64`
  and a much longer causal cache, but the algorithm is length-independent).
  Every cached position `j = 0..=pos` (`pos = seq_len - 1`) is attended
  (causal, current step).
- GQA's `kv_head = qh / group` head-to-KV-head mapping (§9's opening
  paragraph) is a pure indexing detail on top of this same per-head
  arithmetic, not additional numeric behavior, so this vector covers
  exactly one query head against its already-selected KV cache and does
  not separately exercise the mapping.
- `out_bits` layout: `head_dim=8` values, `out_head[0..head_dim)`, index
  order.

**How this vector was derived**: `tests/conformance_attention_block.rs`
includes `src/math.rs` by path (`dot_seq`, `rsqrt_cr`, `softmax_seq` — the
same functions `src/main.rs`'s per-layer attention loop calls) and
reimplements the §9.2/§9.3/§9.4 pipeline above verbatim, computes the
SHA-256 digest per the convention above, and asserts both the bit
patterns and the digest match `vectors/attention_block_v1.expected`. Run
it with:

```
cargo test --test conformance_attention_block
```

This was run once while authoring the vector to derive
`attention_block_v1.expected` from the reference implementation; the test
now exists as a standing self-check that the fixture stays correct across
any future edit to `src/math.rs`. (An `#[ignore]`d
`print_bits_for_generation` test in the same file is not part of the
pinned suite; it exists only to regenerate the fixture if the reference
math ever changes — run with `cargo test --test conformance_attention_block
print_bits -- --ignored --nocapture`.)

### `matvec_v1` (§5.2 Matvec)

- Spec reference: `docs/CIS2_SPEC_v0.2.md` §5.2 (matvec: `y[o] =
  dot_seq(w[o,:], x)` for each output row `o`), which in turn depends on
  §5.1 (strict left-to-right sequential dot product).
- `out_features=3`, `in_features=3` (deliberately small and hand-sized,
  same rationale as the other vectors above).
- Row 0 of `w` is §5.1's own worked order-sensitivity example verbatim
  (`[1e8, 1.0, -1e8]` dotted against `x = [1,1,1]`), which must give
  exactly `0.0_f32` (the `1.0` term is lost to rounding against the `1e8`
  partial sum) — a conforming implementation MUST reproduce this exact
  cancellation, not just "close". Rows 1 and 2 use small
  exactly-representable values.
- `out_bits` layout: 3 values, `y[0]` (the §5.1 example, must be exactly
  `0x00000000`), `y[1]`, `y[2]`.

**How this vector was derived**: `tests/conformance_matvec.rs` includes
`src/math.rs` by path (`dot_seq` — the same function `src/main.rs`'s
per-layer projections/MLP/lm_head calls use) and reimplements §5.2's
per-row matvec verbatim, computes the SHA-256 digest per the convention
above, and asserts both the bit patterns and the digest match
`vectors/matvec_v1.expected`. Run it with:

```
cargo test --test conformance_matvec
```

This was run once while authoring the vector to derive
`matvec_v1.expected` from the reference implementation; the test now
exists as a standing self-check that the fixture stays correct across any
future edit to `src/math.rs`. (An `#[ignore]`d `print_bits_for_generation`
test in the same file is not part of the pinned suite; it exists only to
regenerate the fixture if the reference math ever changes — run with
`cargo test --test conformance_matvec print_bits -- --ignored --nocapture`.)

## The `cis2-conformance` CLI

`src/bin/cis2_conformance.rs` implements the stdin/stdout protocol
documented in `PROTOCOL.md`: given a path to a third-party binary, it
runs every vector above against that binary (one invocation per vector,
vector's `op=` value as argv, `.txt` content on stdin, `out_sha256=...`
expected on stdout) and reports PASS/FAIL per vector plus a summary. See
`PROTOCOL.md` for the full wire contract. `src/bin/cis2_conformance_reference_candidate.rs`
is a reference candidate binary (reusing `src/math.rs` directly, unlike a
real third-party candidate) that exists only to smoke-test
`cis2-conformance` itself against a known-good implementation; it is not
part of the advertised protocol surface.

```
cargo run --release --bin cis2-conformance -- /path/to/your-binary
```

## What's missing (next pass, E21 continues)

Not yet done:

- RoPE §7.4 rotation vector (apply a pinned cos/sin table to a hand-sized
  query/key head slice — pure elementwise arithmetic, not covered by
  `rope_v1` above; `attention_block_v1` above assumes RoPE already
  applied rather than exercising the rotation step itself).
- A dual-ISA (x86_64/aarch64) CI job that runs this suite against the
  in-repo reference implementation on both ISAs, the same way
  `scripts/self_check.sh` / CI already does for the full §13.1 vector
  (explicitly out of scope for this pass).
