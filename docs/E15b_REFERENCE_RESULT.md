# E15b milestone 1 — CIS-2 fp32 reference pass, SmolLM2-135M

Historical development note (from the reference repository's internal research
log).

## What was built

A **full Rust reference implementation** (not a torch/numpy oracle) of a
deterministic fp32 forward pass for `HuggingFaceTB/SmolLM2-135M`,
implementing the CIS-2 §3 normative contract from the E14a design memo
(from a separate internal operator-notes repository's design memo, not published here):

- `src/math.rs` — the normative kernel primitives: strict left-to-right
  sequential dot product / sum (§3.1), no FMA anywhere (§3.2), correctly-rounded
  rsqrt via `1.0/x.sqrt()` (§3.3 route (a)), a pinned polynomial fallback for
  `exp`/`sin`/`cos` (§3.3 route (b) — see "Transcendental route" below),
  exact bf16→fp32 widening via bit-shift (§3.5), and softmax with
  sequential max-scan + sequential-sum denominator.
- `src/denormal.rs` — pins FTZ+DAZ on x86_64 via direct MXCSR bits (§3.4),
  with a documented, unimplemented note on what aarch64's FPCR.FZ leg needs
  (E15c work, not this milestone).
- `src/main.rs` — loads `model.safetensors` + `tokenizer.json` +
  `config.json`, widens every bf16 weight tensor to fp32 (§3.5), runs a
  KV-cached (correctness-preserving; KV cache is a pure memoization of
  already-computed per-position K/V vectors, not a reduction-order change)
  greedy-decode forward pass, and emits the SHA-256 witness chain.

This is architecture-general Llama-family code (RMSNorm → RoPE → GQA
causal attention → SwiGLU MLP, pre-norm, 30 layers, tied embeddings) driven
entirely from `config.json`, not hardcoded to this checkpoint's numbers.

## Exact repro commands

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cd <repo root>

# weights (already present in weights/, gitignored — re-download if needed):
curl -sS -L -o weights/config.json     https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/config.json
curl -sS -L -o weights/tokenizer.json  https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/tokenizer.json
curl -sS -L -o weights/model.safetensors https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/model.safetensors

cargo test --release          # unit tests + tests/no_mul_add.rs grep gate
./scripts/no_fma_gate.sh      # §3.2 mechanical gate: grep + disassembly
nice -n 10 ./target/release/cis2_ref   # runs the reference TWICE internally,
                                        # prints both digests + CIS2_REF line
```

Preconditions observed: `free -m` showed ~1.5–1.7 GB available before the
run; peak `used` during/after the run stayed within the machine's 6.4 GB
total (see "Resource notes" below). Ran under `nice -n 10` per project
convention.

## Weight / artifact provenance

Downloaded directly from the public HF `resolve` URLs (this box was not
authenticated as `aefinityAIINC`; the plain `curl` path worked without
auth since `HuggingFaceTB/SmolLM2-135M` is public).

| file | sha256 |
|---|---|
| `model.safetensors` | `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1` |
| `config.json` | `1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843` |
| `tokenizer.json` | `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c` |

`config.json` confirms the memo's §4 architecture exactly: `hidden_size:
576`, `intermediate_size: 1536`, `num_hidden_layers: 30`,
`num_attention_heads: 9`, `num_key_value_heads: 3`, `hidden_act: silu`,
`rms_norm_eps: 1e-05`, `rope_theta: 100000`, `rope_interleaved: false`,
`tie_word_embeddings: true`, `torch_dtype: bfloat16`, `vocab_size: 49152`.
Header inspection of `model.safetensors` (272 tensors, all `BF16`)
confirms there is **no separate `lm_head.weight` tensor** — the checkpoint
really is tied, matching `tie_word_embeddings: true`; the reference
implementation reuses `model.embed_tokens.weight` for the LM head matmul.

## CIS2_REF result — two runs, identical

Prompt: `"Once upon a time"` → tokenizes (via the checkpoint's own
`tokenizer.json`) to `[6403, 1980, 253, 655]` (4 prompt tokens).

16 greedy-decoded token ids (both runs): `[28, 665, 436, 253, 1838, 8180,
3365, 14176, 30, 2306, 4161, 281, 253, 2066, 2291, 351]`.

```
run1_witness_digest=830d972dbfb0b5598015f33571d08623d2b05d8e239b8d0288562d9ac9786907
run2_witness_digest=830d972dbfb0b5598015f33571d08623d2b05d8e239b8d0288562d9ac9786907
run1_argmax_digest=0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff
run2_argmax_digest=0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff
runs_identical=true
CIS2_REF digest=830d972dbfb0b5598015f33571d08623d2b05d8e239b8d0288562d9ac9786907 prompt_toks=4 gen_toks=16 dtype=fp32
```

This was checked **two ways**:

1. **In-process**: `main()` runs the full decode twice sequentially in one
   process invocation (`run1`/`run2` in the log) — both witness and argmax
   digests match exactly.
2. **Cross-process**: the compiled binary was invoked as two entirely
   separate OS processes; the printed `CIS2_REF digest=...` line was
   identical byte-for-byte between the two invocations.

This satisfies E15b's preregistered pass bar from the memo §5: "op-level
digest identical across two sequential runs on the same host... before any
aarch64 work begins." **Cross-ISA is explicitly NOT tested here** — see
open questions.

The witness digest is a SHA-256 chain seeded with the three artifact
hashes above plus the prompt token ids, then updated at each of the 16
decode steps with (a) the full fp32 logit vector's bit pattern
(`f32::to_bits().to_le_bytes()` for all 49152 vocab entries, LE) and (b)
the chosen token id — directly mirroring CIS-1's witness/receipt chain
structure but over fp32 bits instead of exact integers.

## Confirming the pinned invariants actually took effect

- **FTZ/DAZ on x86_64**: `denormal::pin_ftz_daz()` sets MXCSR bits 15
  (FTZ) and 6 (DAZ) directly via `_mm_setcsr`/`_mm_getcsr`; `main()`
  asserts `mxcsr_ftz_daz_on()` before any decode-path arithmetic runs (see
  the `assert!` right after `denormal::pin_ftz_daz()` in `src/main.rs`).
  Unit test `denormal::tests::ftz_daz_pinned_and_denormal_input_flushed`
  (passing, see below) additionally verifies an adversarial case: the
  smallest positive normal times a tiny value flushes to `+0.0` (not a
  subnormal), and a literal subnormal bit pattern added to `0.0` also
  flushes to `+0.0` — both `black_box`-guarded so LLVM cannot optimize the
  hardware op away.
- **Left-to-right sequential accumulation (§3.1)**: `dot_seq`/`sum_seq` in
  `src/math.rs` are plain `for i in 0..n` loops with no chunking/tree
  structure; `math::tests::dot_seq_is_order_sensitive_by_design` is a
  regression test whose expected output (`0.0` for `[1e8, 1, -1e8]` dotted
  with `[1,1,1]`) is only correct under this exact left-to-right order (a
  pairwise-tree or any reassociation would give a different, nonzero
  answer) — this is a positive assertion that the pinned order, not just
  "some" order, is what runs.
- **bf16→fp32 = exact bit-shift (§3.5)**: `widen_bf16` is literally
  `f32::from_bits((bits as u32) << 16)`, and
  `math::tests::widen_bf16_exact` checks the exact bit pattern for `1.0`,
  `0.0`, and `-0.0`.
- **No FMA (§3.2)**: verified two ways, both currently passing —
  `tests/no_mul_add.rs` (grep gate: zero `.mul_add(` call sites in
  `src/`) and `scripts/no_fma_gate.sh` gate 2 (disassembly of the release
  binary: zero occurrences of `vfmadd|vfnmadd|vfmsub|vfnmsub|fmadd|fmsub`
  anywhere in `.text`, confirmed with `objdump -d`).
- **Transcendental route used**: **route (b), the pinned
  polynomial fallback** — NOT route (a) (adopting CORE-MATH/RLIBM), which
  the memo recommended trying first. No published Rust crate/binding for
  either CORE-MATH or RLIBM was found on crates.io in this sandbox (both
  are C libraries; porting either from scratch was out of scope for one
  milestone). `exp`/`sin`/`cos` in `src/math.rs` therefore use a
  Cephes-pattern range-reduction + fixed-degree-polynomial construction
  (algorithm *shape* only — coefficients independently pinned as literal
  f32 bit patterns, not copied binary), with a `table_digest()` SHA-256
  over every pinned coefficient, printed at runtime as
  `table_digest=0bf9257bc56c4588cb4003e1a105b60169e152222ec734811d0a58e66e1cdeca`.
  `rsqrt` (RMSNorm) **does** use route (a): `1.0f32 / x.sqrt()` composes
  two IEEE-754-mandatory correctly-rounded operations, so it needs no
  pinned table at all (this is the memo's explicit "cheap win" for rsqrt).

## Unit + gate test results

```
$ cargo test --release
running 6 tests
test math::tests::dot_seq_is_order_sensitive_by_design ... ok
test denormal::tests::ftz_daz_pinned_and_denormal_input_flushed ... ok
test math::tests::exp_matches_std_within_tolerance ... ok
test math::tests::rsqrt_matches_expected ... ok
test math::tests::sin_cos_match_std_within_tolerance ... ok
test math::tests::widen_bf16_exact ... ok
test result: ok. 6 passed; 0 failed

running 1 test (tests/no_mul_add.rs)
test no_mul_add_calls_in_src ... ok

$ ./scripts/no_fma_gate.sh
== gate 1: grep for .mul_add( in src/ ==
PASS: no .mul_add( call sites in src/
== gate 2: disassemble release binary, search for FMA instructions ==
PASS: no FMA instructions found in target/release/cis2_ref (0 matches for
vfmadd/vfnmadd/vfmsub/vfnmsub/fmadd/fmsub)
== both FMA gates PASS ==
```

`exp_matches_std_within_tolerance` / `sin_cos_match_std_within_tolerance`
compare the pinned route-(b) functions against the host's std-library
`f32::exp`/`sin`/`cos` (glibc libm) as a **sanity check only** — this
comparison is explicitly NOT the normative claim (host libm is exactly the
kind of non-pinned, platform-varying implementation CIS-2 exists to avoid);
it just confirms the pinned polynomial is a correct approximation of the
real function, not merely self-consistent.

## Resource notes

`free -m` before the run showed ~1.5–1.7 GB available; the run (loading
269 MB of bf16 weights, widening to ~540 MB of fp32, running two full
16-token decodes) completed without OOM, `used` peaking around 4.5–5.0 GB
with the rest of the box's existing load, `available` never hitting zero.
Run under `nice -n 10`. No wall-clock timing numbers are reported here
(Rule A: this program does not assert unreproduced performance figures).

## Honest open questions (what is NOT yet done)

1. **No cross-ISA testing at all.** This milestone's pass bar is
   same-machine, same-process-and-cross-process determinism only (memo
   §5's E14b bar). E15c (cross-ISA CI, `ubuntu-latest` vs
   `ubuntu-24.04-arm`) has not been attempted. The denormal-default
   divergence and transcendental-implementation-availability risks the
   memo flags as "live candidates for a first-attempt miss" (§5) are
   entirely unverified here.
2. **aarch64 FTZ/DAZ path is unimplemented, only documented.**
   `src/denormal.rs` has a `compile_error!` on non-x86_64 targets rather
   than a working FPCR.FZ implementation — this repo will not even build
   for aarch64 yet. The module doc explains exactly what E15c needs (set
   FPCR bit 24 via `msr fpcr, x0`), but it is unbuilt and unverified.
3. **Transcendental route is the fallback (b), not the memo's recommended
   route (a).** No CORE-MATH/RLIBM Rust binding was found; the pinned
   Cephes-pattern polynomial is a defensible, fully-specified,
   table-digested implementation, but it is "one arbitrary correct choice
   pinned by fiat" (memo's own characterization of route (b)) rather than
   the "converges by construction with any other correctly-rounded
   implementation" property route (a) would give. A different clean-room
   CIS-2 reference implementation that also chose route (b) would need to
   copy these exact coefficients (bound into `table_digest`) to agree —
   it would NOT automatically converge the way two independent
   correctly-rounded implementations would.
4. **RoPE's `inv_freq` table is computed via host `f64::powf` at
   model-load time**, not via a CIS-2-pinned transcendental. This runs
   once (32 values, `head_dim/2`), not per-decode-step, so it's a smaller
   surface than the per-step `exp`/`sin`/`cos` calls, but it is still a
   host-libm dependency that could differ across platforms/compiler
   versions and has not been pinned or digested. Should be replaced with
   `exp_pinned(-exponent * ln(theta))`-style construction (needs a pinned
   `ln`, which does not yet exist in `src/math.rs`) or hand-verified to be
   identical across the platforms E15c will target.
5. **No `cis-verify`-style independent from-spec re-derivation exists
   yet.** This is a single implementation producing a digest, not (as
   E14d envisions) a from-spec-text clean-room reimplementation that
   independently reproduces the same bits. The "two runs agree" bar
   proves self-consistency, not spec-completeness — a genuinely
   independent implementation might still diverge on an
   under-specified corner the spec prose hasn't nailed down yet (this is
   exactly the CIS-1 A31 errata pattern the memo warns about, §3.2).
6. **No accuracy/perplexity check against the original bf16/fp16 model
   output.** The pinned route-(b) `exp`/`sin`/`cos` were checked for
   numerical closeness to host libm on a handful of scalar test points
   (`math::tests::*`), not validated end-to-end against a reference
   HuggingFace `transformers` forward pass on the same prompt (no `torch`
   available in this sandbox — `pip`/`uv` install of `torch` was not
   attempted, out of scope for this milestone). The 16 generated token
   ids have not been cross-checked against any other implementation of
   this model; they are only self-consistent (deterministic on this box),
   not yet validated as "correct" relative to the reference PyTorch
   model's greedy output.
7. **KV-cache correctness is asserted by construction, not
   independently tested.** The KV cache stores exactly the post-RoPE
   K/V vectors that a from-scratch (no-cache) recomputation at each
   position would produce — reasoning about this from the code, not from
   a side-by-side no-cache-vs-cache digest comparison. A follow-up
   sanity check (run a no-cache full-recompute path and diff the digest
   against the cached path) would strengthen this claim; not done here.
8. **Pinned trig polynomial's range-reduction accuracy has only been
   checked for `|x| ≲ 14`** (`sin_cos_match_std_within_tolerance` tests
   `i in -20..=20, x = i*0.7`, i.e. up to ~14). RoPE angles in this run
   stay small (position < 20, `inv_freq` ≤ 1), so this is adequate for
   E15b's actual prompt, but the polynomial has not been stress-tested at
   larger magnitudes where the `TWO_PI_HI`/`TWO_PI_LO` two-part reduction
   could lose more precision.

## What milestone 2 (E15c, cross-ISA) needs

- Implement and build-gate the aarch64 FPCR.FZ leg in `src/denormal.rs`
  (currently a `compile_error!` stub).
- Stand up the `ubuntu-latest` / `ubuntu-24.04-arm` GitHub Actions matrix
  (direct reuse of alice-aegis's existing `arm-digest.yml` pattern, per
  the memo §5) and compare `CIS2_REF digest=` across both.
- Decide whether to invest in a real CORE-MATH/RLIBM port (route (a)) to
  remove open question 3's "pinned by fiat" caveat, now that a working
  route-(b) baseline exists to compare against.
- Add the `ln`-based pinned construction for `inv_freq` (open question 4).
