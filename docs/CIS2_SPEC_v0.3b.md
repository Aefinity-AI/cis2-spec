# CIS-2 — Canonical Floating-Point Semantics for fp32 Transformer Inference, v0.3b

**Status: DRAFT, not frozen.** Private research (Aefinity-AI/cis2-fp). HELD —
not for publication without Justin's explicit decision (2026-08-28 publish
policy). This document is normative for the scope stated below: a
conforming, from-spec-text-only clean-room implementation MUST reproduce
`CIS2_REF` (§12) and the `inv_freq table_digest` (§7.2) bit-for-bit on the
pinned test vectors of §13, on both x86-64 and aarch64.

This is v0.3b of the spec, drafted as v0.1 (`docs/CIS2_SPEC_v0.1.md`), then
v0.2 (`docs/CIS2_SPEC_v0.2.md`), then an intermediate v0.3 (superseded,
kept only in §16's changelog and §6.3/§6.7's history notes). **v0.3b closes
CIS-2's one remaining stated technical limitation** (v0.2 §6.7,
`docs/H2_TRANSCENDENTAL_RIGOR.md`): `sin_pinned`/`cos_pinned` now use an
octant range reduction (`r ∈ [-π/4, π/4]`, quadrant index, §6.3),
f64-staged same as v0.3's attempt, PLUS separate degree-7/8 minimax
polynomials (replacing the degree-10 Taylor series both v0.2 and v0.3
used). v0.3 alone (f64-staged reduction, unchanged Taylor polynomial) fixed
the reduction's catastrophic cancellation but left the *fixed Taylor
polynomial's own truncation error* dominant (up to ~2813 ULP even after the
reduction fix) — a PARTIAL result, documented in `docs/E15m_RESULT.md`.
v0.3b's minimax-polynomial-on-a-smaller-octant approach reaches **≤2 ULP**
vs. a correctly-rounded fp32 oracle across the full RoPE position domain
(`pos ∈ [0, 8192)`), closing the bar v0.3 missed. This changes **only**
§6.3 and §6.6 (`table_digest`, which now hashes one f64 octant-reduction
constant plus 6 minimax coefficients in place of the prior reduction
constant(s) and Taylor tables); RoPE's `inv_freq` construction (§7.1,
`ln_pinned`/`exp_pinned`) and `inv_freq_table_digest` (§7.2) are
**unaffected** by anything in this task. `CIS2_REF` and `table_digest`
**both changed at each step** (v0.2→v0.3→v0.3b; §16 has the full changelog
and all digest values); `inv_freq_table_digest` does **not** change at any
step. Full design rationale, determinism justification, and honest
gate-by-gate results (including v0.3's partial miss): `docs/E15m_PREREG.md`
(pre-registered before implementation) and `docs/E15m_RESULT.md`
(post-implementation gate results for both v0.3 and v0.3b). This
document's body is, as with v0.1/v0.2, written to be implementable
**without** access to `src/` — Appendix A cites exact `file:line`s only so
a reviewer can audit that this spec did not silently diverge from what the
reference actually computes.

**What conformance buys:** bit-identical fp32 decode of
`HuggingFaceTB/SmolLM2-135M` given the pinned prompt (§13), reproducible by
an independent from-spec-text implementation, on any ISA that supports
IEEE-754 binary32 arithmetic with FTZ/DAZ control (x86-64 MXCSR, aarch64
FPCR.FZ). Unlike v0.1, this version's RoPE construction (§7) is
**theta-general**: it is defined for any `rope_theta` value read from
`config.json`, not just `100000.0`, and has been exercised against a second
model family (`Qwen/Qwen2.5-0.5B`, `rope_theta = 1000000`) under an
independent oracle (`docs/E15d_bc_RESULT.md`). This is still deliberately
narrower than CIS-1 (§1 of `docs/CIS-1_SPEC_v1.0.md` in the public `Aefinity-AI/alice-aegis` repository, https://github.com/Aefinity-AI/alice-aegis):
floating-point addition is not associative, so this spec does not claim "any
reduction order is safe" — it claims exactly one pinned reduction order, one
pinned transcendental route (now including a pinned general `ln`), and one
pinned digest encoding are collectively sufficient for bit-identity, and
states each of them exactly.

---

## 0. Scope and non-goals (normative)

This spec's primary pinned tuple is unchanged from v0.1: `HuggingFaceTB/SmolLM2-135M`, prompt
`"Once upon a time"`, 16 greedy-decoded tokens, fp32 compute, bf16-on-disk
weights. §7's RoPE construction is now general (any `rope_theta`), so a
second (model, prompt, decode-length) tuple — `Qwen/Qwen2.5-0.5B`,
`rope_theta = 1_000_000` — is evidence-of-correctness (`docs/E15d_bc_RESULT.md`,
`docs/E25_RMSNORM_ASSOCIATION.md` §4) but is **not** itself a pinned §13 test
vector in this version; only the SmolLM2-135M tuple's digests are normative
test vectors here. §14.8 (erratum E-3) records a further consequence: §3 pins
one exact `tokenizer.json` shape, which that second checkpoint does not have,
so a conforming implementation cannot drive it end-to-end at all — work on it
is confined to §4–§11 with the prompt token ids supplied from outside §3. It does **not**
claim:

- Correct rounding of the pinned transcendental polynomials. Accuracy
  itself is no longer a gap: §14.1 (erratum E-4) records an exhaustive
  measurement of `exp`, `ln` and `silu` over **every** f32 bit pattern, and
  §14.4 (erratum E-2) the same for `sin` and `cos` over every f32 argument
  below `2^31`. What is not claimed is that any of them is *correctly
  rounded*, and §14.1(a) records one deliberate, normative divergence from
  mathematical `exp`: §6.2's `x > 88.0` guard clips 94,743 arguments whose
  true `exp` is finite and representable.
- Cross-framework agreement (vs. PyTorch/`transformers`) as a conformance
  requirement — oracle comparisons (§13.3, §14.6, `docs/E15b_m1p5_CORRECTNESS.md`,
  `docs/E15d_bc_RESULT.md`, `docs/E28_LAYER_ORACLE.md`) are evidence of
  correctness, not a conformance requirement; CIS-2 conformance is defined
  relative to this document's own bits, not to any third-party framework's
  output. That evidence now covers every intermediate activation of both §0
  models and not only the stack's output (§14.6, erratum E-5), but ~1e-5
  relative agreement with `transformers` is what it shows and all it could
  show: bit-exactness is claimed between conforming implementations of this
  document, never against a third-party framework.
- Seeded/temperature sampling — decode is greedy-only (argmax every step),
  matching CIS-1's own non-goal (CIS-1 §10).
- Anything about `verify/` or `cis2-verify2` (independently-written
  clean-room implementations built by *not* reading `src/`, used only to
  find gaps in earlier drafts of this document — evidence this spec was
  tested against, not itself normative). A clean-room re-verification of
  this exact v0.2 document against `cis2-verify2` is tracked separately
  (`docs/E15h_verify2_v0.2` lineage) and is not required for this document's
  own normative status.

## 1. Floating-point environment (normative)

**Unchanged from v0.1.** Reproduced verbatim (renumbered where v0.1 section
numbers shifted below, otherwise identical text and identical pinned
values):

1.1. **Format**: IEEE-754 binary32 (`f32`) for all decode-path compute.
Weights are stored on disk as `bfloat16` and widened to `f32` per §4.2
before any arithmetic.

1.2. **Rounding mode**: round-to-nearest-even (RNE), the IEEE-754 default,
for every basic operation (`+`, `-`, `*`, `/`, `sqrt`). No other rounding
mode is set at any point in the decode path. Nothing in this spec changes
the FPU/SIMD rounding-mode control bits from their IEEE-754 default.

1.3. **FTZ/DAZ MUST be pinned on, on every ISA, before any decode-path fp32
arithmetic runs:**
   - x86-64: MXCSR bit 15 (FTZ) and bit 6 (DAZ) both set to 1, via direct
     read-modify-write of the MXCSR register (not the deprecated
     `_MM_SET_*` wrapper macros). A conforming implementation MUST assert
     (readback, not just "we called the setter") that both bits are 1
     before proceeding.
   - aarch64: FPCR bit 24 (FZ) set to 1, via `mrs`/`msr fpcr`. aarch64 has
     no separate DAZ control; FZ alone flushes both denormal inputs and
     denormal outputs for scalar and Advanced SIMD binary32/binary64
     arithmetic. FPCR bit 19 (FZ16, half-precision flush-to-zero) is left
     at its architectural default (this reference never uses fp16) — do
     not set it. A conforming implementation MUST assert (readback) that
     FZ is 1 before proceeding.
   - Any other ISA: undefined by this version of the spec; a conforming
     implementation MUST refuse to run (hard compile-time or run-time
     error) rather than silently proceed without an equivalent control.
   - **Adversarial self-test (MUST run and pass before decode)**: with
     FTZ/DAZ pinned, `f32::MIN_POSITIVE (2^-126) * 1.0e-10` MUST equal
     exactly `+0.0` (not a subnormal), and the smallest positive subnormal
     bit pattern `0x00000001` added to `+0.0` MUST also equal exactly
     `+0.0`. (This must be checked with the operands passed through an
     optimization barrier — e.g. Rust's `std::hint::black_box` — so the
     compiler cannot fold the arithmetic away and mask a broken pin.)

   **ERRATUM E-11 (2026-09-09).** The first half of that self-test, as
   published, **cannot fail**. `f32::MIN_POSITIVE * 1.0e-10` is
   `1.18e-48`, which is below *half* the smallest positive subnormal
   (`7.0e-46`), so it rounds to `+0.0` under RNE whether or not FTZ is set.
   Measured directly, in a build with the pin deliberately gutted and proven
   gutted (`is_pinned() == false`): the probe still reads `0x00000000`. The
   multiplier `1.0e-10` is therefore **REPLACED by `0.5`**:
   `f32::MIN_POSITIVE * 0.5` is `5.88e-39`, a genuine subnormal, which reads
   `0x00000000` when FTZ is pinned and `0x00400000` when it is not. The DAZ
   half of the self-test (`0x00000001 + 0.0`) always did discriminate, and the
   readback assertion above always did enforce the pin, so no digest anywhere
   in this specification changes and no implementation whose pin is correct is
   affected --- the corrected probe newly fails exactly those implementations
   whose FTZ was in fact broken, which is what the self-test exists to catch.
   See docs/E38_REACH_SWEEP.md §5.

1.4. **No FMA contraction, anywhere, in the reference.** Every
multiply-then-add in this spec is defined as **two separate, separately
RNE-rounded operations**: compute the product, round to `f32`; then add,
round to `f32` again. A single fused `a*b+c` with one rounding step is
**non-conforming** even though it may be "more accurate" — it produces
different bits. In Rust terms: never call `f32::mul_add`; write `a*b + c`
as two statements/expressions, which LLVM does not auto-fuse absent an
explicit fast-math flag Rust does not provide. A conforming implementation
in any language MUST mechanically verify (e.g., disassemble the release
binary and grep for `vfmadd*`/`vfnmadd*`/`vfmsub*`/`vfnmsub*`/`fmadd`/
`fmsub` on the target ISA) that the emitted machine code contains zero
fused multiply-add instructions on the decode path. This has now been
mechanically verified across 20 compiler configurations (opt-level ×
target-cpu × ISA) with a reproducing pre-registered digest — see §13.4,
new in v0.2.

1.5. **No fast-math, no reassociation.** The compiler MUST NOT be given
any flag that licenses reassociating floating-point expressions, assuming
no NaN/Inf, or substituting approximate reciprocal/rsqrt hardware
instructions (e.g. `-ffast-math`, `-Ofast`, `-freciprocal-math`). This
spec's reduction orders (§5) are only bit-determining if the compiler
computes exactly the sequence of operations stated, in the stated order. This
is a prohibition on *licensing* reassociation, not a prohibition on
optimization: §13.5 measures that `--release -C target-cpu=native` emits
8-wide AVX2 for the elementwise work and leaves every §5 reduction scalar,
with every intermediate activation bit-identical to a non-vectorized
build. An optimizer denied fast-math can only act where acting does not
move the bits.

1.6. **Division and sqrt**: ordinary IEEE-754 `/` and `sqrt` (both
mandatory-correctly-rounded operations under IEEE-754), never a
reciprocal-approximation instruction (x86 `rcpps`/`rsqrtps`, ARM
`frecpe`/`frsqrte`) and never a library "fast inverse sqrt" trick. Any
conformant IEEE-754 binary32 `sqrt`/`/` implementation on any ISA produces
identical bits for identical inputs, by the standard itself — this is the
one part of the arithmetic this spec does not need to additionally pin.

## 2. Model artifact (normative)

**Unchanged from v0.1** (verbatim, all pinned values identical):

2.1. **Source**: Hugging Face repo `HuggingFaceTB/SmolLM2-135M`, files
fetched from the public `resolve/main/` URLs (no authentication required,
repo is public). No git revision/commit hash is pinned beyond content —
the artifact is identified **only** by the sha256 hashes below, which is
strictly stronger than a mutable branch ref:

| file | sha256 |
|---|---|
| `model.safetensors` | `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1` |
| `config.json` | `1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843` |
| `tokenizer.json` | `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c` |

A conforming implementation MUST verify all three sha256 hashes before use
and MUST fail loudly (not silently proceed) on mismatch.

2.2. **Architecture parameters**, taken from `config.json` (a conforming
implementation MUST read these from the file, not hardcode them, but their
pinned values for this artifact are, for the reader's convenience):

| field | value |
|---|---|
| `hidden_size` | 576 |
| `intermediate_size` | 1536 |
| `num_hidden_layers` | 30 |
| `num_attention_heads` | 9 |
| `num_key_value_heads` | 3 |
| `hidden_act` | `silu` |
| `rms_norm_eps` | `1e-05` (f64 in JSON; §2.3 pins its f32 cast) |
| `rope_theta` | `100000` (`100000.0`) |
| `rope_interleaved` | `false` |
| `max_position_embeddings` | 8192 |
| `tie_word_embeddings` | `true` |
| `torch_dtype` | `bfloat16` |
| `vocab_size` | 49152 |

Derived: `head_dim = hidden_size / num_attention_heads = 576/9 = 64`
(exact integer division); `group = num_attention_heads / num_key_value_heads
= 9/3 = 3` (GQA group size); `half = head_dim/2 = 32`.

2.3. **`rms_norm_eps` f32 cast**: the JSON value `1e-05` is parsed as an
IEEE-754 `f64` (`1.0e-5`), then cast to `f32` via a single RNE rounding
(Rust `as f32`, equivalent to any IEEE-754-conformant `f64`→`f32`
narrowing conversion). Pinned bit pattern:

```
EPS_F32 = 0x3727C5AC   (== (1.0e-5f64) as f32)
```

2.4. **`rope_theta` restriction — REMOVED in v0.2** (v0.1's §2.4 required
`rope_theta == 100000.0` exactly and refused to run otherwise; §7's
theta-general construction removes this restriction, closing v0.1 §14.1 —
see §7 and §16).

2.5. **Tensor names and shapes** (row-major, `[out_features, in_features]`
for every linear layer, matching PyTorch's `nn.Linear.weight` layout — a
matvec is `y[o] = Σ_i w[o,i]·x[i]`, §5.1):

```
model.embed_tokens.weight                              [vocab, hidden]
model.layers.{i}.input_layernorm.weight                [hidden]                for i in 0..30
model.layers.{i}.post_attention_layernorm.weight       [hidden]
model.layers.{i}.self_attn.q_proj.weight               [hidden, hidden]
model.layers.{i}.self_attn.k_proj.weight               [n_kv_heads*head_dim, hidden]
model.layers.{i}.self_attn.v_proj.weight               [n_kv_heads*head_dim, hidden]
model.layers.{i}.self_attn.o_proj.weight               [hidden, hidden]
model.layers.{i}.mlp.gate_proj.weight                  [inter, hidden]
model.layers.{i}.mlp.up_proj.weight                    [inter, hidden]
model.layers.{i}.mlp.down_proj.weight                  [hidden, inter]
model.norm.weight                                      [hidden]
```

There is **no** separate `lm_head.weight` tensor in this checkpoint
(confirmed by safetensors header inspection: 272 tensors total, all
`BF16`, no `lm_head.*` name present) — this matches
`tie_word_embeddings: true`. The LM head matmul (§11) reuses
`model.embed_tokens.weight` directly as the `[vocab, hidden]` output
projection matrix; no separate weight is loaded or derived for it.

Every tensor's on-disk dtype is `BF16` (2 bytes/element, little-endian
16-bit pattern per element, raw `safetensors` payload bytes taken two at a
time as `u16::from_le_bytes`). Every tensor MUST be widened per §4.2
before use; no tensor is used in bf16 form at compute time.

**Informative, v0.2**: a second model family, `Qwen/Qwen2.5-0.5B`
(different `rope_theta`, GQA shape, `attention_bias=true` on q/k/v
projections, optionally untied `lm_head.weight`), has been run through the
same reference code path unmodified beyond config-driven parameters
(`docs/E15d_bc_RESULT.md`). This is evidence the reference's architecture
handling generalizes; it is not a second pinned §13 test vector in this
version.

## 3. Tokenizer, prompt, and decode protocol (normative)

3.1. **Tokenizer.** **v0.2.1 (H4, this section): fully re-specified,
closing H3 BLOCKER-2** (`docs/H3_SPEC_AUDIT.md`; carried forward from
`verify3/SPEC_GAPS_v0.2.md` items 1–2). v0.1/v0.2's text ("loaded as a
complete, self-contained HuggingFace `tokenizers`-format tokenizer
(BPE-family; the exact merge/vocab table is entirely contained in this one
file — no external vocab/merges files, no additional special-token config
beyond what `tokenizer.json` itself specifies)") is retained but is no
longer the full normative surface: it is necessary but not sufficient, since
it does not define the BPE algorithm, the pretokenizer, the byte-level
mapping, or the merge tie-break rule. Those are pinned below.

3.1.1. **Artifact.** The `tokenizer.json` file shipped with the checkpoint,
bound by content hash: `sha256(tokenizer.json) =
9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c` (§2.1's
table, repeated here for locality). A conforming implementation MUST verify
this hash before use. This one file is self-contained: it embeds the full
vocab (49152 entries), the full ordered merge-rank list (48900 pairs), the
pre-tokenizer config, and the 17 special/added tokens (§3.1.6) — no external
vocab.json/merges.txt/tokenizer_config.json is consulted.

3.1.2. **Reference library binding (normative fallback).** The reference
implementation calls the Rust `tokenizers` crate, version **0.23.1**
(`Cargo.lock`; upstream source
`https://crates.io/crates/tokenizers/0.23.1`), specifically
`tokenizers::Tokenizer::from_file(&tokenizer_path)` then
`.encode(prompt, false)` (`src/main.rs:487-488,536`). Per this spec's
preference for algorithm over library citation: §3.1.3–3.1.6 below give the
byte-level BPE algorithm in full, transcribed from that crate's source
(`src/pre_tokenizers/byte_level.rs`, `src/pre_tokenizers/digits.rs`,
`src/models/bpe/word.rs`, `src/models/bpe/model.rs`, all at tag `v0.23.1`)
and independently verified in this audit to reproduce §3.3's pinned
token ids using the Python `tokenizers` binding (same crate, same version
family) loading the actual downloaded `tokenizer.json`. If any future
implementer finds an algorithmic edge case this section under-specifies,
`tokenizers` crate v0.23.1's published source is the normative
tie-breaker — this citation, not just the informal "HuggingFace
`tokenizers`-format" phrase, is part of the normative surface (the same
pattern this spec already uses for `sha256` as an external cited standard,
§2.1).

3.1.3. **Pipeline, in order:** normalizer (none — `tokenizer.json`'s
`normalizer` field is `null`) → pre-tokenizer (§3.1.4, splits the input
string into a sequence of "words") → per-word byte-level BPE encode
(§3.1.5) → no post-processor (`tokenizer.json`'s `post_processor` field is
`null`; this is what makes `add_special_tokens=false`, §3.3, produce zero
inserted tokens rather than a template).

3.1.4. **Pre-tokenizer.** `tokenizer.json`'s `pre_tokenizer` field is
`{"type": "Sequence", "pretokenizers": [
{"type": "Digits", "individual_digits": true},
{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true,
"use_regex": true}]}`. Applied in list order to the raw Unicode input
string:
  a. **Digits** (`individual_digits=true`): split the string at every
     maximal run of Unicode-numeric characters (`char::is_numeric`),
     *isolating* each individual digit as its own one-character segment
     (contrast with `individual_digits=false`, which would keep a digit
     run together — not used here). Non-digit segments pass through
     unsplit. (No digits occur in the pinned prompt `"Once upon a time"`,
     so this stage is a no-op on the §13.1 test vector but MUST still be
     applied for other inputs.)
  b. **ByteLevel** (`add_prefix_space=false`, `use_regex=true`): for each
     segment from (a), do NOT prepend a space (this checkpoint's config
     differs from the GPT-2 default of `add_prefix_space=true`); then
     split it further using the literal GPT-2 pre-tokenizer regex,
     applied with Unicode property classes, isolated-delimiter semantics
     (each match becomes its own segment, unlike the ordinary
     split/no-delimiter mode):
     ```
     's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
     ```
     (verbatim from `tokenizers` crate `src/pre_tokenizers/byte_level.rs`,
     itself citing `https://github.com/openai/gpt-2/blob/master/src/encoder.py#L98`).
     Then, for every resulting segment, remap **every UTF-8 byte** of that
     segment's text through the byte→unicode table of §3.1.5.a, producing
     a string of the same byte-count length (each input byte becomes
     exactly one output Unicode codepoint). This byte-remapped string is
     the final "word" handed to the BPE model (§3.1.5).

3.1.5. **Byte-level BPE model** (`tokenizer.json`'s `model` field: `type =
"BPE"`, `dropout = null`, `unk_token = null`, `continuing_subword_prefix =
null`, `end_of_word_suffix = null`, `fuse_unk = false`,
`byte_fallback = false`, `ignore_merges = false`). `dropout = null` means
the merge process below is fully deterministic (no probabilistic merge
skipping); `unk_token = null` and `byte_fallback = false` are moot because
the byte-level remap of 3.1.4.b guarantees every input byte already maps to
some single-character token that is a base entry in `vocab` (step a below),
so the "no matching vocab entry" branch is never taken for any input.
  a. **Byte→unicode map construction** (verbatim from
     `tokenizers` crate `bytes_char()`, itself following
     `https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9`):
     starting from the empty list, let `bs` = the 188 byte values in the
     three ranges `0x21..=0x7E` (`!`..`~`), `0xA1..=0xAC`, `0xAE..=0xFF`,
     each mapped to itself as a Unicode codepoint (`char::from_u32(b as
     u32)`); then, in ascending byte order `0..=255`, for every byte `b`
     NOT already in `bs`, append `b` to `bs` and map it to codepoint
     `256 + n` where `n` is a counter starting at 0 and incremented after
     each such assignment. This produces a total bijection over all 256
     byte values (188 map to themselves as printable-ASCII/Latin-1
     codepoints, the remaining 68 — control chars, space, DEL, and the
     `0x7F..=0xA0`/`0xAD` gap — map to codepoints `256..323`, e.g. byte
     `0x20` (space) maps to U+0120 `Ġ`, byte `0x0A` (newline) maps to
     U+010A `Ċ`). Decode uses the inverse map; both directions are
     computed once from this same construction, never independently
     hand-tuned per direction.
  b. **Initial symbol sequence.** For each byte-remapped "word" string
     from §3.1.4.b: split it into individual Unicode characters (one
     character = one original input byte, by construction of 3.1.4.b/3.1.5.a);
     look each single character up in `vocab` (the base single-byte
     vocabulary entries, ids 18–273 in this tokenizer's vocab, are exactly
     the 256 codepoints of 3.1.5.a, so this lookup always succeeds — see
     the `unk`/`byte_fallback` note above); the resulting list of
     `(vocab_id, byte_length=1)` pairs, in original left-to-right order, is
     the word's initial symbol sequence.
  c. **Merge loop (deterministic, rank-ordered, leftmost-tie-break).**
     `tokenizer.json`'s `model.merges` is an ordered list of 48900 symbol
     pairs; its list position IS the pair's merge rank (rank 0 = highest
     priority, applied first; rank 48899 = lowest). Given the initial
     symbol sequence from (b):
     i.   Build a priority queue seeded with every adjacent pair in the
          current symbol sequence that appears in `model.merges`, each
          keyed `(rank, position)`.
     ii.  Repeatedly pop the queue entry with the **lowest rank**; if two
          queued entries have equal rank (impossible here since merge
          ranks are a total order over distinct pairs, but stated for
          completeness as the crate's own tie-break), the entry with the
          **lower (leftmost) position** wins.
     iii. Before applying a popped entry, re-validate it still describes
          an adjacent, unmerged pair at that position (earlier merges may
          have invalidated it); if stale, discard and continue.
     iv.  Apply the merge: replace the two symbols with one symbol whose
          vocab id is `model.merges[rank]`'s resulting token id (from
          `tokenizer.json`'s corresponding `model.vocab` entry for the
          concatenated string) and whose byte-length is the sum of the
          two merged symbols'; enqueue any newly-adjacent mergeable pairs
          this creates (with the previous and/or next symbol) at their
          own rank.
     v.   Repeat ii–iv until the queue is empty. The final symbol sequence
          (in order) is that word's token id sequence.
  d. **Concatenation.** The prompt's full token id sequence is the
     concatenation, in original left-to-right order, of every word's token
     id sequence from (c), across all words produced by §3.1.4.

3.1.6. **Special/added tokens.** `tokenizer.json`'s `added_tokens` array
pins 17 special tokens, ids 0–16 (`<|endoftext|>`, `<|im_start|>`,
`<|im_end|>`, `<repo_name>`, `<reponame>`, `<file_sep>`, `<filename>`,
`<gh_stars>`, `<issue_start>`, `<issue_comment>`, `<issue_closed>`,
`<jupyter_start>`, `<jupyter_text>`, `<jupyter_code>`, `<jupyter_output>`,
`<empty_output>`), each with `special: true`. Per §3.3,
`add_special_tokens = false` means **none** of these are inserted for the
pinned prompt — no BOS (`<|endoftext|>`, id 0, is this tokenizer's only
BOS/EOS-like token and is never emitted for this prompt), no EOS, no
chat-template tokens. This is why the pinned `prompt_token_ids` (§3.3) has
exactly 4 entries, not 5 or more.

3.1.7. **Worked example — self-check before running the model.** Applying
§3.1.3–3.1.6 to the pinned prompt `"Once upon a time"` (§3.2):
  - No digits (§3.1.4.a is a no-op).
  - ByteLevel regex (§3.1.4.b) with `add_prefix_space=false` splits the
    ASCII string into 4 words: `"Once"`, `" upon"`, `" a"`, `" time"`
    (each of the latter three carries its leading space per the ` ?\p{L}+`
    alternative). Byte-remapping (§3.1.5.a) maps each literal space byte
    `0x20` to `Ġ` (U+0120) and leaves all other ASCII letters unchanged
    (they are in the `0x21..=0x7E` self-mapped range), giving the 4
    byte-remapped words `"Once"`, `"Ġupon"`, `"Ġa"`, `"Ġtime"`.
  - Per-word BPE merge (§3.1.5.b–d) on this tokenizer's `vocab`/`merges`
    reduces each word to exactly **one** token each (each whole word is
    itself a `vocab` entry reachable by the merge sequence — a property of
    this particular checkpoint's trained merge table for these four common
    words, not a general guarantee): `"Once"` → id `6403`, `"Ġupon"` → id
    `1980`, `"Ġa"` → id `253`, `"Ġtime"` → id `655`.
  - Concatenation (§3.1.5.d) with no special tokens (§3.1.6, §3.3) gives
    ```
    prompt_token_ids = [6403, 1980, 253, 655]
    ```
    — bit-for-bit the pinned value already stated in §3.3 and reproduced
    in §13.1. An implementer can check their own tokenizer stage against
    this worked example (word segmentation, byte remap, and final ids)
    *before* running any model math, isolating tokenizer bugs from
    numeric-core bugs.

3.1.8. **Cross-check performed for this section (informative, not itself
part of the normative text).** The exact `tokenizer.json` cited by §3.1.1's
hash was fetched from `https://huggingface.co/HuggingFaceTB/SmolLM2-135M/
resolve/main/tokenizer.json`; its sha256 was confirmed to equal
`9ca9acddb6...` (§3.1.1) byte-for-byte; loading it with the Python
`tokenizers` library (`Tokenizer.from_file(...).encode("Once upon a time",
add_special_tokens=False)`) reproduced `[6403, 1980, 253, 655]` exactly,
confirming both this section's transcription of the algorithm and its
worked example (§3.1.7) against a live run of the actual pinned artifact —
not merely against source-code reading.

3.2. **Prompt**: the literal ASCII string `Once upon a time` (no
leading/trailing whitespace beyond what is written here, no chat template
applied, no system prompt, no BOS/EOS token added or implied by any
wrapper). This is raw next-token completion, not chat-formatted.

3.3. **BOS/special-token handling**: tokenization MUST be performed with
`add_special_tokens = false` (i.e. the tokenizer's raw BPE encode of the
prompt string only — no BOS, no EOS, no any other special token
prepended/appended). Pinned result:

```
prompt_token_ids = [6403, 1980, 253, 655]     (4 tokens, u32)
```

3.4. **Decode protocol**: greedy only (argmax every step, §11.2), no
sampling, no temperature, no top-k/top-p. Exactly **16** tokens are
generated after the 4-token prompt, unconditionally — **EOS is not
checked and does not stop generation early** (matching CIS-1's own Tier-3
"EOS ignored" precedent, CIS-1 spec §8). Position indices are
`0, 1, 2, 3` for the 4 prompt tokens (prefill) and `4, 5, ..., 19` for the
16 generated tokens, assigned strictly in generation order (prompt
positions first, generated positions immediately following, no gaps, no
re-indexing).

3.5. **KV-cache equivalence (informative, not itself a conformance
requirement)**: the reference recomputes attention incrementally with a
KV cache that stores exactly the post-RoPE K/V vectors a from-scratch
full-recompute-per-step forward pass would produce (pure memoization, not
a reduction-order change). §14.5 (v0.1 numbering) independently confirms
the cached and oracle (full-recompute, no cache) paths agree. A conforming
implementation MAY use a KV cache or MAY recompute from scratch each
step; both are conformant iff they produce bit-identical logits — this
spec pins the *math*, not the caching strategy.

## 4. bf16 → fp32 widening (normative)

**Unchanged from v0.1** (verbatim):

4.1. **Formula**: exact, lossless bit-shift, not a rounding conversion.
For a bf16 value with raw 16-bit pattern `b` (as loaded via
`u16::from_le_bytes` on the 2 little-endian bytes of that element in the
safetensors payload):

```
f32_bits = (b as u32) << 16
widened  = f32::from_bits(f32_bits)
```

This is exact because bf16's sign(1)/exponent(8)/mantissa(7) layout is
identical to fp32's top 16 bits with the low 16 mantissa bits defined as
zero — bf16's exponent field has the same width and bias as fp32's, so
there is no exponent range issue and no rounding decision to make. This
MUST be implemented as the literal bit operation above, not as a call
through any "convert" library routine that might round or normalize
differently.

4.2. Every tensor listed in §2.5 is widened element-wise via §4.1 at load
time before any arithmetic touches it. No tensor is used in bf16 form.

## 5. Numeric reduction primitives (normative)

**Unchanged from v0.1** (verbatim):

5.1. **Sequential dot product.** For vectors `a`, `b` of equal length `n`:

```
acc = 0.0_f32
for i in 0..n:
    p   = a[i] * b[i]     # separate multiply, RNE-rounded to f32
    acc = acc + p          # separate add, RNE-rounded to f32
return acc
```

Strictly left-to-right, index order `0, 1, ..., n-1`. No pairwise tree, no
chunking, no reordering by magnitude. This is the **only** conforming
reduction order for every dot product and every "sum of products" loop in
this spec (matvec rows §5.2, attention score dot §9.2, V-mix §9.4).
`dot_seq` is order-sensitive by design: dotting `[1e8, 1.0, -1e8]` against
`[1.0, 1.0, 1.0]` in this order gives exactly `0.0_f32` (the `1.0` term is
lost to rounding against the `1e8` partial sum) — any other order gives a
different, nonzero answer; a conforming implementation MUST reproduce
`0.0_f32` on this specific input as a regression check.

5.2. **Matvec.** `y[o] = dot_seq(w[o, :], x)` for `o = 0..out_features`,
where `w[o, :]` is row `o` of the row-major `[out_features, in_features]`
weight tensor (§2.5). Each output element is one independent §5.1 dot
product; rows may be computed in any order or in parallel relative to
each other (row order does not affect any single row's bits), but each
row's own reduction MUST use §5.1's exact order.

5.3. **Sequential sum.** For a vector `a` of length `n` (used for
sum-of-squares in RMSNorm §8 and the softmax denominator §10):

```
acc = 0.0_f32
for i in 0..n:
    acc = acc + a[i]      # separate add, RNE-rounded to f32
return acc
```

Same left-to-right, no-reassociation rule as §5.1.

5.4. **Elementwise add** (residual connections, §9.5/§10.3, and optional
QKV bias, §9.1 new-in-v0.2): plain `out[i] = a[i] + b[i]` for every `i`,
one IEEE-754 add each, no reduction involved.

## 6. Transcendental functions (normative)

There are now **three** routes (v0.1 had two); this spec pins all three
exactly. §6.1/§6.2/§6.3 are unchanged from v0.1; §6.5 is new in v0.2.

### 6.1 rsqrt — route (a), correctly-rounded, no table

```
rsqrt(x) = 1.0_f32 / x.sqrt()
```

Composed from two IEEE-754-mandatory correctly-rounded operations
(`sqrt`, then `/`). Because both are mandatory-correctly-rounded under the
standard, **any** conformant IEEE-754 binary32 implementation on any ISA
produces identical bits for identical `x` — no coefficient table, no
digest, no pinning needed beyond "use the standard's `sqrt` and `/`, not
an approximate hardware reciprocal-sqrt instruction" (already stated in
§1.6). `rsqrt(64.0) == 0.125` exactly (`0x3E000000`) is a conformance
check.

### 6.2 exp — route (b), pinned Cephes-pattern polynomial

For `x` a finite `f32`:

1. If `x` is NaN, return NaN.
2. If `x > 88.0`, return `+Infinity`.
3. If `x < -88.0`, return `0.0`.
4. Range reduction: find integer `k` and remainder `r` such that
   `x ≈ k·ln(2) + r`, `|r|` small, via:
   ```
   t   = x * EXP_LOG2E          # x / ln(2), separate mul
   z0  = t + 0.5
   k   = floor(z0)               # k as f32, truncated toward -inf at z0
   kc1 = k * EXP_C1
   r0  = x - kc1
   kc2 = k * EXP_C2
   r   = r0 - kc2                 # two-part ln(2) split, Cephes pattern
   ```
5. Polynomial evaluation (Horner, strict left-to-right, descending
   coefficient index, no FMA):
   ```
   r2   = r * r
   poly = EXP_P[0]
   for i in 1..=5:
       poly = poly * r + EXP_P[i]
   m1     = poly * r2
   poly2  = m1 + r
   result = poly2 + 1.0
   ```
   (This computes `exp(r) ≈ 1 + r + r²·P(r)` with `P` the degree-5
   polynomial `EXP_P[0]·r⁵ + EXP_P[1]·r⁴ + ... + EXP_P[5]`.)
6. Reconstruct `exp(x) = 2^k · result` via **exact bit manipulation** of
   the `f32` exponent field (`ldexp_exact`, §6.2.1) — not a library
   `ldexp`/multiply-by-power-of-2-computed-as-a-float call.

Pinned coefficients (`f32` bit patterns, hex, big-endian digit order of
the 32-bit pattern as conventionally written — i.e. `0xSEEEEEEE MMMMMMM`
read as one `u32`):

```
EXP_LOG2E = 0x3FB8AA3B   (1/ln2)
EXP_C1    = 0x3F318000   (ln2 hi)
EXP_C2    = 0xB95E8083   (ln2 lo, negative)
EXP_P[0]  = 0x39506967
EXP_P[1]  = 0x3AB743CE
EXP_P[2]  = 0x3C088908
EXP_P[3]  = 0x3D2AA9C1
EXP_P[4]  = 0x3E2AAAAA
EXP_P[5]  = 0x3F000000   (== 0.5 exactly)
```

#### 6.2.1 `ldexp_exact(x, k)` — exact scale-by-power-of-2

```
if x == 0.0: return x
bits     = x.to_bits()
exp_bits = (bits >> 23) & 0xFF          # 8-bit biased exponent field
new_exp  = exp_bits + k                  # k is a signed integer
if new_exp <= 0:   return 0.0             # underflow — FTZ/DAZ (§1.3) governs anyway
if new_exp >= 0xFF: return (x.is_sign_negative() ? -Infinity : +Infinity)
new_bits = (bits & !(0xFF << 23)) | (new_exp << 23)   # replace exponent field only
return f32::from_bits(new_bits)
```
Sign and mantissa bits are untouched; only the 8-bit exponent field is
replaced. This is exact (no rounding) whenever the result stays in the
normal-or-flush range, which the `exp()` domain clamp (steps 2–3) and the
saturation branches above guarantee.

### 6.3 sin/cos — route (b), octant reduction + SEPARATE minimax polynomials (CHANGED in v0.3b — supersedes v0.3, closes v0.2 §6.7's finding)

**History (repeated here for §16's changelog):**
- **v0.2's construction** (no longer normative): a two-part-π split
  (`TWO_PI_HI`/`TWO_PI_LO`) done entirely in `f32`. Found
  (`docs/H2_TRANSCENDENTAL_RIGOR.md`, v0.2 §6.7) to suffer catastrophic
  cancellation growing with the RoPE angle's magnitude, up to 1104 ULP /
  1e-4 relative error by decode position 19, and ~0.58 relative
  (unusable) toward `max_position_embeddings = 8192` — an accuracy
  defect, not a determinism defect.
- **v0.3's construction** (no longer normative, superseded by v0.3b
  below): fixed the reduction by staging it through `f64` (exact widen,
  correctly-rounded f64 div/round/mul/sub, correctly-rounded narrow back
  to f32) while keeping v0.2's degree-10 Taylor polynomials on the
  resulting `r ∈ [-π, π]` unchanged. This closed the *reduction's*
  accuracy defect (reduced remainder now accurate to a few ULP of f64,
  utterly negligible next to f32's own ULP) but left the *fixed Taylor
  polynomial's own truncation error* as the dominant remaining term,
  worst near `|r| ≈ π` and near `sin(r)`/`cos(r)` zero crossings — up to
  ~2813 raw ULP / 486 ULP filtered away from zero crossings even after
  the reduction fix, missing this spec family's pre-registered ≤2-ULP
  accuracy bar (`docs/E15m_PREREG.md`, `docs/E15m_RESULT.md`'s v0.3
  section — reported honestly as a partial result, not silently narrowed).

**v0.3b's construction (normative).** Reducing to a *quarter*-period
octant (`r ∈ [-π/4, π/4]`, quadrant index `k mod 4`) instead of a full
period (`r ∈ [-π, π]`) lets a much lower-degree **minimax** polynomial
(fit to minimize worst-case error over the whole reduced interval, rather
than a Taylor series truncated at an arbitrary degree) reach far better
accuracy across that smaller domain. Still f64-staged (same determinism
argument as v0.3), still no FMA, still no fast-math/reassociation (§1.4/
§1.5):

**Reduction** (Cody–Waite pattern, f64-staged, strict left-to-right):

```
xd    = x as f64                  # exact widening cast, f32 -> f64, zero rounding error
k     = round(xd / PI_2_F64)      # f64 division (correctly rounded), then round-to-
                                   # nearest-integer, ties away from zero (f64::round()
                                   # semantics: pure function of the input bit pattern,
                                   # no ISA-dependent instruction selection, §1.5)
khi   = k * PI_2_F64               # f64 multiply (correctly rounded)
r64   = xd - khi                   # f64 subtract (correctly rounded)
r     = r64 as f32                 # single correctly-rounded f64 -> f32 narrowing cast
quadrant = ((k as i64) % 4 + 4) % 4  # k is exact (small integer in f64), cast to i64 losslessly
```

```
PI_2_F64 = 0x3FF921FB54442D18   (f64 bit pattern; == 1.5707963267948966, the
                                 correctly-rounded f64 value of pi/2)
```

**Why f64-staging suffices instead of full Payne-Hanek**: unchanged
argument from v0.3 (`docs/E15m_PREREG.md`) — the RoPE angle is bounded,
`|x| = |pos · inv_freq[i]| < max_position_embeddings = 8192` (§2.2, §7.1),
which leaves ample f64 mantissa headroom regardless of whether the modulus
is `2π` or `π/2`.

**Polynomials** (Cephes `sinf`/`cosf`'s minimax coefficients — long
public-domain algorithm shape, coefficients independently pinned as
literal f32 bit patterns per this spec's own convention, §6.2's preamble):
for `r ∈ [-π/4, π/4]`, strict left-to-right, no FMA:

```
# sin(r) = r + r^3 * (SIN_C0 + r^2 * (SIN_C1 + r^2 * SIN_C2))
r2      = r * r
inner   = SIN_C2
inner   = inner * r2 + SIN_C1
inner   = inner * r2 + SIN_C0
r3      = r2 * r
term    = inner * r3
sin_r   = r + term

# cos(r) = 1 - r^2/2 + r^4 * (COS_C0 + r^2 * (COS_C1 + r^2 * COS_C2))
r2      = r * r
inner   = COS_C2
inner   = inner * r2 + COS_C1
inner   = inner * r2 + COS_C0
r4      = r2 * r2
term    = inner * r4
half_r2 = 0.5 * r2
step1   = 1.0 - half_r2
cos_r   = step1 + term
```

```
SIN_C0 = 0xBE2AAAA3   (== -0.16666655242443085, r^3 coefficient)
SIN_C1 = 0x3C08839E   (== 0.008332161232829094, r^5 coefficient)
SIN_C2 = 0xB94CA1F9   (== -0.00019515295571181923, r^7 coefficient)

COS_C0 = 0x3D2AAAA5   (== 0.04166664555668831, r^4 coefficient)
COS_C1 = 0xBAB6061A   (== -0.0013887316454201937, r^6 coefficient)
COS_C2 = 0x37CCF5CE   (== 2.44331567955669e-05, r^8 coefficient)
```

**Quadrant sign/swap** (standard `sin(k·π/2 + r)`/`cos(k·π/2 + r)`
identities for `k mod 4 ∈ {0,1,2,3}`):

```
quadrant = 0: sin(x) = sin_r         cos(x) = cos_r
quadrant = 1: sin(x) = cos_r         cos(x) = -sin_r
quadrant = 2: sin(x) = -sin_r        cos(x) = -cos_r
quadrant = 3: sin(x) = -cos_r        cos(x) = sin_r
```

**Determinism**: identical argument to v0.3's (§6.3 history above,
`docs/E15m_PREREG.md`) — every reduction op is an exact widening cast, an
IEEE-754-mandatory correctly-rounded f64 arithmetic operation,
`f64::round()` (deterministic bit-pattern function), or a single
correctly-rounded narrowing cast; the polynomial evaluation is ordinary
strict-left-to-right f32 arithmetic, no FMA, no reassociation.

**Accuracy** (§6.7's re-measurement): max ULP vs. a correctly-rounded fp32
oracle over the full RoPE position domain (`pos ∈ [0, 8192)`, both pinned
`rope_theta` values) is now **≤2 ULP** (measured max 1.5 ULP on a dense
grid, `scripts/h2m_accuracy_harness.py`), closing the accuracy bar this
task pre-registered and v0.3 (Taylor-polynomial-only fix) missed.

### 6.4 SiLU

```
silu(x) = x / (1.0 + exp_pinned(-x))
```
i.e.: `neg = -x`; `e = exp_pinned(neg)`; `denom = 1.0 + e` (separate add);
`return x / denom` (one IEEE-754-mandatory-correctly-rounded division, not
a reciprocal-multiply).

### 6.5 ln — route (b), pinned Cephes-pattern polynomial (NEW in v0.2, closes v0.1 §14.1)

For `x` a finite `f32`, `x >= 0`:

1. If `x` is NaN or `x < 0.0`, return NaN.
2. If `x == 0.0`, return `-Infinity`.
3. **Exact frexp** (`frexp_exact`, §6.5.1): split `x = m · 2^e`, `m ∈
   [0.5, 1.0)`, via bit manipulation only (no rounding, no library call).
4. Mantissa range fix-up (strict, no FMA):
   ```
   if m < LOG_SQRTHF:      # m < sqrt(0.5)
       e = e - 1
       m = m + m - 1.0      # m := 2m - 1
   else:
       m = m - 1.0
   ```
5. Polynomial evaluation (Horner, strict left-to-right, ascending-then-
   folded per the Cephes logf structure, no FMA):
   ```
   z    = m * m
   poly = LOG_P[0]
   for i in 1..=8:
       poly = poly * m + LOG_P[i]
   y      = poly * m           # y := poly(m) * m
   y      = y * z
   fe     = e as f32            # exact integer->f32 cast (e is small, always exact)
   t1     = fe * LOG_Q1
   y      = y + t1
   half_z = 0.5 * z
   y      = y - half_z
   result = m + y
   t2     = fe * LOG_Q2
   result = result + t2
   ```
   (This computes `ln(x) = ln(m) + e·ln(2)`, with `ln(m)` via the degree-9
   Cephes polynomial `m + m·z·P(m) - z/2` and `e·ln(2)` split as
   `e·LOG_Q1 + e·LOG_Q2` — the same two-part-ln(2) split pattern as
   `EXP_C1`/`EXP_C2`, deliberately reusing identical bit patterns: `LOG_Q1
   == EXP_C2`, `LOG_Q2 == EXP_C1`.)

Pinned coefficients (`f32` bit patterns, hex):

```
LOG_SQRTHF = 0x3F3504F3   (sqrt(0.5))
LOG_Q1     = 0xB95E8083   (ln2 lo, negative — identical bits to EXP_C2)
LOG_Q2     = 0x3F318000   (ln2 hi — identical bits to EXP_C1)
LOG_P[0]   = 0x3D9021BB
LOG_P[1]   = 0xBDEBD1B8
LOG_P[2]   = 0x3DEF251B
LOG_P[3]   = 0xBDFE5D4F
LOG_P[4]   = 0x3E11E9BF
LOG_P[5]   = 0xBE2AAE50
LOG_P[6]   = 0x3E4CCEAD
LOG_P[7]   = 0xBE7FFFFC
LOG_P[8]   = 0x3EAAAAAA
```

**Which `rope_theta` values are conformant.** This construction is used
exactly once per model load, on exactly one input (`ln_pinned(rope_theta as
f32)`, §7.1) — it is not a per-decode-step hot path. It has been validated
(unit test, ≤2e-6 relative tolerance against host `f64::ln` cast down, §6.5
citing `src/math.rs` `ln_matches_std_within_tolerance`) at `x ∈ {0.001, 0.1,
0.5, 0.999, 1.0, 1.5, 2.0, 10.0, 100.0, 100_000.0, 1_000_000.0}`. The two
values that matter for this spec's actual pinned models are
`rope_theta = 100_000.0` (SmolLM2-135M, §13's pinned test vector) and
`rope_theta = 1_000_000.0` (Qwen2.5-0.5B, §0's informative second-model
evidence, `docs/E15d_bc_RESULT.md`); **both are conformant** under this
tolerance. This spec does **not** claim `ln_pinned` is correctly rounded.
It is, however, no longer restricted to the tested set above: `ln_pinned`
has since been evaluated at **every** f32 bit pattern and is within one ULP
of the f64 value narrowed to f32 at all 2,130,706,432 positive normals, and
**0 ULP** — not merely within 2e-6 — at both pinned `rope_theta` values. See
§14.1 erratum E-4 and `docs/E27_EXP_LN_RANGE.md`; the corresponding caution
for `sin`/`cos` was v0.1 §14.4 and was discharged the same way (erratum
E-2). A clean-room implementer targeting an arbitrary `rope_theta` should
still note that §6.5 returns `-Infinity` for every subnormal input under
§1.3's DAZ.

#### 6.5.1 `frexp_exact(x)` — exact mantissa/exponent split

```
bits     = x.to_bits()
exp_bits = (bits >> 23) & 0xFF                      # 8-bit biased exponent field, as i32
mantissa_bits = (bits & 0x807FFFFF) | (126 << 23)   # force exponent field to 126 (bias 127-1)
mantissa = f32::from_bits(mantissa_bits)             # in [0.5, 1.0)
exponent = exp_bits - 126
return (mantissa, exponent)
```
Exact (bit reinterpretation only, no rounding), matching §6.2.1's
`ldexp_exact` in spirit (the inverse bit-field operation). Precondition:
`x > 0.0` and finite (checked by the domain guard in step 1–2 above before
this is called).

### 6.6 Table digest (NOW BOUND into the witness chain — see §12.1, closes v0.1 §14.3; CHANGED in v0.3b)

SHA-256 over the LE bytes of every pinned coefficient above, in this exact
declared order — `[EXP_LOG2E, EXP_C1, EXP_C2]`, then `EXP_P[0..6]`, then
**(changed in v0.3b)** `PI_2_F64` (8 little-endian bytes of the `f64` bit
pattern, replacing v0.3's `TWO_PI_F64` and, before that, v0.2's two 4-byte
`f32` half-constants `[TWO_PI_HI, TWO_PI_LO]`), then `SIN_C0`, `SIN_C1`,
`SIN_C2` (replacing v0.2/v0.3's `SIN_COEF[0..11]`), then `COS_C0`,
`COS_C1`, `COS_C2` (replacing `COS_COEF[0..11]`), then `LOG_SQRTHF`, then
`LOG_Q1`, then `LOG_Q2`, then `LOG_P[0..9]` (all `f32`, 4 little-endian
bytes each), concatenated, one continuous SHA-256 stream:

```
table_digest = 23c7bfaf5cef0095fd021af2eb1808abb4928bae4219756d86bdac670a06b35d
```

**v0.3 vs v0.3b**: the *set* of constants hashed changed (one 8-byte
`PI_2_F64` plus 6 minimax coefficients replace `TWO_PI_F64` plus the two
11-entry Taylor tables), so this digest differs from v0.3's
`986abc500e...` (and v0.2's `465d358ccd...`) for that reason alone. This
digest remains one of the raw 32-byte inputs folded directly into
`CIS2_REF` — see §12.1 (unchanged from v0.2's binding).

### 6.7 Accuracy characterization (informative; v0.2 addendum, UPDATED in v0.3 then v0.3b — closes the H2 finding)

§6.2–§6.5 pin exact coefficients; §6.2's addendum (originally added in
v0.2) quantifies their error vs. a correctly-rounded fp32 reference. **v0.3
re-ran this characterization after §6.3's f64-staged reduction fix and
found the fixed degree-10 Taylor polynomial's own truncation error now
dominant (up to ~2813 raw ULP / 486 ULP filtered away from zero
crossings) — a PARTIAL result, missing this task's pre-registered ≤2-ULP
bar. v0.3b (octant reduction + separate minimax polynomials, §6.3) closes
that gap.** Full method, oracle, and domain derivation in
`docs/H2_TRANSCENDENTAL_RIGOR.md` (v0.2 finding) and
`docs/E15m_RESULT.md` (v0.3 partial result + v0.3b resolution).
`exp_pinned`/`ln_pinned`/`rsqrt_cr` are unchanged from v0.2 (their code did
not change); `sin_pinned`/`cos_pinned` are re-measured below, over the full
`pos ∈ [0, 8192)` domain (v0.2's table only covered `pos ≤ 19`):

| function | domain measured | max ULP vs. correctly-rounded fp32 | max relative error |
|---|---|---|---|
| `rsqrt_cr` | all finite x>0 | 0 (correctly rounded by construction) | 0 |
| `exp_pinned` | **all 2^32 f32** (E27) | **1** | **1.19e-7** (= 2^-23) |
| `ln_pinned` | **all 2^32 f32** (E27) | **1** over all positive normals; **0** at both pinned thetas | **1.19e-7** / 0 at both thetas |
| `silu_pinned` | **all 2^32 f32** (E27) | **2** off §6.2's clip band | **2.33e-7** |
| `sin_pinned` (v0.3, Taylor, informative/superseded) | RoPE angle, pos ∈ [0, 8192) | up to 2813.5 raw (486.5 filtered) | 2.37e-4 |
| `sin_pinned` (v0.3b, minimax, NORMATIVE) | same | **≤1.5** | **1.2e-7** |
| `cos_pinned` (v0.3, Taylor, informative/superseded) | same | up to 407 raw (144.5 filtered) | 3.14e-5 |
| `cos_pinned` (v0.3b, minimax, NORMATIVE) | same | **≤1.5** | **1.2e-7** |

**Resolution**: `docs/E15m_PREREG.md`/`docs/E15m_RESULT.md` document both
the f64-staging fix (v0.3, partial) and the octant-reduction +
minimax-polynomial fix (v0.3b, closes the bar) and their shared
cross-ISA determinism justification (both stages are f64-staged
identically; only the reduction period and the polynomial degree/domain
differ). `table_digest` (§6.6) changed at each step; `inv_freq_table_digest`
(§7.2) never changed (RoPE's `inv_freq` construction only calls
`ln_pinned`/`exp_pinned`, neither of which changed at any point in this
task). This closes the "if decode length were ever extended toward
`max_position_embeddings = 8192`" caveat v0.2 §6.7 flagged for future
work — v0.3b measures that full range directly and meets a ≤2-ULP bar
across it, rather than extrapolating from `pos ≤ 19`.

## 7. RoPE (normative, now theta-general — closes v0.1 §14.1)

7.1. **`inv_freq` construction.** For `i = 0..head_dim/2` (32 values for
SmolLM2-135M's `head_dim = 64`; a different `head_dim` scales this
identically):

```
ln_theta = ln_pinned(rope_theta as f32)          # §6.5, NEW: general, not a literal
for i in 0..head_dim/2:
    frac    = (2*i as f32) / (head_dim as f32)     # e.g. i=0 -> 0.0, i=1 -> 2/64
    neg_arg = -(frac * ln_theta)                    # separate mul, then negate
    inv_freq[i] = exp_pinned(neg_arg)                # §6.2, the SAME pinned exp
```

**v0.1 vs v0.2**: v0.1 pinned `LN_THETA` as a bare literal f32 bit pattern
(`0x413834F1`) valid **only** for `rope_theta == 100000.0`, and mandated the
loader assert this exact value and refuse to run otherwise (v0.1 §2.4).
v0.2 **removes that restriction**: `ln_theta` is computed at load time from
whatever `rope_theta` is present in `config.json`, via the pinned `ln`
polynomial (§6.5) — not any host `pow`/`powf`/`ln` call. For
`rope_theta = 100000.0` specifically, `ln_pinned(100_000.0)` MUST agree with
the old v0.1 literal `0x413834F1` to within the same ≤2e-6 relative
tolerance used elsewhere in this spec (it is not required to be bit-
identical to the old hardcoded literal, since the literal was the
RNE-nearest f32 to the true `ln(100000)` while `ln_pinned` is a polynomial
approximation — the two need not collide at the last bit; what matters is
that `inv_freq_table_digest`, §7.2, still reproduces bit-for-bit run-to-run
and cross-ISA, which it does, §13.1).

Mathematically this is still `inv_freq[i] = theta^(-2i/head_dim) =
exp(-(2i/head_dim)·ln(theta))`, evaluated with the pinned `exp_pinned`
polynomial (§6.2) and now also the pinned `ln_pinned` polynomial (§6.5)
rather than any host `pow`/`powf`/`ln` call — this is a deliberate, pinned
choice (superseding both the v0.1 literal draft and an earlier,
non-normative pre-v0.1 draft that used host `f64::powf`; see Appendix A for
the exact history).

7.2. **`inv_freq` table digest.** SHA-256 over the LE bytes of each of the
`head_dim/2` `inv_freq` values, `f32.to_le_bytes()`, in index order, one
continuous stream. **Unchanged construction from v0.1** (this digest was
already raw bytes, not hex-ASCII, in v0.1 — see §14.2 (v0.1 numbering) /
Appendix B item 3). For SmolLM2-135M (`head_dim=64`, `rope_theta=100000.0`,
32 values):

```
inv_freq_table_digest = da9f6dcfde0425588815509e874515cdcd3d6b8818b6d0136590052e7bbf6f12
```

Identical bit-for-bit to v0.1's value for this same model — evidence that
switching from the bare `LN_THETA` literal to the general `ln_pinned(rope_theta)`
call did not change SmolLM2-135M's `inv_freq` table at all (§16 changelog
note). **Now bound into `CIS2_REF`** (§12.1) — v0.1 computed and printed
this digest but never fed it into the witness chain (v0.1 §14.3); v0.2
closes that gap.

7.3. **Per-position cos/sin table.** For a query/key head being rotated
at sequence position `pos` (a `usize`, cast to `f32` exactly — `pos` never
exceeds 19 in this spec's fixed 20-position decode, well within `f32`'s
exact-integer range):

```
for i in 0..half (half = head_dim/2 = 32):
    angle    = (pos as f32) * inv_freq[i]
    cos_v[i] = cos_pinned(angle)     # §6.3
    sin_v[i] = sin_pinned(angle)     # §6.3
```

7.4. **Rotation (rotate-half convention, `rope_interleaved = false`).**
Given a `head_dim`-length slice `head[0..head_dim]` for one attention
head, with `half = head_dim/2`:

```
for i in 0..half:
    x1 = head[i]
    x2 = head[i + half]
    t1 = x1 * cos_v[i]
    t2 = (-x2) * sin_v[i]
    out[i]        = t1 + t2
    t3 = x2 * cos_v[i]
    t4 = x1 * sin_v[i]
    out[i + half] = t3 + t4
head[0..head_dim] = out[0..head_dim]   # in place, after computing all `out`
```

This is applied to **every** query head (9 heads, each `head_dim=64`
slice of the 576-wide `q` vector) and **every** key head (3 heads, each
`head_dim=64` slice of the 192-wide `k` vector) independently, once per
decode step, at that step's `pos`. Values are never rotated more than
once (the KV cache stores post-RoPE `k`; §3.5).

## 8. RMSNorm (normative)

**Unchanged from v0.1** (verbatim):

For input vector `x` of length `n = hidden = 576`, gain vector `weight`
of the same length, and `eps` = `EPS_F32` (§2.3):

```
for i in 0..n: sq[i] = x[i] * x[i]
ss   = sum_seq(sq)              # §5.3, strict left-to-right
mean = ss / (n as f32)          # one IEEE-754 division
inv  = rsqrt(mean + eps)        # §6.1: 1.0 / (mean+eps).sqrt()
for i in 0..n:
    scaled = x[i] * inv
    out[i] = scaled * weight[i]     # (x[i]*inv)*weight[i] — THIS association, not x[i]*(inv*weight[i])
```

The multiply order `(x[i] * inv) * weight[i]` (not `x[i] * (inv *
weight[i])`) is pinned explicitly: floating-point multiplication is not
associative under rounding, so these two orders can differ in the low bit
in general, even though no divergence from this specific reordering has
been observed on this model/prompt.

Applied twice per layer (`input_layernorm` before attention, in §2.5's
naming `post_attention_layernorm` before the MLP) and once after the final
layer (`model.norm.weight`), all using the same procedure and the same
`eps`.

## 9. Attention (GQA, causal, per decode step) (normative)

Let `qh` range over `n_heads = 9` query heads, `kv_head = qh /
group` (`group = 3`) map each query head to its shared KV head (integer
division), `head_dim = 64`.

9.1. **Projections.** `q = matvec(q_proj, ln1, 576, 576)`; `k =
matvec(k_proj, ln1, 192, 576)`; `v = matvec(v_proj, ln1, 192, 576)`
(§5.2), where `ln1` is this layer's post-RMSNorm hidden state (§8).
**New in v0.2 (informative for SmolLM2-135M, normative for any checkpoint
that carries these tensors)**: if the checkpoint provides
`self_attn.{q,k,v}_proj.bias` tensors (Qwen2-family
`attention_bias=true`), each is added elementwise (§5.4) to the
corresponding projection immediately after the matvec, before RoPE; if the
checkpoint has no such tensors (Llama-family, including SmolLM2-135M),
this step is a no-op and the pinned §13 test vector is unaffected. RoPE
(§7) is then applied in place to each of the 9 query-head slices of `q`
and each of the 3 key-head slices of `k`, at the current step's `pos`. `v`
is never rotated.

9.2. **Score.** For query head `qh` at position `pos`, against every
cached key position `j = 0..=pos` (causal: only positions ≤ current,
enforced by the KV cache containing exactly those positions, not by an
explicit mask value):

```
scale     = rsqrt(head_dim as f32)     # computed once: rsqrt(64.0) == 0.125 exactly
d         = dot_seq(q_head, k_j)        # §5.1, over the 64-dim head slice
scores[j] = d * scale                    # separate multiply, applied AFTER the dot
```

9.3. **Softmax** (`softmax_seq`, in place over `scores[0..=pos]`):

```
max_v = scores[0]
for v in scores[1..]:
    if v > max_v: max_v = v      # strict >, so the FIRST occurrence of the max wins ties
for v in scores: v = exp_pinned(v - max_v)     # §6.2
denom = sum_seq(scores)           # §5.3
for v in scores: v = v / denom     # elementwise division, NOT multiply-by-reciprocal
```

9.4. **V-mix.** For each output dimension `d = 0..head_dim`:

```
acc = 0.0_f32
for j in 0..=pos:
    p   = scores[j] * v_cache[j][kv_head][d]
    acc = acc + p                                # separate mul, separate add — same
                                                   # left-to-right order as §5.1, written
                                                   # as an explicit loop rather than a
                                                   # dot_seq call, but semantically identical
out_head[d] = acc
```

9.5. **Output projection and residual.** Concatenate all 9 `out_head`
slices into a 576-wide `attn_out`; `o = matvec(o_proj, attn_out, 576,
576)`; `h = elementwise_add(h, o)` (§5.4).

## 10. MLP / SwiGLU (normative)

**Unchanged from v0.1** (verbatim):

Given this layer's post-`post_attention_layernorm` hidden state `ln2`
(576-wide):

```
gate = matvec(gate_proj, ln2, 1536, 576)     # §5.2
up   = matvec(up_proj,   ln2, 1536, 576)
for i in 0..1536:
    hid[i] = silu_pinned(gate[i]) * up[i]     # §6.4, then one separate multiply
down = matvec(down_proj, hid, 576, 1536)
h    = elementwise_add(h, down)                # §5.4
```

## 11. LM head and argmax (normative)

11.1. After the final layer, `hn = rmsnorm(h, model.norm.weight, eps)`
(§8). `logits = matvec(lm_head_weights, hn, vocab, hidden)` (§5.2). For
SmolLM2-135M (`tie_word_embeddings=true`), `lm_head_weights` is the tied
`embed_tokens.weight` matrix directly (§2.5) — no separate weight, no
transpose (the embedding table is already `[vocab, hidden]`, the exact
shape a `matvec` row-dot needs). **New in v0.2 (informative for
SmolLM2-135M, normative for any checkpoint with
`tie_word_embeddings=false`)**: if the checkpoint config sets
`tie_word_embeddings=false` and provides a separate `lm_head.weight`
tensor (`[vocab, hidden]`), that tensor is used instead of `embed_tokens`;
this branch is untested by this version's pinned §13 test vector
(SmolLM2-135M is tied) but is exercised by the informative Qwen2.5-0.5B
evidence (§0), which is itself tied (`tie_word_embeddings=true` for the
base, non-Instruct checkpoint) — so even that evidence does not exercise
the untied branch end-to-end; a clean-room implementer targeting an
untied checkpoint should treat this branch as spec-described but
less-validated than the tied path.

11.2. **Argmax**, strict left-to-right scan, first-occurrence-wins on
exact ties:

```
best_idx = 0
best_val = logits[0]
for idx, v in enumerate(logits):
    if v > best_val:          # strict >, never >=
        best_val = v
        best_idx = idx
next_token_id = best_idx as u32
```

## 12. Digest / receipt format (normative)

All digests are SHA-256 (32 raw bytes), rendered as lowercase hex (64
ASCII characters) **only for display/printing** — the witness chain itself
consumes raw bytes, not the hex rendering (§12.1, this is the v0.2 change
from v0.1). There are **three separate, independently computed** digests;
none of them contains any of the others as a sub-input except where
explicitly stated.

12.1. **Witness chain / `CIS2_REF` digest.** One continuous
`Sha256::update()` stream (not a step-wise re-hash of
`running_digest || new_data` — this is a single hash object updated
repeatedly, then finalized once), fed in exactly this order:

1. `weights_sha256` — the **raw 32-byte SHA-256 digest** of
   `model.safetensors` (§2.1). **CHANGED in v0.2**: v0.1 fed the
   **64-character lowercase hex ASCII string** (the digest's UTF-8/ASCII
   bytes, 64 bytes) instead of these 32 raw bytes; this closes v0.1 §14.2.
2. `tokenizer_sha256` — same raw-32-byte encoding, `tokenizer.json`'s
   digest.
3. `config_sha256` — same raw-32-byte encoding, `config.json`'s digest.
4. `table_digest` — the **raw 32 bytes** of §6.6's pinned
   exp/sin/cos/ln coefficient-table digest. **NEW in v0.2** (closes v0.1
   §14.3): v0.1 computed and printed this value but never fed it into the
   witness chain.
5. `inv_freq_table_digest` — the **raw 32 bytes** of §7.2's RoPE
   `inv_freq` table digest. **NEW in v0.2** (closes v0.1 §14.3, same gap
   as item 1).

   **E15k correction (this pass):** this doc previously listed items
   1-5 in the order table_digest, inv_freq_table_digest, weights,
   tokenizer, config. That prose never matched the actual reference
   implementation (`src/main.rs`, which is the source of the pinned
   `CIS2_REF=a0c563ef...` value below) — the reference feeds
   weights/tokenizer/config first, then table/inv_freq, as corrected
   above. A from-spec-text-only clean-room (`verify3/`, E15j) followed
   the old (wrong) prose exactly and reproduced every other digest
   bit-for-bit (`table_digest`, `inv_freq_table_digest`, `argmax_digest`,
   all 16 `generated_token_ids`) but got a different `CIS2_REF`
   (`a532d4a4...`) purely from this item-order mismatch — see
   `docs/E15k_DIVERGENCE_LOCALIZATION.md`. Fixed here; `verify3/model.c`
   corrected to match and reconfirmed bit-exact on all 4 CI cells
   (x86_64/aarch64 × gcc/clang).
6. Every prompt token id, in order, each as **`u32` little-endian, 4
   bytes** (`t.to_le_bytes()`) — 4 tokens × 4 bytes = 16 bytes total for
   this prompt. **Unchanged from v0.1.**
7. For each of the `gen_toks` decode steps (16 for the pinned §13 test
   vector), in step order (`step = 0..gen_toks`):
   a. The **full fp32 logit vector for this step** (49152 entries for
      SmolLM2-135M, in vocab-index order), each entry's raw bit pattern as
      `u32` little-endian (`v.to_bits().to_le_bytes()`), concatenated —
      49152 × 4 = 196608 bytes per step for this vocab size. This is the
      logit vector **before** this step's argmax pick (i.e., the vector
      argmax was just computed against), not the vector for the *next*
      position. **Unchanged from v0.1.**
   b. The chosen `next_token_id` for this step, as **`u32` little-endian,
      4 bytes**. **Unchanged from v0.1.**

Finalize once after all steps; the resulting 32-byte digest, hex encoded,
is `CIS2_REF`. For the pinned SmolLM2-135M / `"Once upon a time"` /
16-token test vector:

```
CIS2_REF = d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df
```

printed as `CIS2_REF digest=d8274305... prompt_idx=0 prompt_toks=4
gen_toks=16 dtype=fp32`. **This digest is not backward-compatible with
v0.3's `90f7484e...`, v0.2's `a0c563ef...`, or v0.1's `ba88708bf4...`** —
v0.3b changed `table_digest` (§6.6, the octant-reduction constant plus 6
minimax coefficients replacing v0.3's single f64 reduction constant plus
the two Taylor tables), which feeds directly into this witness chain (item
4 below); v0.2→v0.3 separately changed `table_digest`'s constant set (f64
reduction constant replacing two f32 halves); the v0.1→v0.2 transition
changed the item order and the item 3–5 encoding (raw bytes, not
hex-ASCII); see §16 for the full breakdown of which change contributed
how. `generated_token_ids` and `argmax_digest` for this 16-token pinned
test vector are **unchanged from v0.1/v0.2/v0.3** (none of this task's
`sin`/`cos` accuracy fixes are large enough at `pos ≤ 19` to flip any of
the 16 greedy argmax decisions — see `docs/E15m_RESULT.md`).

12.2. **Argmax-token digest.** **Unchanged construction and unchanged
value from v0.1** (this digest was never affected by either v0.2 change —
it does not touch `table_digest`, `inv_freq_table_digest`, or any artifact
hash). A **separate** `Sha256` instance (not derived from or continuing
the witness chain above), fed the LE `u32` bytes of every token id in
`prompt_token_ids ++ [16 generated token ids]` (20 tokens total, in that
order, prompt first), one continuous stream, finalized once:

```
argmax_digest = 0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff
```

12.3. **Table digests** (§6.6, §7.2) are now **inputs to** `CIS2_REF`
(§12.1 items 1–2, new in v0.2) but remain independent of, and not inputs
to, the argmax-token digest (§12.2) — printed separately as before, in
addition to being folded into the witness chain.

12.4. **Determinism check (MUST).** The reference computes the entire
decode (§3.4) **twice** in the same process (`run1`, `run2`) and MUST
assert all three of: `witness_digest_run1 == witness_digest_run2`,
`argmax_digest_run1 == argmax_digest_run2`, and
`token_ids_run1 == token_ids_run2`, before printing any result — this is
the same-host determinism bar, a precondition for the stronger cross-ISA
claim (§13.2), not itself the interesting claim. **Unchanged from v0.1.**

## 13. Test vectors (normative)

13.1. **Full run, prompt `"Once upon a time"`, 16 greedy tokens:**

```
weights_sha256   = 80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1
config_sha256    = 1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843
tokenizer_sha256 = 9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
prompt_token_ids = [6403, 1980, 253, 655]
generated_token_ids (16, in order) =
    [28, 665, 436, 253, 1838, 8180, 3365, 14176, 30, 2306, 4161, 281, 253, 2066, 2291, 351]
table_digest           = 23c7bfaf5cef0095fd021af2eb1808abb4928bae4219756d86bdac670a06b35d
inv_freq_table_digest  = da9f6dcfde0425588815509e874515cdcd3d6b8818b6d0136590052e7bbf6f12
argmax_digest          = 0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff
CIS2_REF (witness_digest) = d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df
```

Note `generated_token_ids` and `argmax_digest` are **unchanged from v0.3,
v0.2, and v0.1** — none of v0.2's, v0.3's, or v0.3b's changes altered the actual
computed logits or greedy decisions at this 16-token/pos≤19 length, only
the receipt's hashing (v0.2, §16) or the reduction's accuracy at longer
positions than this pinned vector exercises (v0.3, §6.3/§6.7).

13.2. **Cross-run/cross-ISA conformance bar.** All of §13.1's values MUST
reproduce bit-for-bit: (a) across two sequential runs in the same process
(§12.4); (b) across two separate OS process invocations on the same host;
(c) across x86-64 and aarch64, same source, unmodified, both with §1.3's
FTZ/DAZ pin actually in effect (verified by the adversarial self-test,
§1.3) and §1.4's zero-FMA gate passing (disassembly check). **Unchanged
requirement from v0.1**, now additionally exercised by §13.4's 20-cell
compiler-invariance matrix.

13.3. **Step-0 full-logit-vector spot check (informative, not part of
`CIS2_REF`)**: an independent oracle comparison
(`docs/E15b_m1p5_CORRECTNESS.md`) against `torch`/`transformers` fp32
forward pass of the same checkpoint found, at generation step 0 (right
after prefill, before the first generated token), across all 49152
logits: `max_abs_diff ≈ 8.965e-05` (vocab index 40082: this reference
`5.486028671264648`, oracle `5.486118316650391`), `max|logit| ≈
22.531156539916992`, relative diff `≈ 3.98e-06` — evidence of
*correctness* (this is a valid forward pass), not evidence of bit-identity
with `torch` (which this spec never claims; `torch` uses its own BLAS
reduction order and libm, out of scope). This spec does not define a
`--dump-step0` flag as part of its normative surface; the debug env-var
hook (`CIS2_DUMP_STEP0_LOGITS`, Appendix A) that produced this comparison
is informative tooling, not part of conformance. **Unchanged from v0.1.**
A second, independently-run oracle comparison against Qwen2.5-0.5B
(`docs/E15d_bc_RESULT.md`, greedy 16/16 token match, logit relative
diff ≈4.4e-6) is informative evidence for §7's theta-general RoPE, not a
§13.1 test vector.

This clause compares only the *output* of the layer stack. §14.6 (erratum
E-5) records the corresponding comparison of every *intermediate* activation
— 7,467 tensors on the first model and 5,985 on the second, worst relative L2
error 9.923e-5, with no layer diverging from or compensating for the oracle
(`docs/E28_LAYER_ORACLE.md`).

13.4. **Compiler-invariance matrix (NEW in v0.2, informative but
strongly evidential)**: the pinned SmolLM2-135M test vector's `CIS2_REF`
and `inv_freq_table_digest` reproduce bit-for-bit across a 20-cell matrix
of `{x86_64, aarch64} × {opt-level 0,1,2,3,s} × {target-cpu generic,
native}`, with zero FMA instructions in every cell's disassembly (§1.4).
Preregistered digest values and the full per-cell table are recorded in
`docs/E15d_v0.2_DIGESTS.md` (this branch); the v0.1-era version of this
same check (`docs/E15d_a_COMPILER_INVARIANCE.md`) targeted the old
`ba88708b...` digest and is superseded by the v0.2 rerun.

13.5. **Optimizer-invariance at every intermediate (NEW in v0.3b,
informative but strongly evidential).** §13.4 compares two digests across a
compiler matrix. This clause compares *every intermediate activation*
across code generation targets. Three binaries built from one source tree —
a default `--release` build, a `-C target-cpu=native` build on an AVX2 part
(397 `%ymm` instructions emitted), and a `-C target-cpu=native` build on a
part without AVX2 (none) — produce, on both of §0's models, dumps that are
identical byte for byte: 7,467 tensors / 51,750,154 B for SmolLM2-135M and
5,985 tensors / 103,942,814 B for Qwen2.5-0.5B, over eight runs on two
microarchitectures. Zero FMA instructions in all three binaries (§1.4).

The same check then covers all ten `{opt-level 0,1,2,3,s} × {target-cpu
generic, native}` cells of §13.4's matrix on x86_64: **ten distinct
binaries, one dump**, every cell reproducing
`5386d3b0e529d9817af86f2ba1381193c2b174b0f26d12e22a441616dafb2f64` and the
pinned §13.1 digests, with zero FMA instructions throughout. The emitted
AVX2 instruction count rises with optimization pressure — 286 at
`opt-level s`, 344 at 1, 397 at 2, 529 at 3 — while `dot_seq`'s
multiply/add stay scalar in every cell, varying only in unroll factor,
which changes the instruction count without changing the order of the
additions.

The mechanism is the point, and it is a property of this specification
rather than of these builds. §5.1's `acc = acc + p` is a serial
floating-point dependency, so an optimizer denied fast-math (§1.5) cannot
vectorize the reduction: `matvec` is scalar in all three binaries,
including the AVX2 one. §8's sum-of-squares splits into an elementwise map
and §5.3's fold, and the AVX2 build vectorizes the map 8-wide while
emitting the fold as a scalar `vaddss` chain. **This spec pins exactly
those operations whose order changes the result and leaves free exactly
those whose order does not**, so an optimizer can act only where acting is
a no-op on the bits. A conforming implementation therefore need not ship an
unoptimized build; `--release -C target-cpu=native` was measured to be
conforming.

Two controls accompany the result, because an invariance claim is empty
without them. (a) The AVX2 vector loop is on the executed path: replacing
one of its `vmulps` instructions with `ud2` in a copy of the binary
terminates the run with SIGILL. (b) The dump is sensitive: flipping one
low-order mantissa bit of one weight in the 269 MB artifact moves 570 of
the 7,467 tensors — entering at `L27.down_proj` in a single element,
saturating L28 and L29, and moving all 19 `logits` vectors — while changing
**no** argmax decision. That last observation is the measured form of
§12.1's requirement to hash the full logit vector rather than the emitted
token ids: the corruption was visible in 100 % of the logit vectors and 0 %
of the decisions. Full method, disassembly and provenance in
`docs/E29_OPTIMIZER_INVARIANCE.md`; the per-tensor comparison tool is
`scripts/diff_dumps.py`.

The same comparison was then extended over *inputs*. Six prompt/length
configurations (different lengths, a digit-heavy prompt, a code prompt, a
single-token prompt with 32 generated tokens; all ASCII, so §3.1.4's open
question could not confound the result) were each run four times — the
generic binary on both hosts, plus each host's `target-cpu=native` build.
The six configurations produce six *different* dumps, totalling 558,394,570
bytes of intermediate activations, and within each configuration all four
runs agree on every byte.

The remaining half of §13.4's matrix has since been closed, and closed as a
standing gate rather than a measurement: the `intermediates` job in
`.github/workflows/verify.yml` runs **all twenty cells** —
`{x86_64, aarch64} × {opt-level 0,1,2,3,s} × {target-cpu generic, native}` —
on GitHub-hosted runners of both ISAs, and fails the build if any cell's
dump digest moves. Twenty distinct binaries, one dump. On aarch64 the
identity does not rest on the code having stayed scalar: NEON is
architecturally baseline there, and every cell, `-O0` included, emits
875–936 vector instructions. Zero FMA-family instructions in all twenty.

Scope limits, stated so this is not over-read: every cell ran the same
`rustc`/LLVM, so this widens the ISA and code-generation axes and not the
compiler axis (`verify3`'s four `{x86_64, aarch64} × {gcc, clang}` cells are
the independent-compiler axis, and §0's four-implementation convergence is
the independent-implementation axis, though neither compares
intermediates); six ASCII prompts, at most 64 generated tokens, on two
models; and of the remaining codegen flags, `lto=fat`, `lto=thin` and
`codegen-units=1` have since been measured over eight further cells (same
dump, same digests, zero FMA — thin LTO with `target-cpu=native` emits 1,539
AVX2 instructions, 3.9× the plain native build, and computes the same bits),
and profile-guided optimization over eight further cells (E33: instrumented
`profile-generate` builds, `profile-use` at both `target-cpu` settings, and
the same combined with thin and fat LTO — same dump, same digests, zero FMA,
including cells whose profile was trained on a *different* prompt than the
one verified, and where the profile names `ops::matvec` as 86 % of all
counted activity). Every codegen flag named here is now measured; any flag
outside `{opt-level, target-cpu, lto, codegen-units, profile-use}` remains
untested, and only `opt-level` and `target-cpu` are continuously gated.

## 14. Known gaps and internal inconsistencies (informative — read before treating this as complete)

Renumbered from v0.1's §14; items resolved by v0.2 are marked **CLOSED**
and kept for history, per §16's changelog discipline.

14.1. **`exp_pinned` and `ln_pinned` accuracy is measured exhaustively, at
every f32.** §6.2's `exp_pinned`, §6.5's `ln_pinned` and §6.4's `silu_pinned`
have each been evaluated at *all 2^32 f32 bit patterns* under the §1.3 pinned
environment against an f64 accuracy oracle. Outside §6.2's two guard bands,
every one of the 3,257,925,634 comparable `exp_pinned` arguments and all
2,130,706,432 positive-normal `ln_pinned` arguments are within **one ULP** of
the f64 value narrowed to f32, with no degradation across the argument range.
Both functions are exactly monotone over the whole finite domain.
`ln_pinned` is **0 ULP** at both pinned `rope_theta` values (`100_000.0`,
`1_000_000.0`), superseding the "≤2e-6 relative tolerance" figure quoted in
§6.5 and in v0.3b's §14.1. `silu_pinned` is within **two ULP** everywhere it
is not sitting on §6.2's clip band.

Two deliberate divergences from mathematical `exp`, both pinned:

(a) §6.2 step 2 clips at `x > 88.0`, but `ln(f32::MAX) = 88.7228390520684`.
Exactly **94,743** arguments in `[0x42B00001, 0x42B17217]` =
[88.0000076, 88.7228317] therefore return `+Infinity` where the true value is
finite and representable. §10's softmax cannot reach this band --- it
evaluates `exp_pinned(v - max_v)` with `max_v` the maximum over the same
vector, so its argument is always `≤ 0` --- but §6.4's SiLU can: an FFN
intermediate `x ∈ [-88.7228317, -88.0000076]` makes `exp_pinned(-x)` land
inside it. A read-only census of SmolLM2-135M decodes over six
prompt/length configurations (the §13.1 reference decode
`"Once upon a time"`/16 among them, plus a 43-character prompt, a
digit-heavy prompt, a code prompt, `"Once upon a time"`/64 and `"A"`/32)
records **0** of its **18,892,800** `silu_pinned` arguments in that band —
the most negative argument seen anywhere is -31.406876, some 57 units short
of the band — and **0** of its **2,355,480** softmax `exp_pinned` arguments
below `-88.0`; the instrumented build reproduces each configuration's pinned
digests exactly. So §13.1's digests do not depend on the clip. That is one
model, six ASCII prompts and at most 64 generated tokens, and does not
establish unreachability in general. This
clip is **normative and MUST be reproduced**; a clean-room implementation
that returns the finite value will not reproduce the pinned digests.

(b) §6.2 step 3 returns `0.0` for `x < -88.0`. Every value *this* guard
destroys is subnormal and would be flushed by §1.3 in any case; the measured
count of arguments where it returned zero and the oracle was nonzero is **0**.
The visible discontinuity in §6.4 comes from (a), not from this guard:
`silu_pinned(-88.0)` is `0x83354DDC` (≈ -5.328e-37) while
`silu_pinned(-88.0000076)` is `-0.0`, because `exp_pinned(88.0000076)` is
clipped to `+Infinity`. Note that `5.328e-37` is a **normal** f32 (about 45×
`f32::MIN_POSITIVE`), so §1.3's FTZ does not flush it. The error is
numerically negligible and, being produced identically by every conforming
implementation, does not affect bit-exact agreement.

Under §1.3's DAZ, `ln_pinned` returns `-Infinity` for all 16,777,214
subnormal inputs of both signs, because §6.5's `x == 0.0` guard is an SSE
compare and DAZ makes a subnormal operand compare equal to zero. That is
§1.3 acting on §6.5's guard, not a property of the polynomial;
`frexp_exact` (§6.5.1) is bit manipulation and is unaffected.

Tier-1 op-level goldens pinning all of this -- 48 `(input_bits,
output_bits)` pairs, with two structural mutation controls -- are in
`cis2-verify/src/mathpin.rs`, `mod exp_ln_range`.

**ERRATUM E-4 (2026-09-09).** Through v0.3b as published, this clause read
"`ln_pinned`'s validated domain is a finite, explicitly-tested set of `x`
values (§6.5), not a general accuracy proof ... **PARTIALLY CLOSED**". That
was accurate about what had been measured and wrong in three ways about what
is true: the stated tolerance was two orders of magnitude looser than the
truth; the clause cautioned about `ln_pinned`, which runs once per model
load, while saying nothing about `exp_pinned`, which runs twice per decode
step (§10 softmax and §6.4 SiLU); and it did not know that §6.2's high guard
clips below the representable range. §6.2, §6.4 and §6.5 are unchanged; only
this limitations note was. See CHANGELOG.md, "Errata against v0.3b", and
docs/E27_EXP_LN_RANGE.md.

**ERRATUM E-10 (2026-09-09).** The reach census quoted in (a) and the
zero count quoted in (b) were both taken on **one model, six ASCII prompts and
at most 64 generated tokens**, and the clause said so. Extending the same
instrument to **20 cells across both §0 models** --- ten per model, adding
Japanese, Cyrillic, Arabic, emoji, accented-Latin and mathematical-symbol
prompts, a 32-character repeat, a punctuation repeat, and a 256-token decode ---
censused **145,385,472** `silu_pinned` arguments and **43,523,148** softmax
`exp_pinned` arguments, against the 18,892,800 and 2,355,480 quoted above.

Two of the numbers above therefore need correcting, and one claim needs
retracting:

* The SiLU side is **strengthened, not changed**: still **0** arguments in
  §6.2's clip band, on either model, in any of the 20 cells. The deepest
  `silu_pinned` argument seen anywhere is now **-32.173088** (SmolLM2, Japanese
  prompt) rather than -31.406876 --- still some 56 units short of the band.
* "**0** of its 2,355,480 softmax `exp_pinned` arguments below `-88.0`" is
  **false in the wider scope.** Qwen2.5-0.5B decoding `"Once upon a time"` for
  256 tokens produces a softmax argument of **-88.369385**, and **2** of that
  cell's 22,626,240 arguments fall below `-88.0` and so trip (b)'s LOW guard.
  It is a monotone trend in decode length on that one prompt and model, not a
  freak: 16 tokens reaches -55.623780, 128 reaches -80.235170, 192 reaches
  -82.705530, 256 reaches -88.369385.
* (b)'s "the measured count of arguments where it returned zero and the oracle
  was nonzero is **0**" is likewise **now 2**. (b)'s *substance* is unaffected,
  though not for the reason given here: the true `exp(-88.369385)` is about
  4.2e-39, a subnormal, and the guard destroys it --- but §1.3 never sees it.
  See erratum **E-12**.

A second counter, absent when this clause was written, records arguments in
`[-88.0, -87.33654022216797)`, where no guard fires and `exp_pinned` is
evaluated. The same Qwen cell puts **2** arguments there. E22's M01 mutant ---
§1.3's pin gutted, and instrumented to prove the pin is really absent --- was
re-run on that cell: the witness and argmax digests are **byte-identical to the
pinned build**.

> **This paragraph originally called that band the FTZ-*dependent* window and
> said "the result is a subnormal that §1.3 flushes", concluding that "a
> denormal genuinely arises in a real decode". That is wrong and is withdrawn
> by erratum E-12 below. The band and its count of 2 are real; the mechanism
> attributed to them is not.**

Nothing normative changes. The clip in (a) and the guard in (b) are unaltered,
and every digest in §13 is unaffected. What changes is the scope of the
supporting measurement and the retraction of a count. The clause's own caveat
--- "does not establish unreachability in general" --- was right to be there.
See CHANGELOG.md and docs/E38_REACH_SWEEP.md.

**ERRATUM E-12 (2026-09-09).** (b) says every value §6.2's low guard destroys
"would be flushed by §1.3 in any case", and erratum E-10 above extended that
reading to the band `[-88.0, -87.33654022216797)`, calling it the window where
§1.3's FTZ/DAZ pin is digest-relevant. **Both statements name the wrong
mechanism, and `exp_pinned` cannot reach the state they describe.**

§6.2's final step is `ldexp_exact(result, k)`, which is bit manipulation, not a
floating-point operation, and returns exactly `+0.0` whenever the reconstructed
exponent field would be `<= 0`. `result` always carries exponent field 126 or
127, so `exp_pinned` returns either `+0.0` or a **normal** f32. Evaluating
`exp_pinned` on **all 4,294,967,296** f32 bit patterns yields **0** subnormal
outputs; the smallest nonzero magnitude it can produce anywhere is `0x00800026`
(exponent field 1, at `x = 0xc2aeac4f`). **No FTZ decision is ever taken on
`exp_pinned`'s output.** Direct measurement agrees: with the pin cleared and
restored around each evaluation, every point in the band returns `0x00000000`
both ways.

The consequences are confined to the stated reason:

* (b)'s conclusion --- the guard is harmless --- is **correct**. Its reason is
  not: the values are destroyed by `ldexp_exact`'s zero branch, before §1.3 is
  consulted. §1.3 does not participate.
* The band counter measures arguments **whose true `exp` is subnormal**, not
  subnormals computed and flushed. It is FTZ-*independent*, like the guarded
  side it was introduced to contrast with.
* "A denormal genuinely arises in a real decode" is **withdrawn**. Through
  §6.2, on the measured vector, none did.
* E-10's *reach* result is untouched: the softmax argument really does reach
  -88.369385, and (a)'s and (b)'s counts really are 2.
* No digest, coefficient, guard or required behaviour changes.

Where §1.3 **can** change bits in this pipeline is §10's elementwise division
`w[i] = exp_i / denom`: `exp_pinned` yields a normal, `denom >= 1` because
max-subtraction puts `exp(0) = 1` in the sum, and a smallest-normal numerator
over a denominator above 1 is subnormal --- an SSE operation FTZ governs. That
quantity had never been counted. Measured over the same Qwen cell,
**22,626,240** softmax weights produced **0** subnormal quotients and **0**
where FTZ would have changed the stored bits; the smallest nonzero weight is
`1.5562307486661955e-38`, a factor of **1.32** above the subnormal boundary. So
on this vector §1.3 is not digest-relevant through §10 --- which fully explains
E-10's byte-identical M01 digests --- and it is **untriggered, not shown
unnecessary**: a third of a binade more spread in one attention row would
trigger it. §14's caveat that a SAME digest is weaker than "no denormal ever
arose" therefore stands, now with the measurement that makes it precise. See
CHANGELOG.md and docs/E39_FTZ_IS_NOT_WHERE_WE_SAID.md.

**ERRATUM E-13 (2026-09-09).** E-12 closed one operation (§10's division) and
left the rest of the pipeline uncounted, so "untriggered, not shown unnecessary"
was stated for §10 alone. The remaining operations have now been counted, and
the scope of that sentence widens to the whole decode.

Every §5.1 product and partial sum, every §8 RMSNorm intermediate and every §10
softmax weight of the §13.1 decode was classified exactly, on both §0
checkpoints: **562,531,070,920** intermediates, **0** subnormal and **0** where
FTZ would have changed the stored bits. Every §2.5 weight operand the decode
reads was scanned first, since DAZ acts on inputs: **628,547,776** operands, **0**
subnormal. The operations no counter reaches --- §11's residual adds, §7's RoPE
rotations, §9.2's score scaling, §10's max-subtraction, and the polynomial
internals of §6.2/§6.3 --- are covered instead by a differential: the §14.6 layer
dump run twice, once with §1.3's FTZ and DAZ cleared for the whole decode, is
**byte-identical** across all **7,467** named intermediate tensors of the
normative vector (and 5,985 of the Qwen vector), and those dumps contain **0**
stored subnormals and **0** exact zeros in 38,775,808 f32 values.

So on the normative vector **§1.3 is not digest-relevant anywhere**, not merely
through §10. The requirement stands unchanged and for the unchanged reason:
§1.3 exists so that the *platform* cannot answer the question, and a conforming
implementation must pin FTZ/DAZ whether or not its inputs exercise the pin. What
is now measured rather than assumed is the weaker and more useful statement a
conformance tier can rely on: **no conforming implementation's digest is hostage
to FTZ/DAZ on this vector.** The nearest approach is still E-12's factor of
1.32, so this remains untriggered rather than unnecessary. No digest,
coefficient, guard or required behaviour changes. See CHANGELOG.md and
docs/E40_MATVEC_RMSNORM_REACH.md.

14.2. **Digest byte encoding for artifact hashes: CLOSED.** v0.1 fed the
64-character hex **string's** ASCII bytes into the witness hash, not the
32 raw digest bytes — an ambiguity only recoverable by reading
`src/main.rs`'s `sha256_file` return type. v0.2's `sha256_file` now
returns `[u8; 32]` directly (Appendix A), and the witness chain (§12.1)
consumes those raw bytes; hex encoding is applied only at print time via a
local `hex::encode` helper. Closed by construction, not by convention —
there is no longer a hex `String` in the artifact-hash code path for the
witness chain to accidentally consume.

14.3. **Table digests not bound into `CIS2_REF`: CLOSED.** v0.1's
`table_digest` and `inv_freq_table_digest` were computed and printed
entirely independently of the witness chain — a receipt holder could not
detect, from `CIS2_REF` alone, whether a verifier used the exact pinned
polynomial/RoPE-table coefficients of §6/§7 or some other transcendental
implementation producing the same logits. v0.2 folds both digests into
the witness chain's seed (§12.1 items 1–2), ahead of the artifact hashes.
**Residual caveat**: this closes the "not bound at all" gap, but a
`CIS2_REF` mismatch still does not, by itself, tell a verifier *which* of
the now-five seed inputs (2 table digests + 3 artifact hashes) diverged —
a verifier wanting to localize a mismatch should still compare
`table_digest`/`inv_freq_table_digest`/artifact hashes individually (all
five are still printed, §12.3), not rely on `CIS2_REF` alone to diagnose
*why* it differs.

14.4. **Trig polynomial accuracy is measured, exhaustively, to `|x| <
2^31`.** §6.3's Cody-Waite reduction has been evaluated at *every* f32 bit
pattern `x` with `0 <= x < 2^31` -- 1,325,400,064 arguments, subnormals
included -- against an f64 accuracy oracle under the §1.3 pinned
environment. The worst absolute error is **9.4218e-8 for `|x| < 2^20`** and
**2.0925e-7 for `|x| < 2^31`**: 0.790 and 1.756 ULP at 1.0 respectively.
Relative (ULP) error is much larger near the zeros of `sin` and `cos` --
2617 ULP at worst -- but there the absolute error is *smaller*, by four to
five orders of magnitude, because the f32 grid is finer near zero; ULP is
not a meaningful figure of merit for these functions and absolute error is.

Because §7.1 makes `inv_freq[0]` exactly `1.0` and every later entry
smaller, the largest RoPE angle a decode evaluates is its sequence length.
The `|x| < 2^20` figure therefore covers **every RoPE angle any context up
to 1,048,576 positions can produce**, for any `rope_theta`. Direct
enumeration of the angle multiset for both §0 models at `L = 131,072`
agrees: max absolute error 9.3815e-8.

`sin_pinned` is exactly odd and `cos_pinned` exactly even *in value* over
every f32 with `2^-126 <= |x| < 2^31`. Two exceptions are documented, and
neither affects any reachable RoPE angle: on the zero/subnormal class
§1.3's DAZ makes `sin_pinned` return `+0.0` for both signs, and at three
arguments above `2^29` (620046660, 1175634300, 1240093300) the reduced
argument underflows and only the sign of a zero result fails to mirror.

Tier-1 op-level goldens pinning this behaviour at the worst-case arguments
are in `cis2-verify/src/mathpin.rs`, `mod large_angle`.

**ERRATUM E-2 (2026-09-09).** Through v0.3b as published, this clause read
"Trig polynomial accuracy is only validated for `|x| ≲ 14` ... A clean-room
implementer targeting a longer sequence than this spec's 20 positions
should not assume this polynomial's accuracy holds unchanged," carried
unchanged from v0.1. That was accurate about what had been measured and
badly misleading about what is true: the reduction does not degrade at all
across nine orders of magnitude of argument. §6.3 is unchanged; only this
limitations note was. See CHANGELOG.md, "Errata against v0.3b", and
docs/E26_TRIG_RANGE.md.

14.5. **RMSNorm multiply order (§8) is pinned, and its bit-level necessity
is measured.** `(x[i]*inv)*weight[i]` and `x[i]*(inv*weight[i])` are not
provably identical for arbitrary fp32 operands under rounding, and they in
fact differ by one ULP on about 35 % of the operand triples an actual decode
produces — 231,014 of the 667,584 RMSNorm elements in the §13.1 vector.
Substituting the other association changes the §13.1 witness digest from
`d82743059d…` to `570c0bbb0d…`. It does **not** change the argmax digest or
the generated token ids for this vector, so the violation is invisible to any
check that hashes only the model's outputs — one of the reasons §12.1 hashes
the full logit vector. The second model §0 names behaves the same way:
`Qwen/Qwen2.5-0.5B` on the same prompt and decode length diverges on 287,859
of its 834,176 RMSNorm elements (34.51 %), moves its witness
digest from `c9dff099d9…` to `4df2b260ee…`, and leaves its argmax digest and
all sixteen generated token ids unchanged. (That run supplies its prompt token
ids rather than deriving them; see §14.8.) See
`docs/E25_RMSNORM_ASSOCIATION.md`.
**ERRATUM E-1 (2026-09-09).** Through v0.3b as published, this clause read
"…its bit-level necessity is unconfirmed… no divergence between the two
orders has actually been observed on either model tested," carried unchanged
from v0.1. That was wrong: E22's M09 mutation is this exact reassociation and
had already moved the digest. §8 is unchanged; only this limitations note was.
See CHANGELOG.md, "Errata against v0.3b".

14.6. **Every intermediate activation has been compared against the oracle,
on both models. CLOSED.** §13.3's checks look only at the two ends of the
pipe --- greedy token ids and one step's full-vocab logit vector. That left a
compensating pair of errors inside the layer stack, one layer diverging and a
later one bringing the result back, outside the reach of the evidence. It is
now inside it.

Every named intermediate of the forward pass --- `embed`, and per layer
`ln1`, `q_proj`, `k_proj`, `v_proj`, `attn_out`, `o_proj`, `resid_attn`,
`ln2`, `gate_proj`, `up_proj`, `mlp_act`, `down_proj`, `resid_mlp`, and
`final_norm`/`logits` --- has been dumped at every position of a full decode
of **both** §0 models and compared against the same tensors taken from a
`transformers` fp32 forward pass by module hook:

| | SmolLM2-135M | Qwen2.5-0.5B |
|---|---|---|
| tensors compared | 7,467 | 5,985 |
| worst relative L2 error, any tensor | **2.228e-5** | **9.923e-5** |
| `embed` agreement | exactly 0 | exactly 0 |
| worst amplification of the carried-in error by any layer | **4.07x** | **3.22x** |

The last row is the one that closes the clause. A divergent layer shows its
own tensors far above the error it was handed; a compensating layer shows the
opposite. Measured, **every layer's first computed tensor is within
0.73-1.18x of the error handed to it**, on all 30 layers of the first model
and all 24 of the second, with the worst whole-layer amplification bounded by
4.07x and uniform with depth. There is no divergent layer and no compensating
layer.

The residual stream shows apparent spikes (8.79x at layer 9 of the first
model) which are **cancellation, not divergence**: there the two addends have
norms 486.5 and 414.1 and their sum has norm 105.6, so a 4.6x cancellation
inflates the relative measure by 4.6x while every input to the add sits at
1-3e-6. Evidence and method: `docs/E28_LAYER_ORACLE.md`. Instrumentation:
`cis2-verify` `--features layerdump`, plus `scripts/oracle_layers.py` and
`scripts/compare_layers.py`.

Scope, unchanged by this: agreement with `transformers` is ~1e-5 relative and
**must not** be bit-exact --- the oracle uses different kernels and a
different summation order. CIS-2's bit-exactness claim is between conforming
implementations of *this document*, not between this document and PyTorch.
This is one prompt and 19 positions on each model, so a compensating pair
that appears only at some other context length or activation pattern is not
excluded; and the Qwen run supplies its prompt token ids from outside §3
(§14.8), so it attests to §4-§11 only.

**ERRATUM E-5 (2026-09-09).** Through v0.3b as published, and unchanged in
kind since v0.1, this clause read:

> The oracle correctness checks (§13.3) are defensible spot-checks, not
> exhaustive. They confirm greedy token-id agreement and one step's
> full-vocab logit agreement to ~4e-6 relative on two model families now
> (SmolLM2-135M, Qwen2.5-0.5B) --- neither checks every intermediate layer's
> activations against the oracle, so a compensating pair of errors elsewhere
> in the layer stack that happens to preserve step-0's output and all argmax
> decisions cannot be completely ruled out by this evidence alone.
> **Unchanged in kind from v0.1 (was §14.6 there); now covers two models
> instead of one.**

That statement was correct when written, and the measurement it asked for has
now been made. Nothing normative changes: no digest, coefficient, or required
behaviour is affected. What changes is the strength of the evidence behind
§13.3, and the fact that this clause is no longer an open gap.

**ERRATUM E-9 (2026-09-09).** Three numbers above are **per-prompt maxima
quoted as general bounds**. As published they read "every layer's first
computed tensor is within **0.73-1.18x** of the error handed to it" and "the
worst whole-layer amplification bounded by **4.07x**" (SmolLM2) / **3.22x**
(Qwen). Each is the value measured on the single prompt E28 ran. Re-measuring
the identical instrument over six prompt/length cells on each model
(`docs/E36_ORACLE_PROMPT_SWEEP.md`, `docs/E37_QWEN_ORACLE_SWEEP.md`; 144,825
tensors, 312 (cell, layer) ratio rows) gives:

| | as published | measured over six cells per model |
|---|---|---|
| first-computed-tensor / carried-in ratio | 0.73-1.18x | **0.51-1.70x** |
| worst whole-layer amplification, SmolLM2 | 4.07x | **6.48x** |
| worst whole-layer amplification, Qwen | 3.22x | **4.83x** |

E28's own cell reproduces 4.07x, 3.22x and the 0.73 low end exactly, so these
are that prompt's values and not a transcription error. The table earlier in
this clause is likewise per-prompt: SmolLM2's worst relative L2 error rises to
**3.406e-5** across six prompts (E36), while Qwen's **9.923e-5** turns out to be
the six-cell maximum already.

**The conclusion of this clause is unchanged, and now rests on a better test.**
The width of the ratio interval was never the evidence; whether an extreme is a
property of the *weights* is. A divergent or compensating layer must appear at
the same layer, in the same direction, on every input. Measured, it does not:
Qwen's layer 22 hands back 0.73x the error it was given on one prompt and 1.61x
on another, and SmolLM2's layer 3 spans 0.68x to 1.48x — activation-dependent
scatter in a ratio of two small relative errors, not a layer that creates or
destroys error. The compensating pair is excluded instead by the residual-stream
test, run on all 324 (cell, layer) rows of both sweeps: a compensating pair would
show a residual spike larger than that layer's own-tensor error times its
cancellation factor, and `max resid_l2 / (own_l2 x cancellation)` is **0.848**
on SmolLM2 and **0.557** on Qwen — every row below 1, none unexplained. Read the
row above as: *no layer's own tensors depart from the error handed to it by more
than about 1.7x in either direction, and no layer's departure reproduces across
inputs.* The "one prompt and 19 positions on each model" scope note below is
correspondingly widened to six prompts and up to 67 fed positions on each; it is
still not exhaustive, and the Qwen half still attests to §4-§11 only (§14.8).
Nothing normative changes: no digest, coefficient, or required behaviour is
affected. See CHANGELOG.md, "Errata against v0.3b".

14.7. **This spec's own history.** Carried forward from v0.1: earlier
states of the reference computed `inv_freq` via unpinned host `f64::powf`,
producing a *different* `CIS2_REF`
(`830d972dbfb0b5598015f33571d08623d2b05d8e239b8d0288562d9ac9786907`) than
v0.1's `ba88708b...` (produced after switching to `exp_pinned`-based
`inv_freq`). v0.2 adds a third data point:
`a0c563ef804f50413b7fb6619ae4afe9b51b1ffa7655e944221393e85d6261da`
(§13.1, this document), produced after (a) generalizing `inv_freq` to any
`rope_theta` via `ln_pinned` and (b) the two receipt-format changes
(§12.1). The `argmax_digest` (`0b9c8f3a...`) has been unchanged across
*all three* `CIS2_REF` states — the greedy token ids have never flipped
across any of these reference-internal changes, only the full logit/receipt
bit patterns did. This is recorded here because it demonstrates, a second
time, the exact failure mode §14.3 (now closed) used to warn about: a
`CIS2_REF`-only comparison cannot localize *which* internal change moved
the digest without also comparing the finer-grained digests
individually.

14.8. **§3 admits exactly one `tokenizer.json`, and §0's second model is not
it. ERRATUM E-3 (2026-09-09).** §3.1.3 pins `normalizer: null` and §3.1.4 pins
the `pre_tokenizer` value `Sequence[Digits(individual_digits = true),
ByteLevel(add_prefix_space = false, use_regex = true)]` — the shape
`HuggingFaceTB/SmolLM2-135M` ships. `Qwen/Qwen2.5-0.5B`, which §0 names as the
second model and `docs/E15d_bc_RESULT.md` uses as evidence of correctness,
ships an NFC normalizer and `Sequence[Split(<GPT-4-style regex>, Isolated),
ByteLevel(use_regex = false)]`. A conforming implementation of §3 therefore
**must refuse that checkpoint's tokenizer**, and cannot run that model
end-to-end from its artifacts. This was always true of the text; it was never
written down, and a reader could reasonably have taken §0's second-model
sentence to mean otherwise. The scope is unchanged — §13.1 was and is the only
normative test vector, and §0 already said the Qwen tuple is not one — but the
boundary is now stated. Work on such a checkpoint is confined to §4–§11 and
must supply the prompt token ids from outside §3, saying so; a receipt produced
that way attests to §4–§11 and nothing of §3. A future version that wants a
second normative tuple has to generalize §3.1.3/§3.1.4 from a pinned literal to
a small enumerated set, with the same "refuse rather than reinterpret"
discipline for anything outside it. See `docs/E25_RMSNORM_ASSOCIATION.md` §4.1
and CHANGELOG.md, "Errata against v0.3b".

## 15. Conformance (normative)

An implementation is CIS-2 v0.2 conforming iff, from this document's text
alone (no access to `src/`):

1. It reproduces every value in §13.1 bit-for-bit, on x86-64.
2. It reproduces every value in §13.1 bit-for-bit, on aarch64, unmodified
   source, with the ISA-specific FTZ/DAZ leg of §1.3 actually exercised
   (not a no-op stub).
3. Its release binary contains zero FMA instructions on the decode path
   (§1.4, disassembly-verified).
4. Its self-test (§1.3's adversarial denormal check) passes.
5. It passes §12.4's same-process two-run determinism check.

This is a narrower and more mechanical bar than CIS-1's three-tier scheme
(CIS-1 §8) because CIS-2 has exactly one pinned (model, prompt, length)
tuple as its normative §13.1 test vector, rather than CIS-1's
op-goldens/selftest/token-digest split; a future version should factor out
op-level goldens (individual
`exp_pinned`/`sin_pinned`/`cos_pinned`/`ln_pinned`/`rsqrt`/`dot_seq` unit
vectors) as their own tier, independent of the full 30-layer decode, the
way CIS-1's Tier 1/Tier 2 do — not done in this version. **Unchanged
structure from v0.1**, item list identical; only the referenced §13.1
values changed underneath it.

## 16. Version history

- **v0.3 (2026-08-29, PARTIAL — superseded by v0.3b below)** — attempted
  to close CIS-2's one remaining stated technical limitation (v0.2 §6.7 /
  `docs/H2_TRANSCENDENTAL_RIGOR.md`): `sin_pinned`/
  `cos_pinned`'s §6.3 range reduction is now staged through `f64` instead
  of two `f32` half-constants, fixing catastrophic-cancellation accuracy
  loss that grew with RoPE position (up to 1104 ULP / 1e-4 relative by
  position 19 in v0.2; ~0.58 relative, unusable, toward position 8192).
  Design pre-registered before implementation in `docs/E15m_PREREG.md`;
  gate results (determinism, accuracy, oracle-correctness) in
  `docs/E15m_RESULT.md`.

  **What changed**: §6.3 (reduction only, not the degree-10 Taylor
  polynomials); §6.6 (`table_digest` now hashes one 8-byte `TWO_PI_F64`
  constant in place of v0.2's two 4-byte `TWO_PI_HI`/`TWO_PI_LO`
  constants); §6.7 (accuracy table re-measured over the full
  `pos ∈ [0, 8192)` domain instead of v0.2's `pos ≤ 19`).

  **What did not change**: §7 (RoPE `inv_freq` construction —
  `ln_pinned`/`exp_pinned` untouched), `inv_freq_table_digest`
  (`da9f6dcfde...`, bit-identical to v0.1/v0.2), `argmax_digest`
  (`0b9c8f3a...`) and `generated_token_ids` for the 16-token/pos≤19 pinned
  test vector (the f64-staged fix is far more accurate at short range but
  not different enough from v0.2's already-adequate short-range values to
  flip any of the 16 greedy decisions at this length).

  **Net effect on test vectors (§13.1)**: `CIS2_REF` changed
  (`a0c563ef80...` → `90f7484e4c...`); `table_digest` changed
  (`465d358ccd...` → `986abc500e...`); `inv_freq_table_digest`,
  `argmax_digest`, `generated_token_ids` unchanged from v0.2.

  Also in this branch: `.github/workflows/e15d-compiler-invariance.yml`
  and `.github/workflows/e15c-cross-isa.yml` `TARGET_DIGEST` updated to
  `90f7484e4c...` (`TARGET_INVFREQ_DIGEST` unchanged); `verify2/` and
  `verify3/` clean-rooms updated from this spec's §6.3 text only (not by
  reading `src/math.rs`), logged in each crate's own `CLEANROOM_LOG.md`;
  new `scripts/h2m_accuracy_harness.py` (mpmath oracle, pos up to 8192);
  `scripts/oracle_compare.py`/`scripts/oracle_compare_qwen.py` extended to
  a 2048-token horizon.

  **PARTIAL result (gate 2 missed)**: the accuracy harness
  (`scripts/h2m_accuracy_harness.py`) found max relative error improved
  ~55-70x (1.30e-2 -> 2.37e-4 sin, 1.36e-2 -> 3.14e-5 cos) but max ULP was
  still up to 2813 (486 filtered away from zero crossings) — the
  pre-registered <=2 ULP bar was NOT met. Root cause: the reduction itself
  became ~exact (matches an arbitrary-precision reduction to ~1 ULP of the
  reduced argument `r`); the *fixed degree-10 Taylor polynomial's own
  truncation error*, worst near `|r| ~ pi` and at sin/cos zero crossings,
  became the dominant term. Reported honestly rather than narrowing the
  claim; superseded by v0.3b below per coordinator directive.

- **v0.3b (2026-08-29)** — closes CIS-2's one remaining stated technical
  limitation, meeting the pre-registered accuracy bar v0.3 missed. Same
  pre-registration (`docs/E15m_PREREG.md`), refined mechanism: §6.3 now
  reduces to an octant (`r ∈ [-pi/4, pi/4]`, quadrant index `k mod 4`,
  still f64-staged, same Cody-Waite pattern and determinism argument as
  v0.3, just mod `pi/2` instead of mod `2*pi`), then evaluates sin(r)/
  cos(r) with SEPARATE degree-7/8 Cephes `sinf`/`cosf` minimax polynomials
  (not Taylor truncations), then selects/signs the result by quadrant.

  **What changed**: §6.3 (reduction period and polynomial, both); §6.6
  (`table_digest` now hashes one 8-byte `PI_2_F64` constant plus 6 pinned
  f32 minimax coefficients — `SIN_C0/C1/C2`, `COS_C0/C1/C2` — in place of
  v0.3's `TWO_PI_F64` plus the two 11-entry Taylor tables); §6.7 (accuracy
  table adds the v0.3b row, marked NORMATIVE, alongside v0.3's superseded
  row for comparison).

  **What did not change**: §7 (RoPE `inv_freq` construction), everything
  else in v0.3's "what did not change" list; `argmax_digest`/
  `generated_token_ids` for the 16-token pinned test vector remain
  unchanged across v0.1/v0.2/v0.3/v0.3b.

  **Net effect on test vectors (§13.1)**: `CIS2_REF` changed
  (`90f7484e4c...` -> `d82743059d...`); `table_digest` changed
  (`986abc500e...` -> `23c7bfaf5c...`); `inv_freq_table_digest`,
  `argmax_digest`, `generated_token_ids` unchanged from v0.3/v0.2/v0.1.

  **Gate (2) result**: `scripts/h2m_accuracy_harness.py` re-run to
  position 8192 on both pinned `rope_theta` values (SmolLM2-135M 1e5,
  Qwen2.5-0.5B 1e6) measures max **1.5 ULP** (both sin and cos, both
  models), max relative error ~1.2e-7 — meets the <=2 ULP bar.

  Also in this branch: `.github/workflows/e15d-compiler-invariance.yml`
  and `.github/workflows/e15c-cross-isa.yml` `TARGET_DIGEST` updated to
  `d82743059d...`; `verify2/`/`verify3/` clean-rooms updated a second time
  from this spec's v0.3b §6.3 text only, logged in each crate's own
  `CLEANROOM_LOG.md`; all three implementations (`cis2_ref`, `verify2`,
  `verify3`) confirmed bit-for-bit matching locally on x86_64
  (`d82743059d...`) — see `docs/E15m_RESULT.md`'s v0.3b section for CI
  run ids (cross-ISA confirmation pending in CI).

- **v0.1 (2026-08-28)** — first draft. Closes every item in
  `verify/SPEC_GAPS.md` (digest byte encoding, transcendental route and
  exact coefficients, RoPE `inv_freq` construction and digest, RMSNorm
  elementwise order) against the actual reference implementation at the
  commit cited in v0.1's Appendix A. Known-incomplete: v0.1 §14's items,
  especially 14.1 (theta-general RoPE) and 14.3 (table digests not bound
  into the receipt), left open for v0.2.

- **v0.2 (2026-08-28)** — three changes, all against `origin/cm/e15h-ref-fixes`
  (Appendix A):
  1. **Table digests folded into `CIS2_REF`** (closes v0.1 §14.3). §12.1
     items 1–2 (new), §6.6/§7.2 (both digests now witness-chain inputs,
     appended AFTER the three artifact hashes: the seeding order is
     weights, tokenizer, config, then table_digest, then
     inv_freq_table_digest — see §12.1 and src/main.rs:555-559.
     [E15k/H3 correction: an earlier draft of this line said "table
     digest first"; that was wrong and never matched the reference.]).
  2. **RoPE `inv_freq` is now theta-general** (closes v0.1 §14.1). New
     §6.5 pinned `ln_pinned` polynomial (Cephes-pattern logf, exact
     `frexp_exact` bit-split, degree-9 Horner in the reduced mantissa);
     §7.1's `LN_THETA` bare literal removed, replaced by
     `ln_pinned(rope_theta as f32)`; §2.4's `rope_theta==100000.0`-only
     assert removed. Validated on SmolLM2-135M (`rope_theta=100000`,
     `inv_freq_table_digest` bit-identical to v0.1) and, informatively,
     Qwen2.5-0.5B (`rope_theta=1000000`, `docs/E15d_bc_RESULT.md`).
  3. **Witness header now feeds raw 32-byte digest bytes, not 64-char
     hex-ASCII strings** (closes v0.1 §14.2/§12.1). `sha256_file()` now
     returns `[u8; 32]`; hex encoding moved to a display-only helper.

  **Net effect on test vectors (§13.1)**: `CIS2_REF` changed
  (`ba88708bf4...` → `a0c563ef80...`); `table_digest` changed
  (`0bf9257bc5...` → `465d358ccd...`, because the coefficient set it
  covers grew to include the new `ln` table — not because of the raw-bytes
  change, which affects `CIS2_REF`'s artifact-hash inputs, not
  `table_digest`'s own internal computation); `inv_freq_table_digest`
  **unchanged** (`da9f6dcfde...`, both because its own construction was
  already raw-bytes in v0.1 and because `ln_pinned(100000.0)` reproduces
  the same `inv_freq` values `LN_THETA`'s literal did, to the precision
  that matters); `argmax_digest` and `generated_token_ids` **unchanged**
  (`0b9c8f3a...`; none of the three v0.2 changes touch what the model
  actually computes, only how the receipt hashes it).

  Also in this branch (not spec content, but shipped alongside v0.2):
  the E15d(a) compiler-invariance workflow's pre-registered target updated
  from v0.1's `ba88708b...` to v0.2's `a0c563ef...`
  (`.github/workflows/e15d-compiler-invariance.yml`); a fresh
  20-cell-matrix rerun and the E15d(b)/E15d(c) cross-ISA reruns against
  this v0.2 reference are recorded in `docs/E15d_v0.2_DIGESTS.md`.

- **v0.2.1 (2026-08-28, H4)** — doc-only, no digest change. Closes H3
  BLOCKER-2 (`docs/H3_SPEC_AUDIT.md`; carried forward from
  `verify3/SPEC_GAPS_v0.2.md` items 1–2): §3.1 is fully rewritten from a
  one-paragraph "HuggingFace `tokenizers`-format, BPE-family" pointer into
  a self-contained byte-level BPE specification — exact byte→unicode map
  construction, the literal GPT-2/ByteLevel pretokenizer regex plus the
  `Digits(individual_digits=true)` pre-split this checkpoint's
  `tokenizer.json` actually configures, the rank-ordered/leftmost-tie-break
  merge algorithm, special/added-token handling under
  `add_special_tokens=false`, and a worked example deriving
  `prompt_token_ids = [6403, 1980, 253, 655]` from the literal prompt
  string. §3.1.2 pins the reference library as a normative fallback
  (`tokenizers` crate v0.23.1) for any edge case the algorithm text does
  not resolve, per this spec's existing pattern of citing external
  standards (`sha256`, §2.1). No test-vector value in §13 changed; this
  entry only makes an already-correct, already-pinned token-id list
  reproducible from spec text alone. Verified by re-fetching the exact
  `tokenizer.json` (hash-checked against §2.1/§3.1.1) and confirming the
  Python `tokenizers` binding reproduces the pinned ids.

---

## Appendix A — source citations (informative; not required to implement this spec)

Every normative choice above is taken from this repo's `src/` at the
commit checked out on branch `cm/e15i-spec-v0.2-doc` (based on
`origin/cm/e15h-ref-fixes`, commit `98f541f`). Reference implementation
state: post-E15h refactor (`docs/E15h_REFACTOR_v0.2_RESULT.md`), i.e. raw-
byte digest hashing, table-digest binding, and theta-general `ln_pinned`
all present. Citations below cover only what changed or is new versus
v0.1's Appendix A; unlisted sections (§1, §2.1–2.3/2.5, §3, §4, §5, §6.1–
6.4, §8, §9.2–9.4, §10, §11.2) are unchanged from v0.1 and cite the same
`file:line`s v0.1's own Appendix A already gives.

- §2.4 (removed restriction): v0.1's `rope_theta==100000.0` assert is gone
  from `src/main.rs`; `rope_theta` is read at `src/main.rs:331` and passed
  directly to `math::ln_pinned` at `src/main.rs:345`, with no equality
  assertion in between.
- §6.5 `ln_pinned`: `src/math.rs:272-303` (`ln_pinned`), coefficient
  consts `src/math.rs:240-254` (`LOG_SQRTHF`, `LOG_Q1`, `LOG_Q2`,
  `LOG_P[0..9]`), unit test `src/math.rs:404-428`
  (`ln_matches_std_within_tolerance`, the tested-`x` set cited in §6.5).
- §6.5.1 `frexp_exact`: `src/math.rs:256-267`.
- §6.6 table digest (extended): `src/math.rs:337-367` (`table_digest`,
  the three new `h.update(...)` lines for `LOG_SQRTHF`/`LOG_Q1`/`LOG_Q2`/
  `LOG_P` are `src/math.rs:360-365`).
- §7.1 `inv_freq` construction (generalized): `src/main.rs:334-354`
  (comment block explaining the E15d(c) generalization, then `let
  ln_theta = math::ln_pinned(rope_theta as f32);` at `src/main.rs:345`,
  the `inv_freq` loop at `:346-354`).
- §7.2 `inv_freq_table_digest` (now bound into witness, still same
  construction): `src/main.rs:358-365` (digest computation, unchanged
  from v0.1's own Appendix A citation), `src/main.rs:516`
  (`witness.update(inv_freq_table_digest)`, new call site).
- §9.1 QKV bias (new, informative for SmolLM2-135M which has none):
  `src/main.rs:90-97` (`add_bias_opt`), call sites `src/main.rs:155,157,159`.
  `LayerWeights.{q,k,v}_bias: Option<Vec<f32>>` fields `src/main.rs:35-37`;
  loader `src/main.rs:376-380` (`load_bias_opt`), populated
  `src/main.rs:407-409`.
- §11.1 untied LM head (new, informative for SmolLM2-135M which is tied):
  `Model.lm_head: Option<Vec<f32>>` field `src/main.rs:56`; load-time
  branch on `tie_word_embeddings` `src/main.rs:415-423`; consumption at
  forward time `src/main.rs:219-220`.
- §12.1 raw-byte witness header: `sha256_file` now returns `[u8; 32]`,
  `src/main.rs:245-250` (doc comment explicitly citing "CIS-2 v0.2 §12.1"
  at `:241-244`); witness seeding order `src/main.rs:555-561`
  (three artifact hashes first — weights, tokenizer, config — then
  `table_digest`, then `inv_freq_table_digest`, then prompt tokens); display-only hex helper `src/main.rs:252-261`
  (local `mod hex`).
- §13.1 test vector values: `docs/E15h_REFACTOR_v0.2_RESULT.md` (full
  run transcript, `runs_identical = true`, this branch's commit
  `f227b43`+`36b5a0f`).
- §13.4 compiler-invariance matrix: `.github/workflows/e15d-compiler-invariance.yml`
  (`TARGET_DIGEST` updated to `a0c563ef...` on `cm/e15i-spec-v0.2-doc`,
  superseding the v0.1-targeted value recorded in
  `docs/E15d_a_COMPILER_INVARIANCE.md`); results in
  `docs/E15d_v0.2_DIGESTS.md` (this branch).
- Everything else (§1 FTZ/DAZ, §2.1/2.2/2.5 artifact+config+tensors, §3
  tokenizer/prompt/decode, §4 bf16 widening, §5 reductions, §6.1–6.4
  rsqrt/exp/sin/cos, §8 RMSNorm, §9.2–9.4 attention score/softmax/V-mix,
  §10 MLP, §11.2 argmax): unchanged `file:line`s from v0.1's own Appendix
  A, reproduced there in full and not re-cited here to avoid drift between
  two documents describing the same unchanged lines.

## Appendix B — where this spec found the reference itself inconsistent or unpinned

(Every item cross-referenced to its §14 discussion above; v0.1's four
items are carried forward with their resolution status noted; no new
items were found during the v0.2 refactor beyond what v0.1 already
flagged.)

1. **RoPE `inv_freq` is theta-specific by construction: CLOSED in v0.2.**
   v0.1 flagged that `LN_THETA` was a bare literal despite surrounding code
   describing the implementation as "architecture-general... driven
   entirely from `config.json`". v0.2's `ln_pinned(rope_theta)` closes
   this specific inconsistency — the RoPE leg is now actually
   config-driven, matching the surrounding claim. Residual caveat: "closed"
   here means "no longer hardcoded to one value," not "proven correct for
   all values" — see §14.1's narrower, honest restatement. §14.1.
2. **Table digests not folded into `CIS2_REF`: CLOSED in v0.2.** §12.1
   items 1-2 now bind both digests. Residual caveat noted in §14.3: a
   `CIS2_REF` mismatch alone still does not localize *which* input
   diverged. §14.3.
3. **Digest byte encoding (hex-ASCII vs. raw bytes): CLOSED in v0.2.**
   `sha256_file` now returns `[u8; 32]`; there is no longer a hex `String`
   for the witness-chain call site to consume, so this can no longer
   silently regress to the v0.1 behavior without changing the function's
   own return type (a much harder mistake to make silently than v0.1's
   "call site happened to pick `.as_bytes()` on a hex `String`"). §14.2.
4. **`rsqrt`/`sqrt`/`/` remain the only operations in this whole spec that
   are unconditionally, provably cross-ISA-identical by the IEEE-754
   standard itself** — unchanged by v0.2; `exp`, `sin`, `cos`, and now
   `ln` are all pinned "by fiat" and have no such guarantee outside this
   document's specific coefficients. Carried forward verbatim as a
   standing caution, now covering one more transcendental (`ln`) than
   v0.1's version of this item.
