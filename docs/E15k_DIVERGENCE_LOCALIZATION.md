# E15k — Localizing the verify3 (C) vs Rust reference witness-digest divergence

Historical development note (from the reference repository's internal research
log, branches `cm/e15k-localize-divergence` (pass 1) and
`cm/e15k2-witness-bisect` (pass 2, this doc's resolution)). No timing
numbers; CI-only (x86_64/aarch64, reference-repository CI run 33200155942).

## Pass 1 (recap): steps 0-1 bit-for-bit identical, end to end

An env-gated `CIS2_DUMP_LAYERS` side-channel dumped a SHA-256 digest per
{token embedding, each of the 30 blocks' post-attention/post-MLP hidden
states, final-norm output, full 49152-entry pre-argmax logit vector} for
both the C clean-room (`verify3/model.c`) and the Rust reference
(`src/main.rs`), at decode steps 0 and 1. All ~124 dumped tensors matched
bit-for-bit. This ruled out RMSNorm, RoPE, attention, MLP, final norm, and
the LM head matvec as the divergence source for the first two steps, but
left `CIS2_REF` itself mismatched (C=`a532d4a4...` vs ref=`a0c563ef...`)
unexplained.

## Pass 2: root cause found — witness-chain item ORDER, not the forward pass

Per-task hypothesis A (receipt encoding) was correct. Direct code
inspection of the witness-chain construction in both implementations:

- `src/main.rs` (Rust reference, `run()` closure): feeds
  `weights_sha256, tokenizer_sha256, config_sha256, table_digest,
  inv_freq_table_digest` — **in that order** — before the prompt tokens
  and per-step logit/token-id pairs.
- `verify3/model.c` (`cis2_run_decode`, pre-fix): fed
  `table_digest, inv_freq_table_digest, weights_sha256, tokenizer_sha256,
  config_sha256` — the **reverse group order**, copied verbatim from
  `docs/CIS2_SPEC_v0.2.md` §12.1's written prose.

Every one of these five 32-byte inputs was already proven bit-identical
between the two implementations (pass 1, plus separately-printed
`table_digest`/`inv_freq_table_digest`/`weights_sha256` etc. matching in
E15j). SHA-256 is order-sensitive: hashing the same five 32-byte blocks
in a different sequence produces a completely unrelated 32-byte digest,
with zero correlation to how "close" the orders are. This fully explains
the entire prior observation: every other digest (table, inv_freq,
argmax, all 16 generated token ids) matched, because none of those
depend on witness-chain item order — only `CIS2_REF` did, and it was the
only mismatching field.

**Which order is correct?** `docs/E15d_v0.2_DIGESTS.md` and the E15d(a)
20-cell compiler-invariance CI matrix (run 33195649908) both confirm the
*actual reference binary* (`src/main.rs`, unmodified) reproduces
`CIS2_REF=a0c563ef...` repeatedly, cross-compiler, cross-ISA. Since that
binary's code order is weights→tokenizer→config→table→inv_freq, **the
spec's written §12.1 prose (table→inv_freq→weights→tokenizer→config) was
itself wrong** — a documentation bug, not a reference-binary bug. The
clean-room C implementation (E15j) was in fact a *faithful, correct*
transcription of the spec text as written; the spec text just didn't
match the reference it was purportedly describing.

## Classification: class A (spec defect), NOT class B (verify3 bug)

The C code was correct relative to the document it was built from. The
defect was in `docs/CIS2_SPEC_v0.2.md` §12.1 itself: normative prose
describing witness-chain item order 1-5 did not match the reference
implementation that the same section pins a test vector against.

## Fix applied

1. `docs/CIS2_SPEC_v0.2.md` §12.1: reordered items 1-5 to
   weights_sha256, tokenizer_sha256, config_sha256, table_digest,
   inv_freq_table_digest (matching `src/main.rs`), with an inline
   correction note explaining the prior error and pointing here.
2. `verify3/model.c` `cis2_run_decode`: reordered the five
   `cis2_sha256_update` calls feeding the witness chain to match.

## Result: FULL cross-language MATCH achieved

Reference-repository CI run 33200155942 (private CI, not reproducible
externally), same 4-cell matrix as E15j (x86_64/aarch64 × gcc/clang). All 4
cells:

```
CIS2_VERIFY3 digest=a0c563ef804f50413b7fb6619ae4afe9b51b1ffa7655e944221393e85d6261da prompt_idx=0 prompt_toks=4 gen_toks=16 dtype=fp32
CIS2_VERIFY3 conformance=PASS
```

`a0c563ef804f50413b7fb6619ae4afe9b51b1ffa7655e944221393e85d6261da` is
exactly the pinned `CIS2_REF` from spec §13.1. **E15j is now a full
4/4-digest, cross-language (Rust vs from-spec-text-only C11), cross-ISA
(x86_64/aarch64), cross-compiler (gcc/clang) conformance PASS** —
`table_digest`, `inv_freq_table_digest`, `argmax_digest`, and `CIS2_REF`
(witness digest) all bit-exact on all 4 cells, up from 3/4 in E15j's
original report.
