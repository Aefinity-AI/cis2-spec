# Expected digests

All digests below are SHA-256, lowercase hex. These are the pinned values
a conforming clean-room implementation of `docs/CIS2_SPEC_v0.2.md` MUST
reproduce bit-for-bit. No timing numbers are recorded anywhere in this
repository by policy.

## Primary normative test vector (§13.1): SmolLM2-135M, `gen_toks=16`

Model: `HuggingFaceTB/SmolLM2-135M`, prompt `"Once upon a time"`, 16
greedy-decoded tokens, fp32 compute.

```
weights_sha256        = 80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1
config_sha256          = 1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843
tokenizer_sha256        = 9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
table_digest            = 465d358ccd63721256dbd2abbc77ad5de755adf3230635f95b12b1727dfa1ea3
inv_freq_table_digest   = da9f6dcfde0425588815509e874515cdcd3d6b8818b6d0136590052e7bbf6f12
argmax_digest           = 0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff
generated_token_ids     = 28,665,436,253,1838,8180,3365,14176,30,2306,4161,281,253,2066,2291,351

CIS2_REF (witness digest) = a0c563ef804f50413b7fb6619ae4afe9b51b1ffa7655e944221393e85d6261da
```

This is the value `scripts/self_check.sh` builds `verify3/` and compares
against.

## Informative: 128-token prompt set (SmolLM2-135M, spec v0.2)

Five prompts of varying length, `gen_toks=128` each, confirmed identical
between x86_64 and aarch64 CI runners. These are informative
cross-ISA evidence, not §13.1 normative test vectors.

| prompt_idx | prompt_toks | `CIS2_REF` digest (`gen_toks=128`) |
|---|---|---|
| 0 | 4   | `e8d4f83fe623c329e2a56acb0efb0aa11614eda168e009ffcd014da110d5ac7d` |
| 1 | 11  | `e805baec6a932ff74d933ff0b51df045a75603f5409536a5325e46ab5a7ccc05` |
| 2 | 11  | `cf1ef40e0fda59fc72218e8f7daa9935e3f15083a1d92c4cde9729d8b015b803` |
| 3 | 19  | `ee1247798e49f766adf10b64fb9cce07ee9cf80a34394e5e38c47485939c6c01` |
| 4 | 215 | `135f50b5850e9e7bf31ee4fd9b935a002c7627cce6e1b7260e665e726c2c450b` |

## Informative: Qwen2.5-0.5B (theta-general RoPE exercise, `rope_theta=1000000`)

Not a §13.1 test vector; exercises §7's general RoPE construction against
a second model family with a different `rope_theta` and vocab size.

| gen_toks | `CIS2_REF` digest |
|---|---|
| 16  | `2c9f623589520aba9cc557edf53185808086f5f6f81cab5e78ee136d0375344c` |
| 128 | `dccd55e68a44eae3d62e36092295b40f980a07e1be72aadf37fe14ca306b20e7` |

## Scale/horizon evidence: Qwen2.5-1.5B and SmolLM2-135M at longer horizons

A later hardening pass confirmed x86_64/aarch64 digest equality (not
listed here as literal hex — the source report records pass/fail
assertions, not the raw digest values) for:

- Qwen2.5-1.5B (~1.5B parameters) at `gen_toks` = 16, 128, and 512, plus a
  PyTorch fp32 oracle comparison at 16 and 512 tokens.
- SmolLM2-135M at `gen_toks=512`.

These are cited here as evidence of scale- and horizon-robustness, not as
reproducible literal digests in this document; a reproducer wanting exact
hex values for these cases should regenerate them by running the
reference construction at the stated (model, `gen_toks`) pairs and
checking x86_64 vs. aarch64 equality directly, per §13.2 of the spec.

## Sources

- Reference-implementation CI results and hardware logs (private
  repository, not included here — see README "Scope").
- `docs/E15j_CLEANROOM_C_RESULT.md`-equivalent CI run (C clean-room,
  `verify3/`), CI run id `33200155942` (post-§12.1-fix, 4/4 bit-exact on
  x86_64/aarch64 x gcc/clang).
- Rust clean-room (`verify2/`) CI run id `33194049057` (spec v0.2, both
  ISAs).
- Cross-ISA/compiler-invariance matrix CI run id `33195649908` (20/20
  cells).
- Longer-horizon/scale hardening CI run id `33214545380` (9/9 jobs).
