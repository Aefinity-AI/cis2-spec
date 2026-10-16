# Expected digests

All digests below are SHA-256, lowercase hex. These are the pinned values
a conforming clean-room implementation of `docs/CIS2_SPEC_v0.3b.md` (current)
MUST reproduce bit-for-bit. No timing numbers are recorded anywhere in this
repository by policy. Historical v0.2 digests are kept below for reference.

## Primary normative test vector (§13.1, spec v0.3b): SmolLM2-135M, `gen_toks=16`

Model: `HuggingFaceTB/SmolLM2-135M`, prompt `"Once upon a time"`, 16
greedy-decoded tokens, fp32 compute. Only `table_digest` and `CIS2_REF`
changed from v0.2 (RoPE range-reduction fix, see CHANGELOG.md v0.3/v0.3b);
`inv_freq_table_digest`, `argmax_digest`, and `generated_token_ids` are
unchanged from v0.1/v0.2.

```
weights_sha256          = 80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1
config_sha256            = 1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843
tokenizer_sha256          = 9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
table_digest              = 23c7bfaf5cef0095fd021af2eb1808abb4928bae4219756d86bdac670a06b35d
inv_freq_table_digest     = da9f6dcfde0425588815509e874515cdcd3d6b8818b6d0136590052e7bbf6f12
argmax_digest             = 0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff
generated_token_ids       = 28,665,436,253,1838,8180,3365,14176,30,2306,4161,281,253,2066,2291,351

CIS2_REF (witness digest) = d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df
```

This is the value `scripts/self_check.sh` builds `verify3/` and compares
against.

## Historical: spec v0.2 primary test vector (superseded)

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

Superseded by the v0.3b vector above; `scripts/self_check.sh` now targets
v0.3b's `CIS2_REF`.

**Note:** the informative scale/horizon vectors below were generated and
confirmed against spec v0.2 only; they have not been re-run against v0.3b as
part of this port. The Qwen2.5-0.5B section was re-run against v0.3b on
2026-10-16 and now lists v0.3b values; its earlier v0.2 digests are kept there
as history.

## GPU confirmation of the §13.1 normative vector

The `CIS2_REF` value above was reproduced bit-for-bit by an independent CUDA
implementation on an NVIDIA Tesla P100 (sm_60) on 2026-09-08, alongside a
byte-identical per-step trace. Full result, scope limits and provenance:
`docs/GPU_RESULT.md`. One further informative vector was pinned by that run —
the same model and prompt at a longer horizon:

```
SmolLM2-135M, "Once upon a time", gen_toks=128, spec v0.3b
CIS2_REF = 22f69ad87a615d66a77efaca8b1172d22bdfb4bd5092ceffd302e35669a050f6
```

Two further digests from that run belong to a **candidate** pinned-tree
reduction order that is not part of v0.3b and is not normative; they are
recorded in `docs/GPU_RESULT.md` only.

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

## Informative: SmolLM2-360M (second-model-scale evidence, same family/tokenizer)

Not a §13.1 test vector (that role belongs to SmolLM2-135M above); exercises
the reference construction (`src/`, not the clean-rooms) against a second,
genuinely different checkpoint in the same model family — 32 layers vs. 9,
hidden_size 960 vs. 576, `rope_theta=100000` (same as 135M) — sharing only the
tokenizer with the primary vector. Confirmed bit-identical between an AVX2
x86_64 machine and a scalar (no AVX2/FMA) x86_64 machine on 2026-09-22.

```
Model: HuggingFaceTB/SmolLM2-360M, prompt "Once upon a time", gen_toks=16, spec v0.3b, dtype=fp32

weights_sha256          = 7aaff6661428bed033abba9522bec81938678642cca3181fe752b6ca9e1e540f
config_sha256            = 34f7801487078de7e434e19162c497e5cc6ff397080e40e8586627cb68a5168a
tokenizer_sha256          = 9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
table_digest              = 23c7bfaf5cef0095fd021af2eb1808abb4928bae4219756d86bdac670a06b35d
inv_freq_table_digest     = da9f6dcfde0425588815509e874515cdcd3d6b8818b6d0136590052e7bbf6f12
argmax_digest             = 90681903ebafdc4f8d8141be882e19b95b7a2c0f0761f16fa213df196db794a8
generated_token_ids       = 28,281,253,1666,1869,2025,28,665,436,253,9077,1379,1217,260,476,61

CIS2_REF (witness digest) = 3d159b94fd7a241fcc380bb6bc158cf03a6b4c13c03f46325c8a91d75c477dad
```

Reproduced via `src/` (`cis2_ref --weights-dir <dir>`, default `--gen-toks 16`)
on:
- x86_64, AVX2, `aefinity-box` (box1) — 23.9s wall.
- x86_64, scalar (no AVX2/FMA — Intel Celeron N4020), `aefinity-box2` (box2),
  same commit, cross-machine replay — 43.2s wall.

Both machines produced the identical witness/argmax/table/inv_freq_table
digests above, byte-for-byte, in a single run per machine (not just the
in-process two-run check). `table_digest` and `inv_freq_table_digest` are
identical to the primary SmolLM2-135M vector, as expected — both are
independent of model weights (the transcendental-polynomial table is a
fixed constant, and `rope_theta=100000` is shared with 135M).

## Informative: Qwen2.5-0.5B (theta-general RoPE exercise, `rope_theta=1000000`)

Not a §13.1 test vector; exercises §7's general RoPE construction against
a second model family with a different `rope_theta` and vocab size.
Qwen2.5-0.5B, prompt `"Once upon a time"`, greedy decode, weights sha256
prefix `88c14255`, prompt-ids sha256 prefix `53ea8e67`.

**Spec v0.3b values.** Each value was produced identically by every build
that ran it (2026-10-16): the C++ host build with g++ on two AVX2+FMA hosts
(lab hosts C and D) and two SSE4.2-only hosts (lab hosts A and H), and the
CUDA sm_61 build on host C. The "tree" rows use the tree-reduction variant
(`-DCIS2_REDUCTION_TREE`), as for the SmolLM2-135M candidates in
[`docs/REPRODUCTIONS.md`](docs/REPRODUCTIONS.md). The CUDA build ran on host C
only; all other rows are host builds.

| gen_toks | variant | `CIS2_REF` digest | status |
|---|---|---|---|
| 16  | sequential | `c9dff099d927a6dda91514c260c7207de7aca4b361291e57fae390a352392e62` | reproduced on all five builds; equals the witness digest published in `docs/E28_LAYER_ORACLE.md` and `docs/E37_QWEN_ORACLE_SWEEP.md` |
| 128 | sequential | `6083b14e67813edf4507ffb1432bcb23fd4b25bf2dc30a9ab22e32244e048eb2` | informative candidate; the four host builds and the CUDA build agree |
| 16  | tree | `df2534e41eede40674885c7de2a5a448a4c32129e0b44e95b54692b9890e9a80` | informative candidate; the four host builds and the CUDA tree build agree |
| 128 | tree | `a62ed378daf4cc7f326c22659ac054251b6d20dbb976768f3f6a622626947859` | informative candidate; the four host builds and the CUDA tree build agree |

No clean-room implementation has run the 128-token or tree rows yet, so the
spec does not pin them. Run table: the Qwen2.5-0.5B section of
[`docs/REPRODUCTIONS.md`](docs/REPRODUCTIONS.md).

**v0.2-only digests (superseded).** An earlier version of this section listed
`2c9f623589520aba9cc557edf53185808086f5f6f81cab5e78ee136d0375344c` (16 tokens)
and `dccd55e68a44eae3d62e36092295b40f980a07e1be72aadf37fe14ca306b20e7`
(128 tokens). Those two values were generated against spec v0.2 only
([`docs/E15d_v0.2_DIGESTS.md`](docs/E15d_v0.2_DIGESTS.md)); they were not
re-run against v0.3b, and the v0.3b 16-token value above differs from the
v0.2 one. Treat them as v0.2 history, not as v0.3b expectations.

## Informative: SmolLM2-135M long-horizon digests (spec v0.3b, `gen_toks` 256 to 2048)

Same model, prompt, artifacts and settings as the §13.1 vector, decoded for
256, 512, 1024 and 2048 greedy tokens with the C++ reference port
(`cis2_host.cpp`: host build with g++, and the CUDA build of the same
source). These values are informative, not normative: no clean-room
implementation has run them yet, so the spec does not pin them. Each value
was produced identically by every build that ran it; the builds and the
cleaned logs are listed in the long-horizon section of
[`docs/REPRODUCTIONS.md`](docs/REPRODUCTIONS.md). The 16-token row of
every one of those runs equals the pinned `d82743059d1db929…`.

| gen_toks | `CIS2_REF` digest | builds that agree (2026-10-16) |
|---|---|---|
| 256  | `8824a22600faa57bcee3a00c8139f4ee3ddc4d1be3188ec1426aa596c3515cc2` | g++ host build on two AVX2+FMA hosts (lab hosts C and D) and on two SSE4.2-only hosts (lab hosts A and H); CUDA sm_61 (host C) |
| 512  | `d8cdafd4f5f7204b44c2ff7287cff310ecad7868006c462e1aaf67c0bf6ad4a6` | same five builds |
| 1024 | `d46c3208976dfeac0e02ef262c15a02c611fca32124c81b7bd3d077356e1471f` | same five builds |
| 2048 | `4924cacf1ad9d90ba0e90516612b296aa2dca5068c8158ecdf9d4d44f142f86a` | same five builds |

A disagreement at any of these lengths on a build that reproduces the
16-token vector would be recorded here as a divergence, not silently
dropped.

## Informative: coverage vector (tiny constructed model, `gen_toks` 8)

A 175 KB model with seeded random weights (seed 20261016), hidden 64, 2 layers,
vocab 32, 4-token prompt `3,11,22,8`, files under
[`vectors/coverage-v1/`](vectors/coverage-v1/README.md). The lm_head rows 5 and
17 are byte-identical (an exact logit tie that is the maximum at every step),
and row 29 holds bf16 subnormals (a subnormal logit unless the section 1.3
FTZ/DAZ pin flushes it). It is informative, not normative. The value below was
produced by the C++ host build with the sequential reduction order on four
hosts (two AVX2+FMA, two SSE4.2-only), and the 8 generated ids are `5` each
time. Write-up: [`docs/results/2026-10-16-COVERAGE-VECTOR.md`](docs/results/2026-10-16-COVERAGE-VECTOR.md).

| gen_toks | `CIS2_REF` digest | builds that agree (2026-10-16) |
|---|---|---|
| 8 | `4660d6c33b6c0d51529b4be43687c004c8b7c0d8804439999b5af32683528734` | g++ host build on lab hosts C and D (AVX2+FMA) and lab hosts A and H (SSE4.2 only) |

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
