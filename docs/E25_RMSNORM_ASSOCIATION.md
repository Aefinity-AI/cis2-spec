# E25 — §14.5 closed: the RMSNorm multiply order is bit-level necessary, and we can say by how much

CIS-2 v0.3b §8 pins RMSNorm's final scale as `(x[i]*inv)*weight[i]`. §14.5 —
carried unchanged from v0.1 through v0.2 to v0.3b — recorded that the pin's
*necessity* had never been demonstrated:

> 14.5. **RMSNorm multiply order (§8) is pinned but its bit-level necessity is
> unconfirmed.** `(x[i]*inv)*weight[i]` vs. `x[i]*(inv*weight[i])` are not
> provably identical for arbitrary fp32 operands under rounding, but no
> divergence between the two orders has actually been observed on either model
> tested.

That sentence is false as of 2026-09-09. It was already false when it was last
published: **E22's M09 mutation is exactly this reassociation**, and it moved
the witness digest. §14.5 was simply never updated after E22 ran. This document
records the measurement that closes the gap, at three levels of evidence.

## 1. Scalar witnesses — the two orders differ on ordinary RMSNorm operands

Pinned as `cis2-verify/src/ops.rs::tests::orders_are_not_interchangeable`.
Five `(x, inv, weight)` triples at the magnitudes RMSNorm actually sees
(unit-ish activations, `inv` in [0.2, 5], weights near 1) for which the two
associations give different bits, and six for which they agree — the second set
kept so the test cannot be satisfied by a table that happens to diverge
everywhere. Every operand is routed through `std::hint::black_box` in both
directions, per this repository's house rule, so the answers come from the
pinned runtime FPU and not from LLVM's constant folder.

| x | inv | weight | `(x*inv)*w` | `x*(inv*w)` |
|---|---|---|---|---|
| `0x3EFF136D` | `0x409CE023` | `0x3F882890` | `0x402645A5` | `0x402645A6` |
| `0xBFB48836` | `0x3FFAD914` | `0x3F6E2B49` | `0xC02493D5` | `0xC02493D6` |
| `0x3E0DF647` | `0x3E71F232` | `0x3F8CEF0D` | `0x3D13B9C5` | `0x3D13B9C6` |
| `0xBF1F9348` | `0x3F4271B8` | `0x3F85D672` | `0xBEFD7738` | `0xBEFD7737` |
| `0x3E611015` | `0x40398E47` | `0x3F93B432` | `0x3F3C3E5D` | `0x3F3C3E5C` |

Every pair differs by exactly one ULP, in both directions. One ULP in a hidden
state is enough to change a logit, which is the reason the pin exists.

A sampled estimate, for scale: over 200,000 pseudo-random triples drawn at
those magnitudes, 69,981 — **34.99%** — diverge. The two associations are not
"almost always equal with a rare corner case"; they disagree on about a third of
ordinary inputs.

## 2. The real function — `rmsnorm` uses the pinned order and the choice is observable

Pinned as `cis2-verify/src/ops.rs::tests::rmsnorm_uses_the_pinned_order`. An
8-wide vector is pushed through `ops::rmsnorm`; `inv` is then re-derived by the
same §8 route so the *only* difference between the two candidate outputs is the
association. The test asserts that the function's output matches the pinned
association at every index and that at least one index distinguishes the two
(4 of 8 do). Mutation control: flipping `ops::rmsnorm` to the other association
fails the test at index 1 (`0xBF8C27BF` vs `0xBF8C27BE`), so the test is not
vacuous.

## 3. A full pinned decode — how many elements, and does it reach the digest?

`cis2-verify/examples/assoc_census.rs`, built with `--features census`. The
census build instruments RMSNorm to count elements and divergences, and can run
the *whole* decode in the non-conforming order so the resulting digests can be
compared. It is a measurement instrument, not a conforming verifier, and says so
on its own first line of output.

Spec §13.1 vector — SmolLM2-135M, prompt `"Once upon a time"`, `gen_toks=16`:

```
CENSUS decodes_per_run=2 (spec 12.4 determinism check)
CENSUS rmsnorm_elements_total=1335168
CENSUS rmsnorm_elements_per_decode=667584
CENSUS elements_per_decode_where_the_two_orders_differ=231014 (34.6045%)
CENSUS pinned   witness=d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df
CENSUS altorder witness=570c0bbb0dbe32b71ae1c0302db4e4d2ee2de0386dcfd309bde01579e021232c
CENSUS witness_digest_moved=true
CENSUS argmax_digest_moved=false
CENSUS tokens_moved=false
```

The per-decode count is independently derivable and checks out: SmolLM2-135M has
30 layers with two RMSNorms each plus one final norm, so 61 calls per position;
`hidden_size=576`; and the §13.1 run does 3 prefill positions plus 16 decode
steps = 19 forward passes. 61 x 576 x 19 = 667,584. (An earlier draft of this
document reported the raw counter, 1,335,168, without noting that `verify::run`
decodes twice to satisfy §12.4's same-host determinism MUST. The instrument now
prints both figures.)

Three things in that block are worth separating.

**The instrumentation does not perturb the run.** The pinned pass reproduces
`d82743059d…`, the published §13.1 witness digest, exactly. Whatever the census
build measures, it measures the conforming decode.

**The divergence rate on real operands matches the sampled estimate.**
231,014 of 667,584 RMSNorm elements — 34.60% — distinguish the two orders in a
single 16-token decode. The sampled figure in §1 was 34.99%. Nothing about the
real weight and activation distribution makes this rare.

**The two independent implementations agree even on the non-conforming path.**
The alternative-order witness digest from this Rust clean-room verifier is
`570c0bbb0dbe32b71ae1c0302db4e4d2ee2de0386dcfd309bde01579e021232c`. That is
bit-for-bit the digest E22's M09 mutant of the *C reference implementation*
produced on cm-box1 on 2026-09-09. Two implementations that share no code, on
two different hosts and two different languages, converge on the same digest for
a deliberately non-conforming association. This is a stronger statement about
specification sufficiency than the conforming case alone: the spec text is
precise enough that even a named deviation from it is reproducible across
implementations.

**But the generated tokens do not move.** `argmax_digest` and
`generated_token_ids` are identical under both associations. A user watching
only the output text would see nothing. This is the same phenomenon as the
one-flipped-weight-bit result already in `cis2-verify/tests/end_to_end.rs`, and
it is the argument for §12.1 hashing the full logit vector rather than the token
ids: **the argmax digest is blind to this violation and the witness digest is
not.**

## 4. Second model — Qwen2.5-0.5B

*(pending; §14.5's wording spoke of "either model tested", and the second model
in §0 is `Qwen/Qwen2.5-0.5B`. Result to be appended.)*

## What this changes

- **CHANGELOG erratum E-1** records the defect and the corrected §14.5 text; the
  correction is applied in place in `docs/CIS2_SPEC_v0.3b.md` with an E-1
  marker. §8 itself is unchanged. **No pinned digest moves.**
- **`cis2-verify` gains two tests** in `src/ops.rs`, both run by default CI.
- **`cis2-verify` gains a `census` feature** and one example behind it. The
  feature is off by default, the example carries `required-features`, and the
  default release binary still reproduces `d82743059d…` and still passes
  `tools/check_no_fma.sh` with 0 FMA instructions.

## What this does *not* establish

- It does not show that the pinned order is the *right* order in any numerical
  sense. Neither association is more accurate; the point of §8 is that a
  specification must choose one, and that the choice is observable.
- The 35 % figure is a property of these weights and this prompt. It is not a
  bound. A different checkpoint could plausibly sit anywhere between 0 % and
  ~50 %; what E25 rules out is the "no divergence observed" claim, not a
  particular rate.
- `tokens_moved=false` is a fact about this 16-token vector, not a guarantee.
  A longer decode could compound one-ULP hidden-state differences into a
  different argmax. E25 does not test that, and the honest reading is that the
  argmax digest *happened* not to move here, not that it cannot.

## Provenance (Rule B)

- Host: `penguin`, Debian 13 Crostini container on ChromeOS, x86_64,
  Linux 6.6.143. **This is a VM; no timing figure is reported from it, and none
  appears in this document.**
- Toolchain: `rustc 1.98.0 (88d9e12ae 2026-08-18)`, `--release`
  (`opt-level = 2`), no fast-math or reassociation flags (see
  `cis2-verify/Cargo.toml`).
- Repository: `cis2-spec`, branch `cm/cis2-verify-standalone`, parent commit
  `eb5dc11`.
- Artifacts: SmolLM2-135M, `model.safetensors`
  `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`,
  `config.json`
  `1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843`,
  `tokenizer.json`
  `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c` — all
  three identical to the digests `EXPECTED_DIGESTS.md` pins.
- Date: 2026-09-09.
- The `570c0bbb0d…` cross-check comes from E22 (`docs/E22_NECESSITY_MATRIX.md`),
  run on cm-box1 (`aefinity-box`, i5-5200U, gcc 14.2.0) 2026-09-09T07:39:31Z →
  08:11:53Z, mutant M09, whose `MUTATION.txt` is the reassociation quoted above.
- Reproduce:
  ```
  cargo test --release                                   # the two ops tests
  cargo run --release --features census --example assoc_census -- weights
  ```
