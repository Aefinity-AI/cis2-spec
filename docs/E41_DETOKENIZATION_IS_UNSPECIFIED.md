# E41 — CIS-2 pins a decode to token ids and never says how to read them

**Status: complete.** E40 closed the last stated gap in §1.3's reach. This
experiment starts from the opposite end: not "is the arithmetic pinned tightly
enough", but "is the specification *complete enough to build on*". The occasion
was concrete — binding an agent-episode receipt to a CIS-2 fp32 decode requires
reading the generated text to find tool calls, and `cis2-verify` had no
`decode`.

**Result.** v0.3b specifies text → ids completely (§3.1.3–§3.1.5) and
specifies ids → text nowhere. §3.1.5.a's "Decode uses the inverse map" is about
the 256-entry *byte* table only; §3.4's "Decode protocol" is greedy generation,
not detokenization; §12 hashes token **ids**. A conforming verifier never has
to produce a character, so **two conforming implementations can disagree about
what a receipt says while agreeing on every digest in it.**

Three cases make that concrete rather than pedantic, all measured on the two
checkpoints §0 names:

| case | measured | consequence |
|---|---|---|
| a codepoint straddles a token boundary | **313** of SmolLM2's 49,152 vocab entries are not valid UTF-8 alone (0.64 %) | a per-token decoder is wrong on a shipped conformance checkpoint |
| an emittable id has no string | **271** of Qwen's 151,936 logit indices are in neither `model.vocab` nor `added_tokens` | argmax can emit an id nothing maps |
| a special token is emitted | **17** of 17 SmolLM2 `added_tokens` decode to text that re-encodes to *different* ids | after rendering, a control token is indistinguishable from text spelling it |

The inverse itself is well defined in both artifacts, so this is a gap in the
specification, not a defect in the checkpoints. Recorded as **erratum E-14**.
`Tokenizer::decode` is implemented as a documented extension, with seven tests
and three killed mutants.

## 1. Establish the inverse exists before calling anything a gap

`examples/e41_detok_probe.rs` parses `tokenizer.json` with the crate's own
`json::parse` and asks whether an id → string map exists at all.

| checkpoint | vocab pairs | distinct ids | id collisions | gaps in `0..=max` | byte table distinct chars | undecodable vocab strings |
|---|---|---|---|---|---|---|
| SmolLM2-135M | 49,152 | 49,152 | **0** | **0** | 256 | **0** |
| Qwen2.5-0.5B | 151,643 | 151,643 | **0** | **0** | 256 | **0** |

Both report `INVERSE-WELL-DEFINED=true`. Had a collision existed, the right
finding would have been "the artifact is not invertible"; it does not, so the
finding is about the specification.

`added_tokens` is where the two checkpoints part company:

| checkpoint | `added_tokens` | overlap `model.vocab` | beyond `model.vocab` | content differs |
|---|---|---|---|---|
| SmolLM2-135M | 17 | **17** | 0 | 0 |
| Qwen2.5-0.5B | 22 | **0** | **22** (ids 151,643..=151,664) | 0 |

A decoder built from `model.vocab` alone is total on SmolLM2 and misses every
special token on Qwen. §3 refuses Qwen's `tokenizer.json` outright (§3.1.3 pins
`normalizer: null`, §3.1.4 pins one pre-tokenizer shape), so this does not
affect the conforming path — but it shows the choice of table is a real choice
the spec never makes.

## 2. The 271 ids nothing maps

§12.1's argmax ranges over `config.vocab_size` logits. Qwen's tied
`model.embed_tokens.weight` is `[151936, 896]`, confirmed in the safetensors
header, so the logit vector really is 151,936 wide. Mapped ids: 151,643 from
`model.vocab` plus 22 `added_tokens` = 151,665. **271 emittable ids
(151,665..=151,935) have no string in either table.**

The obvious rebuttal is that those rows are untrained padding and could never
win. Measured, that rebuttal does not hold as a *weights-level* guarantee:

| row set | count | ‖row‖ range |
|---|---|---|
| unmapped ids 151,665..=151,935 | 271 | `[3.009416e-1, 3.009682e-1]` |
| mapped ids (random sample) | 200 | min `2.999703e-1`, median `4.641488e-1`, max `6.205543e-1` |

The unmapped rows are not zero, and the smallest sampled *mapped* row is
smaller than every unmapped row. They cannot be excluded by inspecting the
weights.

What *is* true is that they lose, by a lot, on this vector.
`examples/e41_unmapped_logits.rs` walks all 16 steps of a Qwen decode and ranks
the best unmapped id in the full logit vector:

```
E41-LOGITS vocab_size=151936 mapped=151665 added_tokens=22 UNMAPPED=271
E41-LOGITS steps=16 best-unmapped-rank=115765 (at step 15)
           smallest-argmax-minus-unmapped-logit=2.0655226e1 unmapped-won-argmax=0
E41-LOGITS control=PASS (probe reports rank 1 when the set holds the argmax)
```

**Untriggered on this vector, not shown impossible** — the same standing E-12
and E-13 record for FTZ/DAZ, and stated the same way deliberately.

### 2.1 The control

"A counter that reads zero must be shown able to fire" (E35 §3). `rank=115765`
is not a zero, but the same objection applies: a rank probe that could never
return 1 would print a large number regardless. The example therefore re-runs
the identical `best_in_set` code against a set constructed to contain the
argmax at every step, and asserts it reports rank 1. That assertion is live, not
commented: with it removed the probe would still print its headline.

## 3. `Tokenizer::decode`

Three decisions, each where v0.3b is silent, each pinned by a test.

**Bytes are concatenated across the whole id sequence before UTF-8 is
validated.** Not per token — 313 of SmolLM2's 49,152 entries are fragments of a
codepoint and are not valid UTF-8 alone. `decode_bytes` and `decode` are
separate entry points for this reason.

**An id with no string is an error, not a substitution.** `DetokError::UnmappedId(id)`.
Substituting U+FFFD would make the 271-id case invisible at exactly the moment
it matters.

**A `model.vocab` mapping two strings to one id is refused in `from_json`.**
Such a vocab has no inverse; picking one string would be choosing where the
spec is silent. Neither shipped checkpoint triggers this (§1), so it costs
nothing today and refuses a malformed artifact tomorrow.

### 3.1 Verification on the real checkpoint

Not only the 2.8 kB CI fixture — `examples/e41_detok_real.rs` runs against
SmolLM2's actual 2.1 MB `tokenizer.json`:

```
E41-DETOK SmolLM2-135M vocab=49152 merges=48900
E41-DETOK SmolLM2-135M ids-decodable-to-bytes=49152 undecodable=0
           lone-token-is-valid-utf8-and-reencodes=48822 lone-token-not-utf8=313
E41-DETOK SmolLM2-135M corpus-round-trips=8/8
```

49,152 − 313 − 48,822 = **17**, which is exactly the `added_tokens` set — §4.

The normative §13.1 decode now renders. Its digests are unchanged by this work
(`witness d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df`,
`argmax 0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff`,
`conformance=PASS`), and its 16 generated ids decode to:

> `", there was a little girl named Lily. She lived in a big house with"`

with `reencode-matches=true`. The 4-token prompt likewise round-trips to
`"Once upon a time"`.

## 4. The case with a consequence: special tokens

§3.3 pins `add_special_tokens = false`, so §3 never *produces* a special-token
id. §11.2's argmax can emit one. The two facts together are not contradictory,
but their combination is unspecified and lossy.

All 17 of SmolLM2's `added_tokens` are ids 0..=16, all present in
`model.vocab`, and all decode to their own literal text. None re-encodes to
itself:

| id | decodes to | re-encodes to |
|---|---|---|
| 0 | `<\|endoftext\|>` | `[44, 108, 486, 1714, 2692, 108, 46]` |
| 1 | `<\|im_start\|>` | `[44, 108, 306, 79, 3738, 108, 46]` |
| 2 | `<\|im_end\|>` | `[44, 108, 306, 79, 486, 108, 46]` |
| 3 | `<repo_name>` | `[44, 22139, 79, 1245, 46]` |
| … | (17 total) | |

So `encode(decode(ids)) == ids` fails on exactly these 17 ids and no others in
the vocabulary — the count is measured, not assumed.

The consequence is not cosmetic. **Once a decode has been rendered to text, a
model-emitted control token is indistinguishable from ordinary generated text
that spells it.** A consumer that re-reads generated text — which is every
agent loop, including the one this work exists to build — must carry the ids,
not only the string. That is a property of the *specification's* silence: the
digest chain is over ids and is unaffected, which is precisely why the gap can
sit unnoticed behind a passing conformance run.

## 5. Tests, and the mutants that prove they can fail

`tests/detokenize.rs`, 7 tests, running in CI against synthetic fixtures and
the committed 2.8 kB tiny fixture — no 500 MB download.

Passing tests prove nothing on their own. Three mutants of the implementation,
each run against the unmodified test file:

| mutant | change | killed by |
|---|---|---|
| M1 | validate UTF-8 per token instead of on the concatenation | `a_codepoint_split_across_tokens_…` **and** `round_trip_over_a_corpus` |
| M2 | skip an unmapped id instead of erroring | `an_id_with_no_string_is_an_error_not_a_substitution` |
| M3 | drop the `from_json` collision check | `a_vocab_with_no_inverse_is_refused` |

Each mutant was killed by the intended test and no other test regressed
spuriously (M1: 5 passed / 2 failed; M2 and M3: 6 passed / 1 failed). M1's
second casualty is not incidental — the corpus contains multi-byte characters,
so a per-token validator fails there too, which is the same defect seen twice.

No mutant kills `a_special_token_decodes_to_text_that_does_not_re_encode_to_it`,
because that test asserts a property of §3.3's `add_special_tokens = false`
rather than of `decode`; its inertness guard is its own control, which round
trips an ordinary token through the same calls.

## 6. Limits

1. Two checkpoints, both from §0. The three counts (313, 271, 17) are
   properties of those artifacts, not of BPE tokenizers in general.
2. The 271-id measurement is on one prompt at gen=16. It shows those ids do not
   win there; it does not show they cannot win. No search over prompts was run.
3. The embedding-norm comparison samples 200 mapped rows, not all 151,665. It
   establishes that *some* mapped row is smaller than every unmapped row, which
   is what the argument needs, and nothing stronger.
4. `Tokenizer::decode` inverts `model.vocab` alone. That is total on SmolLM2
   and would miss Qwen's 22 special tokens — untestable through `from_json`,
   which refuses Qwen's file at §3.1.3/§3.1.4 before any of this is reached.
5. The CI tests use synthetic vocabularies for the split-codepoint, unmapped-id
   and special-token cases. The real-checkpoint runs in §3.1 and §4 are
   evidence in this document, not gates in CI, because the 2.1 MB
   `tokenizer.json` is not committed.
6. Whether v0.4 should specify detokenization at all, or declare it out of
   scope, is a decision for the spec owner. This experiment establishes that
   the current text does neither.
7. No claim is made that this gap is unusual, or that other inference
   specifications close it. It was not surveyed.

## 7. Provenance (Rule B)

* Host: `penguin`, Debian 13, x86_64, crosvm VM. **No timing figures are
  reported from this machine** (Rule A); nothing in this document is a
  measurement of time.
* `rustc 1.98.0 (88d9e12ae 2026-08-18)`, `--release`.
* Base commit: `0c8ecf5fb2132b2fc71ce4f728000279822de3d1`.
* Artifacts:
  * SmolLM2-135M `tokenizer.json` sha256
    `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`,
    `model.safetensors`
    `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`,
    `config.json`
    `1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843`.
  * Qwen2.5-0.5B `tokenizer.json` sha256
    `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`,
    `model.safetensors`
    `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`,
    `config.json`
    `479dcf0c5286339e41ad3992cd08ae88a467c4187587936248e2b7c96283484b`.
* Binaries:
  * `cis2-verify` `020505b7c6ad205aa14f4b899caabb63074e321ad55f276c6b0136c336cf957f`
  * `e41_detok_probe` `5391fd950e1840410f280605b456ede642f25c2b756d945f9f04f8024534c875`
  * `e41_detok_real` `d5c31003eeed32f5b28f3d3035f35357d42b187da55e84c53cf3810d37cdfb9a`
  * `e41_unmapped_logits` `63e22c33752613e58890a5266cd2bac80fb9ec7d2e96a1e6b2d27ecc8b01a438`
* Logs: `~/e41-probe-repo.log`, `~/e41-unmapped-qwen16-repo.log`,
  `~/e41-detok-real-smol.log`, `~/e41-ref-decode.log`.
* Gates at the point of commit: **60/60 default** (47 lib + 7 detokenize + 6
  end-to-end) and **65/65 with `census`** (52 lib + 7 + 6);
  `layerdump` builds; `check_no_fma.sh` PASS (0 FMA instructions);
  `cis2-verify selftest` PASS
  (`table_digest=23c7bfaf5cef0095fd021af2eb1808abb4928bae4219756d86bdac670a06b35d`,
  `spec_5_1_cancellation=PASS`); §13.1 reference decode `conformance=PASS`
  with digests unchanged.
