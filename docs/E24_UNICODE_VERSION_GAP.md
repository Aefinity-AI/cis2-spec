# E24 — §3.1.4 defines segmentation by Unicode property classes and pins no Unicode version

**Status:** finding, bounded. Not a reproduction failure. No published digest moves.
**Found by:** CI disagreeing with the development box, 2026-09-09.
**Provenance:** all measurements below on `penguin` (Debian 13, x86_64, a crosvm
guest) and on GitHub `ubuntu-24.04` / `ubuntu-24.04-arm` runners. No timing
number appears in this document and none may be taken from these runs (Rule A).

## 1. What happened

`cis2-verify/tools/gen_unicode.py` regenerates `src/unicode.rs` from the Unicode
Character Database and checks it byte-for-byte, so that "generated from the UCD"
is a verifiable statement rather than a comment. Its first CI run failed on both
architectures, identically:

```
MISMATCH: src/unicode.rs differs from this script's output
          (this Python carries Unicode 15.0.0).
-//! Unicode version: 15.1.0
+//! Unicode version: 15.0.0
-pub static LETTER: [(u32, u32); 660] = [
+pub static LETTER: [(u32, u32); 659] = [
-    (0x2EBF0,0x2EE5D), (0x2F800,0x2FA1D), ...
+                       (0x2F800,0x2FA1D), ...
```

Runs `34340397775` (x86_64 and aarch64 legs). The two legs failed the same way,
so this is not an ISA difference.

## 2. The exact delta

Measured by generating the three tables twice in one session, once against
`unicodedata2==15.0.0` and once against the interpreter's own database
(15.1.0), and differencing the codepoint sets:

| table | predicate | 15.0.0 | 15.1.0 | delta |
|---|---|---|---|---|
| LETTER | `\p{L}`, `char::is_alphabetic` | 136,104 cp | 136,726 cp | **+622: U+2EBF0..U+2EE5D** |
| NUMBER | `\p{N}`, `char::is_numeric` | 1,831 cp | 1,831 cp | none |
| WHITESPACE | `\s` (White_Space) | 25 cp | 25 cp | none |

The added block is CJK Unified Ideographs Extension I, introduced in Unicode
15.1. It is the entire difference the pre-tokenizer can see between those two
UCD releases.

## 3. The spec gap

v0.3b §3.1.4 specifies the pre-tokenizer as the GPT-2 regex

```
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

"applied with Unicode property classes", and specifies the Digits stage by
`char::is_numeric`. Neither the clause nor any other part of the document names
a Unicode version. The spec pins the model, the tokenizer JSON, the floating
point environment, the reduction order and the argmax rule — but the meaning of
`\p{L}` is left to whatever UCD the implementor's regex engine or standard
library happens to carry.

Two implementations that follow §3.1.4 exactly, built against different UCD
releases, therefore segment some inputs differently. That is a
specification-sufficiency defect of the same shape as the ones E22 found,
reached from a different direction.

## 4. How far the damage actually goes — measured, not assumed

**No published digest moves.** The §13.1 prompt is `"Once upon a time"`, pure
ASCII, far from any codepoint whose classification changed. Every reproduction
claim in the estate stands unaltered.

**Stronger, and specific to this checkpoint: for the pinned SmolLM2-135M
vocabulary the delta cannot change a token id at all.** Two measurements:

1. Encoding `"𮯰"`, `"x𮯰y"`, `" 𮯰𮯱"` and `"𮯰1"` with `cis2-verify`'s
   tokenizer, once with the committed 15.1.0 LETTER table and once with that
   one range deleted (i.e. the 15.0.0 table), gives **identical ids in all four
   cases** — `[187,123,124,125]`, `[104,187,123,124,125,105]`,
   `[216,187,123,124,125,187,123,124,126]`, `[187,123,124,125]`. Segment
   boundaries move; ids do not.

2. The reason is in the merge table, and it generalises across the whole added
   block. Every codepoint in U+2EBF0..U+2EE5D encodes as UTF-8 `F0 AE|AF xx`,
   which the §3.1.5.a byte→unicode map sends to `ð` followed by characters in
   the `®`/`¯`/`°` region. Of the checkpoint's 48,900 merges, exactly three
   involve `ð`: `('ð','Ł')`, `('ð','Ŀ')` and `('Ġ','ðŁ')` — that is, `F0 9F`
   (emoji) and `F0 9D` (mathematical alphanumerics), preceded optionally by a
   space. None of them can fire on `F0 AE` or `F0 AF`. Since no merge spans the
   boundary either side, the boundary's position is unobservable in the output.

So for this checkpoint the exposure at the 15.0 → 15.1 step is nil. It is nil
**by property of the vocabulary**, not by anything the specification says. A
future UCD release that reclassifies a codepoint whose bytes *do* participate in
merges would diverge, and nothing in v0.3b would have warned an implementor.

## 5. What was fixed in this commit range

Only the checkability of the generated file:

- `gen_unicode.py` declares `REQUIRED_UCD = "15.1.0"`, prefers the standalone
  `unicodedata2` package over the interpreter's database, and exits 2 with an
  explanation rather than diffing tables drawn from two different Unicode
  versions — which would test the runner's Python, not this repository.
- The `cis2-verify` CI job installs `unicodedata2==15.1.0` before running it.

## 6. What is proposed for v0.4, and is Justin's call

Amend §3.1.4 to pin the database, normatively:

> The property classes `\p{L}`, `\p{N}` and `\s` in the pre-tokenizer regex, and
> the `char::is_numeric` predicate of the Digits stage, are evaluated against
> **Unicode 15.1.0**. `\p{L}` is General_Category in {Lu, Ll, Lt, Lm, Lo};
> `\p{N}` is General_Category in {Nd, Nl, No}; `\s` is the White_Space property,
> which is 25 codepoints and does **not** include U+001C..U+001F. An
> implementation whose regex engine or standard library carries a different
> Unicode version MUST supply these three sets explicitly rather than defer to
> the host.

and add to §15:

> 9. Its `\p{L}`, `\p{N}` and `\s` sets are those of the Unicode version named
>    in §3.1.4.

The cost of adopting is small and is already paid inside the estate: the
reference implementation, `verify2`, `verify3` and `cis2-verify` all carry
explicit tables or a pinned engine. The cost falls on a third-party
implementation that defers to a host regex engine — which is precisely the
implementation the clause needs to catch.

## 7. Wording discipline

This is not a claim that anyone's reproduction was wrong, and not a claim that
the specification is unsound. It is a claim that one clause is
under-determined, that the under-determination is presently harmless for the
pinned artifacts, and that it is harmless for a reason the specification does
not state.
