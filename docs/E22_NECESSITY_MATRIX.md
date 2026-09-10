# E22 — Necessity matrix: which normative clauses does the v0.3b conformance vector actually prove?

Run: `e22-necessity-matrix`, cm-box1 (`aefinity-box`, i5-5200U, gcc 14.2.0),
2026-09-09T07:39:31Z → 08:11:53Z. verify3 staged from the public v0.3b tree.
Baseline reproduced `d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df`
with **0 FMA instructions in the binary**.

## Method

Sixteen single-clause mutations, each violating one numbered requirement of CIS-2 v0.3b, each built
and run against the pinned test vector. A clause is **exercised** by the vector if breaking it moves
the witness digest. A clause that can be broken with no observable effect is not thereby redundant —
it is *untested by this vector*, which is a statement about the conformance suite, not the spec.

## Result

**12 of 16 mutations moved the digest. 3 did not. 1 was declared skipped.**

| ID | Clause | Digest | Reading |
|---|---|---|---|
| M02 | §1.4 no-FMA / `-ffp-contract=off` | DIFFERS, `fma_insns=33` | exercised; the no-FMA build rule is load-bearing and mechanically checkable |
| M03, M04 | §5.1 left-to-right reduction | DIFFERS | exercised |
| M05 | §6.2 | DIFFERS (table + inv_freq moved) | exercised |
| M06 | §6.3 | DIFFERS (table moved) | exercised |
| M08 | §7.2 | DIFFERS (inv_freq moved) | exercised |
| M09 | §8 RMSNorm | DIFFERS | exercised |
| M10, M11 | §12.1 witness order | DIFFERS | exercised |
| M12 | §12.1/§14.3 | DIFFERS | exercised |
| M13 | §12.1 item 7a | DIFFERS | exercised |
| M14 | §4 bf16 widening | DIFFERS | exercised |
| **M01** | **§1.3 MXCSR FTZ/DAZ pin + self-test** | **SAME** | **exercised (E38); no digest movement** |
| **M15** | **§11 argmax tie-break** | **SAME** | **not exercised** |
| **M16** | **§9 attention scale via pinned rsqrt** | **SAME** | **VACUOUS MUTATION — see correction below** |
| M07 | §6.1 | skipped | declared SKIPPED by the leg |

Positive controls — unmutated source at four optimization levels — all reproduced the baseline:

```
-O0 / -O1 / -O3 / -Os   →  d82743059d…  SAME
```

That is a separate result worth keeping: the v0.3b reference is **optimization-level invariant** on
this compiler.

## Why the three SAME results are SAME

Each has a concrete, nameable cause. None indicates a defective clause.

**M01 — FTZ/DAZ pinning replaced with no-ops.** The pin and its abort-on-failure self-test were
gutted; the digest did not move. The reason is that the pinned vector produces no denormal that
reaches a result, so flush-to-zero has nothing to flush. The clause protects against inputs this
vector does not contain.

**M15 — argmax tie-break `>` changed to `>=`** (first maximal index → last maximal index). The digest
did not move because **no tie occurs** in any of the 16 argmaxes over a 49152-wide fp32 logit vector.
The clause is correct and necessary in general; this vector simply never reaches the branch.

**M01 — precision. UPDATED 2026-09-09 by [E38](E38_REACH_SWEEP.md), then CORRECTED the same day by
[E39](E39_FTZ_IS_NOT_WHERE_WE_SAID.md) (erratum E-12).** "SAME digest" is evidence that no denormal
*changed a result*, which is slightly weaker than "no denormal ever arose"; the mutant was not
instrumented to tell those apart.

E38 rebuilt M01 with probes that *prove the pin is really gone* (`is_pinned()==false`,
`f32::MIN_POSITIVE*0.5` reading `0x00400000` instead of `0x00000000`, `f32::from_bits(1)+0.0`
reading `0x00000001`) and ran it on the deepest cell it found — Qwen2.5-0.5B, `"Once upon a time"`,
256 generated tokens, 2 of 22,626,240 softmax `exp_pinned` arguments in `[-88.0, -87.33654)`.
Witness and argmax digests are byte-identical to the pinned build. **That measurement stands.**

E38's *reading* of it does not. It concluded "a denormal did arise, and it did not change the
receipt". `exp_pinned` ends in `ldexp_exact`, which returns exactly `+0.0` whenever the reconstructed
exponent field would be `<= 0`, so it never returns a subnormal — checked over all 2^32 arguments —
and that band is FTZ-*independent*. **No denormal arose through §6.2, and the claim is withdrawn.**

E39 pointed a counter at the operation §1.3 can actually act on, §10's `w[i] = exp_i / denom`, gave
it pinned-vs-unpinned ground truth on four boundary rows and through the real `softmax_seq`, and ran
it on the same cell: **0** subnormal quotients and **0** FTZ-decisive ones in 22,626,240, smallest
nonzero weight `1.5562307486661955e-38` — a factor of **1.32** above the subnormal boundary. So
M01's null is now *explained*: there was nothing for §1.3 to flush. That is the stronger statement
about this vector and a narrower one about §1.3, which is **untriggered here, not unnecessary** —
a third of a binade more spread in one attention row would trigger it. §1.3 therefore still has **no
end-to-end necessity witness**.
E38 also found that the FTZ half of §1.3's own self-test could not fail (erratum E-11).
This row stays **SAME**; what changes is that it is no longer *unexercised*, and that its null is
now attributed to the right cause.
It is also worth stating that this clause is **not** left unguarded by the spec: §15 item 4 requires
§1.3's adversarial self-test to pass independently of the §13.1 digest, and M01 gutted that self-test
too. A conforming implementation cannot make M01's edit and still claim conformance. The gap is in
what the *digest* can see, not in the conformance bar.

**M16 — CORRECTED 2026-09-09. The first published reading of this row was wrong.**

The mutation as executed replaced the call `cis2_rsqrt((float)head_dim)` with the expression
`1.0f / sqrtf((float)head_dim)`. Re-reading the reference source after the fact:

```c
/* verify3/mathpin.c, §6.1 */
float cis2_rsqrt(float x)
{
    return 1.0f / sqrtf(x);
}
```

The mutation substituted the function for its own body. It is a **vacuous mutation**: it could not
have moved the digest for *any* `head_dim`, perfect square or not, and it tested nothing about §9 or
§6.1. The originally published explanation — that the digest held because `head_dim = 64` is a
perfect square — was a plausible-sounding rationalisation of a result that has a much duller cause,
and the forward-looking claim that "the clause would bite immediately for any head_dim whose inverse
square root is not exact" was simply false.

**What the row should have tested, and what that would have shown.** The route that genuinely differs
from §6.1 is one computed at a different precision — `(float)(1.0 / sqrt((double)head_dim))`, a single
rounding from f64 rather than two roundings in f32. Measured, on this host in C with FTZ/DAZ pinned
and `-ffp-contract=off`, and independently reproduced by `cis2-verify`'s software square root (which
never calls a hardware `sqrtf`):

| head_dim | §6.1 route `1.0f/sqrtf(d)` | f64 route | |
|---|---|---|---|
| 16, 32, 48, 64, 80, 128, 256 | — | — | SAME |
| 24 | `0x3E5105EB` | `0x3E5105EC` | DIFFER (1 ULP) |
| 72 | `0x3DF15BF0` | `0x3DF15BEF` | DIFFER |
| 96 | `0x3DD105EB` | `0x3DD105EC` | DIFFER |
| 112 | `0x3DC18490` | `0x3DC1848F` | DIFFER |
| 136 | `0x3DAF9D54` | `0x3DAF9D53` | DIFFER |

So the "perfect square" reasoning is the right explanation for *that* mutation — 64 is one of the
values where even a differently-rounded route agrees — but it was never the explanation for the
mutation actually run.

This turns a dull row into the sharpest finding in the matrix: **§6.1's only stated conformance
check is `rsqrt(64.0) == 0.125`, and 64 is precisely a value at which every plausible implementation
route agrees.** The spec's one rsqrt check cannot discriminate between a conforming implementation
and a non-conforming one. `head_dim = 96` and `head_dim = 112` are not synthetic corners; both occur
in shipping checkpoints.

Both golden sets below are now pinned as tests in `cis2-verify/src/mathpin.rs`
(`op_level_goldens`), computed by that crate's independent software square root and matching the C
reference bit-for-bit.

## What this means

The conformance vector proves 12 clauses necessary and is **silent on two** (§1.3 and §11); the
third silent row, M16, turned out to be a vacuous mutation that tested nothing. That silence is a
property of a single 4-token prompt against a single 135M checkpoint with `head_dim = 64`, not a
property of the specification. Correcting M16 also surfaced a real defect in the spec's own
conformance surface: §6.1's single stated check is at the one value that cannot discriminate.

The honest sentence for the paper and for any conformance claim is:

> A single-vector conformance pass demonstrates that an implementation agrees on the clauses that
> vector exercises. We measured which ones those are: 12 of 16 mutations are caught; §1.3 and §11's
> tie-break are not reachable from this vector, for reasons we can state exactly; and one mutation we
> originally counted as a null result was a vacuous edit that tested nothing, which we found by
> re-reading our own source and have corrected in place.

Saying "our test vector validates the spec" without this table would be an overclaim. Publishing the
table is stronger than the overclaim would have been, because it shows the suite has been attacked
by its own authors.

## Follow-up (proposed, needs Justin only if it changes v0.3b's frozen vector set)

Three targeted additions would close the gap without touching the existing pinned vector:

1. **A denormal-bearing vector** — an input whose intermediate activations reach the subnormal range,
   so §1.3 becomes observable.
2. **A tie vector** — a synthetic logit vector with two exactly-equal maxima, so §11's tie-break is
   reachable. This can be an op-level golden; it does not need a full decode.
3. **An rsqrt route-discrimination golden** — `rsqrt` at `head_dim` ∈ {24, 72, 96, 112, 136}, the
   values where the §6.1 f32-composed route and an f64 route disagree by 1 ULP. §6.1's present
   `rsqrt(64.0) == 0.125` check cannot discriminate and must not be a suite's only rsqrt test.

Items 2 and 3 are **done**: both are pinned as tests in `cis2-verify/src/mathpin.rs`
(`op_level_goldens`), computed independently of any hardware `sqrtf` and agreeing with the C
reference bit-for-bit. Note that §15 item 4 already requires §1.3's self-test, so item 1 was a
digest-visibility gap rather than an unguarded clause.

Item 1 (a denormal-bearing decode vector) is **closed by measurement, negatively** — see
[E35](E35_DENORMAL_REACH.md). §6.2's low guard clamps `a < -88.0` before `exp` runs, so that side is
FTZ-independent; the side E35 took to be where §1.3 is digest-relevant is the unguarded band
`[-88.0, -87.33654022216797)`, which no counter had measured. (Erratum E-12: that band is
FTZ-independent too — see the M01 paragraph above.) E35 added counters for that band and
for §6.4's mirrored one, gave them positive controls that fire on constructed arguments, and swept
the same six prompts E30 used: zero of 21,248,280 `exp_pinned` evaluations land in either window,
with a nearest approach of 26.79 in ln-space. So M01's null result above is explained rather than
merely reproduced — the vector never reaches an argument where FTZ and no-FTZ differ — and a
denormal-bearing *decode* vector is not obtainable from ordinary prompts on this checkpoint.
**Superseded in part 2026-09-09: [E38](E38_REACH_SWEEP.md) obtains one** — same instrument, 20 cells
across both §0 models including non-ASCII prompts and a 256-token decode, and the Qwen 256-token cell
puts 2 arguments in that band. E35's negative result stands for SmolLM2 and for every decode of 192
tokens or fewer that was tried; what it cannot support is the general phrase "not constructible".
Item 1 is therefore **closed positively** for *reach into the band*, and the M01 re-run on that cell
(above) is what it buys. **Erratum E-12 narrows this:** reaching the band is not reaching a denormal
— `ldexp_exact` returns `+0.0` there — so item 1's real question, whether a denormal ever arises,
was still open after E38. E39 answers it for the §10 route on this vector: **no**, 0 of 22,626,240
weights, nearest miss a factor of 1.32. It remains open for §8 matvec and §7 rmsnorm intermediates,
which no counter reaches.
§1.3
stays covered at op level, where `fpenv`'s `clearing_the_pin_changes_the_answers` already shows that
clearing the pin moves every FTZ case and no inert one. The `exp` route is what E35 measures; a
denormal arising in a matvec product or an rmsnorm intermediate remains covered only by M01's
weaker "no denormal *changed a result*".

All three are additive Tier-1 op-level goldens of the kind §15 already contemplates, so they extend
the suite rather than break the frozen `CIS2_REF`.

## Provenance

- Host: cm-box1 `aefinity-box`, Intel i5-5200U, gcc (Debian 14.2.0-19) 14.2.0, x86_64.
- Leg dir: `~/legs/e22-necessity-matrix` (`RESULT.txt`, `MUTATION.txt` per mutant, `mut/`, `pc-O{0,1,3,s}/`).
- No timing figures are reported from this run.
- M16 correction and the rsqrt route table: penguin (Crostini), `gcc -O2 -ffp-contract=off` with
  MXCSR FTZ+DAZ set, cross-checked against `cis2-verify`'s `softfp::sqrt_cr` (no libm, no hardware
  `sqrtf`). Values agree across all three routes (C hardware sqrt, Rust software sqrt, and an
  exact-rational check).
