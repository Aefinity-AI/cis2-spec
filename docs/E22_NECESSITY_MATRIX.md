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
| **M01** | **§1.3 MXCSR FTZ/DAZ pin + self-test** | **SAME** | **not exercised** |
| **M15** | **§11 argmax tie-break** | **SAME** | **not exercised** |
| **M16** | **§9 attention scale via pinned rsqrt** | **SAME** | **not exercised** |
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
gutted; the digest did not move. The reason is that the pinned vector never produces a denormal
anywhere in the decode, so flush-to-zero has nothing to flush. The clause protects against inputs
this vector does not contain.

**M15 — argmax tie-break `>` changed to `>=`** (first maximal index → last maximal index). The digest
did not move because **no tie occurs** in any of the 16 argmaxes over a 49152-wide fp32 logit vector.
The clause is correct and necessary in general; this vector simply never reaches the branch.

**M16 — `cis2_rsqrt((float)head_dim)` replaced with `1.0f / sqrtf((float)head_dim)`.** The digest did
not move because `head_dim = 64` is a perfect square and `1/8 = 0.125` is exactly representable, so
both routes return bit-identical results. The clause would bite immediately for any head_dim whose
inverse square root is not exact.

## What this means

The conformance vector proves 12 clauses necessary and is **silent on three**. That silence is a
property of a single 4-token prompt against a single 135M checkpoint with `head_dim = 64`, not a
property of the specification.

The honest sentence for the paper and for any conformance claim is:

> A single-vector conformance pass demonstrates that an implementation agrees on the clauses that
> vector exercises. We measured which ones those are: 12 of 16 mutations are caught; §1.3, §9's
> pinned-rsqrt routing and §11's tie-break are not reachable from this vector, for reasons we can
> state exactly.

Saying "our test vector validates the spec" without this table would be an overclaim. Publishing the
table is stronger than the overclaim would have been, because it shows the suite has been attacked
by its own authors.

## Follow-up (proposed, needs Justin only if it changes v0.3b's frozen vector set)

Three targeted additions would close the gap without touching the existing pinned vector:

1. **A denormal-bearing vector** — an input whose intermediate activations reach the subnormal range,
   so §1.3 becomes observable.
2. **A tie vector** — a synthetic logit vector with two exactly-equal maxima, so §11's tie-break is
   reachable. This can be an op-level golden; it does not need a full decode.
3. **A non-perfect-square `head_dim` vector** — so §9's routing through the pinned rsqrt is
   observable. Also cheapest as an op-level golden.

All three are additive Tier-1 op-level goldens of the kind §15 already contemplates, so they extend
the suite rather than break the frozen `CIS2_REF`.

## Provenance

- Host: cm-box1 `aefinity-box`, Intel i5-5200U, gcc (Debian 14.2.0-19) 14.2.0, x86_64.
- Leg dir: `~/legs/e22-necessity-matrix` (`RESULT.txt`, `MUTATION.txt` per mutant, `mut/`, `pc-O{0,1,3,s}/`).
- No timing figures are reported from this run.
