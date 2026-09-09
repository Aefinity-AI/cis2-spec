# Proposed for v0.4 — Tier-1 op-level goldens

**Status: PROPOSAL. Not normative. v0.3b is published and frozen; nothing in
this document changes it.** This is drop-in text for a future v0.4, plus the
evidence for why it is needed. It requires a decision, because items 6 and 7
below tighten the conformance bar: an implementation that passes v0.3b's §15
could fail v0.4's.

## 1. Why

v0.3b §15 already says the right thing and then declines to do it:

> a future version should factor out op-level goldens (individual
> `exp_pinned`/`sin_pinned`/`cos_pinned`/`ln_pinned`/`rsqrt`/`dot_seq` unit
> vectors) as their own tier … **not done in this version**.

The E22 necessity matrix turned that from a tidiness argument into a
correctness one. Two normative clauses are **provably unreachable** by the
§13.1 full-run test vector, so §15's five items cannot detect an
implementation that gets either of them wrong:

- **§6.1 `rsqrt`.** The clause's *only* stated conformance check is
  `rsqrt(64.0) == 0.125` (`0x3E000000`). 64 is a perfect square and 1/8 is
  exactly representable, so that value is one of the few in the plausible
  `head_dim` range where every plausible implementation route agrees. It
  cannot discriminate a conforming implementation from a non-conforming one.
- **§11.2 argmax**, `>` (first-maximal) vs `>=` (last-maximal). No exact tie
  occurs in 16 argmaxes over 49152 fp32 logits, so the pinned vector never
  reaches the branch the clause is about.

### 1.1 Measured: where the routes actually diverge

`head_dim` fed through §6.1's composed-f32 route (`1.0_f32 / x.sqrt()`, two
roundings in f32) against a single-rounded f64 route
(`(float)(1.0 / sqrt((double)d))`). FTZ/DAZ pinned per §1.3,
`gcc -O2 -ffp-contract=off`:

| head_dim | §6.1 route | f64 route | |
|---|---|---|---|
| 16, 32, 40, 48, 56, 64, 80, 88, 104, 120, 128, 160, 192, 256 | — | — | same |
| 24  | `0x3E5105EB` | `0x3E5105EC` | **DIFFER** (1 ULP) |
| 72  | `0x3DF15BF0` | `0x3DF15BEF` | **DIFFER** |
| 96  | `0x3DD105EB` | `0x3DD105EC` | **DIFFER** |
| 112 | `0x3DC18490` | `0x3DC1848F` | **DIFFER** |
| 136 | `0x3DAF9D54` | `0x3DAF9D53` | **DIFFER** |

96 and 112 are not synthetic corners. Both are `head_dim` values in published
checkpoints, verified from the configs themselves:

- `EleutherAI/gpt-neox-20b` — `hidden_size` 6144, `num_attention_heads` 64
  → head_dim **96**.
- MPT-30B (`jondurbin/mpt-30b-qlora-compatible`, a redistribution of the
  MosaicML config) — `d_model` 7168, `n_heads` 64 → head_dim **112**.

Those two architectures are not themselves CIS-2 targets (MPT uses ALiBi, not
§7 RoPE). The point is narrower and survives that: `head_dim` 96 and 112 are
shapes real models ship in, so a CIS-2 implementation retargeted to one of
them would silently leave the range where §6.1's single check has any power.

Scope of the two halves of that table, stated exactly. The six rows named in
§2.2's normative golden table — 24, 64, 72, 96, 112, 136 — were computed by
all three routes in §1.2 and agree bit-for-bit. The `same` row is a wider
sweep computed by route 3 (exact rational arithmetic) alone; it is included to
show that the divergences are sparse and that 64 is not the only agreeing
value, and nothing normative rests on it. `16`, `64` and `256` from that row
are additionally pinned as agreeing by route 2.

Each divergence is 1 ULP in the attention scale. That is not visibly wrong; it
is exactly the size of error that survives a smoke test and moves a digest.

### 1.2 Three independent routes agree on the pinned values

Rule B. The `§6.1 route` column was produced by, and agrees bit-for-bit
across:

1. the C reference `verify3/mathpin.c` (`cis2_rsqrt`) using hardware `sqrtf`,
   compiled `gcc -O2 -ffp-contract=off` with MXCSR FTZ|DAZ set;
2. `cis2-verify`'s **software** square root (`softfp::sqrt_cr`) — no libm, no
   hardware `sqrtf` instruction;
3. Python exact rational arithmetic, rounding once to f32.

### 1.3 Measured: §13.1 cannot exercise §1.3 either

§1.3 pins FTZ and DAZ and requires an adversarial self-test of the pin. The
self-test is a local check. The question for a *conformance suite* is whether
reproducing `CIS2_REF` proves anything about the pin. It does not.

Two measurements, both on the published §13.1 vector and the pinned
SmolLM2-135M checkpoint:

1. The checkpoint contains **0 subnormal bf16 values** out of 134,515,008
   across 272 tensors (and 0 exact zeros). DAZ can never fire on a weight.
2. An instrumented build of `cis2-verify` counted every multiply and add on
   the decode path over the full pinned run: **5,120,807,040 multiplies and
   5,118,239,304 adds, with zero flush events** in mul, add, div or exp. The
   smallest nonzero product was `|2.0876757e-14|` (bits `0x28BC0A86`) — about
   twenty-four orders of magnitude above the subnormal floor `0x00800000`. The
   instrumented build still emitted
   `d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df`, so the
   probe did not perturb the arithmetic it was measuring.

So an implementation that ignores §1.3 entirely reproduces the §13.1 digest.
The clause is not *unguarded* — §15 item 4 requires the self-test — but a
third party who reproduces `CIS2_REF` has demonstrated nothing about FTZ or
DAZ, and a suite whose only check for a clause is at an input where the clause
cannot fail is not a check.

## 2. Proposed normative text

### 2.1 Amend §6.1

After the sentence "`rsqrt(64.0) == 0.125` exactly (`0x3E000000`) is a
conformance check.", add:

> That single check is **necessary but not sufficient**: 64 is a perfect
> square and 1/8 is exactly representable, so a non-conforming route (for
> example one rounding once from f64 rather than twice in f32) reproduces it.
> An implementation MUST also reproduce §13.5's `rsqrt` goldens, which are
> chosen at inputs where the routes measurably differ.

### 2.2 New §13.5 — Op-level goldens (normative)

> **13.5. Op-level goldens (Tier 1).** These vectors are normative and are
> independent of the §13.1 full-run vector. They exist because §13.1
> provably cannot exercise the clauses below: no `head_dim` other than 64
> is fed to §6.1 in the pinned run, and no exact argmax tie occurs in it.
>
> **13.5.1 `rsqrt` (§6.1).** For each `x`, `rsqrt(x)` MUST have exactly
> these bits:
>
> | `x` | `rsqrt(x)` bits |
> |---|---|
> | `24.0`  | `0x3E5105EB` |
> | `64.0`  | `0x3E000000` |
> | `72.0`  | `0x3DF15BF0` |
> | `96.0`  | `0x3DD105EB` |
> | `112.0` | `0x3DC18490` |
> | `136.0` | `0x3DAF9D54` |
>
> An implementation that computes `1/sqrt(x)` with an f64 intermediate, or
> with a hardware reciprocal-square-root approximation (already forbidden by
> §1.6), fails at every row except `64.0`.
>
> **13.5.2 Argmax tie-break (§11.2).** For
> `logits = [1.0, 3.0, 3.0, 2.0, 3.0, -1.0]` (all exactly representable),
> §11.2 MUST return index **1**. An implementation using `>=` instead of `>`
> returns 4. A vector with no exact tie cannot distinguish the two rules,
> which is why this golden is separate from §13.1.

>
> **13.5.3 Denormal handling (§1.3).** Under the §1.3 pin, each of these
> operations MUST produce exactly `0x00000000`:
>
> | a | op | b | required result | what it tests |
> |---|---|---|---|---|
> | `0x20000000` | `*` | `0x1F800000` | `0x00000000` | product 2^-127 flushed (FTZ) |
> | `0x33800000` | `*` | `0x0C000000` | `0x00000000` | product 2^-127 flushed (FTZ) |
> | `0x00800001` | `+` | `0x80800000` | `0x00000000` | normal - normal cancels into 2^-149 (FTZ) |
> | `0x00000001` | `+` | `0x00000000` | `0x00000000` | subnormal input read as zero (DAZ/FZ) |
> | `0x00000001` | `*` | `0x3F800000` | `0x00000000` | subnormal input read as zero (DAZ/FZ) |
> | `0x00000001` | `*` | `0x7F000000` | `0x00000000` | subnormal input read as zero (DAZ/FZ) |
>
> An implementation that never pins, or that pins FTZ without DAZ, returns
> `0x00400000`, `0x00400000`, `0x00000001`, `0x00000001`, `0x00000001` and
> `0x34800000` for these six rows respectively. The last row is the loudest:
> without DAZ a full *normal* number, 2^-22, appears out of a value the pinned
> environment reads as zero.
>
> Operands MUST NOT be compile-time constants visible to the optimiser. A
> constant-folded `a * b` is evaluated by the compiler under its own rules,
> which do not honour a runtime control-register pin, and such a test reports
> the compiler's answer whether or not the pin works.

### 2.3 Amend §15

Add:

> 6. It reproduces every value in §13.5.1 bit-for-bit.
> 7. It returns §13.5.2's required index.
> 8. It reproduces every value in §13.5.3 bit-for-bit.

and replace the closing paragraph's "not done in this version" with a note
that `rsqrt` and argmax are now covered, and that
`exp_pinned`/`sin_pinned`/`cos_pinned`/`ln_pinned`/`dot_seq` goldens remain
unwritten.

## 3. What this does and does not cover

**Covered.** The two clauses E22 showed are unreachable from §13.1, plus
the spec's §1.3, which section 1.3 of this document shows is equally
unreachable.

**Not covered — still open.**

- `exp_pinned`, `sin_pinned`, `cos_pinned`, `ln_pinned`, `dot_seq` have no
  op-level goldens. §13.1 *does* exercise all of them, so they are not in the
  same category as `rsqrt`/argmax — a wrong `exp` moves `CIS2_REF`. They are
  wanted for diagnosis (which op broke), not for coverage.
- A denormal-bearing *decode* vector remains unwritten. §13.5.3 covers the
  arithmetic clause at op level; it does not produce a full-run digest that
  moves when §1.3 is ignored. Writing one means shipping a second checkpoint
  whose activations reach the subnormal range, which is a larger change than
  this proposal.

## 4. Compatibility, and the decision this needs

Adding §15 items 6 and 7 means an implementation that passes v0.3b's §15
could fail v0.4's. That is the intended effect — the whole finding is that
v0.3b's bar has two holes — but it is a real tightening and is Justin's call,
not mine.

Two known-conforming implementations already pass the proposed items:

- `cis2-verify` (Rust, clean-room, third independent implementation) pins all
  of §13.5 as tests: `mathpin.rs`, `mod op_level_goldens`. 37 tests pass.
- `verify3/mathpin.c` (C reference) reproduces §13.5.1 bit-for-bit.

So the cost of adopting this is not "re-verify the estate"; it is already
verified. The cost is that a *third-party* implementation which happened to
use an f64 route, and which passes v0.3b, would newly fail.

**Recommended:** adopt. The alternative is shipping a conformance suite whose
single check for a clause is at the one input where the clause cannot fail.

## 5. Provenance

- §13.5.3 goldens and the pinned/unpinned discrimination test:
  `cis2-verify/src/fpenv.rs`, `mod tests` (commit `3afbf66`, branch
  `cm/cis2-verify-standalone`). Run on x86_64 in both debug and release.
  The aarch64 leg is asserted by the `cis2-verify` CI job on `ubuntu-24.04-arm`.
- The two §1.3 measurements were taken with an instrumented copy of
  `cis2-verify` that is deliberately NOT in this repo (scratch only): it adds
  counters to `dot_seq`, `sum_seq`, `rmsnorm` and `softmax_seq`, which would
  otherwise be dead weight in a clean-room verifier.
- Divergence table and goldens: `cis2-verify/src/mathpin.rs`,
  `mod op_level_goldens` (branch `cm/cis2-verify-standalone`).
- Why the clauses are unreachable: `docs/E22_NECESSITY_MATRIX.md`, rows M15
  and M16, including the M16 correction — the originally published reading of
  that row was wrong, and the corrected reading is what produced this
  proposal.
- head_dim citations: `config.json` of the two Hub repos named in §1.1, read
  2026-09-09.
- No timing numbers appear in this document (Rule A).
