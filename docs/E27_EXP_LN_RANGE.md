# E27 — §14.1 closed: `exp_pinned` and `ln_pinned` measured at every f32, and the one place §6.2's guard is wrong

E26 closed §14.4 by replacing a sampled claim about `sin`/`cos` with an
exhaustive one. E27 is the same move for §6.2 `exp_pinned`, §6.5 `ln_pinned`
and §6.4 `silu_pinned`: **every one of the 2^32 f32 bit patterns**, evaluated
under the §1.3 pinned environment against an f64 accuracy oracle.

The headline is not the accuracy. The accuracy is boring, which is the good
outcome. The headline is that the sweep found a **defect in §6.2's high
guard** that no sampled test could have found, and a **discontinuity in
`silu_pinned`** that follows from it.

---

## 0. What §14.1 said, and what was wrong with it

v0.3b §14.1, as published:

> **`ln_pinned`'s validated domain is a finite, explicitly-tested set of `x`
> values (§6.5), not a general accuracy proof.** The two values that matter
> for this spec's models (`100_000.0`, `1_000_000.0`) are both tested to
> ≤2e-6 relative tolerance against host `f64::ln` cast to f32 … **PARTIALLY
> CLOSED**.

Three things are wrong with it.

1. **The tolerance is off by two orders of magnitude in the safe direction.**
   `ln_pinned` is not "within 2e-6" at those two thetas. It is **0 ULP** —
   bit-identical to the correctly-rounded f32 — at both.

2. **It worries about the wrong function.** `ln_pinned` is called *once per
   model load* (`model.rs:299`, on `cfg.rope_theta`). `exp_pinned` is on the
   decode hot path **twice**: `ops.rs:126`'s softmax evaluates
   `exp_pinned(*v - max_v)` for every score in every attention head at every
   step, and §6.4's `silu_pinned(x)` calls `exp_pinned(-x)` for every FFN
   intermediate. §14.1 carried a caution about the cold function and none
   about the hot one.

3. **It says nothing about the guards.** §6.2 steps 2 and 3 are unconditional
   clips. One of them is wrong, and §14.1 did not know.

---

## 1. §6.2's high guard clips 94,743 values below the representable range

§6.2 step 2 reads:

> 2. If `x > 88.0`, return `+Infinity`.

But `ln(f32::MAX) = 88.72283905206835`. Every f32 argument strictly between
88.0 and that value has an `exp` that is finite and exactly representable in
f32, and §6.2 returns `+Infinity` for all of them.

Measured, and confirmed independently by binary search in Python:

| | |
|---|---|
| count | **94,743** |
| first | `0x42B00001` = 88.00000762939453 |
| last | `0x42B17217` = 88.72283172607422 |
| `exp` at the last | ≈ 3.4028e38, i.e. just under `f32::MAX` |

This is a real defect in the spec's arithmetic, not a documentation gap.

**Is it reachable?** Only one of the two call sites is structurally safe, and
the first version of this document got that wrong.

- **Softmax cannot reach it.** `ops.rs:126` evaluates `exp_pinned(v - max_v)`
  where `max_v` is the maximum over the same vector, so the argument is always
  `≤ 0`. The high guard is unreachable from here by construction.
- **SiLU can reach it.** §6.4 evaluates `exp_pinned(-x)`. An FFN intermediate
  `x ∈ [-88.7228317, -88.0000076]` makes `-x` land inside the clip band, `exp`
  returns `+Infinity`, `denom = 1.0 + Infinity = Infinity`, and
  `silu_pinned(x) = x / Infinity = -0.0`. That is a 0.72-wide window on the
  real line — unlikely for a trained model's FFN pre-activations, but *not
  structurally excluded*, and "unlikely" is not "unreachable".

The disposition is unchanged, but the reason is narrower than "nobody gets
there": **pin the behaviour, state it as normative, and do not change it.**
Every conforming implementation performs the same clip and returns the same
`-0.0`, so bit-exact agreement — the property CIS-2 actually claims — is not
affected at all. What is affected is agreement with mathematical `silu`, and
that is what §14.1 now records. Widening the guard to `ln(f32::MAX)` would be
more faithful to `exp` and would move every published digest. E27 pins the
current behaviour with op-level goldens at both edges (`0x42B00000`,
`0x42B00001`, `0x42B17217`).

**Measured: the §13.1 reference decode does not enter the band.** "Not
structurally excluded" is not the same as "happens", and the difference is
measurable, so it was measured. `cis2-verify/src/model.rs` and
`cis2-verify/src/ops.rs` carry a read-only census behind the existing
`census` cfg feature (`ops::census::note_silu`,
`ops::census::note_softmax_exp_arg`); the default build is unchanged and the
instrumented run reproduces the pinned `witness-digest` and `argmax-digest`
exactly, which is the check that it only reads. Over the full §13.1 decode of
SmolLM2-135M (`"Once upon a time"`, 16 generated tokens):

| | |
|---|---|
| `silu_pinned` calls | 1,751,040 |
| argument range | [-25.979437, 49.15266] = [`0xC1CFD5E3`, `0x42449C53`] |
| arguments in the clip band [-88.7228317, -88.0000076] | **0** |
| arguments `< -88.0` at all | **0** |
| softmax `exp_pinned` arguments | 102,600 |
| softmax arguments hitting the low guard (`< -88.0`) | **0** |

The observed SiLU range stops 62 units short of the band on the left, so this
model on this prompt is nowhere near it.

**Extended to six inputs (E30, 2026-09-09).** One prompt is a weak census, so
the same instrumented build was run over six prompt/length configurations —
different lengths, a digit-heavy prompt, a code prompt, and a single-token
prompt with 32 generated tokens, all ASCII so that §3.1.4's open Unicode
question (§14.8) could not confound the result:

| prompt | gen | `silu_pinned` calls | SiLU argument range | in clip band | `< -88.0` | softmax `exp` args | softmax low guard |
|---|---|---|---|---|---|---|---|
| `Once upon a time` | 16 | 1,751,040 | [-25.979437, 49.15266] | **0** | **0** | 102,600 | **0** |
| `The quick brown fox jumps over the lazy dog` | 16 | 2,211,840 | [-26.987856, 49.108852] | **0** | **0** | 162,000 | **0** |
| `1234567890 + 9876543210 =` | 16 | 3,502,080 | [-22.085716, 49.034283] | **0** | **0** | 400,140 | **0** |
| `def f(x): return x * 2` | 16 | 2,304,000 | [-23.441261, 48.924583] | **0** | **0** | 175,500 | **0** |
| `Once upon a time` | 64 | 6,174,720 | [-31.406876, 49.15266] | **0** | **0** | 1,230,120 | **0** |
| `A` | 32 | 2,949,120 | [-28.012724, 49.148216] | **0** | **0** | 285,120 | **0** |

**18,892,800** SiLU calls and **2,355,480** softmax `exp_pinned` arguments,
**none** in the clip band and none below `-88.0`. The most negative SiLU
argument seen anywhere is `-31.406876`, still ~57 units short of the band's
upper edge at `-88.0000076` — and it appears in the longest run, which is the
direction the range would drift if length were the driver. The first row
reproduces the single-prompt census above exactly, which is the control.

That is six prompts on one model: it establishes that the pinned digests of
§13.1 are not affected by the clip, and it does **not** establish that no
model or input reaches it. The disposition above stands on the normative
argument, not on this measurement.

**The low guard is harmless, and the sweep proves it.** §6.2 step 3 returns
`0.0` for `x < -88.0`. `exp(-88) = 6.05e-39`, which is below
`f32::MIN_POSITIVE = 1.175e-38`, so every value the low guard destroys sits
inside the subnormal region that §1.3's FTZ flushes to zero anyway. Measured
count of arguments where the guard returned `0.0` and the oracle was nonzero:
**0**.

---

## 2. What was run

`cis2-verify/examples/exp_ln_exhaustive.rs`, six parts, all exhaustive over
the full 2^32 f32 bit patterns, 8 worker threads each calling
`fpenv::pin_and_selftest()`:

| part | what |
|---|---|
| A | `exp_pinned` at every f32, bucketed by input binade |
| B | the softmax subdomain `x ≤ 0` (`ops.rs:126`) reported separately |
| C | `ln_pinned` at every f32 — positive normals against the oracle, guard domain against the value §6.5 *requires* |
| D | `silu_pinned` at every f32 |
| E | monotonicity of `exp` and `ln` over the whole finite domain (oracle-free) |
| F | the exactness facts `exp_pinned(±0.0) == 1.0` and `ln_pinned(1.0) == +0.0` |

The program asserts `checked == 2^32` before printing totals; the run
reports `checked=4294967296`.

**Two measurement decisions, both of which matter.**

*The oracle is flushed on bits, not on comparisons.* The oracle is
`(x as f64).exp()` narrowed to f32 and then §1.3-flushed. Writing that flush
as `if v != 0.0 && v.abs() < MIN_POSITIVE` would be **wrong under DAZ**: a
subnormal operand compares equal to zero, so the branch never fires and the
flush silently does not happen. It is written on `to_bits()` instead.

*The figure of merit differs by function.* For `exp`, results span 2^-149 to
2^128, so **relative** error is the only meaningful column and absolute error
is noise. For `ln` near its zero at `x = 1`, the result is tiny, so
**absolute** error is the right column and ULP falsely reports collapse —
exactly E26's lesson, restated.

---

## 3. Result: one ULP everywhere that is not a guard

### `exp_pinned` (A, B)

```
exp ALL finite          n=3257925634  max|ulp|=1  @0x33800000  max|rel|=1.1921e-7
exp x <= 0 (softmax)    n=2139095041  max|ulp|=1  @0xB4200001  max|rel|=1.1921e-7
exp x >= 0 (silu path)  n=1118830593  max|ulp|=1  @0x33800000  max|rel|=1.1921e-7
```

Every one of the 3,257,925,634 arguments for which both `exp_pinned` and the
oracle are finite is within **one ULP**. `1.1921e-7` is `2^-23` exactly — one
ULP at 1.0. The per-binade table shows `max|ulp| = 1` in every binade from
`[0, 2^-14)` up to `[2^6, 2^7)`; there is no degradation with argument
magnitude at all.

The 2^32 arguments account exactly:

```
3,257,925,634  finite pairs
       94,743  clipped to +Inf by §6.2 step 2 (§1 above)
1,020,169,704  oracle and pinned both infinite or both zero
   16,777,214  NaN in, NaN out
            1  +Infinity in
--------------
4,294,967,296  = 2^32
```

### `ln_pinned` (C)

```
ln ALL positive normals  n=2130706432  max|ulp|=1  @0x00800186  max|abs|=7.6294e-6
ln x in [2^-1, 2^1) ZERO n=16777216    max|ulp|=1  @0x3F000841  max|abs|=5.9605e-8
ln theta=1e5  pinned=0x413834F1  ulp=0  rel=2.7531e-8
ln theta=1e6  pinned=0x415D0C55  ulp=0  rel=1.3887e-8
```

All 2,130,706,432 positive normals within one ULP. Both pinned `rope_theta`
values are **0 ULP** — this is what supersedes §14.1's "≤2e-6 relative
tolerance".

Two guard facts, both measured:

- `wrong-value-on-guard-domain = 0`. On `x ≤ 0`, NaN, and `+Inf`, `ln_pinned`
  returns exactly what §6.5 specifies, at every argument.
- `subnormals-returning--Inf = 16,777,214` = `2·(2^23 − 1)` — every subnormal
  of both signs. This is §1.3's DAZ, not a bug in §6.5: the `x == 0.0` test is
  an SSE compare, DAZ makes a subnormal operand compare equal to zero, and
  §6.5's zero branch is taken. Note that `frexp_exact` (§6.5.1) is pure bit
  manipulation on `to_bits()`, so DAZ does **not** reach it — only the guard
  comparisons.

### Monotonicity (E)

```
exp monotone: violations=0
ln  monotone: violations=0
```

Over the entire finite domain, seeded across slice boundaries so no adjacent
pair is skipped. This is an oracle-free property: it holds of the pinned
polynomial itself, independent of any reference implementation.

### Exactness (F)

```
exp(-0.0)=0x3F800000   exp(+0.0)=0x3F800000   ln(1.0)=0x00000000
```

`exp_pinned(±0.0)` is exactly `1.0`. This is load-bearing: it is why §7.1's
`inv_freq[0]` is exactly `1.0`, which is why E26's "the largest RoPE angle a
decode evaluates is its sequence length" holds, which is why §14.4's
`|x| < 2^20` bound covers a 1M-token context.

### `silu_pinned` (D)

```
silu ALL finite, x >= -88  n=3257925633  max|ulp|=2  max|rel|=2.3276e-7  max|abs|=1.9073e-6 @0x41851592
silu x < -88 (6.2 clip)    n=1020264447  max|ulp|=53824898 @0xC2B00001   max|rel|=1.0000e0
```

Two ULP everywhere the §6.2 clip is not involved — expected, since §6.4 is
`x / (1.0 + exp_pinned(-x))`, three roundings on top of `exp`'s one. From
`x = 64.0` upward `silu_pinned(x) == x` exactly.

The clip band is reported on its own line so the per-binade table states a
bound on the *polynomial*, not on the *guard*. Inside it, the 53,824,898-ULP
figure is entirely §6.2 step 3 firing on `-x > 88.0`, and it produces a
**discontinuity**:

| x | `silu_pinned(x)` | true value |
|---|---|---|
| `-88.0` (`0xC2B00000`) | `0x83354DDC` ≈ −5.328050e-37 | −5.328050e-37 |
| `-88.0000076` (`0xC2B00001`) | `-0.0` | −5.328009e-37 (`0x83354D82`) |

One ULP of *input* takes the output from −5.33e-37 to −0.0. Both figures
confirmed independently in Python (bit distance from zero of `0x83354D82` is
53,824,898 — exact match with the sweep).

**This value is not flushed.** `5.328e-37` is about 45× `f32::MIN_POSITIVE`
(`1.175e-38`) — it is a perfectly ordinary normal f32, exponent field 6, and
§1.3's FTZ never touches it. An earlier draft of this document claimed the
opposite; that was wrong, and it mattered, because it was the argument for
calling the discontinuity harmless.

The honest statement is different and still reassuring: the error is
numerically negligible (5.3e-37 in one FFN intermediate, against activations
of order 1) and, more to the point, it is *identical in every conforming
implementation*, so it costs nothing in the only currency CIS-2 trades in —
bit-exact cross-implementation agreement. It is pinned by goldens at both
sides of the boundary.

---

## 4. What now enforces this in CI

`cis2-verify/src/mathpin.rs`, `mod exp_ln_range` — 48 op-level goldens as
`(input_bits, output_bits)` pairs, routed through `black_box` in both
directions:

- `EXP_GOLDENS` (26): the worst-error arguments; ten spread across the
  reduction integer `k` at ±7, ±14, ±29, ±58, ±115; and eight guard
  boundaries including both edges of the 94,743-value clip band.
- `LN_GOLDENS` (12): worst-error arguments, both pinned thetas, `ln(1.0)`,
  `ln(f32::MIN_POSITIVE)`, `ln(f32::MAX)`, and four guard-domain entries.
- `SILU_GOLDENS` (10): worst-error arguments, the `silu(x) == x` region, and
  both sides of the §6.2 clip discontinuity.

`cis2-verify/examples/silu_reach.rs` (`--features census`) re-runs the
reachability census on demand. It is an example, not a test: it needs the §0
artifacts, so CI cannot run it, and it asserts nothing beyond printing the
counts and the two digests.

**The goldens must be structural, not coefficient-level, to be worth
anything.** §6.6's `table_digest` already binds every published coefficient
into `CIS2_REF`, so a wrong *coefficient* is already caught. What a digest
cannot see is a wrong *algorithm* with a right *table*. Two mutation controls
therefore mutate the algorithm:

| mutation | goldens moved | bar |
|---|---|---|
| `exp` with a single-constant range reduction (`r = x − k·(C1+C2)` instead of the two-part Cephes split) | **9 of 26** | `caught >= 9` |
| `ln` without the `m < LOG_SQRTHF` mantissa fixup | **5 of 12** (5 of the 8 the mutation can reach; 4 are guard-domain entries) | `caught >= 5` |

Both bars are the measured value, not an aspiration. Getting here took two
attempts: the first control mutated `EXP_P[5]` by one ULP and moved **0 of
16** goldens — a 1-ULP change to the `0.5` term is ~0.12 ULP of the result,
so a coefficient-level mutation is simply not detectable and it was
misleading to propose one. The second attempt caught 3 of 16 because the
golden table was thin at large `|k|`; adding the ten spread arguments took it
to 9 of 26.

---

## 5. Proposed §14.1 replacement

> **14.1. `exp_pinned` and `ln_pinned` accuracy is measured exhaustively, at
> every f32.** §6.2's `exp_pinned`, §6.5's `ln_pinned` and §6.4's
> `silu_pinned` have each been evaluated at **all 2^32 f32 bit patterns**
> under the §1.3 pinned environment against an f64 accuracy oracle. Outside
> §6.2's two guard bands, every one of the 3,257,925,634 comparable
> `exp_pinned` arguments and all 2,130,706,432 positive-normal `ln_pinned`
> arguments are within **one ULP** of the f64 value narrowed to f32, with no
> degradation across the argument range. Both functions are exactly monotone
> over the whole finite domain. `ln_pinned` is **0 ULP** at both pinned
> `rope_theta` values (`100_000.0`, `1_000_000.0`), superseding the "≤2e-6
> relative tolerance" figure of §6.5 and v0.3b's §14.1. `silu_pinned` is
> within **two ULP** everywhere it is not sitting on §6.2's clip band.
>
> **Two deliberate divergences from mathematical `exp`, both pinned:**
>
> (a) §6.2 step 2 clips at `x > 88.0`, but `ln(f32::MAX) = 88.7228390520684`.
> Exactly **94,743** arguments in `[0x42B00001, 0x42B17217]` =
> [88.0000076, 88.7228317] therefore return `+Infinity` where the true value
> is finite and representable. §10's softmax cannot reach this band — it
> evaluates `exp_pinned(v − max_v)` with `max_v` the maximum over the same
> vector, so its argument is always `≤ 0` — but §6.4's SiLU can: an FFN
> intermediate `x ∈ [-88.7228317, -88.0000076]` makes `exp_pinned(-x)` land
> inside it. This clip is **normative and MUST be reproduced**; a clean-room
> implementation that returns the finite value will not reproduce the pinned
> digests.
>
> (b) §6.2 step 3 returns `0.0` for `x < -88.0`. Every value *this* guard
> destroys is subnormal and would be flushed by §1.3 in any case; the measured
> count of arguments where it returned zero and the oracle was nonzero is
> **0**. The visible discontinuity in §6.4 comes from (a), not from this
> guard: `silu_pinned(-88.0)` is `0x83354DDC` (≈ −5.328e-37) while
> `silu_pinned(-88.0000076)` is `-0.0`, because `exp_pinned(88.0000076)` is
> clipped to `+Infinity`. Note that `5.328e-37` is a **normal** f32 (about 45×
> `f32::MIN_POSITIVE`), so §1.3's FTZ does not flush it. The error is
> numerically negligible and, being produced identically by every conforming
> implementation, does not affect bit-exact agreement.
>
> Under §1.3's DAZ, `ln_pinned` returns `-Infinity` for all 16,777,214
> subnormal inputs of both signs, because the `x == 0.0` guard is an SSE
> compare. This is §1.3 acting on §6.5's guard, not a property of the
> polynomial; `frexp_exact` (§6.5.1) is bit manipulation and is unaffected.
>
> Tier-1 op-level goldens pinning all of this are in
> `cis2-verify/src/mathpin.rs`, `mod exp_ln_range`.
>
> **ERRATUM E-4 (2026-09-09).** Through v0.3b as published, this clause read
> "`ln_pinned`'s validated domain is a finite, explicitly-tested set of `x`
> values (§6.5), not a general accuracy proof … **PARTIALLY CLOSED**". That
> was accurate about what had been measured and wrong in three ways about
> what is true: the tolerance was two orders of magnitude loose, the clause
> cautioned about the cold-path function while saying nothing about the
> twice-per-decode hot-path one, and it did not know that §6.2's high guard
> clips below the representable range. §6.2, §6.4 and §6.5 are unchanged;
> only this limitations note is. See CHANGELOG.md, "Errata against v0.3b",
> and docs/E27_EXP_LN_RANGE.md.

---

## 6. What this does **not** establish

- **Not correct rounding.** The claim is "never more than one ULP from the
  f64 value narrowed to f32", not "correctly rounded". The oracle is the host
  glibc `f64::exp`/`f64::ln` narrowed and §1.3-flushed; where that narrowing
  double-rounds, the oracle itself may differ from the correctly-rounded f32.
- **Not a cross-machine result.** Every number here is from `penguin`. The
  sweep is deterministic and takes no input, so replication is mechanical —
  but it has not yet been run on a second microarchitecture, and until it has,
  the possibility that some binade's worst case is microarchitecture-dependent
  is only argued (all operations are IEEE-pinned scalar SSE, no FMA — see
  `cis2-verify/tools/check_no_fma.sh`), not measured.
- **Not a statement about §6.2 being *right*.** §1's finding is that it is
  wrong, in a way that is now pinned and normative. A future spec version that
  widens the guard to `ln(f32::MAX)` would be more faithful to `exp` and would
  break every published digest.
- **Not a proof that the clip band is never entered.** It is unreachable
  through softmax by construction, and reachable in principle through SiLU
  (§1). The instrumented decode enters it **0** times out of 18,892,800 SiLU
  calls across six prompt/length configurations (§1), with the most negative
  argument seen anywhere stopping ~57 units short of the band — but that is
  one model, six ASCII prompts, and at most 64 generated tokens. No claim is
  made that a different model, prompt, or length cannot reach it.
- **Not a claim about `silu_pinned` composition.** Two ULP is measured on the
  function in isolation, not on its accumulation through an FFN.

---

## 7. Provenance (Rule B)

| | |
|---|---|
| Host | `penguin` — ChromeOS Crostini (crosvm VM), Debian 13 trixie, x86_64, 8 cores. **VM: no timing numbers reported** (Rule A). |
| Toolchain | `rustc 1.98.0 (88d9e12ae 2026-08-18)`, `--release`, `--offline` |
| FP environment | `x86_64 MXCSR FTZ(bit 15)+DAZ(bit 6)`, `fpenv::pin_and_selftest()` per worker thread |
| Repo / parent commit | `cis2-spec`, `cm/cis2-verify-standalone`, `01a2dee` |
| Sweep program | `cis2-verify/examples/exp_ln_exhaustive.rs` (571 lines, sha256 `1bb87ab29166b665784f0e92250784299d5469d32dc6f1657bde3752b9019229`) |
| Invocation | `cargo run --release --example exp_ln_exhaustive` |
| Raw log | `~/e27-explnx.log` |
| Reach census | `cargo run --release --offline --features census --example silu_reach -- ../weights`, same host and toolchain, `cis2-verify/examples/silu_reach.rs`; the instrumented build reproduced `witness-digest d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df` and `argmax-digest 0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff` unchanged |
| Reach census, six inputs (E30) | same `silu_reach` build, same host and toolchain, over `scripts/e30_prompts.txt` via `scripts/e30_prompt_sweep.sh`; raw log `~/e30-sweep.log`; each configuration's instrumented run reproduced that configuration's `witness-digest`/`argmax-digest` unchanged, and the `Once upon a time`/16 row reproduced the single-prompt values above exactly |
| Oracle | host glibc f64 `exp`/`ln`, narrowed to f32, then §1.3-flushed on bits |
| Arguments enumerated | 4,294,967,296 per function (asserted `checked == 2^32`) |
| Cross-machine replication | The same release binary (sha256 `d436ba3e2588f8e612610b5225ad2e6a67eb6312ac2e559d804a4d8790f12fd3`) was copied to `cm-box2` (Celeron, 2 cores, **no AVX2**, Debian 13, glibc 2.41-12+deb13u3 — the same glibc as `penguin`, so the same f64 oracle) and re-run. The resulting log is **byte-identical** to `~/e27-explnx.log`: all 137 lines, every `max|ulp|`, every witness bit pattern, `checked=4294967296`. Raw log `~/e27-explnx-box2.log`. This rules out microarchitectural divergence (AVX2 vs scalar dispatch); it does **not** test a second compiler or a second libm, since the binary and the glibc are the same. |
| Independent cross-checks | clip-band count and endpoints re-derived by binary search in Python; the 53,824,898-ULP silu figure and both `0x83354D..` values re-derived in Python; the 2^32 accounting summed by hand |
| Gates after this change | `cis2-verify/tools/check_no_fma.sh` PASS (0 FMA instructions); 47 lib tests + 6 end-to-end tests pass; `cis2-verify run ../weights` reproduces `witness-digest d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df`, `conformance=PASS` |

The sweep program is deterministic and takes no input; anyone with the crate
and a pinned FP environment reproduces every table above exactly.
