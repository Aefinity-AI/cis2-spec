# E26 — §14.4 closed: the pinned trig reduction is good to under 1 ULP-of-unity across every angle a 1M-token context can produce

**Date:** 2026-09-09
**Repo:** `cis2-spec`, branch `cm/cis2-verify-standalone`, parent `be1f4df`
**Artifacts:** `cis2-verify/examples/trig_exhaustive.rs`, goldens in `cis2-verify/src/mathpin.rs::large_angle`

---

## 0. What §14.4 said, and what was actually wrong with it

v0.3b §14.4, carried unchanged from v0.1:

> **Trig polynomial accuracy is only validated for `|x| ≲ 14`** (unit tests
> sweep `x = i * 0.7` for `i` in `-20..=20`). […] the two-part-π reduction
> (§6.3) has not been stress-tested at larger magnitudes where it could lose
> more precision. A clean-room implementer targeting a longer sequence than
> this spec's 20 positions should not assume this polynomial's accuracy holds
> unchanged.

Every sentence of that is *true as written* — it is a statement about what had
been measured, not a claim about the algorithm. The problem is that it is the
only thing the specification says about the range of §6.3, and a clean-room
implementer reading it has to conclude that the pinned reduction is unproven
outside a 20-position decode. It is the clause that would stop anyone from
using CIS-2 on a real context length.

It is also five orders of magnitude too pessimistic, which this note now shows
by measurement rather than by argument.

Why the range question is exactly the sequence-length question: §7.1 sets
`inv_freq[i] = exp_pinned(-((2i/head_dim) * ln_pinned(theta)))`, so
`inv_freq[0] = exp_pinned(-0.0) = 1.0` exactly and every later entry is
smaller. The RoPE angle at position `pos` is `pos * inv_freq[i]`, so the
largest angle a decode ever evaluates **is its sequence length**. "Does §6.3
hold at `|x| < 2^20`" and "does CIS-2 hold for a 1,048,576-token context" are
the same question.

---

## 1. The first answer was wrong, and the way it was wrong is the point

The first attempt (`examples/trig_range.rs`, kept in the tree, now marked
SUPERSEDED) sampled 4096 arguments per binade, scattered by a golden-ratio
stride, and reported:

```
[2^0 .. 2^16)   max|ulp| sin = 1,  cos = 1   (one 2-ulp cos at [2^13,2^14))
```

That reads as "the reduction is essentially perfect out to a 64k context",
and it is not what an exhaustive sweep of the same range finds:

```
TRIG-X EXHAUSTIVE |x|<2^16 max|ulp| sin=66 (at 4.4767695e3)  cos=2283 (at 5.2516434e4)
```

4096 samples out of 8,388,608 per binade simply never landed near a zero of
`sin` or `cos`, where the result is tiny and a fixed absolute error is a large
number of ULP. The sampled conclusion was not a weaker version of the truth;
it was a different and wrong statement about the shape of the error. Nothing
that licenses a MUST should be established by sampling when the domain is
finite and enumerable — and for a single-argument f32 function it always is.

So the sampler was thrown away and the whole domain was enumerated.

---

## 2. What was run

`cis2-verify/examples/trig_exhaustive.rs` does three things under the §1.3
pinned FP environment (MXCSR FTZ+DAZ; each worker thread pins for itself,
because MXCSR is per-thread state and an unpinned worker is not running the
spec's arithmetic):

* **(A) Symmetry.** For every argument, compare `sin_pinned(-x)` against
  `-sin_pinned(x)` and `cos_pinned(-x)` against `cos_pinned(x)`, on bits.
  This is what turns a positive-only sweep into a statement about every
  finite f32, and it needs no oracle.
* **(B) Accuracy.** For every f32 bit pattern `x` with `0 <= x < 2^E`,
  compare against `(x as f64).sin()` / `.cos()` rounded to f32, recording both
  the ULP distance and the absolute error, bucketed by binade.
* **(C) The RoPE angle multiset.** Enumerate `(pos as f32) * inv_freq[i]` for
  every `pos < 131,072` and every `i`, for both models §0 pins, with
  `inv_freq` built by the crate's own §7.1 routine — not by a formula
  retyped into the example.

(B) subsumes (C) for any context up to `2^E`, per §0's `inv_freq[0] == 1.0`
argument. (C) is run anyway because it is the claim a reader actually wants
and it costs nothing.

**On the oracle.** `(x as f64).sin()` is an *accuracy* oracle, never a
conformance one. The widening f32→f64 cast is exact, and the host libm's f64
`sin` carries a full Payne–Hanek reduction with ~1e-16 relative error — nine
orders of magnitude below an f32 ULP. Nothing pinned depends on it, and a
disagreement here is a claim about accuracy, never about conformance. The
double rounding (f64 result → f32) can shift a reported ULP count by one; it
cannot touch the absolute-error column, which is where every conclusion below
is drawn.

---

## 3. Result: absolute error is flat, ULP error is not, and only the first is a statement about the reduction

Every f32 in `[0, 2^20)` — **1,233,125,376 arguments**, subnormals included:

| range | max \|ulp\| sin | max \|ulp\| cos | max \|abs\| sin | max \|abs\| cos |
|---|---|---|---|---|
| `[0, 2^-14)`   | 0 | 0 | 3.7896e-14 | 1.8626e-9 |
| `[2^-4, 2^-3)` | 1 | 1 | 3.9499e-9 | 5.9773e-8 |
| `[2^0, 2^1)`   | 1 | 1 | 7.4877e-8 | 5.0429e-8 |
| `[2^4, 2^5)`   | 2 | 2 | 9.2556e-8 | 8.6598e-8 |
| `[2^8, 2^9)`   | 34 | 4 | 9.2407e-8 | 9.2457e-8 |
| `[2^12, 2^13)` | 66 | 50 | 9.2392e-8 | 9.2307e-8 |
| `[2^15, 2^16)` | 66 | 2283 | 9.3143e-8 | 9.3118e-8 |
| `[2^19, 2^20)` | 2283 | 2617 | 9.2719e-8 | 9.3223e-8 |
| **overall** | **2283** | **2617** | **9.3815e-8** (at 2100.902) | **9.4218e-8** (at 130659.61) |

(Full 37-row table in `cis2-verify/examples/trig_exhaustive.rs` output; the
rows above are every fourth binade plus the extrema.)

The two columns tell opposite stories and only one of them is about §6.3:

* **Absolute error is flat at ≤ 9.4218e-8 across nine orders of magnitude of
  argument.** An f32 ULP at 1.0 is `2^-23 = 1.1920929e-7`, so the worst case
  over the entire range is **0.790 ULP-of-unity** — below the 1-ULP spacing of
  the f32 grid the answer has to land on. The reduction does not degrade at
  all across `[0, 2^20)`.
* **ULP error grows to 2617 and is not a statement about the reduction.** It
  is large exactly where `sin` or `cos` is near a zero, and there the
  *absolute* error is orders of magnitude *below* the flat figure, because the
  f32 grid is that much finer near zero. The two columns peak at disjoint
  arguments:

  | x | function | pinned | f64 oracle | \|ulp\| | \|abs\| |
  |---|---|---|---|---|---|
  | 801176.8  | cos | 1.0302756e-7   | 1.0304615583947203e-7  | **2617** | 1.8596e-11 |
  | 105032.87 | sin | -3.245077e-8   | -3.244265847844376e-8  | **2283** | 8.1125e-12 |
  | 52516.434 | cos | -1.6225385e-8  | -1.6221329239221883e-8 | **2283** | 4.0562e-12 |
  | 4476.7695 | sin | 1.15454895e-7  | 1.1545536480926656e-7  | **66**   | 4.6939e-13 |
  | 130659.61 | cos | 7.1727526e-1   | 7.172753560966018e-1   | 2        | **9.4218e-8** |
  | 2100.902  | sin | 7.334515e-1    | 7.334513918188813e-1   | 2        | **9.3815e-8** |

  At the worst ULP point in the whole `[0, 2^20)` sweep the absolute error is
  1.86e-11 — five thousand times smaller than the worst case, which occurs
  where the result is of order 1 and the ULP count is 2. A relative-error
  figure of merit is the wrong instrument for a function with zeros, and
  quoting the ULP column alone would report that the reduction falls apart at
  large angles when the opposite is true.

Carrying the same sweep to the whole positive f32 range below `2^31`
(1,325,400,064 arguments) shows where it eventually does give way, and even
that is gentle:

```
[2^20, 2^25)    max|abs| ~9.4e-8    (unchanged)
[2^25, 2^26)    max|abs|  9.5085e-8
[2^27, 2^28)    max|abs|  1.0775e-7
[2^29, 2^30)    max|abs|  1.3584e-7
[2^30, 2^31)    max|abs|  2.0925e-7   <- worst over the whole range
```

Worst case anywhere below `2^31`: **2.0925e-7, i.e. 1.756 ULP-of-unity.** The
onset is where it should be: `khi = k * (π/2 as f64)` carries a rounding error
of about `k * 2^-52 * (π/2)`, which reaches the f32 rounding scale `2^-24`
around `k ≈ 2^28`, i.e. `x ≈ 2^28.7`. Measurement and arithmetic agree.

### The RoPE angle sets, enumerated directly

```
ROPE SmolLM2-135M head_dim=64 theta=1e5 L=131072 angles=4194304 inv_freq[0]=1
     max_angle=1.31071e5  max|ulp| sin=12 cos=12   max|abs| sin=9.3815e-8 cos=9.2446e-8
ROPE Qwen2.5-0.5B head_dim=64 theta=1e6 L=131072 angles=4194304 inv_freq[0]=1
     max_angle=1.31071e5  max|ulp| sin=12 cos=252  max|abs| sin=9.1375e-8 cos=9.2446e-8
```

Both models, at a 131,072-position context — 64× longer than SmolLM2-135M's
training length and 16× the spec's own 20-position decode — sit at the same
9.4e-8 absolute error as everything else. Note `inv_freq[0]` printing as
exactly `1`, which is the identity that makes (B) cover (C).

---

## 4. Symmetry: exact, with two measured exceptions

Over every f32 with `2^-126 <= x < 2^31`:

```
symmetry, |x| >= 2^-126: value-differs=0    (sin odd and cos even, in value)
symmetry, |x| >= 2^-126: zero-sign-only=3   at 620046660, 1175634300, 1240093300
symmetry, zero/subnormal: violations=8388608 (= 2^23)
```

* **Zero value-level violations.** `sin_pinned` is exactly odd and
  `cos_pinned` exactly even. This is not assumed from the algebra of
  round-ties-away; it is checked at 1.3 billion arguments.
* **Three sign-of-zero exceptions**, all above `2^29`. At those arguments the
  reduced argument underflows to `±0` and the polynomial returns a zero whose
  sign does not mirror; the *values* agree. Nothing below `2^20` is affected,
  so no RoPE angle a 1M-position context can produce is affected.
* **`2^23` subnormal-class violations**, which are `2^23` bit patterns
  (zero and every positive subnormal) and are a property of §1.3, not of §6.3:
  DAZ flushes the input and `sin_pinned` returns `+0.0` regardless of sign.
  The spec's FTZ/DAZ pin is what makes this deterministic, and determinism —
  not signed-zero fidelity — is what CIS-2 requires. It is recorded here so
  that an implementer who preserves the sign of zero knows they will diverge.

---

## 5. What now enforces this in CI

An exhaustive sweep is a measurement and cannot live in `cargo test`. What
went into the crate is what a conformance suite needs
(`cis2-verify/src/mathpin.rs`, `mod large_angle`, 2 tests):

* `reduction_goldens_hold_out_to_a_one_million_position_context` — 12
  `(x, sin bits, cos bits)` triples: the seven arguments at which the
  exhaustive sweep recorded its worst absolute or worst ULP error, plus 1.0,
  19.0 (the old §14.4 edge), 400.0, 65535.0 (a 64k context) and 1048575.0
  (a 1M context).
* `sin_is_exactly_odd_and_cos_exactly_even_for_normal_inputs` — the symmetry
  the sweep proved, asserted on bits, with the DAZ exception asserted rather
  than assumed.

Every operand is routed through `core::hint::black_box` in both directions per
CONTRIBUTING §3: `sin_pinned(f32::from_bits(K))` on a literal is exactly the
shape LLVM is entitled to fold, and a folded constant reports the compiler's
arithmetic rather than the pinned runtime FPU's.

**Mutation control.** The goldens are not vacuous. Replacing §6.3's f64-staged
reduction with the classic single-precision one —

```rust
// non-conforming: reduce in f32
let r = x - (k as f32) * core::f32::consts::FRAC_PI_2;
```

— makes both tests fail (`sin_pinned at 0x45034E6F: left 1060882370, right
1060881274`), and the symmetry test fails too, at `x = 4476.7695`, where the
mutant returns `-0.0` for `sin(-x)` against `+0.0` for `-sin(x)`.

**And the §13.1 vector catches it as well.** Running the mutated verifier
against the pinned SmolLM2-135M artifacts:

```
generated-token-ids 28,665,436,253,1838,8180,3365,14176,30,2306,4161,281,253,2066,2291,351   (unchanged)
argmax-digest       0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff          (unchanged)
witness-digest      62844473d979411c8ea7ebe036aa587ddc19976ad6bc8931ee84d957d44fa124          (moved)
conformance FAIL: CIS2_REF: got 62844473..., spec pins d8274305...
```

This was worth checking because the prior expectation was the opposite: a
20-position decode only reaches angles ≤ 19, so an f32-staged reduction was
expected to be invisible to §13.1 and to need op-level goldens the way §6.1's
`rsqrt` route and §11's tie-break do. It is not invisible — the f32 π/2
constant is already wrong in the last bits at `x = 19` — so the op-level
goldens in §5 are defence in depth here, not the only line of defence.

It is, however, the **third** independent instance of the same pattern: a real
arithmetic violation that moves the §12.1 witness digest while leaving the
argmax digest and the generated token ids untouched. The other two are a
single flipped weight bit (E15e) and the RMSNorm reassociation (E25). Three
different kinds of violation, three times the same answer: **output-only
checking does not see any of them.** That is the standing argument for §12.1
hashing full logit vectors rather than argmaxes, and it is now supported by
three disjoint failure modes rather than one.

---

## 6. Proposed §14.4 replacement

> **14.4. Trig polynomial accuracy is measured, exhaustively, to `|x| < 2^31`.**
> §6.3's Cody–Waite reduction has been evaluated at **every** f32 bit pattern
> `x` with `0 <= x < 2^31` (1,325,400,064 arguments, subnormals included)
> against a f64 accuracy oracle, under the §1.3 pinned environment. The worst
> absolute error is **9.4218e-8 for `|x| < 2^20`** and **2.0925e-7 for
> `|x| < 2^31`** — 0.790 and 1.756 ULP at 1.0 respectively. Relative (ULP)
> error is much larger near the zeros of `sin` and `cos` and is not a
> meaningful figure of merit for these functions; absolute error is.
>
> Because §7.1 makes `inv_freq[0]` exactly `1.0` and every later entry
> smaller, the largest RoPE angle a decode evaluates is its sequence length.
> The `|x| < 2^20` figure therefore covers **every RoPE angle any context up
> to 1,048,576 positions can produce**, for any `rope_theta`. Direct
> enumeration of the angle multiset for both §0 models at `L = 131,072`
> agrees: max absolute error 9.3815e-8.
>
> `sin_pinned` is exactly odd and `cos_pinned` exactly even in value over
> every f32 with `2^-126 <= |x| < 2^31`. Two exceptions are documented and
> neither affects any reachable RoPE angle: on the zero/subnormal class §1.3's
> DAZ makes `sin_pinned` return `+0.0` for both signs, and at three arguments
> above `2^29` (620046660, 1175634300, 1240093300) the reduced argument
> underflows and only the sign of a zero result fails to mirror.
>
> Tier-1 op-level goldens pinning this behaviour at the worst-case arguments
> are in `cis2-verify/src/mathpin.rs`, `mod large_angle`.
>
> **ERRATUM E-2 (2026-09-09).** Through v0.3b as published, this clause read
> "Trig polynomial accuracy is only validated for `|x| ≲ 14` … A clean-room
> implementer targeting a longer sequence than this spec's 20 positions should
> not assume this polynomial's accuracy holds unchanged," carried unchanged
> from v0.1. That was accurate about what had been measured and badly
> misleading about what is true. §6.3 is unchanged; only this limitations note
> is. See CHANGELOG.md, "Errata against v0.3b", and docs/E26_TRIG_RANGE.md.

---

## 7. What this does **not** establish

* **It is not a proof.** It is an exhaustive enumeration of a finite domain on
  one host, which is stronger than a bound for the arguments enumerated and
  weaker than a bound for arguments outside them. Nothing is claimed for
  `|x| >= 2^31`.
* **It says nothing about a different reduction.** These are the errors of
  *this* pinned routine. An implementation using Payne–Hanek or a longer
  Cody–Waite chain will be more accurate and, for that reason, **non-conforming**
  — CIS-2 pins bits, not accuracy. Being closer to the true `sin` is a
  conformance failure here.
* **The oracle is a host libm.** A host whose f64 `sin` were badly wrong would
  move the error columns. It would not move the goldens in §5, the symmetry
  result in §4, or any pinned digest, none of which consult the oracle.
* **Accuracy is not the reason the reduction is pinned.** §6.3 is normative
  because two implementations must produce identical bits, and that would be
  true of a worse reduction too. This note removes a *caution* from §14.4; it
  does not add a requirement to §6.3.
* **No timing figures.** None were taken; penguin is a crosvm VM (Rule A).

---

## 8. Provenance (Rule B)

| | |
|---|---|
| Host | `penguin` — ChromeOS Crostini (crosvm VM), Debian 13 trixie, x86_64. **VM: no timing numbers reported** (Rule A). |
| Toolchain | `rustc 1.98.0 (88d9e12ae 2026-08-18)`, `--release`, `--offline` |
| FP environment | `x86_64 MXCSR FTZ(bit 15)+DAZ(bit 6)`, `fpenv::pin_and_selftest()` per worker thread |
| Repo / parent commit | `cis2-spec`, `cm/cis2-verify-standalone`, `be1f4df` |
| Sweep program | `cis2-verify/examples/trig_exhaustive.rs` (`cargo run --release --example trig_exhaustive [exponent]`, default 16) |
| Superseded program | `cis2-verify/examples/trig_range.rs` (sampled; kept, marked SUPERSEDED, numbers not to be quoted) |
| Oracle | host glibc f64 `sin`/`cos`, result rounded to f32 |
| Arguments enumerated | 1,233,125,376 (`E=20`); 1,325,400,064 (`E=31`); 4,194,304 RoPE angles per model at `L=131072` |
| Model artifacts (§3 mutation control only) | `model.safetensors 80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`, `config.json 1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843`, `tokenizer.json 9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c` |
| Baseline witness digest | `d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df` (conformance=PASS, unchanged by this work) |
| Test suite after this change | 43 lib tests + 6 end-to-end tests, all passing |

The sweep programs are deterministic and take no input other than the
exponent; anyone with the crate and a pinned FP environment reproduces the
tables exactly.
