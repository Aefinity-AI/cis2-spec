# E39 — `exp_pinned` cannot return a subnormal, so the band E29/E35/E38 counted is not where §1.3 acts

**Status: the mechanism claim in E29, E35, E38 and erratum E-10 is wrong, and
§14.1(b)'s stated reason is wrong with it. The conclusions they support all
survive; the reason given for them does not. This document replaces the reason
with a measured one and points the counter at the operation where §1.3 can
actually change bits.**

## 1. The claim, as we have been making it since E29

Every one of these says the same thing, and E38 pushed it into the specification
as erratum E-10 four hours ago:

> `SUBNORMAL BAND [-88.0, -87.33654)` … >0 means this decode computed a
> subnormal exp that §1.3's FTZ flushed to zero, so FTZ/DAZ is digest-relevant
> here.

E35 built its headline on it — that band is "the window where §1.3's FTZ/DAZ pin
is digest-relevant". E38 found 2 arguments in it, called that "a denormal arose
in a real decode", and reported that M01 (§1.3 gutted) left the digest
unchanged. §14.1(b) of the specification says the neighbouring thing about the
guarded side: the values §6.2's low guard destroys "would be flushed by §1.3 in
any case".

## 2. `exp_pinned` never returns a subnormal

§6.2's final step is `ldexp_exact(result, k)`, and `ldexp_exact` is not a
floating-point operation — it is bit manipulation with its own zero branch
(`cis2-verify/src/mathpin.rs`):

```rust
let new_exp = exp_bits + k;
if new_exp <= 0 {
    return 0.0;
}
```

`result` is the polynomial's value on the reduced argument and always has an
exponent field of 126 or 127. So the reconstructed field is either `<= 0`, in
which case the function returns **exactly `+0.0`**, or `>= 1`, in which case the
result is a **normal** f32. There is no path that produces a subnormal, and
therefore no FTZ decision is ever taken on `exp_pinned`'s output. This is a
property of the code, not of the floating-point environment.

Measured, with the pin cleared and restored around each evaluation
(`fpenv::probe_unpinned`, mutant tree only):

| `x` | where it sits | `exp_pinned(x)` pinned | with FTZ/DAZ **off** | differ |
|---|---|---|---|---|
| -87.33654 | last argument with a normal result | `0x00800026` | `0x00800026` | no |
| -87.34 | just inside the "subnormal band" | `0x00000000` | `0x00000000` | **no** |
| -87.6 | mid band | `0x00000000` | `0x00000000` | **no** |
| -87.999999 | band bottom | `0x00000000` | `0x00000000` | **no** |
| -88.0 | §13 golden, guard does not fire | `0x00000000` | `0x00000000` | **no** |
| -88.369385 | the argument E38's cell reaches | `0x00000000` | `0x00000000` | **no** |

Sweeping `[-88.5, -87.0]` at 1e-4 spacing, the smallest nonzero value
`exp_pinned` returns on that grid is `0x008000a6` — exponent field **1**, the
smallest normal. The exhaustive sweep of §5.2 lowers that to `0x00800026`, at
`x = 0xc2aeac4f`, and confirms exponent field 1 is the floor over all 2^32
arguments. `0x00800026` is also exactly the table's first row above, so the grid
sweep, the exhaustive sweep and the hand-checked boundary agree.

**So the band counter measures arguments whose *true* `exp` is subnormal. It
does not, and never did, measure a subnormal being computed and flushed.** The
band is FTZ-**independent**, exactly like the guarded side it was introduced to
be contrasted with. E29 drew the boundary in the right place for the wrong
reason and everything downstream inherited it.

## 3. What this does and does not overturn

**Overturned — the reason, in four places.**

* E35's framing of `[-88.0, -87.33654)` as "the window where §1.3's FTZ/DAZ pin
  is digest-relevant". It is not that window.
* E38 §3.2 and §4, and erratum **E-10**: "the result is a subnormal that §1.3's
  FTZ flushes to zero", and "a denormal arose in a real decode". Through the
  `exp` route on this vector, no denormal arose. The two arguments E38 found are
  real and the digests it reports are real; the sentence attributing them to
  §1.3 is not.
* §14.1(b)'s "Every value *this* guard destroys is subnormal and would be
  flushed by §1.3 in any case". The values are subnormal as real numbers, and
  they are destroyed — but by `ldexp_exact`'s `new_exp <= 0` branch, before
  §1.3 is ever consulted. The clause's conclusion (the guard is harmless) is
  correct; its stated reason is not.
* The counter's own doc comment in `ops.rs`, which asserts the same thing.

**Not overturned — everything the reason was used to support.**

* E38's M01 result stands as measured: witness `ab76b80b…` and argmax
  `5991edc7…`, byte-identical pinned and unpinned, on a mutant proven live.
* E38's reach result stands: the softmax argument really does reach -88.369385
  at 256 tokens on Qwen, and §14.1(a)'s "0 below -88.0" really is falsified in
  the wider scope. E-10's *reach* half is untouched.
* The SiLU-side result stands: 0 in §6.2's clip band across 145,385,472
  arguments.
* E-11 (§1.3's inert self-test probe) stands and is unrelated.
* No digest, coefficient, or required behaviour changes anywhere.

And it *explains* M01 properly for the first time. The digest did not move
because through §6.2 there was nothing for FTZ to do — not because a denormal
arose and rounded away downstream. E22's caveat ("SAME digest is weaker than no
denormal ever arose") is therefore still open on the `exp` route, and E38's
claim to have settled it from the other side is withdrawn.

## 4. Where §1.3 *can* act in §10, and the counter that measures it

§10's final step is an elementwise division, `w[i] = exp_i / denom`.
`exp_pinned` gives a normal or zero; `denom` is a §5.1 sum of those, and because
`exp(0) = 1` is always one of the terms, `denom >= 1`. A numerator at the
smallest normal (`1.1755e-38`) over a denominator above 1 is **subnormal**, and
that division is an SSE operation, so FTZ decides its result. A 300-position
attention row is enough: `1.1755e-38 / 300 = 3.9e-41`.

So the FTZ-relevant quantity is the softmax **weight**, not the `exp` argument,
and no experiment in this repository has ever counted it.

E39 adds `census::note_softmax_weight`, which forms the exact quotient in f64
(both operands widen exactly) and classifies it:

* `true-subnormal-quotients` — the exact quotient is nonzero and below
  `f32::MIN_POSITIVE`.
* **`FTZ-DECISIVE`** — additionally at or above half the smallest subnormal
  (`7.0065e-46`), so the f32 rounding is a *nonzero subnormal* and §1.3 changes
  the stored bits. Below that threshold the value rounds to `+0.0` with or
  without FTZ and FTZ decides nothing — the same trap that made §1.3's own
  self-test inert (erratum E-11).

### 4.1 The counter is shown able to fire before it is believed

E35's rule, applied to E35's successor. Ground truth is `DIFFER`, measured by
evaluating the same division with the pin cleared and restored:

| numerator | denominator | exact quotient | predicate says decisive | pinned | unpinned | differ |
|---|---|---|---|---|---|---|
| `MIN_POSITIVE` | 1e0 | 1.1755e-38 | no | `0x00800000` | `0x00800000` | no |
| `MIN_POSITIVE` | 1e2 | 1.1755e-40 | **yes** | `0x00000000` | `0x000147ae` | **yes** |
| `MIN_POSITIVE` | 1e7 | 1.1755e-45 | **yes** | `0x00000000` | `0x00000001` | **yes** |
| `MIN_POSITIVE` | 1e30 | 1.1755e-68 | no | `0x00000000` | `0x00000000` | no |

The predicate agrees with ground truth on all four rows, including the row that
separates "subnormal" from "subnormal enough to matter". And through the real
`softmax_seq`, not only hand-built division: a 301-element vector with one entry
at -87.336525 and 300 at 0.0 gives `d_ftz_decisive=1`, stored value
`0x00000000`.

(The first version of this control read `DIFFER=false` on every row. The
operands were compile-time constants and LLVM folded the division before the
register was ever consulted. `black_box` on both operands in both directions
fixed it — the same house rule `fpenv`'s own tests document, and the second time
in two experiments that an unprotected probe was silently inert.)

## 5. The measurement: 22,626,240 softmax weights, and §1.3 decides none of them

Same vector as E38 — Qwen2.5-0.5B, `"Once upon a time"`, 256 generated tokens,
supplied prompt ids `12522,5193,264,882` — re-run with the new counter on the
same instrumented build:

```
REACH softmax WEIGHTS n=22626240 true-subnormal-quotients=0 FTZ-DECISIVE=0
                      min-nonzero-|q|=1.5562307486661955e-38
```

* **`true-subnormal-quotients = 0`.** Not one of the 22,626,240 quotients landed
  below `f32::MIN_POSITIVE`. `FTZ-DECISIVE = 0` follows.
* The smallest nonzero weight the decode produced is `1.5562307486661955e-38`,
  bits `0x00a97562`, exponent field **1** — the smallest normal binade, one
  binade above subnormal.

The digests are unchanged from E38's run, as they must be — the counter is a
read-only observer:

```
REACH witness-digest ab76b80ba0d5fd826bee8166760b0ed3be4385c064a4df7245fa4ae4cc0335bc
REACH argmax-digest  5991edc7b6176aeb4197d36da484c3ebdcd1ac3a435695d0bd6c12dd586b502e
```

### 5.1 What this settles, and what it leaves open

E22 item 1 asked whether §1.3's FTZ/DAZ pin is digest-relevant on a real decode
— whether M01's null result means "the pin does not matter" or only "no denormal
happened to arise". E38 answered it with the wrong counter. The right counter
answers it this way:

> On this vector, through §10, **no denormal ever arose**. M01's byte-identical
> digests are fully explained: there was nothing for §1.3 to flush.

That is a stronger and more honest statement than E38's. It is also a narrower
one: it says §1.3 is not digest-relevant *here*, not that it never is. The
nearest miss is the measurement that keeps it open —

> `1.5562307486661955e-38 / 1.1754943508222875e-38 = 1.324`

The smallest weight this decode produced is a factor of **1.32** above the
subnormal boundary. A row 33 % more spread — a longer context, a sharper
attention head, a model with a larger score range — puts a weight into the
subnormal range and makes §1.3 decide bits. §1.3 is therefore not shown
unnecessary; it is shown untriggered on one vector, by a margin of a third of a
binade.

### 5.2 The structural claim, exhaustively

§2's argument is about code, not about this vector, so it can be checked
completely. `examples/e39_exhaustive.rs` evaluates `exp_pinned` on **all
4,294,967,296** f32 bit patterns and classifies each output:

```
E39-EXH arguments=4294967296 subnormal_outputs=0 zeros=1020351409
E39-EXH smallest nonzero output=0x00800026 (exp_field=1) at x=-8.733654e1 (0xc2aeac4f)
E39-EXH detector control: bits=0x00000001 exp_field=0 classified_subnormal=true
```

`exp_pinned` returns a subnormal for **no** f32 argument, and the detector that
says so is shown able to classify one. The claim in §2 is not an inference from
six sample points; it is a property of every input the function has.

### 5.3 The instrument is in the repository, and the port was checked

The measurements above were taken in `~/e39`, a copy of `cis2-verify` at E38's
commit. `note_softmax_weight`, `weight_counts` and `probe_unpinned` are now in
the repository tree itself, the last two gated behind the existing `census`
feature so a conforming build (§1.3) cannot contain a helper that clears the pin
— `cargo build --release --examples` without `census` does not compile them, and
was checked to fail closed that way.

The port was verified rather than assumed:

* `examples/e39_ctrl` run from the repository tree reproduces all four
  ground-truth rows and case-C exactly as §4.1 reports them.
* The census binary built in-repo reproduces E38's 16-token cell
  byte-for-byte: witness `c9dff099d927a6dda91514c260c7207de7aca4b361291e57fae390a352392e62`,
  argmax `9619177f959f63d175d9442e73a6063f31654b08ba4e38717e88bd2e39b0aff5`.
* §13.1's reference decode is unchanged after the port:
  `d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df` /
  `0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff`,
  `selftest` passes, suite 53/53 without `census` and 56/56 with it.

## 6. Limits

1. **One vector, one model, one prompt.** §5 measures Qwen2.5-0.5B at 256
   tokens. §5.1's factor of 1.32 is the reason not to generalise from it.
2. **§10 only.** `note_softmax_weight` counts the softmax division. Subnormals
   arising inside §8 matvec accumulation, §7 RMSNorm, or §5.1 summation are
   still uncounted by any experiment in this repository. §1.3 could be
   digest-relevant at one of those and no counter would see it. E22 item 1 is
   narrowed, not closed.
3. **The SiLU route cannot produce subnormals either.** §6.4 evaluates
   `exp_pinned(-x)` and divides `x / (1 + e)` with a denominator `>= 1`; the
   numerator is the activation itself, so a subnormal quotient requires a
   subnormal-ish activation, which the measured SiLU range
   `[-3.0068079e1, 4.196279e1]` excludes. This is reasoning, not a measurement —
   no SiLU-side weight counter was built.
4. **The f64 quotient is the *exact* quotient, and that is the point.** Both f32
   operands widen exactly and f64 division of two values in this range is exact
   to well beyond f32's subnormal resolution, so the predicate classifies what
   the f32 division *would have stored*, not an approximation of it. It is
   evaluated outside the f32 division, so the census cannot itself perturb the
   result — and the unchanged digests confirm it did not.
5. **`FTZ-DECISIVE` is a lower bound on §1.3's relevance, by construction.** It
   counts only quotients that FTZ would flush to a *different* stored value.
   DAZ — denormal *inputs* treated as zero — is not counted at all; on this
   vector no operand anywhere was subnormal, so DAZ had nothing to act on
   either, but that follows from §5's zero rather than from a DAZ counter.
6. **Instrumented build, not a conforming verifier.** The census binary carries
   the `census` feature and prints `build=instrumented`. It is not the artifact
   §11 conformance is claimed for.
7. **Tokenizer supplied, not derived.** As in E35–E38, the prompt ids are passed
   in because §3.1.3/§3.1.4 refuse this `tokenizer.json`. This is a §4–§11 run.

## 7. Provenance (Rule B)

| item | value |
|---|---|
| machine | `penguin` — crosvm VM, Debian 13 trixie, x86_64, 8 cores, 6570 MB |
| toolchain | `rustc 1.98.0 (88d9e12ae 2026-08-18)`, release profile |
| base commit | `707390c1fbad3cc524ade24bbc68e21e46720d12` (E38) |
| instrument tree | `~/e39`, copy of `cis2-verify` at that commit + census hooks |
| model | Qwen2.5-0.5B, `model.safetensors` sha256 `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342` |
| | `config.json` sha256 `479dcf0c5286339e41ad3992cd08ae88a467c4187587936248e2b7c96283484b` |
| census binary | `examples/silu_reach` sha256 `67b01fa0de4557d89dc2e17badd004b7857d64079a16ac51a7fd594c3b1354f3` |
| control binary | `examples/e39_ctrl` sha256 `952cad57d1e715def677ccc0ca645ca7584408c34184fee884d3f91f85ca153c` |
| exhaustive binary | `examples/e39_exhaustive` sha256 `d34f07202eed57219ec8843747caba55133dc28ae6e2026e763aae302f6ad19a` |
| census log | `~/e39-qwen256.log` |
| exhaustive log | `~/e39-exhaustive.log` |
| date | 2026-09-09 |

**No timing numbers are reported (Rule A): penguin is a VM.**
