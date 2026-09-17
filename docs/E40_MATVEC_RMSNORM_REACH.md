# E40 — the two operations E39 could not reach: §5.1 reductions and §8 RMSNorm

**Status: complete.** E39 established where §1.3's FTZ/DAZ pin *cannot* act
(`exp_pinned`'s output, exhaustively) and measured the one place in §10 where it
can (the softmax division: 0 FTZ-decisive quotients in 22,626,240, nearest miss
a factor of 1.32). It left a stated limit: **§5.1 reduction and §8 RMSNorm
intermediates are uncounted by any experiment in this repository**, so E22 item
1 — does a denormal ever arise in a real decode? — was narrowed, not closed.

E40 counts them, and finds a defect in E39's own instrument on the way.

**Result.** Across both §0 checkpoints, **562,531,070,920** classified
intermediates — §5.1 products and partial sums, §8 RMSNorm intermediates, §10
softmax weights — together with **628,547,776** weight operands, contain **no
subnormal at all**, and none within a factor of 1.32 of one. Independently of
every counter, a pinned and an unpinned dump of all 7,467 named intermediate
tensors of the normative §13.1 decode are **byte-identical**. On the normative
vector, §1.3's FTZ/DAZ is not digest-relevant.

## 1. The cheapest exact probe first: are any *weights* subnormal?

Every product in §5.2 has a weight as one operand. §1.3's DAZ reads a subnormal
**input** as zero, so a single subnormal weight would mean DAZ acts on every
product that uses it — no decode required to find out. Operands are scanned as
the decode sees them, after §2.5 widening, which is what `safetensors::load`
produces.

| checkpoint | tensors | operands | subnormal | exact zeros | smallest nonzero (normal) |
|---|---|---|---|---|---|
| §13.1 reference (SmolLM2-135M) | 272 | 134,515,008 | **0** | 0 | `0x30880000` = 9.895302e-10 |
| Qwen2.5-0.5B | 290 | 494,032,768 | **0** | 1,677 | `0x312b0000` = 2.4883775e-9 |

**0 of 628,547,776 weight operands across both §0 models are subnormal**, and the
smallest nonzero weight in either is some 29 orders of magnitude above the
subnormal boundary (`1.1755e-38`). The detector is shown able to classify one
(`0x00000001` → subnormal) in the same run.

So DAZ takes no decision on any §5.2 weight operand in either model. That is a
static property of the checkpoints, independent of prompt, decode length and
tokenizer — the first result in this series that does not carry a "one vector"
caveat.

## 2. That leaves products, partial sums, and RMSNorm intermediates

A product of two normals can still be subnormal, and a partial sum driven near
zero by cancellation can be too — E38 §4.1 named exactly that gap and did not
test it. So E40 instruments the intermediates themselves:

* `dot_seq` (§5.1, and therefore all of §5.2 matvec): **every product** `a[i]*b[i]`
  and **every partial sum** `acc + p`.
* `rmsnorm` (§8): `x[i]*x[i]`, `x[i]*inv`, and `scaled*weight[i]`.

Each is classified by the E39 predicate: the exact value is formed in f64, and
counted `FTZ-DECISIVE` only if it is a nonzero subnormal **at or above half the
smallest subnormal** (`7.0065e-46`), which is where FTZ changes the stored bits
rather than merely agreeing with round-to-nearest.

## 3. The control found a defect in E39's instrument

E35's rule — a counter that reads zero must be shown able to fire — applied
again, and this time it caught something. The first control run reported an
RMSNorm case whose stored bits **differ** pinned vs unpinned
(`0x00000000` vs `0x006ce3eb`) and scored it `ftz_decisive=0`.

The cause: the census widened its operands with `x as f64`, which compiles to
`cvtss2sd` — an SSE conversion, and **§1.3's DAZ makes it read a subnormal
operand as zero**. A census built on `as f64` is therefore blind to exactly the
DAZ cases it exists to count. It sees `0.0`, concludes "not subnormal", and
reports a clean zero.

E39's `note_softmax_weight` shipped with this defect (commit `2e33e72`).

**Fix.** `census::widen_exact` decodes the f32 bit pattern with integer
arithmetic and rebuilds the value from a normal double, so every operand of
every SSE instruction in the census is an integer or a normal double and no
MXCSR bit can act on it.

**Does it change E39's published result?** No, and this is checked rather than
argued both ways. By construction: `as f64` and `widen_exact` differ only on a
subnormal operand, and E39's operands are `exp_pinned`'s output — never
subnormal, proven over all 2^32 arguments — over a denominator `>= 1`. By
measurement: re-running the 16-token cell with the corrected widening gives the
same `min-nonzero-|q| = 6.96465140988929e-25` and the same zero counts.

### 3.1 The corrected control, against pinned-vs-unpinned ground truth

| case | exact value | predicate | pinned | unpinned | differ |
|---|---|---|---|---|---|
| `MIN_POSITIVE * 1` | 1.1755e-38 | not decisive | `0x00800000` | `0x00800000` | no |
| `MIN_POSITIVE * 1e-2` | 1.1755e-40 | **DECISIVE** | `0x00000000` | `0x000147ae` | **yes** |
| `MIN_POSITIVE * 1e-7` | 1.1755e-45 | **DECISIVE** | `0x00000000` | `0x00000001` | **yes** |
| `MIN_POSITIVE * 1e-10` | 1.1755e-48 | not decisive | `0x00000000` | `0x00000000` | no |
| RMSNorm, subnormal weight | — | **DECISIVE** | `0x00000000` | `0x006ce3eb` | **yes** |
| `dot_seq` cancellation | — | **DECISIVE** | `0x00800000` | `0x00000001` | **yes** |

All six agree. The last two are the rows the old instrument got wrong.

## 4. Making it cheap enough to run at all

Classifying 37.5 billion intermediates through an atomic counter each is not
affordable. The per-element cost is instead **two integer compares** on the bit
patterns:

* `maybe_tiny_mul(a, b)`: the product is at least `2^(ea + eb - 254)`, so
  `ea + eb >= 128` already guarantees a magnitude of at least `2^-126` — a
  normal. The filter admits everything up to `ea + eb <= 130`, two binades of
  margin. A zero exponent field always passes, so DAZ cases are never filtered
  out.
* `maybe_tiny_add(a, b)`: `|a + b| <= 2·max(|a|,|b|)`, so a subnormal sum needs
  the larger operand below `2^-125` — exponent field at most 2, filtered at 4
  for margin.

The exact f64 classification runs only for intermediates the filter admits. The
intermediate *total* is accumulated once per `dot_seq` call rather than once per
element. Both filters are integer arithmetic on bit patterns and cannot
themselves be perturbed by MXCSR; both are conservative supersets, so a
filtered-out intermediate is provably not subnormal.

## 5. The census on a real decode

`silu_reach` is the instrumented decode harness E29–E39 used. It now also
reports the two new counters. Both models of §0 were run; the counters are over
*every* call the decode makes, not a sample.

A row reads `true-subnormal` for "the exact value of this intermediate is
nonzero and below `f32::MIN_POSITIVE`" and `FTZ-DECISIVE` for the narrower
condition E39 §4 defines: additionally at or above `HALF_MIN_SUBNORM`, so that a
conforming f32 store *would* have held a nonzero subnormal and §1.3 flushed it.
Only `FTZ-DECISIVE > 0` makes §1.3 digest-relevant.

| model / length | §5.1 reduction intermediates | §8 RMSNorm intermediates | §10 softmax weights | true-subnormal | FTZ-DECISIVE |
|---|---|---|---|---|---|
| Qwen2.5-0.5B, gen=16  | **37,557,395,456** | **5,005,056** | **127,680** | **0** | **0** |
| Qwen2.5-0.5B, gen=256 | **514,639,978,496** | **68,226,816** | **22,626,240** | **0** | **0** |
| SmolLM2-135M (§13.1), gen=16 | **10,233,603,072** | **4,005,504** | **102,600** | **0** | **0** |

The 256-token run classifies **514,639,978,496** §5.1 intermediates — every
product and every partial sum of every `dot_seq` call in the decode — and finds
no subnormal among them. Note what this does *not* rely on: the classification is
of the intermediate itself, so whatever the activation operand happened to be,
the product and the running sum are counted. The §1 weight scan is a separate and
strictly stronger statement about the other operand — DAZ acts on inputs, so a
subnormal weight would matter even where no intermediate underflows, and there
are none.

The digests are unchanged from E38's: `witness
ab76b80ba0d5fd826bee8166760b0ed3be4385c064a4df7245fa4ae4cc0335bc`, `argmax
5991edc7b6176aeb4197d36da484c3ebdcd1ac3a435695d0bd6c12dd586b502e` at gen=256.
A census build that moved a digest would be a census build that changed the
arithmetic it is supposed to be observing.

### 5.1. What this does and does not close

It closes the limit E39 stated. E22 item 1 asked whether a denormal arises in a
real decode; for §5.1 reductions, §8 RMSNorm, §10 softmax weights, §6.2
`exp_pinned` outputs and the §2.5 weights, the answer on these two vectors is
**no, and not within a factor of 1.32**.

It does not make §1.3 removable from the specification. §1.3's job is to make
the *platform* answer, not the model, and the argument for keeping it is
unchanged: a platform that flushes is a platform whose results depend on a
control register, which is exactly the class of divergence CIS-2 exists to
close. What E39 and E40 establish is narrower and more useful — **on the
normative vector, §1.3 is not digest-relevant**, so a future conformance tier
can state that no implementation's digest is hostage to it.

### 5.2. Re-verifying E39 with the corrected widening

§3's defect was in the instrument E39 shipped, so E39's published number is
re-derived with `widen_exact` in place rather than asserted to be unaffected.
Both ways agree:

| run | min nonzero softmax weight | before fix | after fix |
|---|---|---|---|
| Qwen gen=16  | `6.96465140988929e-25` | same | same |
| Qwen gen=256 | `1.5562307486661955e-38` | same | same |

This is the expected outcome and §3 says why: `note_softmax_weight`'s operands
are an `exp_pinned` output — which E39 proved exhaustively is never subnormal —
over a denominator at least 1, so there was never a subnormal operand for DAZ
to hide. The defect was real, and in this one call site it could not bite. It
could bite in the three call sites E40 adds, which is why the control found it
before any E40 number was published.

### 5.3. The differential that does not depend on a counter at all

Every number above comes from a counter at a site I chose to hook. The sites I
did not hook are unmeasured by construction, and §6 lists them. There is one
measurement that does not have that shape.

`layer_dump` (§14.6, E28) writes every named intermediate tensor of the decode
to a file — `ln1`, `q_proj`, `k_proj`, `v_proj`, `attn_out`, `o_proj`,
`resid_attn`, `ln2`, `gate_proj`, `up_proj`, `mlp_act`, `down_proj`,
`resid_mlp` per layer per position, plus `embed`, `final_norm` and `logits`.
With `CIS2_E40_UNPINNED=1` the whole decode runs inside
`fpenv::probe_unpinned`, so §1.3's FTZ and DAZ are cleared for the duration and
MXCSR is restored afterwards. Running both and comparing the files asks one
question with no counter in it: **did clearing §1.3 change any stored
intermediate, anywhere in the stack?**

Each tensor name is emitted twice, because §12.4's determinism check runs each
decode twice; E28 §§1–2 checks every such pair bit-identical (0 that differ on
either model). So the `records` column is twice the `distinct tensors` column and
the `f32 values` column counts both copies. The unpinned arm runs both of its
passes unpinned, so the duplicate check holds within each arm as well as across
them.

| model | records | distinct tensors | f32 values | pinned dump | unpinned dump | differ? |
|---|---|---|---|---|---|---|
| SmolLM2-135M (§13.1) | 14,934 | 7,467 | 12,855,552 | `5386d3b0…dafb2f64` | `5386d3b0…dafb2f64` | **no — byte-identical** |
| Qwen2.5-0.5B | 11,970 | 5,985 | 25,920,256 | `5ccf6520…d048d550e` | `5ccf6520…d048d550e` | **no — byte-identical** |

Both runs print `mxcsr-inside-decode is_pinned=true` / `is_pinned=false`
respectively and `pin-restored-after=true`, so the identical result is not the
unpinned mode having silently failed to take effect; `pin_and_selftest()` is
called once in `main` and never inside `verify::run`, so nothing re-establishes
the pin under the probe. Both arms of the SmolLM2 pair reproduce the §13.1
reference digests (`witness
d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df`, `argmax
0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff`) and both arms
of the Qwen pair reproduce E28's Qwen digests (`witness c9dff099…`, `argmax
9619177f…`), so the tap is still write-only and an unpinned decode of either
model still lands on the same receipt.

A second fact falls out of the same files. Across all 38,775,808 stored f32
values of the two decodes — 12,855,552 for SmolLM2, 25,920,256 for Qwen — there
are **0 subnormals and 0 exact zeros**. The stack never stores a value in the
subnormal range at any named boundary, which is the same conclusion the counters
reach, reached without them. (That there is not one exact zero either is worth
noting on its own: a zero at a named boundary would be the signature of an
underflow that *had* been flushed.)

### 5.3.1. The differential's own positive control

A null result over 14,934 records is only evidence if the comparison could have
come out the other way (E35 §3). `examples/e40_ldctrl.rs` exercises the same
three mechanisms — `probe_unpinned`, `tap::emit`, file comparison — on the §8
RMSNorm input of §3.1 case R, which does underflow:

```
E40-LDCTRL pinned   is_pinned_during=true  y[0]=0x00000000 y[1]=0x3f7ffff8
E40-LDCTRL UNPINNED is_pinned_during=false y[0]=0x006ce3eb y[1]=0x3f7ffff8
E40-LDCTRL pin-restored-after=true
E40-LDCTRL bytes_pinned=38 bytes_unpinned=38 differing_byte_offsets=[30, 31, 32] FILES-DIFFER=true
E40-LDCTRL subnormals_stored_in_unpinned_file=1 (must be >0)
E40-LDCTRL OK --- the differential is sensitive to 1.3
```

The control asserts on both conditions, so it fails the build rather than
passing quietly: the files must differ, **and** a subnormal must actually be
present in the unpinned file — otherwise any difference at all would satisfy it
for the wrong reason. `tap::emit` writes `x.to_bits()`, so a stored subnormal
reaches the file as its raw pattern rather than being normalised on the way
out; the three differing byte offsets are inside the emitted payload.

## 6. Limits

1. **Two vectors, not a theorem.** Every count is over the §13.1 decode of two
   specific checkpoints at two specific lengths. A different prompt, a longer
   context or a different checkpoint can reach values these do not. The §1
   weight scan is the one exception — it is a property of the checkpoints
   themselves and holds for every prompt over them.

2. **The counters cover five operation classes, not all of them.** Hooked:
   §5.1's products and partial sums (via `dot_seq`), §8's `x*x`, `x*inv` and
   `scaled*weight`, §10's weight division, and §6.2's `exp_pinned` output
   (E39, exhaustively). **Not hooked:** the residual adds of §11, §7's RoPE
   sine/cosine rotations, §9.2's attention score scaling by `1/sqrt(d)`, §10's
   max-subtraction `s - max`, and the polynomial internals of `exp_pinned` and
   `ln_pinned` below the `ldexp_exact` that E39 analysed. §5.3's layer-dump
   differential is what covers these, and it covers them at tensor boundaries
   rather than per operation — a subnormal that arose and was consumed
   *between* two named boundaries would be invisible to it. The combination
   narrows the gap considerably; it does not eliminate it.

3. **`sum_seq` is not hooked, and is argued rather than measured.** `sum_seq`
   appears in two places the counters reach only indirectly. In §10 it sums
   `exp_pinned` outputs, which are non-negative and — by E39's exhaustive sweep
   — never subnormal, so the first nonzero partial sum is already at least
   `f32::MIN_POSITIVE` and every later one is no smaller: a monotone
   non-decreasing sequence cannot enter the subnormal range from above. In §8 it
   sums `sq`, whose elements are the `x*x` products the RMSNorm hook does
   classify, likewise non-negative, so the same argument applies. This is a
   proof about sign, not a measurement, and it holds only because both sums are
   over non-negative terms. It does **not** extend to any sum with mixed signs,
   where cancellation can drive a partial sum arbitrarily small — §3.1 case S is
   exactly that case, and it is why `maybe_tiny_add` exists rather than a
   sign-based shortcut.

4. **`FTZ-DECISIVE` is the right counter and it is strictly narrower than
   "subnormal".** An intermediate below `HALF_MIN_SUBNORM` rounds to `+0.0` with
   or without FTZ, so counting it would overstate §1.3's reach. Both counters
   read zero here, so the distinction does not change this result; it would
   change a future one.

5. **The census build is not a conforming verifier** and says so on every line
   of output. `--features census` adds counters and compiles in
   `fpenv::probe_unpinned`, which a conforming build must not contain; the
   examples that use either are declared `required-features`, so a default build
   fails to compile them rather than silently producing an unpinned binary. The
   digests are the check that the instrumentation is observational: they are
   unchanged from E38's at both lengths.

6. **The Qwen runs are not conformance runs.** §3.1.3/§3.1.4 refuse that
   checkpoint's `tokenizer.json`, so prompt token ids are supplied and only
   §§4–11 are exercised. The SmolLM2 run is the normative §13.1 vector with
   tokenization derived by this crate.

7. **One machine, one ISA, one compiler.** All of E40 ran on `penguin`. E29–E34
   establish that the digests are invariant to optimisation level, LTO, compiler
   and ISA, but the *counts* here were not re-derived on another target. A
   platform whose `dot_seq` is vectorised differently would produce the same
   final bits (§5.1 pins the reduction order) but could in principle visit
   different partial sums — `maybe_tiny_add` would have to be re-run there to
   say so.

8. **No timing numbers.** `penguin` is a crosvm VM (Rule A).

## 7. Provenance (Rule B)

| item | value |
|---|---|
| Host | `penguin` — ChromeOS Crostini (crosvm VM), Debian 13 trixie, x86_64, 8 cores, 6570 MB. **VM: no timing numbers reported** (Rule A). |
| Toolchain | `rustc 1.98.0 (88d9e12ae 2026-08-18)`, `--release`, `--offline` |
| Features | `--features census` (counters; `silu_reach`, `e40_wscan`, `e40_ctrl`), `--features layerdump` (`layer_dump`, `e40_ldctrl`) |
| FP environment | `x86_64 MXCSR FTZ(bit 15)+DAZ(bit 6)`, `fpenv::pin_and_selftest()`; the unpinned arm clears both via `fpenv::probe_unpinned` and restores MXCSR after |
| Repo / branch | `cis2-spec`, `cm/cis2-verify-standalone`, base commit `a4bcd21e41961d6cbe5cef5de3a5735a65df6872` |
| Checkpoint — SmolLM2-135M (§13.1) | `model.safetensors` `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`, `config.json` `1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843` |
| Checkpoint — Qwen2.5-0.5B | `model.safetensors` `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`, `config.json` `479dcf0c5286339e41ad3992cd08ae88a467c4187587936248e2b7c96283484b` |
| Binary `silu_reach` | `9afefacacb823279bbab216fa7c47f4721c77de05968937d14faa2de79450727` |
| Binary `layer_dump` | `d2678daef504882e1590ccf359a047ec75e34bc7439196f409fb0eb70c71a43f` |
| Binary `e40_wscan` | `76361b07fc80e91d1f9b804d1aaf4dd264e92cfaf84cb51aa4337b2dba7151c6` |
| Binary `e40_ctrl` | `b4a06cb478532027948e75c206060028a3c2041fd7b7803a9a1781f3a4e8b7e7` |
| Binary `e40_ldctrl` | `1c9f3d06b5e0f6de190f92cdfa218e92ad47d95515eb7b504cbef363a5893ccf` |
| Census logs | `~/e40-census-gen16.log` (both models, gen=16), `~/e40-qwen256.log` (Qwen gen=256) |
| Layer-dump logs | `~/e40-layerdiff.log` (SmolLM2), `~/e40-layerdiff-qwen.log` (Qwen) |
| Layer dumps | SmolLM2 `~/e40-ld-pinned.bin` / `~/e40-ld-unpinned.bin`, both 51,750,154 B, both `5386d3b0e529d9817af86f2ba1381193c2b174b0f26d12e22a441616dafb2f64`; Qwen `~/e40-ld-qwen-pinned.bin` / `~/e40-ld-qwen-unpinned.bin`, both 103,942,814 B, both `5ccf65207a6638d7c2aa6111e0b08e867686b0efe5c07e4e9b2e6c6d048d550e` |
| Dump summariser | `scripts/e40_dump_stats.py` (stdlib only) |
| Gates after this change | `cis2-verify/tools/check_no_fma.sh` PASS; default build 53/53, census build 58/58, layerdump build compiles and `e40_ldctrl` passes both assertions; `cis2-verify selftest` PASS; §13.1 reference decode reproduces `witness d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df` / `argmax 0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff` |

Everything above is re-derivable from the branch: build with `--features
census` and run `silu_reach` on either checkpoint; build with `--features
layerdump` and run `layer_dump` twice, once with `CIS2_E40_UNPINNED=1`, then
`cmp` the two files and summarise either with `scripts/e40_dump_stats.py`. Run
`e40_ctrl` and `e40_ldctrl` first — both assert, so an instrument that has gone
inert fails rather than reporting a clean null.
