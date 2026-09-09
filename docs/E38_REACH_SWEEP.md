# E38 — the §6.2 guard is reachable: what a long enough decode on the other model does to §14.1(a) and E35

**Status: complete. Result: two of §14.1(a)'s three scope limits close cleanly,
and the third does not — it turns out to have been holding up a false negative.
A 256-token Qwen2.5-0.5B decode puts real softmax arguments below §6.2's low
guard and inside the FTZ-dependent subnormal band that E35 declared "not
constructible".**

Twenty cells across both §0 models, ten prompts each — Japanese, Cyrillic,
Arabic, astral-plane emoji, accented Latin, mathematical symbols, two
degenerate-repetition prompts, the §13.1 reference decode, and a 256-token
decode. **145,385,472 SiLU arguments and 43,523,148 softmax arguments**, against
the 18,892,800 and 2,355,480 the published census rests on.

## 1. What §14.1(a) said, and the three limits it declared

> A read-only census of SmolLM2-135M decodes over six prompt/length
> configurations records **0** of its **18,892,800** `silu_pinned` arguments in
> that band — the most negative argument seen anywhere is -31.406876, some 57
> units short of the band — and **0** of its **2,355,480** softmax
> `exp_pinned` arguments below `-88.0` […] That is one model, six ASCII prompts
> and at most 64 generated tokens, and does not establish unreachability in
> general.

`E35_DENORMAL_REACH.md` states the same conclusion more strongly, as its
headline: *"Zero of 21,248,280 `exp_pinned` evaluations across six prompts land
in the window where §1.3's FTZ/DAZ pin is digest-relevant, and the nearest
approach is 26.79 away in ln-space. E22 follow-up item 1 […] is not
constructible from these prompts on this checkpoint."*

Three limits, none of them needing a decision from anyone — only a machine and
more runs. E38 runs them.

## 2. Method, and why every cell supplies its token ids

`silu_reach` takes prompt token ids as an optional 4th argument. Every cell here
uses it, for two independent reasons:

- A **non-ASCII** prompt would otherwise depend on §3.1.4's still-open Unicode
  question, and E-7 deliberately kept its six cells ASCII to avoid confounding a
  reachability result with a tokenization one. Supplying the ids removes the
  confound instead of avoiding it: the census asks what §4–§11 compute, and it
  can be asked about any token sequence regardless of how §3 would derive it.
- A **Qwen** cell must supply them regardless (§14.8 / erratum E-3).

Such a run exercises §4–§11 only and is not a conformance run. Two controls make
the supplied-id path trustworthy rather than assumed:

| control | required | result |
|---|---|---|
| SmolLM2 cell 1 run through §3 (ids derived) vs. supplied | identical digests | `d8274305…` / `0b9c8f3a…` both ways — **§13.1's own pinned pair** |
| every cell: ids the census reports using vs. ids handed to it | equal | **20/20 `SUPPLIED-ID CHECK PASS`** |

The band counters carry E35's positive controls, which still pass
(`cargo test --features census census`, 3/3).

## 3. Result

### 3.1 The SiLU side is not reached, on either model, by anything tried

`silu IN 6.2 CLIP BAND [-88.7228317, -88.0000076]`: **n=0 in all 20 cells**, over
145,385,472 arguments. The most negative SiLU argument anywhere is **-32.173088**
(SmolLM2, Japanese) — 56 units short of the band, against the published
-31.406876. Sevenfold more arguments, a second architecture, six writing systems
and a 4x longer decode move that figure by 0.77.

### 3.2 The softmax side *is* reached

| cell | softmax args | most negative arg | `< -88.0` | in `[-88.0, -87.33654)` |
|---|---|---|---|---|
| published §14.1(a) census (6 ASCII SmolLM2 cells) | 2,355,480 | -39.709507 | 0 | 0 |
| E38 SmolLM2, 10 cells | 19,654,380 | **-54.981120** (Japanese) | 0 | 0 |
| E38 Qwen, 9 short cells | 1,242,528 | -57.128740 (accented Latin) | 0 | 0 |
| **E38 Qwen, `"Once upon a time"`/256** | **22,626,240** | **-88.369385** | **2** | **2** |

Two arguments fall past §6.2's low guard, and two more land in the band between
`-88.0` and `ln(f32::MIN_POSITIVE)` where §6.2 *computes* rather than clamps and
the result is a subnormal that §1.3's FTZ flushes to zero.

### 3.3 It is a trend with decode length, not a freak cell

Same model, same prompt, same ids, four lengths:

| generated tokens | softmax args | most negative arg | in band |
|---|---|---|---|
| 16 | 127,680 | -55.623780 | 0 |
| 128 | 5,810,112 | -80.235170 | 0 |
| 192 | 12,841,920 | -82.705530 | 0 |
| 256 | 22,626,240 | **-88.369385** | **2** |

The approach deepens at every length measured. Four points is not a law, but it
is enough to say the previous censuses were not near a ceiling — they were
short. The prompt is the §13.1 reference prompt; nothing exotic was needed.

### 3.4 Non-ASCII prompts approach faster than ASCII ones

The deepest SmolLM2 argument, -54.98, is the Japanese cell, against -39.71 for
the published six ASCII cells — the ASCII restriction was understating the
approach by 15 units in ln-space on the softmax axis while leaving the SiLU axis
essentially unchanged. E-7 kept its cells ASCII to protect a *tokenization*
result; the cost was a *reachability* result measured on the easy inputs.

## 4. So does §1.3 matter here? M01, with a mutant that is shown to be live

E22's necessity matrix mutated §1.3 (mutant **M01**), saw the digest not move,
and recorded the row as **"not exercised"** with an honest caveat:

> "SAME digest" is evidence that no denormal *changed a result*, which is
> slightly weaker than "no denormal ever arose"; the mutant is not instrumented
> to tell those apart.

E35 could not settle it either, because no decode it measured produced a
denormal at all. §3.2's cell does. So M01 was rebuilt and run on it.

**The mutant is instrumented to prove it is live before its null result is
believed** — the same discipline E35 applied to its counters, applied to a
mutant. `pin_and_selftest` is replaced by a body that *clears* FTZ and DAZ and
returns `Ok`, and the census prints three probes:

| probe | pinned build | M01 mutant | means |
|---|---|---|---|
| `fpenv::is_pinned()` | `true` | **`false`** | MXCSR really is unpinned |
| `f32::MIN_POSITIVE * 0.5` | `0x00000000` | **`0x00400000`** | a subnormal is computed and **kept**, not flushed — FTZ is off |
| `f32::from_bits(1) + 0.0` | `0x00000000` | **`0x00000001`** | a subnormal input survives — DAZ is off |

And a negative control: on the 16-token cell, which reaches no band, mutant and
pinned builds must agree. They do, at `c9dff099…` / `9619177f…`.

**Result on the 256-token cell, which reaches the band twice:**

```
pinned   witness ab76b80ba0d5fd826bee8166760b0ed3be4385c064a4df7245fa4ae4cc0335bc
M01      witness ab76b80ba0d5fd826bee8166760b0ed3be4385c064a4df7245fa4ae4cc0335bc
pinned   argmax  5991edc7b6176aeb4197d36da484c3ebdcd1ac3a435695d0bd6c12dd586b502e
M01      argmax  5991edc7b6176aeb4197d36da484c3ebdcd1ac3a435695d0bd6c12dd586b502e
```

Bit-identical. **E22's two possibilities are now separated by measurement: a
denormal *did* arise — twice — and it *did not* change the result.** That is
strictly more than E22 could say, and it is the weaker of the two outcomes for
§1.3: this decode is a witness that the band is reachable, not that the pin is
necessary.

### 4.1 Why that is unsurprising, stated as a mechanism rather than a hope

§10 subtracts the row maximum before exponentiating, so `exp(0) = 1` is always
one of the terms and the denominator is always `>= 1`. A numerator in the
subnormal band is at most `1.18e-38`, so the normalized weight is at most
`1.18e-38`, and the term it contributes to the weighted sum is below the f32 ulp
of any accumulator of ordinary magnitude. Pinned or not, the term rounds away.

This is an explanation, not a proof, and the gap is worth naming: an accumulator
driven near zero by cancellation would have a small enough ulp for such a term
to survive. E38 does not exclude that; it measures one decode where it did not
happen. §1.3 stays justified where it was — as a portability requirement that
keeps implementations bit-identical regardless of their host's default MXCSR —
and it still has **no end-to-end necessity witness**.

## 5. A side finding: half of §1.3's self-test cannot fail

Building the probe table exposed something in §1.3's own adversarial self-test.
Its "denormal OUTPUT must flush" check computes `f32::MIN_POSITIVE * 1.0e-10`
and requires the result's bits to be zero. That product is `1.18e-48` — *below
half the smallest positive subnormal* (`7.0e-46`), so it rounds to `+0.0` with
or without FTZ. Measured directly in the unpinned mutant:

```
REACH M01-PROBE spec-1.3-selftest-probe_bits=0x00000000   (is_pinned=false)
```

The check passes on a machine with FTZ off. It is inert.

This does not weaken §1.3 as enforced: the readback check
(`back & REQUIRED != REQUIRED`) catches a cleared bit directly, and the DAZ half
of the self-test does discriminate (`0x00000001` unpinned vs `0x00000000`
pinned). But the clause presents the arithmetic as *adversarial* evidence
independent of the readback, and on the FTZ half it is not evidence at all. A
`* 0.5f32` in place of `* 1.0e-10f32` makes it discriminate — `5.88e-39` is a
subnormal, flushed when pinned and kept when not. Filed as **erratum E-11**;
the fix is in the specification text and the reference, and no digest moves.

## 6. Limits, stated as limits

1. **Twenty cells is not all prompts.** Two checkpoints, ten cells each, greedy
   decode only. E38 replaces "not constructible from these prompts" with
   "constructible from *one* of these prompts", which is a change of kind, not a
   census. Nothing here bounds how deep a hostile prompt could go.
2. **The event is rare even where it happens.** In the cell that reaches the
   band, 2 of **22,626,240** softmax exponent arguments land in
   `[-88.0, -87.33654)` and 2 more trip the LOW guard — about 1.8e-7 of the
   arguments in that one decode, and 0 in the other nineteen cells.
3. **The clip band on the §6.4 side remains unreached.** Deepest SiLU argument
   over all 20 cells is `-32.173088`, still ~56 units in ln-space from
   `-88.0`. §14.1(a)'s scope note is corrected for the softmax axis and stands,
   with better evidence, for the SiLU axis.
4. **The census build is not a conforming verifier.** It prints
   `REACH build=instrumented (NOT a conforming verifier)` for that reason. The
   digests it emits match the conforming build's on the control cell, which is
   what licenses comparing them, but no conformance claim rests on it.
5. **Cells that supply token ids exercise §4–§11 only.** §3's tokenizer path is
   bypassed there by design (§14.8 / erratum E-3 for Qwen; for the non-ASCII
   cells, to keep §3.1.4's open Unicode question out of the measurement). Those
   are not conformance runs. Both spec-3 controls — SmolLM2 cell 1 derived and
   supplied — give §13.1's own pinned pair, `d8274305…` / `0b9c8f3a…`.
6. **M01's null is one cell's null.** Byte-identical digests pinned and unpinned
   on a decode containing two flushed subnormals is evidence about *that*
   decode. §4.1 gives the mechanism that makes it expected, and names the case
   (an accumulator driven near zero) that would break it and was not tested.
7. No timing figures appear anywhere in this document. penguin is a crosvm
   guest; measurements of speed from it are not admissible under Rule A. Every
   number here is a count, a digest, or an exactly-representable float.

## 7. Provenance

Machine **penguin** (Crostini/crosvm guest, Debian 13, x86_64, 8 cores).
Toolchain `rustc 1.98.0 (88d9e12ae 2026-08-18)`, built
`--release --offline --features census` (and the same with the M01 patch, in a
separate tree at `~/e38-mut`, never in the repo tree). Weights: SmolLM2-135M at
`cis2-spec/weights`, Qwen2.5-0.5B at `~/qwen05b`, both fp32, both the
checkpoints §13.1 and E28 already pin. Cell files were built by each model's own
`AutoTokenizer` (`transformers 5.16.1`, `~/venvs/torch`).

| artifact | sha256 |
|---|---|
| `scripts/e38_mkcells.py` | `997565a430bb83a0b8af6b3902bd7402da25d9581ea70444023a89a0a05f20db` |
| `scripts/e38_reach_sweep.sh` | `797e5707a72f994cdbb18d2a1d08ed5eecf2a2e9961e20c32000aeba1b7d8161` |
| `scripts/e38_verify_m01.sh` | `34ca629e76722408bf530ce88a5b75e0b0a07f6ed85f27f95bfdd7c371d74f2c` |
| `docs/logs/e38-cells-smollm2.txt` | `cbfe4e22e2482388cc911f76e50e68576f0c76d8c5ac8929d42f56ab595644e7` |
| `docs/logs/e38-cells-qwen.txt` | `63c1c947a917d025cdd2685f5da516be3f14a8d950eb82a44c25174ed473b90a` |
| `docs/logs/e38-reach-sweep.log` | `07fea07de27a271dfbcdf3a744ea0dc8cc025dd412db59f38e8c4cd219ad7f6a` |
| `docs/logs/e38-verify.log` | `889498d859819a5f1569240b2bd2e282930b68a3ccc2ae2e7a44e370034532d2` |

Every figure quoted above is in one of those two logs. Errata raised by this
experiment: **E-10** (§14.1(a) and E35's reachability claim) and **E-11**
(§1.3's inert FTZ self-test probe).
