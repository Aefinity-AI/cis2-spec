# Changelog

This repository tracks the CIS-2 specification and its independent
clean-room verifiers. The reference implementation the spec was audited
against lives in a separate repository (see README "Scope"); this
changelog covers the spec text and verifiers published here.

## Errata against v0.3b

Numbered corrections to the published v0.3b text. **No erratum here moves a
pinned digest**; every value in section 13 is unchanged.

### E-1 (2026-09-09) --- section 14.5 was wrong: the RMSNorm multiply order *is* bit-level necessary

**What v0.3b said**, carried unchanged from v0.1:

> 14.5. **RMSNorm multiply order (section 8) is pinned but its bit-level
> necessity is unconfirmed.** `(x[i]*inv)*weight[i]` vs. `x[i]*(inv*weight[i])`
> are not provably identical for arbitrary fp32 operands under rounding, but no
> divergence between the two orders has actually been observed on either model
> tested.

**Why that is wrong.** A divergence had in fact been observed before v0.3b was
tagged --- E22's M09 mutation is exactly this reassociation, and it moved the
witness digest --- and section 14.5 was simply never updated. Measured since
(`docs/E25_RMSNORM_ASSOCIATION.md`): the two associations differ by one ULP on
**231,014 of the 667,584** RMSNorm elements a section 13.1 decode produces,
about 35%, and substituting the other association changes the witness digest
from `d82743059d...` to `570c0bbb0d...`. The Rust clean-room verifier and the C
reference implementation agree bit-for-bit on that non-conforming digest as
well as on the conforming one.

**What it should say:**

> 14.5. **RMSNorm multiply order (section 8) is pinned, and its bit-level
> necessity is measured.** `(x[i]*inv)*weight[i]` and `x[i]*(inv*weight[i])`
> differ by one ULP on about 35% of the operand triples an actual decode
> produces --- 231,014 of the 667,584 RMSNorm elements in the section 13.1
> vector. Substituting the other association changes the section 13.1 witness
> digest from `d82743059d...` to `570c0bbb0d...`. It does **not** change the
> argmax digest or the generated token ids for this vector, so the violation is
> invisible to any check that hashes only the model's outputs --- one of the
> reasons section 12.1 hashes the full logit vector. See
> `docs/E25_RMSNORM_ASSOCIATION.md`.

The corrected text is applied in place in `docs/CIS2_SPEC_v0.3b.md` with an
E-1 marker. Section 8 itself is unchanged; only the limitations note was wrong.

**Addendum (same day) --- the second model section 0 names says the same thing.**
The false clause spoke of "either model tested". The second model is
`Qwen/Qwen2.5-0.5B`: a different family (QKV bias, GQA with `n_kv_heads = 2`,
`rope_theta = 1e6`, tied LM head), 24 layers, `hidden_size = 896`. Same prompt,
same 16 greedy steps: the two associations differ on **287,859 of its 834,176**
RMSNorm elements (34.51%, against 34.60% for SmolLM2-135M), the witness digest
moves from `c9dff099d9...` to `4df2b260ee...`, and the argmax digest and all
sixteen generated token ids do not move. Those sixteen token ids are also
token-for-token the ones the C reference implementation produced for this
checkpoint in `docs/E15d_bc_RESULT.md`. The witness digests are not comparable
across that boundary --- E15d(c) is a v0.2-era artifact and section 12.1's
witness chain changed between v0.2 and v0.3b, exactly as section 14.7 records
for SmolLM2 --- so the token ids are the quantity that carries the
cross-implementation claim. Section 14.5 now carries the second-model figure.

### E-2 (2026-09-09) --- section 14.4 was five orders of magnitude too pessimistic about the trig range

**What v0.3b said**, carried unchanged from v0.1:

> 14.4. **Trig polynomial accuracy is only validated for `|x| <= 14`** (unit
> tests sweep `x = i * 0.7` for `i` in `-20..=20`). [...] the two-part-pi
> reduction (section 6.3) has not been stress-tested at larger magnitudes
> where it could lose more precision. A clean-room implementer targeting a
> longer sequence than this spec's 20 positions should not assume this
> polynomial's accuracy holds unchanged.

**Why it is misleading.** Every sentence is accurate about what had been
measured. But it is the only thing the specification says about the range of
section 6.3, so a clean-room implementer reading it must conclude the pinned
reduction is unproven outside a 20-position decode -- which is the clause that
would stop anyone using CIS-2 at a real context length. Section 7.1 makes
`inv_freq[0]` exactly `1.0` and every later entry smaller, so the largest RoPE
angle a decode evaluates *is* its sequence length: the range question and the
context-length question are the same question.

**What was measured** (docs/E26_TRIG_RANGE.md). Every f32 bit pattern `x` with
`0 <= x < 2^31` -- 1,325,400,064 arguments, subnormals included -- evaluated
under the section 1.3 pinned environment against an f64 accuracy oracle:

* worst absolute error **9.4218e-8** for `|x| < 2^20` (0.790 ULP at 1.0),
  flat across nine orders of magnitude of argument;
* worst absolute error **2.0925e-7** for `|x| < 2^31` (1.756 ULP at 1.0);
* worst *relative* (ULP) error 2617, always near a zero of the function,
  where the absolute error is four to five orders of magnitude *smaller*
  than the flat figure -- 1.8596e-11 at the 2617-ULP point;
* `sin_pinned` exactly odd and `cos_pinned` exactly even in value over every
  f32 with `2^-126 <= |x| < 2^31`, with two documented exceptions that touch
  only the sign of a zero result and no reachable RoPE angle;
* direct enumeration of the RoPE angle multiset for both section 0 models at
  `L = 131,072`: worst absolute error 9.3815e-8, the same figure.

A first attempt that *sampled* 4096 arguments per binade reported max 1 ULP
through `2^16` and was thrown away: sampling never landed near a zero, so it
got the shape of the error wrong. For a single-argument f32 function the
domain is finite and enumerable, and nothing that licenses a MUST should be
established by sampling.

**Replacement text.** See section 14.4 as it now stands in
docs/CIS2_SPEC_v0.3b.md, which carries this erratum inline. Section 6.3 is
unchanged; only the limitations note was. Tier-1 op-level goldens at the
worst-case arguments were added in `cis2-verify/src/mathpin.rs`,
`mod large_angle`, together with a mutation control showing that an
f32-staged reduction fails them -- and also moves the section 12.1 witness
digest while leaving the argmax digest and the generated token ids untouched,
the third independent violation to show that pattern.

### E-3 (2026-09-09) --- section 3 admits exactly one `tokenizer.json`, and section 14 never said so

**What v0.3b said.** Section 3.1.3 pins `normalizer: null`; section 3.1.4 pins
the `pre_tokenizer` value `Sequence[Digits(individual_digits = true),
ByteLevel(add_prefix_space = false, use_regex = true)]`. Section 0 separately
names `Qwen/Qwen2.5-0.5B` as a second model used as evidence of correctness.

**What is missing.** Qwen2.5-0.5B's `tokenizer.json` ships an NFC normalizer and
`Sequence[Split(<GPT-4-style regex>, Isolated), ByteLevel(use_regex = false)]`.
A conforming implementation of section 3 must therefore *refuse* that
checkpoint --- and the clean-room verifier does, with
`tokenizer.json: spec 3.1.3 requires a null normalizer`, before any arithmetic
runs. So section 0's second model cannot be driven end-to-end from its
artifacts by a conforming CIS-2 v0.3b verifier. That was always true of the
text and was never stated; a reader could reasonably have read section 0 the
other way.

**What changes.** Nothing normative. Section 3 is unchanged and the refusal is
the correct behaviour --- the same "refuse rather than reinterpret" discipline
section 2.5 applies to a non-BF16 dtype. A new section 14.8 states the boundary,
and says what work on such a checkpoint is confined to (sections 4-11, prompt
token ids supplied from outside section 3, and a receipt that attests to
sections 4-11 and nothing of section 3). A future version wanting a second
normative tuple must generalize sections 3.1.3/3.1.4 from a pinned literal to a
small enumerated set.

### E-4 (2026-09-09) --- section 14.1 cautioned about the cold function, and did not know section 6.2's high guard is wrong

**What v0.3b said.** Section 14.1: "`ln_pinned`'s validated domain is a finite,
explicitly-tested set of `x` values (section 6.5), not a general accuracy
proof. The two values that matter for this spec's models (`100_000.0`,
`1_000_000.0`) are both tested to <=2e-6 relative tolerance against host
`f64::ln` cast to f32 ... **PARTIALLY CLOSED**."

**What is wrong with it.** Three things.

1. The tolerance is two orders of magnitude loose. `ln_pinned` is **0 ULP** ---
   bit-identical to the correctly-rounded f32 --- at both pinned thetas.
2. It cautions about the wrong function. `ln_pinned` runs once per model load
   (on `cfg.rope_theta`). `exp_pinned` runs twice per decode step: section 10's
   softmax evaluates `exp_pinned(v - max_v)` for every attention score, and
   section 6.4's SiLU calls `exp_pinned(-x)` for every FFN intermediate.
   Section 14.1 said nothing about it.
3. It did not know that section 6.2's high guard is wrong. Step 2 returns
   `+Infinity` for `x > 88.0`, but `ln(f32::MAX) = 88.7228390520684`. Exactly
   **94,743** f32 arguments in `[0x42B00001, 0x42B17217]` = [88.0000076,
   88.7228317] are clipped to infinity although their true `exp` is finite and
   exactly representable.

**What was measured.** `exp_pinned`, `ln_pinned` and `silu_pinned` were each
evaluated at all 2^32 f32 bit patterns under the section 1.3 pinned
environment against an f64 accuracy oracle. Outside section 6.2's two guard
bands, all 3,257,925,634 comparable `exp_pinned` arguments and all
2,130,706,432 positive-normal `ln_pinned` arguments are within **one ULP**;
both functions are exactly monotone over the whole finite domain;
`silu_pinned` is within **two ULP** off the clip band. Evidence and method:
`docs/E27_EXP_LN_RANGE.md`.

**What changes.** Nothing normative. **Section 6.2's clip stays as written and
is now stated to be normative.** Section 10's softmax cannot reach the
affected band (its argument is always `<= 0`), but section 6.4's SiLU can, for
an FFN intermediate in `[-88.7228317, -88.0000076]`; there `silu_pinned`
returns `-0.0` in place of a normal f32 of magnitude ~5.328e-37. A read-only
census of the section 13.1 reference decode records **0** of its 1,751,040
SiLU arguments in that band (observed range [-25.979437, 49.15266]) and
**0** softmax arguments below `-88.0`, with the pinned digests reproduced ---
so section 13.1 does not depend on the clip, though that is one model on one
prompt and is not a general unreachability result. Every
conforming implementation produces the same `-0.0`, so bit-exact agreement ---
the property CIS-2 claims --- is unaffected; widening the guard would be more
faithful to `exp` and would move every pinned digest. Section 14.1 is
rewritten to state the exhaustive result, to record the clip and the
`silu_pinned` discontinuity at `x = -88` as deliberate divergences that MUST
be reproduced, and to record that section 1.3's DAZ makes `ln_pinned` return
`-Infinity` for all 16,777,214 subnormal inputs. Section 6.5's carried-forward
"no equivalent sweep has been run" caution and section 6.7's accuracy table
are corrected to match. 48 op-level goldens with two structural mutation
controls now enforce the behaviour in CI
(`cis2-verify/src/mathpin.rs`, `mod exp_ln_range`).

### E-5 (2026-09-09) --- section 14.6's compensating-error gap is closed by measurement

**What v0.3b said.** Section 14.6, unchanged in kind since v0.1: "The oracle
correctness checks (section 13.3) are defensible spot-checks, not exhaustive.
They confirm greedy token-id agreement and one step's full-vocab logit
agreement to ~4e-6 relative on two model families now (SmolLM2-135M,
Qwen2.5-0.5B) --- neither checks every intermediate layer's activations
against the oracle, so a compensating pair of errors elsewhere in the layer
stack that happens to preserve step-0's output and all argmax decisions
cannot be completely ruled out by this evidence alone."

**What is wrong with it.** Nothing. It was an accurate statement of a real
hole --- section 13.3 looks only at the two ends of the pipe --- and it was
the last open gap of its kind in section 14. It is corrected here because the
measurement it asks for has now been made, not because it was mistaken.

**What was measured.** Every named intermediate of the forward pass, at every
position of a full decode, on both section 0 models, against a `transformers`
fp32 forward pass hooked at the corresponding modules:

- **13,452 tensors compared** (7,467 SmolLM2-135M, 5,985 Qwen2.5-0.5B), 0
  name or shape mismatches.
- **Worst relative L2 error over all of them: 9.923e-5** (2.228e-5 on the
  first model). `embed` agrees exactly on both.
- **No layer creates error.** Every layer's first computed tensor is within
  0.73-1.18x of the error it was handed, across all 30 layers of the first
  model and all 24 of the second; worst amplification of the carried-in error
  across a whole layer is 4.07x / 3.22x, uniform with depth. A divergent
  layer would show a large ratio at that layer alone, and a compensating one
  a ratio far below 1; neither occurs.
- The residual stream's apparent 8.79x spike at layer 9 is **cancellation,
  not divergence**: addends of norm 486.5 and 414.1 sum to norm 105.6, so the
  relative measure inflates by the 4.6x cancellation while every input to the
  add sits at 1-3e-6.
- The verifier runs each decode twice (section 12.4), so every tensor appears
  twice in its dump; **0 of the 13,452 duplicates differ**, which re-confirms
  determinism at the granularity of every intermediate rather than of the
  receipt.
- The oracle's own argmax reproduces all 16 generated ids on both models
  without reference to the verifier's output.

Evidence and method: `docs/E28_LAYER_ORACLE.md`. Instrumentation:
`cis2-verify --features layerdump` (a write-only activation tap; the
instrumented build reproduces every pinned digest, which is the check that it
only reads), `scripts/oracle_layers.py`, `scripts/compare_layers.py`.

**What changes.** Nothing normative --- no digest, coefficient or required
behaviour. Section 14.6 is rewritten from an open gap to a CLOSED clause
carrying the measurement; section 13.3 gains a forward pointer to it; section
0's cross-framework bullet records that the evidence now covers the interior
of the stack and restates that ~1e-5 agreement with `transformers` is what it
shows and all it could show, since bit-exactness is claimed between
conforming implementations of this document and never against a third-party
framework.

### E-6 (2026-09-09) --- section 1.5 is a limit on the optimizer's licence, not on optimization; measured at every intermediate

**What v0.3b said.** Section 1.5 forbids fast-math and reassociation, and
section 13.4 shows the two pinned digests reproducing across a 20-cell
compiler matrix. Neither passage said what an implementer needs to know
next --- whether a conforming build may be optimized at all --- and section
13.4's evidence is two 32-byte digests, not the computation between them.

**What is wrong with it.** Nothing is retracted. Both clauses stand as
written. What is added is the measurement they imply and did not make, and
a mechanism that explains why the result is structural rather than lucky.

**What was measured.** Three binaries from one source tree --- a default
`--release` build, a `-C target-cpu=native` build on an AVX2 part, and a
`-C target-cpu=native` build on a part without AVX2 --- run on both section
0 models, eight runs across two microarchitectures:

- **Every intermediate identical byte for byte.** SmolLM2-135M 7,467
  tensors / 51,750,154 B, sha256 `5386d3b0...`, all four runs. Qwen2.5-0.5B
  5,985 tensors / 103,942,814 B, sha256 `5ccf6520...`, all four runs. The
  AVX2 binary emits 397 `%ymm` instructions; the other two emit none. Zero
  FMA instructions in all three (section 1.4).
- **Why.** `matvec` is scalar in all three builds, including the AVX2 one:
  section 5.1's `acc = acc + p` is a serial floating-point dependency an
  optimizer denied fast-math may not reassociate. Section 8's sum-of-squares
  splits into an elementwise map and section 5.3's fold, and the AVX2 build
  vectorizes the map 8-wide while emitting the fold as a scalar `vaddss`
  chain. The specification pins exactly the operations whose order changes
  the result and leaves free exactly those whose order does not.
- **The vector path executes.** Replacing one `vmulps` of that loop with
  `ud2` in a copy of the binary terminates the run with SIGILL.
- **The dump is sensitive.** One flipped low-order mantissa bit of one
  weight in the 269 MB artifact moves 570 of 7,467 tensors --- entering at
  `L27.down_proj` in a single element, saturating L28 and L29, and moving
  all 19 `logits` vectors --- while changing no argmax decision at all. That
  is the measured form of section 12.1's requirement to hash the full logit
  vector rather than the emitted token ids.

**What changes in the document.** New section 13.5 records the result, the
mechanism and its scope limits. Section 1.5 gains the sentence that its
prohibition is on licensing reassociation and not on optimization, with a
pointer to 13.5. Full method, disassembly and provenance in
`docs/E29_OPTIMIZER_INVARIANCE.md`; the per-tensor comparison tool added
for it is `scripts/diff_dumps.py`.

**Extended the same day.** The entry above covered `--release` only, and
flagged opt-level as unmeasured at this granularity. That is now measured:
all ten `{opt-level 0,1,2,3,s} x {target-cpu generic, native}` cells of
section 13.4's matrix were re-run on x86_64. **Ten distinct binaries, one
dump** --- every cell reproduces `5386d3b0...` and the pinned section 13.1
digests, with zero FMA instructions throughout. Emitted AVX2 instructions
rise with optimization pressure (286 at `s`, 344 at 1, 397 at 2, 529 at 3)
while `dot_seq`'s multiply/add stay scalar in every cell, varying only in
unroll factor --- which changes the instruction count without changing the
order of the additions. This crate's `[profile.release]` is `opt-level = 2`,
so the two `--release` binaries of the entry above are the opt-level 2 row,
and the sweep reproduces their digests exactly.

**Scope.** Both hosts ran the same `rustc`/LLVM, so this is two code
generation targets and not two independent compilers; section 13.4 remains
the wider compiler axis and section 0's four-implementation convergence the
independent-implementation axis. One prompt, two models; the ten-cell sweep
is x86_64 only, so the aarch64 half of section 13.4's matrix has not been
re-run at intermediate granularity.

### E-7 (2026-09-09) --- the optimizer-invariance result and the section 14.1 reach census both extended from one input to six

E-6 (section 13.5) and E-5's reach census (section 14.1) each rested on a single
prompt. Neither claim needs a new decision to widen, only more runs, so both
were re-run over six prompt/length configurations: `"Once upon a time"`/16
(the section 13.1 reference decode), a 43-character sentence/16, a digit-heavy
prompt/16, a code prompt/16, `"Once upon a time"`/64 and `"A"`/32. All six
are ASCII, deliberately, so that section 3.1.4's still-open Unicode question
(section 14.8) could not confound a codegen or reachability result.

**Optimizer invariance (section 13.5).** Each configuration was run four times
--- the generic binary `c095d7f9...` on both hosts, plus `penguin`'s AVX2
`target-cpu=native` build `eaf8129c...` (397 `%ymm`) and `cm-box2`'s
`target-cpu=native` build `85df5f53...` (SSE only on a Celeron N4020,
0 `%ymm`). Twenty-four runs produced **six** dump digests --- one per
configuration, all six different from one another --- and within each
configuration all four runs agree on every byte of every intermediate
activation. Total compared: 558,394,570 bytes per sweep.

**Reach census (section 14.1).** Across the same six configurations the
instrumented build records **18,892,800** `silu_pinned` calls and
**2,355,480** softmax `exp_pinned` arguments, with **0** in section 6.2's
clip band `[-88.7228317, -88.0000076]` and **0** below `-88.0`. The most
negative SiLU argument observed anywhere is -31.406876, some 57 units short
of the band, and it occurs in the longest run --- the direction the range
would drift if context length were the driver.

**What changed in the text.** Section 13.5 gains a paragraph for the
six-input extension and its scope limit now reads "six ASCII prompts, at most
64 generated tokens, on two models" rather than "one prompt". Section
14.1(a)'s census figures are replaced with the six-configuration totals and
its honest limit now reads "one model, six ASCII prompts and at most 64
generated tokens". **No pinned digest moves**; the reference configuration
reproduces every value E-5 and E-6 recorded, which is the control on the
harness.

Method, per-configuration tables and provenance: `docs/E29_OPTIMIZER_INVARIANCE.md`
section 2b and `docs/E27_EXP_LN_RANGE.md` section 1. Re-derivable with
`scripts/e30_prompt_sweep.sh` / `scripts/e30_prompt_sweep_box2.sh` over
`scripts/e30_prompts.txt`.

### E-8 (2026-09-09) --- section 13.5's last scope limit closed: the aarch64 half of section 13.4's matrix, at intermediate granularity, as a CI gate

E-6 measured every intermediate activation across ten `{opt-level} x
{target-cpu}` cells, but on x86_64 only, and said so. That limit needed a
machine rather than a decision, so it was closed.

**All twenty cells of section 13.4's matrix** --- `{x86_64, aarch64} x
{opt-level 0,1,2,3,s} x {target-cpu generic, native}` --- now run at
intermediate granularity in `.github/workflows/verify.yml` (job
`intermediates`), on GitHub-hosted runners of both ISAs. Twenty distinct
binaries; every one produces
`5386d3b0e529d9817af86f2ba1381193c2b174b0f26d12e22a441616dafb2f64` --- all
51,750,154 bytes, 7,467 tensors --- reproduces the section 13.1 witness
digest, and disassembles to zero FMA-family instructions. Run 34372521833,
20/20 success.

Two observations recorded because they were measured, not because they
change the claim: on aarch64 the identity does **not** rest on the code
having stayed scalar --- NEON is architecturally baseline, so every cell
including `-O0` emits 875-936 vector instructions, with no
`generic`/`native` cliff of the kind x86_64 shows; and `target-cpu=native`
on the CI x86_64 runner emits 404/524/284 `%ymm` where the same source on
`penguin` emitted 397/529/286. The instruction counts are host-dependent.
The 51,750,154 bytes are not.

An independent cross-check, kept out of the evidence above on purpose: the
same aarch64 binary cross-compiled on `penguin` and run under `qemu-aarch64`
user-mode emulation reproduces the dump exactly. Emulated softfloat is the
wrong thing to trust for a bit-exactness claim, so the CI runners carry the
result --- but the crate's 47 library and 6 end-to-end tests pass under that
emulator too, including the six pinned FTZ/DAZ denormal goldens in
`cis2-verify/src/fpenv.rs`, so section 1.3's FPCR FZ pin is exercised on the
aarch64 path rather than assumed.

**This is now a gate.** Section 13.4 gates two digests across those cells;
this gates every intermediate in each of them. The difference is the one
E-6 measured: one flipped weight mantissa bit moves 570 tensors and **zero**
argmax decisions, so a divergence the new job catches is one an
output-digest job would let through. No pinned digest moves.

**Extended the same day (E32).** The scope note above listed LTO, PGO and
`codegen-units=1` as untested. LTO is the one that actually threatens the
result --- it changes what the optimizer can *see*, inlining across crate
boundaries, which is exactly where a reassociation invisible to per-unit
compilation could appear. Eight further cells, `{lto=fat, lto=thin,
codegen-units=1, lto=fat + codegen-units=1} x {generic, native}`: eight
distinct binaries, all reproducing the same dump and the same witness digest,
zero FMA. Thin LTO with `target-cpu=native` emits **1,539** AVX2
instructions --- 3.9x the plain native build of E-6, the most vectorized
binary measured in this work --- and computes the same bits. Under fat LTO
`matvec`, `dot_seq` and `rmsnorm` no longer exist as symbols, and the
whole-binary FP mix is still 102 `vaddss` against 21 `vaddps`: scalar chains
where section 5 pins an order, packed arithmetic where it does not, with
every boundary the optimizer could have crossed removed. **PGO remains
untested.**

Method, per-cell tables and provenance: `docs/E29_OPTIMIZER_INVARIANCE.md`
sections 2c and 2d. Re-derivable with `scripts/e32_lto_sweep.sh`.

## Repository releases

Version numbers above name the *specification* document. The section
below records what the repository shipped alongside it.

### 2026-09-09 --- release `v0.3b` (first tagged release)

The v0.3b spec text is unchanged; every pinned digest in section 13 is
unchanged. What is new in the repository since the spec was frozen:

- **Op-level conformance vectors** (`tests/conformance/`): five pinned
  input/expected pairs -- `matvec_v1`, `rmsnorm_v1`, `rope_v1`,
  `exp_pinned_v1`, `attention_block_v1` -- with a written protocol
  (`PROTOCOL.md`), so an implementer can localize a divergence to one
  operation instead of bisecting a whole decode. Previously the only
  conformance surface was the end-to-end `CIS2_REF` digest: pass or fail,
  no diagnosis.
- **Dual-ISA CI** (`.github/workflows/verify.yml`): every job now runs
  natively on both `ubuntu-24.04` (x86_64) and `ubuntu-24.04-arm`
  (aarch64) and asserts `uname -m` matches, so the cross-ISA claim is
  re-checked by a third party on every push rather than asserted from a
  local log. Jobs: `reference`, `conformance`, `verify2`, and `verify3`
  built with both gcc and clang -- 10 jobs, all green on `main`.
- **A GPU leg** (`docs/GPU_RESULT.md`): a CUDA port written from the
  v0.3b spec text reproduces the normative witness digest
  `d82743059d...` bit-for-bit on an NVIDIA Tesla P100-PCIE-16GB (sm_60,
  CUDA 12.8), with a byte-identical per-step trace against the CPU
  reference. This is a first-party result, not an independent
  replication; the CUDA source is not published. See that document's
  Scope section for what it does and does not establish.
- **A Hugging Face dataset card** (`docs/HF_DATASET_CARD.md`) and
  `scripts/publish_hf.sh`, which assembles the conformance payload from
  the working tree and refuses to upload if the v0.3b digest is missing
  from either the card or `EXPECTED_DIGESTS.md`.

## v0.2 (2026-08-28)

Three normative changes relative to v0.1, all affecting the pinned test
vector's `CIS2_REF` witness digest:

1. **Table digests folded into `CIS2_REF`.** The pinned transcendental
   coefficient-table digest (§6.6) and the RoPE `inv_freq` table digest
   (§7.2) are now inputs to the witness chain (§12.1 items 4-5), seeded
   after the three artifact hashes (weights, tokenizer, config).
2. **RoPE `inv_freq` is theta-general.** A new pinned `ln_pinned`
   polynomial (§6.5) replaces a bare literal that only worked for one
   `rope_theta` value, so the construction now supports any
   `rope_theta` read from `config.json`.
3. **Witness header uses raw 32-byte digests, not hex-ASCII strings.**
   Closes an encoding ambiguity in v0.1's witness chain (§12.1 items 1-3).

**§12.1 witness-order correction (post-v0.2, folded into this document):**
an earlier draft of §12.1 stated the witness chain's item order as
table-digest-then-inv_freq-then-artifact-hashes; the reference
implementation this spec was audited against actually feeds the three
artifact hashes first, then `table_digest`, then `inv_freq_table_digest`.
This mismatch was caught by an independent clean-room implementation built
strictly from the spec text: it reproduced every other pinned value
(`table_digest`, `inv_freq_table_digest`, `argmax_digest`, all 16
generated token ids) bit-for-bit but computed a different `CIS2_REF`
purely because it faithfully followed the incorrect prose. §12.1 in this
document states the corrected order; see docs/CIS2_SPEC_v0.2.md §12.1's
inline note and §16 for detail. No pinned digest value changed as a result
of this correction — only the prose describing the order was wrong.

Net effect on test vectors vs. v0.1: `CIS2_REF` changed
(`ba88708bf4...` -> `a0c563ef80...`); `table_digest` changed
(new `ln` coefficients folded in); `inv_freq_table_digest`, `argmax_digest`,
and `generated_token_ids` unchanged.

## v0.2.1 (2026-08-28)

Doc-only, no digest change. §3.1 rewritten from a one-paragraph pointer
into a self-contained byte-level BPE tokenizer specification (byte<->unicode
map, GPT-2 ByteLevel pre-tokenizer regex plus digit pre-split, rank-ordered
merge algorithm, special-token handling, and a worked example). No §13
test-vector value changed.

## v0.3 (2026-08-29, PARTIAL -- superseded by v0.3b)

Attempted to close CIS-2's one remaining stated technical limitation
(v0.2 section 6.7): `sin_pinned`/`cos_pinned`'s range reduction is
staged through f64 instead of two f32 half-constants, fixing
catastrophic-cancellation accuracy loss that grew with RoPE position (up
to 1104 ULP by position 19 in v0.2; ~0.58 relative, unusable, toward
position 8192).

**What changed:** section 6.3 (reduction only, not the degree-10 Taylor
polynomials); section 6.6 (`table_digest` now hashes one 8-byte
`TWO_PI_F64` constant in place of v0.2's two 4-byte halves).

**Net effect on test vectors:** `CIS2_REF` changed (`a0c563ef80...` ->
`90f7484e4c...`); `table_digest` changed (`465d358ccd...` ->
`986abc500e...`); `inv_freq_table_digest`, `argmax_digest`,
`generated_token_ids` unchanged from v0.2.

**PARTIAL result:** the reduction itself became accurate, but the fixed
degree-10 Taylor polynomial's own truncation error (worst near `|r| ~
pi` and at sin/cos zero crossings) remained dominant -- up to 2813 ULP,
missing the pre-registered <=2 ULP accuracy bar. Superseded by v0.3b.

## v0.3b (2026-08-29)

Closes CIS-2's one remaining stated technical limitation, meeting the
accuracy bar v0.3 missed. Refines the mechanism: section 6.3 now reduces
to an octant (`r` in `[-pi/4, pi/4]`, quadrant index `k mod 4`, still
f64-staged, same Cody-Waite pattern as v0.3 but mod `pi/2` instead of mod
`2*pi`), then evaluates `sin(r)`/`cos(r)` with separate degree-7/8
Cephes `sinf`/`cosf` minimax polynomials (not Taylor truncations),
selected and signed by quadrant.

**What changed:** section 6.3 (reduction period and polynomial, both);
section 6.6 (`table_digest` now hashes one 8-byte `PI_2_F64` constant
plus 6 pinned f32 minimax coefficients in place of v0.3's `TWO_PI_F64`
plus two 11-entry Taylor tables).

**What did not change:** `inv_freq_table_digest`, `argmax_digest`,
`generated_token_ids` -- unchanged from v0.1/v0.2/v0.3.

**Net effect on test vectors:** `CIS2_REF` changed (v0.3's
`90f7484e4c...` -> `d82743059d...`); `table_digest` changed (v0.3's
`986abc500e...` -> `23c7bfaf5c...`).

An accuracy harness (mpmath oracle, RoPE position domain up to
`max_position_embeddings=8192`) confirms max 1.5 ULP for both sin and
cos, meeting the pre-registered <=2 ULP bar. `verify2` (Rust) and
`verify3` (C) clean-rooms were both updated a second time from the
v0.3b spec text only (not from the reference implementation) and
reproduce `d82743059d...` bit-for-bit locally on x86_64; see each
crate's `CLEANROOM_LOG.md`. See `docs/CIS2_SPEC_v0.3b.md` (supersedes
`docs/CIS2_SPEC_v0.2.md` as the current normative document, which is
kept for history) and `docs/E15m_RESULT.md`/`docs/E15m_PREREG.md` for
full detail.

## v0.1 (2026-08-28)

First draft. Established the normative scope: fp32 transformer decode of
`HuggingFaceTB/SmolLM2-135M` on the pinned prompt, digest and receipt
format, floating-point environment (FTZ/DAZ, no FMA), reduction order,
transcendental polynomials, RoPE, RMSNorm, attention, MLP, and argmax.
