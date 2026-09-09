# E29 — What the optimizer is allowed to do, measured at every intermediate

**Date:** 2026-09-09 · **Branch:** `cm/cis2-verify-standalone` · **Status:** complete

---

## 0. The question

§1.5 forbids fast-math and reassociation, and §5 pins the reduction orders.
§13.4 then shows that the two pinned digests reproduce across a 20-cell
matrix of `{x86_64, aarch64} × {opt-level 0..3,s} × {target-cpu generic,
native}`. Both are necessary. Neither answers the question an implementer
actually asks first:

> If the reduction order is pinned, does conformance mean shipping an
> unoptimized build?

and neither answers the question an auditor asks second:

> §13.4 compares two digests. A digest is 32 bytes. What happened to the
> other 51,750,122 bytes of the computation?

E28 closed the second question against an independent oracle (`transformers`)
to ~1e-5, which is what an accuracy check can show. E29 asks a sharper
version of it: not "does an independent implementation agree closely", but
"do *differently compiled* builds of this implementation agree **exactly**,
at every intermediate" — and then explains, from the disassembly, why.

The answer to the implementer's question turns out to be **no**: the
optimizer may vectorize, and does, and the bits do not move. The reason is
structural, not lucky, and is visible in the emitted code.

---

## 1. Method

`cis2-verify --features layerdump` (E28's tap, unchanged) records every
intermediate of the §13.1 decode: for each position and layer, `ln1`,
`q_proj`, `k_proj`, `v_proj`, `attn_out`, `o_proj`, `resid_attn`, `ln2`,
`gate_proj`, `up_proj`, `mlp_act`, `down_proj`, `resid_mlp`, plus `embed`,
`final_norm` and `logits` once per position.

Three binaries were built from the same source tree:

| build | where | `RUSTFLAGS` |
|---|---|---|
| baseline | penguin **and** box2 | *(none)* |
| native-AVX2 | penguin (i5-10210U, AVX2) | `-C target-cpu=native` |
| native-SSE | box2 (Celeron N4020, **no AVX2**) | `-C target-cpu=native` |

Each was run on two models (§0's SmolLM2-135M and Qwen2.5-0.5B), giving
eight dumps. The dumps were compared by `sha256sum` and, structurally, by
the new `scripts/diff_dumps.py`.

Both machines carry byte-identical model artifacts
(`model.safetensors` `80521b40…` for SmolLM2, `88c14255…` for Qwen), and
run the same glibc (2.41-12+deb13u3) and the same
`rustc 1.98.0 (88d9e12ae 2026-08-18)`.

---

## 2. Result: three binaries, one dump

| | sha256 | `%ymm` | FMA | `matvec` FP ops |
|---|---|---|---|---|
| baseline | `c095d7f9e113d7629c6f3e3ea949c30aa1b2a311417bf5b934efd67bba924d0c` | 0 | 0 | 5 `addss`, 5 `mulss` |
| native-AVX2 (penguin) | `eaf8129c66d7e986aad442b70afbca01aa6919cf8732c85fabca5982561d3ed0` | **397** | 0 | 9 `vaddss`, 9 `vmulss` |
| native-SSE (box2) | `85df5f53edc443c47b0eb9263c26c830e3606c988c948315bac304f88a57b6ab` | 0 | 0 | 1 `addss`, 1 `mulss` |

Three distinct binaries. Two microarchitectures. One of them emits 397
256-bit AVX2 instructions; the other two emit none.

| model | tensors | dump size | sha256, **all four runs** |
|---|---|---|---|
| SmolLM2-135M | 7,467 | 51,750,154 B | `5386d3b0e529d9817af86f2ba1381193c2b174b0f26d12e22a441616dafb2f64` |
| Qwen2.5-0.5B | 5,985 | 103,942,814 B | `5ccf65207a6638d7c2aa6111e0b08e867686b0efe5c07e4e9b2e6c6d048d550e` |

Eight runs, two digests. Every one of the 13,452 recorded tensors — every
f32 bit of every intermediate of both decodes — is identical across the
three builds and the two machines. Every run printed the pinned
`witness-digest` and `argmax-digest` (§13.1), which is the check that the
instrumentation itself changed nothing.

### 2a. Extended to the full opt-level axis

The first pass covered `--release` only. §13.4's compiler matrix also varies
`opt-level`, so the same experiment was re-run over all ten cells of
`{0, 1, 2, 3, s} × {generic, native}` on SmolLM2-135M. This crate's
`[profile.release]` sets `opt-level = 2`, so the two `--release` binaries of
§2 are the `opt-level 2` row, and the sweep reproduces their digests exactly.

| opt-level | target-cpu | binary sha256 (first 16) | `%ymm` | FMA | `matvec`/`dot_seq` FP ops |
|---|---|---|---|---|---|
| 0 | generic | `f9e7f38965c4331c` | 0 | 0 | 1 `addss`, 1 `mulss` |
| 0 | native | `473cfb3ae5d6124d` | 147 | 0 | 1 `vaddss`, 1 `vmulss` |
| 1 | generic | `f95d8587a7b5347c` | 0 | 0 | 5 `addss`, 5 `mulss` |
| 1 | native | `06bc8e7e7b7fd2c2` | 344 | 0 | 9 `vaddss`, 9 `vmulss` |
| 2 | generic | `c095d7f9e113d762` | 0 | 0 | 5 `addss`, 5 `mulss` |
| 2 | native | `eaf8129c66d7e986` | 397 | 0 | 9 `vaddss`, 9 `vmulss` |
| 3 | generic | `74715903bcfce9c5` | 0 | 0 | 5 `addss`, 5 `mulss` |
| 3 | native | `979299faa78eb1dc` | **529** | 0 | 9 `vaddss`, 9 `vmulss` |
| s | generic | `87e7d74188e99938` | 0 | 0 | 1 `addss`, 1 `mulss` |
| s | native | `2aee547352f5eac1` | 286 | 0 | 1 `vaddss`, 1 `vmulss` |

**Ten distinct binaries. One dump.** Every cell produced
`5386d3b0e529d9817af86f2ba1381193c2b174b0f26d12e22a441616dafb2f64`, 51,750,154
bytes, and printed the pinned `witness d8274305…` / `argmax 0b9c8f3a…`. Zero
FMA instructions in all ten.

The AVX2 instruction count rises monotonically with optimization pressure to
529 at `opt-level 3` — the optimizer is doing progressively more, on
progressively more of the program — and the reduction column does not move:
`dot_seq`'s multiply/add stay scalar in all ten cells, differing only in
unroll factor (1× at `opt-level 0` and `s`, 5× or 9× elsewhere), which
changes the instruction count without changing the order of the additions.

Two reporting notes, so the table is not over-read. At `opt-level 0` the
compiler does not inline `dot_seq` into `matvec`, so the FP ops of that row
are counted in `dot_seq` itself; and the 147 `%ymm` instructions of the
`0/native` cell are in the tokenizer's JSON parsing, the allocator and struct
moves — the arithmetic kernels at `opt-level 0` are not vectorized at all.
The informative cells are therefore `1/2/3/s × native`, where between 286 and
529 AVX2 instructions coexist with an unchanged dump.

The two baseline binaries are themselves byte-identical, although they were
built on different machines from different absolute source paths
(`~/projects/cis2-spec` vs `~/e28-src`) — an incidental reproducible-build
result, not the point of this experiment.

---

### 2b. Extended over inputs: six prompts, four runs each (E30)

§2 and §2a hold one input fixed. The obvious objection is that
`"Once upon a time"` might simply be an input that never produces an
intermediate where a reassociation would have been visible. That objection is
answered the same way the others were — by running more cells, not by
argument.

Six prompt/length configurations were chosen to move the arithmetic around:
different token counts, different context lengths, a digit-heavy prompt, a
code prompt, and a single-token prompt with 32 generated tokens. All are
ASCII, deliberately: §3.1.4's Unicode handling is still an open decision
(§14.8), and a tokenization question would have confounded a codegen result.

Each configuration was run **four** times — the same three binaries as §2
(`c095d7f9…` generic, byte-identical on both hosts; `eaf8129c…` penguin AVX2
`target-cpu=native`, 397 `%ymm`; `85df5f53…` box2 `target-cpu=native`, which
on a Celeron N4020 is SSE only, 0 `%ymm`), with the generic binary run on
both hosts. Twenty-four runs; six dump digests.

| prompt | gen | dump sha256 (16) | bytes | witness | argmax |
|---|---|---|---|---|---|
| `Once upon a time` | 16 | `5386d3b0e529d981` | 51,750,154 | `d8274305` | `0b9c8f3a` |
| `The quick brown fox jumps over the lazy dog` | 16 | `b5f70db17fd3589a` | 65,370,684 | `d385cef0` | `486f2b3a` |
| `1234567890 + 9876543210 =` | 16 | `860bbffaf566dc78` | 103,508,168 | `4caad3f9` | `3d55e42e` |
| `def f(x): return x * 2` | 16 | `88d45398a7b7763b` | 68,094,790 | `7675e353` | `0bb484df` |
| `Once upon a time` | 64 | `44a65ccc90f63915` | 182,507,242 | `fa75e57e` | `8b59866e` |
| `A` | 32 | `1b06bf6ba6bc63a1` | 87,163,532 | `84e4b87c` | `a28275bb` |

Six configurations, six *different* dumps — the inputs do reach different
arithmetic, which is what makes the comparison worth making — and within each
configuration all four runs agree on every byte of every intermediate. The
first row reproduces §2's and §2a's known values, which is the control that
the harness is comparing what it claims to.

Aggregate: 558,394,570 bytes of intermediate activations per sweep,
bit-identical across two microarchitectures and three codegen targets.

The same six runs also carry the §14.1 reach census, which is E27's result and
is written up there: 18,892,800 `silu_pinned` calls and 2,355,480 softmax
`exp_pinned` arguments across the six configurations, **0** in §6.2's clip
band and **0** below `-88.0`.

---

### 2c. The aarch64 half, on real silicon, and as a standing gate (E31)

§2a's ten cells were x86_64 only. That was the last scope limit in §6 that
needed no new decision — only a machine — so it was closed rather than
argued, and closed in the place that keeps it closed: **all 20 cells of
§13.4's matrix now run in CI at intermediate granularity**,
`{x86_64, aarch64} × {opt-level 0,1,2,3,s} × {target-cpu generic, native}`,
on GitHub-hosted runners of both ISAs.

Twenty distinct binaries. One dump.

| opt | target-cpu | x86_64 binary (16) | `%ymm` | aarch64 binary (16) | NEON |
|---|---|---|---|---|---|
| 0 | generic | `a776cbb8e0af8e49` | 0 | `5e4f66d4e72c4b7f` | 875 |
| 0 | native | `ca81c9ae12183a1d` | 147 | `b9016dac1bb57d20` | 875 |
| 1 | generic | `86b0d799b046f6c4` | 0 | `ccaf4da301a8d940` | 902 |
| 1 | native | `f2fda3541e8edcd6` | 344 | `8c958caa4d98dcc5` | 894 |
| 2 | generic | `54c8d2d08accd729` | 0 | `4a342fa92c9d2960` | 927 |
| 2 | native | `849558a265db8031` | 404 | `d94e09cce1652fd3` | 890 |
| 3 | generic | `4abab7f0e580eb19` | 0 | `b7057178a24ea474` | 936 |
| 3 | native | `7db4ed2ca4880713` | 524 | `10f8dd3d5284e8a9` | 898 |
| s | generic | `d2e4507ff93c6497` | 0 | `c0d2e3d95b8564e8` | 889 |
| s | native | `46a580aee17f0b60` | 284 | `b40fb0b50c7e089f` | 885 |

Every one of the twenty cells produced
`5386d3b0e529d9817af86f2ba1381193c2b174b0f26d12e22a441616dafb2f64` —
all 51,750,154 bytes, all 7,467 tensors — and reproduced the §13.1 witness
digest, with **zero** FMA-family instructions in every cell's disassembly.

Two things in that table are worth reading carefully, and neither is a
semantic claim:

- **aarch64 vectorizes at every optimization level, including `-O0`, and
  without `target-cpu`.** NEON is architecturally baseline in AArch64, so
  there is no `generic`/`native` cliff of the kind x86_64 shows (0 `%ymm`
  at every generic cell, because SSE2 is *its* baseline and the counter only
  looks for AVX). The identity therefore is not resting on aarch64 having
  quietly stayed scalar — it emits 875–936 vector instructions throughout
  and still lands on the same bytes.
- **On aarch64 `native` often emits *fewer* vector instructions than
  `generic`** (927 → 890 at opt-level 2). That is a scheduling and
  instruction-selection difference on a known microarchitecture, not a
  reduction in vectorization, and it is reported here only because the count
  is what was measured. The count is host-dependent; the dump is not.

The x86_64 `native` counts also differ slightly from §2a's — 404 and 524 and
284 here against 397 and 529 and 286 on `penguin` — because `target-cpu=native`
resolves to a different part on the CI runner. Same source, different feature
set, different instruction selection, same 51,750,154 bytes. That divergence
in the *counts* alongside identity in the *bytes* is the cleanest single
illustration of what §1.5 buys.

**Independently, under emulation.** Before the CI matrix existed, the same
aarch64 binary was cross-compiled on `penguin`
(`aarch64-unknown-linux-gnu`, linker `aarch64-linux-gnu-gcc`) and run under
`qemu-aarch64` user-mode emulation: same dump digest, same witness and argmax
digests, zero FMA. That run is *not* the evidence above — emulated softfloat
is exactly the wrong thing to trust for a bit-exactness claim — but it is a
useful independent execution environment, and one control from it is worth
keeping: the crate's 47 library tests and 6 end-to-end tests pass under
`qemu-aarch64`, including the six pinned FTZ/DAZ denormal goldens in
`src/fpenv.rs` that each go through `black_box` in both directions. §1.3's
FPCR FZ pin is therefore exercised on the aarch64 path rather than assumed.

**This is now a gate, not a measurement.** The `intermediates` job in
`.github/workflows/verify.yml` fails the build if any cell's dump digest
moves. §13.4 gates two digests; this gates 51,750,154 bytes. The difference
matters for the reason §5 measured: one flipped weight mantissa bit moves 570
tensors and **zero** argmax decisions, so a divergence this job catches is one
an output-digest job would let through.

---

### 2d. Link-time optimization, the setting most likely to break this (E32)

§2c's scope note listed LTO, PGO and `codegen-units=1` as untested. LTO is
the one of those that actually threatens the result, and it is worth saying
why before the numbers: opt-level changes what the optimizer does *within* a
compilation unit, while LTO changes what it can *see* — every function
inlined into every caller, across crate boundaries, which is precisely the
condition under which a reassociation opportunity invisible to per-unit
compilation could appear.

Eight more cells, `{lto=fat, lto=thin, codegen-units=1, lto=fat +
codegen-units=1} × {generic, native}`:

| flags | target-cpu | binary (16) | `%ymm` | FMA | dump |
|---|---|---|---|---|---|
| `lto=fat` | generic | `3ac32ea09a2a8b52` | 0 | 0 | `5386d3b0…` |
| `lto=fat` | native | `3d6ae879f14db7ec` | **1388** | 0 | `5386d3b0…` |
| `lto=thin` | generic | `3f3d5fae61bf1c6e` | 0 | 0 | `5386d3b0…` |
| `lto=thin` | native | `0d36165792274cec` | **1539** | 0 | `5386d3b0…` |
| `codegen-units=1` | generic | `cf6ce388fd339e3b` | 0 | 0 | `5386d3b0…` |
| `codegen-units=1` | native | `9d9291e05bf5309c` | 387 | 0 | `5386d3b0…` |
| `lto=fat, cgu=1` | generic | `ab40afca6e2de1c0` | 0 | 0 | `5386d3b0…` |
| `lto=fat, cgu=1` | native | `008393200659a4b1` | 1362 | 0 | `5386d3b0…` |

Eight distinct binaries, every one reproducing witness `d8274305…` and the
same 51,750,154 bytes. Thin LTO with `target-cpu=native` emits **1,539**
AVX2 instructions — 3.9× the 397 of §2's plain native build, and the most
vectorized binary measured anywhere in this document. It computes the same
bits.

Under fat LTO the interesting functions no longer exist as symbols: `matvec`,
`dot_seq` and `rmsnorm` are inlined into their callers, so §3's per-function
disassembly cannot be repeated cell-for-cell. The whole-binary floating-point
mix still shows the shape §3 describes — **102 `vaddss` against 21 `vaddps`**,
95 `vmulss` against 16 `vmulps` — scalar chains where §5 pins an order,
packed arithmetic where it does not, even with every boundary the optimizer
could have crossed removed.

That is the §1.5 mechanism stated as strongly as this document can state it:
give LLVM the whole program, an AVX2 target and permission to inline
everything, and it still may not reassociate a floating-point reduction,
because nothing in the build licensed it to. That leaves PGO, which §2e
measures.

---

### 2e. Profile-guided optimization, the last untested flag (E33)

Everything above is a *static* decision: the optimizer chose from the source
and the target description. PGO adds the one input those cells could not
supply — a measurement of where the program actually spends its time — and
that is the input most likely to make a compiler restructure a hot loop.

The profile says exactly what one would fear. `llvm-profdata show` on the
merged profile reports 293 functions, 3,173 blocks and 11,868,112,190
counted events, and the largest internal block count in the whole program is

```
_RNvNtCslqp2CDIoJoR_11cis2_verify3ops6matvec, max count = 10220470272
```

`ops::matvec` — 1.02 × 10¹⁰, two orders of magnitude above the next entry
(`sha256::compress`, 5.5 × 10⁸). That is the function §3 shows to be scalar
and §5 shows to pin the reduction order. So PGO enters this experiment
knowing that 86 % of the program's counted activity is inside the one loop
whose serial dependency chain is the obvious thing to break.

Two profiles were trained, both on the same instrumented binary: `once` from
the reference prompt, and `cross` from `1234567890 + 9876543210 =` — a
different prompt, a different sequence length, and by E30 a different dump
digest. The `cross` cells are the sharper test: the profile that guides the
build was collected on a workload that is *not* the workload being verified.

| cell | target-cpu | binary (16) | `%ymm` | FMA | dump |
|---|---|---|---|---|---|
| `profile-generate` (instrumented) | generic | `8423f510bb4aaf96` | 40 | 0 | `5386d3b0…` |
| `profile-generate` (instrumented) | native | `0a71b3d5bebba233` | 434 | 0 | `5386d3b0…` |
| `profile-use=once` | generic | `569c7a5794b02971` | 0 | 0 | `5386d3b0…` |
| `profile-use=once` | native | `6496f08d546d97be` | 398 | 0 | `5386d3b0…` |
| `profile-use=cross` | generic | `0bcfc13ee57da0d5` | 0 | 0 | `5386d3b0…` |
| `profile-use=cross` | native | `a451167f6ec1e555` | 364 | 0 | `5386d3b0…` |
| `profile-use=once` + `lto=thin` | native | `537ed4055464df95` | **1529** | 0 | `5386d3b0…` |
| `profile-use=cross` + `lto=fat` | native | `dd74a61ad4ca6a05` | 1316 | 0 | `5386d3b0…` |

Eight more distinct binaries, every one reproducing witness `d8274305…`,
argmax `0b9c8f3a…` and the same 51,750,154 bytes.

Two controls, because "the flag was silently dropped" is the failure mode
that would make this table meaningless:

1. **The profile changed the code.** At `opt-level=3, target-cpu=native` the
   non-PGO build of §2a is `979299fa…` with 529 `%ymm`; the same settings
   with `profile-use=once` give `6496f08d…` with 398, and with
   `profile-use=cross` `a451167f…` with 364. Three different binaries, three
   different vector-instruction counts — the profile is being read, and a
   profile trained on a *different prompt* produces a *different binary* from
   one trained on the verified prompt. Both compute the same bits.
2. **rustc validates the flag.** `-C profile-use=/nonexistent/nope.profdata`
   fails the build with `error: file … does not exist`, so a mistyped path
   cannot masquerade as a passing PGO cell.

A third observation falls out for free: the *instrumented* binary is itself
conforming. Running it on the `cross` prompt reproduced E30's
`860bbffaf566dc78…` and witness `4caad3f9…` — counter increments on every
basic block, and the arithmetic is unmoved.

Under fat LTO with a cross-trained profile the floating-point mix is
**102 `vaddss` against 21 `vaddps`, 95 `vmulss` against 16 `vmulps`** —
identical to E32's fat-LTO census. PGO moved block layout and inlining (the
binary and its `%ymm` count both differ from E32's `008393200659a4b1…`) and
did not move the scalar/packed split by a single instruction. That split is
what §1.5 pins, and it is not a performance decision the profile is allowed
to revise.

With this cell, every codegen flag §13.5 nominates is measured:
`opt-level`, `target-cpu`, `lto`, `codegen-units` and `profile-use`.

---

## 3. Why it holds: what the optimizer actually did

This is the part that generalizes beyond these two machines.

`ops::matvec` — with `dot_seq` inlined into it, and by a wide margin the
dominant cost of the decode — is **scalar in all three builds**, including
the AVX2 one. The source is §5.1's pinned order:

```rust
let mut acc = 0.0f32;
for i in 0..a.len() {
    let p = a[i] * b[i];
    acc = acc + p;
}
```

`acc = acc + p` is a serial floating-point dependency. Vectorizing it would
require reassociating the additions, which LLVM may not do without
fast-math — which §1.5 forbids the implementer from enabling. So the
compiler leaves it alone. Not by configuration: by construction.

`ops::rmsnorm` shows the complementary half. Its AVX2 FP instruction mix is
13 `vmulps`, 10 `vaddss`, 3 `vmulss`, 2 `vdivss` — packed *and* scalar in
the same function. The disassembly says exactly which is which:

```
3401b: vmulps %ymm0,%ymm0,%ymm0        # square 8 lanes
34023: vmulps %ymm2,%ymm2,%ymm2
3402b: vmovups %ymm0,(%r14,%rsi,1)     # ...and STORE them
34046: sub    $0xffffffffffffff80,%rsi # 32 floats per iteration
3404d: jne    34000
```

The vector loop squares 32 elements per iteration and **stores** the
squares. It never accumulates them. That matches the source exactly:

```rust
for i in 0..n { sq[i] = x[i] * x[i]; }   // map   — order-independent
let ss = sum_seq(&sq);                    // fold  — order-pinned by 5.3
```

The map is elementwise, so its result does not depend on lane width or
iteration order, and the compiler is free to widen it — 8-wide `vmulps`
here, with a 4-wide `xmm` remainder loop and a scalar tail. The fold is
§5.3's sequential sum, so the compiler is not free, and it emits the scalar
`vaddss` chain. The same split appears in `forward` (20 `vaddps` for
elementwise residual adds, 26 `vaddss` for reductions).

**The spec constrains exactly the operations whose order changes the
result, and leaves free exactly those whose order does not.** That is a
sufficiency property of the specification, and it is the reason the
identity above is robust rather than coincidental: an optimizer can only
touch the parts where touching them is a no-op on the bits.

The practical consequence for an implementer: `--release` with
`-C target-cpu=native` is conforming and was measured to be conforming.
Conformance does not cost the elementwise vectorization; it costs only the
scalar reduction, which the specification requires by design.

---

## 4. That the vectorized path executes (not merely exists)

397 `%ymm` instructions in a binary is not evidence that any of them ran; a
skeptic can reasonably ask whether the vectorized loop is guarded by a
length check that the real shapes never satisfy.

Direct test. In a copy of the native-AVX2 binary, the first `vmulps` of the
`rmsnorm` vector loop above (a 16-byte block occurring exactly once in the
file, at offset `0x3301b`) was overwritten with `ud2; nop; nop`
(`0f 0b 90 90` — same length, guaranteed-invalid opcode). Running that copy
on the §13.1 decode:

```
Illegal instruction     layer_dump_ud2 weights e28-ud2.bin
exit=132                # 128 + SIGILL(4)
```

The process dies on that instruction. The 8-wide AVX2 squaring loop is on
the executed path of the run that produced dump `5386d3b0…`.

This establishes execution for that instruction, and so for the vector loop
containing it. It does not individually establish execution of all 397.

---

## 5. Negative control: is the dump sensitive enough for §2 to mean anything?

An identity result is worthless without a calibration: a quantity that
never changes is also "identical". So the same experiment was run against a
deliberately corrupted artifact.

One byte of `model.safetensors` was flipped at offset 200,000,000
(`0xcd → 0xcc`) — a single low-order mantissa bit of a single f32 weight.
Comparing that dump to the reference with `scripts/diff_dumps.py`:

| | |
|---|---|
| bytes differing | **2,680,060 of 51,750,154** (5.2 %) |
| tensors differing | **570 of 7,467** |
| first differing record | `p0.L27.down_proj` — **1 of 576 elements** |
| spread | all of L28 and L29 (247/247 each), all 19 `final_norm`, all 19 `logits` |
| worst | `p17.L29.resid_attn`, max \|Δ\| = 1.731873e-03, 568/576 elements |
| `witness-digest` | **moves** → `c459040fb8c14e799126a241f796f644720a61a387b68f305c0045a64f41a76b` |
| `argmax-digest` | **unchanged** → `0b9c8f3a…` |
| generated token ids | **unchanged** |

The perturbation enters at the layer whose weight was touched, in one
element, and has saturated the residual stream two layers later. Every
logit vector of every position moved.

And not one token changed.

This is a direct, quantified statement of why §12.1 hashes the full logit
vector rather than the emitted token ids: a corrupted weight was visible in
100 % of the logit vectors and in 0 % of the decisions. A receipt that
committed only to token ids would have accepted this artifact. The
digest-level version of this observation was already recorded for the
clean-room implementation; E29 measures where in the network it happens.

It also calibrates §2: the dump moves for a one-bit change in a 269 MB
artifact, so eight runs agreeing on all 51,750,154 bytes is a measurement,
not a tautology.

---

## 6. What this does **not** establish

- **Not a second compiler.** Every cell — local and CI — ran
  `rustc 1.98.0 (88d9e12ae 2026-08-18)` and hence the same LLVM. §2c widens
  the *ISA* and *code generation* axes, not the compiler axis. The independent-compiler axis lives elsewhere: `verify3/`'s
  four CI cells (`{x86_64, aarch64} × {gcc, clang}`, §13.3) build a
  clean-room C implementation with two compilers that are not LLVM-Rust,
  and the four-implementation convergence recorded in §0 is the
  independent-implementation axis. Neither of those compares intermediates.
- **Not exhaustive over inputs**, though no longer a single input: §2b ran
  six prompt/length configurations, each on three binaries across two
  microarchitectures, and each configuration's four runs agree at every
  intermediate. Six prompts on two models is still not a claim about all
  inputs; the longest context measured is 64 generated tokens, and every
  prompt is ASCII (§3.1.4 is an open decision, §14.8).
- **Not exhaustive over optimizer settings.** All twenty cells of §13.4's
  matrix — both ISAs × five opt-levels × `{generic, native}` — now run at
  intermediate granularity (§2a locally, §2c in CI on real runners of both
  ISAs), and §2d adds eight more cells for `lto=fat`, `lto=thin` and
  `codegen-units=1`, and §2e eight more for `profile-generate` and
  `profile-use` (including a profile trained on a different prompt than the
  one verified). Every codegen flag §13.5 nominates is now measured; **any
  flag outside `{opt-level, target-cpu, lto, codegen-units, profile-use}` is
  untested**, and of those five only the first two are gated in CI — §2d's
  and §2e's cells are measurements, re-runnable from
  `scripts/e32_lto_sweep.sh` and `scripts/e33_pgo_sweep.sh`, not standing
  checks.
- **The `ud2` test covers one instruction**, and so one vector loop, not all
  397 `%ymm` instructions.
- **The negative control is one bit at one offset.** It calibrates
  sensitivity; it is not a claim that every possible corruption is detected.
  E23 R4's 618-mutant matrix is the systematic version of that question for
  the episode receipt.
- **No timing claims.** penguin is a crosvm VM and box2 was not quiesced
  (Rule A).

---

## 7. Provenance (Rule B)

| | |
|---|---|
| Hosts | `penguin` — ChromeOS Crostini (crosvm VM), Debian 13 trixie, x86_64, i5-10210U, 8 cores, AVX2. `cm-box2` (192.168.11.69) — bare metal, Debian 13, Celeron N4020, 2 cores, 3.7 GB, **no AVX2**. |
| glibc | 2.41-12+deb13u3 on both |
| Toolchain | `rustc 1.98.0 (88d9e12ae 2026-08-18)`, `cargo 1.98.0`, `--release --offline --features layerdump`, both hosts |
| FP environment | `x86_64 MXCSR FTZ(bit 15)+DAZ(bit 6)`, `fpenv::pin_and_selftest()`, every run |
| Artifacts | SmolLM2 `model.safetensors` `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`; Qwen `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`; verified identical on both hosts before any run |
| Binaries | baseline `c095d7f9…` (bit-identical on both hosts), penguin native `eaf8129c…`, box2 native `85df5f53…`; 0 FMA instructions in all three. §2a adds seven more penguin binaries across `opt-level 0,1,3,s`, also 0 FMA |
| Opt-level sweep | `~/e29-sweep.sh`, log `~/e29-optlevel-sweep.log`, 10 cells, all reproducing dump `5386d3b0…` and the pinned digests |
| CI matrix (§2c) | `.github/workflows/verify.yml` job `intermediates`, run `34372521833` on `cm/cis2-verify-standalone`, 20/20 cells success; runners `ubuntu-24.04` (x86_64) and `ubuntu-24.04-arm` (aarch64), real hardware, `rustc 1.98.0`; every cell asserts the dump digest and the §13.1 witness digest and greps its own disassembly for FMA |
| qemu cross-check (§2c) | `aarch64-unknown-linux-gnu` cross-build on `penguin`, run under `qemu-aarch64` user-mode with `-L /usr/aarch64-linux-gnu`; same dump digest; 47 lib + 6 end-to-end tests pass under the same emulator, including the `src/fpenv.rs` FTZ/DAZ denormal goldens |
| LTO sweep (§2d) | `scripts/e32_lto_sweep.sh`, log `~/e32-lto-sweep.log`, 8 cells on `penguin`, all reproducing dump `5386d3b0…` and witness `d8274305…`, 0 FMA; a ninth fat-LTO build at the profile's own `codegen-units` (`b129e583…`, 1,372 `%ymm`) likewise |
| PGO sweep (§2e) | `scripts/e33_pgo_sweep.sh`, log `~/e33-pgo-sweep.log`, 8 cells on `penguin`; `llvm-profdata` from the matching `llvm-tools` component (rustc 1.98.0 carries LLVM 22.1.8, so the distro's LLVM 19 `llvm-profdata` cannot read these `.profraw` files); profiles `once` (87,672 bytes) and `cross` (87,384 bytes, trained on `1234567890 + 9876543210 =`); all 8 cells reproduce dump `5386d3b0…` and witness `d8274305…`, 0 FMA. Discrimination: `-C profile-use` on a nonexistent path fails the build, and the three `opt-level=3, native` binaries (`979299fa…`/529 `%ymm` non-PGO, `6496f08d…`/398 with `once`, `a451167f…`/364 with `cross`) are three distinct binaries |
| Prompt sweep (§2b) | `scripts/e30_prompt_sweep.sh` (penguin) and `scripts/e30_prompt_sweep_box2.sh` (box2) over `scripts/e30_prompts.txt`; logs `~/e30-sweep.log`, `cm-box2:~/e30-box2.log`; 6 configurations × 4 runs, 6 dump digests, every configuration's four runs identical |
| Dumps | SmolLM2 `5386d3b0e529d9817af86f2ba1381193c2b174b0f26d12e22a441616dafb2f64` (4/4 runs), Qwen `5ccf65207a6638d7c2aa6111e0b08e867686b0efe5c07e4e9b2e6c6d048d550e` (4/4 runs) |
| Digests reproduced by every run | SmolLM2 `witness d8274305…`, `argmax 0b9c8f3a…`; Qwen `witness c9dff099…`, `argmax 9619177f…` |
| Negative control | flipped byte at offset 200,000,000, `0xcd → 0xcc`; witness `c459040f…`, argmax unchanged |
| Tool | `scripts/diff_dumps.py`, exercised on both a matching pair (`VERDICT identical`, 7,467/7,467) and a differing pair (`VERDICT differs`, 570/7,467) |
| Qwen runs | supply prompt token ids `[12522, 5193, 264, 882]`; §3.1.3/3.1.4 refuse that `tokenizer.json`, so those runs exercise §4–11 only and are not conformance runs (§14.8 / E-3) |
| Gates | `cis2-verify/tools/check_no_fma.sh` PASS; lib + end-to-end tests pass; `cis2-verify run ../weights` reproduces the pinned digests, `conformance=PASS` |

Re-derivable end to end: build with and without `-C target-cpu=native`, run
`layer_dump` on each, `sha256sum` the dumps, and run `scripts/diff_dumps.py`
on any pair. The `ud2` and bit-flip controls are two short scripts over the
same binaries.
