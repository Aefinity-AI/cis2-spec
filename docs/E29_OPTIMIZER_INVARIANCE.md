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

- **Not a second compiler.** Both machines ran the same
  `rustc 1.98.0 (88d9e12ae 2026-08-18)` and hence the same LLVM. This is
  two *code generation targets*, not two independent compilers. §13.4's
  matrix is the wider axis; the four-implementation convergence recorded in
  §0 is the independent-implementation axis.
- **Not exhaustive over inputs.** One prompt, 19 positions, two models —
  the same scope limit E28 carries.
- **Not exhaustive over optimizer settings**, though less narrow than it
  was: §2a re-ran all ten `{opt-level 0,1,2,3,s} × {generic, native}` cells
  of §13.4's matrix at intermediate granularity on x86_64. The aarch64 half
  of §13.4's matrix has not been re-run this way, and no non-default
  codegen flags beyond `target-cpu` were varied.
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
