# GPU result — the §13.1 digest reproduced on an NVIDIA GPU

**Status: PASS. Run of 2026-09-08 12:07 UTC, NVIDIA Tesla P100-PCIE-16GB
(sm_60, Pascal), CUDA 12.8.**

An independent CUDA implementation of `docs/CIS2_SPEC_v0.3b.md` reproduces the
**primary normative `CIS2_REF` digest of this repository, bit-for-bit, on a
GPU** — the same value `scripts/self_check.sh` checks `verify3/` against on a
CPU.

```
CIS2_REF, SmolLM2-135M, "Once upon a time", gen_toks=16, spec v0.3b

  CPU reference (Rust, x86_64 and aarch64)  d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df
  CPU host build (g++, x86_64)              d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df
  GPU (nvcc, Tesla P100, sm_60)             d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df
```

## All four vectors

| mode | `gen_toks` | `CIS2_REF` | CPU | GPU |
|---|---|---|---|---|
| sequential (v0.3b normative) | 16 | `d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df` | PASS | PASS |
| sequential (v0.3b normative) | 128 | `22f69ad87a615d66a77efaca8b1172d22bdfb4bd5092ceffd302e35669a050f6` | PASS | PASS |
| pinned tree (v0.4 **candidate**, not normative) | 16 | `3f3b1ffceca02e8e0e78d5c653963480ea988ef41a378acc5393006ad378e1b3` | PASS | PASS |
| pinned tree (v0.4 **candidate**, not normative) | 128 | `112661fbfcb11440ae7d27c43e45135f8a6ff9f6bc01189d756c76dac96a0fdf` | PASS | PASS |

The 16-token sequential digest is the §13.1 normative vector recorded in
`EXPECTED_DIGESTS.md`. The other three are **informative**: the 128-token
sequential digest extends the same prompt to a longer horizon, and the two
tree-mode digests belong to a *candidate* reduction order that is **not part of
v0.3b** and may or may not become v0.4.

The expected values and the CPU reference traces were generated on x86-64
(g++ and clang++) and frozen on 2026-09-03, five days before the GPU run.

## Identity is byte-level, not only digest-level

The per-step trace — every intermediate the spec pins, not just the final
witness — is the same file on CPU and GPU:

```
sha256(GPU 16-tok trace)  = 9a559a9e284c5e5b2a78b29c0dbab0e2bb2ce0c2e48069dd6bac20331cad46b6
sha256(CPU 16-tok trace)  = 9a559a9e284c5e5b2a78b29c0dbab0e2bb2ce0c2e48069dd6bac20331cad46b6

sha256(GPU tree trace)    = 40c931cd080da04f8e926f84e7f6f5605eada74cd8f858baa4be0d6ae955f0bf
sha256(CPU tree trace)    = 40c931cd080da04f8e926f84e7f6f5605eada74cd8f858baa4be0d6ae955f0bf
```

The two modes hash differently from each other, which is the control that the
comparison discriminates.

## How device execution was established

A GPU claim is only as good as the evidence that the code ran on the GPU.
Two independent checks, both machine-recorded:

1. **`cuobjdump --dump-sass` on the shipped binaries** (not on a probe built
   for the occasion) finds 13 kernel instantiations covering the whole forward
   pass — embed, RMSNorm, matvec, RoPE, scores, softmax, weighted sum, KV
   store, SiLU-multiply, add, add-bias. There is no host fallback path in the
   nvcc build.
2. **`nvidia-smi` sampled once per second** during the 128-token runs: mean GPU
   utilization 58.2 % (max 87 %) sequential and 61.6 % (max 89 %) tree, across
   10 and 11 samples.

This is occupancy evidence for *where the code ran*. It is not a performance
measurement, and no timing number is claimed here or anywhere in this
repository.

## The FMA gate, on the real binaries

The spec forbids FMA contraction in the pinned reductions. Compiled with
`-fmad=false -ftz=true -prec-div=true -prec-sqrt=true`, the shipped binaries
contain 63 `FFMA` instructions each, and none of them are in a reduction:

| kernel | FFMA | MUFU |
|---|---|---|
| matvec, scores, attention weighted-sum | **0** | — |
| embed, add, add-bias, KV store | **0** | — |
| RMSNorm (2 of 3 instantiations) | **0** | — |
| SiLU-multiply | 12 | 3 |
| softmax | 32 | 7 |
| RoPE | 1 | 3 |
| RMSNorm (1 instantiation) | 18 | 8 |

Every kernel that contains `FFMA` also contains `MUFU` — the reciprocal and
reciprocal-square-root seed instructions whose Newton–Raphson refinement steps
`-prec-div=true -prec-sqrt=true` emit for `/` and `sqrt`. Those refinements are
correctly rounded by construction; the bit-identical digest and the
byte-identical trace are the empirical evidence that they are, in this build,
on this device.

## Scope — what this does and does not show

**Shows.** A specification written and audited for CPU fp32 — pinned reduction
order, no FMA contraction, no flush-to-zero, correctly rounded division and
square root, pinned transcendental polynomials, pinned RoPE tables — is
*sufficient* for an implementation on a fundamentally different execution model
to reproduce the CPU receipt bit-for-bit.

**Does not show.** Anything about performance; the run was correctness-only.
Anything about Ampere, Hopper or Blackwell; this is one Pascal device.
Anything about tensor-core paths; this port uses none. Anything about batched,
multi-stream, or multi-GPU execution. Anything about any model other than
SmolLM2-135M, or any horizon beyond 128 tokens.

One GPU generation, one driver and compiler version, one model, one prompt.

## Prior art, and the wording of the claim

Reproducible and deterministic GPU inference is prior-occupied ground. Gensyn's
`repops` demonstrates a hash-matched CPU↔CUDA fp32 forward pass; Microsoft's
RepDL provides reproducible linear-algebra operators with CPU and CUDA
backends; vLLM and SGLang both ship batch-invariant determinism modes. This
repository makes **no "first" and no "only" claim** about deterministic GPU
inference, and none should be inferred from this document.

The narrower claim being made is about the *specification*: the artifact a
third party implements against here is a written document, and that document
turned out to carry enough information to cross an ISA boundary, a compiler
boundary, a language boundary, and now a CPU/GPU boundary without any of the
implementations consulting each other.

## Availability of the GPU implementation

The CUDA port is **not included in this repository** and is not published.
What is published is what a third party needs to check the claim: the
specification, the CPU reference, two clean-room CPU implementations, and the
digests above. Anyone can write their own CUDA implementation from
`docs/CIS2_SPEC_v0.3b.md` and compare against `d82743059d1db929…`; that is the
intended way to falsify or confirm this result, and a matching independent GPU
implementation is exactly the contribution described in
"How to submit your own clean-room implementation" in the README.

## Run provenance

- Kaggle kernel `aefinityaiinc/e18b-gpu`, GPU enabled, **internet disabled**,
  finished 2026-09-08 12:07:11 UTC; container image pinned by digest
  `sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`.
- Machine-readable verdicts (`nofma_cuda_pass`, `sequential_pass`,
  `tree_candidate_pass`, all four digest comparisons, both `nvidia-smi`
  summaries) were emitted by the run itself, not transcribed by hand.
- Earlier attempts v6 and v7 were **false positives** and are withdrawn: the
  link step was missing `-x cu`, so the "GPU" binary was compiled as host C++
  and executed on the CPU. v8 adds `-x cu`, gates the real binaries rather than
  a standalone probe, and adds the utilization sampling. Do not cite v6 or v7.
- The earlier v5 run (2026-09-06, same device) reached the same four digests
  but established device execution from source structure alone; its adversarial
  review recorded that gap, and v8 closes it.
