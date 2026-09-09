---
license: apache-2.0
pretty_name: CIS-2 — conformance vectors for bit-identical fp32 transformer inference
language:
- en
tags:
- reproducibility
- determinism
- verification
- inference
- floating-point
- specification
- conformance
size_categories:
- n<1K
---

# CIS-2 — conformance vectors for bit-identical fp32 transformer inference

Floating-point transformer inference is usually treated as unavoidably
nondeterministic across hardware. Reduction order, FMA contraction, denormal
handling and platform math libraries all differ between x86_64 and aarch64, and
between compilers, so "the same model on the same input" in practice means
"agrees to within a tolerance", not bit-for-bit.

**CIS-2 is a written specification that removes those degrees of freedom**, and
this repository holds the artifacts a third party needs to check whether their
own implementation conforms: the spec text, five op-level conformance vectors
with pinned expected outputs, the expected end-to-end digests, and the GPU
result.

Everything here is Apache-2.0. Source of truth and CI:
**https://github.com/Aefinity-AI/cis2-spec**

## The claim

```
CIS2_REF, SmolLM2-135M, prompt "Once upon a time", 16 greedy tokens, spec v0.3b
  d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df
```

That digest is a SHA-256 witness chain folded over the **complete fp32 logit
vector at every decode step** — not the argmax token, the whole vector. It is
currently reproduced by:

| implementation | written from | platforms |
|---|---|---|
| Rust reference | — | x86_64, aarch64 (native runners) |
| Rust clean-room `verify2/` | the spec text alone | x86_64, aarch64 |
| C11 clean-room `verify3/` | the spec text alone | x86_64, aarch64 · gcc and clang |
| CUDA port (not published) | the spec text alone | NVIDIA Tesla P100, sm_60, CUDA 12.8 |

The two clean-room implementations were written without access to the reference
source or to each other. Public CI re-checks all of the CPU rows on every push.
The GPU row is documented in `GPU_RESULT.md`; on that run the per-step trace was
**byte-identical** to the CPU trace, not merely equal at the final digest.

## What is pinned

- Reduction order — strictly left-to-right, sequential.
- FMA contraction — forbidden, and gated by `objdump` in CI.
- Denormals — FTZ/DAZ on, pinned via MXCSR (x86) and FPCR.FZ (aarch64).
- Transcendentals — `sin`/`cos` by octant reduction plus separate Cephes-pattern
  minimax polynomials; `exp` and `ln` by pinned Cephes-pattern polynomials;
  `rsqrt` correctly rounded with no table. Every coefficient is pinned as an f32
  hex literal and hashed into the witness chain.
- RoPE `inv_freq` — pinned table, theta-general.
- Tokenization — byte-level BPE pinned at the byte level.

Full normative text: `CIS2_SPEC_v0.3b.md` (§13.1 carries the pinned vector).

## Files

| file | what it is |
|---|---|
| `CIS2_SPEC_v0.3b.md` | the normative specification |
| `EXPECTED_DIGESTS.md` | pinned end-to-end digests, including the GPU confirmation |
| `GPU_RESULT.md` | the 2026-09-08 NVIDIA Tesla P100 run, with its scope limits |
| `PROTOCOL.md` | the stdin/stdout wire contract a third-party binary implements |
| `vectors/README.md` | per-op field layout of each vector file |
| `vectors/*.txt` | five op-level input vectors: matvec, rmsnorm, rope, exp_pinned, attention_block |
| `vectors/*.expected` | the pinned expected output for each |

The vectors are plain text `key=value` files with float fields given as exact
hex bit patterns, so parsing introduces no rounding of its own. They exist so an
implementation can be checked **op by op** — you find out *which* operation
diverges, instead of only that a 64-character digest came out wrong.

## How to check your own implementation

```bash
git clone https://github.com/Aefinity-AI/cis2-spec
cd cis2-spec
cargo run --release --bin cis2-conformance -- /path/to/your-binary
```

`PROTOCOL.md` is the complete contract; you do not need to read any of this
project's Rust to implement against it.

## Scope, stated plainly

fp32 scalar reference semantics, not a fast kernel. Greedy decoding. Models
checked up to 1.5B parameters. The GPU leg is one Pascal device, one toolchain,
correctness only — no tensor cores, no batching, no timing number is claimed
anywhere in this project. The CUDA port itself is deliberately not published, so
a second GPU implementation written from the spec would be a genuine independent
check rather than a re-run of ours; that is the contribution we are asking for.

## Prior art

Reproducible and deterministic inference is prior-occupied ground. Gensyn's
`repops` demonstrates a hash-matched CPU/CUDA fp32 forward pass; Microsoft's
RepDL provides reproducible linear-algebra operators with CPU and CUDA backends;
vLLM and SGLang both ship batch-invariant determinism modes; and
arXiv:2606.00279 verifies bit-exact GPU inference by emulating vendor silicon
tables. **No "first" and no "only" claim is made here**, and none should be
inferred.

The narrower thing CIS-2 is testing is whether a *written document* can carry
enough information for strangers to converge on identical bits — across an ISA
boundary, a compiler boundary, a language boundary, and a CPU/GPU boundary,
with no implementation consulting another.

## Falsification bounty

There is a standing $50-per-distinct-root-cause bounty for breaking this:
https://github.com/Aefinity-AI/alice-aegis/blob/main/CHALLENGE.md — write your
own implementation from `CIS2_SPEC_v0.3b.md`, in any language for any device,
and get a different digest. If the disagreement is because the spec text permits
two readings, that is the finding most worth paying for: it means the document
is not yet sufficient, which is the entire thing CIS-2 claims to be.

---

Aefinity AI Inc. · Justin Brian Thompson
