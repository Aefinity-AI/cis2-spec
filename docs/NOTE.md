# What bit-identical inference and verifiable receipts buy us for trust

## 1. The problem

When an AI model runs in production today, the run itself is a black box.
You are told which model answered a request, but you cannot check that
claim after the fact: the weights on disk can be swapped, a log file can be
edited, and "we ran model X, version Y" is an assertion by the operator,
not something an outside party can check. Even inside a single
organization, "same model, same answer" is usually taken on trust (a claim
in a changelog or a deployment ticket) rather than something a second,
disinterested machine can check against the computation that happened.
This matters for anyone who has to sign off on an AI system: an auditor
asking "did the model that was certified last month produce this output," a
regulator asking "can this be reproduced," or an incident responder asking
"was this output tampered with after generation." None of those questions
can be answered by inspecting logs alone, because logs describe what an
operator says happened, not what a piece of checkable evidence shows
happened.

## 2. What CIS-2 provides

CIS-2 is a written specification (not a product) for running a
floating-point ("fp32") transformer model forward pass such that two
people who have never seen each other's code can each build an
implementation from the spec text alone, run the same model on the same
input, and get **bit-for-bit identical** output: the exact last bit of
every number the model produces, not just the same generated words. This
is stronger than "the outputs looked similar." Floating-point arithmetic is
order-sensitive, so getting the literal same bits out requires pinning the
reduction order, the numerical routines, and the output encoding
explicitly enough that a stranger can reproduce them without ever seeing a
reference implementation.

Two clean-room implementations (one in Rust, one in C, each written by an
isolated coding agent that had access only to the specification text, not
to each other's work or to any reference implementation) converge on the
same digest for the same pinned test case. On top of the raw computation,
CIS-2 defines two digest types: a **witness digest**, which hashes the full
logit vector (the model's raw, un-rounded numeric output) at every
generation step, and a coarser **argmax digest**, which hashes only which
token "won" at each step. The witness digest is the stricter check: it can
catch a change that alters the underlying numbers without changing which
word gets picked.

Separately, a **receipt** is a hash chain that binds together the model's
identity (a hash of its weights), the exact inputs, every intermediate
computation step, and any external tool results the model used along the
way, into one verifiable artifact. A receipt can be handed to a different
machine, one with no GPU and no relationship to the machine that generated
it, and that machine can recompute the chain with the public verifier and
confirm, offline, that the receipt is genuine and unaltered.

## 3. What has been measured

Each row is backed by a file in this repository.

| Claim | Evidence | Where it lives |
|---|---|---|
| Bit-identical fp32 digest from two clean-room implementations, for a pinned SmolLM2-135M test vector | `verify2/` (Rust) and `verify3/` (C), each written from spec text alone, reproduce the pinned witness digest `d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df`, on x86_64 and aarch64 CI runners, gcc and clang | [`README.md`](../README.md), [`EXPECTED_DIGESTS.md`](../EXPECTED_DIGESTS.md) |
| The self-check reproduces the pinned digests on several of the maintainer's own machines, with and without AVX2 | box1, box2, box3, penguin; plus an aarch64 Android phone through the `verify3/` path | machine table in [`README.md`](../README.md) |
| Row-batched fp32 on an NVIDIA T4 GPU reproduces the CPU reference witness digest bit-for-bit; a different batching strategy (grouped GEMM) and int8 quantization do not match, as expected | host CPU build versus CUDA build | [GPU batched fp32 determinism](results/2026-09-19-GPU-BATCHED-FP32-DETERMINISM.md) |
| A single corrupted weight bit is caught by the witness digest in every case tested, where text replay and a cheaper top-candidates check each miss a share of cases | head-to-head over logit-changing bit flips | [single-weight corruption](results/2026-09-13-BITFLIP-HEADTOHEAD.md) |
| A valid receipt replayed under the wrong item's identity is caught only after adding context binding to the hash | receipt design without binding passes the swap; with binding it is caught | [receipt-verified EVAL-60](results/2026-09-13-RECEIPT-VERIFIED-EVAL60.md) |

None of these rows is a timing benchmark.

## 4. Known limits

A verified receipt shows **which model ran, on what inputs, producing what
outputs, unaltered since generation.** It does not show that the decision
itself was good, safe, fair, or resistant to a jailbreak or a bad prompt: a
model can be verifiably, bit-for-bit itself while still giving a wrong or
harmful answer. CIS-2 is a provenance and integrity check, not a content or
safety check.

The scope tested so far is narrower than "AI in general." Bit-identity has
only been demonstrated for **fp32** inference. No claim is made about
lower-precision formats being bit-identical across environments (int8
quantization was specifically shown to diverge from the fp32 reference in
the GPU sweep above, by design). The CPU path is the reference
implementation the spec is audited against; the GPU path has only been
shown to match under one batching strategy on the GPU architectures
documented in the linked results, and a faster batching strategy does not
match the CPU reference.

Replay has a real cost, which is why the design uses a hash digest for
everyday checks and reserves full replay for cases that need the stronger
check.

The specification is young. The reproducers of the primary claim are two
clean-room implementations built by the same operator on the same day.
Their isolation is a read-access boundary enforced during development, not
authorship by unrelated organizations (see the "Provenance" section of
[`README.md`](../README.md)). No outside, unaffiliated reproduction has
happened yet. The context-binding fix for the receipt identity-swap gap is
described in its result note, which says where it stands.

## 5. How to check it yourself

```sh
git clone https://github.com/Aefinity-AI/cis2-spec && cd cis2-spec
./selfcheck.sh
```

This fetches the pinned test weights, builds the clean-room verifiers with
your system toolchain, and runs them against the pinned test vector; expect
a summary line reading `passed` with a zero `failed` count. A shorter,
digests-only path is `scripts/self_check.sh`, which ends with a line
reading `PASS: all digests match the pinned CIS-2 v0.3b test vector.` If you
get anything other than a clean PASS, that is itself useful information:
file an issue with the full output.
