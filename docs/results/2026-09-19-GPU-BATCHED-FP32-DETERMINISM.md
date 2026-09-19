# GPU batched fp32 determinism: host CPU vs Tesla T4, row-batched vs grouped vs int8

A batched-forward-pass sweep of the CIS-2 reference matmul on an NVIDIA
Tesla T4, comparing the witness digest of a CPU host build against a CUDA
build across three batching/precision strategies: row-batched fp32
(sequential per-row reduction, reused across the batch), grouped fp32
(a different reduction tree via batched/strided GEMM), and int8.

Rule A: this is an identity/correctness experiment. No timing figures are
reported or implied anywhere below.

## 1. Question

CIS-2's witness digest is defined over the exact sequence of floating-point
operations a reference implementation performs. Batching multiple
sequences through a matmul, or moving the matmul from CPU to GPU, can
change the order operations are summed in, or the numeric representation
used — either of which can change the digest even when the model and
inputs are unchanged. Does a straightforward row-batched fp32 GPU
implementation still reproduce the pinned CPU witness digest bit-for-bit
at batch sizes greater than 1? And do the batching/precision strategies
that are *expected* to change the reduction order (grouped-GEMM batching,
int8 quantization) actually diverge, as the digest is designed to catch?

## 2. Setup

**Model:** the CIS-2 reference model (hidden_size=576), the same
architecture and pinned reference weights used elsewhere in this repo's
conformance suite.

**Hardware:** NVIDIA Tesla T4 (`sm_75`), accessed via a Kaggle GPU
notebook; CPU host build compiled with g++, CUDA build compiled with
nvcc targeting `sm_75`. The GPU driver source and the Kaggle run recipe
used to reproduce this are not yet published.

**Batching strategies:**
- **fp32-seq (row-batched):** B independent forward passes computed with
  the same sequential per-row reduction as the batch=1 reference — this
  strategy is expected to reproduce the batch=1 digest exactly, since it
  reuses the same reduction order per row regardless of batch size.
- **fp32-grouped:** a batched/strided-GEMM-style matmul, which sums
  partial products in a different order than the sequential reduction —
  expected to diverge from the batch=1 digest, because changing
  floating-point summation order can change the rounded result even
  when the mathematical inputs are identical.
- **int8:** the batched matmul computed in int8, dequantized to compare
  against the fp32 reference — expected to diverge, because quantization
  changes the numeric representation entirely, not just the reduction
  order.

**Scope:** this sweep exercises a single matmul (`q_proj` at layer 0) of
the reference model, not a full forward pass.

**Pinned reference digest** (sequential, 16 generated tokens),
established by this repo's existing conformance suite: `d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df`.

## 3. Results

**Host gate.** Before the GPU sweep ran, the CPU host build's own
sequential-reduction digest for the 16-token witness chain was checked
against the pinned digest above:

```
HOST_GATE digest=d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df expected=d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df MATCH
```

**Sweep table** (host g++ build; the CUDA nvcc build on the T4 produced
byte-identical digests to the host build for every one of the 10 configs
below):

| config | digest (first 16 hex) | vs. cpu batch1 | diffs |
|---|---|---|---|
| host_control_batch1_fp32 | 9a8e923498185888... | MATCH | 0/576 |
| batch1_fp32_seq | 9a8e923498185888... | MATCH | 0/576 |
| batch1_fp32_grouped | f1ba46ce729afea3... | MISMATCH | 455/576 |
| batch1_int8 | 413114502d874b10... | MISMATCH | 576/576 |
| batch4_fp32_seq | 34a369808e935938... | MATCH | 0/576 |
| batch4_fp32_grouped | 6bf560c9f29a3d8d... | MISMATCH | 455/576 |
| batch4_int8 | 31d0109235031d12... | MISMATCH | 576/576 |
| batch8_fp32_seq | ada5b5de31aa3306... | MATCH | 0/576 |
| batch8_fp32_grouped | 23fae4007ee7f42c... | MISMATCH | 455/576 |
| batch8_int8 | c9e186c9c39cf36d... | MISMATCH | 576/576 |

fp32-seq matched the batch=1 reference exactly at batch sizes 1, 4, and 8
(0/576 diffs each), on both the host build and the T4 CUDA build.
fp32-grouped and int8 mismatched at every batch size tested (455/576 and
576/576 diffs respectively), also on both builds. This matches the
expectation stated in Setup exactly: the row-batched strategy reuses the
same reduction order per row regardless of batch size, so it reproduces
the reference digest; the grouped and int8 strategies use a genuinely
different computation (different summation order, or a different numeric
representation), so the digest correctly reports a difference — the
digest is doing what it is designed to do, not failing.

Host and CUDA builds produced bit-identical digests for every one of the
10 configurations tested.

## 4. Whole-model follow-on (row-batched, no dedicated report)

A separate, later run extended the row-batched fp32 strategy from the
single-matmul scope above to a whole-model forward pass (not just layer
0's q_proj), again on a Tesla T4 via Kaggle, at batch sizes 1, 4, and 8.
Row 0 of the batched output reproduced the pinned CPU sequential digest
above (`d82743059d1db929...`), and the host build's digest matched the
CUDA build's digest for all three batch sizes. This result is recorded in
this project's internal log rather than a dedicated report; the specific
build/run recipe is not yet published.

## 5. What this does NOT show

- The primary sweep (Section 3) covers one matmul (`q_proj`, layer 0) of
  the reference model, not a full forward pass; the whole-model
  confirmation (Section 4) covers only the row-batched fp32 strategy, not
  grouped or int8 at whole-model scope.
- Only one GPU architecture (Tesla T4, `sm_75`) was tested; no claim is
  made about other GPU architectures or CUDA compute capabilities.
- The grouped-GEMM and int8 mismatches are the *expected* outcome given
  how those strategies compute the matmul differently — this experiment
  does not characterize whether a grouped-GEMM or int8 implementation
  could be made to match the reference (that would require deliberately
  constraining its reduction order/representation, not attempted here).
- No timing numbers are reported, and none should be inferred (Rule A).
- The GPU driver source and the exact Kaggle notebook/dataset recipe used
  to produce these results are not yet published.

## 6. Reproduce

The CUDA driver source and Kaggle run recipe used for this sweep are not
yet published. The pinned CPU-side sequential reference digest
(`d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df`) is
established by this repo's existing conformance suite; see
`EXPECTED_DIGESTS.md` in this repo for the pinned reference digests this
sweep was checked against.

## 7. Plain English

We ran the same small piece of a language model's math on two very
different pieces of hardware — a regular CPU and an NVIDIA T4 GPU — and
in three different ways of handling a batch of requests at once. When we
batched requests the "simple" way (running each one independently, the
same way as a single request), the GPU produced the exact same
cryptographic fingerprint as the CPU, every time, at batch sizes 1, 4, and
8. When we batched them the way GPUs are normally optimized to do it —
grouping multiple requests into one larger matrix operation — the
fingerprint changed, because that approach genuinely adds up numbers in a
different order, which can shift the last few decimal digits of a
floating-point result. The same happened when we used a lower-precision
number format (int8) instead of full precision. Both of those changes are
expected and correct: the fingerprint's job is to notice when the actual
math that ran is different, and it did its job. The takeaway is that GPU
determinism against a CPU reference is achievable, but only for
implementations that are deliberately built to preserve the same order of
operations — the moment you take a shortcut that's normally invisible to
users (a faster batching strategy, a smaller number format), it becomes
visible to this kind of check.
