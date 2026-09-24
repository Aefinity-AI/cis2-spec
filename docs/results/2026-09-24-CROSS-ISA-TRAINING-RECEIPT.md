# Cross-machine and cross-ISA training receipts: torch default vs pinned vs a from-scratch fixed-order C trainer

Three related experiments asking the same question at increasing levels
of control: does a training run's checkpoint hash reproduce bit-for-bit
across different CPUs? A default-kernel PyTorch run, a pinned-kernel
PyTorch run, and a from-scratch fixed-order fp32 C trainer using CIS-2's
pinned deterministic primitives, each on a different set of hosts.

Rule A: this is an identity/correctness experiment. No timing figures are
reported or implied anywhere below.

## 1. Question

If the same training script, same seed, same data, and same step count
run on two different CPUs, do the resulting checkpoints hash to the same
value? Three variants:

- unmodified PyTorch (default CPU kernel dispatch),
- PyTorch with the CPU kernel pinned to a single dispatch path,
- a from-scratch fixed-order fp32 trainer using only IEEE-exact ops and
  CIS-2's pinned polynomial `exp`/`ln` (no platform libm transcendentals).

## 2. Setup common to all three

A tiny language-model trainer, checkpointed at steps 50/100/150/200,
checkpoint identity checked as a sha256 digest over the raw parameter
bytes. Same seed (`20260924`-derived) and same deterministic synthetic
corpus generator across every run in every variant.

## 3. cap-1: PyTorch, default CPU kernel dispatch

**Hosts:** box1 (aefinity-box, Intel i5-5200U, x86_64, AVX2/FMA present)
vs box2 (aefinity-box2, Intel Celeron N4020, x86_64, no AVX2 exposed).
Both `torch 2.14.0+cpu`, `train/run_receipt.sh` run unmodified.

| step | box1 loss | box2 loss | loss equal? | box1 sha256 | box2 sha256 |
|------|-----------|-----------|-------------|-------------|-------------|
| 50  | 5.0175909996032715 | 5.0175909996032715 | YES | `c4c4117048c14f8cbd3b018a0bb3d585b71174cd322c9df9e8652053c4e153c6` | `ed848a52814591be06627b7099ec99ececd23f26ec615889e603e1a51593c4e9` |
| 100 | 4.793025493621826  | 4.793025493621826  | YES | `b849047a6a02eecddfcdba082eb753a2d4294138807cd4543a9aaa884e263384` | `3280de16020a20ab60f4cbdaad5d4515eae858dd7143945a908544c8cc51beb8` |
| 150 | 4.643331527709961  | 4.643331527709961  | YES | `95df5169218a1be9caaf34e6a133f6771caab556bb099d740974c8eae158f2b0` | `f1792b2e7fa58c28c8649fac08f91a0e30d661d7ecc10e5ac0aa17e4cb62cf3b` |
| 200 | 4.622854709625244  | 4.622854709625244  | YES | `13af7f00cd53aab54691d4745b70c353c96684dac0b60de99650ab06ecb8a8ee` | `10e9f13ed17fadfc223470a3bd01f4332947aededafacc0db240dbe57cf690f0` |

**MISMATCH.** The four printed loss scalars are bit-identical (same
Python `repr()` at every checkpoint on both hosts), but all four
checkpoint sha256 digests differ, starting at the first recorded
checkpoint (step 50). Consistent with box1's AVX2/FMA CPU kernels
picking a different vectorized matmul/reduction path than box2's
scalar-only path, producing different low-order rounding in the weights
that the loss scalar's own reduction/print precision happened not to
surface. Both machines were independently self-consistent (same-machine
reruns matched); the divergence is specific to cross-machine kernel
dispatch under PyTorch's default CPU backend.

## 4. cap-1b: PyTorch, pinned CPU kernel

Same script and same two hosts (box1, box2), with the CPU kernel pinned:
`ATEN_CPU_CAPABILITY=default`, `MKL_CBWR=COMPATIBLE`,
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`NUMEXPR_NUM_THREADS=1`, `PYTHONHASHSEED=0`. Confirmed under this env,
`torch.backends.cpu.get_cpu_capability()` reports `DEFAULT` on both
hosts.

| step | box1 sha256 | box2 sha256 | equal? |
|------|-------------|-------------|--------|
| 50  | `1786c50e1abe973552d3409ef0798d9b6d8086f0dad06cb4214b9b1f62b64dbc` | `1786c50e1abe973552d3409ef0798d9b6d8086f0dad06cb4214b9b1f62b64dbc` | YES |
| 100 | `45a0339aa616fe458b29c9e09bd0f595dfca941bc6b49baaa2e7b2b133a69e9c` | `45a0339aa616fe458b29c9e09bd0f595dfca941bc6b49baaa2e7b2b133a69e9c` | YES |
| 150 | `c9ff41e03f50488c3dfcc909bb785affa70f915260d096d1fb384faa6b6d25a3` | `c9ff41e03f50488c3dfcc909bb785affa70f915260d096d1fb384faa6b6d25a3` | YES |
| 200 | `188a297ccdcfc8f7824178094ca30ca7d4bcf07c90b3b039c4568ea75b5a6bb1` | `188a297ccdcfc8f7824178094ca30ca7d4bcf07c90b3b039c4568ea75b5a6bb1` | YES |

**MATCH — cross-machine, not cross-ISA.** All four checkpoints are
byte-identical between box1 (AVX2/FMA) and box2 (no AVX2) once the CPU
kernel dispatch path and thread count are pinned. Both hosts are
x86_64; this result shows pinning is sufficient to reproduce this fp32
training receipt across two different x86_64 CPUs, not across
instruction-set architectures.

## 5. cap-1c: from-scratch fixed-order fp32 C trainer

A small byte-level MLP language-model trainer (embedding -> tanh hidden
layer -> softmax over 256 byte classes), written from scratch in C11,
trained with plain SGD in fixed left-to-right summation order,
single-threaded, seq_len=64, 200 steps, checkpointed every 50 steps.
13,088 fp32 parameters total. Not a reimplementation of any existing
model architecture; exact numeric parity with the cap-1/cap-1b torch
run was out of scope by design.

Determinism measures:

- fp32 only, `-ffp-contract=off`, `-fno-fast-math`,
  `-fexcess-precision=standard`, `-mno-fma` on x86_64 — the same flags
  `verify3/`'s Makefile already uses.
- No libm `expf`/`tanhf`/`logf`. The trainer calls CIS-2 v0.2's pinned
  fp32 primitives for `exp`/`ln` (fixed bit-pattern-coefficient
  polynomial, range reduction, exact `ldexp` via integer exponent-field
  manipulation); `tanh(x)` is built from that pinned `exp` as
  `(e^{2x}-1)/(e^{2x}+1)`. The only remaining libm calls (`sqrtf`,
  `floorf`, `isnan`) are IEEE-754-mandated exact operations, not
  approximations. FTZ/DAZ is pinned and self-tested at process start
  (MXCSR on x86_64, FPCR.FZ on aarch64).
- Checkpoint digest = sha256 over the raw fp32 bytes of all five
  parameter arrays in fixed field order, native-endian; every target
  host is little-endian.
- Same deterministic synthetic-corpus generator (independently
  reimplemented in C from the same spec used elsewhere in this
  programme, not copied from a shared binary blob).

**Hosts:** penguin (x86_64, no AVX2 in this VM), box1 (x86_64,
AVX2/FMA), box2 (x86_64, no AVX2), box3 (x86_64, no AVX2), and an
aarch64 phone (static binary, no build-time flags differing beyond the
architecture-conditional ISA flag already used by `verify3/`:
`-mno-fma` on x86_64, `-march=armv8-a` on aarch64).

| step | loss | checkpoint_sha256 (all 5 hosts) |
|------|------|----------------------------------|
| 50  | 5.492920 | `fd64cf9e032ee182e2623aaada3e71918d4b8d31c25d8dfc876fed74a85e3e12` |
| 100 | 5.438680 | `d36bd7fd164ba77481a9d8f25f2fbd1acd83561f68cb8ceca9b3ba99be57705c` |
| 150 | 5.373738 | `3887f7b3f512a6a2ed4bb72572966f46b90366f6a546aafe758ae7587db76ccd` |
| 200 | 5.264791 | `f665ffecfabba7a4a6c0248c73107ecf8953efd9ef20b861a50fbc3ce818fd3e` |

**MATCH.** All four checkpoint sha256 digests are identical across all
five hosts: three x86_64 machines without AVX2, one x86_64 machine with
AVX2/FMA, and one aarch64 machine — a cross-ISA result. Loss decreased
monotonically over the run (starting near `ln(256) ≈ 5.545`, i.e.
near-uniform-random init), so the model is training, not producing a
static digest. The checkpoints were independently re-verified by a
second clean build of the trainer on both an x86_64 host and the
aarch64 phone, reproducing the same four digests.

## 6. cap-1d: CIS-2 inference replays on the same aarch64 phone

For context, three already-published CIS-2 `verify3/` inference paths
were also replayed on the same aarch64 phone used above (cross-compiled
statically, `aarch64-linux-gnu-gcc`, otherwise identical flags to the
x86_64 reference build). All three matched the x86_64 reference
digests:

- **int8 deterministic path:** witness digest
  `387c468fa782b42d14eab4a4c67b24f7a79b99b1f4c303f5b86e7f2e98c6bf00`,
  argmax digest
  `9192f1c64a40b04fb2d12984c4f8988e2c9e1f7943090c6ee202dc8124a4dd86` —
  MATCH.
- **KV-cache incremental decode:** witness digest
  `d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df`,
  argmax digest
  `0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff`,
  table digest
  `23c7bfaf5cef0095fd021af2eb1808abb4928bae4219756d86bdac670a06b35d`,
  inv-freq-table digest
  `da9f6dcfde0425588815509e874515cdcd3d6b8818b6d0136590052e7bbf6f12`
  — MATCH, and these reproduce this repo's published `EXPECTED_DIGESTS`
  values for this path.
- **Speculative decode (exact):** across 5 prompts, all plain-argmax and
  spec-argmax digests matched each other and the x86_64 reference on
  every prompt (`ids_match`/`witness_digest_match` PASS on all 5) —
  MATCH.

These are inference replays, not training, and are reported here only
as corroborating cross-ISA evidence on the same physical aarch64
device used for the cap-1c training result above.

## 7. Limits

- The cap-1c trainer is tiny (13,088 parameters) and the run is short
  (200 steps); this is not a claim about arbitrarily large models or
  step counts.
- cap-1c is a from-scratch architecture (byte-level MLP), not the same
  model as cap-1/cap-1b's PyTorch trainer; it demonstrates
  self-consistency across hosts for this specific trainer/config, not
  numeric parity with a PyTorch reference.
- No GPU host was tested in any of these three experiments; all results
  above are CPU-only.
- The cap-1c trainer's source is not released in this repo.
- No mismatch was found in cap-1b, cap-1c, or cap-1d, so nothing above
  is pinned as a general guarantee — each is reported as an observed
  result for its specific configuration and host set.

## Acknowledgments

A special thank you to Charles Seaman and Linda Blanchard, whose contributions have helped Aefinity AI stay on track.

And a very special thank you to **Bonnie Rae Power**: an amazing woman, a great friend and neighbor, without whom Aefinity AI would have never had a chance to ever get started. Thank you, Bonnie, for your advice, care, encouragement, guidance, intuitive wisdom, and financial assistance.
