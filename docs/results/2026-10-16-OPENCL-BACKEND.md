# OpenCL backend: the pinned digests on pocl, NVIDIA OpenCL and an Intel iGPU (2026-10-16)

An OpenCL 1.2 port of the C++ reference (`cis2_host.cpp`) runs SmolLM2-135M
under the CIS-2 arithmetic on three OpenCL devices and reproduces the pinned
`CIS2_REF` digests on all of them, with one required change on the Intel iGPU:
its hardware divide and square root are not correctly rounded, so that device
runs a software build. With the hardware divide and square root the iGPU
diverges from the host on the 16-token vector, and the divergence is localised
to the input normalisation of layer 0.

Rule A: no timing figures. Section 4 holds the builder run (lab host C); section 6 holds the pre-registered leg rows that confirmed it on four hosts (verdict MET, ledger batch 39).
The rows from the builder run on lab host C are listed in section 4. The
per-host rows from the pre-registered leg are not in yet and are marked by the
placeholder in section 6.

## 1. What was built

- **Kernels.** OpenCL 1.2 kernels (`-cl-std=CL1.2`) of the same pinned
  arithmetic as the host port: same reduction order, same operation order, no
  fused multiply-add, no fast-math. Device kernels are built with
  `-cl-denorms-are-zero`, and the flush-to-zero points follow the x86 host.
- **Hardware divide/sqrt build** (`make opencl`). Requires the device to report
  correctly rounded single-precision divide and square root, and requests it
  with `-cl-fp32-correctly-rounded-divide-sqrt`. The `reduce_pi_2` step uses
  the device's native binary64 (`-DCIS2_CL_NATIVE_F64=1`).
- **Software divide/sqrt build** (`make opencl-soft`, `CIS2_SOFT_DIVSQRT=1`).
  Implements correctly rounded fp32 divide and square root in integer
  arithmetic, and a software binary64 `reduce_pi_2` for devices without fp64.
  It does not depend on the device's divide, square root or fp64 hardware.
  With fp64 present it can still use the native binary64 path, and a
  `--cl-soft-f64` switch forces the software one.
- **Unit test** (`make test_soft`): compares the software routines with the
  host's hardware `divss`, `sqrtss` and double arithmetic, on the host and
  compiled for each device.

## 2. Devices

| label | platform / device | OpenCL | correctly rounded div/sqrt | fp64 |
|---|---|---|---|---|
| pocl | Portable Computing Language, `cpu-haswell-Intel(R) Core(TM) i7-7700HQ` | 3.0 PoCL | yes | yes |
| NVIDIA | NVIDIA CUDA platform, GeForce GTX 1050 | 3.0 CUDA | yes | yes |
| iGPU | Mesa rusticl (`RUSTICL_ENABLE=iris`), Intel HD Graphics 630 (KBL GT2) | 3.0 | **no** | **no** |

On the iGPU the device also reports no denormal support. Mesa itself prints
that OpenCL support on the iris driver is incomplete. Nothing here is a claim
that any of these implementations meets the OpenCL conformance requirements;
the only property used is the one measured below.

All three are on one machine, lab host C (i7-7700HQ, AVX2 + FMA, GTX 1050,
Intel HD 630).

## 3. Unit test of the software routines

Host software routines against host hardware (`gpu/test_soft --n 3`):

| routine | cases |
|---|---|
| divide | 7,297,086 |
| square root | 25,370,682 |
| `reduce_pi_2` | 4,542,018 |
| binary64 to binary32 narrowing (near 2^-126 and 2^128) | 3,000,000 |

Edge sets: powers of two and neighbours, subnormals, minimum and maximum
normal, infinities and NaN, zero signs, and mantissa sweeps for square root at
exponents 1, 2, 126, 127, 128, 129, 253 and 254. **Mismatches: 0.**

Software routines compiled for the device, against host hardware:

| device | result |
|---|---|
| pocl | PASS, 0 mismatches |
| pocl with `--cl-soft-f64` | PASS |
| NVIDIA GTX 1050 | PASS, 0 mismatches |
| Intel HD 630 | PASS, 0 mismatches |

Informational: the devices' own hardware divide and square root against the
host, same case sets.

- pocl: 0 mismatches.
- NVIDIA hardware path: 666,366 divide and 251,898 square-root mismatches. Every
  one involves a subnormal operand or result, or the flush-to-zero boundary
  (for example `0x00ffffff / 2` returns `0x00800000` on the device where the
  host's flush-to-zero gives 0). There are 0 mismatches on normal operands.
- Intel HD 630 hardware path: 1,203,879 divide and 1,938,585 square-root
  mismatches. The path is not correctly rounded: `0x00ffffff / 0x00fffffe`
  returns `3f800000`, the correctly rounded result is `3f800001`.

## 4. Digests (builder run on lab host C, sequential reduction)

Pins: 16 tokens `d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df`,
128 tokens `22f69ad87a615d66a77efaca8b1172d22bdfb4bd5092ceffd302e35669a050f6`.
Weights sha256 `80521b40…` and the 16-token prompt ids file
(sha256 `4b6618b6…`) were checked before the runs. Every MATCH row carries the
full 64-hex digest.

| row | build | 16 tokens | 128 tokens | status |
|---|---|---|---|---|
| host g++ | none | MATCH | MATCH | confirmed by the leg (section 6) |
| pocl | hardware div/sqrt | MATCH | MATCH | confirmed by the leg (section 6) |
| NVIDIA GTX 1050 | hardware div/sqrt | MATCH | MATCH | confirmed by the leg (section 6) |
| Intel HD 630 | software div/sqrt, software binary64 | MATCH | MATCH | confirmed by the leg (section 6) |
| pocl | software div/sqrt | MATCH | MATCH | confirmed by the leg (section 6) |
| pocl | software, `--cl-soft-f64` | MATCH | not run | builder run only (not in the leg) |
| Intel HD 630 | hardware div/sqrt (negative control) | DIVERGE | not run | confirmed by the leg (section 6), same digest on both UHD 600 units |

MATCH means the printed digest equals the pinned value above, and the rows
reproduce the pinned digest. The negative control produced
`8322617cc9f3ba3355eab391d066329b3d0cb8d0a9020c5765ffae60fc750acc` at 16
tokens.

## 5. Negative control: where the iGPU hardware path diverges

Same build as the NVIDIA and pocl rows (hardware divide and square root), run
on the Intel HD 630, with the step-0 tensors dumped and compared with the host.

- Divergence at position 0, layer 0, operation `norm1`: 512 of 576 elements
  differ.
- Element index 0: host `0x3cad866e`, device `0x3cad866f`, one unit in the last
  place. The device's reciprocal square root or divide is not correctly
  rounded.
- With the software build on the same iGPU, every tensor dumped at position 0
  is identical to the host.

Reading: the hardware build does not fail loudly on this device. It runs, and
produces a different digest. What separates the two builds is the device's
divide and square root, which section 3 measures directly.

## 6. Pre-registered leg

The leg **w1x-opencl** is registered in the public ledger (batch 37; verdict
MET in batch 39). Its bar covers lab host C (hq) only. The other hosts are supplementary and are not part
of the bar:

- lab host B (i9-9980HK, AVX2 + FMA): RTX 2070 OpenCL and pocl;
- lab hosts A and H (Celeron N4020 and N4020C, SSE4.2 only): Intel UHD 600
  iGPU rows, and pocl on an SSE4.2-only CPU.

Final per-host rows (the same rows are in `docs/REPRODUCTIONS.md`):

| date (UTC) | host class | implementation | compiler / toolchain | source commit | weights sha256 prefix | gen_toks | digest | outcome |
|---|---|---|---|---|---|---|---|---|
| 2026-10-16 | Lab host C: Intel Core i7-7700HQ (AVX2 + FMA), pocl CPU device | OpenCL port, hardware div/sqrt build (`make opencl`) | pocl 3.0 (Debian 13), g++ 14.2.0 host side | host sources sha256 prefix 10e08ea20584151f (see note 2); OpenCL kernels private (results note section 1) | 80521b40 | 16 (run twice) | `d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df` | MATCH |
| 2026-10-16 | Lab host C: Intel Core i7-7700HQ (AVX2 + FMA), pocl CPU device | OpenCL port, hardware div/sqrt build (`make opencl`) | pocl 3.0 (Debian 13), g++ 14.2.0 host side | host sources sha256 prefix 10e08ea20584151f (see note 2); OpenCL kernels private (results note section 1) | 80521b40 | 128 | `22f69ad87a615d66a77efaca8b1172d22bdfb4bd5092ceffd302e35669a050f6` | MATCH |
| 2026-10-16 | Lab host C: Intel Core i7-7700HQ (AVX2 + FMA), pocl CPU device | OpenCL port, software div/sqrt build (`make opencl-soft`) | pocl 3.0 (Debian 13), g++ 14.2.0 host side | host sources sha256 prefix 10e08ea20584151f (see note 2); OpenCL kernels private (results note section 1) | 80521b40 | 16 (run twice) | `d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df` | MATCH |
| 2026-10-16 | Lab host C: Intel Core i7-7700HQ (AVX2 + FMA), pocl CPU device | OpenCL port, software div/sqrt build (`make opencl-soft`) | pocl 3.0 (Debian 13), g++ 14.2.0 host side | host sources sha256 prefix 10e08ea20584151f (see note 2); OpenCL kernels private (results note section 1) | 80521b40 | 128 | `22f69ad87a615d66a77efaca8b1172d22bdfb4bd5092ceffd302e35669a050f6` | MATCH |
| 2026-10-16 | Lab host C GPU: NVIDIA GeForce GTX 1050, NVIDIA OpenCL | OpenCL port, hardware div/sqrt build | NVIDIA OpenCL 3.0 CUDA platform (driver 550.163.01) | host sources sha256 prefix 10e08ea20584151f (see note 2); OpenCL kernels private (results note section 1) | 80521b40 | 16 (run twice) | `d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df` | MATCH |
| 2026-10-16 | Lab host C GPU: NVIDIA GeForce GTX 1050, NVIDIA OpenCL | OpenCL port, hardware div/sqrt build | NVIDIA OpenCL 3.0 CUDA platform (driver 550.163.01) | host sources sha256 prefix 10e08ea20584151f (see note 2); OpenCL kernels private (results note section 1) | 80521b40 | 128 | `22f69ad87a615d66a77efaca8b1172d22bdfb4bd5092ceffd302e35669a050f6` | MATCH |
| 2026-10-16 | Lab host C iGPU: Intel HD Graphics 630 (KBL GT2), Mesa rusticl | OpenCL port, software div/sqrt build | Mesa 25.0.7 rusticl (`RUSTICL_ENABLE=iris`) | host sources sha256 prefix 10e08ea20584151f (see note 2); OpenCL kernels private (results note section 1) | 80521b40 | 16 (run twice) | `d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df` | MATCH |
| 2026-10-16 | Lab host C iGPU: Intel HD Graphics 630 (KBL GT2), Mesa rusticl | OpenCL port, software div/sqrt build | Mesa 25.0.7 rusticl (`RUSTICL_ENABLE=iris`) | host sources sha256 prefix 10e08ea20584151f (see note 2); OpenCL kernels private (results note section 1) | 80521b40 | 128 | `22f69ad87a615d66a77efaca8b1172d22bdfb4bd5092ceffd302e35669a050f6` | MATCH |
| 2026-10-16 | Lab host C iGPU: Intel HD Graphics 630, Mesa rusticl | OpenCL port, HARDWARE div/sqrt build (negative control) | Mesa 25.0.7 rusticl | host sources sha256 prefix 10e08ea20584151f (see note 2); OpenCL kernels private (results note section 1) | 80521b40 | 16 | `8322617cc9f3ba3355eab391d066329b3d0cb8d0a9020c5765ffae60fc750acc` | DIVERGE (expected: the device does not round divide/sqrt correctly; see results note section 5) |
| 2026-10-16 | Lab host A: Intel Celeron N4020 (SSE4.2, no AVX), pocl CPU device | OpenCL port, hardware div/sqrt build | pocl 3.0 (Debian 13), g++ 14.2.0 host side | host sources sha256 prefix 10e08ea20584151f (see note 2); OpenCL kernels private (results note section 1) | 80521b40 | 16 (run twice) | `d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df` | MATCH |
| 2026-10-16 | Lab host A: Intel Celeron N4020 (SSE4.2, no AVX), pocl CPU device | OpenCL port, hardware div/sqrt build | pocl 3.0 (Debian 13), g++ 14.2.0 host side | host sources sha256 prefix 10e08ea20584151f (see note 2); OpenCL kernels private (results note section 1) | 80521b40 | 128 | `22f69ad87a615d66a77efaca8b1172d22bdfb4bd5092ceffd302e35669a050f6` | MATCH |
| 2026-10-16 | Lab host A: Intel Celeron N4020 (SSE4.2, no AVX), pocl CPU device | OpenCL port, software div/sqrt build | pocl 3.0 (Debian 13), g++ 14.2.0 host side | host sources sha256 prefix 10e08ea20584151f (see note 2); OpenCL kernels private (results note section 1) | 80521b40 | 16 (run twice) | `d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df` | MATCH |
| 2026-10-16 | Lab host A: Intel Celeron N4020 (SSE4.2, no AVX), pocl CPU device | OpenCL port, software div/sqrt build | pocl 3.0 (Debian 13), g++ 14.2.0 host side | host sources sha256 prefix 10e08ea20584151f (see note 2); OpenCL kernels private (results note section 1) | 80521b40 | 128 | `22f69ad87a615d66a77efaca8b1172d22bdfb4bd5092ceffd302e35669a050f6` | MATCH |
| 2026-10-16 | Lab host A iGPU: Intel UHD Graphics 600 (GLK), Mesa rusticl | OpenCL port, software div/sqrt build | Mesa rusticl (Debian 13), `RUSTICL_ENABLE=iris` | host sources sha256 prefix 10e08ea20584151f (see note 2); OpenCL kernels private (results note section 1) | 80521b40 | 16 (run twice) | `d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df` | MATCH |
| 2026-10-16 | Lab host A iGPU: Intel UHD Graphics 600 (GLK), Mesa rusticl | OpenCL port, software div/sqrt build | Mesa rusticl (Debian 13), `RUSTICL_ENABLE=iris` | host sources sha256 prefix 10e08ea20584151f (see note 2); OpenCL kernels private (results note section 1) | 80521b40 | 128 | `22f69ad87a615d66a77efaca8b1172d22bdfb4bd5092ceffd302e35669a050f6` | MATCH |
| 2026-10-16 | Lab host A iGPU: Intel UHD Graphics 600, Mesa rusticl | OpenCL port, HARDWARE div/sqrt build (negative control) | Mesa rusticl (Debian 13) | host sources sha256 prefix 10e08ea20584151f (see note 2); OpenCL kernels private (results note section 1) | 80521b40 | 16 | `8322617cc9f3ba3355eab391d066329b3d0cb8d0a9020c5765ffae60fc750acc` | DIVERGE (expected; same divergent digest as host C's HD 630) |
| 2026-10-16 | Lab host H: Intel Celeron N4020 (SSE4.2, no AVX), second unit, pocl CPU device | OpenCL port, hardware div/sqrt build | pocl 3.0 (Debian 13), g++ 14.2.0 host side | host sources sha256 prefix 10e08ea20584151f (see note 2); OpenCL kernels private (results note section 1) | 80521b40 | 16 (run twice) | `d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df` | MATCH |
| 2026-10-16 | Lab host H: Intel Celeron N4020 (SSE4.2, no AVX), second unit, pocl CPU device | OpenCL port, hardware div/sqrt build | pocl 3.0 (Debian 13), g++ 14.2.0 host side | host sources sha256 prefix 10e08ea20584151f (see note 2); OpenCL kernels private (results note section 1) | 80521b40 | 128 | `22f69ad87a615d66a77efaca8b1172d22bdfb4bd5092ceffd302e35669a050f6` | MATCH |
| 2026-10-16 | Lab host H: Intel Celeron N4020 (SSE4.2, no AVX), second unit, pocl CPU device | OpenCL port, software div/sqrt build | pocl 3.0 (Debian 13), g++ 14.2.0 host side | host sources sha256 prefix 10e08ea20584151f (see note 2); OpenCL kernels private (results note section 1) | 80521b40 | 16 (run twice) | `d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df` | MATCH |
| 2026-10-16 | Lab host H: Intel Celeron N4020 (SSE4.2, no AVX), second unit, pocl CPU device | OpenCL port, software div/sqrt build | pocl 3.0 (Debian 13), g++ 14.2.0 host side | host sources sha256 prefix 10e08ea20584151f (see note 2); OpenCL kernels private (results note section 1) | 80521b40 | 128 | `22f69ad87a615d66a77efaca8b1172d22bdfb4bd5092ceffd302e35669a050f6` | MATCH |
| 2026-10-16 | Lab host H iGPU: Intel UHD Graphics 600 (GLK), Mesa rusticl | OpenCL port, software div/sqrt build | Mesa rusticl (Debian 13), `RUSTICL_ENABLE=iris` | host sources sha256 prefix 10e08ea20584151f (see note 2); OpenCL kernels private (results note section 1) | 80521b40 | 16 (run twice) | `d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df` | MATCH |
| 2026-10-16 | Lab host H iGPU: Intel UHD Graphics 600 (GLK), Mesa rusticl | OpenCL port, software div/sqrt build | Mesa rusticl (Debian 13), `RUSTICL_ENABLE=iris` | host sources sha256 prefix 10e08ea20584151f (see note 2); OpenCL kernels private (results note section 1) | 80521b40 | 128 | `22f69ad87a615d66a77efaca8b1172d22bdfb4bd5092ceffd302e35669a050f6` | MATCH |
| 2026-10-16 | Lab host H iGPU: Intel UHD Graphics 600, Mesa rusticl | OpenCL port, HARDWARE div/sqrt build (negative control) | Mesa rusticl (Debian 13) | host sources sha256 prefix 10e08ea20584151f (see note 2); OpenCL kernels private (results note section 1) | 80521b40 | 16 | `8322617cc9f3ba3355eab391d066329b3d0cb8d0a9020c5765ffae60fc750acc` | DIVERGE (expected; same divergent digest as host C's HD 630) |
| 2026-10-16 | Lab host B: Intel Core i9-9980HK (AVX2 + FMA), pocl CPU device | OpenCL port, hardware div/sqrt build | pocl 3.0 (Debian 13), g++ 14.2.0 host side | host sources sha256 prefix 10e08ea20584151f (see note 2); OpenCL kernels private (results note section 1) | 80521b40 | 16 (run twice) | `d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df` | MATCH |
| 2026-10-16 | Lab host B: Intel Core i9-9980HK (AVX2 + FMA), pocl CPU device | OpenCL port, hardware div/sqrt build | pocl 3.0 (Debian 13), g++ 14.2.0 host side | host sources sha256 prefix 10e08ea20584151f (see note 2); OpenCL kernels private (results note section 1) | 80521b40 | 128 | `22f69ad87a615d66a77efaca8b1172d22bdfb4bd5092ceffd302e35669a050f6` | MATCH |
| 2026-10-16 | Lab host B: Intel Core i9-9980HK (AVX2 + FMA), pocl CPU device | OpenCL port, software div/sqrt build | pocl 3.0 (Debian 13), g++ 14.2.0 host side | host sources sha256 prefix 10e08ea20584151f (see note 2); OpenCL kernels private (results note section 1) | 80521b40 | 16 (run twice) | `d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df` | MATCH |
| 2026-10-16 | Lab host B: Intel Core i9-9980HK (AVX2 + FMA), pocl CPU device | OpenCL port, software div/sqrt build | pocl 3.0 (Debian 13), g++ 14.2.0 host side | host sources sha256 prefix 10e08ea20584151f (see note 2); OpenCL kernels private (results note section 1) | 80521b40 | 128 | `22f69ad87a615d66a77efaca8b1172d22bdfb4bd5092ceffd302e35669a050f6` | MATCH |
| 2026-10-16 | Lab host B GPU: NVIDIA GeForce RTX 2070 Max-Q, NVIDIA OpenCL | OpenCL port, hardware div/sqrt build | NVIDIA OpenCL 3.0 CUDA platform (driver 550.163.01) | host sources sha256 prefix 10e08ea20584151f (see note 2); OpenCL kernels private (results note section 1) | 80521b40 | 16 (run twice) | `d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df` | MATCH |
| 2026-10-16 | Lab host B GPU: NVIDIA GeForce RTX 2070 Max-Q, NVIDIA OpenCL | OpenCL port, hardware div/sqrt build | NVIDIA OpenCL 3.0 CUDA platform (driver 550.163.01) | host sources sha256 prefix 10e08ea20584151f (see note 2); OpenCL kernels private (results note section 1) | 80521b40 | 128 | `22f69ad87a615d66a77efaca8b1172d22bdfb4bd5092ceffd302e35669a050f6` | MATCH |

Every row above is a lab-run row (pre-registered leg, ledger batches 38 and 39); the software unit test (`make test_soft`) reported PASS on every device listed with a MATCH row. The run that produced the three iGPU rows was the second run of the leg: run 1 on each host produced no iGPU row because the worker account could not open the DRM render node (an infrastructure failure, recorded in the ledger as such); the bar was unchanged between runs. Host B's Intel UHD 630 row is still queued and will be added when it has run.

## 7. Scope

One model, one prompt, 16 and 128 tokens, sequential reduction, one
implementation (the OpenCL port of the C++ reference). The builder rows are
from one machine and three devices. The claim is limited to what the tables
show: three OpenCL devices reproduce the pinned digests (the iGPU only with
the software divide, square root and binary64 routines), and one device's
hardware divide and square root, measured as not correctly rounded, produces a
different digest. No statement is made about other devices, drivers or OpenCL
versions, and none about OpenCL conformance of any implementation. No
clean-room implementation ran these builds. The C++ port source is not part of
this tree (see note 2 in `docs/REPRODUCTIONS.md`), so the OpenCL sources are
not published here either; the sources sha256 prefix of the builder tree is
`76dcae0f7981358e` (same file concatenation as that note).

Related: `docs/REPRODUCTIONS.md` (CUDA and host rows), `docs/GPU_RESULT.md`.
