# CIS-2 — Canonical Floating-Point Semantics for fp32 Transformer Inference

## Reproduce this in <15 minutes

```
git clone https://github.com/Aefinity-AI/cis2-spec && cd cis2-spec && ./selfcheck.sh
```

Expected output ends with (trimmed to the machine-fingerprint and summary
lines; full log runs longer with build output in between):

```
== machine fingerprint ==
cpu_model: Intel(R) Core(TM) i5-5200U CPU @ 2.20GHz
isa_flags(relevant): avx,avx2,fma,sse4_2
os: Linux aefinity-box 6.12.94+deb13-amd64 #1 SMP PREEMPT_DYNAMIC Debian 6.12.94-1 (2026-06-20) x86_64 GNU/Linux

== SUMMARY: 9 passed, 0 failed ==
PASS: all reproducible pinned digests match this repository's own EXPECTED_DIGESTS.md.

real	3m30.629s
user	0m53.911s
sys	0m2.095s
```

**Machines this has been run on**

| Machine | CPU | ISA | OS/kernel | selfcheck.sh wall-clock (demo convenience, not throughput) | Full run |
| --- | --- | --- | --- | --- | --- |
| box1 | Intel i5-5200U | x86_64 AVX2 | Linux 6.12.94+deb13-amd64 | 3m30s | trimmed output in PR #21 |
| box2 | Intel Celeron N4020 | x86_64 scalar (no AVX2) | Linux 6.12.94+deb13-amd64 | 5m58s (margin under the <10min target is thin) | trimmed output in PR #21 |
| box3 | Intel Celeron N4020C | x86_64 scalar (sse4_2 only, no AVX2) | Linux 6.12.94+deb13-amd64 | 8m32s (margin under the <10min target is thin) | trimmed output in this PR |
| penguin | Intel i5-10210U | x86_64 AVX2 | Linux 6.6.147-09642-gea7f90d2e99e (ChromeOS Crostini container, Debian 13) | 1m58s | trimmed output in PR #21 |
| Lightning AI Studio (AWS) | Intel Xeon Platinum 8488C (4 vCPU) | x86_64 AVX2/AVX-512 | Ubuntu 24.04, 6.8.0-1063-aws | — (only `scripts/self_check.sh` run, not `selfcheck.sh`) | [2026-09-25 note](docs/results/2026-09-25-LIGHTNING-STUDIO-SELFCHECK.md) |
| phone | TBD | aarch64 (Android) | — | not yet run | pending |

One caveat on the penguin row: that machine's resolver could not reach the
Hugging Face LFS CDN host at run time, so the three pinned artifacts were
copied to it over the LAN instead of downloaded. `scripts/fetch_weights.sh`
accepts a pre-existing file only when its sha256 equals the pin recorded in
that script, so the fetch step still verified the same three hashes; every
later step ran normally on that host.

## Shorter path: digests only

This is the narrower of the two entry points: it checks the four pinned
digests with the C verifier only, where `./selfcheck.sh` above runs all
nine checks across both clean-room implementations.

```sh
git clone https://github.com/Aefinity-AI/cis2-spec && cd cis2-spec
scripts/self_check.sh
```

That single command fetches + sha256-verifies the pinned
`HuggingFaceTB/SmolLM2-135M` weights from Hugging Face (no account or
token needed), builds the C clean-room verifier (`verify3/`) with your
system `gcc`, runs it against the pinned §13.1 test vector, and diffs
every digest against `EXPECTED_DIGESTS.md`. Expected last line:

```
PASS: all digests match the pinned CIS-2 v0.3b test vector.
```

Requires `git`, `gcc`, `make`, `curl`; `objdump` is used for an extra FMA
check if present. No GPU and no Rust toolchain required for this path.
Measured end-to-end wall time on a fresh clone: ~5 minutes (demo
convenience, not throughput), dominated by the ~257 MB weight download
over the tester's network connection, not by the build or the verifier run
themselves.

**Independent reproduction.** The machine/compiler/ISA classes this
repository documents as already having reproduced the primary `CIS2_REF`
digest above are listed in `EXPECTED_DIGESTS.md`, `docs/GPU_RESULT.md`,
and `.github/workflows/verify.yml` (currently: x86_64 and aarch64
GitHub Actions runners, gcc and clang, and — as of 2026-09-08 — an
NVIDIA Tesla P100 CUDA implementation). This is not a claim that no other
environment could diverge; it is a record of which environments have been
checked and are re-checked on every push by CI.
**Claim:** given a normative specification (`docs/CIS2_SPEC_v0.3b.md`) for
an fp32 transformer forward pass — pinned floating-point environment,
pinned reduction order, pinned transcendental polynomials, pinned digest
format — independently-written implementations reproduce **bit-identical**
full-logit output digests:

- across instruction set architectures (x86_64 and aarch64),
- across compilers and optimization levels (gcc, clang, rustc; `-O0`
  through `-O3`/`-Os`),
- across languages (Rust and C, written by separate clean-room passes that
  never read each other's source or the reference implementation),
- across model families and decode horizons (see `EXPECTED_DIGESTS.md`),
- and, as of 2026-09-08, across the CPU/GPU boundary: a CUDA implementation
  on an NVIDIA Tesla P100 reproduces the primary normative `CIS2_REF`
  digest below bit-for-bit, with a byte-identical per-step trace
  (see `docs/GPU_RESULT.md`),

for a pinned (model, prompt, decode-length) test vector, matching a
PyTorch/`transformers` oracle within floating-point tolerance.

**Where to get it.** Tagged release:
[`v0.3b`](https://github.com/Aefinity-AI/cis2-spec/releases/tag/v0.3b). The
spec, the five op-level conformance vectors, `EXPECTED_DIGESTS.md` and
`docs/GPU_RESULT.md` are also mirrored on Hugging Face as
[`aefinityAIINC/cis2-conformance`](https://huggingface.co/datasets/aefinityAIINC/cis2-conformance),
regenerated from this repository by `scripts/publish_hf.sh`.

**What the clean-rooms in this repository actually verify:** `verify2/`
(Rust) and `verify3/` (C) — the two independent, from-spec-text-only
implementations shipped here — reproduce the primary bit-identical
`CIS2_REF` digest for `HuggingFaceTB/SmolLM2-135M`, prompt `"Once upon a
time"`, 16 greedy-decoded tokens, on both x86_64 and aarch64 (`verify3/`
additionally under both gcc and clang). This is the pinned §13.1 test
vector in `docs/CIS2_SPEC_v0.3b.md` and `EXPECTED_DIGESTS.md`. The 20-cell
compiler/ISA invariance matrix, the Qwen2.5-0.5B/1.5B results, and the
128-/512-token decode horizons were obtained with `src/`, the Rust
reference implementation included in this repository, not with the
clean-rooms; they are listed as **informative** evidence in
`EXPECTED_DIGESTS.md`, not as claims about `verify2/`/`verify3/`.

This is a narrower and harder claim than integer/bitwise determinism:
floating-point addition is not associative, so this does not claim "any
reduction order is safe." It claims that *one* pinned reduction order,
*one* pinned transcendental route, and *one* pinned digest encoding are
collectively sufficient for bit-identity — and states each of them
explicitly enough that a reader who has never seen a reference
implementation can reproduce them.

## Scope

This repository contains the **specification**, the **reference
implementation** (`src/`) the spec was audited against, and **two
independent clean-room verifiers**. `src/` lets readers cite exact source
locations for their own sanity-checking (see the spec's Appendix A) and
lets anyone reproduce the informative 20-cell compiler/ISA matrix and
second-model-family results in `EXPECTED_DIGESTS.md`. Reproducing `src/`
is not required to trust the spec, though: `verify2/` (Rust) and
`verify3/` (C) were each written from the specification text alone,
without access to `src/` or to each other, and are the artifacts this
repository asks readers to trust and extend independently of the
reference. Not included: `docs/paper/` (the in-progress paper draft) and
the private repository's own CI workflow definitions (their published
run results and logs are included under `hardware_logs/` and
`docs/logs/`, but the `.yml` files themselves were internal-repo-specific
and are not reproduced here — `.github/workflows/verify.yml` in this
repository is the public equivalent).

## Reproduce manually (equivalent to `scripts/self_check.sh`)

```sh
git clone https://github.com/Aefinity-AI/cis2-spec && cd cis2-spec
scripts/fetch_weights.sh weights          # fetches + sha256-verifies HuggingFaceTB/SmolLM2-135M
make -C verify3                            # builds the C clean-room (verify3/)
./verify3/cis2_verify3 weights/model.safetensors weights/config.json weights/tokenizer.json
```

Expected result: `CIS2_VERIFY3 digest=d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df`,
matching `EXPECTED_DIGESTS.md`. `verify2/` (Rust) reproduces the same
digest; see `.github/workflows/verify.yml` for the exact build/run
sequence on both x86_64 and aarch64 CI runners.

No throughput numbers (tokens/sec and the like) are published anywhere in
this repository. The only timings published are end-to-end wall-clock
figures — the machine table above and the shorter-path note — labelled in
both places as (demo convenience, not throughput): they tell you how long
to wait, not how fast anything is. The reference and both clean-room
verifiers are scalar, unoptimized-for-speed implementations whose only goal
is bit-exact, auditable determinism, not throughput.

`scripts/self_check.sh` extracts digests from verifier output using
`grep -P` (PCRE lookbehind) when GNU grep is available, and automatically
falls back to a portable `grep -E` + `sed` form (equivalent for this
fixed-format output) when it is not — so it also runs on macOS/BSD grep.

## Provenance

Both clean-room verifiers in this repository — `verify2/` (Rust) and
`verify3/` (C) — were written by isolated AI coding agents (Claude, via
Claude Code), each operating from `docs/CIS2_SPEC_v0.3b.md` alone, with no
access to the private reference implementation and no access to each
other's work. Both agents were operated by the same sole author/operator
(Aefinity AI Inc.) on the same day; the independence claim rests on the
clean-room read-access boundary enforced during each implementation pass,
not on separate human authors or separate calendar days. Each
implementation directory contains a `CLEANROOM_LOG.md` recording, in
order, every file each agent read and why, as the evidence for that
boundary — see `verify2/CLEANROOM_LOG.md` and `verify3/CLEANROOM_LOG.md`.

## Threat model summary

What this claim rules out:

- A verifier silently using a different reduction order, transcendental
  approximation, or rounding mode that happens to produce the same
  *tokens* but different *logit bits* — the witness digest (`CIS2_REF`,
  spec §12.1) folds in the full fp32 logit vector for every decode step,
  not just the argmax winners, so it cannot be gamed by token-level luck.
- Silent divergence introduced by a specific ISA (denormal/FTZ-DAZ
  handling differs between x86 MXCSR and ARM FPCR by default; the spec
  pins both), a specific compiler's instruction selection (FMA fusion is
  explicitly disallowed and disassembly-checked), or a specific
  optimization level.

What this claim does **not** rule out (see spec §14 "Known gaps" for the
full, honest list):

- General correctness of the pinned transcendental polynomials outside
  the input ranges actually exercised by the tested decodes.
- Cross-framework agreement with PyTorch as a *conformance* requirement —
  that is evidence collected alongside the spec, not part of what
  "conforming" means.
- Any claim about sampling, batching, quantization, or non-greedy decode —
  out of scope by construction (spec §0).
- A `CIS2_REF` mismatch, by itself, localizing *which* of the five seeded
  inputs diverged (spec §14.3) — a verifier debugging a mismatch should
  compare `table_digest`, `inv_freq_table_digest`, and the three artifact
  hashes individually, not just the final witness digest.

## How to submit your own clean-room implementation

1. Read `docs/CIS2_SPEC_v0.3b.md` only. Do not read `verify2/` or
   `verify3/`'s source before or during your implementation — treat them
   the same way this repository's own authors treat the (private,
   unpublished) reference implementation: off-limits until your
   implementation is complete.
2. Keep a `CLEANROOM_LOG.md` in your implementation's directory listing,
   in order, every file you read and why (see `verify2/CLEANROOM_LOG.md`
   and `verify3/CLEANROOM_LOG.md` for the expected format and level of
   detail).
3. Your implementation MUST, per spec §15:
   - reproduce every value in §13.1 bit-for-bit on x86-64;
   - reproduce every value in §13.1 bit-for-bit on aarch64, unmodified
     source;
   - contain zero FMA-family instructions on the decode path in its
     release binary (disassembly-verified, e.g. with `objdump -d`);
   - pass its own adversarial denormal self-test (spec §1.3);
   - pass the same-process two-run determinism check (spec §12.4).
4. If your implementation's `CIS2_REF` does not match
   `EXPECTED_DIGESTS.md`, first check whether `table_digest`,
   `inv_freq_table_digest`, and `argmax_digest`/`generated_token_ids`
   individually match — a full match on those but not on `CIS2_REF` is
   most likely a witness-chain item-order or byte-encoding bug (see spec
   §12.1's item order, corrected in v0.2 after exactly this failure mode
   was caught by an earlier clean-room pass), not a numeric error in your
   forward pass.
5. Open an issue or pull request with your implementation, its
   `CLEANROOM_LOG.md`, and its CI run reproducing (or failing to
   reproduce) `EXPECTED_DIGESTS.md`. Mismatches are useful — they are how
   this spec's own item-order bug was found (see `CHANGELOG.md`).

## Try to break it

CIS-2 predicts a specific hash on hardware nobody here has run it on
yet. `CHALLENGE.md` defines what counts as a divergence vs. an
out-of-scope failure; `HALL-OF-DIVERGENCE.md` credits every verified
report by name (credit-only: no cash, no prize, no payment). Report via
[`divergence-report.yml`](.github/ISSUE_TEMPLATE/divergence-report.yml)
or [`reproduction-report.yml`](.github/ISSUE_TEMPLATE/reproduction-report.yml)
— or open one directly:
[divergence](../../issues/new?template=divergence-report.yml) /
[reproduction](../../issues/new?template=reproduction-report.yml).

## Paper

arXiv link: pending — updated on publication. See `CITATION.cff`.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.

## Repository layout

```
docs/CIS2_SPEC_v0.3b.md   the normative specification (current)
docs/CIS2_SPEC_v0.2.md   superseded, kept for history
CHANGELOG.md             v0.1 -> v0.2 -> v0.2.1 -> v0.3 -> v0.3b changes
EXPECTED_DIGESTS.md       pinned + informative digest values
src/                      reference implementation (Rust), the spec's own
                          audit trail (Appendix A cites src/*.rs:line)
verify2/                  clean-room verifier #1 (Rust), written from the
                          spec text alone, no access to src/
verify3/                  clean-room verifier #2 (C11), written from the
                          spec text alone, no access to src/ or verify2/
docs/GPU_RESULT.md        CPU/GPU bit-identity result (P100, 2026-09-08)
docs/                     spec, changelog, and E15* evidence/result notes
docs/logs/, hardware_logs/  raw CI/local-run logs backing those notes
weights/                  fetch manifest only; no weight files committed
scripts/fetch_weights.sh  sha256-verified weight fetch
scripts/self_check.sh     local build + digest reproduction check
tools/evidence_pack.py    builds a schema-validated JSON/Markdown
                          evidence pack (no timing numbers) for one
                          CIS-2 receipt from a receipt dir + verifier
                          output
   (verify3/); see below for the src/ reference's own run command
.github/workflows/verify.yml  CI: builds src/ + both clean-rooms on
                               x86_64 + aarch64 (+ gcc/clang for verify3),
                               fails on any digest mismatch
```

## Results

- [Single-weight corruption: text replay vs top-k logit check vs witness
  digest](docs/results/2026-09-13-BITFLIP-HEADTOHEAD.md) — moat-3/moat-3b
  bitflip experiments on a pruned BitNet-2B checkpoint.
- [Receipt-verified EVAL-60 and tamper
  detection](docs/results/2026-09-13-RECEIPT-VERIFIED-EVAL60.md) — 60/60
  receipts verify, tamper detection improves from 3/4 to 4/4 caught after
  adding receipt/item context binding.
- [Realistic embedding substitution: text replay vs top-5 logit check vs
  witness digest](docs/results/2026-09-17-EMBED-SUBSTITUTION-HEADTOHEAD.md)
  — quantization/pruning/fine-tune/surgical-row-patch experiments on a
  pruned BitNet-2B checkpoint's embedding table; only the witness digest
  catches every case, including unexercised single-row patches.
- [GPU batched fp32 determinism: host CPU vs Tesla
  T4](docs/results/2026-09-19-GPU-BATCHED-FP32-DETERMINISM.md) — row-batched
  fp32 matches the CPU reference digest bit-for-bit at batch 1/4/8 on a
  T4; grouped-GEMM and int8 batching mismatch as designed.

## Related tooling

[`receipt-view`](https://github.com/Aefinity-AI/alice-aegis/blob/cm/rc1-receipt-viewer/demo/agent-trace/receipt-view.py)
(in the `alice-aegis` repository, `demo/agent-trace/`) is a single-file,
stdlib-only Python tool that turns an `agent_trace` agent-episode receipt
— the sibling artifact to this repo's CIS-2 witness digests, hash-chaining
a whole K-step agent episode rather than one forward pass — into a
plain-English report: what ran, whether the hash chain is intact or
broken, the first broken step and what changed, and whether the check ran
offline. It shells out to the existing `agent_trace verify` (no
reimplemented hashing/replay) and its own README documents a 4-case
mutation test (changed token, changed tool-result, a flipped bit in an
intermediate decode-chain digest, and a reordered step), each correctly
caught and correctly attributed to the right step. See
[`demo/agent-trace/receipt-view-README.md`](https://github.com/Aefinity-AI/alice-aegis/blob/cm/rc1-receipt-viewer/demo/agent-trace/receipt-view-README.md).

## Build and run the reference implementation (`src/`)

```sh
scripts/fetch_weights.sh weights   # if not already fetched
cargo build --release -j2
nice ./target/release/cis2_ref     # weights/, prompt "Once upon a time", 16 tokens
```

Expected: a `CIS2_REF digest=d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df`
line, matching `EXPECTED_DIGESTS.md` and the `verify2`/`verify3` clean-room
result above. No timing numbers are printed or recorded by this
repository (Rule A); `nice` is used only to be a considerate neighbor on
shared machines, not to produce a timing measurement.

## Acknowledgments

A special thank you to Charles Seaman and Linda Blanchard, whose contributions have helped Aefinity AI stay on track.

And a very special thank you to **Bonnie Rae Power**: an amazing woman, a great friend and neighbor, without whom Aefinity AI would have never had a chance to ever get started. Thank you, Bonnie, for your advice, care, encouragement, guidance, intuitive wisdom, and financial assistance.
