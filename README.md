# CIS-2 — Canonical Floating-Point Semantics for fp32 Transformer Inference

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

## Reproduce in 5 commands

```sh
git clone <this-repo-url> cis2-spec && cd cis2-spec
scripts/fetch_weights.sh weights          # fetches + sha256-verifies HuggingFaceTB/SmolLM2-135M
make -C verify3                            # builds the C clean-room (verify3/)
./verify3/cis2_verify3 weights/model.safetensors weights/config.json weights/tokenizer.json
scripts/self_check.sh                      # does all of the above and diffs against EXPECTED_DIGESTS.md
```

Expected result: `CIS2_VERIFY3 digest=d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df`,
matching `EXPECTED_DIGESTS.md`. `verify2/` (Rust) reproduces the same
digest; see `.github/workflows/verify.yml` for the exact build/run
sequence on both x86_64 and aarch64 CI runners.

No timing numbers (tokens/sec, wall-clock, etc.) are published anywhere in
this repository. The reference and both clean-room verifiers are scalar,
unoptimized-for-speed implementations whose only goal is bit-exact,
auditable determinism, not throughput.

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
   (verify3/); see below for the src/ reference's own run command
.github/workflows/verify.yml  CI: builds src/ + both clean-rooms on
                               x86_64 + aarch64 (+ gcc/clang for verify3),
                               fails on any digest mismatch
```

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
