# Try to break CIS-2

CIS-2 (`docs/CIS2_SPEC_v0.3b.md`) specifies fp32 transformer inference
precisely enough — pinned floating-point environment, pinned reduction
order, pinned transcendental polynomials, pinned digest format — that
independent implementations, on different machines, produce **bit-identical**
witness and argmax digests for a pinned (model, prompt, decode-length)
test vector. That is a falsifiable claim: it predicts a specific hash
value on hardware nobody here has run it on yet.

Try to break it. Verified divergences and reproductions are credited by name in HALL-OF-DIVERGENCE.md.

There is no cash, no prize, and no payment attached to reporting a
divergence or a reproduction — the incentive is credit only, recorded
in `HALL-OF-DIVERGENCE.md` under your name (or handle) with a link to
your evidence.

## What counts as a divergence

**In scope — a spec divergence:**

- You run this repository's self-check (the `selfcheck.sh` entry point
  described in the README) on a conforming machine, every build/gate
  step passes, and one or more pinned digests (witness `CIS2_REF`,
  `argmax_digest`, `table_digest`, `inv_freq_table_digest`) comes out
  **different** from the value pinned in `EXPECTED_DIGESTS.md`.
- You find a genuine ambiguity in `docs/CIS2_SPEC_v0.3b.md` — wording
  that a second good-faith, spec-only implementer could reasonably read
  two different ways, producing two different but each internally
  consistent results. Cite the section number.
- An op-level conformance vector (`tests/conformance/vectors/`) that a
  spec-conforming implementation cannot reproduce, despite following the
  spec's operation-level description exactly.

**In scope, different class — a conformance-environment failure:**

- The self-check fails because the *machine*, not the spec, is
  non-conforming: an FMA-family instruction shows up in the built
  binary's disassembly, FTZ/DAZ (flush-to-zero / denormals-are-zero)
  is not pinned the way §1.3/§1.4 require, or the compiler
  auto-vectorized a multiply-add into a fused instruction the spec
  disallows.
- This is still worth reporting, and still gets credited — as a
  **reproduction report** with the environment detail attached, in the
  divergences table if the cause is unresolved or the reproductions
  table once explained — because it demonstrates the gate itself
  (spec §1.4's FMA check) is doing its job. It is not evidence the
  specification is wrong.

**Out of scope — not a CIS-2 divergence:**

- Network or download failures.
- A sha256 mismatch on a weight file — that is a bad or truncated
  download, not a divergence; re-fetch and retry before reporting.
- A missing or wrong toolchain (no `gcc`, no `cargo`, wrong version
  such that the build itself fails, rather than running and disagreeing).
- A different model, different weights, or a different prompt than the
  ones pinned in `EXPECTED_DIGESTS.md` / spec §13.1.
- Non-fp32 paths: quantized, int8, or mixed-precision runs.
- Non-greedy decode, sampling, or batching — out of scope by spec §0.
- GPU paths not already covered by `docs/GPU_RESULT.md`.
- Timing differences of any kind — see `README.md`'s Rule A: no
  tokens/sec figures, wall-clock only when explicitly labeled
  "(demo convenience, not throughput)".

## How to report

Open a GitHub issue using the `divergence-report.yml` or
`reproduction-report.yml` template under
`.github/ISSUE_TEMPLATE/` (or start from
`/issues/new?template=divergence-report.yml`). Attach, unedited:

1. The full `selfcheck.sh` stdout+stderr — the whole thing, not a
   trimmed excerpt.
2. The machine fingerprint block the script prints at the end (CPU
   model, ISA flags, OS/uname).
3. `git rev-parse HEAD` of your checkout.
4. Your compiler version: `gcc --version` (or `cc --version`).
5. The sha256 of every fetched weight file
   (`model.safetensors`, `config.json`, `tokenizer.json`).

Reports missing any of the above will most likely just get a request for
the missing piece before anything can be reproduced on our end.

## What happens next

We attempt to reproduce your result on our own fleet. We say publicly
whether it reproduced. If it does, either the specification or
`selfcheck.sh`/the reference implementation gets fixed and the fix is
recorded as an errata entry in `CHANGELOG.md`, or — if the cause turns
out to be environment-specific (an FMA-generating compiler flag, an
unpinned FTZ/DAZ setting, and so on) — the report is recorded in
`HALL-OF-DIVERGENCE.md` with that explanation. Either way, you are
credited by name.

## Honesty note

We will publish divergences that make CIS-2 look bad, including ones we
cannot immediately explain. A specification that only holds up when
nobody tests it is worthless. If you break it, that is the useful
outcome — not an embarrassment to be managed.
