# Contributing

The most valuable contribution to this project is an implementation written
by someone who has never seen ours, and the second most valuable is a
specification defect. Both have a process below.

## 1. An independent implementation

This is the point of the repository. The full procedure is in `README.md`
under **"How to submit your own clean-room implementation"** — read that, not
this section, for the requirements. The short version:

- Read `docs/CIS2_SPEC_v0.3b.md` and nothing else. `src/`, `verify2/`,
  `verify3/` and `cis2-verify/` are off-limits until you are done.
- Keep a `CLEANROOM_LOG.md` recording every file you read, in order, and why.
- Satisfy every item in spec §15 before you open the PR.
- Open the PR whether your digests match or not. A mismatch is a result.

## 2. A specification defect

Open an issue. Say which clause, what two conforming implementations could do
differently under it, and — if you can — a concrete input that separates them.
A defect backed by a measurement that shows the divergence is worth far more
than one backed by an argument, but the argument alone is still welcome.

Defects become numbered errata in `CHANGELOG.md`. The rule is that goldens do
not move silently: if an erratum changes a pinned digest, the change, the
cause and the new value all land in the same commit as the version bump.

## 3. Code changes to the verifiers

- CI must be green on both `ubuntu-24.04` and `ubuntu-24.04-arm` before
  merge. The digest checks are the gate; a build that compiles but prints a
  different digest is a failure, not a warning.
- No FMA-family instruction may appear on the decode path of a release
  binary. CI disassembly-checks this; do not work around the check.
- Do not add a dependency to `cis2-verify/`. It is zero-dependency on
  purpose, so that a reader auditing it has a finite amount to audit.
- Floating-point arithmetic in a test must not be reachable by the compiler's
  constant folder. Route every operand through `std::hint::black_box` in both
  directions — see the doc comment on `cis2-verify/src/fpenv.rs`'s test module
  for the failure this prevents, which we shipped once and caught by
  measurement.

## House rules for anything written into this repository

These are the rules the maintainers hold themselves to, and PRs are held to
them too.

**Provenance on every number.** A result reported in a doc or a log names the
host, the architecture, the commit, and the artifact hashes it was produced
from. A number with no provenance gets removed rather than trusted.

**No timing numbers.** Nothing in this repository publishes tokens/sec,
wall-clock, or any other performance figure. The implementations here are
scalar and deliberately unoptimized; a timing number measured from them would
be meaningless, and one measured on a virtual machine would be worse than
meaningless. If you want to benchmark a conforming implementation, do it
elsewhere and say what iron it ran on.

**No unqualified superlatives.** Not "the first", not "the only". Write the
claim you can defend with an artifact in this repository, at the width the
artifact actually supports, and let a reader who checks it find it true. The
existing wording in `README.md` — which distinguishes what the clean-rooms
verify from what the reference implementation verifies, and marks the latter
informative — is the standard to match.

**Name the limitation in the same document as the result.** Spec §14 ("Known
gaps") and the "What this claim does not rule out" section of `README.md` are
load-bearing, not disclaimers. A PR that adds a result adds its limits too.

## Code of conduct

See `CODE_OF_CONDUCT.md`.

## License

By contributing you agree that your contribution is licensed under
Apache-2.0, the same terms as the rest of the repository (`LICENSE`,
`NOTICE`).
