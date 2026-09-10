# Security policy

## What "a vulnerability" means for this repository

CIS-2 is a specification plus verifiers that read a model file and print a
digest. It is not a network service, it does not process untrusted input in
production, and it holds no secrets. The interesting failure modes are
therefore not the usual ones, and this policy names both kinds explicitly so
that a reporter knows which channel to use.

### Report privately (GitHub private vulnerability reporting)

Use the **Security → Report a vulnerability** button on
<https://github.com/Aefinity-AI/cis2-spec>. This opens a private advisory
visible only to the maintainers.

Use it for:

- **Memory-safety or parsing bugs** in `verify3/` (C11) or in the safetensors
  and `tokenizer.json` readers of any verifier — a malformed model file that
  causes an out-of-bounds read/write, an unchecked length, or an integer
  overflow in an offset computation. `verify3/` parses attacker-shaped binary
  input by design, so this is the realistic memory-safety surface.
- **A way to make a verifier print a matching digest for a model or decode it
  did not actually compute.** This is the security property the whole
  repository exists to provide. A verifier that can be induced to emit
  `d827430…` without doing the pinned arithmetic defeats the point, whether by
  a caching bug, an early-exit, a digest-chain reuse, or an input-aliasing
  trick. Treat this as the highest-severity class here.
- **Supply-chain issues** in `scripts/fetch_weights.sh` — anything that would
  let a fetched artifact pass the sha256 check it should have failed, or that
  writes outside the target directory.

Please include the model/prompt/decode-length that reproduces it, the
architecture and compiler, and the digest you observed. We will acknowledge
within 7 days and aim to have a fix or a public advisory within 90 days.

### Report publicly (open an issue)

Open a normal issue for **specification defects**. These are not
vulnerabilities under the definition above, they are the thing this project
most wants to receive, and keeping them public is how the record stays
honest:

- A clause that is ambiguous enough that two good-faith implementations could
  read it differently and diverge.
- A clause that pins nothing where the hardware or the compiler is free to
  choose — the class of bug documented in `docs/E24_UNICODE_VERSION_GAP.md`
  (`§3.1.4` named no Unicode version) and in
  `docs/PROPOSED_v0.4_TIER1_OP_GOLDENS.md` (`§13.1` cannot exercise several
  normative clauses, so a conforming-looking implementation could ignore
  them and still pass).
- A digest in `EXPECTED_DIGESTS.md` that your independent implementation does
  not reproduce. Read step 4 of "How to submit your own clean-room
  implementation" in `README.md` first — it lists the failure mode that has
  historically accounted for most mismatches — but if it survives that,
  please file it. A mismatch found by an outside party is more valuable to
  this project than any number of internal green runs.

## Supported versions

The current specification version (`docs/CIS2_SPEC_v0.3b.md`) and the tip of
`main` are supported. Superseded spec versions are kept in `docs/` for
history and are not maintained; errata are recorded in `CHANGELOG.md` and
applied forward only.

## What we will not do

Fix a reported digest mismatch by changing `EXPECTED_DIGESTS.md` to match the
observed value. If a pinned digest is wrong, the cause is found and named in
`CHANGELOG.md` before any value moves, and the version number moves with it.
