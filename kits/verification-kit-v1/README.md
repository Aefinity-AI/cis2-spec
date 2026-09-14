# Third-party verification kit v1 — CIS-1 / CIS-2 receipts

Everything here is static, offline, and runs without building anything.
No network access is required or used by any script in this kit.

## Contents

- `bin/x86_64/cis-verify`, `bin/aarch64/cis-verify` — standalone CIS-1
  witness-receipt verifier, static (`crt-static`, no `PT_INTERP`, no
  dynamic section, no dynamic linker needed at all). x86_64 build is
  `x86_64-unknown-linux-musl`; aarch64 build is
  `aarch64-unknown-linux-gnu` with `crt-static` (glibc static, not musl —
  cross toolchain constraint, still fully static per `readelf`).
- `bin/x86_64/agent_trace`, `bin/aarch64/agent_trace` — CIS-2
  agent-episode trace verifier (same build approach per architecture).
- `verify.sh` auto-selects `bin/$(uname -m)/` — the correct binaries for
  the machine it runs on. If your architecture isn't one of the two
  bundled here, `verify.sh` fails loudly and tells you what it looked
  for, rather than silently running the wrong binary.
- `model/` — the small "tinybit" (~3.8M param) M7 model triple
  (`MODEL.SAF`, `EMBED.BIN`, `VOCAB.BIN`) both receipts were run against.
- `receipts/cis1_witness.receipt` — a golden CIS-1 `AEGIS-WITNESS v1-CIS`
  receipt (greedy decode, 64 tokens).
- `receipts/cis2_agent_trace.receipt` + `receipts/chain.tsv` — a CIS-2
  `AEGIS-TRACE v2` agent-episode receipt and the lookup table it used.
- `verify.sh` — runs both verifiers against the bundled receipts, diffs
  the output against pinned `EXPECTED_DIGESTS.txt`, fails loudly on any
  mismatch.
- `tamper-demo.sh` — makes 4 tampered copies of the bundled receipts (one
  distinct tamper class each) and shows all 4 correctly FAIL.
- `SHA256SUMS.txt` — sha256 of every other file in this kit. Check with
  `sha256sum -c SHA256SUMS.txt` from this directory.

## Run it

```
./verify.sh          # both verifiers, PASS expected, diffed against pins
./tamper-demo.sh      # 4 tampers, all 4 must FAIL (exit 0 means they did)
```

## What a PASS proves

A receipt that verifies proves a conforming computation over the exact
bound artifacts (hashed in the receipt) reproduced the exact bound outputs
— every generated token id and every folded digest — bit-for-bit, using a
second, independent implementation (`cis-verify`/`agent_trace` share no
code with the engine that produced the receipts).

## What a PASS does NOT prove

- Does not prove *which physical machine* ran the original computation —
  that is platform attestation's job, a separate layer this format does
  not attempt (the `host`/`commit` fields are bound into CIS-2 receipts so
  a receipt cannot be silently relabelled to a different host, but nothing
  here attests that a *given* host string corresponds to real hardware).
- Not a zero-knowledge proof — the model and prompt are hashed, not
  concealed.
- Does not protect against an adversary who controls both the artifacts
  and the verifier together.
- These two verifiers are spec transcriptions built with the CIS-1/CIS-2
  spec and receipt formats as reference material, not clean-room audits by
  a party working blind to the reference implementation — that is
  evidence the spec is independently re-implementable, not the same claim
  as an external auditor's clean-room review.
- The CIS-2 verbatim-argument grounding check (`--strict-grounding`) is
  sound but not complete: off by default here (lenient mode), so a
  fabricated-but-plausible tool argument may only WARN, not FAIL, unless
  you re-run with `--strict-grounding`.
- This kit uses the small tinybit (~3.8M param) model only. It does not
  demonstrate throughput, timing, or behavior at BitNet-2B scale.

## Verifying a full 2B-scale receipt

Too large to ship in this kit. The reference implementation, the 2B model
artifacts, and 2B-scale receipts (including a bare-metal/UEFI boot leg)
live in the `alice-aegis` repository — see `docs/paper/08_limitations.md`
for exactly what is and is not established at 2B scale, and `cis-verify/`
in that repo for the same verifier's full source and golden fixtures.
`cis-verify` and `agent_trace` both take the same three-artifact-plus-
receipt argument shape shown here; only the artifact triple and receipt
change for a 2B run.

## Integrity of this kit itself

`SHA256SUMS.txt` in this directory covers every other file in the kit.
`verify.sh` checks it automatically as its last step.

The whole kit is also shipped as `kits/verification-kit-v1.tar.gz` in this repository. Its checksum is kept in the detached file `kits/verification-kit-v1.tar.gz.sha256` (the README cannot carry a hash of an archive that contains the README):

```
sha256sum -c kits/verification-kit-v1.tar.gz.sha256
```

Note: because this hash is text *inside* the kit, and the kit is what
gets tarballed, this line necessarily describes the immediately-prior
build, not the exact bytes of the tarball you are looking at right now
(a self-referential hash cannot describe its own container). Treat it
as a sanity pointer, not the canonical value — the canonical value for
any given release is whatever `sha256sum verification-kit-v1.tar.gz`
reports on the artifact you actually downloaded, and what's recorded in
the PR/release notes for that artifact.

## Verified on

`./verify.sh` (both checks: bundled verifiers PASS with the pinned
digests, and the sha256 manifest) and `./tamper-demo.sh` (all 4 tampers
correctly rejected) have been independently re-run, from a clean git
checkout of this kit, on:

- x86_64 Linux — two separate machines (the machine this kit was built
  on, and a second, independent machine).
- aarch64 Linux — a Samsung Android phone (via `sh verify.sh`; no bash
  on device), running the cross-built `bin/aarch64/` binaries under the
  device's real kernel (no emulation on-device).
