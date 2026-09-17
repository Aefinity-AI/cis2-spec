# What "independent" means for this crate, and what it does not

`cis2-verify` reproduces every pinned digest of CIS-2 v0.3b from the specification text. This document
states exactly what that does and does not establish, because the value of the result is entirely in
the precision of the claim.

## What is true, and mechanically checkable

**No shared source.** This crate shares no source file, no module, and no function with `cis2_ref`
(the reference implementation at the repository root) or with `verify2`/`verify3`. It is a separate
Cargo package with its own `[workspace]` table, so the build system enforces the separation rather
than leaving it to discipline.

**No shared dependencies.** `[dependencies]` is empty. The reference uses `safetensors`, `tokenizers`,
`sha2` and `serde_json`; this crate re-derives all four capabilities:

| Capability | Reference | Here |
|---|---|---|
| SHA-256 | `sha2` crate | `src/sha256.rs`, from FIPS 180-4 |
| JSON | `serde_json` | `src/json.rs` |
| safetensors container | `safetensors` crate | `src/safetensors.rs` |
| byte-level BPE | `tokenizers` crate (with `onig`) | `src/tokenizer.rs`, from spec 3.1, including a hand-written matcher for the GPT-2 pre-tokenizer regex |
| Unicode letter/number/space classes | `onig` | `src/unicode.rs`, generated range tables |
| `sqrt`, `exp`, `ln` | libm / reference tables | `src/softfp.rs`, `src/mathpin.rs` |

A shared dependency would be a shared implementation. Two programs that both call `sha2` agree about
SHA-256 because they are the same code, not because they read the same document.

**No FMA, checked in the emitted machine code.** `tools/check_no_fma.sh` disassembles the built binary
and fails on any FMA mnemonic. It reports PASS with 0 hits on this crate, and it has a negative
control: it correctly FAILs on a C binary built with `-mfma -ffp-contract=fast`. A check that cannot
fail proves nothing, so the negative control is part of the claim.

**The agreement is exact, and sensitive.** All four pinned digests and all sixteen token ids match.
A single flipped low mantissa bit in one bf16 weight moves the witness digest, and `verify` rejects a
receipt whose artifact hashes disagree with what is on disk.

## What is NOT claimed

**This is not a clean-room implementation in the legal sense.** A clean room requires that the
implementer never had access to the original. That is not the situation here: this crate was written
inside the same project as the reference, by an author with prior exposure to it. Calling it a
clean-room reimplementation would be an overclaim, and the estate's whole argument depends on not
making overclaims.

What this crate demonstrates is narrower and still worth having: **that the specification text is
detailed enough to build a working, bit-exact implementation from, using none of the reference's
code or dependencies.** It is evidence of specification sufficiency. It is not evidence of
implementer independence.

**The independent-implementer evidence is elsewhere, and it is the stronger asset.** The claim that
*strangers* implemented this document down to identical bits rests on the earlier third-party
implementations and the x86 / ARM / NVIDIA P100 reproductions — not on this crate. When the two are
cited together, they should be cited for different things:

- third-party implementations → *strangers can do it* (independence)
- this crate → *the document alone is sufficient, with no shared machinery* (sufficiency)

Conflating them would weaken both.

## What the specification did not pin down

Places where the spec was ambiguous enough that this implementation had to refuse rather than guess.
Each is a hard error with a section citation, and each is a candidate erratum:

- **Tied embeddings vs. a present `lm_head.weight`.** §11.1 does not say which wins if a checkpoint
  sets `tie_word_embeddings = true` *and* ships an `lm_head.weight` tensor. This crate refuses to
  load such a checkpoint rather than pick one.
- **Numeric literal precision in `config.json`.** The pinned config writes `initializer_range` with 17
  significant digits, more than a single correctly-rounded decimal→binary conversion covers. The
  parser leaves such a literal unconverted; asking for its value fails loudly, while a field nobody
  reads passes through. The spec does not state a precision limit for config numerics.
- **`rope_theta` written as a bare integer** (`100000`, not `100000.0`). Handled, but the spec does not
  say whether an integer spelling is permitted for a float-valued field.

## Reproducing

```
cargo test --release          # 76 tests (55 lib, 8 compare, 7 tokenizer fixture, 6 end-to-end)
cargo build --release
tools/check_no_fma.sh
./target/release/cis2-verify selftest
./target/release/cis2-verify run ../weights -o receipt.txt
./target/release/cis2-verify verify ../weights receipt.txt
./target/release/cis2-verify check receipt.txt          # no weights needed
./target/release/cis2-verify compare a.txt b.txt        # no artifacts at all
```

`selftest` needs no weights: it pins the floating-point environment, self-tests FTZ/DAZ against
denormal inputs, and reproduces §6.6's `table_digest` from the pinned polynomials alone.

`check` needs no weights either, and is the tier to reach for when the question is "is this receipt
worth replaying". It audits canonical form and every field recomputable without the 270 MB
checkpoint: §6.6's table from nothing at all, §7.2's `inv_freq` table from `config.json`, §3.3's
prompt token ids from `tokenizer.json`, and — when the receipt names the three artifact hashes §2.1
pins together with §3.2's prompt and §3.4's length — every remaining field, because §13.1 already
fixes them. It then prints the **residual**: the fields it did not establish, and therefore what a
subsequent `verify` would still buy. A skip is printed as loudly as a failure, so a receipt that
passes with five fields skipped is visibly not the same as one that passes with none. What `check`
can never establish, at any tier, is that the logits behind `witness-digest` came from running the
model; only `verify` does that.

`compare` needs no artifacts at all --- not even `config.json` --- because it answers a question
about two documents rather than about a run. CIS-2's fields split into the **inputs** that name the
computation (the three artifact hashes, the prompt, `gen-toks`, `dtype`, `spec-version`), the
**derived** fields that are functions of those inputs alone (§3.3's prompt token ids, §6.6's table,
§7.2's `inv_freq` table), and the **outputs** that only a forward pass produces
(`generated-token-ids`, `argmax-digest`, `witness-digest`). §1.4 then does the work: identical
inputs must give identical outputs on any conforming target, so two receipts that agree on every
input and disagree on any output cannot both be conforming. `compare` reports that as
`CONTRADICTION` and exits 1.

It is as careful about the verdicts it will not reach:

- **`DIFFERENT-RUN`** when any input differs --- the outputs were never required to agree, so no
  contradiction is established, and the exit status is 0. When `weights-sha256` is the *only*
  input that moved, the finding names it as the model-substitution signature, while saying plainly
  that whether the substitution was permitted is a contract question the receipts cannot answer.
- **`WITNESS-COLLISION`** when the receipts name different weights and carry the same
  `witness-digest`. §12.1 digests the full logit stream, so this is not two models coinciding. The
  finding gives both readings --- two safetensors files can serialise the same tensors and hash
  differently, since `weights-sha256` hashes the file rather than the parameters --- and names the
  discriminator, rather than asserting the accusatory one.
- **No verdict at all** from an agreeing `generated-token-ids` or `argmax-digest` under differing
  weights. Those fields carry the argmax stream; two related checkpoints agreeing on a short greedy
  continuation is ordinary. It is surfaced as an observation, never as a finding. `tests/compare.rs`
  holds that case as a negative control, because a tool that accuses falsely is worth nothing in
  the dispute it exists to serve.

What `compare` cannot do is say *which* of two contradicting receipts is honest. That needs the
weights, and the answer is `verify` on each; what `compare` buys is knowing which field the dispute
turns on, and whether paying for a replay would settle anything.

No timing figures are published from this crate's runs on the development machine, which is a
virtualized container (see the project's Rule A).
