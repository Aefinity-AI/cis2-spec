# E15e — clean-room independent CIS-2 fp32 reference implementation

Historical development note (from the reference repository's internal research
log). No timing numbers anywhere (Rule A).

## What this milestone answers

E15b/E15c built and cross-ISA-validated one CIS-2 fp32 reference
implementation. E15e asks: does a *second, independently written*
implementation — built by reading only the design memo and the E15b/E15c
result docs, with the original's `src/`/`tests/` never opened — converge
on the same bits? This is the "third implementation" / clean-room bar the
memo's own §5 (E14d) and CIS-1's own A31 precedent call for.

## Files read to build this (see `verify/CLEANROOM_LOG.md` for full detail)

1. `docs/design/CIS2_FP_RECEIPT_DESIGN.md` — **does not exist** in this
   repo, any branch. The actual memo lived in a separate internal
   operator-notes repository, not published here. Read that file instead,
   since it is unambiguously "the normative memo" the task described, just
   at a different path.
2. `docs/E15b_REFERENCE_RESULT.md` — read in full from this branch's
   working tree.
3. `docs/E15c_RESULT.md` — **not present** in `origin/cm/e15c-cross-isa`'s
   current tip (commit `f83e8dd`). Retrieved via
   `git show 05187dc:docs/E15c_RESULT.md`, a commit reachable only from a
   stale local branch ref, not from any remote branch — `origin/cm/e15c-
   cross-isa` appears to have been reset backward after `05187dc` was
   pushed, dropping the E15c result commit from the remote. Flagged internally for operator follow-up.
4. Root `Cargo.toml` — read only to confirm there is no workspace to join
   and to match dependency versions for a standalone crate.
5. `./target/release/cis2_ref --help` (no output; confirms only the
   `weights/` directory convention).

No file under this repository's `src/` or `tests/` was opened, grepped, or
diffed.

## What was built

`verify/` — a standalone Rust crate, `cis2-verify`, not part of any Cargo
workspace (there isn't one). Implements, from scratch:
- FTZ/DAZ pin (x86_64 MXCSR bits 15/6; aarch64 FPCR bit 24) — memo §3.4.
- bf16 → fp32 exact bit-shift widening — memo §3.5.
- Strict left-to-right sequential sum/dot product, no `.mul_add()`
  anywhere — memo §3.1(a)/§3.2, mechanically gated (grep + `objdump`
  disassembly check for FMA instructions, both in CI and locally; both
  gates PASS — zero FMA instructions in the release binary).
- rsqrt via `1.0f32 / x.sqrt()` — memo §3.3's "cheap win" (route a).
- `exp`/`sin`/`cos` via host `std`/libm (glibc) — this implementation's
  own choice of route (a), since (matching E15b's own finding
  independently) no CORE-MATH/RLIBM Rust crate was available in this
  sandbox either.
- Full SmolLM2-135M forward pass (30-layer Llama/GQA architecture, RoPE
  non-interleaved, SwiGLU MLP, tied embeddings), no KV-cache (full
  recompute per step — mathematically equivalent per the memo/E15b's own
  reasoning, not independently re-verified here).
- Its own SHA-256 witness-chain digest and RoPE `inv_freq` table digest,
  reverse-engineered from E15b's prose description (not a byte-exact spec
  — see `verify/SPEC_GAPS.md`).

## Result — comparison against the E15c preregistered target

| Quantity | Target (E15c) | This clean-room (x86_64 = aarch64, identical) | Match? |
|---|---|---|---|
| Weight/config/tokenizer sha256 | `80521b40...`/`1d556eab...`/`9ca9acdd...` | identical | **MATCH** |
| Tokenization of `"Once upon a time"` | `[6403, 1980, 253, 655]` | identical | **MATCH** |
| Greedy-decoded 16 token ids | `[28, 665, 436, 253, 1838, 8180, 3365, 14176, 30, 2306, 4161, 281, 253, 2066, 2291, 351]` | identical, bit-for-bit | **MATCH** |
| `argmax_digest` (sha256, u32 LE, prompt+gen token ids, reconstructed) | `0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff` | `0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff` | **MATCH** |
| `inv_freq table_digest` | `da9f6dcfde0425588815509e874515cdcd3d6b8818b6d0136590052e7bbf6f12` | `6d20bb9d2a7ec35a70fcc399fed93e17664fa74f95942d9ec77f17ae7a882668` | **MISMATCH** |
| `CIS2_REF` / full fp32 logit witness digest | `ba88708bf4159a4057d4eef727820ebc5ce1744ae334bfe0cc0776e03509dd0f` | `241a73387b22b4f81cd96094efeb6603c3f88a0d10c9fca9a7a1feff5fd7d7bf` | **MISMATCH** |

**Cross-ISA reproducibility of this clean-room implementation itself:**
bit-identical across x86_64 (local + GitHub `ubuntu-latest`) and aarch64
(GitHub `ubuntu-24.04-arm`) — CI run
reference-repository CI run 33190989443 (private CI, not reproducible externally). This
implementation reproduces E15c's own cross-ISA-identity result
independently, even though it does not match E15c's specific bit pattern.

## Verdict: MISMATCH, but an informative one

Per the task's preregistered outcomes, this is a MISMATCH on the two
digests meant to prove bit-exact convergence (`CIS2_REF`, `inv_freq`).
However, three independent facts are **not** mismatches and narrow down
exactly where the divergence enters:

1. Weight/config/tokenizer provenance and tokenization agree exactly —
   the divergence is not in artifact identity or tokenization.
2. The full 16-token greedy-decode sequence agrees exactly, and so does
   an independently reconstructed `argmax_digest` — the divergence is not
   in the decode-path's discrete decisions.
3. The divergence is therefore in the **fp32 logit bit patterns
   themselves** (which feed the full witness digest) and in the
   `inv_freq` table's exact values/encoding — i.e., **the transcendental
   function route** (memo §3.3) and the RoPE `inv_freq` construction
   (unspecified in the memo at all; see below), not the reduction order,
   not FMA, not denormal handling, not bf16 widening (all of which, per
   the memo, are either exact bit operations or mechanically gated and
   were not flagged as differing).

## Spec gaps (full detail in `verify/SPEC_GAPS.md`)

1. **Digest byte encoding is nowhere fully specified.** The memo states
   "SHA-256... chained" in prose; the exact seed order, hash-vs-hex-bytes
   choice, and integer width/endianness for token ids are not given as
   spec text anywhere this task permitted reading. This implementation's
   own choice (u32 LE, single continuous stream, no artifact hashes) for
   the *argmax* digest happened to reproduce the target exactly; the same
   style of guess for the full witness digest (which does need the
   artifact hashes per E15b's prose) did not reproduce, likely because of
   item 2, not because of the encoding.
2. **Transcendental route is not actually pinned — the load-bearing
   gap.** §3.3 recommends correctly-rounded route (a) but no Rust
   crate was available (confirmed independently); the fallback, route
   (b), is *explicitly* "pinned by fiat" per the memo's own words — its
   coefficients are not, and by design cannot be, given in spec text.
   Two clean-room implementations that each independently choose a
   transcendental implementation have no shared reference to converge on.
   This is the single most likely explanation for both digest mismatches.
3. **RoPE `inv_freq` formula/construction is never stated in the memo at
   all** (only its `sin`/`cos` *usage* is discussed in §2's nondeterminism
   table). The standard Llama formula was used here (host `f64::powf`);
   E15b's own doc calls its *own* version of this "a host-libm dependency
   that... has not been pinned or digested," an admitted open item in the
   reference itself, not just in this clean-room attempt.
4. **RMSNorm elementwise multiply order** (`(x*scale)*weight` vs.
   `x*(scale*weight)`) is unstated; minor, not believed to be the actual
   source of the observed mismatch.
5. **Process gap:** `origin/cm/e15c-cross-isa` on GitHub no longer
   contains the E15c result commit (`docs/E15c_RESULT.md`,
   `hardware_logs/e15c_crossisa_2026-08-28.log`) — it was retrievable only
   via a stale local branch ref during this task. Needs attention outside
   this task's scope (see `verify/CLEANROOM_LOG.md` item 0; flagged
   internally for operator follow-up).

## What would resolve the mismatch (not attempted here, out of scope)

Per the task's constraint, this repository's `src/` was never opened to
"debug" this mismatch. If CIS-2 wants two independent implementations to
converge on the *exact* fp32 bits (not just the argmax decisions), the
memo needs to either (a) name a specific, portable, correctly-rounded
transcendental implementation as the literal normative reference (with a
working Rust binding), or (b) publish its route-(b) LUT/polynomial
coefficients as spec text, the way CIS-1 publishes its integer LUT
constants in-spec — and separately state the receipt's exact byte
encoding as a worked example, not prose.

## Hardware logs

`hardware_logs/e15e_2026-08-28.log` — raw local + CI outputs for both ISA
legs plus the argmax-digest reconstruction.
