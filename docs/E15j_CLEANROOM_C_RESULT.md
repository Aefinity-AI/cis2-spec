# E15j — Clean-room C11 implementation of CIS-2 v0.2 (`verify3/`)

Historical development note (from the reference repository's internal research
log, branch `cm/e15j-cleanroom-c`). No timing numbers; CI-only, no local
model runs.

## What this is

A from-spec-text-only clean-room reimplementation of `docs/CIS2_SPEC_v0.2.md`
in C11 (`verify3/`), following the same clean-room discipline as
`verify/`/`verify2/` but in a different language and with a fresh
implementer. Only `docs/CIS2_SPEC_v0.2.md` and `docs/E15d_v0.2_DIGESTS.md`
were read from this repo (see `verify3/CLEANROOM_LOG.md`). SHA-256 vendored
(public-domain, Brad Conte's `crypto-algorithms`); a hand-written minimal
JSON reader; a hand-rolled byte-level BPE tokenizer (ASCII-sufficient
pre-tokenizer — see `verify3/SPEC_GAPS_v0.2.md`).

## CI run

Workflow `.github/workflows/e15j-cleanroom-c.yml`, dispatched on
`cm/e15j-cleanroom-c`. Run
reference-repository CI run 33197817365 (private CI, not reproducible externally)
(and an earlier path-bug run, 33197529043/33197526034, fixed by commit
`4c499b1`). 4-cell matrix: `{x86_64, aarch64} × {gcc, clang}`, all fetching
the pinned SmolLM2-135M weights (sha256 asserted before use) and gated on
zero FMA-family instructions in the release binary's disassembly (PASS on
all 4 cells).

## Result: PARTIAL conformance — 3 of 4 digests match bit-for-bit, cross-ISA/cross-compiler-identical

All 4 cells (x86_64 gcc, x86_64 clang, aarch64 gcc, aarch64 clang) produced
**byte-identical output** for every field:

| field | expected (spec §13.1) | got (all 4 cells) | match |
|---|---|---|---|
| `table_digest` | `465d358ccd63721256dbd2abbc77ad5de755adf3230635f95b12b1727dfa1ea3` | same | **YES** |
| `inv_freq_table_digest` | `da9f6dcfde0425588815509e874515cdcd3d6b8818b6d0136590052e7bbf6f12` | same | **YES** |
| `argmax_digest` | `0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff` | same | **YES** |
| `generated_token_ids` | `28,665,436,253,1838,8180,3365,14176,30,2306,4161,281,253,2066,2291,351` | same, all 16 | **YES** |
| `CIS2_REF` (witness digest) | `a0c563ef804f50413b7fb6619ae4afe9b51b1ffa7655e944221393e85d6261da` | `a532d4a4a88a15221ea6b6e211daa36aecf6fb64fefa5f85f065252008927d65` | **NO** |

Local unit tests (`verify3/selftest.c`, no model weights needed) already
reproduced `table_digest` and `inv_freq_table_digest` bit-for-bit before
any CI run, and additionally confirmed `ln_pinned(100000.0)` reproduces
the v0.1-era literal `0x413834F1` bit-exactly, and `exp_pinned` matches
glibc `expf` bit-for-bit at every tested point from `-89.0` to `89.0` —
strong evidence the transcendental core (§6) is transcribed correctly.

**Diagnosis**: since `argmax_digest` and `generated_token_ids` match
exactly (this requires all 20 tokens — 4 prompt + 16 greedy picks — to be
identical to the reference), the forward pass is very likely correct or
extremely close; the greedy decode never diverges across 16 sequential
steps. Since `CIS2_REF` additionally folds in the full 49152-entry fp32
logit vector for every one of the 16 steps (§12.1 item 7a), a divergence
there that never flips an argmax winner is consistent with a residual,
small, **unlocalized** numeric deviation somewhere in the forward pass
(most likely a reduction-order or rounding subtlety in RMSNorm, attention
softmax/V-mix, or MLP — all individually re-checked against the spec text
in this implementation and found textually correct) that is invisible to
`argmax_digest` but visible to `CIS2_REF`. This implementer could not
localize the exact divergence without either (a) an independent oracle
(forbidden — no `src/`/`verify/`/`verify2/` access, no local model run)
or (b) a diagnostic side-channel comparing per-layer intermediate state,
which is out of scope for a single clean-room pass.

This is being reported as a **partial-conformance result**, not a pass:
per spec §15 items 1-2, `CIS2_VERIFY3` does **not** yet reproduce every
value in §13.1 bit-for-bit on either ISA. Items 3 (zero FMA), 4
(self-test), and 5 (two-run determinism) all pass.

## Fix attempts

0 of the allotted 3 spec-conformance fix attempts were spent chasing this
specific witness-digest mismatch (one infra-only fix was applied: the
weights file path in the workflow, unrelated to spec conformance). Given
the difficulty of localizing a non-argmax-flipping numeric divergence
without an oracle, and the task's time budget, this is reported as-is for
follow-up rather than spending further blind fix/redispatch cycles.

## Recommendation for a follow-up pass

A future pass could add a `--dump-layer-state <n>` debug hook (informative
tooling only, not spec-normative, matching the precedent of the Rust
reference's `CIS2_DUMP_STEP0_LOGITS`) and manually diff hidden-state
digests layer-by-layer against a second, deliberately-different-order
implementation (e.g. `verify2/`) to localize which operation's reduction
order or rounding differs — without ever reading `verify2/`'s source, only
comparing its printed digest output.
