# Agent-Trace Receipt v2: item/prompt/session binding

Companion to `RECEIPT-GATEWAY.md`. Covers the receipt-format additions
(collectively "v2") that let a verifier reject a receipt that is
internally self-consistent but was generated for, or replayed under, the
wrong item, prompt, or session — a confused-deputy gap the base
agent-trace format does not close on its own.

## Header fields

A v2 receipt keeps the existing genesis/step-line format and adds:

- `table-sha256 <64 hex>` — hash of the tool/allowlist table the episode
  was generated against (pre-existing; table-less v0 episodes omit it).
- `ctx <64 hex>` — folded into the trace genesis, so it cannot be edited
  without breaking the hash chain. Computed as
  `ctx = sha256(item_id || prompt_bytes || session_nonce)`
  (see `compute_item_ctx` in `agent_trace.rs`). `item_id`, `prompt_bytes`,
  and `session_nonce` are the same values the caller already has when it
  requests the episode; nothing new needs to be issued or distributed.
- Existing step lines (`toks=`, `tool=`, `in=`, `out=`, `decode-chain=`)
  are unchanged.

A receipt with no `ctx` line is a pre-v2 (v0/v1) receipt and is verified
under the old rules — `ctx` binding is opt-in per verify invocation via
`--expect-ctx`/`--expect-prompt-hash`, not a format-breaking change.

## Verification order

`agent_trace verify` and the gateway that wraps it check, in this order,
each one short-circuiting the ones after it:

1. **Structure** — hex/length/UTF-8 validation of every field before any
   replay is attempted. Failures print `FAIL structure: <field>` (e.g.
   malformed hex, wrong K/N, non-UTF-8 bytes, duplicate/unknown fields).
2. **Ctx match** — if `--expect-ctx <64hex>` (or `--expect-item ID
   --expect-prompt-file F --nonce N`, which the verifier reduces to the
   same `compute_item_ctx` value) was given, the receipt's `ctx` header
   must match exactly, or verification fails fast with a ctx-mismatch
   message, before replay.
3. **Prompt-hash match** — if `--expect-prompt-hash <64hex>` was given,
   `sha256(prompt)` must match, or the run fails fast with `FAIL
   structure: --expect-prompt-hash mismatch (...)`, again before replay.
4. **Trace-chain replay** — the full sequence of steps (tool calls,
   decode chain, token ids) is independently recomputed and compared
   byte-for-byte to the receipt's own claims.
5. **Hash-chain integrity** — each step's chain hash is recomputed from
   the previous step plus that step's content; any edit anywhere in the
   trace breaks every hash after it.

The gateway (see `RECEIPT-GATEWAY.md`) always passes `--expect-ctx` and
`--expect-prompt-hash` together, derived from the session/item/prompt it
already knows for the decision it is making — never one without the
other, since binding only the prompt but not the item (or vice versa)
would leave a rebinding gap.

## Fuzzing evidence

- Gateway fuzz (mutated inputs to the gateway's own CLI/receipt-reading
  path): 2544 cases across 8 mutation classes (byte flips, truncation,
  non-UTF-8, oversized fields, duplicated step lines, ctx-line edits,
  allowlist-file corruption, malformed nonce/counter). One real panic
  found (a non-UTF-8 CLI argument) and fixed with a regression test; 0
  wrongful ALLOWs after the fix.
- Verifier fuzz (mutated receipt bodies fed straight to `agent_trace
  verify`): 11,601 cases across 10 mutation classes. 0 panics, 0
  wrongful PASS on any mutated receipt. Caveat, stated because it
  matters: 3 of the 10 classes (byte flips, truncated chains, and one
  of five oversized-field buckets) did not individually complete. A
  mutation landing in a receipt field that is not structurally
  validated forces a full model replay before the verifier can reject
  it, and those runs exceeded the harness timeout. Those classes were
  resolved by completing a 40-case random recheck (40/40 clean) and
  extrapolating, not by running every case to completion. The 0-panic
  and 0-wrongful-PASS result for those three classes therefore rests on
  a sample, not on exhaustive completion.

Both fuzz passes only exercise the verifier/gateway's handling of
malformed or randomly mutated input; see limits below for what they do
not show.

## Limits

- Exercised against a single model family (BitNet-2B) on a single
  machine's toolchain; not yet run against other model families or
  independently rebuilt toolchains.
- Fuzzing here means random/structural mutation, not an adversary
  deliberately searching for a bypass — no adversarially-optimized
  tampering has been attempted against this format.
- The verifier that checks these rules has one implementation so far; it
  has not yet been independently re-implemented by a third party from
  the format description alone.
- No timing or performance numbers are claimed anywhere in this note or
  in the underlying work; ctx/prompt-hash checks are a correctness
  property (fail fast vs. fail late), not a performance optimization.
