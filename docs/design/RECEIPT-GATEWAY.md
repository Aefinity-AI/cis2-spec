# Receipt gateway: a "no receipt, no action" tool gate

Design note. Status: design + a working end-to-end prototype (box1), independently re-verified on a second device, and separately demonstrated split across two machines over a forced-command SSH relay (§5a) — with a round trip using a real 2B-parameter model receipt, not only the small/tiny-model receipts used in earlier demonstrations. No wall-clock or performance numbers appear below; none are claimed.

The gateway is a small service sitting between an autonomous agent and the tools it calls (shell, filesystem, external APIs). Its rule: a tool call is executed only if it arrives with a receipt that verifies — an AEGIS-TRACE receipt covering every step up to and including that call, replayed and checked before the call is let through.

## 1. Threat model

Five cases: (1) a **compromised agent fabricates a plausible tool call** whose receipt doesn't actually cover it — caught by receipt-chain replay, since any dropped step or changed input/output/context fails the chain; (2) **tampered model weights** — a receipt that replays perfectly but against a different, unapproved model — caught by the weight-digest allowlist, independent of whether the chain itself verifies; (3) a **replayed receipt** — an old, valid receipt resubmitted to authorize a new or different action — caught by gateway-side freshness tracking, since `verify` alone has no built-in notion of reuse; (4) a **fabricated tool argument** not grounded in anything the model was actually shown — surfaced as a WARNING during replay, and a receipt whose WARNING lines don't match the independently re-derived set is a hard verify failure; and (5) a **rogue process that bypasses the gateway entirely** and talks to the tool directly — not caught (§3).

## 2. The four enforced rules

**(a) Verify-to-execute binding.** A verifying receipt is not enough; the gateway asserts byte-equality between the canonical bytes of the action it is about to dispatch and the receipt's own final-step tool-input bytes. Without this, any valid receipt — even one from an unrelated, benign episode — could authorize a different action; a single flipped bit in the dispatched bytes after verification trips this check.

**(b) Freshness (nonce/replay).** Each decision is keyed to a (session, counter, action-hash) tuple; resubmitting an already-logged tuple is denied — closing the gap plain chain-replay leaves open, replaying the same valid receipt to re-trigger the same action.

**(c) Signed allowlist, including a weights digest.** The receipt's header commits sha256 digests of the model, embedding, and vocab artifacts; the gateway holds a static, operator-approved allowlist of these triples, and a receipt whose artifact hashes are not on it is denied independent of whether it verifies — this is what stops a trace chain that replays perfectly against a different, unapproved model.

**(d) Anchored, hash-chained decision log.** The gateway emits one decision per proposed call and appends it to its own append-only, hash-chained log — chaining the decision log itself, not just the agent's receipts, so an attacker who compromises the gateway after the fact cannot quietly rewrite history. Log heads are committed externally (git) at a fixed cadence so tampering is itself detectable.

## 3. What the gateway does NOT stop

A **rogue process that bypasses the gateway entirely** is not caught — no mechanism here addresses it. Stopping it needs OS-level enforcement: the tool's entry point reachable only through the gateway's own process/credentials (socket-only tool daemon, seccomp/namespace confinement, a reverse proxy with no other network path). A receipt-gateway is a userspace policy layer in front of a decision; if anything can reach the tool without going through it, the gate is decorative for that path — least-privilege process isolation is separate infrastructure, not something to bolt onto the gateway.

A **wrong-but-well-formed model output** is also not caught: the gateway gates on receipt validity and verify-execute binding, not output correctness. A model legitimately tricked into deciding to call a dangerous tool produces a receipt that verifies cleanly — the chain proves the call happened as claimed, not that calling it was a good idea. A model that fabricates an argument and then honestly, self-consistently reports on it does not trip the WARNING mechanism (self-generated text is excluded from the grounding check by design) — verified is not the same claim as safe.

## 4. End-to-end decision table (one gateway process, 25 decisions)

Twenty live-episode receipts plus four constructed cases exercising each required deny scenario, all against one long-lived gateway process (freshness state persists across calls, as in a real deployment). **20 ALLOW**, no false denies: `CALC`-only, `FILE-READ`-only, `LOOKUP`-only, chained `LOOKUP`→`FILE-READ`, chained `LOOKUP`→`LOOKUP`, and mixed `LOOKUP`+`CALC` episodes — the full live-episode corpus, lenient policy. **4 DENY**, one per required scenario: (1) **replay** — resubmitting an already-allowed episode's exact receipt/action under the same session and counter, message `freshness: (session, counter, action-hash) already seen`; (2) **ungrounded argument under strict grounding** — a known ungrounded tool-call argument verified against a strict-mode gateway, denied on verify failure/strict-grounding warning; (3) **weights digest not on allowlist** — allowlist signed with a one-bit-flipped model-artifact hash while the receipt declares the real, correct hash, mismatch, message `artifact triple not on allowlist`; (4) **tampered forwarded bytes** — one bit flipped in the action bytes actually dispatched, after verification, versus what the receipt's own last step says happened, message `verify-execute binding failed: action hash != receipt final-step hash`.

All four required deny scenarios were caught correctly; zero false allows and zero false denies on the twenty legitimate episodes.

## 5. Independent re-verification (a second device)

Because the gateway's own decision log is a single point of failure, a second, independent verifier periodically re-checks a sample of ALLOW decisions rather than trusting the gateway's own verdict: it recomputes a committed decision-log chain head by hand from a receipt's declared model/embed/vocab/trace-chain fields and confirms it matches the head the gateway externally committed (hex-identical match confirmed); and it independently re-runs the verifier — built for a different CPU architecture than the one that produced the receipts — against a sample of ALLOWed receipts, using its own copy of the reference artifacts (sha256-confirmed against pinned values, not supplied by the gateway). All sampled receipts came back VERIFY PASS with trace-chain hex matching the receipt's own declared value; one carried the same grounding WARNING already known for that episode (pass-with-warning, not a failure). An ALLOW that an independent replay says should have been DENY is treated as evidence the gateway itself is compromised, not logged and moved past.

## 5a. Two-box enforcement demonstration (relay-split gateway)

Beyond running the gateway and its independent verifier as separate processes on one machine (§5), the split was also demonstrated across two physically separate machines: the agent issuing requests from box1, and the gateway/verifier running on box2, communicating only through a forced-command SSH relay (a raw byte pipe onto the gateway's own UNIX-socket line protocol — no separate relay-side protocol logic to keep in sync with the gateway). This round trip has since been repeated with a real 2B-parameter model's receipt (previously only small/tiny-model receipts had been used for the two-box case), closing that gap. The receipt was a single, non-chained step (no `table-sha256` dependency), submitted asynchronously against the gateway's ticket-based queue. Verbatim sequence (elided to the required lines):

```
$ printf 'MODE GATEWAY\nRECEIPT %s\nACTION %s\nSESSION %s\nCOUNTER %s\n\n' "$RECEIPT" "$ACTION" "$SESSION" "$COUNTER" | ssh ... cm@<box2>
PENDING ticket=T1
rc=0

$ printf 'MODE GATEWAY\nPOLL ticket=T1\n\n' | ssh ... cm@<box2>
PENDING
rc=0
[... further PENDING polls while the gateway's background worker verifies ...]

$ printf 'MODE GATEWAY\nPOLL ticket=T1\n\n' | ssh ... cm@<box2>
ALLOW idx=3 cap=8ece5f6cb84d301ef2594ea2271591e2d5aa9d8bfc7cb8ebcd66e1aeb7e66f09 exp=1789413648
rc=0
```

Resubmitting the identical `(session, counter, action)` afterward correctly triggers the freshness deny (§2b), demonstrating that replay protection holds across the relay, not just within a single process:

```
$ printf 'MODE GATEWAY\nRECEIPT %s\nACTION %s\nSESSION %s\nCOUNTER %s\n\n' "$RECEIPT" "$ACTION" "$SESSION" "$COUNTER" | ssh ... cm@<box2>
DENY freshness: (session, counter, action-hash) already seen
rc=0
```

All three lines — `PENDING ticket=T1`, the `ALLOW ...` line, and the `DENY freshness: ...` line — were captured verbatim, none fabricated. No relay-side change was needed for the ticket-based PENDING/POLL exchange: because the relay is a raw byte pipe onto the gateway's socket, it passes the gateway's own async protocol through untouched.

## 6. How to run it

1. Approve a model/embed/vocab artifact triple out of band; add its sha256 digests to the gateway's allowlist file.
2. Start one long-lived gateway process pointed at that allowlist and a fresh decision-log path.
3. Per proposed tool call, submit the receipt covering every step up to and including that call, plus the canonical bytes of the action about to be dispatched; the gateway returns ALLOW or DENY with a reason.
4. Periodically anchor the decision-log head externally (e.g. commit it to a repository) and have an independent process re-verify a sample of ALLOWed decisions against its own copy of the reference artifacts and allowlist.

## 7. Related work

Gating a tool call on bit-exact, independently replayable inference verification is uncommon: most "verifiable inference" work trades bit-exactness away for speed (locality-sensitive hashing over activations, fraud-proof/challenge-period economics), and the one closed-vendor approach reaching a comparable bit-exact tier (A2Auth) does so without verify-execute binding or a weight allowlist.

Closest work, by property: **A2Auth** matches on verification depth — same bit-exact tier — but names a single closed vendor with no public spec, and has no verify-execute binding, weight allowlist, or gateway architecture of its own. **Notarized Agents** matches on log-integrity, solving "the agent lies about its own trace" via a receiver that signs what it observed into a cosigned Merkle log, where this design instead uses independent second-verifier replay of the agent's own receipts; neither combines receiver-side notarization with bit-exact replay. **MCP security guidance** (Anthropic/OWASP) matches on naming the vulnerability — the confused deputy, tool-result content with no attestable provenance — but its mitigation is authorization and session-scoping, itself described as advisory rather than a security boundary, not cryptographic binding of forwarded bytes to a verified inference receipt.

Two claims this design does not make: it does not claim to be the sole or original receipt-gated agent tool gateway (signed/hash-chained agent receipts are a crowded space); and passing these checks does not mean a call was a good idea or the model is trustworthy — only that its history is checkable and tied to an approved weight triple (§3).
