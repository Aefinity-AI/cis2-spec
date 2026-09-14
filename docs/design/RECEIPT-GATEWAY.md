# Receipt gateway: a "no receipt, no action" tool gate

Design note. Status: design + a working end-to-end prototype (box1), independently re-verified on a second device. No wall-clock or performance numbers appear below; none are claimed.

The gateway is a small service sitting between an autonomous agent and the tools it calls (shell, filesystem, external APIs). Its rule: a tool call is executed only if it arrives with a receipt that verifies — an AEGIS-TRACE receipt covering every step up to and including that call, replayed and checked before the call is let through.

## 1. Threat model

Five cases: (1) a **compromised agent fabricates a plausible tool call** whose receipt doesn't actually cover it — caught by receipt-chain replay, since any dropped step or changed input/output/context fails the chain; (2) **tampered model weights** — a receipt that replays perfectly but against a different, unapproved model — caught by the weight-digest allowlist, independent of whether the chain itself verifies; (3) a **replayed receipt** — an old, valid receipt resubmitted to authorize a new or different action — caught by gateway-side freshness tracking, since `verify` alone has no built-in notion of reuse; (4) a **fabricated tool argument** not grounded in anything the model was actually shown — surfaced as a WARNING during replay, and a receipt whose WARNING lines don't match the independently re-derived set is a hard verify failure; and (5) a **rogue process that bypasses the gateway entirely** and talks to the tool directly — not caught (§3).

## 2. The four enforced rules

**(a) Verify-to-execute binding.** A verifying receipt is not enough; the gateway asserts byte-equality between the canonical bytes of the action it is about to dispatch and the receipt's own final-step tool-input bytes. Without this, any valid receipt — even one from an unrelated, benign episode — could authorize a different action; a single flipped bit in the dispatched bytes after verification trips this check.

**(b) Freshness (nonce/replay).** Each decision is keyed to a (session, counter, action-hash) tuple; resubmitting an already-logged tuple is denied — closing the gap plain chain-replay leaves open, replaying the same valid receipt to re-trigger the same action.

**(c) Signed allowlist, including a weights digest.** The receipt's header commits sha256 digests of the model, embedding, and vocab artifacts; the gateway holds a static, operator-approved allowlist of these triples, and a receipt whose artifact hashes are not on it is denied independent of whether it verifies — this is what stops a trace chain that replays perfectly against a different, unapproved model.

**(d) Anchored, hash-chained decision log.** The gateway emits one decision per proposed call and appends it to its own append-only, hash-chained log — chaining the decision log itself, not just the agent's receipts, so an attacker who compromises the gateway after the fact cannot quietly rewrite history. Log heads are committed externally (git) at a fixed cadence so tampering is itself detectable.

## 3. What the gateway does NOT stop

- A **compromised gateway host** — an attacker who controls the gateway process itself can allow anything and rewrite or suppress its own decision log; nothing in §2 stops this by itself, which is why §5's independent second-verifier replay exists as a separate, out-of-band check rather than a property of the gateway.
A **rogue process that bypasses the gateway entirely** is not caught — no mechanism here addresses it. Stopping it needs OS-level enforcement: the tool's entry point reachable only through the gateway's own process/credentials (socket-only tool daemon, seccomp/namespace confinement, a reverse proxy with no other network path). A receipt-gateway is a userspace policy layer in front of a decision; if anything can reach the tool without going through it, the gate is decorative for that path — least-privilege process isolation is separate infrastructure, not something to bolt onto the gateway.

A **wrong-but-well-formed model output** is also not caught: the gateway gates on receipt validity and verify-execute binding, not output correctness. A model legitimately tricked into deciding to call a dangerous tool produces a receipt that verifies cleanly — the chain proves the call happened as claimed, not that calling it was a good idea. A model that fabricates an argument and then honestly, self-consistently reports on it does not trip the WARNING mechanism (self-generated text is excluded from the grounding check by design) — verified is not the same claim as safe.

## 4. End-to-end decision table (one gateway process, 25 decisions)

Twenty live-episode receipts plus four constructed cases exercising each required deny scenario, all against one long-lived gateway process (freshness state persists across calls, as in a real deployment). **20 ALLOW**, no false denies: `CALC`-only, `FILE-READ`-only, `LOOKUP`-only, chained `LOOKUP`→`FILE-READ`, chained `LOOKUP`→`LOOKUP`, and mixed `LOOKUP`+`CALC` episodes — the full live-episode corpus, lenient policy. **4 DENY**, one per required scenario: (1) **replay** — resubmitting an already-allowed episode's exact receipt/action under the same session and counter, message `freshness: (session, counter, action-hash) already seen`; (2) **ungrounded argument under strict grounding** — a known ungrounded tool-call argument verified against a strict-mode gateway, message `agent_trace verify: VERIFY FAIL or strict-grounding WARNING present`; (3) **weights digest not on allowlist** — allowlist signed with a one-bit-flipped model-artifact hash while the receipt declares the real, correct hash, mismatch, message `artifact triple not on allowlist`; (4) **tampered forwarded bytes** — one bit flipped in the action bytes actually dispatched, after verification, versus what the receipt's own last step says happened, message `verify-execute binding failed: action hash != receipt final-step hash`.

All four required deny scenarios were caught correctly; zero false allows and zero false denies on the twenty legitimate episodes.

## 5. Independent re-verification (a second device)

Because the gateway's own decision log is a single point of failure, a second, independent verifier periodically re-checks a sample of ALLOW decisions rather than trusting the gateway's own verdict: it recomputes a committed decision-log chain head by hand from a receipt's declared model/embed/vocab/trace-chain fields and confirms it matches the head the gateway externally committed (hex-identical match confirmed); and it independently re-runs the verifier — built for a different CPU architecture than the one that produced the receipts — against a sample of ALLOWed receipts, using its own copy of the reference artifacts (sha256-confirmed against pinned values, not supplied by the gateway). All sampled receipts came back VERIFY PASS with trace-chain hex matching the receipt's own declared value; one carried the same grounding WARNING already known for that episode (pass-with-warning, not a failure). An ALLOW that an independent replay says should have been DENY is treated as evidence the gateway itself is compromised, not logged and moved past.

## 6. How to run it

1. Approve a model/embed/vocab artifact triple out of band; add its sha256 digests to the gateway's allowlist file.
2. Start one long-lived gateway process pointed at that allowlist and a fresh decision-log path.
3. Per proposed tool call, submit the receipt covering every step up to and including that call, plus the canonical bytes of the action about to be dispatched; the gateway returns ALLOW or DENY with a reason.
4. Periodically anchor the decision-log head externally (e.g. commit it to a repository) and have an independent process re-verify a sample of ALLOWed decisions against its own copy of the reference artifacts and allowlist.

## 7. Related work

Gating a tool call on bit-exact, independently replayable inference verification is uncommon: most "verifiable inference" work trades bit-exactness away for speed (locality-sensitive hashing over activations, fraud-proof/challenge-period economics), and the one closed-vendor approach reaching a comparable bit-exact tier (A2Auth) does so without verify-execute binding or a weight allowlist.

Closest work, by property: **A2Auth** matches on verification depth — same bit-exact tier — but names a single closed vendor with no public spec, and has no verify-execute binding, weight allowlist, or gateway architecture of its own. **Notarized Agents** matches on log-integrity, solving "the agent lies about its own trace" via a receiver that signs what it observed into a cosigned Merkle log, where this design instead uses independent second-verifier replay of the agent's own receipts; neither combines receiver-side notarization with bit-exact replay. **MCP security guidance** (Anthropic/OWASP) matches on naming the vulnerability — the confused deputy, tool-result content with no attestable provenance — but its mitigation is authorization and session-scoping, itself described as advisory rather than a security boundary, not cryptographic binding of forwarded bytes to a verified inference receipt.

Two claims this design does not make: it does not claim to be the sole or original receipt-gated agent tool gateway (signed/hash-chained agent receipts are a crowded space); and passing these checks does not mean a call was a good idea or the model is trustworthy — only that its history is checkable and tied to an approved weight triple (§3).

## 8. Enforcement boundary

The receipt-gateway logic above is a userspace decision layer; §3 already
concedes that a rogue process bypassing the gateway entirely is not
caught by receipt verification alone. This section describes the
complementary enforcement layer that closes that gap on a real
deployment: OS-level confinement of the agent, tool-side capability
tokens, and a daemon whose only network surface is a local socket — plus
a set of escape tests exercised against a live install, including a
two-box configuration where the agent and the gateway's signing keys are
split across separate hosts.

**Agent sandbox.** The agent runs inside a systemd service unit with
`RestrictAddressFamilies=AF_UNIX` and `PrivateNetwork=yes`, so any
attempt to open an AF_INET socket is refused at the kernel level before
it can reach the network at all. Where AppArmor is installed and its
enforcing profile is loaded, exec of anything other than the approved
tool-shim binaries (and, for the shell shim, a narrowly scoped
sub-profile reachable only after that shim's own capability check has
already passed) is denied by the LSM. Together these mean the sandboxed
agent process has no raw network path and no way to invoke an arbitrary
shell — every tool action must go through a shim.

**Capability tokens.** On every ALLOW decision, the gateway mints a
single-use, expiring, HMAC-signed capability token bound to the specific
action it just approved (an index the gateway assigns, the hash of the
approved action bytes, and an expiry). Each tool shim independently
recomputes the action hash from the bytes it is actually about to
execute, verifies the token's signature and expiry, and checks its own
persisted record of already-consumed indices before doing anything —
so a captured but already-used token, a forged token, or a token for a
different action all fail closed at the shim, independent of whether the
gateway itself is ever asked again.

**Gateway daemon.** The gateway runs as a long-lived daemon reachable
only over a unix domain socket — it exposes no TCP listener, so there is
no network-reachable surface to probe for the decision service itself.
Every proposed action is checked against a submitted agent-trace receipt
(verify-execute binding, freshness, and the signed artifact allowlist, as
in §2) before a capability token is ever issued; a request that fails any
of those checks gets a DENY and no token.

### Escape tests

Five probes were run against live installs to confirm the above holds in
practice, not just on paper.

- **Network escape.** Attempting to open a raw network socket from
  inside the sandboxed agent unit fails immediately with `EAFNOSUPPORT`
  — confirmed at the syscall level, not merely by an error return code
  from a wrapper.
- **Shell exec.** Attempting to exec an arbitrary shell binary from
  inside the sandboxed unit, with the AppArmor profile enforcing, is
  denied: `/bin/bash: Permission denied`, exit 126 — while the profile's
  own explicitly allowed shell entry point still runs, showing the
  denial is the mandatory-access-control policy and not a missing
  binary or a broken environment.
- **Forged or replayed-signature capability token.** Presenting a tool
  shim with no token, or with a token whose signature does not verify,
  is refused: `REFUSE: no capability token / signature invalid`. Nothing
  executes.
- **Reused capability token.** Presenting a genuinely valid, single-use
  token a second time — after it has already authorized one action — is
  refused: `REFUSE: capability token already consumed`. The token works
  once and never again.
- **Two-box split.** In a configuration where the agent runs on one host
  ("box A") and the gateway daemon plus its signing keys live on a
  separate host ("box B"), reachable only through a forced-command SSH
  channel, five separate exfiltration and bypass attempts from box A —
  copying the key files off box B, opening a local port-forward through
  the channel, connecting directly to any TCP port on box B (the gateway
  listens only on a unix socket, so there is no such port to reach),
  forcing an interactive shell or pty over the channel, and authenticating
  with an unrelated key not present on box B — were all refused. The
  legitimate protocol, by contrast, worked exactly as designed: a
  correctly signed request over the same channel produced a real ALLOW
  with a minted capability token, and resubmitting that identical request
  produced a freshness DENY on replay. The two-box configuration proves
  the split holds end to end, not just for the illegitimate paths.

### Residual limits

This layer does not defend against everything, and the design note would
be dishonest if it implied otherwise.

- **A compromised gateway host is not defended against.** In a
  single-box deployment the gateway process holds both signing keys in
  plaintext at runtime; root or process compromise there can forge any
  token it wants. The two-box configuration reduces this by moving the
  keys off the host where agent-controlled content lives, but a
  compromise of the box that does hold the keys is still a full break —
  there is no independent, real-time detection of that case.
- **Tools reachable outside the shims are out of scope.** The enforcement
  described here only applies to actions that go through a wrapped tool
  shim. Any other way of reaching a resource — direct OS access, a
  binary not covered by the AppArmor profile, a service left listening
  outside this design — is not mediated by any of this and must be
  treated as a separate, unaddressed surface.
- **No hardware-rooted attestation.** There is no vendor-rooted hardware
  attestation (no TPM-anchored chain) anywhere in this fleet. The trust
  root is the gateway process plus its AppArmor/systemd policy — software
  and LSM mediation on top of the existing kernel and root account — not
  silicon. Root on either host can, in principle, unload the enforcing
  profile, edit the unit, or read key material directly; nothing here is
  a hypervisor- or TPM-rooted boundary.

## Residual limits

- **Verification scale on the enforcement host.** The 2B-model receipt path (20 live episodes, four deny classes) was proven on the single-box gateway on the stronger machine. The two-box demonstration above used a small model on the weaker enforcement host, because a full 2B replay there takes minutes and the daemon serves one request at a time. A production enforcement host needs comparable compute to the agent host, or asynchronous verification.
