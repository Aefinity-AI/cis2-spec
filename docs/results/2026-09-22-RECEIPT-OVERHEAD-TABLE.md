# Receipt overhead table: cost of AEGIS-TRACE receipts on a real agent episode

**Rule A note:** this is a timing/overhead report — no correctness or
"first" claims are made. Every number below names the machine it was
measured on and points to a committed log in the `alice-aegis` repo. No
number is estimated or extrapolated.

## What was measured

One representative agent episode — `scenario1-tool-use` (K=3 steps, N=24
tokens/step budget, a few-shot `CALC` prompt), reused verbatim from
`alice-aegis` PR #109 (rc-2: three agent-trace format-2 episodes), regenerated
against the same pinned `bitnet2B` artifact triple (matches
`.github/workflows/bitnet2b-receipt.yml` `artifacts-bitnet2b-2026-08-27`
pins). The regenerated receipt is byte-identical to rc-2's original in
every field except the folded-in `commit` hash and the resulting trace-chain
digest (rebuilt from a later commit) — the underlying episode content
(prompt, tokens, tool call, tool output) did not change.

- **box1** = `aefinity-box` (Intel i5-5200U, AVX2/FMA).
- **box2** = `aefinity-box2` (Intel Celeron N4020, no AVX2/FMA), reached at
  `192.168.10.21`.

## Table

| Machine | Gen time WITH receipts | Gen time WITHOUT receipts | Receipt size | Verify time |
|---|---|---|---|---|
| box1 (aefinity-box) | 44.74s wall (`agent_trace gen`, K=3 N=24) | 17.84s wall (`cis_decode`, N=72, same prompt) | 1599 bytes | 40.08s wall, PASS (local, direct) |
| box2 (aefinity-box2) | n/a — box2 never runs 2B generation (HOST RULE) | n/a | same file, copied from box1 | ALLOW (PASS), ~12m13s submit-to-ALLOW, polling-bounded (see below) |
| phone | unreachable, not attempted (no `adb` on box1 or box2) | unreachable | unreachable | unreachable |

box2's verify result: per the box2 HOST RULE, box2 must never run
`agent_trace verify` directly against the 2B artifact triple — the only
permitted path is `cm-gateway.service`'s single background verify worker,
reached over a local unix socket with a `RECEIPT`/`POLL` protocol (no
client-side wall clock on the RPC itself, only PENDING/ALLOW/DENY). rc-2
already ran this same scenario shape through the same daemon and observed
ALLOW after ~7.5 minutes of polling. This tick resubmitted the
byte-identical (mod genesis-commit-hash) receipt and observed ALLOW after
~12m13s (submitted 14:43:24Z, ALLOW 14:55:37Z, poll interval 20s) — slower
than rc-2's figure for the same scenario shape; no attempt was made to
control for other daemon-queue activity between the two ticks (the daemon
serializes all verify work through one worker), so both numbers are
reported as observed rather than as a single "the" box2 figure. Full
transcript in `alice-aegis`
`demo/agent-trace/out/rc3/logs/box2-daemon-verify.log` (committed alongside
this doc's source data).

## Exact commands

```sh
# box1, WITH receipts:
agent_trace gen $MODEL $EMBED $VOCAB 3 24 "$PROMPT" > scenario1-tool-use.receipt

# box1, WITHOUT receipts (same prompt, same total token budget K*N=72,
# plain single-pass decode, no multi-step tool loop, no receipt):
cis_decode $MODEL $EMBED $VOCAB 72 "$PROMPT"

# box1, verify:
agent_trace verify $MODEL $EMBED $VOCAB scenario1-tool-use.receipt

# box2, verify (via the daemon, the only box2-HOST-RULE-permitted 2B path):
printf 'RECEIPT /tmp/rc3-scenario1-tool-use.receipt\nACTION \nSESSION cm-rc3-s1\nCOUNTER 1\n\n' \
  | socat - UNIX-CONNECT:$XDG_RUNTIME_DIR/cm-gateway.sock
printf 'POLL ticket=T4\n\n' | socat - UNIX-CONNECT:$XDG_RUNTIME_DIR/cm-gateway.sock
```

Both binaries (`agent_trace`, `cis_decode`) built with
`cargo build --release` from `alice-aegis` `aegis-linux`, worktree
`cm/rc3-overhead-table` off `main` commit `e72427e` (the rc-2 merge). Full
logs, including `/usr/bin/time -v` output (peak RSS: 1,406,336 KB for the
WITH-receipts gen run on box1) and the raw receipt file, are committed in
`alice-aegis` under `demo/agent-trace/out/rc3/`.

## Honest summary

The "with vs. without receipts" gen comparison is for the whole
`scenario1-tool-use` episode as actually run (a K=3-step tool-use loop with
per-step re-prefill), not an isolated ablation of the receipt-writing code
alone — `agent_trace`'s episode loop also re-prefills from scratch at each
step, which the plain single-pass `cis_decode` baseline does not do. So the
observed 2.51x (44.74s vs 17.84s) figure is an upper bound on receipt cost,
not a clean measurement of "the hash chain alone." A closer same-model,
single-step (K=1) isolation was already reported separately (`cost-1`,
2026-09-17, aefinity-box): 31.2% overhead, 946 bytes/receipt, 14.865s
verify. Both figures are given here rather than picking one, because they
measure different things and neither should be read as "the" number for
receipt overhead — it depends on episode shape (step count, re-prefill
count). Receipt size (1599 bytes for this K=3 episode) and box1 verify time
(40.08s, PASS) are direct measurements with no caveat. box2's verify result
is polling-bounded, not a precise wall-clock `time` figure, because the
daemon protocol used (the only box2-HOST-RULE-permitted 2B path) has no
synchronous RPC. Phone replay was not attempted beyond checking for `adb`
(absent on both boxes) — reported as unreachable, not estimated. Energy
draw was not measured this tick (a separate QUEUE addendum item covers
that); no number is fabricated in its place.
