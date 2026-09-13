# Receipt-verified EVAL-60 and tamper detection

Rule A: this is an identity/correctness experiment. No timing figures are
reported or implied anywhere below.

## 1. Question and setup

If every item in an agent-tool-use eval emits a CIS/`agent_trace` receipt,
and every receipt must verify before the item counts toward the score: does
the verifier actually reject tampered receipts, including a receipt that is
internally self-consistent but has been filed under the wrong item? 60-item
suite, template T1 (single tool-call agent turns: calculator, lookup,
mixed), pruned BitNet-2B checkpoint, host `aefinity-box` (box1). Each
item's inference emits an `agent_trace` receipt (trace-chain hash over
genesis + decode steps + tool observation + output tokens); `agent_trace
verify` replays each receipt against the pinned model/embed/vocab artifacts
before the item is scored.
<!-- state/reports/2026-09-13-safe2-verified-evals-box1.md -->

## 2. Result: receipt-verified score

| Metric | Value |
|---|---|
| Receipt verification | 60/60 PASS |
| Score (receipt-verified outputs, arg_match AND output_match) | 45/60 |
<!-- state/reports/2026-09-13-safe2-verified-evals-box1.md -->

No item failed verification here, so this alone does not show a failure —
see the tamper experiments below.

## 3. Tamper detection: v1 (no context binding) vs v2 (context binding)

The 60 verified receipts were copied and exactly one tamper was applied to
each of 4 distinct items, then re-verified with the same `agent_trace
verify` invocation.

| # | Tamper type | Item | v1 result | v1 verifier message | v2 result | v2 verifier message |
|---|---|---|---|---|---|---|
| 1 | Flip one output token | `calc_easy_02` | FAIL | `step 0 divergence: toks-match=false tool-match=true in-match=true out-match=true decode-chain-match=true` / `VERIFY FAIL — replay diverged from the receipt` | FAIL | same |
| 2 | Change one hex byte of tool arg | `calc_easy_03` | FAIL | `step 0 divergence: toks-match=true tool-match=true in-match=false out-match=true decode-chain-match=true` / `VERIFY FAIL — replay diverged from the receipt` | FAIL | same |
| 3 | Truncate chain by one step | `mixed_02` | FAIL | `FAIL structure: receipt claims K=2 but has 1 step lines` | FAIL | same |
| 4 | Replay a valid receipt under a different item's filename | `lookup_hit_01` | **PASS (not caught)** | `receipt trace-chain 13d5237334f306c4` / `local trace-chain 13d5237334f306c4` / `VERIFY PASS — replay reproduced 1 steps and the full trace chain bit-for-bit` (identical to genuine `lookup_hit_02: PASS`) | **FAIL (caught)** | `VERIFY FAIL — ctx mismatch (receipt item-ctx cd6b93c578504e2b vs expected e5aa09bcb711acf9)` |
<!-- state/reports/2026-09-13-safe2-verified-evals-box1.md (v1 rows); state/reports/2026-09-13-safe2c-ctx-binding-box1.md (v2 rows) -->

Overall: v1 = 57/60 PASS, **3/4 tampers caught**. v2 = 56/60 PASS, **4/4
tampers caught**.
<!-- state/reports/2026-09-13-safe2-verified-evals-box1.md, state/reports/2026-09-13-safe2c-ctx-binding-box1.md -->

**The fix (v2, `AEGIS-TRACE` v3):** `item-ctx = sha256(item_id ||
prompt_bytes || session_nonce)` is folded into trace genesis itself,
binding a receipt to the item/session it was generated for. `gen
--item-id/--nonce` sets it; `verify --expect-ctx <hex>` (or
`--expect-item/--expect-prompt-file/--nonce`) fails fast with "ctx
mismatch" if it doesn't match. Retrofit via a pure hash recompute; clean
set re-verifies 60/60 PASS under `--expect-ctx`.
<!-- state/reports/2026-09-13-safe2c-ctx-binding-box1.md -->

## 4. What this does NOT show

- Single host (box1) generated all receipts here — no cross-machine claim.
- Template T1 only, not the full eval suite's other templates.
- The 4 tampers are hand-made (probing token/tool-arg/chain-length/identity
  code paths), not adversarially optimized or randomly sampled.
- No timing/benchmark numbers (Rule A).

## 5. Re-verify
```sh
git clone <alice-aegis-repo-url> alice-aegis && cd alice-aegis
git fetch origin
git worktree add -f ../aa-safe2c origin/cm/safe2c-ctx-binding   # 13ed3c47ae5a
cd ../aa-safe2c
bash eval/receipts/run_verify_v3.sh                 # clean, 60/60 PASS
cat eval/receipts/eval-60-2b-T1-box1-v3-tampered/verify.log  # 4/4 caught
```

Branch history: `cm/safe2-verified-evals` (282bac27759b) →
`cm/safe2b-tamper-demo` (811a8c2b405a) → `cm/safe2c-ctx-binding`
(13ed3c47ae5a), on alice-aegis `main` c4b7fe708fa2. Not yet merged — see
alice-aegis PR "agent-trace: receipt-verified EVAL-60 + tamper demo +
receipt context binding (v2)".
