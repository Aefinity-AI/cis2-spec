# EVAL-60, receipt-verified (2026-09-13)

60-item tool-use eval (T1 template, 2B model, single box), scored strictly
from receipt-verified outputs.
- 60/60 receipts verify PASS (trace-chain reproduces bit-for-bit on replay).
- 45/60 items scored correct against expected tool/arg/output.
- Buckets: calc_easy, calc_hard, calc_overflow, distractor, lookup_hit,
  lookup_miss, lookup_near, mixed.

## Tamper-detection check (separate demo run, not merged)
Four distinct single-field tampers applied to four otherwise-valid receipts
from this batch, then re-verified:

| tamper | result |
|---|---|
| flip one output token id | caught (FAIL) |
| change one hex digit of tool arg | caught (FAIL) |
| truncate the last step, stale trace-chain | caught (FAIL) |
| replay a valid receipt verbatim under a different item's id/filename | **not caught (PASS)** |

**3 of 4 tamper types are caught.** The one that isn't: receipts don't bind
to an item id. A receipt copied verbatim from one item and filed under
another item's filename still verifies PASS. Real, open gap, not fixed yet.

## How to re-verify
alice-aegis `cm/safe2-verified-evals` branch / PR into `main`,
`eval/receipts/eval-60-2b-T1-box1/` (summary.tsv + per-item receipts). Run
the repo's verify tool against those to reproduce 60/60 PASS independently.

## Limits
- No timing/latency numbers published here or in the linked receipts
  (never public per Rule A).
- Single host, single box, T1 template only — not run across boxes,
  templates, or model sizes.
- Known gap: receipts don't bind to item id; wrong-item-id replay is not
  detected by current verification (see table above).
