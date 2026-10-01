# Continuous Validation Experiment — 2026-09-27

**Host:** penguin (ChromeOS Crostini)  
**CPU:** Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz  
**ISA:** avx, avx2, fma, sse4_2  
**OS:** Linux penguin 6.6.147-09642-gea7f90d2e99e x86_64  
**Repository:** Aefinity-AI/cis2-spec @ main (clone of 2026-09-27)

---

## Summary

The CIS-2 public self-check, the weight-pin gates, and `scripts/cis2_monitor.sh` were run by hand on one ordinary older host (no GPU). No scheduler (cron, systemd timer) was used.

- **Two baseline self-checks** were started by hand about 6.5 minutes apart on one host; both printed the same pinned digests.
- **One byte in one file** (`model.safetensors`) was changed on purpose. The verifier's sha256 pin check caught it (`CIS2_VERIFY3 FATAL: model.safetensors sha256 mismatch`) before any forward pass ran.
- **The monitor script was run twice by hand** on the restored clean files and logged PASS both times.

Scope limits: this is evidence that the sha256 pin gate works. It is not new evidence about logit or witness digests, because the corrupted run never reached the forward pass. Corruption of `config.json` or `tokenizer.json`, tampered pins or `EXPECTED_DIGESTS.md`, and a tampered verifier binary were not tested. The monitor script itself was never run against corrupted weights (see Phase 3).

---

## Phase 1 — Baseline (repeated self-check)

### Commands

```bash
git clone https://github.com/Aefinity-AI/cis2-spec && cd cis2-spec
./selfcheck.sh          # run 1 — 2026-09-27T08:41:34Z
./selfcheck.sh          # run 2 — 2026-09-27T08:48:04Z
```

### Results

| Run | UTC time | Summary | CIS2_VERIFY3 digest |
|-----|----------|---------|---------------------|
| 1 | 08:41:34 | 10 passed, 0 failed | `d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df` |
| 2 | 08:48:04 | 10 passed, 0 failed | `d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df` |

Additional pinned digests (both runs):

| Digest | Value |
|--------|--------|
| argmax_digest | `0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff` |
| table_digest | `23c7bfaf5cef0095fd021af2eb1808abb4928bae4219756d86bdac670a06b35d` |
| inv_freq_table_digest | `da9f6dcfde0425588815509e874515cdcd3d6b8818b6d0136590052e7bbf6f12` |
| generated_token_ids | `28,665,436,253,1838,8180,3365,14176,30,2306,4161,281,253,2066,2291,351` |
| model.safetensors sha256 | `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1` |

**Phase 1 result: PASS** — two hand-started runs, about 6.5 minutes apart on one host, produced matching digests (n=2, same host, same day).

---

## Phase 2 — Integrity-fault injection

### Procedure

```bash
cp weights/model.safetensors weights/model.safetensors.bak
printf '\xff' | dd of=weights/model.safetensors bs=1 seek=1000000 conv=notrunc status=none
sha256sum weights/model.safetensors
# → f262f76b67887396a7c0de5e24d44eb4c72ec9fbcf6dbd3b978a1d40d87a7493

# Direct verifier only (no fetch path)
./verify3/cis2_verify3 \
  weights/model.safetensors \
  weights/config.json \
  weights/tokenizer.json
```

### Results

```text
CIS2_VERIFY3 FATAL: model.safetensors sha256 mismatch,
  got=f262f76b67887396a7c0de5e24d44eb4c72ec9fbcf6dbd3b978a1d40d87a7493
CIS2_VERIFY3 selftest=PASS ftz_daz=pinned
```

The verifier stopped on a weight file that did not match the pinned artifact hash, before the forward pass. (The `selftest=PASS` line is the verifier's own startup self-test, printed after the FATAL line; it is not a verdict on the weights.)

**Related observation (fetch path):** When `./selfcheck.sh` was run against the same corrupted file earlier, the fetch step detected the pin mismatch, re-downloaded the correct artifact, and the subsequent check PASSed on clean weights. Because of this, the self-check (and therefore the monitor, which wraps it) repairs corrupted weights silently and would likely log PASS, masking the corruption in monitor mode. Both layers are integrity controls, but they behave differently:

| Gate | Behavior under one-byte corruption |
|------|-------------------------------------|
| `scripts/fetch_weights.sh` / selfcheck fetch | Re-fetch pinned weights; self-check continues on good file |
| `verify3` artifact hash at load | **FATAL** — no forward pass |

**Phase 2 result: Detected (run stopped with a hash mismatch)** — the one-byte change in `model.safetensors` was caught by the sha256 pin before any logits were produced.

*(For evidence that full-logit witness digests catch alterations that leave generated text unchanged when corrupted weights are intentionally allowed through, see the earlier published head-to-heads: [bitflip](2026-09-13-BITFLIP-HEADTOHEAD.md), [embedding substitution](2026-09-17-EMBED-SUBSTITUTION-HEADTOHEAD.md).)*

---

## Phase 3 — Continuous monitor loop

### Restore and monitor

```bash
cp weights/model.safetensors.bak weights/model.safetensors
sha256sum weights/model.safetensors
# → 80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1

./selfcheck.sh    # SUMMARY: 10 passed, 0 failed
bash scripts/cis2_monitor.sh
cat monitor_logs/cis2_monitor_summary.log
```

### Monitor summary log

```text
20260927T084928Z  PASS  digest_hint=6ebb67358c6bbab8  log=.../cis2_monitor_20260927T084928Z.log
20260927T090007Z  PASS  digest_hint=6ebb67358c6bbab8  log=.../cis2_monitor_20260927T090007Z.log
```

**Phase 3 result: PASS** — two monitor runs, started by hand about 11 minutes apart, logged PASS on the restored clean stack and exited zero.

Notes on what this does and does not show:

- The monitor script was not run against corrupted weights, so its FAIL path was not exercised in this experiment. Because the fetch step re-downloads a file whose hash does not match, a monitor run on corrupted weights would likely repair them and report PASS.
- `digest_hint=6ebb67358c6bbab8` in the log above is not a model or output digest. It is the suffix of the cargo test binary name (`conformance_rope-6ebb67358c6bbab8`), which the script's old "last hex string in the log" rule picked up. The script now parses the witness digest line printed by the self-check and prints `digest_hint=unknown` if it finds none.
- PASS means the self-check printed its own all-digests-match line, compared against this repository's `EXPECTED_DIGESTS.md`. It is a consistency check, not an independent one.

---

## Conclusions

1. **Baseline digests matched** across two hand-started runs on this host under the public CIS-2 self-check.
2. **The tooling ran on older hardware** without a GPU: self-check + `scripts/cis2_monitor.sh`.
3. **The sha256 pin gates fired on one one-byte change in `model.safetensors`:** both the fetch pin path and the verifier load-time artifact hash detected it. Other files and attack surfaces were not tested.
4. **Not yet shown:** scheduled operation, multi-host or long-horizon data, and monitor behavior on corrupted input. Wiring the monitor to cron and disabling the fetch self-repair in monitor mode would be needed before treating FAIL as an incident signal.

This is a small, hand-run example of the hash gate and self-check working together, not a demonstration of an operational continuous-validation deployment.

---

## Re-check 2026-10-01

An independent re-run on a fresh clone at commit 205b024 reproduced the monitor PASS and the corruption FATAL (sha256 mismatch, got=f262f76b...7493, run stopped before any digest line). All printed digests (table, inv_freq, argmax, witness, generated token ids, model.safetensors sha256) matched `EXPECTED_DIGESTS.md`. The 09-27 run times and the second baseline run could not be re-checked.

---

## Artifacts retained on the experiment host

- `phase2_corrupt_verify3.log` — FATAL sha256 mismatch
- `monitor_logs/cis2_monitor_*.log` — full monitor runs
- `monitor_logs/cis2_monitor_summary.log` — one-line PASS/FAIL history

## Related public materials

- [Continuous Validation Experiment outline](../public/Continuous_Validation_Experiment.md)
- [Continuous Validation Monitor](../public/Continuous_Validation_Monitor_Sketch.md) / `scripts/cis2_monitor.sh`
- [CIS-2 and AI Safety](../public/CIS2_and_AI_Safety.md)
- [Design Partner Adoption Path](../public/Design_Partner_Adoption_Path.md)
