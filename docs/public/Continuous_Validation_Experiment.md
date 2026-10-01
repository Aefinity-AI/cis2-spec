# Continuous Validation Experiment Outline
**Runnable on existing older hardware**  
Aefinity AI · September 2026

---

## Goal

Demonstrate a minimal continuous-validation loop for bit-identical inference integrity that:

1. Re-uses the existing CIS-2 self-check and witness digests.
2. Detects a controlled integrity failure (weight or environment change).
3. Can be run repeatedly on ordinary older CPUs without a GPU.
4. Produces a clear PASS / FAIL report suitable for an auditor or design partner.

## Experiment design

### Phase 1 — Baseline continuous check (already possible today)

1. Run `./selfcheck.sh` (or `scripts/self_check.sh`) on the current pinned test vector.
2. Record the machine fingerprint, wall-clock time, and final digests.
3. Re-run the same check after a controlled, non-functional change (for example: a different compiler flag that is still expected to match, or a second run on the same machine).
4. Confirm digests remain identical.

**Success criterion:** Repeated runs on the same machine and environment produce matching digests every time.

### Phase 2 — Integrity-fault injection

1. Start from a known-good run and record the `CIS2_REF` witness digest.
2. Introduce one of the following minimal changes (re-use the existing failure-demo patterns from the published head-to-head results):
   - Single-bit flip or small scale change in an embedding row that is not exercised by the short test prompt (see [bitflip head-to-head](../results/2026-09-13-BITFLIP-HEADTOHEAD.md) and [embedding substitution head-to-head](../results/2026-09-17-EMBED-SUBSTITUTION-HEADTOHEAD.md)), or
   - A deliberate change to the floating-point environment or reduction order that the specification forbids.
3. Re-run the exact same verification procedure.
4. Compare side-by-side:
   - Generated tokens (often still identical)
   - Simple text / top-k style check (often still passes)
   - Full CIS-2 witness digest (should fail)

**Success criterion:** Token-level checks can still pass while the full-logit digest fails — demonstrating that continuous monitoring of the witness digest catches a class of integrity failures that text matching misses.

### Phase 3 — Scheduled / repeated monitoring (continuous loop)

1. Use the shipped monitor:
   ```bash
   bash scripts/cis2_monitor.sh
   ```
   Documented in [Continuous_Validation_Monitor_Sketch.md](Continuous_Validation_Monitor_Sketch.md). It:
   - Runs on a schedule or on demand
   - Appends a one-line summary (timestamp, PASS/FAIL, digest short-hash) under `monitor_logs/`
   - Exits non-zero on FAIL so it can be wired into ordinary monitoring
2. Optionally emit a minimal evidence pack for each run using the existing `tools/evidence_pack.py`.
3. Treat any FAIL as an incident that requires investigation (exactly as a production integrity monitor would).

**Success criterion:** The loop can be left running and will surface a deliberate integrity change within one cycle.

## Hardware notes

All phases are designed for the same older equipment already used for CIS-2 self-checks (Intel i5 / Celeron-class CPUs, no GPU required). Wall-clock times on the order of a few minutes per full self-check are expected and acceptable for a continuous-validation demonstration.

## Report template (fill after running)

```markdown
# Continuous Validation Experiment — [Date]

## Summary
A continuous-validation loop based on the CIS-2 self-check and full-logit witness digests was exercised on [machine].
Baseline runs remained bit-identical.
A controlled integrity change was introduced and detected by the witness digest while token-level checks still passed.

## Phase 1 — Baseline
- Commands:
- Digests (short):
- Result: PASS / FAIL

## Phase 2 — Fault injection
- Exact change made:
- Token-level check: PASS / FAIL
- CIS-2 witness digest: PASS / FAIL
- Conclusion: [one sentence]

## Phase 3 — Monitoring loop
- Script / schedule used: scripts/cis2_monitor.sh
- Number of cycles:
- First FAIL detected at:

## Artifacts
All logs, digests, and evidence packs are recorded under [path] (default: monitor_logs/).
```

## Why this increases usability and value

- It turns the existing offline self-check into a demonstrable continuous monitor.
- It re-uses the strongest existing integrity results (bitflip / embedding substitution) instead of inventing new machinery.
- It produces auditor-ready PASS/FAIL evidence that can be shown to design partners or safety researchers.
- It stays within the constraints of a small lab using older equipment.

Once a completed report with real hashes is published, it becomes a concrete public example of continuous validation of bit-identical inference integrity built directly on CIS-2.
