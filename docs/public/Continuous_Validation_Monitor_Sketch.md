# Continuous Validation Monitor
**Minimal wrapper around the existing CIS-2 self-check**  
Aefinity AI · September 2026

---

## Purpose

Turn the existing one-shot `./selfcheck.sh` (or `scripts/self_check.sh`) into a repeatable integrity monitor that:

- Runs on demand or on a schedule
- Logs a short PASS/FAIL summary with timestamp and digest fingerprint
- Exits non-zero on failure so it can be wired into ordinary monitoring or cron
- Stays runnable on the same older hardware already used for CIS-2 self-checks

A runnable implementation is shipped as [`scripts/cis2_monitor.sh`](../../scripts/cis2_monitor.sh). It re-uses the existing public self-check without modifying the normative specification or verifiers.

## Usage

```bash
# One-shot (from repository root)
bash scripts/cis2_monitor.sh

# Or after chmod +x scripts/cis2_monitor.sh
./scripts/cis2_monitor.sh

# Simple daily cron example (adjust path)
# 0 6 * * * /path/to/cis2-spec/scripts/cis2_monitor.sh
```

Logs are written under `monitor_logs/` by default (override with `CIS2_MONITOR_LOG_DIR`). On FAIL the script exits non-zero. Wire it to whatever notification or incident system you already use.

## What it does *not* do

- It does not replace the normative self-check or the clean-room verifiers.
- It does not claim to detect every possible integrity failure; it re-runs the existing public checks.
- It does not require a GPU or special hardware.

## Relation to the continuous-validation experiment

This monitor implements Phase 3 of the experiment outline in [Continuous_Validation_Experiment.md](Continuous_Validation_Experiment.md). Run Phases 1–2 first (baseline + controlled fault injection) so that a FAIL is meaningful when it appears in the monitor log.

## Optional next improvements

- Emit a minimal evidence pack on every run using `tools/evidence_pack.py`.
- Add a machine-fingerprint line to the summary log.
- Keep only the last N logs to bound disk use.

Once a completed continuous-validation experiment report with real hashes is published, this monitor is the operational counterpart that design partners or auditors can adopt with minimal friction.
