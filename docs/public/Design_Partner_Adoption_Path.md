# Design Partner Adoption Path
**From first self-check to continuous integrity monitoring**  
Aefinity AI · September 2026

---

## Who this is for

Organizations that need checkable provenance for high-stakes or regulated AI decisions, safety researchers building reproducible evaluation infrastructure, and teams evaluating receipt-gated or verifiable inference at the edge.

## Path in four steps

### Step 1 — Verify the claim yourself (15 minutes)

```
git clone https://github.com/Aefinity-AI/cis2-spec && cd cis2-spec && ./selfcheck.sh
```

A PASS means an independent machine produced bit-identical results under the CIS-2 specification. No GPU, no account, and no special hardware are required.

See also: [Why Bit-Identical Inference Matters for Trust](Why_Bit_Identical_Inference_Matters_for_Trust.md)

### Step 2 — Understand the integrity gap that token-level checks miss

Review the published head-to-head results:

- [Single-weight corruption](../results/2026-09-13-BITFLIP-HEADTOHEAD.md) — a flipped weight bit can leave generated text unchanged while breaking the full-logit witness digest.
- [Embedding substitution](../results/2026-09-17-EMBED-SUBSTITUTION-HEADTOHEAD.md) — only the full-logit digest catches every tested alteration, including unexercised rows.

These demonstrate why continuous monitoring of witness digests adds assurance beyond matching final text.

### Step 3 — Run the continuous-validation experiment (small-lab scale)

Follow the three-phase outline in [Continuous_Validation_Experiment.md](Continuous_Validation_Experiment.md), or review the **completed run** already published:

- **[2026-09-27 Continuous Validation on penguin (i5-10210U)](../results/2026-09-27-CONTINUOUS-VALIDATION.md)** — baseline bit-identity, one-byte weight corruption detected by the verifier artifact-hash gate, monitor PASS after restore.

Tooling for your own runs:

1. Baseline repeated self-checks (`./selfcheck.sh`)
2. Controlled integrity-fault injection (direct `verify3` on a corrupted weight file)
3. Scheduled monitoring with [`scripts/cis2_monitor.sh`](../../scripts/cis2_monitor.sh)

All phases are designed for ordinary older CPUs.

### Step 4 — Adopt continuous monitoring and receipts

- Run `bash scripts/cis2_monitor.sh` on a schedule (or on demand) and treat FAIL as an incident. Logs go under `monitor_logs/` by default.
- For agentic or multi-step workflows, combine CIS-2 witness digests with the receipt / agent-trace tooling in the sibling [alice-aegis](https://github.com/Aefinity-AI/alice-aegis) repository.
- For external scrutiny, use the paths in [How Outsiders Can Help](How_Outsiders_Can_Help.md) (reproduction reports, divergence reports, clean-room implementations).
- For safety-case language, see [Safety Case Contribution](Safety_Case_Contribution.md) and [CIS-2 and AI Safety](CIS2_and_AI_Safety.md).

## What you get

- An independently checkable numerical execution layer
- A concrete way to detect silent changes that text-matching can miss
- A path from one-shot verification to continuous integrity monitoring
- A published end-to-end continuous-validation experiment on real older hardware
- No requirement for GPUs or proprietary stacks to perform the core checks

## What you do not get

CIS-2 does not solve alignment, jailbreaks, or hallucination. It makes the floating-point computation itself continuously checkable so that those harder questions can be asked about a fixed, agreed-upon artifact.

## Next conversation

Contact: https://aefinity-ai.github.io/pilot.html (or a GitHub issue on Aefinity-AI/cis2-spec)  
Company site: https://aefinity-ai.github.io/  
Full public materials: https://github.com/Aefinity-AI/cis2-spec/tree/main/docs/public

Aefinity AI Inc. · Orange, Texas
