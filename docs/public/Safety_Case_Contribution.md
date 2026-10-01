# CIS-2 as Safety-Case Evidence
**What computational integrity can contribute to an AI safety case**  
Aefinity AI · September 2026

---

## Purpose

A safety case argues that a system is acceptably safe for a defined use. Typical evidence includes evaluations, mitigations, monitoring plans, and accountability mechanisms. This note states, without over-claiming, what CIS-2 can supply as evidence and what it cannot.

## Evidence CIS-2 can support

| Safety-case need | CIS-2 contribution |
|------------------|--------------------|
| **Reproducible evaluations** | Pinned semantics and clean-room verifiers so that a safety benchmark or red-team test can be re-run to the same bits on independent hardware. |
| **Post-deployment integrity monitoring** | Continuous self-check + full-logit digests that surface silent weight, software, or environment changes (see continuous-validation materials). |
| **Provenance of a decision** | Witness digests and receipts that bind an output to a model, input, and pinned execution, verifiable offline without trusting the original operator. |
| **Third-party / independent verification** | No-GPU, cross-ISA, clean-room path so an auditor or external lab can check the claim without the original stack. |
| **Change detection** | Demonstrated sensitivity of the full-logit digest to alterations that leave generated text unchanged (bitflip and embedding-substitution head-to-heads). |
| **Incident reconstruction** | Offline re-verification of a recorded receipt or digest after the fact. |

## Evidence CIS-2 does *not* supply

| Safety-case need | Why CIS-2 is insufficient alone |
|------------------|--------------------------------|
| Alignment / intent matching | Behavioral property of training and policy, not of floating-point semantics. |
| Jailbreak and adversarial robustness | Requires behavioral testing and defenses, not bit-identity of the forward pass. |
| Factual reliability / hallucination rates | Output quality metrics, independent of numerical reproducibility. |
| Capability thresholds and dangerous-capability evals | Content of evaluations, not the integrity of their numerical substrate. |
| Human oversight design | Sociotechnical process, not a computational specification. |

## How to use this in a safety case

1. **Cite the specification and clean-room results** as evidence that the numerical execution layer is independently checkable under pinned conditions.
2. **Cite the continuous-validation experiment and monitor** (once run with real hashes) as evidence of an operational integrity monitor.
3. **Cite the integrity head-to-heads** as evidence that token-level matching alone is not a sufficient integrity check.
4. **Pair with behavioral safety evidence** — evaluations, refusal tests, monitoring of harmful outputs — rather than treating computational integrity as a substitute.

A minimal statement that remains accurate:

> “The system’s forward-pass computation is subject to bit-identical verification under the CIS-2 specification. Independent parties can re-verify digests offline. Continuous self-checks are used to detect silent changes in the computational stack. This supports evaluation reproducibility, post-deployment integrity monitoring, and auditability of individual inferences. It does not by itself establish alignment, robustness to jailbreaks, or factual reliability.”

## Related public materials

- [CIS-2 and AI Safety](CIS2_and_AI_Safety.md) — broader placement in the safety landscape
- [Continuous Validation of Bit-Identical Inference](Continuous_Validation_of_Bit_Identical_Inference.md)
- [Design Partner Adoption Path](Design_Partner_Adoption_Path.md)
- [Why Bit-Identical Inference Matters for Trust](Why_Bit_Identical_Inference_Matters_for_Trust.md)

Aefinity AI Inc. · Orange, Texas
