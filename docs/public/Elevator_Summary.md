# CIS-2 in One Page
**Bit-exact, independently verifiable fp32 transformer inference**

---

## The problem
Most claims about what an AI system actually computed rest on trust in the operator. Different machines, compilers, or environments can produce different numerical results even when the model weights and input are identical. When the underlying computation is not reproducible, independent checking becomes difficult or impossible.

## What CIS-2 provides
A normative specification for the floating-point semantics of an fp32 transformer forward pass. When an implementation follows the specification:

- The same model + same input produce **bit-identical** full-logit outputs across different machines, compilers, languages, and even CPU vs GPU.
- An independent party can re-run the computation offline and check a short cryptographic digest.
- No GPU, no special hardware, and no trust in the original operator are required for verification.

Two independent clean-room implementations (C and Rust), written from the specification text alone, already reproduce the same digests.

## Why this matters for safety and trust
- It reduces the “black box” problem at the level of the computation itself.
- It makes safety evaluations and red-team results more reproducible.
- Combined with receipts and agent traces, it supports offline auditability of decisions.
- It raises the cost of silent changes to weights, software, or the execution environment.
- Continuous validation turns one-shot checks into an ongoing integrity monitor that can detect drift or tampering over time.

It does **not** solve alignment, jailbreaks, or hallucination. Those are separate problems. CIS-2 makes the numerical execution itself checkable — a foundation layer within the broader AI safety landscape (see [CIS-2 and AI Safety](CIS2_and_AI_Safety.md)).

## How to verify yourself (under 15 minutes)
```
git clone https://github.com/Aefinity-AI/cis2-spec && cd cis2-spec && ./selfcheck.sh
```
A PASS means an independent machine produced bit-identical results under the specification.

## Concrete evidence already published
- Bit-identical results across x86-64 (AVX2 and scalar), aarch64, and NVIDIA GPUs (Tesla P100 / T4).
- Head-to-head demonstrations that a single flipped weight bit or an unexercised embedding-row change can leave generated text unchanged while breaking the full-logit witness digest.
- **Completed continuous-validation experiment** (2026-09-27, Intel i5-10210U): baseline bit-identity, one-byte weight corruption detected by the verifier artifact-hash gate, continuous monitor PASS after restore — [report](../results/2026-09-27-CONTINUOUS-VALIDATION.md).
- Public challenge and issue templates for external reproductions and divergence reports.

## Research paper status
A related manuscript is under confidential peer review. **Venue and submission details are not listed publicly** during double-blind review. See [Paper Status](Paper_Status.md). An arXiv ID will be linked when a preprint is public.

## Who this is for
- Organizations that need checkable provenance for high-stakes or regulated AI decisions.
- Safety researchers who want reproducible evaluation infrastructure.
- Design partners interested in receipt-gated agent tooling or verifiable inference at the edge.

## Links
- Specification & verifiers: https://github.com/Aefinity-AI/cis2-spec
- Paper status: https://github.com/Aefinity-AI/cis2-spec/blob/main/docs/public/Paper_Status.md
- Continuous validation report: https://github.com/Aefinity-AI/cis2-spec/blob/main/docs/results/2026-09-27-CONTINUOUS-VALIDATION.md
- AI safety framing: https://github.com/Aefinity-AI/cis2-spec/blob/main/docs/public/CIS2_and_AI_Safety.md
- Design partner path: https://github.com/Aefinity-AI/cis2-spec/blob/main/docs/public/Design_Partner_Adoption_Path.md
- Company site: https://aefinity-ai.github.io/
- Contact: https://aefinity-ai.github.io/pilot.html (or a GitHub issue on Aefinity-AI/cis2-spec)

Aefinity AI Inc. · Orange, Texas · 2026
