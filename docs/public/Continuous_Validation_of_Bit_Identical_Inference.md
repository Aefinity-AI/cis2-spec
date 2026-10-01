# Continuous Validation of Bit-Identical Inference Integrity
**Research note and concrete CIS-2 extension directions**  
Aefinity AI · September 2026

---

## Why continuous validation matters

One-shot reproducibility is useful. Continuous validation is stronger.

Public and commercial interest in bit-identical (or deterministic) inference has risen sharply in 2025–2026. Multiple independent lines of work now treat the ability to re-check what was actually computed as a governance and safety property, not just a debugging convenience.

The core problem is operational: even when a specification requires bit-identity under pinned conditions, real serving stacks introduce nondeterminism through dynamic batching, tensor-parallel size changes, kernel variants, software updates, and quantization. Continuous validation means detecting when those conditions drift, and making past results checkable after the fact.

## What the recent literature shows

Key findings from 2025–2026 work:

- A transformer forward pass is pure arithmetic. With weights, input bytes, reduction order, floating-point environment, and software stack pinned, logits are deterministic.
- Production stacks (vLLM, SGLang, Hugging Face Transformers, etc.) are not deterministic by default. Throughput incentives favor faster, non-invariant kernels.
- Full-logit or activation digests catch alterations that token-level or top-k checks miss (exactly the class of failures already demonstrated in the CIS-2 bitflip and embedding-substitution results).
- Optimistic challenge models (operator posts result; anyone can force deterministic re-execution) make continuous validation economically practical at scale.
- Batch-invariant and tensor-parallel-invariant kernels are emerging; they reduce live-load nondeterminism but still benefit from an independent, specification-driven verification layer.

CIS-2 already owns one of the cleanest foundations in this space: cross-ISA, cross-compiler, cross-language bit-identity; clean-room independent verifiers; and full-logit witness digests that are offline-checkable without a GPU.

## Concrete CIS-2 extension directions

These extensions stay within the resources of a small lab and increase the usability and continuous-validation strength of the existing work.

### 1. Continuous witness emission (highest near-term leverage)

Make every high-stakes (or every Nth) inference emit a CIS-2-style witness digest / evidence pack that can be verified offline by anyone.

- Re-use the existing `tools/evidence_pack.py` and receipt formats.
- Publish a minimal “emit + verify” workflow that a design partner or auditor can adopt without reading the full specification.
- Success metric: an outsider can take a published receipt and re-verify it on ordinary hardware in under 15 minutes.

### 2. Challenge-friendly packaging

Design digests and evidence packs so an external party can demand and independently re-execute a specific past request with minimal friction.

- Document the exact pinned inputs needed for a challenge (weights hash, prompt bytes, decode length, environment flags).
- Provide a one-command challenge path that mirrors the existing self-check.
- Align with the optimistic-verification pattern already appearing in commercial verifiable-AI systems.

### 3. Drift-detection outer loop

Treat the public self-check and a small set of integrity probes as a continuous monitor.

- Run the self-check on a schedule (or on every deployment / environment change) using the [monitor sketch](Continuous_Validation_Monitor_Sketch.md).
- Add a short set of integrity probes drawn from the existing bitflip and embedding-substitution experiments.
- Any digest mismatch becomes a production incident, not just a research finding.

### 4. Batch- and topology-aware conformance vectors (medium-term)

Extend the normative test vectors so they remain informative under controlled changes in batch size or parallel topology, while still requiring bit-identity under the pinned CIS-2 semantics.

- Keep the existing single-sequence, greedy, offline vectors as the normative core.
- Add informative (non-normative) vectors that document behavior under batching once batch-invariant kernels are available.

### 5. Public continuous-validation experiment and adoption path

A concrete experiment that can be run on existing older hardware is outlined in [Continuous_Validation_Experiment.md](Continuous_Validation_Experiment.md).  
A four-step path for design partners is in [Design_Partner_Adoption_Path.md](Design_Partner_Adoption_Path.md).

## What this does not claim

Continuous validation of bit-identical inference does not solve alignment, jailbreaks, or hallucination. It makes the numerical execution itself continuously checkable. That is a necessary but not sufficient condition for high-assurance AI systems.

## Links

- Specification and verifiers: https://github.com/Aefinity-AI/cis2-spec
- Existing integrity demonstrations: [bitflip head-to-head](../results/2026-09-13-BITFLIP-HEADTOHEAD.md), [embedding substitution head-to-head](../results/2026-09-17-EMBED-SUBSTITUTION-HEADTOHEAD.md)
- Plain-language trust note: [Why Bit-Identical Inference Matters for Trust](Why_Bit_Identical_Inference_Matters_for_Trust.md)
- Monitor sketch: [Continuous_Validation_Monitor_Sketch.md](Continuous_Validation_Monitor_Sketch.md)
- Design partner path: [Design_Partner_Adoption_Path.md](Design_Partner_Adoption_Path.md)
- How outsiders can help: [How_Outsiders_Can_Help.md](How_Outsiders_Can_Help.md)

Aefinity AI Inc. · Orange, Texas
