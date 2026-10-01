# CIS-2 and AI Safety
**Where bit-identical inference integrity sits in the broader safety landscape**  
Aefinity AI · September 2026

---

## The honest placement

AI safety is not one problem. It is a cluster of problems that interact:

| Safety concern | Typical focus | Does CIS-2 address it directly? |
|----------------|---------------|----------------------------------|
| Alignment / value loading | Model behavior matches human intent | No |
| Jailbreaks & adversarial robustness | Resistance to harmful or evasive prompts | No |
| Hallucination & factual reliability | Truthfulness of outputs | No |
| Interpretability | Understanding internal computations | Partially (makes the forward pass inspectable) |
| Evaluation reproducibility | Same test, same numerical result | **Yes** |
| Monitoring & oversight | Detecting drift, tampering, silent changes | **Yes** |
| Accountability & provenance | Reconstructing what was computed | **Yes** |
| High-assurance / regulated deployment | Checkable claims for auditors | **Yes** |
| Scalable oversight | Humans supervising powerful systems | Indirectly (reliable evidence base) |

CIS-2 is a **foundation-layer contribution**. It makes the numerical execution of the model checkable. It does not make the model aligned, refuse harmful requests, or stop hallucinating. Those remain separate research and engineering problems.

What it does is remove a specific, often overlooked failure mode: the inability of independent parties to confirm that a claimed output really came from a claimed model and input under pinned semantics.

## Why this matters for overall safety

Modern safety practice increasingly depends on evidence that can be re-examined:

- **Frontier safety frameworks** emphasize evaluations, post-deployment monitoring, incident response, and accountability.
- **Risk-management standards** (NIST AI RMF profiles, ISO-aligned frameworks, national and international safety reports) call for continuous monitoring, change management, and auditable records.
- **Governance and assurance** efforts treat “show me what was actually computed” as a prerequisite for credible claims about risk, especially when adversaries or silent software changes are in scope.

If the underlying computation is not reproducible, then:

- Safety benchmarks can disagree across machines for reasons unrelated to the model’s behavior.
- Post-incident reconstruction becomes ambiguous.
- Independent auditors must take the operator’s word for the numerical result.
- Continuous monitoring cannot distinguish benign environmental drift from real integrity failures.

Bit-identical, independently verifiable inference closes that gap at the level of the forward pass. Continuous validation (scheduled self-checks, witness digests, challenge-friendly evidence packs) turns the property into an ongoing monitor rather than a one-shot demo.

## How CIS-2 maps onto safety practice

### 1. Evaluation infrastructure

Safety evaluations and red-team tests are only as trustworthy as their reproducibility. CIS-2 provides pinned semantics and clean-room verifiers so that “the same test” can mean the same bits, not merely similar text. That reduces a class of noise that otherwise confounds comparisons across labs, hardware, and time.

### 2. Post-deployment monitoring and drift detection

A continuous-validation loop around the public self-check and full-logit digests can surface silent changes in weights, software, or environment—the same class of failures already demonstrated in the published bitflip and embedding-substitution head-to-heads. This is operational monitoring of computational integrity, complementary to behavioral monitoring (toxicity, refusal rates, capability probes).

### 3. Accountability and provenance

Combined with receipts and agent traces, CIS-2 supports offline reconstruction of what was computed. An auditor, regulator, or affected party can re-verify a chain without trusting the original operator and without needing a GPU. That is a concrete contribution to accountability requirements that appear across frontier safety policies and risk-management frameworks.

### 4. High-assurance and regulated settings

Where systems must support safety cases, third-party review, or regulated decision trails, checkable computation is a prerequisite. CIS-2 is designed for exactly that setting: offline, cross-machine, clean-room verifiable digests under a written specification. See [Safety Case Contribution](Safety_Case_Contribution.md) for explicit mapping onto safety-case evidence needs.

### 5. What remains outside scope

CIS-2 does not replace:

- Alignment research and preference optimization
- Adversarial robustness and jailbreak defenses
- Mechanistic interpretability
- Capability evaluations and dangerous-capability thresholds
- Sociotechnical governance and human oversight design

It supplies a more reliable computational substrate on which those efforts can rest.

## Relation to continuous validation

One-shot verification is useful. Continuous validation is the safety-relevant form of the same idea.

The public materials under `docs/public/` provide:

- A research framing for continuous validation of bit-identical inference
- A practical experiment outline runnable on ordinary older hardware
- A minimal monitor sketch for scheduled self-checks
- A design-partner adoption path from first self-check to ongoing monitoring
- A short FAQ for safety researchers

Together they turn a specification-level integrity property into something that can be operated, audited, and extended—without claiming to solve the full safety problem.

## Bottom line for AI safety

Public fears about AI often center on opacity, unpredictability, and the inability of outsiders to check claims. Behavioral safety work addresses what the model *does*. Computational integrity work addresses whether we can *know* what was computed.

Both are necessary. CIS-2 is a contribution to the second. Making that contribution continuous, independently checkable, and usable by non-experts is how it can meaningfully support broader AI safety goals without over-claiming.

## Links

- [Safety Case Contribution](Safety_Case_Contribution.md)
- [Safety Researcher FAQ](Safety_Researcher_FAQ.md)
- [Why Bit-Identical Inference Matters for Trust](Why_Bit_Identical_Inference_Matters_for_Trust.md)
- [Continuous Validation of Bit-Identical Inference](Continuous_Validation_of_Bit_Identical_Inference.md)
- [Design Partner Adoption Path](Design_Partner_Adoption_Path.md)
- [Continuous Validation Experiment](Continuous_Validation_Experiment.md)
- [Continuous Validation Monitor Sketch](Continuous_Validation_Monitor_Sketch.md)
- Specification: https://github.com/Aefinity-AI/cis2-spec

Aefinity AI Inc. · Orange, Texas
