# FAQ for Safety Researchers
**CIS-2, computational integrity, and continuous validation**  
Aefinity AI · September 2026

---

## What problem does CIS-2 actually address?

Independent parties often cannot confirm that a claimed model output really came from a claimed model and input under known numerical semantics. Different machines, compilers, or environments can diverge even when weights and prompts match. CIS-2 pins the floating-point semantics of an fp32 transformer forward pass so that conforming implementations produce **bit-identical** full-logit digests that can be re-verified offline, without a GPU and without trusting the original operator.

## Is this alignment research?

No. Alignment concerns whether the model’s behavior matches human intent. CIS-2 concerns whether the numerical computation is independently checkable. Both matter for overall safety; they are different layers.

## Does it help with jailbreaks or hallucinations?

Not directly. Those are behavioral properties. CIS-2 can make evaluations of jailbreaks or hallucination rates more reproducible (same test, same bits), but it does not improve the model’s refusal behavior or factual accuracy.

## Why full-logit digests instead of matching generated text?

Token-level or top-k checks can pass while the underlying computation has changed. Published head-to-heads show that a single flipped weight bit or an unexercised embedding-row change can leave the generated text unchanged while breaking the full-logit witness digest. Continuous monitoring of the digest therefore catches a class of integrity failures that text matching misses.

See: [bitflip head-to-head](../results/2026-09-13-BITFLIP-HEADTOHEAD.md), [embedding substitution head-to-head](../results/2026-09-17-EMBED-SUBSTITUTION-HEADTOHEAD.md).

## How does continuous validation fit?

One-shot self-checks are useful. Continuous validation turns the same property into an ongoing integrity monitor: scheduled self-checks, drift detection, and challenge-friendly evidence packs. A practical experiment outline and a minimal monitor sketch are published for small-lab hardware.

See: [Continuous Validation research](Continuous_Validation_of_Bit_Identical_Inference.md), [Experiment](Continuous_Validation_Experiment.md), [Monitor sketch](Continuous_Validation_Monitor_Sketch.md).

## Can this appear in a safety case?

Yes, as evidence for evaluation reproducibility, post-deployment integrity monitoring, provenance of individual inferences, and independent verification—**not** as evidence for alignment, adversarial robustness, or factual reliability. Suggested language is in [Safety Case Contribution](Safety_Case_Contribution.md).

## What hardware is required to verify?

Ordinary CPUs are sufficient for the public self-check and continuous-validation experiment. No GPU is required for the core offline verification path. Cross-ISA and clean-room results are already published.

## How do I try it?

```
git clone https://github.com/Aefinity-AI/cis2-spec && cd cis2-spec && ./selfcheck.sh
```

A PASS means an independent machine produced bit-identical results under the specification. For a structured path from first check to continuous monitoring, see [Design Partner Adoption Path](Design_Partner_Adoption_Path.md).

## How can outsiders contribute?

Reproduction reports, divergence reports, and clean-room implementations are welcome. See [How Outsiders Can Help](How_Outsiders_Can_Help.md).

## Where does this sit in the broader AI safety landscape?

See [CIS-2 and AI Safety](CIS2_and_AI_Safety.md) for an explicit mapping onto evaluation, monitoring, accountability, and high-assurance deployment—and for a clear statement of what remains outside scope.

Aefinity AI Inc. · Orange, Texas
