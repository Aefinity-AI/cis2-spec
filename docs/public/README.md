# Public Materials — CIS-2 and Trust

These documents explain the contribution of bit-identical inference in plain language, situate it within broader AI safety, and provide practical next steps for independent verifiability and continuous validation.

## Start here

| Document | Purpose |
|----------|---------|
| [Elevator Summary](Elevator_Summary.md) | One-page overview for design partners and safety researchers |
| [Paper Status](Paper_Status.md) | Preprint / review status (venue omitted during double-blind review) |
| [CIS-2 and AI Safety](CIS2_and_AI_Safety.md) | Where computational integrity sits in the overall AI safety landscape |
| [Safety Case Contribution](Safety_Case_Contribution.md) | What CIS-2 can and cannot supply as evidence in an AI safety case |
| [Safety Researcher FAQ](Safety_Researcher_FAQ.md) | Short answers on scope, limits, continuous validation, and how to try it |
| [Design Partner Adoption Path](Design_Partner_Adoption_Path.md) | Four-step path from first self-check to continuous monitoring |
| [Why Bit-Identical Inference Matters for Trust](Why_Bit_Identical_Inference_Matters_for_Trust.md) | Plain-language explanation of what CIS-2 contributes to trust and what it does not solve |
| [Continuous Validation of Bit-Identical Inference](Continuous_Validation_of_Bit_Identical_Inference.md) | Research note mapping recent literature onto concrete CIS-2 extension directions |
| [Continuous Validation Experiment](Continuous_Validation_Experiment.md) | Practical experiment outline runnable on existing older hardware |
| [Continuous Validation Monitor](Continuous_Validation_Monitor_Sketch.md) | Minimal script (`scripts/cis2_monitor.sh`) for scheduled integrity monitoring |
| [How Outsiders Can Help](How_Outsiders_Can_Help.md) | Clear paths for external reproductions, divergence reports, and clean-room contributions |
| [Self-check README Improvements](Selfcheck_README_Improvements.md) | Clearer messaging for non-experts running the public self-check |
| [Failure-Demonstration Experiment Template](Failure_Demo_Template.md) | Ready-to-run experiment showing integrity failures that token-level checks can miss |
| [CIS-2 Next-Level Actions](CIS2_Next_Level_Actions.md) | Prioritized practical steps for a small lab |

## Completed continuous-validation experiment

- **[2026-09-27 Continuous Validation on penguin (i5-10210U)](../results/2026-09-27-CONTINUOUS-VALIDATION.md)** — baseline bit-identity, one-byte weight corruption detected by verifier artifact hash (FATAL), monitor PASS after restore. No GPU.

## Existing concrete integrity demonstrations already in this repository

These results are the strongest public evidence that the full-logit witness digest catches alterations that simpler checks miss:

- [Single-weight corruption head-to-head](../results/2026-09-13-BITFLIP-HEADTOHEAD.md) — a flipped weight bit can leave the generated text unchanged while breaking the witness digest.
- [Realistic embedding substitution head-to-head](../results/2026-09-17-EMBED-SUBSTITUTION-HEADTOHEAD.md) — quantization, pruning, fine-tune and surgical row-patch cases; only the full-logit digest catches every alteration, including unexercised rows.
- [Receipt-verified EVAL-60 and tamper detection](../results/2026-09-13-RECEIPT-VERIFIED-EVAL60.md) — 60/60 receipts verify; tamper detection improves after binding context into the receipt.

## How to verify yourself

```
git clone https://github.com/Aefinity-AI/cis2-spec && cd cis2-spec && ./selfcheck.sh
```

A PASS means an independent machine produced bit-identical results under the CIS-2 specification.

For continuous monitoring: `bash scripts/cis2_monitor.sh`  
For paper status: [Paper Status](Paper_Status.md)  
For a structured adoption path: [Design Partner Adoption Path](Design_Partner_Adoption_Path.md)  
For the broader safety framing: [CIS-2 and AI Safety](CIS2_and_AI_Safety.md)
