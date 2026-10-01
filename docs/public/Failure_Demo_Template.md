# Failure-Demonstration Experiment Template
**Goal:** Show a concrete integrity failure that ordinary token-level checks can miss, but that the CIS-2 full-logit witness digest catches.

---

## Experiment design (ready to run)

### Setup
1. Start from the current pinned CIS-2 test vector (SmolLM2-135M, prompt “Once upon a time”, 16 greedy tokens).
2. Record the original `CIS2_REF` witness digest and the generated token IDs.
3. Create a minimally altered weight file that changes an embedding row (or a scale factor) that is **not exercised** by the short test prompt, or that stays within the same argmax for the 16-token sequence.
4. Re-run the exact same CIS-2 verification procedure on the altered weights.
5. Compare three things side-by-side:
   - Final generated tokens
   - Simple text-replay / top-k style check
   - Full CIS-2 witness digest over the logits

### Expected outcome (the useful case)
- Generated tokens remain identical (or change only in ways a casual observer would not notice).
- Token-level or top-k checks still pass.
- The full-logit CIS-2 witness digest fails.

This demonstrates that token-level matching alone is not sufficient to detect certain classes of alteration, while the bit-identical full-logit digest is.

---

## Report template (fill in after you run it)

```markdown
# Integrity Failure Demonstration — [Date]

## Summary
A minimal change was introduced into the model weights.  
The generated tokens under the standard 16-token greedy test remained [identical / nearly identical].  
A simple token-level check passed.  
The CIS-2 full-logit witness digest failed.

This shows that bit-identical verification over the logits can detect alterations that token-level checks miss.

## Exact change made
- File: [path to altered weights]
- Change: [e.g. single-bit flip in embedding row N, or scale multiplied by 1.0000001 on an unused row]
- Rationale for choosing this change: [why it was expected to preserve tokens but alter logits]

## Commands used
[exact commands for original run and altered run]

## Results

| Check                        | Original | Altered | Match? |
|-----------------------------|----------|---------|--------|
| Generated token IDs         | ...      | ...     | Yes/No |
| Simple text / top-k check   | Pass     | Pass    | Yes    |
| CIS-2 witness digest (CIS2_REF) | [hash] | [hash] | **No** |

## Plain-language conclusion
Token-level checks can miss this class of alteration.  
The full-logit CIS-2 digest does not.  
Independent verification of the numerical computation therefore provides stronger integrity assurance than matching the final text alone.

## Reproducibility
All original and altered artifacts, commands, and digests are recorded in this repository under [path].
```

---

## How to use this
1. Run the experiment on your hardware.
2. Fill in the template above with the real numbers and hashes.
3. Publish the completed report under `docs/public/` or `docs/results/`.
4. Link it from the main README and from the plain-language trust note.

This becomes a concrete, citable example that outsiders can understand without needing to read the full specification.
