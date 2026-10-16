# Negative result: naive post-training ternary weights collapse a small model

Date: 2026-10-16. Status: negative result, recorded so others do not repeat it.

## What was tried

We took SmolLM2-135M, the small model used for the CIS-2 pinned test case,
and quantized its weight matrices after training to three levels (-1, 0,
+1) with one scale per output row, using a simple mean-magnitude threshold,
in the style of published ternary-weight work. We used no calibration data,
no fine-tuning, and no distillation. Activations were quantized to int8 and
the matrix products were accumulated in integers in a fixed order, so that
the computation stayed deterministic. We compared this against the same
model with int8 weights, which had earlier stayed close to the fp32 model.

## What was observed

- Determinism held. Repeated runs in separate processes produced identical
  digests on the one machine we tried.
- Size was substantially smaller than the int8 model, as expected for two
  bits per weight.
- Quality collapsed. On a short public-domain text slice, perplexity rose by
  several orders of magnitude relative to fp32, the ternary model's top
  prediction never agreed with the fp32 model's on that slice, and greedy
  generation degenerated into one repeated token after the opening generated
  token. The int8 model, by contrast, degraded only slightly.

## Why it does not help

Bit-exact reproducibility is worthless if the model no longer works. Zeroing
or sign-quantizing every weight of an already trained model discards too
much magnitude information at this model size, and the loss is not
recovered without training. The int8 model gives a far better trade between
quality and size under the same no-calibration conditions.

## Limits of this note

- One small model, one machine, one short text slice. The eval slice was
  short because the unpacking was slow on the scalar CPU used.
- This says nothing about ternary methods that train the model at low bit
  width from the start or fine-tune after quantizing. It is evidence only
  against naive post-training ternarization.
- This is not part of the CIS-2 specification or any conformance claim.
  Exact figures are withheld from this note.
