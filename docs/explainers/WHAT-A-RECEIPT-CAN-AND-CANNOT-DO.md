# What a receipt can and cannot do

AI companies say things like "this answer came from our safety-tuned
model" or "we didn't swap in a cheaper model to save money." Right now,
almost nobody outside the company can check whether that is true. This
note is about a way to make that specific kind of claim checkable. It is
not about making AI safe, honest, or correct. Those are different, harder
problems, and nothing here solves them.

## Three terms, plainly

**CIS-2** is a written specification for running a model's calculation so
that anyone who reads the spec, with no access to the original company's
code, can build their own program, run the same model on the same input,
and get an output that matches down to the exact bit. That is stricter
than "close enough" or "the same words."

A **receipt** is a record, generated automatically while a model runs,
that chains together a hash of the exact weights used, the exact input,
every step the model took, and any outside tool results it relied on. It
is small and it travels separately from the model. Someone with no GPU and
no relationship to the original machine can check it against a weights
file and a public verifier program.

The **no-receipt-no-action gateway** is a rule you can put in front of a
system: no receipt, no output gets used. If a step in a pipeline (say, an
agent calling a tool, or a model answering a customer) does not produce a
valid receipt, whatever it produced is refused rather than acted on. It
turns "trust us" into "check this," for the narrow question of what ran.
Design notes for the gateway are in [`docs/design/`](../design/RECEIPT-GATEWAY.md).

## What this makes checkable

If a vendor says "model version X, weights hash Y, produced this specific
output for this specific input," a receipt plus the gateway lets an
outside party (a regulator, a journalist, a customer's own auditor)
confirm or contradict that claim without needing access to the vendor's
servers, source code, or cooperation. They need only the receipt, the
weights file (or its published hash), and a public verifier. Run the
verifier on ordinary hardware and it either reproduces the bit-for-bit
output the receipt claims, or it does not.

If a vendor claims model A ran but the weights hash in the receipt matches
model B, that is a checkable contradiction, not a matter of taking anyone's
word for it. If the vendor claims "nothing was changed after generation"
and a single byte of the record was altered, the hash chain breaks and the
check fails. Checking does not require expensive compute.

## What this does not do

A receipt tells you what ran. It says nothing about whether what ran was a
good idea. A model can be exactly, verifiably itself (identical weights,
identical computation, a receipt that checks out cleanly) and still give a
wrong, biased, or harmful answer. Receipts are a provenance and integrity
check: which model, which weights, which input, which output, unaltered
since it was produced. They do not evaluate whether the model is honest,
whether its training data was appropriate, or whether its output is safe
to act on. A vendor with a spotless receipt record can still ship a bad
model; the receipt just means you can show which bad model it was, and
that its output was not tampered with afterward. This is a narrower, more
modest claim than "trustworthy AI," and it should be described that way.

## The failure modes we found in our own testing

We do not only report the successes. Two gaps we found in our own testing,
and published:

Replaying the model's plain text output and comparing words is a weak
check. In controlled tests where we deliberately corrupted single bits of
a model's weights, text-level replay missed many of the corruptions,
because a corrupted weight can shift internal numbers without changing
which word the model picks as its top choice. A cheaper "top candidates"
check did better but still missed a meaningful share. Only the full
numeric witness digest, which hashes every underlying number and not just
the words that were chosen, caught every corruption we tested. The lesson
is that not every kind of "check" is equally strong, and a vendor's claim
to have "verified" an output is only as good as which check they used. The
counts are in the
[single-weight corruption write-up](../results/2026-09-13-BITFLIP-HEADTOHEAD.md).

Separately, an early version of our receipt design had a real hole: a
genuine, unaltered receipt from one task could be filed under a different
task's name and would still verify as valid, because the receipt only
checked its own contents, not which task it was supposed to belong to. We
found this ourselves and closed it in our test setup by adding a binding
value, derived from the task's own identity and inputs, into the hash
chain, so a receipt filed under the wrong task name now fails to verify.
The [write-up](../results/2026-09-13-RECEIPT-VERIFIED-EVAL60.md) says
where that fix stands. We name this gap because a checking system is only
as good as the parts of a claim it actually binds together.

## How to check a vendor's claim yourself, in an afternoon

1. Ask the vendor for the receipt covering the specific output in
   question, plus a hash or copy of the model weights they say produced it.
2. Get the public verifier program (open source; no special hardware
   needed) and confirm it is the version matching the spec the receipt
   claims to follow.
3. Run the verifier against the receipt and the weights on your own
   machine. A laptop is enough.
4. Compare the digest the verifier produces against the digest recorded in
   the receipt. A match means the specific output was produced by those
   specific weights on that specific input, unaltered since. A mismatch,
   or a refusal to hand over a receipt at all, is itself informative.
5. Ask what kind of check was used to catch tampering (full witness
   digest, or a weaker text or top-candidates replay). Per the failure
   modes above, this matters for how strong the claim actually is.

That afternoon of work gets you a checkable answer to "did this specific
model produce this specific output, unaltered." It does not get you an
answer to "is this model good, fair, or safe." That remains a separate
question, and no receipt answers it.

To try the verifier itself on a pinned test case, see the self-check in the
[README](../../README.md).
