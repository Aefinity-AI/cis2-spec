# CIS-2: bit-identical fp32 inference, in one page

## Claims

CIS-2 is a written specification for running an fp32 transformer forward
pass such that two implementations built from the spec text alone, without
seeing each other's code or a reference implementation, produce
**bit-for-bit identical** output digests for a pinned test case
(SmolLM2-135M, prompt "Once upon a time", 16 greedy-decoded tokens). This
is stronger than "the words came out the same": it is the full fp32 logit
vector, down to the last bit, at every decode step.

What is NOT being claimed: that the model's answer was correct, safe, or
resistant to a jailbreak; that this holds for any precision other than
fp32; that it holds for any model larger than the sizes actually tested;
or that "isolated" here means separate organizations. The two clean-room
implementations were written by isolated coding agents operated by the
same person. The isolation is a read-access boundary enforced during
development, not separate authorship.

## Evidence

Four digests are pinned in the spec (section 13.1), shown here as their
leading 16 hex characters (full values in
[`EXPECTED_DIGESTS.md`](../EXPECTED_DIGESTS.md)):

- witness (`CIS2_REF`): `d82743059d1db929...`
- argmax: `0b9c8f3ac90d0b9c...`
- table: `23c7bfaf5cef0095...`
- inv_freq_table: `da9f6dcfde042558...`

The self-check script has been run end to end by the maintainer on four of
the maintainer's own machines (box1, box2, box3, penguin), covering x86_64
with and without AVX2, and reproduced the pinned digests. The machine table,
including a note on one machine where the weights were copied over the LAN,
is in the [README](../README.md). Each run finishes with a summary line
reading `passed` with a zero `failed` count, and a line reading `PASS`.

Public: github.com/Aefinity-AI/cis2-spec, and the mirrored artifacts at
huggingface.co/datasets/aefinityAIINC/cis2-conformance.

## Limits

This is a provenance and integrity check, not a content or safety check: a
model can be bit-identical and still be wrong. Everything measured so far is
fp32 only; nothing is claimed about lower-precision formats being
bit-identical across environments. The model exercised end to end by
`selfcheck.sh` is SmolLM2-135M; nothing larger has been run through this
exact check. All of the machines that ran `selfcheck.sh` are x86-64 Linux;
CI additionally covers aarch64, and an aarch64 Android phone reproduced the
pinned digests through the `verify3/` path but did not run `selfcheck.sh`.
The two clean-room implementations were built by the same operator, not by
unrelated organizations, so no outside, unaffiliated reproduction has
happened yet. No throughput numbers are published anywhere in this project;
the selfcheck wall-clock times in the README are end-to-end convenience
numbers (demo convenience, not throughput), and both clean-room verifiers
are deliberately scalar and unoptimized.

## How to reproduce

```
git clone https://github.com/Aefinity-AI/cis2-spec
cd cis2-spec && ./selfcheck.sh
```

Expect it to end with a machine fingerprint block followed by a summary line
(`== SUMMARY: N passed, 0 failed ==`) and a `PASS` line. The README explains
the `N` you will see.

Anything other than a clean PASS is itself useful information; file an issue
with the full output. Verified reproductions or verified divergences are
credited by name in the repository's
[`HALL-OF-DIVERGENCE.md`](../HALL-OF-DIVERGENCE.md). No other reward is
offered.
