# How Outsiders Can Help

CIS-2 is designed so that people who have never seen the private reference implementation can still check the claim and improve the record.

## 1. Run the public self-check (lowest friction)

```
git clone https://github.com/Aefinity-AI/cis2-spec && cd cis2-spec && ./selfcheck.sh
```

A PASS means your machine produced bit-identical results under the specification.  
A FAIL is a finding — please report it.

The self-check is intended to run on ordinary older hardware (including machines without AVX2) and does not require a GPU or a Rust toolchain for the shorter path.

## 2. Report a reproduction or a divergence

Use the issue templates:

- [Reproduction report](https://github.com/Aefinity-AI/cis2-spec/issues/new?template=reproduction-report.yml) — when the digests match on new hardware or a new environment.
- [Divergence report](https://github.com/Aefinity-AI/cis2-spec/issues/new?template=divergence-report.yml) — when the digests do not match.

Every verified report is credited by name in `HALL-OF-DIVERGENCE.md` (credit only; no cash prize).

See also `CHALLENGE.md` for what counts as an in-scope divergence versus an out-of-scope difference.

## 3. Submit a clean-room implementation

If you want to write an independent verifier:

1. Read only `docs/CIS2_SPEC_v0.3b.md`. Do not read the existing `verify2/` or `verify3/` source while implementing.
2. Keep a `CLEANROOM_LOG.md` that lists every file you read and why.
3. Your implementation must reproduce the pinned §13.1 digests on both x86-64 and aarch64, contain zero FMA-family instructions on the decode path, and pass the adversarial denormal and same-process determinism checks (spec §15).
4. Open a pull request or issue with the implementation, the log, and the CI (or local) results.

Mismatches are useful. Earlier clean-room work is how the specification’s own item-order bug was found and fixed.

## 4. Point others at the concrete integrity examples

Two existing results already show the difference between token-level checks and the full-logit witness digest:

- [Single-weight corruption head-to-head](../results/2026-09-13-BITFLIP-HEADTOHEAD.md)
- [Realistic embedding substitution head-to-head](../results/2026-09-17-EMBED-SUBSTITUTION-HEADTOHEAD.md)

These are the strongest public demonstrations currently available that the full-logit digest catches alterations that simpler checks can miss.

## 5. Plain-language materials

If you are explaining the work to non-experts, start with:

- [Why Bit-Identical Inference Matters for Trust](Why_Bit_Identical_Inference_Matters_for_Trust.md)

Thank you for testing, reporting, and extending the record.
