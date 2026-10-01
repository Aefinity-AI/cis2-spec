# CIS-2 Next-Level Actions
**Practical steps to increase public impact on safety-related trust**

---

## 1. Public Self-Check Improvements (Priority)

### Goal
Make independent verification so easy that a non-expert can run it in under 10–15 minutes on ordinary older hardware and understand the result.

### Concrete improvements
- Ensure the top-level command remains a single script (`./selfcheck.sh` or equivalent).
- Add a clear human-readable summary at the end of every successful run:
  ```
  RESULT: PASS
  All pinned digests matched.
  This means the computation produced bit-identical results under the CIS-2 specification.
  Machine fingerprint: [cpu / os summary]
  ```
- On failure, print a short plain-English explanation of what mismatched and what that implies.
- Document exact tested machines (including older CPUs such as Celeron and early i5) in a visible table on the README.
- Add a short “What this shows / What this does not show” section directly under the self-check instructions.
- Consider a minimal web page or static demo page that links to the script and explains the output in non-technical language.

### Success metric
A stranger with a standard Linux laptop (or even a modest older machine) can clone, run, and correctly interpret the result without further help.

---

## 2. Failure-Demonstration Experiment (Recommended Initial Demo)

### Goal
Show a concrete integrity failure that ordinary logging or token-level checks can miss, but that CIS-2 full-logit digests catch.

### Suggested experiment: Embedding-row or weight-bit flip
1. Start from a pinned, known-good CIS-2 run (SmolLM2-135M or the current test vector) and record the witness digest.
2. Introduce a minimal, realistic change that does not alter the generated tokens under greedy decoding in the short test sequence (for example: a single-bit flip in an unused embedding row, or a tiny scale change that stays within the same argmax).
3. Re-run under the same CIS-2 procedure.
4. Show three checks side-by-side:
   - Final generated tokens (often still identical)
   - Top-k or text-replay style check (often still passes)
   - Full CIS-2 witness digest over the logits (fails)

### Output
A short markdown report with:
- Exact change made
- Commands used
- Side-by-side results
- Plain-language conclusion: “Token-level checks can miss this class of alteration. The full-logit digest does not.”

This becomes a reusable public example of why bit-identical verification adds safety value beyond ordinary output matching.

---

## 3. Plain-Language Materials (Already Drafted)

- `Why_Bit_Identical_Inference_Matters_for_Trust.md` — a short public note that explains the contribution without over-claiming.
- Use it on the repository, the company site, or as a one-pager for outreach.
- Keep the tone precise: state what is solved and what is not solved.

---

## 4. Immediate Sequencing Recommendation

1. Polish the self-check output and README language so the public path is as clear as possible.
2. Run and publish the initial failure-demonstration experiment (embedding or weight alteration that preserves tokens but breaks the digest).
3. Publish the plain-language note alongside the improved self-check.
4. Update the public challenge / issue templates so external reproductions and divergence reports are easy to submit and are visibly credited.

These four steps stay within the resources of a small lab using older equipment and directly increase the chance that outsiders can see and verify the integrity property for themselves.

---

## What success looks like in 4–8 weeks
- A non-expert can run the self-check and correctly interpret the result.
- At least one concrete, published demonstration exists of an integrity failure that token-level checks miss but CIS-2 catches.
- The plain-language explanation is public and linked from the repository.
- External people have a clear, low-friction path to try the verification themselves and report results.
