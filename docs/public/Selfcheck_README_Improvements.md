# Self-Check Public Improvements (Draft for README)

**Suggested additions to the top of the CIS-2 README and to the self-check output**

---

## 1. Suggested new section for the README (place near the top, right after the one-command self-check)

### What the self-check shows (and what it does not)

**What a successful run shows**
- The same model weights + the same input produce bit-identical numerical results under the CIS-2 specification.
- The computation matches the pinned digests on the machine you just ran it on.
- An independent party can re-verify the result offline without trusting the original operator.

**What a successful run does *not* show**
- That the model is aligned, safe, or free of harmful outputs.
- That the model will behave the same way under sampling, different temperatures, or longer contexts.
- That every possible hardware configuration in the world will match (only that the ones tested so far do, and that the specification is precise enough for others to check).

If the digests match, the floating-point computation itself was the same. That is one concrete piece of trustworthiness. It is not a complete safety assurance.

---

## 2. Recommended human-readable output for a successful self-check

At the very end of a successful `./selfcheck.sh` (or equivalent) run, print something close to this:

```
============================================================
RESULT: PASS
============================================================
All pinned digests matched.

This means the computation produced bit-identical results
under the CIS-2 specification on this machine.

Machine fingerprint:
  CPU : [detected model]
  ISA : [relevant flags]
  OS  : [kernel / distro summary]

You can re-run this check on any other machine.
If the digests still match, the numerical result was the same.
============================================================
```

---

## 3. Recommended human-readable output for a failed self-check

```
============================================================
RESULT: FAIL
============================================================
One or more pinned digests did not match.

This means the numerical result on this machine differs from
the reference under the CIS-2 specification.

Possible causes include:
  - different compiler or optimization settings
  - different floating-point environment (FTZ/DAZ, etc.)
  - a modified or different set of model weights
  - an implementation that does not follow the pinned reduction order

See EXPECTED_DIGESTS.md and the divergence reporting instructions
for how to investigate or report the difference.
============================================================
```

---

## 4. Suggested short note to add under the self-check command in the README

```markdown
The self-check is designed to run on ordinary computers (including older CPUs) without a GPU.  
It downloads only the pinned test weights, builds the independent C verifier, and checks the digests.  
A PASS means an independent machine produced the same bit-level result.
```

These changes make the strongest public property of CIS-2 immediately understandable to non-experts.
