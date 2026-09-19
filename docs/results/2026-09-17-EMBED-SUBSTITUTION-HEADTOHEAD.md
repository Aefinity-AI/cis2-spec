# Realistic embedding substitution: text replay vs top-5 logit check vs witness digest

Five realistic-substitution experiments against a pruned BitNet-2B
checkpoint's continuous bf16 embedding table, again comparing three
independent tampering-detection methods — text/argmax replay (A), a
top-5-logit-index set check (B), and the CIS witness-chain digest (C) —
this time under substitutions that model plausible deployment scenarios
(quantization, pruning, a fine-tune-style patch, a surgical single-row
edit) rather than random single-bit flips.

Rule A: this is an identity/correctness experiment. No timing figures are
reported or implied anywhere below.

## 1. Question

The earlier bitflip head-to-head (see
[2026-09-13-BITFLIP-HEADTOHEAD.md](2026-09-13-BITFLIP-HEADTOHEAD.md))
used single random bit flips. Real deployment substitutions look
different: someone quantizes a checkpoint, prunes it, fine-tunes it, or
patches a handful of rows. Do the same three detectors — text replay,
top-5 logit check, witness digest — still separate the way they did for
bit flips, once the corruption is a whole realistic operation rather than
one flipped bit? And specifically: can a *single* patched embedding row
hide from output-based checks if the test prompts never happen to read it?

## 2. Setup

**Model:** the same pruned BitNet-2B checkpoint family used in the
bitflip head-to-head — ternary attention/FFN weights packed 2 bits/weight
with per-tensor f32 dequantization scales, and one continuous bf16
embedding table (tied embeddings, no separate lm_head), vocab size
50256, hidden size 2560. All substitutions below apply only to this
embedding table: the packed ternary weights only take three exact
values, so "quantize" or "magnitude-prune" is not a meaningful operation
on them without redefining the operation; the embedding table is the one
tensor these operations are normally defined on.

**Checks (A/B/C), as implemented:** same harness pattern and check
definitions as the bitflip experiments — A = generated token ids
identical to baseline at every decode step; B = top-5 logit-index set
identical to baseline at every decode step; C = full witness-chain digest
(i64-logit SHA-256 fold) identical to baseline. "Detect" = the check
differs from baseline.

**Harness.** A new example program derived from the bitflip head-to-head
harness (same `run()` structure and A/B/C definitions), extended with one
substitution routine per class below. The harness is not yet published
in a public repo; described here in prose only.

**Standard 20-prompt set** (classes a, b, d): 20 short single-sentence
prompts spanning narrative, technical, news, and procedural registers,
`max_new=64`, greedy decode.

**Reduced 5-prompt subset** (classes b', b''): prompt indices 0, 4, 8,
12, 16 of the standard 20, chosen for turn-budget reasons, `max_new=64`.

All substitutions were applied to an in-memory working copy only; every
run verified the on-disk embedding file was byte-identical before and
after (`embed_bytes_unchanged_after_run=true`).

## 3. Substitution classes

### (a) INT8 quantization round-trip (n=20 prompts, global substitution)

Symmetric per-tensor INT8 PTQ round-trip on the whole embedding table:
`scale = max_abs(table)/127`, `q = round(v/scale)` clamped to
`[-127,127]`, dequantized back to bf16. `elems=128,655,360`,
`changed=124,674,862` (96.91% of embedding values changed) — a large,
realistic global perturbation.

| detector | detected | fraction | Wilson 95% CI |
|---|---|---|---|
| A (text replay) | 20/20 | 100.0% | 83.9–100.0% |
| B (top-5 logit-index check) | 20/20 | 100.0% | 83.9–100.0% |
| C (witness digest) | 20/20 | 100.0% | 83.9–100.0% |

All three caught every case. Unlike the bitflip experiments, A and B also
reached 100% here — a global, ~97%-of-values quantization perturbation is
far coarser than a single bit flip and visibly changes generated text on
every prompt tested. This does not establish that A/B would catch a
smaller/more surgical quantization; a much coarser corruption was tested.

### (d) 1% magnitude pruning (n=20 prompts, global substitution)

Exact 1% magnitude pruning of the embedding table: full sort of `|v|`
over all values, zero every value at or below the rank-k threshold where
k = round(1% x n). `elems=128,655,360`, `pruned=1,286,554` (exactly
1.0000%).

| detector | detected | fraction | Wilson 95% CI |
|---|---|---|---|
| A (text replay) | 18/20 | 90.0% | 69.9–97.2% |
| B (top-5 logit-index check) | 20/20 | 100.0% | 83.9–100.0% |
| C (witness digest) | 20/20 | 100.0% | 83.9–100.0% |

A missed 2 of 20 prompts: greedy-argmax generation coincidentally matched
the clean baseline's exact token sequence on those two prompts even
though the embedding table differed and the underlying logits (B, C)
had changed. B and C both stayed at 100%. This is direct evidence, in
this harness, that exact-text replay (A) is not a reliable standalone
detector even for a whole-tensor, deterministic, non-adversarial
perturbation once the perturbation is small enough — consistent with the
bitflip experiments' finding that A's miss rate is corruption-magnitude
dependent, now shown on a different kind of corruption.

### (b) Approximated fine-tune patch (n=20 prompts, localized substitution)

No differentiable forward pass or PyTorch install was available on the
box used for this experiment, so a real gradient-based fine-tune could
not be run. Instead, a bag-of-tokens / next-token co-occurrence style
update was used as an explicitly-labeled approximation of a fine-tune
patch, not a real gradient step: for 50 epochs over a 15-sentence
held-out corpus (topically distinct from the 20 detection prompts), each
adjacent token bigram nudged the earlier token's embedding row toward the
later token's row (`embed_row[prev] += lr * (embed_row[next] -
embed_row[prev])`, `lr=1e-3`). `changed=80,244` elements (0.0624% of the
table), `rows_touched=133` of 50256 rows (0.26%).

| detector | detected | fraction | Wilson 95% CI |
|---|---|---|---|
| A (text replay) | 20/20 | 100.0% | 83.9–100.0% |
| B (top-5 logit-index check) | 20/20 | 100.0% | 83.9–100.0% |
| C (witness digest) | 20/20 | 100.0% | 83.9–100.0% |

**This class (b) result is weaker evidence than it looks.** All three
detectors caught every case despite the perturbation touching only 0.06%
of embedding elements, but the 134 unique tokens in the held-out corpus
include common function words that also appear in essentially every one
of the 20 detection prompts — once a patched row is read during any
prompt's forward pass, greedy decoding tends to diverge immediately and
the divergence compounds. This shows detection works when the patched
tokens are common enough to be exercised by nearly any prompt; it says
nothing about a genuinely surgical patch confined to rare or
adversarially-chosen tokens kept out of a defender's test prompts. That
narrower question is addressed directly by classes (b') and (b'') below.

### (c) Different same-shape checkpoint — found infeasible

Substituting the whole embedding table with a different, independently
trained checkpoint of matching shape (vocab_size=50256, hidden_size=2560)
was surveyed but not run: no such checkpoint was available. Every
matching-size embedding file found was byte-identical to the one already
in use (copies, not a different checkpoint). Two larger checkpoints
available have a different vocabulary size (128256, a mismatch), and two
smaller checkpoints available have a wholly different architecture
(shape [49152, 576]). Doing this class honestly would require either
sourcing an independently-trained checkpoint sharing this project's own
non-standard 50256-vocab/2560-hidden pruned architecture, or building a
token-id remapping between a differently-sized vocabulary and this one —
a materially different and larger undertaking, not attempted.

### (b') Surgical single-row +1-LSB patch, unexercised rows (n=8 rows x 5 prompts = 40 trials)

Exactly one embedding row patched per trial, by the smallest
int8-representable nudge (`+1` to the row's int8-quantized code under the
same symmetric per-tensor scale as class (a), then dequantized back to
bf16 — `elems_in_row=2560`, all 2560 elements moved). Row selection: a
documented-seed PRNG (seed 42) sampling 8 of the full 50256 vocabulary
rows without replacement and with no filtering for rarity or commonness —
rows `21349, 35299, 40914, 36468, 32866, 1014, 23725, 6404`. None of the
8 chosen rows' tokens occur in any of the 5 test prompts (confirmed by
tokenizing the prompts directly).

| detector | detected | fraction | Wilson 95% CI |
|---|---|---|---|
| A (text replay) | 0/40 | 0.0% | 0.0–8.8% |
| B (top-5 logit-index check) | 0/40 | 0.0% | 0.0–8.8% |
| C (witness digest) | 40/40 | 100.0% | 91.2–100.0% |

Every one of the 40 trials landed the same way: A PASS, B PASS, C FAIL.
A genuinely surgical, non-cherry-picked single-row patch — one row out of
50256, chosen without any knowledge of the test prompts — was invisible
to both output-based checks on every trial, and caught only by the
witness digest. This is a total split (0/40 vs. 40/40), not a partial
miss.

### (b'') Surgical single-row +1-LSB patch, exercised rows — the control (n=8 rows x 5 prompts = 40 trials)

Same harness and patch method as (b'), but the 8 rows were chosen so each
is actually read by at least one of the 5 test prompts (selected from the
union of token ids appearing in those prompts, PRNG seed 42): rows for
the tokens " L", " the", " bring", " water", " yeast", " and", " lived",
" to".

| detector | detected | fraction | Wilson 95% CI | (b') baseline, n=40 |
|---|---|---|---|---|
| A (text replay) | 11/40 | 27.5% | 16.1–42.8% | 0/40 (0.0%) |
| B (top-5 logit-index check) | 21/40 | 52.5% | 37.5–67.1% | 0/40 (0.0%) |
| C (witness digest) | 40/40 | 100.0% | 91.2–100.0% | 40/40 (100.0%) |

A/B do show a clear increase once the corrupted row is actually read
(0/40 -> 11/40 for A, 0/40 -> 21/40 for B) — this confirms (b')'s 0/40
was not an artifact of A/B being generally blind to single-row int8
patches. But the rate stays far short of near-100%: not every prompt that
contains a patched token's row in its own text reliably re-reads or
re-generates that exact position at a step where a one-quantization-step
nudge moves the argmax or the top-5 set. Detection depends on whether a
row is exercised at all, and, even then, only probabilistically on how
central that token is to the specific greedy decoding path. C caught
every trial regardless, in both (b') and (b'').

## 4. Headline finding, stated plainly

**Across classes (a), (b), and (d), all three detectors performed well —
but those substitutions were global or common-token-biased enough that
almost any prompt exercised the affected region.** The classes designed
to isolate a single, non-cherry-picked embedding row — (b') and its
exercised-row control (b'') — show the real picture: output-based checks
(A, B) detect a single-row substitution only when, and only
probabilistically even then, a given prompt's forward pass actually reads
that row (as literal input or as a token the model happens to generate).
When the row is not exercised by the test prompts, A and B miss it
completely (0/40 each in (b')). The witness digest (C) is the only
detector of the three that caught every trial in every class tested here,
because it binds the model's actual computation rather than depending on
which vocabulary positions a given prompt's decode happens to touch.

## 5. What this does NOT show

- Sample sizes are small (n=20 per global-substitution class; n=40 per
  single-row class, using a 5-of-20 prompt subset for the single-row
  classes) — these are single-checkpoint, single-prompt-set samples, not
  a general characterization of A/B's miss rate across models or prompt
  distributions.
- Class (b)'s 100%/100%/100% result is weak evidence (see above): the
  patched tokens overlapped heavily with ordinary-English vocabulary
  shared by the detection prompts. A narrower, more adversarial variant —
  patch only rare/out-of-distribution tokens deliberately excluded from
  the defender's known test-prompt set — is untested.
- (b')'s row selection was a fixed-seed PRNG over the full vocabulary,
  not an adversary deliberately choosing a row to evade a known defender
  prompt set; that harder, adversarial variant is untested.
- (b'')'s row selection was PRNG over a *subset* of the vocabulary (the
  union of tokens the 5 prompts contain), a different and looser
  selection procedure than (b')'s full-vocabulary sampling, by
  construction (the task required rows that are exercised).
- Class (c) was not run at all (found infeasible on the available
  hardware/checkpoints).
- Only the continuous bf16 embedding table was in scope for every class;
  the packed ternary attention/FFN weights were not substituted, for the
  checkpoint-layout reasons given in Setup.
- No timing numbers are reported, and none should be inferred (Rule A).

## 6. Reproduce

The harness examples used for these experiments (INT8 quantization,
magnitude pruning, approximated fine-tune, and the two surgical
single-row variants) are not yet published in a public repo. The moat-3
bitflip harness pattern they extend is described in
[2026-09-13-BITFLIP-HEADTOHEAD.md](2026-09-13-BITFLIP-HEADTOHEAD.md#6-reproduce).

## 7. Plain English

We took the same pruned BitNet-2B checkpoint and, instead of flipping a
single random bit, applied realistic deployment-style substitutions to
its embedding table: quantizing it to int8, pruning the smallest 1% of
its values, applying an approximated fine-tune patch, and surgically
editing exactly one row out of over 50,000. For the big, global changes
(quantization, pruning, the fine-tune approximation), all three
detectors — text replay, top-5 logit check, and the witness digest —
caught almost everything, because those changes were large or common
enough that any test prompt was likely to touch the affected part of the
model. But when we patched exactly one row, chosen at random with no
regard for what our test prompts happened to say, the two output-based
checks missed it every single time — 0 out of 40 trials — while the
witness digest caught it every time. Even when we deliberately chose rows
we knew our prompts would read, the output-based checks still only caught
it about a quarter to half the time, because it also depends on whether
the model's own generated text happens to land on the exact position that
changed. The honest rule: checks that only look at a model's printed
output are only as good as the prompts you happen to test with; a
detector that hashes the model's actual computation does not have that
blind spot.
