# Single-weight corruption: text replay vs top-k logit check vs witness digest

Two bitflip experiments against a pruned BitNet-2B checkpoint, comparing
three independent ways of asking "did the model's weights get corrupted?":
a text/argmax replay check, a coarser top-k-style logit/receipt check, and
the CIS witness digest (full-logit hash).

Rule A: this is an identity/correctness experiment. No timing figures are
reported or implied anywhere below.

## 1. Question

If a single bit in a deployed model's weights is corrupted, can it be
detected by:

- **replaying the generated text** and comparing it to a known-good
  baseline, or
- **replaying only the top-k candidate logits** at each step (a cheaper,
  coarser statistical check), or

only by an independent **cryptographic witness digest** over the model's
actual internal computation (CIS)?

## 2. Setup

**Model:** pruned BitNet-2B checkpoint, tied embeddings, ternary weight
matrices packed as U8 (2 bits/weight), f32 per-tensor dequantization
scales, bf16 embedding table.
<!-- 2026-09-13-moat3-headtohead-box1.md lines 13-16 -->

**Prompt / decode:** `'Once upon a time'`, 64 new tokens, greedy decode.
<!-- 2026-09-13-moat3-headtohead-box1.md line 18 -->

**Byte -> trit LUT** (how a "trit change" is made in the TRIT_* classes):
2 bits per weight, LSB-first per byte; `00 = 0.0`, `01 = +1.0`, `10 =
-1.0`, `11 = undefined (decodes to 0.0)`.
<!-- raw log header line 2, /tmp/.../scratchpad/moat3b.log -->

**The three checks, as actually implemented (the two experiments define
A/B slightly differently; both are reported below rather than blurred
into one):**

- Experiment 1 ("moat-3", 50 trials): **A** — text/argmax replay match to
  baseline; **B** — top-5 logit-set index match to baseline at every
  decode step (statistical-replay style, coarser than an exact digest,
  finer than plain argmax); **C** — full witness-chain digest match
  (`aegis_core::cis_infer::CisEngine` / `aegis_core::witness::WitnessChain`,
  same code paths as production).
  <!-- 2026-09-13-moat3-headtohead-box1.md lines 23-25 -->
- Experiment 2 ("moat-3b", 295 trials, stratified by tensor class):
  **A** — inference ran to completion without erroring; **B** — the
  (cheaper) receipt/verifier accepted the run; **C** — output actually
  matches the uncorrupted baseline (digest-equivalent ground truth).
  <!-- 2026-09-13-moat3b-stratified-box1.md lines 6-11 -->

**Perturbation classes (moat-3b):** CONTROL (no-op sanity), EMBED (bf16
embedding-table bytes), SCALE_ATTN / SCALE_FFN (f32 per-tensor dequant
scales in attention vs. FFN tensors), TRIT_ATTN / TRIT_FFN (U8-packed
ternary weight bytes in attention vs. FFN tensors, using the LUT above).
<!-- 2026-09-13-moat3b-stratified-box1.md lines 13-22 -->

**Hosts (identity evidence only, no timing claim):** both experiments ran
on host `aefinity-box`; results were independently recounted on a second
host from the raw per-trial logs rather than trusting the harness's own
summary line.
<!-- 2026-09-13-moat3-verify-box2.md line 3; 2026-09-13-moat3b-verify.md lines 6-7, 63-65 -->

## 3. Results

### Experiment 1 — moat-3 (50 trials: 5 no-flip controls, 40 random
low-mantissa-bit flips (bits 0-4), 5 forced high-bit flips (bit 7))

| metric | count / 50 |
|---|---|
| A pass AND C fail (text replay misses a corruption the digest catches) | 3 |
| B pass AND C fail (top-5 replay misses a corruption the digest catches) | 0 |
<!-- 2026-09-13-moat3-headtohead-box1.md lines 34-39; recount CONFIRMED in 2026-09-13-moat3-verify-box2.md lines 14-24 -->

All 5 no-flip controls: PASS/PASS/PASS. All 5 forced high-bit controls
(trials 6-10): FAIL/FAIL/FAIL on all three checks — every method caught
every high-bit flip.
<!-- 2026-09-13-moat3-verify-box2.md lines 26-35 -->

The 3 A-pass/C-fail trials (16, 25, 47) also failed B in this run, so
B-pass-AND-C-fail happened to be 0 at this sample size — a fact about this
50-trial run, not a claim that no such case can exist.
<!-- 2026-09-13-moat3-headtohead-box1.md lines 57-61 -->

Correction from the independent recount: the original report used trial
47 as "the" high-bit example; trial 47 is actually a bit-7 flip in the
ordinary random-trial set, not one of the 5 dedicated `control-highbit`
trials (6-10). The 5 control-highbit trials are correctly counted and are
FAIL/FAIL/FAIL throughout; trial 47 is a separate, correctly-reported
A-pass/C-fail case.
<!-- 2026-09-13-moat3-verify-box2.md lines 37-54 -->

### Experiment 2 — moat-3b (295 trials, stratified by class + a
bit-position sweep, bits 0 and 3-5)

| class | n | C-fail | A-pass & C-fail | B-pass & C-fail | Wilson 95% CI (A-miss rate, of C-fail) | Wilson 95% CI (B-miss rate, of C-fail) |
|---|---|---|---|---|---|---|
| CONTROL | 5 | 0 | 0 | 0 | n/a (no C-fail trials) | n/a |
| EMBED | 80 | 80 | 80 | 80 | [0.954, 1.000] (80/80) | [0.954, 1.000] (80/80) |
| SCALE_ATTN | 70 | 70 | 47 | 1 | [0.555, 0.770] (47/70) | [0.003, 0.077] (1/70) |
| SCALE_FFN | 70 | 33 | 18 | 0 | [0.380, 0.702] (18/33) | [0.000, 0.104] (0/33) |
| TRIT_ATTN | 35 | 35 | 18 | 1 | [0.356, 0.670] (18/35) | [0.005, 0.145] (1/35) |
| TRIT_FFN | 35 | 29 | 19 | 0 | [0.473, 0.801] (19/29) | [0.000, 0.117] (0/29) |
<!-- per-class n/C-fail/A-pass&C-fail/B-pass&C-fail numbers: task-provided recount, verified against raw log in
     /tmp/claude-1000/-home-justinbrianthompson/0ff8b692-930e-43f1-a192-481560f57ba0/scratchpad/moat3b.log
     (381 lines) and confirmed matching 2026-09-13-moat3b-verify.md lines 9-16; Wilson CIs computed for this doc,
     95% (z=1.96), denominator = C-fail (logit-changing trials) per class -->

Aggregate miss rate among all 247 logit-changing trials (C-fail, i.e. the
corruption actually changed model output) across the moat-3b run:

| check | misses / logit-changing trials | Wilson 95% CI |
|---|---|---|
| text/argmax replay (A) | 182 / 247 | [0.679, 0.788] |
| top-k / receipt-style check (B) | 82 / 247 | [0.276, 0.393] |
| CIS witness digest (C) | 0 / 247 | [0.000, 0.015] |
<!-- counts recomputed by awk over the raw per-trial log (295 rows, 247 C-fail rows) and cross-checked
     against the per-class table above (sums: A-pass&C-fail 80+47+18+18+19=182, B-pass&C-fail 80+1+0+1+0=82);
     Wilson CIs computed for this doc, 95% (z=1.96) -->

The digest's 0/247 is exact by construction (C is defined as "does the
digest match"; a digest miss and a C-fail are the same event here). It is
included as the reference row: the informative comparison is how often
the *other two* checks — which measure something weaker, printed text or
a truncated logit set/receipt — fail to notice what the digest does.

**One false-accept case each for TRIT_ATTN and SCALE_ATTN:** main-run
trial #15 (TRIT_ATTN, `model.layers.28.self_attn.o_proj.weight`, byte
552843, bit 0) and sweep trial #29 (SCALE_ATTN,
`model.layers.29.self_attn.o_proj.weight_scale`, byte 0, bit 5) are the 2
receipt-verifier false-accepts (B pass, C fail) outside the always-silent
EMBED class.
<!-- 2026-09-13-moat3b-verify.md lines 42-45 -->

### Experiment 1 vs. Experiment 2 — negatives that must stay attached

- Experiment 1's 40 "random" trials were low-mantissa-bit flips only (bits
  0-4), not the full byte space.
  <!-- 2026-09-13-moat3-headtohead-box1.md line 27 -->
- Experiment 1's B-pass/C-fail = 0 is a 50-trial sample fact, not a
  universal claim; a larger run is the natural follow-up.
  <!-- 2026-09-13-moat3-headtohead-box1.md lines 59-61 -->
- Experiment 2's "B" (receipt-verifies) is not the same check as
  Experiment 1's "B" (top-5 logit-set replay); reported separately, not
  merged into one number.
  <!-- 2026-09-13-moat3b-stratified-box1.md lines 6-11 vs 2026-09-13-moat3-headtohead-box1.md lines 23-24 -->

## 4. The 43 no-effect perturbations

Not every corrupted bit changes the output. In moat-3b, SCALE_FFN had 37
of 70 trials (allPASS) and TRIT_FFN had 6 of 35 trials (allPASS) where the
flipped bit produced a bit-identical witness digest to the clean baseline
— the corruption had no detectable effect at all, on any of the three
checks.
<!-- 2026-09-13-moat3b-verify.md lines 23-33 (SCALE_FFN allPASS=37, TRIT_FFN allPASS=6) -->

This is expected, not a gap: a flipped low-mantissa bit in an f32 scale
value can be numerically negligible after dequantization and rounding
downstream, and a flipped bit in a 2-bits/weight ternary-packed byte can
decode to the *same* trit under the LUT above (or to the reserved `11 =
undefined -> 0.0` encoding, coinciding with the correct value). In both
cases the actual computation is unchanged, so a digest over that
computation correctly reports "no change" — it measures the arithmetic
that ran, not the raw file bytes. A byte-level integrity check would flag
these 43 cases; the CIS digest, by design, does not, because it attests
to computation, not storage.

## 5. What this does NOT show

- **Embedding-row-unused caveat.** EMBED C-fail was 80/80 (every
  corrupted embedding byte changed the output) for the fixed test prompt.
  This is consistent with the prompt's tokens landing on the corrupted
  rows, not evidence that the digest catches corruption of an embedding
  row *never read* by a given prompt — an integrity question (did the
  byte change) distinct from the behavioral one measured here (did the
  output change). This experiment does not test the unused-row case.
  <!-- 2026-09-13-moat3b-verify.md lines 47-58 -->
- Single fixed prompt (`'Once upon a time'`, 64 tokens), single model
  (pruned BitNet-2B), single perturbation mechanism (single-bit flips
  chosen by the harness, not an adversary).
  <!-- 2026-09-13-moat3-headtohead-box1.md line 18; 2026-09-13-moat3b-stratified-box1.md line 3 -->
- No claim about **adversarial (optimized) tampering** — every flip is a
  random or LUT-targeted single bit, not one chosen by an attacker
  searching for a flip that evades all three checks at once.
- No timing numbers are reported, and none should be inferred (Rule A,
  both experiments).
  <!-- 2026-09-13-moat3-headtohead-box1.md lines 8-9; raw log header line 3 -->
- Both experiments used a single hardware family for generation
  (`aefinity-box`); cross-checking was recomputation from the raw log on
  a second host, not re-running inference on different hardware.
  <!-- 2026-09-13-moat3-verify-box2.md line 3; 2026-09-13-moat3b-verify.md lines 6-7 -->
- The top-k-style check misses far fewer corruptions than plain text
  replay (82/247 vs. 182/247) but is still far from catching everything
  the digest catches (0/247).

## 6. Reproduce

Harness (moat-3, 50 trials):
```
cd aegis-linux && cargo build --release --example cis_bitflip_moat3
M=/home/cm/aefinity-artifacts/bitnet2b-2b-artifacts
nice -n 10 ./aegis-linux/target/release/examples/cis_bitflip_moat3 \
  "$M/aegis_pruned_model.cis.safetensors" "$M/embed.bin" "$M/vocab.bin" \
  tools/moat3/f32_tensors.csv 64 'Once upon a time' 40 5 5 20260913 \
  > docs/hardware_logs/moat3_bitflip_headtohead_bitnet2b_aefinity-box_2026-09-13.log 2>&1
```
alice-aegis branch `cm/moat3-headtohead`, commit `14ef1a1`.
<!-- 2026-09-13-moat3-headtohead-box1.md lines 71-83 -->

Harness (moat-3b, 295 trials): alice-aegis branch `cm/moat3b-stratified`,
commit `1567ee1`, log
`docs/hardware_logs/moat3b_stratified_bitnet2b_aefinity-box_2026-09-13.log`.
<!-- 2026-09-13-moat3b-stratified-box1.md lines 3-4; commit confirmed on origin in 2026-09-13-moat3b-verify.md lines 63-65 -->

Pinned model-artifact digests (moat-3 run; full hashes in the log):
- MODEL `facb3597...5868db`
- EMBED `e32b99a2...4d364e077`
- VOCAB `5bde1b03...c2d97d9ae`
<!-- 2026-09-13-moat3-headtohead-box1.md lines 85-89 -->

Both harnesses restore every flipped byte and assert
`model_bytes_unchanged_after_run=true` (moat-3b also asserts
`embed_bytes_unchanged_after_run=true`).
<!-- 2026-09-13-moat3-headtohead-box1.md line 20; raw log lines 15-16 -->

## 7. Plain English

We took a large language model and flipped exactly one bit somewhere in
its weights — sometimes in a scale number, sometimes in a packed ternary
weight, sometimes in an embedding — and asked three different tampering
detectors "did anything change?" A basic check that just compares the
printed text missed the tampering in 3 out of 50 single-bit-flip trials in
one experiment, and in 182 out of 247 cases in a larger, stratified
experiment where the model's actual output had genuinely changed. A
cheaper "check the top handful of likely next words, or accept a
lightweight receipt" approach did better but still missed 82 of those 247
real changes. Our cryptographic witness — which hashes the model's actual
internal computation rather than its final printed words — caught every
single one, 247 out of 247. It also correctly stayed silent on 43 other
flips that genuinely made no difference to the math (the equivalent of
flipping a bit in a number that rounds to the same value either way) —
which is a feature, not a miss, since the digest is attesting to the
computation that ran, not to the raw bytes on disk. What this does not
show: we did not test an attacker deliberately searching for a bit flip
that fools all three checks at once, we ran one prompt on one model on
one machine, and none of this is a speed claim.
