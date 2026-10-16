# Real-weight tamper study: what each receipt level catches (2026-10-16)

Single-element edits to the pinned SmolLM2-135M `model.safetensors`, decoded
under CIS-2 with the C++ reference port, checked at four levels: the artifact
hash, the `CIS2_REF` witness of section 12.1, the witness *body* (the per-step
full fp32 logits and argmax ids of section 12.1 item 7, with the header items
1 to 6 left out), and the greedy-token argmax digest of section 12.2 (which is
what a text replay compares).

Rule A: identity experiment, no timing figures. The mutant files, and which
tensors and elements were edited, are not published; the counts are.

## 1. Pre-registration

Two registrations in the public ledger (`ledger/`):

- **w1x-tamper** (batch 34). Its bar required the header-only control (one
  padding byte of the safetensors JSON header changed, data unchanged) to move
  the artifact hash and *not* move `CIS2_REF`. That clause was mis-specified:
  section 12.1 item 1 feeds the raw `weights_sha256` into the witness chain,
  so any byte change to the file moves `CIS2_REF` by construction. The bar was
  frozen at start, so w1x-tamper is scored **not met** on that clause, with
  every data-mutation clause met. Nothing was amended.
- **w1x-tamper2** (batch 35), registered before any second-run result was
  read, with the control stated correctly (artifact hash and `CIS2_REF` move;
  body and argmax digests do not), a base trial, and the body digest as a
  third level. **Met** on all three hosts.

## 2. Setup

- Artifact: `model.safetensors` sha256 `80521b40…` (the section 2.1 pin);
  prompt and decode: the section 13.1 vector, 16 greedy tokens.
- Implementation: the C++ port (`cis2_host.cpp`, sources sha256 prefix
  `10e08ea20584151f`), host build, pinned flags
  (`-O2 -ffp-contract=off -fexcess-precision=standard -fno-fast-math`). A
  patched copy prints the body digest as one extra line; the receipt
  construction is otherwise untouched.
- Hosts (labels as in `docs/REPRODUCTIONS.md`): host A (Celeron N4020,
  SSE4.2 only), host H (Celeron N4020C, SSE4.2 only), host C (i7-7700HQ,
  AVX2+FMA). Each host drew its own random mutant set.
- Per host, 54 mutants plus one control:
  - **1-ULP**: 24 mutants, one bf16 element incremented by one ULP;
  - **INT8**: 6 mutants, one whole tensor round-tripped through per-tensor
    INT8 quantisation;
  - **bit flip**: 24 mutants, one random bit of one random element flipped;
    the subset that landed in a sign or exponent bit is reported separately
    (11, 13 and 12 of the 24 on hosts A, H and C);
  - **control**: one padding byte in the JSON header changed, data offsets
    and bytes unchanged.
- Base trial: the unmutated copy must reproduce the pinned `CIS2_REF`
  (`d82743059d…`) before any mutant is read.

## 3. Results

Base trial: `d82743059d…` on all three hosts.

| level | 1-ULP (24 per host) | INT8 (6) | bit flip (24) | of which sign/exponent | all 54 | control |
|---|---|---|---|---|---|---|
| artifact hash (sha256 of the file) | 24 / 24 / 24 | 6 / 6 / 6 | 24 / 24 / 24 | 11 / 13 / 12 | 54 / 54 / 54 | moved |
| `CIS2_REF` witness (section 12.1) | 24 / 24 / 24 | 6 / 6 / 6 | 24 / 24 / 24 | 11 / 13 / 12 | 54 / 54 / 54 | moved (item 1 binds the file) |
| witness body (items 6 to 7 only) | 24 / 24 / 24 | 6 / 6 / 6 | 24 / 24 / 24 | 11 / 13 / 12 | 54 / 54 / 54 | unchanged |
| argmax digest (section 12.2), what text replay sees | 0 / 0 / 0 | 3 / 3 / 3 | 1 / 2 / 0 | 1 / 2 / 0 | 4 / 5 / 3 | unchanged |

Cells are host A / host H / host C counts of mutants whose digest at that
level differed from the base trial.

Pooled over the three hosts: the witness body moved for 162 of 162 mutants;
the argmax digest moved for 12 of 162. By class, text replay caught none of
the 72 one-ULP edits, 9 of the 18 INT8 round-trips and 3 of the 72 bit flips.

## 4. Reading

- The artifact hash and `CIS2_REF` catch everything, including the control,
  because the receipt binds the exact artifact bytes (section 12.1 item 1).
  That is by design, and erratum E-15 now says so in plain words.
- The body digest is the level that separates "the file changed" from "the
  computation changed": it moved for every data mutation and stayed put for
  the header-only control, on all three hosts.
- A check that compares generated text (or token ids) misses most
  single-element tampers: roughly 93% here (150 of 162). A one-ULP edit
  changed the greedy tokens in none of 72 trials. The full-logit witness body
  caught all 72.
- Hosts of two ISA classes agree on every body-level count. The argmax-level
  counts differ by host only because each host drew different mutants.

## 5. Scope

One model, one prompt, 16 tokens, 54 mutants per host, three hosts, one
implementation. No clean-room implementation ran these mutants. The claim is
limited to what the table shows: at the full-logit level every tested
single-element edit was caught, at the token level most were not.

Related: `docs/results/2026-09-13-BITFLIP-HEADTOHEAD.md` (BitNet-2B, bit
flips, text replay vs top-k vs witness) and
`docs/results/2026-09-17-EMBED-SUBSTITUTION-HEADTOHEAD.md`.
