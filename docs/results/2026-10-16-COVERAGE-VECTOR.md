# Coverage vector: a constructed input on which the two E22 SAME mutants move the digest (2026-10-16)

[E22](../E22_NECESSITY_MATRIX.md) found that two single-clause mutants left the pinned section 13.1 digest unchanged: M01 (no FTZ/DAZ pin, section 1.3) and M15 (argmax `>=` in place of strict `>`, section 11.2). E22 and E38 explain both as properties of the vector: no subnormal reached a result and no tie occurred in the 16 argmaxes. This note adds a small constructed vector that reaches both branches on purpose, and records that both mutants move the digest on it on four hosts.

Rule A: no timing figures. Informative result: the pinned section 13.1 vector and all pinned digests are unchanged. The result says nothing about how often tied or subnormal logits arise in real checkpoints.

## 1. The vector

Files in [`vectors/coverage-v1/`](../../vectors/coverage-v1/README.md); the model is our own seeded random weights (seed 20261016, generator `gen_coverage_model.py`, numpy only), not a trained or third-party model.

| item | value |
|---|---|
| architecture | Llama, hidden 64, 2 heads, 2 KV heads (head_dim 32, so the section 9 scale `rsqrt(32)` is not a power of two), 2 layers, intermediate 128, vocab 32, eps 1e-5, rope_theta 10000, untied lm_head, bf16 weights |
| exact tie | lm_head rows 5 and 17 are byte-identical, equal to +-2.0 following the sign of the final hidden state, so the tied logit is the maximum at every step; the embeddings of the two tokens differ |
| subnormal logit | lm_head row 29 is bf16 `0x0001` / `0x8001` (f32 subnormal +-2^-133), sign matched to the hidden state so nothing cancels |
| prompt | `ids/coverage-4.txt` = 3,11,22,8; `gen_toks` 8 |
| `model.safetensors` sha256 | `677a5a25208b08594b0cb9375cb28ed427bdfeec72ddcd2833b0513e7031f4ca` |
| `config.json` sha256 | `af88fb256b150dad9050f760c6390ef45a9c539c4d14c0bcea4fa31b125e9695` |
| `tokenizer.json` sha256 (stub, content unused) | `e39fc11fc1c2b004fcf5f6b5a817f79562037c05a256439a8d4cdef8ecb5dd23` |
| `ids/coverage-4.txt` sha256 | `aa1bb87d585ab545985af17145a4e71a83bc1c0e563aec8ca9c2dd1c43f14853` |

Running the generator again reproduces all four files with the hashes above.

## 2. What each mutant does and why the vector exercises it

Diffs: [`m01.diff`](../../vectors/coverage-v1/mutants/m01.diff), [`m15.diff`](../../vectors/coverage-v1/mutants/m15.diff). Each is a single-file edit of the C++ host port; all other files are unchanged.

- **M01 (section 1.3).** The MXCSR FTZ/DAZ pin, its read-back assertion and the self-test are removed from `cis2_fpenv_init`. With the pin, the subnormal logit at index 29 is flushed to exactly `0x00000000`. Without it the logit stays a non-zero subnormal. In a trace of the logits at the 8 decode steps on lab host C, the M01 build differs from the reference at index 29 only, in all 8 steps, with bit patterns `0x00337564`, `0x0015efe9`, `0x00146d04`, `0x001284aa`, `0x0011d68e`, `0x0011e6f4`, `0x0012988e`, `0x0013f4eb`; the other 31 elements are bit-identical. The full logit vector is absorbed into the witness (section 12.1 item 7a), so that one element moves `CIS2_REF`. The generated ids and the argmax digest do not change, because row 29 is never the maximum.
- **M15 (section 11.2).** The argmax comparison `logits[i] > best` becomes `>=`, so the last maximal index wins instead of the lowest. Rows 5 and 17 tie for the maximum at every step. The reference picks 5; the mutant picks 17 already at decode step 0, so the ids, the argmax digest, the witness and every later step differ.

## 3. Result (8 generated tokens, same on all four hosts)

| build | `CIS2_REF` | argmax digest (prefix) | ids (prompt, then 8 generated) |
|---|---|---|---|
| reference (run twice, identical) | `4660d6c33b6c0d51529b4be43687c004c8b7c0d8804439999b5af32683528734` | `05c95d57fe4bb048` | 3,11,22,8, then 5 x 8 |
| M01 | `93ecb8384ec59629f60232aa5a184d50c38ca15cf94ee613de1e41bffe2d47f0` | `05c95d57fe4bb048` | 3,11,22,8, then 5 x 8 |
| M15 | `68dd35c5d2a969754d6fdccfda579642df43e1f4b97cc00016e32b6684b90d85` | `ffc1b1a0069025ca` | 3,11,22,8, then 17 x 8 |

Hosts (all g++ 14.2.0, `-std=c++17 -O2 -ffp-contract=off -fexcess-precision=standard -fno-fast-math`): lab host C (Core i7-7700HQ) and lab host D (Core i5-5200U), both AVX2 + FMA; lab host A (Celeron N4020) and lab host H (Celeron N4020C), both SSE4.2 only. On each host: reference digest equal to the value above, M01 MOVED, M15 MOVED, generated id 5 at step 0, repeat run SAME. The M01 and M15 digests in the table are identical on all four hosts, and the hash prefixes of weights (`677a5a25`), config (`af88fb25`) and ids (`aa1bb87d`) matched on every host.

## 4. E22 addendum

| ID | Clause | SmolLM2-135M vector (E22) | coverage vector (this note) |
|---|---|---|---|
| M01 | section 1.3 FTZ/DAZ pin + self-test | SAME | MOVED on 4 of 4 hosts |
| M15 | section 11.2 argmax tie rule | SAME | MOVED on 4 of 4 hosts |

The E22 reading stands: the SAME results were properties of the vector, not of the clauses. This vector shows that the clauses are load-bearing once an input reaches them.

## 5. Limits

- The model is constructed. It says nothing about how often exact ties or subnormal logits occur in real checkpoints.
- One implementation family (the C++ host port) ran the mutants; the host source is not part of this tree (`docs/REPRODUCTIONS.md`, note 2). No independent implementation has run this vector yet.
- The lab also wrote a numpy float32 forward pass from the spec text that matches the logits bit for bit; it is by the same author on the same data and is not included here.
