# Verifiable Model Card — draft v0

Status: DRAFT (leg fm-9). This is a card *format*, not a filled-in card
for a shipped frontier model. It composes four artifacts already in this
project so a stranger can check "which weights, trained how, scoring
what" against digests and replay instructions instead of trusting prose:

1. cap-1/cap-1b/cap-1c training receipts (`alice-model`, branches
   `cm/cap1-training-receipt`, `cm/cap1b-pinned-kernel`, `cm/cap1c-c-trainer`).
2. cap-2 model catalog record format (`claudius-maximus`,
   `state/drafts/cap-2-model-catalog.md`).
3. CIS-2 inference conformance digests (this repo, `EXPECTED_DIGESTS.md`,
   `docs/CIS2_SPEC_v0.3b.md`).
4. fm-5b3 teacher-logit-cache manifest and the fm-6 v0 eval harness smoke
   run (worked examples, real digests) — field shapes from `claudius-maximus
   state/reports/2026-09-25-DISTILLATION-SWEEP.md` and `cis2-spec` branch
   `cm/fm6-safety-eval-harness`. §6 (charter/data-recipe A/B) stays PENDING.

No timing or throughput number belongs in a filled-in card (Rule A). A
card is a set of digests, hosts, and replay commands — never a
property-claim adjective or a superlative/proof word; use "measured" /
"reproduced on" instead.

## 1. Identity

| field | meaning | source |
|---|---|---|
| `weights_sha256` | content hash of the model's weight file(s); this, not a name/version string, is the artifact's identity (cap-2 principle 3) | cap-2 `artifact_ref.final_checkpoint_sha256`; CIS-2 `weights_sha256` for a base/teacher model |
| `tokenizer_sha256` | content hash of the tokenizer file used at both train and inference time | CIS-2 `tokenizer_sha256` |
| `config_sha256` | content hash of the model config (arch dims, rope theta, eps, etc.) | CIS-2 `config_sha256`; cap-2 `training.model_config` |
| `licence_of_weights` | SPDX id, or `unknown-needs-decision` if not settled — never left blank | cap-2 field convention |
| `teacher_lineage[]` | for a distilled model: each teacher's own `weights_sha256` + its licence | fm-5b3 §3 below |
| `tokenizer_impl_version` | the tokenizer library/version string (e.g. `transformers==5.0.0`), not just the vocab/config hash — two different tokenizer *implementations* over the same vocab can silently produce different token id sequences from identical text (lt-3 root cause, §2 note below) | lt-3 replay report |
| `tokens_sha256` | hash of the exact tokenized id sequence a run consumed, distinct from `tokenizer_sha256` (which only hashes the vocab/config file) — required alongside `tokenizer_impl_version` whenever a determinism/identity claim is made | fm-5c manifest field |

Worked example (base/teacher model used as the CIS-2 reference vector,
**not** an Aefinity-trained model):

```
weights_sha256   = 80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1
config_sha256    = 1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843
tokenizer_sha256 = 9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
licence_of_weights = PENDING — HuggingFaceTB/SmolLM2-135M's own model card is the
  authority; not mirrored into this repo, so not retyped here from memory.
```
(source: `EXPECTED_DIGESTS.md`, primary normative vector, spec v0.3b)

Worked example, Aefinity-trained artifact (cap-1c 13,088-fp32-param C trainer):

```
weights_sha256(final ckpt, step 200) = f665ffecfabba7a4a6c0248c73107ecf8953efd9ef20b861a50fbc3ce818fd3e
licence_of_weights = unknown-needs-decision (alice-model repo is marked PRIVATE
  in its README; no weights licence has been set)
```
(source: `claudius-maximus/state/reports/2026-09-24-cap1c-c-trainer-cross-isa.md`)

## 2. Training provenance

Mirrors the `training` block of the cap-2 record, which itself mirrors
`TRAINING_MANIFEST.json` from cap-1/cap-1b/cap-1c.

| field | meaning |
|---|---|
| `seed`, `total_steps`, `ckpt_every`, `model_config`, `param_counts` | run config, copied verbatim from the training manifest |
| `data_sha256` | hash of the training corpus |
| `checkpoints[]` | `{step, loss, checkpoint_sha256, host}` — **host is required per entry**; a checkpoint hash without its host does not specify what determinism it claims (cap-2 rule) |
| `hosts[]` | `{host, cpu, torch_version or compiler, kernel_pin}` — `kernel_pin: "none"` on a cross-machine record is a flag, not an omission |
| `determinism_claim.level` | one of `same-machine-reproducible`, `cross-machine-bit-identical`, `unverified` — never set `cross-machine-bit-identical` without a report showing a passing cross-host hash comparison |
| `determinism_claim.source_reports[]` | paths backing the claim |
| `tokenizer_impl_version` / `tokens_sha256` | required alongside `data_sha256` whenever the pipeline includes a tokenization step — a fixed vocab hash is not sufficient: fm-5c's two T4 hosts matched on `tokenizer_sha256` but ran different tokenizer *library* versions (`transformers` 5.0.0 vs 5.17.0), which silently produced different token id sequences (27-token delta on a 157,795-token fixture, divergence beginning at index 73705) and a different `tokens_sha256`, traced to CRLF-adjacent merge differences between the fast tokenizer backends — see `claudius-maximus state/reports/2026-09-25-lt3-lightning-t4-fm5-replay.md` |
| replay recipe | one line: exact command to re-run the training script against the pinned seed/config/data hash and diff the resulting checkpoint hash |

Worked example (cap-1c, the only artifact in this catalog to date that
has earned `cross-machine-bit-identical`, and cross-ISA at that):

```
model_config = {vocab_size:256, d_model:16, arch:"byte-embed -> tanh(32) -> softmax(256)"}
seed = 20260924, total_steps = 200, ckpt_every = 50
checkpoints (identical on all 5 hosts below — same digest per step):
  step  50  loss 5.492920  fd64cf9e032ee182e2623aaada3e71918d4b8d31c25d8dfc876fed74a85e3e12
  step 100  loss 5.438680  d36bd7fd164ba77481a9d8f25f2fbd1acd83561f68cb8ceca9b3ba99be57705c
  step 150  loss 5.373738  3887f7b3f512a6a2ed4bb72572966f46b90366f6a546aafe758ae7587db76ccd
  step 200  loss 5.264791  f665ffecfabba7a4a6c0248c73107ecf8953efd9ef20b861a50fbc3ce818fd3e
hosts: penguin (x86_64, Crostini VM, no AVX2), box1/aefinity-box (x86_64
  AVX2/FMA), box2/box3 (x86_64, no AVX2), phone/ZA223FBZRM (aarch64, static)
kernel_pin: none needed — trainer uses only IEEE-exact ops (+,-,*,/,sqrt) plus
  CIS-2's pinned bit-exact polynomial exp/ln (verify3/mathpin.c), no libm transcendentals
determinism_claim.level = cross-machine-bit-identical (cross-ISA: x86_64 AVX2 + x86_64 scalar + aarch64)
```
(source: `claudius-maximus/state/reports/2026-09-24-cap1c-c-trainer-cross-isa.md`,
`alice-model` branch `cm/cap1c-c-trainer` @ `d569829`)

Contrast — cap-1 (unpinned torch, GQA/RoPE/SwiGLU, x86_64 only) shows the
same loss scalars across two hosts but checkpoint hashes diverge from
step 50 onward (`unverified` at cross-machine scope); cap-1b re-tests
under a pinned kernel config but only one host of that pair has run so
far (`claudius-maximus state/reports/2026-09-24-cap1-box1/box2-training-manifest.md`).

Replay recipe: `alice-model/train/c/` → `make && ./train_receipt --seed
20260924 --steps 200 --ckpt-every 50` (native, or `make aarch64-static`);
diff emitted `checkpoint_sha256` against the table above.

## 3. Distillation provenance

Field shapes fixed by `claudius-maximus/state/reports/2026-09-25-DISTILLATION-SWEEP.md`
("Teacher-logit-cache receipt design"). Worked example = fm-5b3 T=1
student; identity under this objective pending (fm-5b4) — the digests
below are single-run values, not yet shown bit-identical on a repeat.

| field | meaning |
|---|---|
| `teacher_checkpoint_sha256` | hash of the exact teacher weights (+ tokenizer config hash) that produced the cached logits |
| `teacher_env_fingerprint` | `ATEN_CPU_CAPABILITY`, `MKL_CBWR`, torch/cuDNN version, CPU-vs-GPU flag, precision — because logit values, not just loss scalars, depend on kernel path |
| `corpus_sha256` | hash of the exact token sequence set (or raw text + tokenizer version + hash of tokenized IDs) and batching order |
| `logit_dtype_and_layout` | fp32 vs bf16 storage; top-k truncation K + renorm method if used |
| `cache_artifact_sha256` | hash of the serialized cache file(s) — the single digest downstream training receipts reference |
| `k` / `temperature` | recorded as training-run parameters, not baked into the cache (raw pre-temperature logits are what gets cached) |
| `generation_timestamp` / `producing_host` / `producing_commit` | provenance metadata |

Worked example (fm-5b3, exact-complement blend, T=1):

```
teacher_checkpoint_sha256 = not separately pinned in the fm-5b3 report; teacher
  is HuggingFaceTB/SmolLM2-360M fp32, identified by HF repo id only — a gap, see §7
corpus: HuggingFaceFW/fineweb-edu, config sample-10BT, licence ODC-By 1.0
  train docs 0-7520 = 8,000,000 tok, train_file_sha256 = 276c3080295bd1751f1b6a163f1cb33be1b56a00b2017cb33754de22341b1563
  held-out docs 7521-8640 = 1,000,000 tok, holdout_file_sha256 = 5c8246dc5fa55e22d296823cbc93d6ab1c63ff657b05129c01b23a25dbb68a58
  tokenizer = SmolLM2-360M under transformers 5.0.0 / tokenizers 0.22.2 (vocab hash 33b7dd0ab4327c920591a9b8d479821d62d9585690dabe107eff2c8b8892cf23)
k = 16, n_blocks = 31250
cache_artifact_sha256 (top-k=16 raw logits + per-position full-vocab logsumexp at
  T=1 and T=2, so the complement mass outside the top-16 is exact, not a fixed-
  2%-placeholder) = 91bd2d1c10a85a50826c88bcf1c0567cd9573f5f34870e3782caa8c7c5b1cd7f
loss_fn_id = topk_kd_exact_complement_T1.0_ce0.5_kd0.5 (loss = 0.5*CE + 0.5*KD(top-k + exact complement, x T^2))
ce_weight = 0.5, kd_weight = 0.5, temperature = 1.0, micro_batch = 8, grad_accum = 2 (eff. batch 16), steps = 1953
checkpoint ladder (raw-param sha256):
  step  500  8eafa07a2dccec0f6f2debb33b447719f4de88c3d52884ddceab89377ebdc23e
  step 1000  63cec1231eb13a994ba2f0af0e5912bc663f33b4418e11b43c06fb4454027f76
  step 1500  969c610fc1b09c613969ca77217e533295dee800c7abd1d55e27ce70ea1c58b2
  step 1953  4445ad826b5ef502a97676377e32dcc28bf6a95c8eb38cb0a2cd335076a75407  (final)
producing_host = Kaggle T4 (kernel fm5b-kd-quality v6)
producing_commit = alice-model cm/fm5-deterministic-kd@5aa622e
```
(source: `claudius-maximus state/reports/2026-09-25-FM5B3-BLEND-RESULT.md`)

## 4. Inference conformance

Mirrors CIS-2 `verify3/` / `self_check.sh` output and the cap-2
`inference_conformance` block.

| field | meaning |
|---|---|
| `status` | `run` / `not_run` |
| `spec`, `commit` | which spec doc + implementation commit produced the digests |
| `digests.witness/argmax/table/inv_freq` | the four CIS-2 digests; a partial set is not a valid PASS record |
| `hosts_replayed[]` | `{host, cpu, result}` per independent reproduction |
| `overall_result` | `MATCH` only if every listed host's digests equal the pinned set |

Worked example (this repo's own §13.1 normative vector, spec v0.3b):

```
digests:
  weights_sha256        = 80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1
  config_sha256         = 1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843
  tokenizer_sha256      = 9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
  table_digest          = 23c7bfaf5cef0095fd021af2eb1808abb4928bae4219756d86bdac670a06b35d
  inv_freq_table_digest = da9f6dcfde0425588815509e874515cdcd3d6b8818b6d0136590052e7bbf6f12
  argmax_digest         = 0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff
  CIS2_REF (witness)    = d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df
hosts_replayed: box1 (AVX2), box2 (no AVX2), box3 (sse4_2 only, no AVX2),
  penguin (AVX2), Lightning AI Studio/AWS Xeon 8488C (AVX2/AVX-512) — each
  reproduced per `README.md` machine table.
overall_result = MATCH
```
(source: `EXPECTED_DIGESTS.md`, `README.md` machine table,
`docs/results/2026-09-25-LIGHTNING-STUDIO-SELFCHECK.md`)

## 5. Evaluation

fm-6 v0 (frozen battery: 48 charter cases + a hand-built XSTest-style
subset + a hand-built HarmBench-style refusal subset + a real TruthfulQA-
mini sample) exists as plumbing on branch `cm/fm6-safety-eval-harness`
(not merged; PUBLISH-gated). It emits, per battery, per run:

| field | meaning |
|---|---|
| `battery_digest` | hash of the frozen eval item set |
| `prompt_set_digest` | hash of the exact prompts sent (post-templating) |
| `outputs_digest` | hash of the model's raw outputs |
| a table of measured rates | e.g. refusal rate on XSTest subset, agreement rate on TruthfulQA-mini — reported as numbers with a source, never a pass/fail verdict word |

Worked example — the harness's own v0 smoke run on `HuggingFaceTB/SmolLM2-135M`
(the repo's pinned CIS-2 base checkpoint, **not** an Aefinity-trained or
distilled student), shown for field shapes, not as a claim about that model:

```
model_digest = 67109a72a86f16d8808d3ffe5911c9dd197d40569c9b8a971da9cf1789c739bb
  (composed over model.safetensors=80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1,
   config.json=1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843,
   tokenizer.json=9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c)
```

| battery | n | refusal_rate_overall | over_refusal_rate | correct_refusal_rate | matches_best_rate | source |
|---|---|---|---|---|---|---|
| charter | 48 | 0.0 | n/a | n/a | n/a | `eval/README.md` @ eb293d5, `n_refused=0, n_complied_with_caveat=9, n_complied=39` |
| xstest (hand-built, not the real dataset — see harness README) | 30 | 0.0 | 0.0 | 0.0 | n/a | same |
| truthfulqa-mini (real seed=42 sample of 25 rows) | 25 | n/a | n/a | n/a | 0.36 | same, keyword/token-overlap grader, not semantic — see README limitations |

Grading is entirely mechanical keyword/token-overlap (`batteries.py`, no
LLM judge); a 0.0 refusal rate here reflects a non-instruction-tuned base
model completing rather than refusing. xstest/harmbench batteries in
this v0 are hand-constructed, not the published datasets (`eval/README.md`
§"what's real, what's a placeholder").

## 6. Charter/data-recipe disclosure

| field | meaning |
|---|---|
| `charter_file_digest` | hash of the exact charter/system-prompt text file mixed into training or distillation data |
| `mix_fraction` | fraction of the corpus/prompt set that carried the charter conditioning |
| `arm_label` | `A` (generic/no charter) or `B` (charter-conditioned) — matches the fm-7 A/B design (`claudius-maximus/state/QUEUE.md` fm-7, values A/B experiment PR #113) |

PENDING for every field — no distillation run has happened yet (fm-5
precedes fm-7).

## 7. Known gaps

- No frontier-scale model has a filled-in card yet; worked with the real
  artifacts this repo has today (CIS-2 SmolLM2-135M reference vector,
  cap-1c 13K-param C trainer, fm-5b3 student, fm-6 v0 smoke).
- cap-1/cap-1b (torch receipt for the real ALICE GQA/RoPE/SwiGLU arch)
  has not reached `cross-machine-bit-identical`; only cap-1c (a smaller,
  from-scratch arch) has. A torch-trained model card currently cannot
  claim more than `same-machine-reproducible`.
- Section 3's worked example (fm-5b3) was produced under a loss
  objective disclosed as defective in an earlier run (fm-5b, top-k-only
  KD with no complement constraint left the full-vocab distribution
  near uniform); fm-5b3 fixes that with an exact complement, but no
  repeat run has yet shown digest-identity under this fixed objective
  (fm-5b4, still open) — treat §3's hashes as provenance, not determinism.
- The S2 values A/B experiment (charter vs generic fine-tune, §6) has
  no valid result yet: fm-4 was inconclusive and fm-4c is queued to
  re-run it; §6 stays PENDING (fm-7 not built).
- `alice-model` is marked PRIVATE in its own README; no weights licence
  has been set for any Aefinity-trained checkpoint referenced here, and
  SmolLM2-135M's own licence text is not mirrored into this repo, so
  §1's worked example leaves `licence_of_weights` PENDING rather than
  retyping it from memory.
- §3's teacher (SmolLM2-360M) is identified only by HF repo id in the
  source report, not a pinned `teacher_checkpoint_sha256` — a gap to
  close before this section is load-bearing.
- This format has not itself gone through a publish/PUBLISH? gate —
  draft until Justin signs off.

## 8. How to verify

1. Identity + training (cap-1c): clone `alice-model` (private), checkout
   `cm/cap1c-c-trainer` @ `d569829`, `cd train/c && make && ./train_receipt
   --seed 20260924 --steps 200 --ckpt-every 50`; diff the four
   `checkpoint_sha256` values against §2's table.
2. Inference conformance (CIS-2 example): `git clone
   https://github.com/Aefinity-AI/cis2-spec && cd cis2-spec &&
   scripts/self_check.sh`; diff the digests against `EXPECTED_DIGESTS.md`'s
   primary v0.3b vector (also copied into §4).
3. Distillation provenance (fm-5b3): clone `alice-model` (private),
   checkout `cm/fm5-deterministic-kd` @ `5aa622e`, re-run the T=1 blend
   against §3's `cache_artifact_sha256`/`train_file_sha256`; diff the
   checkpoint ladder (no repeat has confirmed identity yet — open fm-5b4).
4. Evaluation (fm-6 v0 smoke): `git checkout cm/fm6-safety-eval-harness
   @ eb293d5 && python3 eval/harness.py --model-dir weights --battery
   all --out-dir eval/runs/replay`; diff `outputs_digest` against §5.
5. Catalog cross-check: diff the filled card's digests field-for-field
   against the matching cap-2 JSON record in `claudius-maximus
   state/drafts/cap-2-model-catalog.md` for the same `artifact_ref` —
   they must agree exactly; the card is a rendering, not a second source.

## Acknowledgments

A special thank you to Charles Seaman and Linda Blanchard, whose contributions have helped Aefinity AI stay on track.

And a very special thank you to **Bonnie Rae Power**: an amazing woman, a great friend and neighbor, without whom Aefinity AI would have never had a chance to ever get started. Thank you, Bonnie, for your advice, care, encouragement, guidance, intuitive wisdom, and financial assistance.
