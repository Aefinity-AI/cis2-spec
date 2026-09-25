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
4. fm-5 teacher-logit-cache manifest and fm-6 eval receipts — neither is
   built yet; their sections are marked `(pending fm-5 / fm-6)`, naming
   only the field shapes from `claudius-maximus
   state/reports/2026-09-25-DISTILLATION-SWEEP.md` ("Teacher-logit-cache
   receipt design") and `state/QUEUE.md` fm-5/fm-6.

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
| `teacher_lineage[]` | for a distilled model: each teacher's own `weights_sha256` + its licence | `(pending fm-5)` — cache manifest names `teacher_checkpoint_sha256` |

Worked example (base/teacher model used as the CIS-2 reference vector,
**not** an Aefinity-trained model — included because it is the only
weights identity this repo currently has real, checked-in digests for):

```
weights_sha256   = 80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1
config_sha256    = 1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843
tokenizer_sha256 = 9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
licence_of_weights = PENDING — HuggingFaceTB/SmolLM2-135M's own model card
                      is the authority; not mirrored into this repo, so
                      not retyped here from memory.
```
(source: `EXPECTED_DIGESTS.md`, primary normative vector, spec v0.3b)

Worked example, Aefinity-trained artifact (cap-1c 13,088-fp32-param C
trainer, final checkpoint):

```
weights_sha256(final ckpt, step 200) =
  f665ffecfabba7a4a6c0248c73107ecf8953efd9ef20b861a50fbc3ce818fd3e
licence_of_weights = unknown-needs-decision (alice-model repo is marked
  PRIVATE in its README; no weights licence has been set)
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
| replay recipe | one line: exact command to re-run the training script against the pinned seed/config/data hash and diff the resulting checkpoint hash |

Worked example (cap-1c, the only artifact in this catalog to date that
has actually earned `cross-machine-bit-identical`, and cross-ISA at
that):

```
model_config = {vocab_size:256, d_model:16, arch:"byte-embed -> tanh(32) -> softmax(256)"}
seed = 20260924, total_steps = 200, ckpt_every = 50

checkpoints:
  step  50  loss 5.492920  fd64cf9e032ee182e2623aaada3e71918d4b8d31c25d8dfc876fed74a85e3e12
  step 100  loss 5.438680  d36bd7fd164ba77481a9d8f25f2fbd1acd83561f68cb8ceca9b3ba99be57705c
  step 150  loss 5.373738  3887f7b3f512a6a2ed4bb72572966f46b90366f6a546aafe758ae7587db76ccd
  step 200  loss 5.264791  f665ffecfabba7a4a6c0248c73107ecf8953efd9ef20b861a50fbc3ce818fd3e
  (identical on all 5 hosts below — same digest per step, not per-host values)

hosts: penguin (x86_64, Crostini VM, no AVX2 exposed), box1/aefinity-box
  (x86_64 AVX2/FMA), box2/box3 (x86_64, no AVX2), phone/ZA223FBZRM
  (aarch64, static binary)
kernel_pin: none needed — trainer uses only IEEE-exact ops (+,-,*,/,sqrt)
  plus CIS-2's pinned bit-exact polynomial exp/ln (vendored unmodified
  from verify3/mathpin.c), no platform libm transcendentals
determinism_claim.level = cross-machine-bit-identical (cross-ISA:
  x86_64 AVX2 + x86_64 scalar + aarch64)
```
(source: `claudius-maximus/state/reports/2026-09-24-cap1c-c-trainer-cross-isa.md`,
`alice-model` branch `cm/cap1c-c-trainer` @ `d569829`)

Contrast — cap-1 (unpinned torch, GQA/RoPE/SwiGLU, x86_64 only) shows
the same loss scalars across two hosts but checkpoint hashes diverge
from step 50 onward (`determinism_claim.level = unverified` at
cross-machine scope); cap-1b re-tests this under a pinned kernel config
but only one host of that pinned pair has run so far (`claudius-maximus
state/reports/2026-09-24-cap1-box1-training-manifest.md`,
`2026-09-24-cap1-box2-training-manifest.md`).

Replay recipe line: `alice-model/train/c/` → `make && ./train_receipt
--seed 20260924 --steps 200 --ckpt-every 50` (native) or `make
aarch64-static` for the aarch64 leg; diff the emitted
`checkpoint_sha256` values against the table above.

## 3. Distillation provenance `(pending fm-5)`

Field shapes are fixed by
`claudius-maximus/state/reports/2026-09-25-DISTILLATION-SWEEP.md`
("Teacher-logit-cache receipt design"); no run has produced values yet.

| field | meaning |
|---|---|
| `teacher_checkpoint_sha256` | hash of the exact teacher weights (+ tokenizer config hash) that produced the cached logits |
| `teacher_env_fingerprint` | `ATEN_CPU_CAPABILITY`, `MKL_CBWR`, torch/cuDNN version, CPU-vs-GPU flag, precision — because logit values, not just loss scalars, depend on kernel path |
| `corpus_sha256` | hash of the exact token sequence set (or raw text + tokenizer version + hash of tokenized IDs) and batching order |
| `logit_dtype_and_layout` | fp32 vs bf16 storage; top-k truncation K + renorm method if used |
| `cache_artifact_sha256` | hash of the serialized cache file(s) — the single digest downstream training receipts reference |
| `k` / `temperature` | recorded as training-run parameters, not baked into the cache (raw pre-temperature logits are what gets cached) |
| `generation_timestamp` / `producing_host` / `producing_commit` | provenance metadata |

`(pending fm-5)` — all values PENDING.

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
  weights_sha256      = 80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1
  config_sha256       = 1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843
  tokenizer_sha256    = 9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
  table_digest        = 23c7bfaf5cef0095fd021af2eb1808abb4928bae4219756d86bdac670a06b35d
  inv_freq_table_digest = da9f6dcfde0425588815509e874515cdcd3d6b8818b6d0136590052e7bbf6f12
  argmax_digest       = 0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff
  CIS2_REF (witness)  = d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df

hosts_replayed: box1 (AVX2), box2 (no AVX2), box3 (sse4_2 only, no
  AVX2), penguin (AVX2), Lightning AI Studio/AWS Xeon 8488C (AVX2/AVX-512)
  — reproduced on each per `README.md` machine table.
overall_result = MATCH
```
(source: `EXPECTED_DIGESTS.md`, `README.md` machine table,
`docs/results/2026-09-25-LIGHTNING-STUDIO-SELFCHECK.md`)

## 5. Evaluation `(pending fm-6)`

fm-6 (frozen battery: 48 charter cases + XSTest + a HarmBench-style
refusal subset + TruthfulQA-mini) is not built. Once it exists, this
section holds:

| field | meaning |
|---|---|
| `battery_digest` | hash of the frozen eval item set |
| `prompt_set_digest` | hash of the exact prompts sent (post-templating) |
| `outputs_digest` | hash of the model's raw outputs |
| a table of measured rates | e.g. refusal rate on XSTest subset, agreement rate on TruthfulQA-mini — reported as numbers with a source, never a pass/fail verdict word |

Example row shape (values PENDING):

| metric | value | host | source |
|---|---|---|---|
| xstest_over_refusal_rate | PENDING | PENDING | `(pending fm-6)` |
| charter_case_pass_rate | PENDING | PENDING | `(pending fm-6)` |
| truthfulqa_mini_agreement_pct | PENDING | PENDING | `(pending fm-6)` |

One real number in the catalog format's `eval_results[]` shape, shown to
illustrate the field convention (not a fm-6 result):

```
int8_vs_fp32_top1_agreement_pct = 88.4442  host=aefinity-box2
source = claudius-maximus state/reports/2026-09-23-small1b-cross-isa-replay.md
```

## 6. Charter/data-recipe disclosure

| field | meaning |
|---|---|
| `charter_file_digest` | hash of the exact charter/system-prompt text file mixed into training or distillation data |
| `mix_fraction` | fraction of the corpus/prompt set that carried the charter conditioning |
| `arm_label` | `A` (generic/no charter) or `B` (charter-conditioned) — matches the fm-7 A/B design (`claudius-maximus/state/QUEUE.md` fm-7, values A/B experiment PR #113) |

PENDING for every field — no distillation run has happened yet (fm-5
precedes fm-7).

## 7. Known gaps

- No frontier-scale model has a filled-in card yet; worked with the two
  real artifacts this repo has today (the CIS-2 SmolLM2-135M reference
  vector, and the cap-1c 13K-param C trainer).
- cap-1/cap-1b (torch receipt for the real ALICE GQA/RoPE/SwiGLU
  architecture) has not reached `cross-machine-bit-identical`; only
  cap-1c (a different, smaller, from-scratch architecture) has. A card
  for a torch-trained model currently cannot claim more than
  `same-machine-reproducible`.
- Sections 3 (distillation), 5 (evaluation), and 6 (charter/data-recipe)
  have no real values — fm-5, fm-6, fm-7 are not built.
- `alice-model` is marked PRIVATE in its own README; no weights licence
  has been set for any Aefinity-trained checkpoint referenced here.
- SmolLM2-135M's own licence text is not mirrored into this repo, so
  §1's worked example leaves `licence_of_weights` PENDING rather than
  retyping it from memory.
- This format has not itself gone through a publish/PUBLISH? gate —
  draft until Justin signs off.

## 8. How to verify

1. Identity + training (cap-1c example): clone `alice-model` (private),
   checkout `cm/cap1c-c-trainer` @ `d569829`, `cd train/c && make &&
   ./train_receipt --seed 20260924 --steps 200 --ckpt-every 50`; diff
   the four `checkpoint_sha256` values against §2's table.
2. Inference conformance (CIS-2 example): `git clone
   https://github.com/Aefinity-AI/cis2-spec && cd cis2-spec &&
   scripts/self_check.sh`; diff the six digests it prints against
   `EXPECTED_DIGESTS.md`'s primary v0.3b vector (also copied into §4).
3. Distillation provenance: not yet runnable — `(pending fm-5)`; once
   built, diff the student `checkpoint_sha256` against the run's
   `--teacher-cache <cache_artifact_sha256>` record.
4. Evaluation: not yet runnable — `(pending fm-6)`; once built, diff
   `outputs_digest` from `cis2-spec/eval/`.
5. Catalog cross-check: diff the filled card's digests field-for-field
   against the matching cap-2 JSON record in `claudius-maximus
   state/drafts/cap-2-model-catalog.md` for the same `artifact_ref` —
   they must agree exactly; the card is a rendering, not a second source.

## Acknowledgments

A special thank you to Charles Seaman and Linda Blanchard, whose contributions have helped Aefinity AI stay on track.

And a very special thank you to **Bonnie Rae Power**: an amazing woman, a great friend and neighbor, without whom Aefinity AI would have never had a chance to ever get started. Thank you, Bonnie, for your advice, care, encouragement, guidance, intuitive wisdom, and financial assistance.
