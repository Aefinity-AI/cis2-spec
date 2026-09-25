# Cross-host deterministic knowledge distillation: two GPU hosts, identical student checkpoints — 2026-09-25

Rule A: this is an identity/correctness experiment. No timing or
throughput figures are reported or implied anywhere below.

## 1. Question and setting

Does a cached-teacher-logit knowledge-distillation (KD) training run
produce a bit-identical student checkpoint when repeated on two
different GPU hosts?

- Teacher: `HuggingFaceTB/SmolLM2-360M` (Apache-2.0), fp32, eval mode;
  top-k=64 raw logits cached once.
- Student: a random-init, Llama-config, 135M-parameter-class model
  (hidden=576, layers=30, heads=9, kv_heads=3, intermediate=1536,
  vocab=49152), seed `20260925`.
- Training: 200 steps, seq_len=256, k=64, temperature=2.0, lr=0.0003,
  weight_decay=0.01, `ce_weight=0.0`, loss `forward_kl_T2.0`,
  `data_order_seed=20260925`, no shuffle.
- Fixture: a pre-tokenized, public-domain corpus — Project Gutenberg
  #1661, tokenized once with the teacher's tokenizer and shipped as a
  fixed token-id tensor (616 blocks x 256 tokens) rather than re-run
  through the tokenizer on each host.
- Determinism settings: fp32 only, no AMP/dropout,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
  `torch.use_deterministic_algorithms(True)`, no TF32,
  `cudnn.benchmark=False`.

## 2. Digest table

Same-host repeats (run1 vs run2) and the two hosts are compared side by
side. Digests are full 64-hex sha256, copied verbatim from the source
manifests/logs.

| field | Kaggle T4 run1 | Kaggle T4 run2 | Lightning T4 (AWS) run1 | Lightning T4 (AWS) run2 | Kaggle T4 (kernel v5, re-check) | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| tokens_sha256 | `140e9dac6ad6380dd9cee667ec6e9beffec1297ec2e5cc344fcd33efc31b72a9` | `140e9dac6ad6380dd9cee667ec6e9beffec1297ec2e5cc344fcd33efc31b72a9` | `140e9dac6ad6380dd9cee667ec6e9beffec1297ec2e5cc344fcd33efc31b72a9` | `140e9dac6ad6380dd9cee667ec6e9beffec1297ec2e5cc344fcd33efc31b72a9` | `140e9dac6ad6380dd9cee667ec6e9beffec1297ec2e5cc344fcd33efc31b72a9` | MATCH |
| token_ids_file_sha256 | `f50c84eab13010991dd2a3d633b2e68937afebc6a8c0b68d48f5d4d1a14feb19` | `f50c84eab13010991dd2a3d633b2e68937afebc6a8c0b68d48f5d4d1a14feb19` | `f50c84eab13010991dd2a3d633b2e68937afebc6a8c0b68d48f5d4d1a14feb19` | `f50c84eab13010991dd2a3d633b2e68937afebc6a8c0b68d48f5d4d1a14feb19` | `f50c84eab13010991dd2a3d633b2e68937afebc6a8c0b68d48f5d4d1a14feb19` | MATCH |
| tokenizer_hash | `33b7dd0ab4327c920591a9b8d479821d62d9585690dabe107eff2c8b8892cf23` | `33b7dd0ab4327c920591a9b8d479821d62d9585690dabe107eff2c8b8892cf23` | `33b7dd0ab4327c920591a9b8d479821d62d9585690dabe107eff2c8b8892cf23` | `33b7dd0ab4327c920591a9b8d479821d62d9585690dabe107eff2c8b8892cf23` | `33b7dd0ab4327c920591a9b8d479821d62d9585690dabe107eff2c8b8892cf23` | MATCH |
| teacher_logit_cache_sha256 | `493f042d59535cc7b68baeadd4d3f81d4112aa3341260af682ae3bc0a4ea4d23` | `493f042d59535cc7b68baeadd4d3f81d4112aa3341260af682ae3bc0a4ea4d23` | `493f042d59535cc7b68baeadd4d3f81d4112aa3341260af682ae3bc0a4ea4d23` | `493f042d59535cc7b68baeadd4d3f81d4112aa3341260af682ae3bc0a4ea4d23` | `493f042d59535cc7b68baeadd4d3f81d4112aa3341260af682ae3bc0a4ea4d23` | MATCH |
| checkpoint_raw_param_sha256 | `167074c43f0efdc6afe406982162ab93ab95b5e831e1ff26898a438aee2041ca` | `167074c43f0efdc6afe406982162ab93ab95b5e831e1ff26898a438aee2041ca` | `167074c43f0efdc6afe406982162ab93ab95b5e831e1ff26898a438aee2041ca` | `167074c43f0efdc6afe406982162ab93ab95b5e831e1ff26898a438aee2041ca` | `167074c43f0efdc6afe406982162ab93ab95b5e831e1ff26898a438aee2041ca` | MATCH |
| checkpoint_file_sha256 | `c022c8a3773e7c01a05daebab81fec908071f3645920fd2560ee43c57f51ca18` | `c022c8a3773e7c01a05daebab81fec908071f3645920fd2560ee43c57f51ca18` | `c022c8a3773e7c01a05daebab81fec908071f3645920fd2560ee43c57f51ca18` | `c022c8a3773e7c01a05daebab81fec908071f3645920fd2560ee43c57f51ca18` | `c022c8a3773e7c01a05daebab81fec908071f3645920fd2560ee43c57f51ca18` | MATCH |

"Kaggle T4 (kernel v5, re-check)" is a third independent replay on the
same host class run after the two-host comparison above, using the same
pinned stack; included because the harness reported `machine_shape` was
not honoured by the platform (the manifest's `gpu_name` still read Tesla
T4), so it re-confirms same-host determinism rather than adding a third
architecture. All digests across all five runs and both hosts are equal
for every field in this table.

## 3. The disclosed negative: an un-pinned tokenizer breaks cross-host identity before any GPU math runs

An earlier pass at this same comparison (same corpus bytes, same
teacher, same seeds) used `transformers 5.0.0` on one host and
`transformers 5.17.0` on the other, tokenizing the corpus text fresh on
each host rather than shipping a pre-tokenized fixture. The corpus bytes
and the tokenizer's own vocabulary hash matched across hosts, but the
resulting token counts did not: 157,795 tokens under `transformers
5.0.0` vs 157,822 under `5.17.0` — a 27-token difference, with the earliest
divergence at token index 73,705. Diagnostic re-tokenization of the
identical corpus bytes with both `transformers` versions reproduced this
exact divergence locally. Part of the delta traces to a CRLF-adjacent
merge token (`"\r\n"`) appearing 30 times under 5.0.0 vs 23 times under
5.17.0; the remainder comes from other CRLF-adjacent merges not
itemized further. A short two-token probe string tokenized identically
under both versions, so the difference is context-dependent
pre-tokenization/merge behavior between the two tokenizer backends
(`TokenizersBackend` vs `GPT2Tokenizer`, both reported "fast"), not a
vocabulary change.

Both token counts happened to floor to the same number of fixed-length
blocks, so a block-count check alone would not have caught this — the
token *content* differed while the block *count* matched. Every digest
downstream of tokenization (teacher-logit cache, checkpoint) therefore
differed across hosts, even though the corpus and the tokenizer identity
both matched. This was purely a tokenization-layer effect; it says
nothing about the correctness of the GPU training math on either host.

**Lesson:** a training receipt that hashes only the source corpus and
records a tokenizer vocabulary hash is not sufficient to establish
cross-host reproducibility. The token-id tensor itself must be hashed
(or shipped as a pinned fixture, as done in section 2 above), and the
tokenizer's *implementation* version must be pinned and recorded
alongside its vocabulary hash — two tokenizers can share a vocabulary
hash and still segment the same bytes differently.

## 4. Environment (both hosts)

| | Kaggle T4 | Lightning AI Studio (AWS), T4 |
| --- | --- | --- |
| GPU | Tesla T4 | Tesla T4 |
| Driver | not captured in the Kaggle log | 580.178.04 |
| torch | 2.10.0+cu128 | 2.10.0+cu128 (pip-pinned; stock image shipped 2.8.0+cu128) |
| CUDA (`torch.version.cuda`) | 12.8 | 12.8 |
| cuDNN | 91002 | 91002 |
| transformers | 5.0.0 | 5.0.0 |
| tokenizers | 0.22.2 | 0.22.2 |
| Determinism env | `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `use_deterministic_algorithms(True)`, no TF32, `cudnn.benchmark=False` | same |

## 5. Not claimed

- Only T4-class GPUs are covered here; a separate-architecture run (a
  Kaggle P100 request) did not actually land on a P100 (the platform
  reported `gpu_name: Tesla T4` regardless of the requested machine
  shape) and is still pending.
- No quality claim: the fixture corpus is a single public-domain book
  used as a determinism fixture, not a training corpus chosen for
  quality or scale.
- No claim across `torch` builds in isolation: this result pins `torch`,
  CUDA runtime, `transformers`, and `tokenizers` identically on both
  hosts; an earlier attempt that left `transformers` un-pinned (5.0.0 vs
  5.17.0) diverged (section 3), and `torch` 2.8 vs 2.10 has not been
  tested in isolation with everything else held fixed.
- The training harness itself is private today (an internal recipe, not
  a released tool). What a third party can check today is the fixture
  (the pre-tokenized token-id file and its sha256, and the corpus's own
  public Project Gutenberg source) and the digests in section 2 — the
  harness is not yet released for independent re-execution.

## 6. Relation to CIS-2

CIS-2 establishes bit-identical *inference* receipts across machines
and ISAs for a pinned model and pinned deterministic math. This result
asks the analogous question one layer earlier, on the *training* side:
whether a full multi-step training run (not just a forward pass)
reproduces bit-for-bit across GPU hosts once the software stack and the
tokenized input are pinned as tightly as CIS-2 pins inference.

## Acknowledgments

A special thank you to Charles Seaman and Linda Blanchard, whose contributions have helped Aefinity AI stay on track.

And a very special thank you to **Bonnie Rae Power**: an amazing woman, a great friend and neighbor, without whom Aefinity AI would have never had a chance to ever get started. Thank you, Bonnie, for your advice, care, encouragement, guidance, intuitive wisdom, and financial assistance.
