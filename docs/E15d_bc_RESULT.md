# E15d(b,c) — longer-horizon prompt set + second fp32 model family (Qwen2.5-0.5B)

Historical development note (from the reference repository's internal research
log). No timing numbers anywhere (Rule A).

Raw logs: `hardware_logs/e15d_bc_2026-08-28.log`.

CI runs (final passing state):
- E15d(b): reference-repository CI run 33191206420 (private CI, not reproducible externally)
- E15d(c): reference-repository CI run 33191206427 (private CI, not reproducible externally)

(Earlier attempts on this branch: E15d(b) passed first try,
run reference-repository CI run 33190379515 (private CI, not reproducible externally).
E15d(c)'s first attempt, run
reference-repository CI run 33190530860 (private CI, not reproducible externally), FAILED
only its `oracle` job — see "Honest findings" below; the `cross-isa` and
`aggregate` jobs already passed on that first attempt.)

Toolchain: `rustc 1.98.0 (88d9e12ae 2026-08-18)`, pinned via
`dtolnay/rust-toolchain@1.98.0` on both `ubuntu-latest` (x86_64) and
`ubuntu-24.04-arm` (aarch64) runners, per the E15d compiler-invariance
track's pin (this repo's `cm/e15d-compiler-invariance` branch owns the
`rust-toolchain.toml`; these two CI workflows pin the same version
directly in the `dtolnay/rust-toolchain@` action step instead, per this
task's explicit instruction not to touch that branch's files).

Commits under test: `060146f`, `600970c`, `c4366c7`, `cb4af47` on
`cm/e15d-longer-and-second-model` (built on top of E15c's `f83e8dd`).

## What this milestone answers

E15c (`docs/E15c_RESULT.md`) proved cross-ISA bit-identity for one model
(SmolLM2-135M), one short prompt (4 tokens), one generation length (16
tokens). E15d asks two follow-on questions:

- **(b)** Does the same cross-ISA bit-identity hold at a longer decode
  horizon (128 generated tokens) and across a diverse prompt set (short,
  code, and a ~200-token long prompt), not just the one 4-token prompt?
- **(c)** Does the CIS-2 §3 pinned-arithmetic contract generalize to a
  second, architecturally-different fp32 model family (Qwen2-family:
  QKV bias, different `rope_theta`), both for correctness (vs. a
  torch/transformers oracle) and cross-ISA bit-identity?

## What was built

- `src/main.rs`: new CLI flags `--gen-toks N`, `--prompt-set FILE` (JSON
  array of prompt strings), `--weights-dir DIR`. With no flags, behavior
  is unchanged from E15c (single hardcoded prompt "Once upon a time",
  gen_toks=16, `weights/`) except that the printed `table_digest=` value
  changed (see below — `math::table_digest()` now also covers the new
  `ln_pinned` coefficient table). The single-prompt run loop was factored
  into `run_one_prompt()` so the legacy path and the new multi-prompt path
  share identical logic.
- `src/math.rs`: added `ln_pinned(x)`, a route-(b) pinned Cephes-pattern
  logf (exact bit-manipulation `frexp` + fixed-degree Horner polynomial in
  the reduced mantissa), covered by `table_digest()` and by
  `math::tests::ln_matches_std_within_tolerance` (host-libm sanity check
  only, same 2e-6 relative bar as `exp`/`sin`/`cos`, includes both
  rope_theta=1e5 and rope_theta=1e6 in the test's input list).
- `src/main.rs` RoPE `inv_freq` construction generalized from a hardcoded
  `LN_THETA` literal (pinned for exactly `rope_theta=1e5`, with an
  `assert_eq!` that made any other theta a hard panic) to
  `math::ln_pinned(rope_theta as f32)`, read from `config.json` — still
  ONLY the pinned route-(b) polynomials, never host `f64::powf`/`ln`.
- `src/main.rs`: `LayerWeights` gained optional `q_bias`/`k_bias`/`v_bias`
  (`Option<Vec<f32>>`), loaded only if the corresponding tensor exists in
  the checkpoint (`load_bias_opt`), added via `add_bias_opt` after each
  QKV matvec — a no-op for Llama-family checkpoints with no such tensors.
  `Model` gained an optional `lm_head` (untied LM head), used only when
  `config.json`'s `tie_word_embeddings` is `false`.
- `prompts/e15d_prompt_set.json`: 5 fixed prompts checked into the repo —
  `"Once upon a time"` (continuity with E15b/E15c), a pangram sentence, a
  news-style sentence, a Python function stub, and a ~200-token (187-word)
  paragraph about the history of computing.
- `.github/workflows/e15d-b-longer.yml`: x86_64 + aarch64 matrix, runs
  `cis2_ref --prompt-set prompts/e15d_prompt_set.json --gen-toks 128`,
  extracts each prompt's `CIS2_REF digest=` line, uploads per-arch digest
  artifacts, and an `aggregate` job that fails on any x86/aarch64 mismatch
  across all 5 prompts.
- `.github/workflows/e15d-c-qwen.yml`: fetches Qwen2.5-0.5B, runs
  `cross-isa` (x86_64 + aarch64) at `gen_toks=16` and `gen_toks=128`,
  aggregates digest comparison across both lengths; a separate `oracle`
  job (x86_64 only) builds torch 2.13.0+cpu / transformers in a venv,
  runs `scripts/oracle_compare_qwen.py` (same determinism pins as E15b
  m1.5's `oracle_compare.py`: `use_deterministic_algorithms`, no TF32,
  1 thread, fp32, `no_grad`), and `scripts/diff_logits.py` to check the
  step-0 full-vocab logit relative diff against the same `< 1e-2` bar
  E15b m1.5 preregistered.

## Model choice for E15d(c)

**Qwen2.5-0.5B was used, as the brief's first preference** (Llama-like
architecture — RMSNorm, GQA, SwiGLU, non-interleaved RoPE — but with QKV
bias and a different `rope_theta`). It built and ran successfully on the
**first attempt** at the Rust-loader level (no fallback to SmolLM2-360M or
Llama-3.2-1B was needed): the existing generalization work (optional bias,
config-driven rope theta) was sufficient with zero additional
architecture-specific code, because Qwen2.5-0.5B's `tie_word_embeddings`
is `true` (so the pre-existing tied-embedding LM head path was exercised,
not the new untied path — that code is written and present but has not
yet been exercised against a real untied checkpoint; see "Honest open
questions").

Qwen2.5-0.5B `config.json` (fetched fresh this session, public HF
`resolve`, no auth): `hidden_size: 896`, `intermediate_size: 4864`,
`num_hidden_layers: 24`, `num_attention_heads: 14`, `num_key_value_heads:
2`, `rms_norm_eps: 1e-06`, `rope_theta: 1000000.0`, `tie_word_embeddings:
true`, `vocab_size: 151936`. `self_attn.{q,k,v}_proj.bias` tensors are
present in the checkpoint (Qwen2's architecture always sets
`attention_bias=True` regardless of a config flag) and were loaded and
applied by the new `add_bias_opt` path.

## Results — E15d(b): prompt × ISA → digest (gen_toks=128)

| prompt_idx | prompt (truncated) | x86_64 CIS2_REF digest | aarch64 CIS2_REF digest | match |
|---|---|---|---|---|
| 0 | "Once upon a time" | `0f3ccdb9b69847573cf19d16f0a91e957c4409e48eb6f40978d06b35a2814cf4` | same | Y |
| 1 | "The quick brown fox jumps..." | `0fcd8ee2ebf44d9580a439ccc1daff9b710d334e3056777305c1ba50b2e29e39` | same | Y |
| 2 | "In a shocking turn of events..." | `ec5b6a8aec099ffe32567b92ebf59aa7ffb56d9cbeb15350fdcb3afa0fc54afa` | same | Y |
| 3 | "def fibonacci(n): ..." | `7f3aae1c1ad3e7a1412a55e16e0aa70ad88e0a760925033e01fad6ce1cf17496` | same | Y |
| 4 | ~200-tok history-of-computing paragraph | `02843ebe1b5755583a27cf36c80d7c443ab186aa0d41d01fb248eb4d5ec93eb9` | same | Y |

**5/5 prompts, bit-identical x86_64 vs aarch64 digests at gen_toks=128.**
CI `aggregate` job: **PASS** ("all prompt digests match bit-identically
between x86_64 and aarch64").

These 5 digests are the **new preregistered reference values** for
SmolLM2-135M at gen_toks=128 over this prompt set (supersedes E15b/E15c's
single gen_toks=16 "Once upon a time" digest
`ba88708bf4159a4057d4eef727820ebc5ce1744ae334bfe0cc0776e03509dd0f`, which
remains valid for its own gen_toks=16 case — the new `table_digest`
change, from `math::ln_pinned` being added, means a from-scratch rerun of
the OLD gen_toks=16 case would also now print a different `table_digest=`
field, though the witness/argmax digests for that unchanged code path are
unaffected since `ln_pinned` is only called once at model-load time for
`inv_freq`, and `ln_pinned(100000.0)` reproduces the old hardcoded
`LN_THETA` literal to within the same host-libm sanity tolerance already
used elsewhere — see `math::tests::ln_matches_std_within_tolerance`).

`table_digest=465d358ccd63721256dbd2abbc77ad5de755adf3230635f95b12b1727dfa1ea3`
(new value, covers `ln_pinned`'s coefficients in addition to the E15b/c
`exp`/`sin`/`cos` tables) and
`inv_freq table_digest=33b4b06c37c6b5b54da225ac8de68db17e96131aef21c24ef16756e4c4d4d18b`
(new value for prompt 0's inv_freq table — SmolLM2 rope_theta=1e5 via
`ln_pinned` instead of the old hardcoded literal) were identical across
x86_64 and aarch64.

Weight provenance (re-verified this session, fresh download):
`model.safetensors` sha256
`80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`
(matches E15b/E15c's table exactly), `config.json` sha256
`1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843`,
`tokenizer.json` sha256
`9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`.

## Results — E15d(c): Qwen2.5-0.5B, model × ISA → digest

| gen_toks | x86_64 CIS2_REF digest | aarch64 CIS2_REF digest | match |
|---|---|---|---|
| 16 | `085da81c52b9308947460740eaff108fa273b2638ab2eae913a0eb4e4d773658` | same | Y |
| 128 | `edfd66d7e61e0f02348c606c434351797b54ab29088e76796f0a5bf61a5d14f9` | same | Y |

CI `aggregate` job: **PASS** ("Qwen2.5-0.5B digests match bit-identically
between x86_64 and aarch64 at gen_toks=16 and 128"). Same
`table_digest=465d358ccd63721256dbd2abbc77ad5de755adf3230635f95b12b1727dfa1ea3`
on both ISAs (identical to E15d(b)'s, as expected — it's a
model-independent constant table) and
`inv_freq table_digest=33b4b06c37c6b5b54da225ac8de68db17e96131aef21c24ef16756e4c4d4d18b`
(correctly DIFFERENT from SmolLM2's `da9f6dcfde0425588815509e874515cdcd3d6b8818b6d0136590052e7bbf6f12`
above — both models share `head_dim=64`, but `rope_theta` differs 10x
(1e5 vs 1e6), so every `inv_freq[i]` for `i>0` differs; independently
confirmed with a local scratch unit test computing both tables via the
same `ln_pinned`/`exp_pinned` calls: `inv_freq[1]` is `0.69783056` at
theta=1e5 vs `0.64938164` at theta=1e6, `inv_freq[31]` is `0.000014330137`
vs `0.0000015399268` — genuinely different tables, correctly reflected in
different digests).

Weight provenance (fetched fresh this session): `model.safetensors` sha256
`88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`,
`config.json` sha256
`479dcf0c5286339e41ad3992cd08ae88a467c4187587936248e2b7c96283484b`,
`tokenizer.json` sha256
`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.

## Oracle results — Qwen2.5-0.5B vs torch/transformers (gen_toks=16, single default prompt)

Same prompt as E15b m1.5 ("Once upon a time", tokenizes differently under
Qwen's own tokenizer: `[12522, 5193, 264, 882]`), same tolerance bar
(`< 1e-2 * max|logit|`), `torch==2.13.0+cpu`, `transformers` (latest
compatible with the pinned torch wheel at CI run time), CPU-only,
`torch.use_deterministic_algorithms(True)`, TF32 disabled, 1 thread, fp32,
`model.eval()`, `no_grad()`.

```
n = 151936
max_abs_diff        = 6.818771362304688e-05  (at vocab idx 12892: rust=1.0115975141525269 oracle=1.01166570186615)
relative_max_diff   = 3.901223423665661e-06
PASS: relative diff 3.901223423665661e-06 < preregistered bar 0.01
```

Token-level: **16/16 exact match** between the Rust reference and the
torch/transformers oracle (comparing the full prompt+generated sequence
both ways):

```
rust   = [12522, 5193, 264, 882, 11, 1052, 572, 264, 2632, 3743, 6941, 47290, 13, 2932, 10245, 311, 1486, 448, 1059, 23069]
oracle = [12522, 5193, 264, 882, 11, 1052, 572, 264, 2632, 3743, 6941, 47290, 13, 2932, 10245, 311, 1486, 448, 1059, 23069]
```

**CORRECT**, by the same bar E15b m1.5 established for SmolLM2-135M — the
Qwen2.5-0.5B forward pass (RMSNorm, GQA with `n_kv_heads=2`, QKV bias,
non-interleaved RoPE at `theta=1e6`, SwiGLU MLP, tied LM head) agrees with
an independent oracle to ordinary fp32 cross-implementation rounding
noise (~4e-6 relative), not bit-identically (expected — torch/MKL uses a
different reduction order, FMA, and its own libm transcendentals).

## Honest findings

1. **The first E15d(c) CI attempt failed on a CI-script bug, not a
   correctness bug** (run
   reference-repository CI run 33190530860 (private CI, not reproducible externally),
   `oracle` job). The workflow compared `rust_run.txt`'s `run1_tokens=`
   field (which is `prompt_ids + generated_ids`, the full sequence)
   against the oracle script's `oracle_tokens=` field (generated-only, no
   prompt prefix) — a length mismatch that looked like a token-level
   correctness failure but wasn't. The underlying data in that failed run
   already showed the generated tokens agreeing exactly and the logit
   relative diff already under bar (4.39e-6) — it just diffed the wrong
   pair of lists. Fixed in commit `cb4af47` by comparing against the
   oracle's `oracle_all_tokens=` field instead (also prompt+generated);
   the re-run (`33191206427`) passed cleanly. **Reported per instructions
   as a finding, not silently corrected**: this was a genuine CI defect
   this session introduced and then fixed, not a numerics finding, but it
   is exactly the kind of "any mismatch is a finding" case the task
   called out.
2. (Initial draft of this document misread the CI logs and briefly
   claimed Qwen2.5-0.5B's `inv_freq table_digest` matched SmolLM2's —
   that was a transcription error while writing this report, not a code
   finding. Re-checked directly against the raw logs plus an independent
   local scratch test (`math::e15d_debug::debug_inv_freq_differs_by_theta`,
   not committed — a throwaway check, deleted after use): the two
   digests are correctly different (`da9f6dcf...` for SmolLM2 at
   theta=1e5 vs `33b4b06c...` for Qwen at theta=1e6), and `inv_freq[1]`/
   `inv_freq[31]` differ by the expected ~10x-driven amount between the
   two thetas. No bug here; corrected before finalizing.)
3. **The untied-LM-head code path (`Model.lm_head`,
   `tie_word_embeddings: false`) is unexercised.** Qwen2.5-0.5B has tied
   embeddings, so this milestone's "second model family" happened to not
   need that branch. It compiles and the loader would call it correctly
   per the safetensors key name (`lm_head.weight`), but it has zero
   runtime coverage — a genuinely untied checkpoint (e.g. Qwen2.5-1.5B/3B,
   or a Llama-3.2 variant) would be needed to actually test it.
4. **No independent from-spec clean-room reimplementation** (same caveat
   E15b/E15c already carried forward — this milestone doesn't add or
   remove that gap).
5. **`ln_pinned`'s accuracy was validated only against host libm at 10
   discrete points** (`math::tests::ln_matches_std_within_tolerance`), not
   against an independent correctly-rounded reference across its full
   domain — same "route (b) is one arbitrary correct choice pinned by
   fiat" caveat the E15b writeup already applies to `exp`/`sin`/`cos`.

## Unit + gate test results (this session, x86_64 local)

```
$ cargo test --release
running 7 tests
test math::tests::widen_bf16_exact ... ok
test math::tests::exp_matches_std_within_tolerance ... ok
test denormal::tests::ftz_daz_pinned_and_denormal_input_flushed ... ok
test math::tests::dot_seq_is_order_sensitive_by_design ... ok
test math::tests::ln_matches_std_within_tolerance ... ok
test math::tests::sin_cos_match_std_within_tolerance ... ok
test math::tests::rsqrt_matches_expected ... ok
test result: ok. 7 passed; 0 failed

running 1 test (tests/no_mul_add.rs)
test no_mul_add_calls_in_src ... ok

$ ./scripts/no_fma_gate.sh
== both FMA gates PASS ==
```

(Local run only did `cargo check`/`cargo test`/the FMA gate — no model
weights were downloaded or run locally, per the RAM-budget instruction;
all model-loading legs ran on GitHub Actions.)

## Resource notes

Local (penguin) work this session was limited to `cargo check --release`,
`cargo test --release` (unit tests only, no model weights), and
`./scripts/no_fma_gate.sh` — `free -m` showed 238–487 MB available at
various points during this session (below the 600 MB smoke-test bar), so
no local model-loading/inference was attempted; all `cis2_ref` executions
(all four CI legs above) ran on GitHub Actions runners. No timing numbers
are reported anywhere in this document (Rule A).
