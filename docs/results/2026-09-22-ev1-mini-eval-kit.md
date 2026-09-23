# ev-1: mini eval kit for the pinned CIS-2 reference model

Rule A: no timing/benchmark numbers are reported here.

## 1. Setup

Task: build a small (≤200 item), honest evaluation kit for "the pinned
CIS-2 2B model" and run it through the CIS-2 reference tooling, emitting
per-item receipts.

**Model substitution (honest, per task's own fallback instruction):** no
2B model or checkpoint exists in this repo (`cis2-spec`) or was found on
this box. `cis2-spec`'s own `EXPECTED_DIGESTS.md` pins only
`HuggingFaceTB/SmolLM2-135M` (§13.1 normative vector) plus informative
Qwen2.5-0.5B/1.5B vectors. The 2B BitNet checkpoint referenced elsewhere in
`claudius-maximus` state notes belongs to a different project
(`alice-aegis`/`aegis-forge`), not `cis2-spec`, and is out of scope for
this repo's reference tooling. **This kit therefore runs SmolLM2-135M,
fp32, greedy decode**, via this repo's own `cis2_ref` (built from `src/`,
`cargo build --release --bin cis2_ref`) — the same reference binary/model
`EXPECTED_DIGESTS.md` and `scripts/self_check.sh` already pin.

160 items (≤200 cap) across 4 categories — see `eval/SOURCES.md` for exact
provenance/license per category:

| Category | Items | What it checks |
|---|---|---|
| `truthfulness` | 60 | Sampled from TruthfulQA (Apache-2.0). Heuristic exact-substring scoring only (see caveat below). |
| `self_consistency` | 20 | Same prompt run twice, both within one `cis2_ref` process (its own run1/run2 determinism check) and across two separate process invocations. |
| `tool_use` | 60 | Arithmetic-completion proxy (original items). |
| `refusal` | 20 | 10 "should refuse" / 10 "should comply" narrow, non-operational probes (original items). |

## 2. Honest limitation

STATUS: weight fetch retried on 2026-09-23 and succeeded this time (HF CDN
was flaky in the prior session; the earlier `curl` errors — HTTP/2 stream
resets, DNS resolution failures for `us.aws.cdn.hf.co`, 300s timeouts, and
connection resets — did not recur). All three artifacts fetched via
`scripts/fetch_weights.sh weights` and sha256-verified against
`weights/MANIFEST.sha256`:

```
model.safetensors  80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1
config.json         1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843
tokenizer.json      9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
```

`python3 eval/run_eval.py --mode run` then completed end-to-end (exit 0),
building `cis2_ref` release and scoring all 160 items across the 4
categories. Numbers below are copied verbatim from
`eval/results/summary.json`; nothing here is fabricated or estimated.

## 3. Numbers

Per `eval/results/summary.json`:

| Category | n_items | n_pass | score_frac |
|---|---|---|---|
| `truthfulness` | 60 | 3 | 0.05 |
| `self_consistency` | 20 | 20 | 1.0 |
| `tool_use` | 60 | 5 | 0.0833... |
| `refusal` | 20 | 10 | 0.5 |

As expected for a non-instruction-tuned 135M base model (see §4):
`truthfulness` and `tool_use` scores are near-floor (weak heuristic
proxies against a base model with no instruction tuning); `refusal` sits
at chance (10/20 — the 10 "should comply" items likely pass more often
than the 10 "should refuse" items resist refusal-marker matching, but this
was not broken out further here); `self_consistency` is a perfect 20/20,
i.e. every one of the 20 self-consistency items produced bit-identical
`run1_witness_digest`/`run2_witness_digest`/`run1_argmax_digest` within
the same `cis2_ref` process, which is the actual claim this kit exists to
support (deterministic fp32 CIS-2 reproducibility on this model), not the
task-accuracy numbers.

Full per-item receipts (including `run1_witness_digest`, `run1_argmax_digest`,
`run1_tokens`, `run1_gen_text`, and pass/fail per item) are committed at
`eval/results/{truthfulness,self_consistency,tool_use,refusal}.json`, with
raw stdout/stderr in `eval/logs/*.run.log`.

Note: `eval/logs/self_consistency.run.rerun.log` is `run_eval.py`'s own
internal cross-process check for the `self_consistency` category (each
item's prompt run in two separate `cis2_ref` invocations, per §1 table
above) — it is produced automatically by `--mode run` and is not the same
thing as `python3 eval/run_eval.py --mode verify`. A standalone `--mode
verify` pass (re-deriving digests from scratch and diffing against the
committed `eval/results/*.json`, ideally on a second host such as box2)
has NOT been run as part of this task and is not claimed here — see §4.

## 4. What this does NOT show

- Not the 2B model (see §1) — no claim is made about any 2B-scale or
  instruction-tuned model here.
- `truthfulness`/`refusal` scoring on a non-instruction-tuned 135M base
  model is a weak proxy (verbatim-substring / refusal-marker heuristics),
  not a real accuracy or safety measurement; near-baseline or near-zero
  scores are expected and would not indicate anything wrong with the
  reference implementation.
- `tool_use` here means arithmetic-completion, not actual tool-call-format
  correctness (this base model has no tool-calling format at all).
- Single host; no cross-host claim until `--mode verify` is run on a
  second machine (e.g. box2) against the committed `eval/results/*.json`.

## 5. Re-run / verify

```sh
git clone <cis2-spec-repo-url> cis2-spec && cd cis2-spec
git fetch origin && git worktree add -f ../cis2-spec-ev1 cm/ev1-mini-eval-kit
cd ../cis2-spec-ev1
bash scripts/fetch_weights.sh weights
python3 eval/run_eval.py --mode run       # full run, or:
python3 eval/run_eval.py --mode verify    # receipt-reproducibility check only
```
