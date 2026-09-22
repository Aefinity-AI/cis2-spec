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

<!-- FILL IN AFTER RUN: either the actual run_eval.py --mode run numbers,
     or, if weights fetch does not complete in time, the resume point. -->

STATUS AS OF THIS COMMIT: harness (`eval/run_eval.py`), item files
(`eval/items/*.jsonl`), and the `cis2_ref` decoded-text addition
(`src/main.rs`) are built and committed. The SmolLM2-135M weight fetch
(`scripts/fetch_weights.sh`, sha256-verified against
`weights/MANIFEST.sha256`) has been unreliable on this box in this
session — the HuggingFace CDN connection has repeatedly reset, timed out,
or failed DNS resolution mid-transfer (see raw curl errors below), so no
end-to-end `run_eval.py --mode run` has completed yet. No score in this
document is fabricated; none exist yet for this reason.

```
curl: (92) HTTP/2 stream 1 was not closed cleanly: CANCEL (err 8)
curl: (6) Could not resolve host: us.aws.cdn.hf.co
curl: (28) Connection timed out after 300173 milliseconds
curl: (56) Recv failure: Connection reset by peer
```

**RESUME**: once `weights/model.safetensors` (sha256
`80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`,
269,060,552 bytes) plus `config.json`/`tokenizer.json` are fully fetched
(a resumable `curl -C -` retry loop was left running in the background at
`~/projects/cis2-spec-cm-ev1/weights/`; check
`sha256sum weights/*.{safetensors,json}` against
`weights/MANIFEST.sha256` before trusting it), run:

```sh
cd cis2-spec-cm-ev1
python3 eval/run_eval.py --mode run
```

then fill in the real per-category numbers below from
`eval/results/summary.json`, replacing this section, and commit.

## 3. Numbers

(pending — see §2)

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
