# ev-1: mini eval kit for the pinned CIS-2 reference model

Small (160-item, ≤200 cap), honest evaluation harness for the CIS-2
reference implementation (`cis2_ref`, built from `src/`), run against the
pinned model the rest of this repo already targets.

**Model actually used: `HuggingFaceTB/SmolLM2-135M`, fp32, greedy decode.**
This is NOT the "pinned CIS-2 2B model" mentioned in the original task
description. No 2B checkpoint or weights exist in this repo or were found
on this box (`cis2-spec` only pins SmolLM2-135M as its §13.1 normative
vector, plus informative Qwen2.5-0.5B/1.5B vectors — see
`EXPECTED_DIGESTS.md`). The 2B BitNet checkpoint referenced in
`claudius-maximus` state notes lives in a *different* project
(`alice-aegis`/`aegis-forge`), not `cis2-spec`, and this task was scoped to
`cis2-spec`'s own reference tooling. Per the task's own fallback
instruction, this kit runs the smallest pinned model instead and says so
here, honestly, rather than fabricating a 2B result.

## Layout

- `items/*.jsonl` — the frozen 160-item eval set (4 categories, ≤200 total).
  See `SOURCES.md` for exact provenance + license per category.
- `make_items.py` — the (offline) generator that produced `items/*.jsonl`.
  Not re-run automatically; kept for provenance/reproducibility.
- `run_eval.py` — the one script that runs `cis2_ref` over every item and
  scores it. See below.
- `logs/` — full raw stdout+stderr from every `cis2_ref` invocation.
- `results/*.json` — per-category receipts + scores; `results/summary.json`
  is the aggregate. `results/verify_summary.json` is written by `--mode verify`.
- `thirdparty/TruthfulQA.csv` + `TruthfulQA-LICENSE` — the upstream Apache-2.0
  source file `truthfulness` items were sampled from, vendored for
  reproducibility of `make_items.py`.

## Running

```sh
cd cis2-spec   # repo root
python3 eval/run_eval.py --mode run
```

This builds `cis2_ref` if needed (`cargo build --release --bin cis2_ref`),
fetches/checks weights via `scripts/fetch_weights.sh` if you haven't
already (do this first — see `scripts/self_check.sh` for the exact
sha256-verified fetch), then runs each category as one `cis2_ref
--prompt-set ... --gen-toks N` invocation, decodes generated text, scores
each item, and writes `eval/results/*.json` + `eval/logs/*.log`.

## Verify-only mode (re-run on a second machine, e.g. box2)

```sh
python3 eval/run_eval.py --mode verify
```

This re-runs `cis2_ref` on every item (cheap: 135M params, fp32, ≤24
generated tokens/item) with the exact same prompts and `gen_toks`, and
diffs the freshly produced `run1_witness_digest` / `run1_argmax_digest` /
`run1_tokens` against the values already committed in `eval/results/*.json`.
It does NOT re-derive scores or re-touch the committed item files — it is
strictly a receipt-reproducibility check: same weights sha256 + same
config sha256 + same tokenizer sha256 + same prompt bytes + same
`gen_toks` MUST produce bit-identical witness digests on any conforming
host, per the CIS-2 spec's whole reason for existing. `PASS` means every
digest reproduced exactly; any `FAIL`/mismatch is a real finding, not
something to paper over.

## What this kit does and does not show

- It shows whether `cis2_ref`'s deterministic fp32 forward pass, run on
  this specific tiny base model, reproduces the same witness digests
  across runs/hosts, and gives a first honest (low, expected-to-be-low)
  read on 4 narrow behavior categories.
- It does NOT show anything about an instruction-tuned or 2B-scale model.
  SmolLM2-135M-base has no instruction tuning, no RLHF, no safety
  training — the `refusal` and `truthfulness` category scores are
  expected to be low/near-baseline and should not be read as a real
  safety or accuracy measurement of any deployed system. See
  `docs/results/2026-09-22-ev1-mini-eval-kit.md` for the actual numbers
  and this caveat in full.
