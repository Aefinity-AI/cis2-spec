# fast-1: deterministic-speed profiling + KV-cache speedup (verify3/)

**Status: NOT for publish as-is.** This repo's README and
`EXPECTED_DIGESTS.md` state "No timing numbers are recorded anywhere in
this repository by policy." This document records timing numbers and a
5-6x speedup, so per that policy (and the operator's PUBLISH POLICY for
"a large speed/quality jump") it stays on a local, unpushed branch
(`cm/fast1-deterministic-speed`) pending an explicit decision from Justin
on whether/how to publish. See BLOCKERS/NEEDS note filed alongside this
work.

## Scope

Profiled `verify3/` (the C clean-room reference used by
`scripts/self_check.sh`) running the pinned §13.1 test vector
(`HuggingFaceTB/SmolLM2-135M`, prompt `"Once upon a time"`, 16
greedy-decoded tokens, fp32). Machine: `aefinity-box` (hostname
`aefinity-box`), 4 vCPUs, plain wall-clock `time`, no other legs running
concurrently, normal (not `nice`'d) priority — an internal engineering
profiling number, not a Rule-A protected timing claim.

## Profiling (manual `clock_gettime` instrumentation, not committed)

`perf` is installed but unusable in this sandbox
(`perf_event_paranoid=3`, would require a kernel-setting change outside
this task's scope). Instead, added temporary section timers around
`forward_full()`'s per-layer stages, ran once, discarded the
instrumentation. Wall time breakdown for one full (baseline,
non-incremental) 16-token decode:

```
qkv_proj=3.45s  rope=0.006s  attn_score=0.05s  o_proj=2.10s  mlp=17.25s  norm=0.02s
(total layer-stage time ~22.9s of ~25.6s single-run wall time)
```

MLP matvecs (`gate_proj`/`up_proj`/`down_proj`, each `[1536,576]` or
`[576,1536]`) dominate at ~75% of layer time, consistent with FLOP
counts (MLP does ~3x the multiply-adds per layer that attention's
q/k/v/o projections do combined, for this model's `hidden=576,
intermediate=1536`).

But the FLOP-count breakdown understated the real problem: `config.json`
gives `num_hidden_layers=30`, `vocab_size=49152`. The **original**
`cis2_run_decode()` had no KV cache — every decode step called
`forward_full()` over `tokens[0..ntok-1]` **from scratch**, recomputing
every position's q/k/v, attention, and MLP through all 30 layers, even
though positions `0..ntok-2`'s values are bit-identical to what was
already computed in earlier steps (each (layer, position) tuple's output
depends only on that position's own residual stream and the layer's own
already-finalized K/V for positions `<=` it — never on how many tokens
have been generated since). Over a 16-token decode starting from a
4-token prompt, that's `sum(p for p in 4..19)` ≈ 184 position-forwards
instead of 20 (`4 prompt + 16 generated`) — a ~9x redundancy factor,
which is the actual dominant cost, not cache/tiling behavior inside a
single GEMV.

## The change

Replaced `forward_full()` (recompute-everything-every-step) with
`process_position()` (verify3/model.c) + a per-layer K/V cache
(`decode_kv_cache`, `kv_cache_alloc`/`kv_cache_free`). Each sequence
position is now processed through all 30 layers **exactly once**, in
increasing position order; attention reads K/V for earlier positions from
the cache instead of recomputing them. `cis2_run_decode()` primes the
cache with the 4 prompt positions, then processes exactly one new
position per generation step.

**No floating-point operation, no reduction order, and no accumulation
order changed.** Every scalar value in the new code is produced by
literally the same sequence of arithmetic operations (same `cis2_dot_seq`
left-to-right accumulation, same softmax computation, same RMSNorm
sum-of-squares order, same RoPE application) as before — the change only
stops *recomputing* values that were already computed bit-for-bit
identically in an earlier step. This was verified empirically: the
witness digest, argmax digest, table digest, and inv_freq digest are all
unchanged (see Verification below), and the per-layer `CIS2_DUMP`
side-channel digests (E15k debug instrumentation, not part of any
normative digest) for step-0/step-1 also matched between old and new code
when checked by hand during development.

No FMA was introduced (`-mno-fma -ffp-contract=off` unchanged in
`verify3/Makefile`), no reassociation, no changed transcendental routes.

## Before / after (aefinity-box, plain wall-clock `time`, `verify3/`, `-O2`, gcc)

`main.c` runs the full 16-token decode **twice** in-process (the §12.4
same-process determinism check), so wall time below is for two runs; per
single decode-run tok/s = `16 / (wall_time_for_two_runs / 2)`.

| | wall (2 runs) | wall (1 run) | tok/s |
|---|---|---|---|
| before (original, no KV cache) | 50.4-51.2s (3 runs) | ~25.3s | ~0.63 tok/s |
| after (KV-cache incremental decode) | 8.76-8.88s (3 runs) | ~4.4s | ~3.6 tok/s |

**Speedup: ~5.7x.**

(The 9x position-redundancy estimate above is an upper bound on the
*layer-stage* work; the ~5.7x wall-clock number also includes fixed
per-run costs — model load/bf16-widen of ~270MB of weights, tokenizer
setup, SHA-256 hashing of the whole weights file for the artifact-hash
check — that don't shrink with this change, which is why the observed
speedup is smaller than the ~9x redundancy factor.)

## Verification

```
$ scripts/self_check.sh
...
witness digest: got=d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df expected=d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df
argmax digest:  got=0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff expected=0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff
table digest:   got=23c7bfaf5cef0095fd021af2eb1808abb4928bae4219756d86bdac670a06b35d expected=23c7bfaf5cef0095fd021af2eb1808abb4928bae4219756d86bdac670a06b35d
invfreq digest: got=da9f6dcfde0425588815509e874515cdcd3d6b8818b6d0136590052e7bbf6f12 expected=da9f6dcfde0425588815509e874515cdcd3d6b8818b6d0136590052e7bbf6f12

PASS: all digests match the pinned CIS-2 v0.3b test vector.
```

`generated_token_ids` also unchanged:
`28,665,436,253,1838,8180,3365,14176,30,2306,4161,281,253,2066,2291,351`.
Two-run in-process determinism check (`CIS2_VERIFY3 determinism=PASS`)
also passes, confirming the new code is itself deterministic run-to-run.

## Not done (out of scope for this pass)

- `src/` (the Rust reference) and `verify2/` (Rust clean-room) have the
  same no-KV-cache redundancy; not touched here — this pass only covered
  `verify3/` (C), which is what `scripts/self_check.sh` builds and runs.
- No attempt was made to further tile/block the individual GEMV loops
  (`cis2_matvec`/`cis2_dot_seq`) — profiling showed the redundant
  recomputation, not intra-GEMV cache behavior, was the dominant cost, so
  that was the one speedup attempted per the task's scope.
- `perf`-based profiling was not possible in this sandbox
  (`perf_event_paranoid=3`); manual section timers were used instead and
  then removed before commit (not part of the shipped diff).
