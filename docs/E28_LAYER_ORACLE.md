# E28 — every intermediate activation, both models, against the oracle

**Closes §14.6.** 13,452 named intermediate tensors — every activation of
every layer at every position, on both models §0 names — compared against a
`transformers` fp32 forward pass. Worst disagreement: **9.9e-5** relative
L2, on one tensor of one model. No layer creates error; every layer carries
in what it was handed.

---

## 0. What §14.6 said, and why it was the last real gap

§14.6, unchanged in kind since v0.1:

> "The oracle correctness checks (§13.3) are defensible spot-checks, not
> exhaustive. They confirm greedy token-id agreement and one step's
> full-vocab logit agreement to ~4e-6 relative on two model families now
> (SmolLM2-135M, Qwen2.5-0.5B) — neither checks every intermediate layer's
> activations against the oracle, so a compensating pair of errors elsewhere
> in the layer stack that happens to preserve step-0's output and all argmax
> decisions cannot be completely ruled out by this evidence alone."

That is an honest statement of a real hole, and it is the *last* one of its
kind in §14. Everything else §14 still lists is either closed, a deliberate
pin, or a scope boundary. This one was a live possibility: §13.3 looks only
at the two ends of the pipe. Two errors inside the stack that cancelled —
one layer computing something wrong, a later one bringing the result back —
would pass every check the specification had.

The gap is not closable by more end-to-end testing, however much of it. It
closes only by looking inside.

## 1. Method

Two dumps in one record format, compared offline.

**The verifier half.** `cis2-verify/src/ops.rs` gains a `tap` module and
`src/model.rs` gains sixteen tap points, all behind a new `layerdump` cfg
feature; `examples/layer_dump.rs` runs a decode with the tap open. The tap is
**write-only** — it reads the slice it records and touches nothing the
forward pass will read. The check that this is true is not an assertion in
the code, it is the digests: an instrumented run must still print the pinned
`witness-digest` and `argmax-digest`, and it does, for both models.

Sixteen tensors, per position:

| once per position | per layer |
|---|---|
| `embed`, `final_norm`, `logits` | `ln1`, `q_proj`, `k_proj`, `v_proj`, `attn_out`, `o_proj`, `resid_attn`, `ln2`, `gate_proj`, `up_proj`, `mlp_act`, `down_proj`, `resid_mlp` |

`q_proj` / `k_proj` / `v_proj` are taken **after** §9.1's optional QKV bias,
because the oracle's `q_proj` is an `nn.Linear` whose output already includes
it. SmolLM2 has no QKV bias and Qwen2.5 does, so taking the tap before the
add would have compared different quantities on the second model only. Ten of
the sixteen are the direct output of a named `torch.nn.Module`; three
(`attn_out`, `resid_attn`, `mlp_act`) are the *input* to one; `resid_mlp` is
the decoder layer's own output. None requires interpreting what the oracle
"meant" by a value.

**The oracle half.** `scripts/oracle_layers.py` registers forward and
forward-pre hooks on the corresponding `transformers` modules and writes the
same format. It steps one token at a time with a KV cache over the token
sequence the verifier actually ran — the prompt ids plus the ids the verifier
generated — so each oracle forward corresponds to exactly one verifier
forward at the same position. That is teacher forcing, and it is deliberate:
it isolates the numerics from any divergence in token choice. Token agreement
is then still *checked* rather than assumed, because the oracle's own argmax
at each step is recorded separately.

**The comparison.** `scripts/compare_layers.py`. Per tensor:

- `rel_l2` = `‖v − o‖₂ / ‖o‖₂` — the headline; scale-free and stable
- `max_abs` = `max |v − o|`
- `rel_rms` = `max_abs / rms(o)` — worst element as a fraction of the
  tensor's own scale

The verifier runs each decode twice (§12.4's determinism check), so every
tensor appears twice in its dump. The duplicate is not discarded silently:
it is checked to be bit-identical. **7,467 and 5,985 duplicate records, 0
that differ** — which is §12.4 re-confirmed at the granularity of every
intermediate rather than of the receipt.

## 2. Result

| | SmolLM2-135M | Qwen2.5-0.5B |
|---|---|---|
| layers × positions | 30 × 19 | 24 × 19 |
| tensors compared | **7,467** | **5,985** |
| name/shape mismatches | 0 | 0 |
| duplicate records that differ | 0 | 0 |
| **worst `rel_l2`, any tensor** | **2.228e-5** (`p0.L10.down_proj`) | **9.923e-5** (`p0.L22.o_proj`) |
| worst `rel_rms`, any tensor | 4.094e-4 | 2.705e-3 |
| logits `rel_l2` | ≤ 1.756e-5 | ≤ 5.320e-5 |
| `embed` agreement | **exactly 0** | **exactly 0** |

13,452 tensors, worst case 9.9e-5. `embed` is bit-identical on both, which is
the sanity check that the two sides are reading the same weights in the same
layout — a real mismatch there would be the first thing to show.

Per tensor kind, SmolLM2 (max over all 30 layers and 19 positions):

| kind | n | max `rel_l2` | | kind | n | max `rel_l2` |
|---|---|---|---|---|---|---|
| `down_proj` | 570 | 2.228e-5 | | `ln1` | 570 | 1.252e-5 |
| `logits` | 19 | 1.756e-5 | | `final_norm` | 19 | 1.072e-5 |
| `resid_mlp` | 570 | 1.719e-5 | | `up_proj` | 570 | 1.055e-5 |
| `mlp_act` | 570 | 1.680e-5 | | `q_proj` | 570 | 9.723e-6 |
| `attn_out` | 570 | 1.625e-5 | | `k_proj` | 570 | 9.636e-6 |
| `v_proj` | 570 | 1.625e-5 | | `gate_proj` | 570 | 8.835e-6 |
| `resid_attn` | 570 | 1.562e-5 | | `embed` | 19 | **0** |
| `o_proj` | 570 | 1.461e-5 | | | | |
| `ln2` | 570 | 1.257e-5 | | | | |

Nothing is an outlier. The spread across all fifteen computed kinds is 2.5×.

## 3. The one apparent anomaly, and why it is not one

The residual stream shows a spike. At SmolLM2 layer 9, `resid_mlp`'s relative
error jumps 8.79× over layer 8, and layer 29 jumps 2.4×; on Qwen the same
shape appears near layer 21. Read naively that is exactly what a divergent
layer looks like, so it was checked before anything was claimed.

It is **cancellation**, not divergence. At layer 9, position 0:

| | ‖·‖₂ (oracle) | ‖error‖₂ |
|---|---|---|
| `resid_attn` (the incoming residual) | 486.5 | 7.13e-4 |
| `down_proj` (what is added to it) | 414.1 | 1.12e-3 |
| `resid_mlp` (their sum) | **105.6** | 1.81e-3 |

The two addends nearly cancel — 486.5 + 414.1 of magnitude summing to 105.6,
a 4.6× cancellation — while their *absolute* errors simply add
(7.13e-4 + 1.12e-3 = 1.81e-3, which is the measured figure exactly). Relative
error therefore rises by precisely the cancellation factor, with nothing
computed differently. Every input to that add is at 1–3e-6 relative:

```
  L9.ln1        rel=8.4e-7     L9.ln2        rel=1.3e-6
  L9.q_proj     rel=5.9e-7     L9.gate_proj  rel=1.5e-6
  L9.k_proj     rel=7.2e-7     L9.up_proj    rel=1.3e-6
  L9.v_proj     rel=2.1e-6     L9.mlp_act    rel=2.0e-6
  L9.attn_out   rel=2.1e-6     L9.down_proj  rel=2.7e-6
  L9.o_proj     rel=2.1e-6     L9.resid_mlp  rel=1.7e-5   <- the sum
```

The relative error of a residual stream measures the conditioning of that
add, not the correctness of the layer. `compare_layers.py` therefore prints
the cancellation factor alongside the residual column, so the spike is
readable rather than alarming, and reports the layer's own tensors separately.

## 4. The measurement that actually closes §14.6

A compensating pair of errors has two halves, and both are visible in the
same table: a layer that **diverges** shows its own tensors far above the
error it was handed, and a layer that **compensates** shows the opposite —
a ratio far below 1. So the diagnostic is each layer's error against its
input's:

```
layer   carried-in      own ln1   ratio    worst own  /carried
    1    1.595e-06    1.387e-06    0.87    4.344e-06      2.72
    4    1.603e-06    1.894e-06    1.18    6.462e-06      4.03
    9    1.956e-06    1.855e-06    0.95    5.182e-06      2.65
   10    1.719e-05    1.252e-05    0.73    2.228e-05      1.30
   11    1.131e-05    9.514e-06    0.84    1.625e-05      1.44
   20    5.458e-06    6.007e-06    1.10    9.735e-06      1.78
   29    4.673e-06    5.241e-06    1.12    1.018e-05      2.18
```

(seven of thirty rows; the full table is in the raw log)

**Every layer's `ln1` is within 0.73–1.18× of the error it was handed**, on
all thirty layers of SmolLM2 and all twenty-four of Qwen. Worst amplification
of the carried-in error across a layer's whole computation: **4.07×**
(SmolLM2), **3.22×** (Qwen) — and that figure is uniform with depth, not
concentrated anywhere.

Layer 10's larger absolute figure, the only one that survives the
tripling filter, is inherited: it is handed 1.72e-5 by layer 9's cancellation
and *reduces* it to 1.25e-5 (ratio 0.73). It creates nothing.

There is no divergent layer and no compensating layer. The interior of the
stack tracks the oracle everywhere, so the possibility §14.6 could not
exclude is now excluded on the evidence, for these two models on this prompt.

## 5. What this does **not** establish

- **Not bit-exactness against `transformers`, and it must not be.** The
  oracle uses a different summation order and different kernels; ~1e-5
  relative is what fp32 agreement between two honest implementations looks
  like. CIS-2's bit-exactness claim is between *conforming implementations of
  the specification*, not between the specification and PyTorch. E28 is an
  accuracy check, not a determinism check.
- **Not exhaustive over inputs.** One prompt, 19 positions, two models. A
  compensating pair that only appears at a context length or an activation
  pattern outside this run is not excluded.
- **Not a check of §3.** The Qwen run supplies its prompt token ids from
  outside §3 — §14.8 / erratum E-3 — so it attests to §4–§11 only.
- **Not a claim that the oracle is right.** `transformers` is an independent
  implementation, not ground truth. What E28 shows is agreement between two
  independent implementations at every interior point, which is what §14.6
  asked for and what makes an undetected compensating pair implausible: it
  would have to occur identically in both.
- **Not run in CI.** `layer_dump` needs the §0 artifacts and the oracle needs
  torch, so neither can be a test. Both are reproducible on demand.

## 6. Provenance (Rule B)

| | |
|---|---|
| Host | `penguin` — ChromeOS Crostini (crosvm VM), Debian 13 trixie, x86_64, 8 cores. **VM: no timing numbers reported** (Rule A). |
| Toolchain | `rustc 1.98.0 (88d9e12ae 2026-08-18)`, `--release`, `--offline`, `--features layerdump` |
| FP environment | `x86_64 MXCSR FTZ(bit 15)+DAZ(bit 6)`, `fpenv::pin_and_selftest()` |
| Oracle | `torch 2.14.0+cpu`, `transformers 5.16.1`, fp32, `torch.set_num_threads(1)`, `use_deterministic_algorithms(True)`, TF32 off |
| Repo / branch | `cis2-spec`, `cm/cis2-verify-standalone` |
| Verifier dumps | `~/e28-cis2-layers.bin` (51,750,154 B), `~/e28-cis2-layers-qwen.bin` (103,942,814 B) |
| Oracle dumps | `~/e28-oracle-layers.bin` (25,875,077 B), `~/e28-oracle-layers-qwen.bin` (51,971,407 B) |
| Comparison logs | `~/e28-compare-smollm2.log`, `~/e28-compare-qwen05b.log` |
| Digests reproduced by the instrumented build | SmolLM2 `witness d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df`, `argmax 0b9c8f3ac90d0b9cd5f1719ac327dca1fc639fd87468305fccebbe3d56f67aff`; Qwen `witness c9dff099d927a6dda91514c260c7207de7aca4b361291e57fae390a352392e62`, `argmax 9619177f959f63d175d9442e73a6063f31654b08ba4e38717e88bd2e39b0aff5` |
| Independent token check | the oracle's own argmax at positions 3..18 reproduces all 16 generated ids on both models, with no reference to the verifier's output |
| Gates after this change | `cis2-verify/tools/check_no_fma.sh` PASS; lib + end-to-end tests pass; `cis2-verify run ../weights` reproduces the pinned digests, `conformance=PASS` |

Everything above is re-derivable: build with `--features layerdump`, run
`layer_dump`, run `scripts/oracle_layers.py` on the same token ids, run
`scripts/compare_layers.py` on the two files.
