# E37 — the same six cells on the second model, and what it does to §14.6's closing row

**Status: complete. Result: E36's "still one model" limit is closed, E28's Qwen
`rel_l2` figure survives the sweep, and the row that §14.6 says *closes the
clause* turns out to be two per-prompt numbers quoted as general bounds.**

64,260 intermediate tensors from six prompt/length cells on Qwen2.5-0.5B, every
cell `TOKEN CHECK PASS` and `SUPPLIED-ID CHECK PASS`, 0 shape mismatches, 0
differing duplicate records. Taken with E36 this is 144,825 tensors across two
architectures.

## 1. What E36 left open, in its own words

`E36_ORACLE_PROMPT_SWEEP.md` §4, as it stood before this run (that bullet now
reads CLOSED and points here):

> **Still one model.** All six cells are SmolLM2-135M. E28's Qwen-0.5B cell is
> not re-run here, so "six cells" means six inputs, not six architectures.

E36 also established the discipline that makes this experiment obligatory
rather than optional: *any maximum reported from a single input is a per-input
maximum until a sweep says otherwise*. E28's Qwen figures are exactly such
maxima. E36 corrected the SmolLM2 one and left its sibling unchecked.

## 2. Method, and one addition

Identical to E36 — teacher-forced oracle stepping `prompt_ids +
generated_ids[:-1]` one token at a time with a KV cache, token ids taken from
the verifier's own dump, agreement on *which* token checked against the
oracle's independent per-step argmax, dumps deleted between cells.

Qwen2.5-0.5B's `tokenizer.json` is refused by §3.1.3/§3.1.4 (§14.8 / erratum
E-3), so the prompt token ids are **supplied** from the oracle's tokenizer.
Such a run exercises §4–§11 only and is not a conformance run — the same
footing E28's Qwen cell stood on. E37 adds a check E28 did not have: the ids
`layer_dump` reports having used are compared against the ids it was handed, so
"supplied" cannot silently become "re-derived". **`SUPPLIED-ID CHECK PASS` on
all six cells.**

## 3. Result

| cell | prompt | gen | feed | tensors | worst `rel_l2` | worst `rel_rms` |
|---|---|---|---|---|---|---|
| 1 | `Once upon a time` | 16 | 19 | 5,985 | **9.923e-05** `p0.L22.o_proj` | 2.705e-03 `p0.L21.resid_mlp` |
| 2 | `The quick brown fox…` | 16 | 24 | 7,560 | 7.362e-05 `p0.L22.attn_out` | **2.883e-03** `p0.L22.mlp_act` |
| 3 | `1234567890 + 9876543210 =` | 16 | 38 | 11,970 | 9.787e-05 `p0.L22.down_proj` | 2.528e-03 `p0.L22.resid_attn` |
| 4 | `def f(x): return x * 2` | 16 | 24 | 7,560 | 1.730e-05 `p0.L22.ln1` | 4.181e-04 `p0.L22.ln1` |
| 5 | `Once upon a time` | 64 | 67 | 21,105 | **9.923e-05** `p0.L22.o_proj` | 2.705e-03 `p0.L21.resid_mlp` |
| 6 | `A` | 32 | 32 | 10,080 | 4.639e-05 `p0.L22.resid_attn` | 1.376e-03 `p0.L22.mlp_act` |

Cell 1 reproduces E28's published Qwen numbers exactly — 5,985 tensors,
9.923e-05 at `p0.L22.o_proj`, 2.705e-03 at `p0.L21.resid_mlp`, witness digest
`c9dff099…`, argmax digest `9619177f…`. That is what validates the plumbing
before the other five cells are believed.

### 3.1 The `rel_l2` figure survives; the `rel_rms` figure does not

**E28's 9.923e-05 is the maximum over all six cells**, not merely over its own.
Where E36 found the SmolLM2 prompt understating the worst case by 1.53x, the
Qwen prompt happens to *be* the worst case. That is worth stating plainly
because it is the outcome that makes E36's headline look less impressive: a
single-prompt maximum is not reliably an underestimate, it is simply unknown
until swept. On one model E28 was low; on the other it was exact. Neither could
have been predicted from the other.

The `rel_rms` figure is a different story: cell 2 reaches **2.883e-03**, above
E28's 2.705e-03. A 1.07x correction rather than 1.53x, but the same defect.

### 3.2 A second architecture reproduces E36's context-length control

Cells 1 and 5 are the same prompt at 19 and 67 fed positions. Their worst
`rel_l2` and worst `rel_rms` are **identical, at identical tensors** — exactly
as on SmolLM2. The finding that interior error is set inside a layer rather
than accumulated across positions now holds on two model families.

### 3.3 Qwen concentrates its worst tensor at depth; SmolLM2 does not

Every one of the six Qwen cells puts its worst tensor in **L22 of 24**. On
SmolLM2 the worst tensor moved around (L10, L11, L2, L12, L13 across cells).
The mechanism is visible in the cancellation column: Qwen's residual add at
**L21** cancels by 43.8x–68.1x depending on cell, against SmolLM2's 17.1x–20.6x
at L29, and the layer it hands that inflated relative error to is L22. Deep
cancellation raising the relative measure downstream of itself is E28 §3's
mechanism and not a new phenomenon — but it does mean the two models' error
profiles are shaped differently, and a bound taken from one says little about
the other.

## 4. The adversarial check, and what it does to §14.6's closing row

### 4.1 The check that closes the clause still holds, on every row

A compensating pair would show as a residual-stream spike **larger** than its
layer's own-tensor error times that layer's cancellation factor. Over all 144
(cell, layer) rows here:

```
max  resid_l2 / (own_l2 x cancellation)  =  0.557   (cell 3, L0)
                                  min      0.060
max  resid rel_l2                        =  9.081e-05
```

Every row below 1, as on SmolLM2 (max 0.848 there). Across both sweeps that is
**324 rows, none unexplained**. This is the test that excludes a compensating
pair, and E37 strengthens it rather than disturbing it.

### 4.2 But §14.6's stated bounds are per-prompt numbers

§14.6 leans on a different row, and calls it "the one that closes the clause":

> Measured, **every layer's first computed tensor is within 0.73-1.18x of the
> error handed to it**, on all 30 layers of the first model and all 24 of the
> second, with the worst whole-layer amplification bounded by 4.07x and uniform
> with depth.

Both figures are single-prompt measurements. Over twelve cells:

| | published in §14.6 | measured over six cells |
|---|---|---|
| ln1 / carried-in ratio, both models | 0.73–1.18x | **0.51–1.70x** |
| worst whole-layer amplification, SmolLM2 | 4.07x | **6.48x** (cell 3) |
| worst whole-layer amplification, Qwen | 3.22x | **4.83x** (cell 4) |

E28's own cell reproduces 4.07x, 3.22x and the 0.73 low end exactly, confirming
these are that cell's values and not a typo. **Erratum E-9** is filed against
§14.6.

### 4.3 Does the wider interval break the conclusion? No — and here is the test

The interval width is not itself evidence of a compensating layer. The decisive
question is whether an extreme ratio is a property of the *weights* or of the
*prompt*. A genuinely divergent or compensating layer is structural: it must
appear at the same layer, in the same direction, on every input.

It does not. The same layer takes opposite extremes on different prompts:

| model, layer | lowest across prompts | highest across prompts |
|---|---|---|
| Qwen `L22` | 0.73 (cell 2) | **1.61** (cell 4) |
| Qwen `L23` | 0.73 (cell 1) | 1.53 (cell 4) |
| SmolLM2 `L3` | 0.68 (cell 2) | 1.48 (cell 3) |
| SmolLM2 `L12` | 0.53 (cell 4) | 1.20 (cell 6) |

Qwen's layer 22 — the same weights, the same code — hands back 0.73x the error
it was given on one prompt and 1.61x on another. That is activation-dependent
scatter in a ratio of two small relative errors, not a layer that creates or
destroys error. The cell-1/cell-5 control says the same thing from the other
side: the same prompt at 19 and 67 fed positions gives ratios identical on all
23 Qwen layers, and differing on only 3 of SmolLM2's 29 — and those three are
early layers where the longer run's extra positions raise a per-layer maximum,
not a change of direction. The scatter tracks the prompt, not the run length.

An honest restatement of the §14.6 row is therefore: *no layer's own tensors
depart from the error handed to it by more than about 1.7x in either direction,
and no layer's departure is reproducible across inputs* — with the compensating
pair excluded by §4.1's residual test rather than by the width of this
interval.

## 5. What this establishes, and what it does not

**Establishes.** E36's "still one model" limit is closed: the six-cell
comparison now covers both §0 models, 144,825 tensors, and 324 (cell, layer)
rows in which cancellation accounts for every residual-stream spike. E28's Qwen
`rel_l2` figure is confirmed as a genuine six-cell maximum. Two further
per-prompt numbers in §14.6 are corrected upward.

**Does not establish.**

- **Still not exhaustive.** Six prompts per model. No cell tried so far exceeds
  9.923e-05; that is not a proof that none can.
- **Two models, not many.** Both are decoder-only fp32 checkpoints of the
  shapes §0 admits. Nothing here speaks to an architecture outside §0.
- **§4–§11 only for Qwen.** The prompt token ids are supplied from outside §3
  (§14.8 / E-3), so the Qwen half attests to §4–§11. The `SUPPLIED-ID CHECK`
  makes that explicit rather than implicit, but does not change the scope.
- **Not a determinism check.** ~1e-4 relative against `transformers` is an
  accuracy check. Bit-exactness is a claim between conforming implementations
  of the specification.

## 6. Provenance (Rule B)

| | |
|---|---|
| Host | `penguin` — ChromeOS Crostini (crosvm VM), Debian 13 trixie, x86_64, 8 cores. **VM: no timing numbers reported** (Rule A). |
| Toolchain | `rustc 1.98.0 (88d9e12ae 2026-08-18)`, `--release`, `--offline`, `--features layerdump` |
| FP environment | `x86_64 MXCSR FTZ(bit 15)+DAZ(bit 6)`, `fpenv::pin_and_selftest()`, reported per cell |
| Oracle | `torch 2.14.0+cpu`, `transformers 5.16.1` (`~/venvs/torch`), fp32, `torch.set_num_threads(1)`, `use_deterministic_algorithms(True)`, TF32 off |
| Model | Qwen2.5-0.5B, `~/qwen05b` (`config.json`, `model.safetensors` 988,097,824 B, `tokenizer.json`) |
| Tokenization | SUPPLIED prompt token ids from the oracle's `AutoTokenizer` (§3.1.3/§3.1.4 refuse this `tokenizer.json`; §14.8 / erratum E-3). NOT a conformance run. |
| Repo / branch | `cis2-spec`, `cm/cis2-verify-standalone` |
| Sweep script | `scripts/e37_qwen_oracle_sweep.sh` — sha256 `76c29fb7809698cf735794e2c4ee699bb1770b7b03314c06143e147ea53821dc` (byte-identical to the on-box `~/e37-qwen-sweep.sh`; it reads its cell list from `~/e37-cells.txt`, committed here as `docs/logs/e37-cells.txt`, sha256 `3f718def1c311cc36c2255b588f4631b828387ce460e3a9ba25fe51dbdcc7a8e`) |
| Raw log | `docs/logs/e37-qwen-sweep.log` — sha256 `18cda9636cd580db447c0703d1c24b0fc85625edc557660650e1545c743f9e92` |
| Cell 1 digests | `witness c9dff099d927a6dda91514c260c7207de7aca4b361291e57fae390a352392e62`, `argmax 9619177f959f63d175d9442e73a6063f31654b08ba4e38717e88bd2e39b0aff5` — identical to E28's published pair |
| Cells 2–6 digests | recorded per cell in the raw log; each is the first published run of that cell on this model |
| Independent checks | per cell: `TOKEN CHECK` (oracle's own per-step argmax reproduces every verifier-generated id) and `SUPPLIED-ID CHECK` (the ids the verifier used are the ids it was handed) |

Re-derivable end to end: build `cis2-verify` with `--features layerdump`, copy
`docs/logs/e37-cells.txt` to `~/e37-cells.txt`, then
`bash scripts/e37_qwen_oracle_sweep.sh`.
