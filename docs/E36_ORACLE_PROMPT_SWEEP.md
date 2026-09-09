# E36 — E28's "one prompt, 19 positions", widened to six cells

**Status: complete. Result: E28's input scope note is closed for one model, and
E28's published SmolLM2 worst case was a per-cell maximum, not a per-model
one.** Across 80,565 intermediate tensors from six prompt/length cells, every
interior point of the verifier agrees with an independent `transformers` fp32
oracle to ~1e-5 relative — but the worst case is **3.406e-05**, 1.53x the
2.228e-05 E28 published from its single SmolLM2 prompt. Nothing about §14.6's
conclusion changes; the *number* does.

## 1. What E28 left open, in its own words

`E28_LAYER_ORACLE.md` §5 states the limit itself:

> **Not exhaustive over inputs.** One prompt, 19 positions, two models. A
> compensating pair that only appears at a context length or an activation
> pattern outside this run is not excluded.

That is a scope limit of the second kind: it needs no decision from anyone, only
a machine and more runs. E36 supplies the runs.

The six cells are E30's, unchanged, so this sweep sits on the same inputs as
E30, E33 and E35 and its digests are directly comparable to theirs:

| cell | prompt | gen | why this cell |
|---|---|---|---|
| 1 | `Once upon a time` | 16 | E28's own cell — the reproduction anchor |
| 2 | `The quick brown fox jumps over the lazy dog` | 16 | long English, wide token variety |
| 3 | `1234567890 + 9876543210 =` | 16 | digits only; degenerate repeated-token decode |
| 4 | `def f(x): return x * 2` | 16 | code; punctuation-dense |
| 5 | `Once upon a time` | 64 | cell 1 at 3.5x the context — a free control |
| 6 | `A` | 32 | minimal prompt, one token of context |

## 2. Method: teacher forcing, with the token agreement *checked*

The verifier never feeds the last token it generates, so the oracle is fed
`prompt_ids + generated_ids[:-1]` — one oracle forward per verifier forward —
stepped one token at a time with a KV cache. Two things make this a comparison
rather than an assumption:

- **The token ids come from the verifier's own dump**, parsed out of
  `layer_dump`'s `prompt-token-ids` / `generated-token-ids` lines, so there is
  no second tokenization to disagree about.
- **Agreement is checked, not assumed.** The oracle's per-step argmax is
  printed independently and compared against the verifier's generated ids. If
  the two implementations diverged on *which* token, the per-tensor comparison
  would be comparing different computations and every rel_l2 below would be
  meaningless. All six cells report `TOKEN CHECK PASS`.

Each cell writes both dumps, compares, and deletes them before the next cell
starts, so peak disk stays at one pair — largest is cell 5's 182,507,242 B
verifier dump — rather than the 558,394,570 B of verifier dumps alone that the
six cells would hold at once, plus their oracle counterparts.

## 3. Result

| cell | feed len | tensors | worst `rel_l2` | worst `rel_rms` | token check | shape mism. |
|---|---|---|---|---|---|---|
| 1 | 19 | 7,467 | 2.228e-05 `p0.L10.down_proj` | 4.094e-04 `p0.L9.resid_mlp` | PASS | 0 |
| 2 | 24 | 9,432 | **3.406e-05** `p0.L11.attn_out` | **8.935e-04** `p0.L10.mlp_act` | PASS | 0 |
| 3 | 38 | 14,934 | 1.872e-05 `p23.L2.mlp_act` | 6.633e-04 `p23.L2.mlp_act` | PASS | 0 |
| 4 | 25 | 9,825 | 1.950e-05 `p8.L12.o_proj` | 3.881e-04 `p0.L10.mlp_act` | PASS | 0 |
| 5 | 67 | 26,331 | 2.228e-05 `p0.L10.down_proj` | 4.094e-04 `p0.L9.resid_mlp` | PASS | 0 |
| 6 | 32 | 12,576 | 1.317e-05 `p1.L13.mlp_act` | 1.634e-04 `p1.L26.mlp_act` | PASS | 0 |

**80,565 tensors compared, 0 shape mismatches, 0 differing duplicate records,
and all twelve digests (six witness + six argmax) byte-identical to E30's and
E35's.** `embed` is exactly 0.000e+00 in every cell, as in E28: the two
implementations read the same embedding rows.

Cell 1 reproduces E28's published figures exactly — 7,467 tensors, 2.228e-05 at
`p0.L10.down_proj`, 4.094e-04 at `p0.L9.resid_mlp` — which is what validates
the plumbing before the other five cells are believed.

### 3.1 Reading 1 — E28's worst case was not the worst case

Cell 2 reaches **3.406e-05**, 1.53x E28's 2.228e-05, at a different tensor
(`attn_out`, not `down_proj`) in a different layer. E28 reported its number as
the worst over its run, which was true, and E36 does not contradict it — but
anyone who read 2.228e-05 as *the* agreement bound for this verifier read more
into it than one prompt could carry. The honest statement after E36 is that
across six cells the worst interior disagreement is 3.4e-05 relative, and that
this is still squarely what fp32 agreement between two independent
implementations looks like. **This is the reason a one-prompt maximum should
never be quoted as a bound.**

One qualification, in the direction that makes E36 *less* dramatic and must be
stated anyway: E28's other model already ran higher. Its Qwen2.5-0.5B cell
reported worst `rel_l2` 9.923e-05 — about 3x E36's worst — so 3.406e-05 is not
the largest interior disagreement anyone has published for this verifier, and
E36 is not a discovery that agreement is looser than believed. What E36
corrects is narrower and still worth correcting: the SmolLM2 figure of
2.228e-05 was a per-cell maximum being read as a per-model one, and it is not.

### 3.2 Reading 2 — 3.5x more context does not move the worst case

Cells 1 and 5 are the same prompt at 16 and 64 generated tokens: 19 versus 67
fed positions. Their worst `rel_l2` and worst `rel_rms` are *identical*, at
identical tensors. Positions 0–18 are a shared prefix and compute identically in
both cells, so this says the maximum over the 48 additional positions does not
exceed the maximum over the first 19. Error does not accumulate with context
length here; it is set by what happens inside a layer, not by how many
positions have gone before.

That also answers the "context length" half of E28's scope note directly, on
the one pair of cells where the comparison is controlled.

### 3.3 Reading 3 — the worst tensor is not a prefill artefact

In cells 1, 2 and 5 the worst tensor is at `p0`, which invites the reading that
this is something about the prefill position. Cells 3, 4 and 6 put it at `p23`,
`p8` and `p1`. It is not positional.

### 3.4 The adversarial check: does cancellation still account for every spike?

E28 §3 explained the residual-stream error as *cancellation, not divergence* —
when two large addends nearly cancel, relative error rises by exactly the
cancellation factor of the add. A compensating pair hiding in some other cell
would show up as a residual-stream spike **larger** than its layer's own-tensor
error times that layer's cancellation factor. So over all 180 (cell, layer)
rows:

```
max  resid_l2 / (own_l2 x cancellation)  =  0.848   (cell 1, L0, x1.12)
                                  next     0.797   (cell 6, L0, x1.13)
                                  min      0.039
max  resid rel_l2                        =  2.356e-05  (cell 2, L10)
```

Every row is **below 1**: cancellation fully accounts for the residual-stream
error in every layer of every cell, with nothing left over. Max cancellation is
17.12–20.60 depending on cell, always at layer 29, matching E28 §3's mechanism.
No row shows the unexplained jump that a compensating pair would produce.

## 4. What this establishes, and what it does not

**Establishes.** E28's §14.6 conclusion — two independent implementations agree
at *every* interior point, so an undetected compensating pair would have to
occur identically in both — now rests on six prompts, six context lengths and
80,565 tensors rather than one prompt and 7,467. The context-length half of the
scope note is closed by a controlled pair. E28's headline number is corrected
upward to 3.406e-05.

**Does not establish.** Everything else in E28 §5 still stands unchanged, and
E36 adds two limits of its own:

- **Still one model — CLOSED by E37.** All six cells here are SmolLM2-135M, so
  "six cells" meant six inputs, not six architectures. `E37_QWEN_ORACLE_SWEEP.md`
  reruns the identical six cells on Qwen2.5-0.5B (64,260 further tensors):
  E28's Qwen `rel_l2` of 9.923e-05 *survives* as the six-cell maximum — so a
  single-prompt figure is not reliably an underestimate, only unknown — while
  its `rel_rms` is exceeded (2.883e-03, 1.07x). E37 also finds a third instance
  of the same defect, this time in the published spec: §14.6's "0.73-1.18x" and
  "4.07x / 3.22x" are per-prompt values (erratum E-9).
- **Still not exhaustive.** Six prompts is more than one; it is not all
  prompts. The sweep is now cheap to extend — the script takes cells as
  `gen|prompt` lines — so the honest framing is that no cell tried so far
  exceeds 3.4e-05, not that none can.
- **Not a determinism check.** As in E28: ~1e-5 against `transformers` is an
  accuracy check. Bit-exactness is a claim between conforming implementations
  of the specification, and is covered elsewhere.
- **Not a check of §3.** The token ids come from the verifier's dump; §3 is
  attested separately.

## 5. Provenance (Rule B)

| | |
|---|---|
| Host | `penguin` — ChromeOS Crostini (crosvm VM), Debian 13 trixie, x86_64, 8 cores. **VM: no timing numbers reported** (Rule A). |
| Toolchain | `rustc 1.98.0 (88d9e12ae 2026-08-18)`, `--release`, `--offline`, `--features layerdump` |
| FP environment | `x86_64 MXCSR FTZ(bit 15)+DAZ(bit 6)`, `fpenv::pin_and_selftest()`, reported per cell by `layer_dump` |
| Oracle | `torch 2.14.0+cpu`, `transformers 5.16.1` (`~/venvs/torch`), fp32, `torch.set_num_threads(1)`, `use_deterministic_algorithms(True)`, TF32 off |
| Model / artifacts | SmolLM2-135M, `cis2-spec/weights`, §0 artifacts |
| Repo / branch | `cis2-spec`, `cm/cis2-verify-standalone` |
| Sweep script | `scripts/e36_oracle_sweep.sh` — sha256 `0f4e095c6ee0b0b3611467a3d54f9a5e94f907f3a4fed94799ae5f262115d8dd` (byte-identical to the on-box `~/e36-oracle-sweep.sh` it was copied from) |
| Raw log | `docs/logs/e36-oracle-sweep.log` — sha256 `b61488540d90d8f71ffa7b26c3e69f9b7da38645dc82bb7d8f8a6e33c4cd80a9`, exit code 0 |
| Digests | all twelve identical to E30's and E35's; cell 1 `witness d827430…`, `argmax 0b9c8f3…` also identical to E28's published pair |
| Independent token check | per cell, the oracle's own per-step argmax reproduces every verifier-generated id, with no reference to the verifier's output |

Re-derivable end to end: build `cis2-verify` with `--features layerdump`, then
`bash scripts/e36_oracle_sweep.sh`. It drives `examples/layer_dump`,
`scripts/oracle_layers.py` and `scripts/compare_layers.py` per cell and deletes
each dump pair before the next.
