# E35 — Does a real decode ever produce a denormal? Measuring §1.3's reach, not its guard

**Status: closed by measurement, and then reopened and closed the other way. Zero of
21,248,280 `exp_pinned` evaluations across six prompts land in the window where §1.3's
FTZ/DAZ pin is digest-relevant, and the nearest approach is 26.79 away in ln-space. That
result is correct as measured and every number below stands.**

**CORRECTION 2026-09-09 (erratum E-10).** The sentence that used to end this header —
"E22 follow-up item 1 … is not constructible from these prompts on this checkpoint" — was
carried outward as if it settled the question. It does not.
[E38](E38_REACH_SWEEP.md) ran the same instrument on 20 cells across **both** §0 models,
adding non-ASCII prompts and decode lengths to 256 tokens, and **constructed one**:
Qwen2.5-0.5B, `"Once upon a time"`, 256 generated tokens, softmax argument reaching
**−88.369385**, with **2** of that cell's 22,626,240 arguments inside
`[-88.0, -87.33654)`. The margin this document called large is a property of *these six
prompts on this one checkpoint*, not of the models — §6 below said as much
("a checkpoint with a much wider logit spread would need re-measuring; the instrument now
exists to do that in one run"), and that is precisely what happened. What E38 then found
is that the denormal does **not** move the digest, so §1.3 still has no end-to-end
necessity witness — see E38 §4 and E22's updated M01 row.

## 1. What E22 left open, and why the existing counter could not close it

E22's necessity matrix mutated each spec clause and asked whether the pinned test vector
notices. Mutant M01 gutted §1.3 (FTZ/DAZ) and the digest did not move. E22 recorded the
caveat honestly:

> "SAME digest" is evidence that no denormal *changed a result*, which is slightly weaker
> than "no denormal ever arose"; the mutant is not instrumented to tell those apart.

E29 added a reach census and reported `softmax hit 6.2 LOW guard (arg < -88.0 -> 0.0) n=0`
on every prompt. That looks like it settles the question. It does not, and the reason is
the boundary the counter was drawn at.

§6.2's low guard is `x < -88.0 -> 0.0`. It fires *before* `exp` is evaluated, so on that
side the answer is `0.0` whether or not FTZ is pinned: an unclamped `exp(-88.5)` would have
been subnormal and flushed to the same zero. The guarded side is FTZ-**independent**. It is
the wrong side to count.

The side that matters is the one the guard does not cover.

## 2. The window, computed exactly and pinned against `exp_pinned` itself

`ln(f32::MIN_POSITIVE)` is −87.336544750553100, so `exp(a)` is subnormal for every `a`
below it. §6.2 clamps only `a < −88.0`. Between the two lies a band the spec computes
rather than clamps, and there the result is a subnormal that FTZ flushes to zero and an
unpinned implementation keeps:

| region | §6.2 | `exp_pinned` result | depends on §1.3? |
|---|---|---|---|
| `a < −88.0` | LOW guard fires | `0.0` | no — clamped before `exp` |
| `−88.0 ≤ a < −87.33654022216797` | computed | subnormal → `0.0` pinned, subnormal unpinned | **yes** |
| `a ≥ −87.33654022216797` | computed | normal | no |

The band's exclusive top is pinned as an exact f32 bit pattern, `0xC2AE_AC4F`
(−87.33654022216797), and the test
`ops::census::tests::subnorm_band_edge_is_exp_pinneds_own_boundary` checks it against
`exp_pinned` rather than against `libm`: `exp_pinned(0xC2AE_AC4F)` is normal, the next f32
away from zero is not. The assertion is written as `is_normal()` on both sides so it holds
whether or not the harness thread has §1.3 pinned — with the pin a band argument returns
`+0.0`, without it a subnormal, and neither is normal.

§6.4's SiLU evaluates `exp_pinned(-x)`, so it has the same window mirrored: `x` in
`(87.33654022216797, 88.0]`. Above `88.0` the low guard fires on `-x` and clamps. Both
windows are counted, so this is a statement about the whole `exp` route rather than about
softmax alone.

## 3. A counter that reads zero must be shown able to fire

`n=0` from a dead instrument and `n=0` from a live one are the same observation, so the
band counters carry positive controls that drive real arguments through the real ops:

| control | required | result |
|---|---|---|
| `softmax_seq([0.0, −87.5])` | band 1, guard 0, calls 2 | as required |
| `softmax_seq([0.0, −90.0])` | band 0, guard 1 | as required |
| `softmax_seq([0.0, −80.0])` | band 0, guard 0 | as required |
| `note_silu(87.5)` / `note_silu(−90.0)` / `note_silu(0.0)` | band 1, below-88 1, neither 1 | as required |

`subnorm_band_counter_fires_on_a_constructed_gap` and
`silu_subnorm_band_is_the_softmax_band_mirrored` pin these. The second also asserts
`(-SUBNORM_TOP).to_bits() == SUBNORM_TOP.to_bits() ^ 0x8000_0000`, so the two windows share
one constant instead of two independently rounded literals.

## 4. Result

Six prompts, contexts to 64 generated tokens, all on the pinned SmolLM2-135M artifacts with
§1.3 pinned and self-tested.

| prompt | gen | softmax `exp` args | min softmax arg | margin to band | in band | SiLU calls | max SiLU arg | in band |
|---|---|---|---|---|---|---|---|---|
| `Once upon a time` | 16 | 102,600 | −35.557106 | 51.78 | **0** | 1,751,040 | 49.152660 | **0** |
| `The quick brown fox jumps over the lazy dog` | 16 | 162,000 | −39.668365 | 47.67 | **0** | 2,211,840 | 49.108852 | **0** |
| `1234567890 + 9876543210 =` | 16 | 400,140 | −60.541702 | **26.79** | **0** | 3,502,080 | 49.034283 | **0** |
| `def f(x): return x * 2` | 16 | 175,500 | −37.359097 | 49.98 | **0** | 2,304,000 | 48.924583 | **0** |
| `Once upon a time` | 64 | 1,230,120 | −39.709507 | 47.63 | **0** | 6,174,720 | 49.152660 | **0** |
| `A` | 32 | 285,120 | −35.568474 | 51.77 | **0** | 2,949,120 | 49.148216 | **0** |

Totals: 2,355,480 softmax `exp` arguments and 18,892,800 SiLU calls, 21,248,280 `exp_pinned`
evaluations, **zero** in either subnormal window. The §6.2 low guard also never fired
(`n=0` everywhere), reproducing E29 — but that is now the FTZ-independent half of the
picture rather than the whole of it.

The nearest approach is the arithmetic prompt at −60.541702, which is 26.79 short of the
band. In value terms its `exp` is 5.09e−27, about 4.3e11 times larger than the smallest
normal f32. This is not a near miss.

## 5. The instrument does not perturb the result

The census only reads the values it counts. Every witness digest and argmax digest in this
sweep is byte-identical to the ones E30 recorded for the same six cells — for example
`d82743059d1db929…` / `0b9c8f3ac90d0b9c…` on `Once upon a time` at 16 tokens. A census
build announces `build=instrumented (NOT a conforming verifier)` on its own output and is
not one.

## 6. What this establishes, and what it does not

Establishes:

- On this checkpoint and these six prompts, §1.3 is not observable in the digest through the
  `exp` route, and now says so by measurement rather than by the absence of a counter. E22's
  M01 null result is explained rather than merely reproduced: the mutant did not move the
  digest because the vector never reaches an argument where FTZ and no-FTZ differ.
- E22 follow-up item 1 asked for "a denormal-bearing decode vector". With a 26.79 margin in
  ln-space, ordinary prompts on this checkpoint are not going to supply one by chance.
  Decode-level coverage of §1.3 is the wrong instrument for the job; op-level goldens are the
  right one, and `fpenv`'s `pinned_denormal_goldens` and `clearing_the_pin_changes_the_answers`
  already provide exactly that — the second demonstrates that clearing the pin moves every FTZ
  case and no inert one.
  **Superseded by E38:** *this* checkpoint is not going to supply one; the other §0 checkpoint
  does, at 256 generated tokens. The generalisation from "these six prompts" to "ordinary
  prompts" was the error, and it is the same shape as erratum E-9's — a maximum measured over
  chosen inputs, quoted as a bound over inputs nobody chose. Decode-level coverage turned out
  to be the right instrument after all, once pointed at a long enough decode on a second model.
  (E38 also found that the FTZ half of the op-level self-test cited here could not fail:
  erratum E-11.)

Does not establish:

- **The `exp` route only.** These counters see §6.2's `exp_pinned` arguments and §6.4's SiLU
  arguments. A denormal arising elsewhere — a matvec product, an rmsnorm intermediate, a RoPE
  term — is not counted here. For those, E22's M01 result stands as it was: no denormal
  *changed a result*.
- **Six prompts, one checkpoint, contexts to 64 generated tokens.** The margin is large rather
  than marginal, which is the reason to expect it to hold, but this is measurement and not
  proof. A checkpoint with a much wider logit spread would need re-measuring; the instrument
  now exists to do that in one run.
- **Nothing about performance.** No timing figures are reported from this run.

## 7. Provenance

- Host: penguin (ChromeOS Crostini VM, Debian 13 trixie, x86_64), rustc 1.98.0 (88d9e12ae
  2026-08-18), release profile, `--features census`.
- Repo: `cis2-spec` at `f8f72d5` plus this change; branch `cm/cis2-verify-standalone`.
- Artifacts: `model.safetensors` `80521b40281d6ce7…`, `config.json` `1d556eab73b69c7f…`,
  `tokenizer.json` `9ca9acddb6525a19…` (`weights/MANIFEST.sha256`).
- fp env: `x86_64 MXCSR FTZ(bit 15)+DAZ(bit 6)`, pinned and self-tested at start of every run.
- Sweep script: `~/e35-subnormal-sweep.sh`, sha256 `7dd16ea8b26ef621…`, copied to
  `scripts/e35_subnormal_sweep.sh`. Prompt list: `~/e30-prompts.txt`, the same six cells E30 used.
- Raw log: `docs/logs/e35-subnormal-sweep.log`.
- No timing figures are reported from this run, and none would be reportable from this host.
