# E40 — the two operations E39 could not reach: §5.1 reductions and §8 RMSNorm

**Status: in progress.** E39 established where §1.3's FTZ/DAZ pin *cannot* act
(`exp_pinned`'s output, exhaustively) and measured the one place in §10 where it
can (the softmax division: 0 FTZ-decisive quotients in 22,626,240, nearest miss
a factor of 1.32). It left a stated limit: **§8 matvec and §7 RMSNorm
intermediates are uncounted by any experiment in this repository**, so E22 item
1 — does a denormal ever arise in a real decode? — was narrowed, not closed.

E40 counts them, and finds a defect in E39's own instrument on the way.

## 1. The cheapest exact probe first: are any *weights* subnormal?

Every product in §5.2 has a weight as one operand. §1.3's DAZ reads a subnormal
**input** as zero, so a single subnormal weight would mean DAZ acts on every
product that uses it — no decode required to find out. Operands are scanned as
the decode sees them, after §2.5 widening, which is what `safetensors::load`
produces.

| checkpoint | tensors | operands | subnormal | exact zeros | smallest nonzero (normal) |
|---|---|---|---|---|---|
| §13.1 reference (SmolLM2-135M) | 272 | 134,515,008 | **0** | 0 | `0x30880000` = 9.895302e-10 |
| Qwen2.5-0.5B | 290 | 494,032,768 | **0** | 1,677 | `0x312b0000` = 2.4883775e-9 |

**0 of 628,547,776 weight operands across both §0 models are subnormal**, and the
smallest nonzero weight in either is some 29 orders of magnitude above the
subnormal boundary (`1.1755e-38`). The detector is shown able to classify one
(`0x00000001` → subnormal) in the same run.

So DAZ takes no decision on any §5.2 weight operand in either model. That is a
static property of the checkpoints, independent of prompt, decode length and
tokenizer — the first result in this series that does not carry a "one vector"
caveat.

## 2. That leaves products, partial sums, and RMSNorm intermediates

A product of two normals can still be subnormal, and a partial sum driven near
zero by cancellation can be too — E38 §4.1 named exactly that gap and did not
test it. So E40 instruments the intermediates themselves:

* `dot_seq` (§5.1, and therefore all of §5.2 matvec): **every product** `a[i]*b[i]`
  and **every partial sum** `acc + p`.
* `rmsnorm` (§8): `x[i]*x[i]`, `x[i]*inv`, and `scaled*weight[i]`.

Each is classified by the E39 predicate: the exact value is formed in f64, and
counted `FTZ-DECISIVE` only if it is a nonzero subnormal **at or above half the
smallest subnormal** (`7.0065e-46`), which is where FTZ changes the stored bits
rather than merely agreeing with round-to-nearest.

## 3. The control found a defect in E39's instrument

E35's rule — a counter that reads zero must be shown able to fire — applied
again, and this time it caught something. The first control run reported an
RMSNorm case whose stored bits **differ** pinned vs unpinned
(`0x00000000` vs `0x006ce3eb`) and scored it `ftz_decisive=0`.

The cause: the census widened its operands with `x as f64`, which compiles to
`cvtss2sd` — an SSE conversion, and **§1.3's DAZ makes it read a subnormal
operand as zero**. A census built on `as f64` is therefore blind to exactly the
DAZ cases it exists to count. It sees `0.0`, concludes "not subnormal", and
reports a clean zero.

E39's `note_softmax_weight` shipped with this defect (commit `2e33e72`).

**Fix.** `census::widen_exact` decodes the f32 bit pattern with integer
arithmetic and rebuilds the value from a normal double, so every operand of
every SSE instruction in the census is an integer or a normal double and no
MXCSR bit can act on it.

**Does it change E39's published result?** No, and this is checked rather than
argued both ways. By construction: `as f64` and `widen_exact` differ only on a
subnormal operand, and E39's operands are `exp_pinned`'s output — never
subnormal, proven over all 2^32 arguments — over a denominator `>= 1`. By
measurement: re-running the 16-token cell with the corrected widening gives the
same `min-nonzero-|q| = 6.96465140988929e-25` and the same zero counts.

### 3.1 The corrected control, against pinned-vs-unpinned ground truth

| case | exact value | predicate | pinned | unpinned | differ |
|---|---|---|---|---|---|
| `MIN_POSITIVE * 1` | 1.1755e-38 | not decisive | `0x00800000` | `0x00800000` | no |
| `MIN_POSITIVE * 1e-2` | 1.1755e-40 | **DECISIVE** | `0x00000000` | `0x000147ae` | **yes** |
| `MIN_POSITIVE * 1e-7` | 1.1755e-45 | **DECISIVE** | `0x00000000` | `0x00000001` | **yes** |
| `MIN_POSITIVE * 1e-10` | 1.1755e-48 | not decisive | `0x00000000` | `0x00000000` | no |
| RMSNorm, subnormal weight | — | **DECISIVE** | `0x00000000` | `0x006ce3eb` | **yes** |
| `dot_seq` cancellation | — | **DECISIVE** | `0x00800000` | `0x00000001` | **yes** |

All six agree. The last two are the rows the old instrument got wrong.

## 4. Making it cheap enough to run at all

Classifying 37.5 billion intermediates through an atomic counter each is not
affordable. The per-element cost is instead **two integer compares** on the bit
patterns:

* `maybe_tiny_mul(a, b)`: the product is at least `2^(ea + eb - 254)`, so
  `ea + eb >= 128` already guarantees a magnitude of at least `2^-126` — a
  normal. The filter admits everything up to `ea + eb <= 130`, two binades of
  margin. A zero exponent field always passes, so DAZ cases are never filtered
  out.
* `maybe_tiny_add(a, b)`: `|a + b| <= 2·max(|a|,|b|)`, so a subnormal sum needs
  the larger operand below `2^-125` — exponent field at most 2, filtered at 4
  for margin.

The exact f64 classification runs only for intermediates the filter admits. The
intermediate *total* is accumulated once per `dot_seq` call rather than once per
element. Both filters are integer arithmetic on bit patterns and cannot
themselves be perturbed by MXCSR; both are conservative supersets, so a
filtered-out intermediate is provably not subnormal.
