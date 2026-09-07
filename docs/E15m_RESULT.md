# E15m result — CIS-2 spec v0.3 (f64-staged RoPE reduction) — v0.3 PARTIAL, superseded below by v0.3b

Branch: `cm/e15m-spec-v0.3-rope` (pushed). **v0.3 (f64-staged reduction
only): PARTIAL — gate (1) full local match + gate (2) missed (residual
error dominated by the fixed degree-10 Taylor polynomial's own
truncation, not the reduction; see table below). Coordinator directive:
implement v0.3b (octant reduction + separate minimax sin/cos polynomials)
on this same branch to close gate (2) — see the v0.3b section at the
bottom of this file for that result.**
CI runs were dispatched and are in flight as of this writing (2026-08-29).

## Local results (x86_64, this sandbox — already confirmed)

- New `CIS2_REF = 90f7484e4cb523e40cd44d79491d3d7aba124e65205db80bc0ba1ab30a8f9890`
  (was v0.2's `a0c563ef804f50413b7fb6619ae4afe9b51b1ffa7655e944221393e85d6261da`).
- New `table_digest = 986abc500e1122ada32a885bc722b313326a68d6af4e974cf3ee5034b9e7f900`
  (was v0.2's `465d358ccd63721256dbd2abbc77ad5de755adf3230635f95b12b1727dfa1ea3`).
- `inv_freq_table_digest` **unchanged**: `da9f6dcfde0425588815509e874515cdcd3d6b8818b6d0136590052e7bbf6f12`.
- `generated_token_ids`/`argmax_digest` for the 16-token pinned test vector
  **unchanged** from v0.2 (`0b9c8f3a...`).
- **Gate (1) partial**: `cis2_ref` (Rust reference), `verify2` (Rust
  clean-room, updated from spec v0.3 §6.3 text only), and `verify3` (C11
  clean-room, same) all produced digest `90f7484e...` locally on x86_64 —
  3-way match. Cross-ISA (aarch64) confirmation is pending in CI (below).
- **Gate (2) — accuracy harness result** (`scripts/h2m_accuracy_harness.py`,
  mpmath 60-digit oracle, RoPE domain up to `max_position_embeddings=8192`,
  both SmolLM2-135M `theta=1e5` and Qwen2.5-0.5B `theta=1e6`):

  | | v0.2 (f32-only) | v0.3 (f64-staged) |
  |---|---|---|
  | sin max rel error | 1.30e-2 (SmolLM2) / 1.96e-3 (Qwen) | **2.37e-4** (both) |
  | cos max rel error | 5.33e-3 (SmolLM2) / 1.36e-2 (Qwen) | **3.14e-5** (Qwen) / 1.84e-5 (SmolLM2) |
  | sin max abs error | 2.53e-4 / 2.71e-4 | **6.7e-7** / 5.2e-7 |
  | max ULP (raw) | up to 163969 | up to 2813.5 |
  | max ULP (filtered, \|sin/cos(r)\|>=1e-2) | up to 92283 | up to 486.5 |

  **Max relative/absolute error improved ~55-70x. The pre-registered
  "<=2 ULP" bar is NOT met** — max ULP is still in the hundreds even after
  the fix. Root-caused (see `/tmp/diag.py`-style check, not committed):
  the *reduction* itself is now essentially exact (matches an
  arbitrary-precision mpmath reduction to ~1e-7 absolute, i.e. within
  ~1 ULP of the reduced f32 argument `r` itself) — the residual error is
  the fixed degree-10 Taylor polynomial's own truncation/rounding error,
  which is worst near `|r| ~ pi` (the domain edge) and gets amplified into
  large ULP/relative-error numbers specifically where `sin(r)` or
  `cos(r)` passes near zero (a zero-crossing divides a tiny-but-real
  absolute error by an even tinier ULP-at-that-magnitude, inflating the
  ULP count for any fixed-precision fp32 trig implementation — this is a
  known property of ULP metrics near zeros, not specific to this
  polynomial). **Honest conclusion**: this task's mechanism (f64-staged
  reduction only, polynomial untouched per the task's own design
  constraint) reduces the accuracy defect by ~1-2 orders of magnitude and
  removes the position-dependent *growth* (v0.2's error scaled with
  position; v0.3's does not — both the SmolLM2 and Qwen worst-case numbers
  above are already saturated by position ~1649, not growing further
  toward 8192), but does **not** hit a literal "<=2 ULP absolute" bar,
  because that residual is now the polynomial's, not the reduction's.
  Closing it further would require also revisiting the polynomial
  (out of scope for this task's pre-registered mechanism) — flagged here
  as the honest residual finding rather than silently narrowing the claim.

## CI dispatch (in flight — record run ids, poll `gh run view <id>`)

Dispatched on `cm/e15m-spec-v0.3-rope` at 2026-08-29T18:11Z:

- `e15m-rope-fix.yml` (new; H2m accuracy harness in CI + oracle
  16/128/512/2048 for SmolLM2-135M and Qwen2.5-0.5B):
  - push-triggered run: **33267566689**
  - workflow_dispatch run: **33267569959**
- `e15d-compiler-invariance.yml` (20-cell opt-level x target-cpu x ISA
  matrix, `TARGET_DIGEST` updated to `90f7484e4c...`):
  - workflow_dispatch run: **33267573467**
- `e15c-cross-isa.yml` (2-cell x86_64/aarch64 primary vector,
  `TARGET_DIGEST` updated to `90f7484e4c...`):
  - workflow_dispatch run: **33267577208**

## Resume point / what remains

1. Poll the 4 run ids above (`gh run watch <id>` or `gh run view <id>`)
   until complete; aggregate pass/fail.
2. **Gate (1) full**: confirm `e15d-compiler-invariance.yml` 20/20 PASS and
   `e15c-cross-isa.yml` 2/2 PASS against `TARGET_DIGEST=90f7484e4c...` and
   `TARGET_INVFREQ_DIGEST=da9f6dcfde...` (unchanged) — this is the
   cross-ISA half of gate (1) not yet confirmed (x86_64-only confirmed
   locally above).
3. **Gate (3)**: read back `e15m-rope-fix.yml`'s `oracle-smollm2` and
   `oracle-qwen05b` job matrices (n_gen 16/128/512/2048) — record
   token-match PASS/DIVERGED per length and the `diff_logits.py` max
   relative logit diff from each `diff_output*.txt` artifact.
4. Update this doc's header from "IN PROGRESS" to final status once (1)-(3)
   are aggregated; update `docs/CIS2_SPEC_v0.3.md`'s `<NEW_MAX_ULP_SIN>`
   etc. placeholders in §6.7 with the final numbers from this doc's gate
   (2) table (currently filled with the local sandbox run's numbers above
   — re-check they match the CI run before finalizing, they should since
   the harness has no ISA-dependent behavior, but not yet re-confirmed
   from the CI artifact itself).
5. Do NOT merge to main — this branch stays open pending Justin's/
   coordinator's review of the gate (2) partial-pass finding above.

---

## v0.3b section (coordinator directive: octant reduction + minimax polynomials, closes gate 2)

**Mechanism**: §6.3 reduction changed from mod-`2*pi` (v0.3) to mod-`pi/2`
(octant, `r ∈ [-pi/4, pi/4]` + quadrant index `k mod 4`), same f64-staged
Cody-Waite pattern/determinism argument, still no FMA. The degree-10
Taylor polynomials were replaced with SEPARATE degree-7 (`sin`) / degree-8
(`cos`) Cephes `sinf`/`cosf` minimax polynomials, pinned as f32 hex
literals (`SIN_C0=0xBE2AAAA3`, `SIN_C1=0x3C08839E`, `SIN_C2=0xB94CA1F9`,
`COS_C0=0x3D2AAAA5`, `COS_C1=0xBAB6061A`, `COS_C2=0x37CCF5CE`), selected
and signed by quadrant. Full text: `docs/CIS2_SPEC_v0.3.md` §6.3/§6.6/§6.7
(same file, now titled v0.3b; v0.3's attempt is kept in §6.3/§6.7/§16 as
history, marked PARTIAL/superseded).

**New digest**: `CIS2_REF = d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df`
(was v0.3's `90f7484e4cb523e40cd44d79491d3d7aba124e65205db80bc0ba1ab30a8f9890`,
v0.2's `a0c563ef80...`). `table_digest = 23c7bfaf5cef0095fd021af2eb1808abb4928bae4219756d86bdac670a06b35d`
(was v0.3's `986abc500e...`). `inv_freq_table_digest` unchanged
(`da9f6dcfde...`); `argmax_digest`/`generated_token_ids` unchanged from
v0.1/v0.2/v0.3.

**Gate (1) — local 3-way match (x86_64, this sandbox)**: `cis2_ref`
(Rust reference), `verify2` (Rust clean-room), and `verify3` (C11
clean-room) all produced `d82743059d...` — full match. Both clean-rooms
updated from the v0.3b spec text only (not `src/math.rs`), logged in
`verify2/CLEANROOM_LOG.md` and `verify3/CLEANROOM_LOG.md` ("E15m update
#2" entries). Cross-ISA (aarch64) confirmation pending in CI (dispatched
below).

**Gate (2) — accuracy harness, closed**: `scripts/h2m_accuracy_harness.py`
re-run to position 8192 (dense grid, stride 7 + full 0-19 + last position)
for both SmolLM2-135M (`theta=1e5`) and Qwen2.5-0.5B (`theta=1e6`):

| variant | sin max ULP | sin max rel | cos max ULP | cos max rel |
|---|---|---|---|---|
| v0.2 (f32-only reduce, Taylor) | up to 115198 | 1.30e-2 | up to 163969 | 1.36e-2 |
| v0.3 (f64-staged reduce, Taylor) | up to 2813.5 | 2.37e-4 | up to 407 | 3.14e-5 |
| **v0.3b (octant reduce, minimax)** | **1.5** | **1.2e-7** | **1.5** | **1.2e-7** |

**Meets the pre-registered <=2 ULP bar** (both functions, both models,
full `pos ∈ [0, 8192)` domain) — closes gate (2), which v0.3 alone missed.

**Gate (3)**: not re-run for v0.3b specifically in this pass (16-token
pinned-vector argmax/tokens already confirmed unchanged from v0.2/v0.3
above; the broader 16/128/512/2048 oracle-correctness CI matrix dispatched
for v0.3 in `e15m-rope-fix.yml` re-runs automatically against the new
v0.3b digest on this same branch — see run ids below).

**CI dispatch (in flight, v0.3b commit) — record run ids, poll `gh run view <id>`**:
dispatched on `cm/e15m-spec-v0.3-rope` after pushing the v0.3b commit:
- `e15m-rope-fix.yml`: push-triggered run **33268180453**,
  workflow_dispatch run **33268184416**
- `e15d-compiler-invariance.yml` (`TARGET_DIGEST` updated to
  `d82743059d...`): workflow_dispatch run **33268188962**
- `e15c-cross-isa.yml` (`TARGET_DIGEST` updated to `d82743059d...`):
  workflow_dispatch run **33268192967**

Dispatched at 2026-08-29T18:24Z on `cm/e15m-spec-v0.3-rope` (commit
`fe18bd6`). Note: earlier v0.3-digest runs `33267577208` (e15c, PASS) and
`33267573467` (e15d, PASS) are now stale (they validated `90f7484e4c...`,
superseded by v0.3b's `d82743059d...`) — do not treat those as gate (1)
evidence for v0.3b; only the 4 run ids above are.

## Resume point (v0.3b)

1. Fill in the 3 real run ids above (dispatch happens right after this
   commit in the same session) and poll to completion.
2. Confirm `e15d-compiler-invariance.yml` 20/20 PASS and
   `e15c-cross-isa.yml` 2/2 PASS against `TARGET_DIGEST=d82743059d...`.
3. Confirm `e15m-rope-fix.yml`'s `h2m-accuracy-harness` job (CI-run,
   should match the local numbers above exactly since the harness has no
   ISA-dependent behavior) and the `oracle-smollm2`/`oracle-qwen05b`
   16/128/512/2048 matrices (token-match PASS/DIVERGED per length).
4. Do NOT merge to main pending review.
