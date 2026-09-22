# Hall of Divergence

Credit for people who tried to break CIS-2 (`docs/CIS2_SPEC_v0.3b.md`).
See `CHALLENGE.md` for what counts as a divergence, a reproduction, or a
conformance-environment failure, and how to report one. Only runs with a
full log a reporter (or we) can point at are listed here — no cash, no
prize, no payment: credit only.

## Verified reproductions

| Date | Reporter | Machine (hostname / CPU) | ISA | OS / kernel | Result | Log |
|------|----------|---------------------------|-----|-------------|--------|-----|
| 2026-09-22 | Aefinity AI (internal, box1/box2) | aefinity-box (Intel i5-5200U) | x86_64, AVX2 | Debian 13 trixie, Linux 6.12.94+deb13-amd64 | 9 passed, 0 failed | not yet committed to this public repo — run predates this PR; `selfcheck.sh` lands in PR #21, a public log commit is a follow-up |
| 2026-09-22 | Aefinity AI (internal, box1/box2) | aefinity-box2 (Intel Celeron N4020) | x86_64, scalar (no AVX2) | Debian 13 trixie, Linux 6.12.94+deb13-amd64 | 9 passed, 0 failed | not yet committed to this public repo — run predates this PR; `selfcheck.sh` lands in PR #21, a public log commit is a follow-up |

The two rows above are internal Aefinity AI runs, not third-party
reproductions — they seed the table's format. See the README's own
machine table, `.github/workflows/verify.yml`, and `docs/GPU_RESULT.md`
for the project's other own-fleet results. Third-party reports go here
too, once reproduced.

## Verified divergences

| Date | Reporter | Machine (hostname / CPU) | ISA | OS / kernel | What diverged | Resolution | Log |
|------|----------|---------------------------|-----|-------------|----------------|------------|-----|
| — | — | — | — | — | — | — | — |

*No divergence reported yet.*
