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
| 2026-10-16 | Aefinity AI (internal negative control, lab host C) | Intel HD Graphics 630 iGPU (Mesa rusticl, OpenCL 3.0), hardware divide/sqrt build | x86_64 host, Intel Gen9 iGPU | Debian 13 trixie | Not a divergence of the spec: the device's hardware divide/sqrt is not correctly rounded (`0x00ffffff / 0x00fffffe` gives `3f800000`, want `3f800001`), and the OpenCL build that uses it gives 16-token digest `8322617cc9f3ba3355eab391d066329b3d0cb8d0a9020c5765ffae60fc750acc` instead of the pinned `d82743059d…`; the lowest differing element is position 0, layer 0, `norm1`, index 0 (host `0x3cad866e`, device `0x3cad866f`) | Software divide/sqrt build on the same device reproduces the pinned digests at 16 and 128 tokens | [`docs/results/2026-10-16-OPENCL-BACKEND.md`](docs/results/2026-10-16-OPENCL-BACKEND.md) (builder log not committed; rows pending the pre-registered leg) |

*No third-party divergence reported yet.* The row above is an internal negative control, listed because it is a documented case of a conformance-environment failure in the sense of `CHALLENGE.md`.
