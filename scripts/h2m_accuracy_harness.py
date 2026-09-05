#!/usr/bin/env python3
"""E15m accuracy harness: re-run H2's sin_pinned/cos_pinned ULP/relative-
error characterization (docs/H2_TRANSCENDENTAL_RIGOR.md) after CIS-2 spec
v0.3's f64-staged range reduction fix (docs/CIS2_SPEC_v0.3.md §6.3), over
the full RoPE position domain up to max_position_embeddings=8192, not just
the old pos<=19 pinned-decode-length domain.

Ports the pinned Rust math (src/math.rs sin_pinned/cos_pinned, both the old
v0.2 f32-only reduce_2pi and the new v0.3 f64-staged reduce_2pi) into
Python using numpy.float32 for f32 ops (IEEE-754 binary32, correctly
rounded, matches hardware) and native Python float (a C double, IEEE-754
binary64) for the f64 stage. Oracle: mpmath at 60 decimal digits, giving
the correctly-rounded fp32 sin/cos value by nearest-candidate search.

Not part of CIS-2's normative reference; evidence only.
"""
import math
import sys

import numpy as np
import mpmath as mp

mp.mp.dps = 60

f32 = np.float32

SIN_COEF = [f32(v) for v in [
    1.0, -1.666_666_7e-1, 8.333_333e-3, -1.984_127e-4, 2.755_732e-6,
    -2.505_211e-8, 1.605_904e-10, -7.647_164e-13, 2.811_457e-15,
    -8.220_635e-18, 1.957_294e-20,
]]
COS_COEF = [f32(v) for v in [
    1.0, -5.0e-1, 4.166_667e-2, -1.388_889e-3, 2.480_159e-5, -2.755_732e-7,
    2.087_676e-9, -1.147_075e-11, 4.779_477e-14, -1.561_921e-16,
    4.110_318e-19,
]]

TWO_PI_HI = f32(6.283_185)
TWO_PI_LO = f32(2.383_651e-7)
TWO_PI_F64 = 6.283_185_307_179_586_476_925_286_766_559
PI_2_F64 = 1.570_796_326_794_896_619_231_321_691_639_8

SIN_C0 = f32(-1.666_665_461_1e-1)
SIN_C1 = f32(8.332_160_873_6e-3)
SIN_C2 = f32(-1.951_529_589_1e-4)
COS_C0 = f32(4.166_664_568_298_827e-2)
COS_C1 = f32(-1.388_731_625_493_765e-3)
COS_C2 = f32(2.443_315_711_809_948e-5)


def round_half_away_from_zero(x: float) -> float:
    """Rust f64::round() semantics (ties away from zero)."""
    return math.copysign(math.floor(abs(x) + 0.5), x)


def reduce_2pi_v02(x: f32) -> f32:
    k = f32(round_half_away_from_zero(float(x) / float(TWO_PI_HI)))
    khi = f32(k * TWO_PI_HI)
    r0 = f32(x - khi)
    klo = f32(k * TWO_PI_LO)
    return f32(r0 - klo)


def reduce_2pi_v03(x: f32) -> f32:
    xd = float(x)  # exact widen f32->f64
    k = round_half_away_from_zero(xd / TWO_PI_F64)
    khi = k * TWO_PI_F64
    r64 = xd - khi
    return f32(r64)  # single correctly-rounded narrow


def reduce_pi_2_v03b(x: f32):
    xd = float(x)
    k = round_half_away_from_zero(xd / PI_2_F64)
    khi = k * PI_2_F64
    r64 = xd - khi
    return f32(r64), int(k)


def sin_poly_v03b(r: f32) -> f32:
    r2 = f32(r * r)
    inner = SIN_C2
    inner = f32(f32(inner * r2) + SIN_C1)
    inner = f32(f32(inner * r2) + SIN_C0)
    r3 = f32(r2 * r)
    term = f32(inner * r3)
    return f32(r + term)


def cos_poly_v03b(r: f32) -> f32:
    r2 = f32(r * r)
    inner = COS_C2
    inner = f32(f32(inner * r2) + COS_C1)
    inner = f32(f32(inner * r2) + COS_C0)
    r4 = f32(r2 * r2)
    term = f32(inner * r4)
    half_r2 = f32(f32(0.5) * r2)
    step1 = f32(f32(1.0) - half_r2)
    return f32(step1 + term)


def sin_pinned_v03b(x: f32) -> f32:
    r, k = reduce_pi_2_v03b(x)
    sr, cr = sin_poly_v03b(r), cos_poly_v03b(r)
    q = ((k % 4) + 4) % 4
    return [sr, cr, f32(-sr), f32(-cr)][q]


def cos_pinned_v03b(x: f32) -> f32:
    r, k = reduce_pi_2_v03b(x)
    sr, cr = sin_poly_v03b(r), cos_poly_v03b(r)
    q = ((k % 4) + 4) % 4
    return [cr, f32(-sr), f32(-cr), sr][q]


def horner_cos(r: f32) -> f32:
    r2 = f32(r * r)
    poly = COS_COEF[10]
    for i in range(9, -1, -1):
        poly = f32(f32(poly * r2) + COS_COEF[i])
    return poly


def horner_sin(r: f32) -> f32:
    r2 = f32(r * r)
    poly = SIN_COEF[10]
    for i in range(9, -1, -1):
        poly = f32(f32(poly * r2) + SIN_COEF[i])
    return f32(poly * r)


def sin_pinned(x: f32, reducer) -> f32:
    return horner_sin(reducer(x))


def cos_pinned(x: f32, reducer) -> f32:
    return horner_cos(reducer(x))


def f32_ulp_distance(got: f32, want_mp) -> float:
    """ULP distance between an f32 value and an mpmath high-precision
    reference value, measured in f32 ULPs at the reference's magnitude."""
    want = mp.mpf(want_mp)
    got_mp = mp.mpf(float(got))
    diff = abs(got_mp - want)
    # f32 ULP at the reference magnitude (or at 1.0 if the ref is 0)
    mag = max(abs(want), mp.mpf(1e-38))
    exp = mp.floor(mp.log(mag, 2))
    ulp = mp.mpf(2) ** (exp - 23)
    if ulp == 0:
        return 0.0
    return float(diff / ulp)


def rel_err(got: f32, want_mp) -> float:
    want = mp.mpf(want_mp)
    if want == 0:
        return float(abs(mp.mpf(float(got))))
    return float(abs(mp.mpf(float(got)) - want) / abs(want))


def inv_freq_table(theta: float, head_dim: int):
    half = head_dim // 2
    return [theta ** (-(2 * i) / head_dim) for i in range(half)]


def characterize(theta: float, head_dim: int, max_pos: int, label: str):
    inv_freq = inv_freq_table(theta, head_dim)
    positions = list(range(0, min(20, max_pos))) + list(range(0, max_pos, 7)) + [max_pos - 1]
    positions = sorted(set(p for p in positions if 0 <= p < max_pos))

    # ZERO_CROSS_FLOOR: |sin(r)| or |cos(r)| below this is treated as "near
    # a zero crossing" -- ULP is not a meaningful metric there for ANY
    # fixed-precision fp32 trig function, because the absolute precision
    # floor of the reduced argument r (bounded below by f32's own ULP,
    # ~6e-8 near |r|<=pi) translates into unbounded ULP-of-result as the
    # true function value -> 0. This is a mathematical property of
    # floating-point trigonometric functions, not a defect of this fix
    # (see docs/E15m_RESULT.md gate 2 discussion).
    ZERO_CROSS_FLOOR = 1e-2

    variants = {
        "v0.2 (f32-only)": (lambda x: sin_pinned(x, reduce_2pi_v02), lambda x: cos_pinned(x, reduce_2pi_v02)),
        "v0.3 (f64-staged reduce, Taylor poly)": (lambda x: sin_pinned(x, reduce_2pi_v03), lambda x: cos_pinned(x, reduce_2pi_v03)),
        "v0.3b (octant reduce, minimax poly)": (sin_pinned_v03b, cos_pinned_v03b),
    }

    results = {}
    for tag, (sin_fn, cos_fn) in variants.items():
        max_ulp_sin = max_ulp_cos = 0.0
        max_rel_sin = max_rel_cos = 0.0
        max_ulp_sin_f = max_ulp_cos_f = 0.0  # filtered, away from zero crossings
        max_abs_sin = max_abs_cos = 0.0
        worst_pos = None
        for pos in positions:
            for iv in inv_freq:
                x = f32(pos * iv)
                xmp = mp.mpf(float(x))
                s_oracle = mp.sin(xmp)
                c_oracle = mp.cos(xmp)
                s_got = sin_fn(x)
                c_got = cos_fn(x)
                us = f32_ulp_distance(s_got, s_oracle)
                uc = f32_ulp_distance(c_got, c_oracle)
                rs = rel_err(s_got, s_oracle)
                rc = rel_err(c_got, c_oracle)
                asb = float(abs(mp.mpf(float(s_got)) - s_oracle))
                acb = float(abs(mp.mpf(float(c_got)) - c_oracle))
                if us > max_ulp_sin:
                    max_ulp_sin = us
                if uc > max_ulp_cos:
                    max_ulp_cos = uc
                if abs(s_oracle) >= ZERO_CROSS_FLOOR and us > max_ulp_sin_f:
                    max_ulp_sin_f = us
                if abs(c_oracle) >= ZERO_CROSS_FLOOR and uc > max_ulp_cos_f:
                    max_ulp_cos_f = uc
                if asb > max_abs_sin:
                    max_abs_sin = asb
                if acb > max_abs_cos:
                    max_abs_cos = acb
                if rs > max_rel_sin:
                    max_rel_sin = rs
                    worst_pos = pos
                if rc > max_rel_cos:
                    max_rel_cos = rc
        results[tag] = (max_ulp_sin, max_rel_sin, max_ulp_cos, max_rel_cos, worst_pos,
                         max_ulp_sin_f, max_ulp_cos_f, max_abs_sin, max_abs_cos)

    print(f"== {label} (theta={theta}, head_dim={head_dim}, max_pos={max_pos}) ==")
    for tag, (mus, mrs, muc, mrc, wp, musf, mucf, mabs, mabc) in results.items():
        print(f"  {tag}: sin max_ulp={mus:.1f} (filtered>={ZERO_CROSS_FLOOR}: {musf:.2f}) "
              f"max_rel={mrs:.3e} max_abs={mabs:.3e} | "
              f"cos max_ulp={muc:.1f} (filtered: {mucf:.2f}) max_rel={mrc:.3e} max_abs={mabc:.3e} | "
              f"worst_pos(rel)={wp}")
    return results


if __name__ == "__main__":
    max_pos = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    characterize(100000.0, 64, max_pos, "SmolLM2-135M")
    characterize(1000000.0, 64, max_pos, "Qwen2.5-0.5B")
