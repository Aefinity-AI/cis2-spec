//! Spec 6: the pinned transcendental functions, and spec 6.6's table
//! digest over their coefficients.
//!
//! Every expression here is written as the specification writes it: one
//! IEEE-754 operation per step, strictly left to right, never `mul_add`
//! (spec 1.4 — a fused multiply-add is non-conforming even though it is
//! more accurate, because it produces different bits).

use crate::sha256::Sha256;
use crate::softfp::{floor_f32, round_ties_away_f64, sqrt_cr};

// --- 6.2 exp ---------------------------------------------------------------
pub const EXP_LOG2E: u32 = 0x3FB8AA3B;
pub const EXP_C1: u32 = 0x3F318000;
pub const EXP_C2: u32 = 0xB95E8083;
pub const EXP_P: [u32; 6] = [
    0x39506967, 0x3AB743CE, 0x3C088908, 0x3D2AA9C1, 0x3E2AAAAA, 0x3F000000,
];

// --- 6.3 sin/cos (v0.3b: octant reduction + separate minimax polys) --------
pub const PI_2_F64: u64 = 0x3FF921FB54442D18;
pub const SIN_C0: u32 = 0xBE2AAAA3;
pub const SIN_C1: u32 = 0x3C08839E;
pub const SIN_C2: u32 = 0xB94CA1F9;
pub const COS_C0: u32 = 0x3D2AAAA5;
pub const COS_C1: u32 = 0xBAB6061A;
pub const COS_C2: u32 = 0x37CCF5CE;

// --- 6.5 ln ----------------------------------------------------------------
pub const LOG_SQRTHF: u32 = 0x3F3504F3;
pub const LOG_Q1: u32 = 0xB95E8083;
pub const LOG_Q2: u32 = 0x3F318000;
pub const LOG_P: [u32; 9] = [
    0x3D9021BB, 0xBDEBD1B8, 0x3DEF251B, 0xBDFE5D4F, 0x3E11E9BF, 0xBE2AAE50, 0x3E4CCEAD, 0xBE7FFFFC,
    0x3EAAAAAA,
];

/// 2.3: the JSON `1e-05` parsed as f64 and narrowed to f32 by one RNE
/// rounding. Pinned so a JSON parser bug cannot move it silently.
pub const EPS_F32_BITS: u32 = 0x3727C5AC;

#[inline(always)]
fn f(bits: u32) -> f32 {
    f32::from_bits(bits)
}

/// 6.1 rsqrt, route (a): composed from two IEEE-754-mandatory
/// correctly-rounded operations. `sqrt_cr` is this crate's integer software
/// square root (see `softfp`), never an approximate hardware reciprocal.
#[inline]
pub fn rsqrt(x: f32) -> f32 {
    1.0f32 / sqrt_cr(x)
}

/// 6.2.1 `ldexp_exact`: replace only the 8-bit exponent field.
pub fn ldexp_exact(x: f32, k: i32) -> f32 {
    if x == 0.0 {
        return x;
    }
    let bits = x.to_bits();
    let exp_bits = ((bits >> 23) & 0xFF) as i32;
    let new_exp = exp_bits + k;
    if new_exp <= 0 {
        return 0.0;
    }
    if new_exp >= 0xFF {
        return if bits >> 31 == 1 {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
    }
    let new_bits = (bits & !(0xFFu32 << 23)) | ((new_exp as u32) << 23);
    f32::from_bits(new_bits)
}

/// 6.2 exp, route (b): pinned Cephes-pattern polynomial.
pub fn exp_pinned(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x > 88.0 {
        return f32::INFINITY;
    }
    if x < -88.0 {
        return 0.0;
    }
    let t = x * f(EXP_LOG2E);
    let z0 = t + 0.5;
    let k = floor_f32(z0);
    let kc1 = k * f(EXP_C1);
    let r0 = x - kc1;
    let kc2 = k * f(EXP_C2);
    let r = r0 - kc2;

    let r2 = r * r;
    let mut poly = f(EXP_P[0]);
    for i in 1..=5 {
        poly = poly * r + f(EXP_P[i]);
    }
    let m1 = poly * r2;
    let poly2 = m1 + r;
    let result = poly2 + 1.0;

    ldexp_exact(result, k as i32)
}

/// 6.3's shared reduction: Cody-Waite, staged through f64.
#[inline]
fn octant_reduce(x: f32) -> (f32, i64) {
    let xd = x as f64; // exact widening cast
    let k = round_ties_away_f64(xd / f64::from_bits(PI_2_F64));
    let khi = k * f64::from_bits(PI_2_F64);
    let r64 = xd - khi;
    let r = r64 as f32; // one correctly-rounded narrowing cast
    let quadrant = ((k as i64) % 4 + 4) % 4;
    (r, quadrant)
}

#[inline]
fn sin_poly(r: f32) -> f32 {
    let r2 = r * r;
    let mut inner = f(SIN_C2);
    inner = inner * r2 + f(SIN_C1);
    inner = inner * r2 + f(SIN_C0);
    let r3 = r2 * r;
    let term = inner * r3;
    r + term
}

#[inline]
fn cos_poly(r: f32) -> f32 {
    let r2 = r * r;
    let mut inner = f(COS_C2);
    inner = inner * r2 + f(COS_C1);
    inner = inner * r2 + f(COS_C0);
    let r4 = r2 * r2;
    let term = inner * r4;
    let half_r2 = 0.5 * r2;
    let step1 = 1.0 - half_r2;
    step1 + term
}

/// 6.3 sin.
pub fn sin_pinned(x: f32) -> f32 {
    let (r, q) = octant_reduce(x);
    match q {
        0 => sin_poly(r),
        1 => cos_poly(r),
        2 => -sin_poly(r),
        _ => -cos_poly(r),
    }
}

/// 6.3 cos.
pub fn cos_pinned(x: f32) -> f32 {
    let (r, q) = octant_reduce(x);
    match q {
        0 => cos_poly(r),
        1 => -sin_poly(r),
        2 => -cos_poly(r),
        _ => sin_poly(r),
    }
}

/// 6.4 SiLU. The final step is one IEEE-754 division, not a
/// reciprocal-multiply.
pub fn silu_pinned(x: f32) -> f32 {
    let neg = -x;
    let e = exp_pinned(neg);
    let denom = 1.0 + e;
    x / denom
}

/// 6.5.1 `frexp_exact`.
pub fn frexp_exact(x: f32) -> (f32, i32) {
    let bits = x.to_bits();
    let exp_bits = ((bits >> 23) & 0xFF) as i32;
    let mantissa_bits = (bits & 0x807F_FFFF) | (126u32 << 23);
    (f32::from_bits(mantissa_bits), exp_bits - 126)
}

/// 6.5 ln, route (b): pinned Cephes-pattern polynomial.
pub fn ln_pinned(x: f32) -> f32 {
    if x.is_nan() || x < 0.0 {
        return f32::NAN;
    }
    if x == 0.0 {
        return f32::NEG_INFINITY;
    }
    let (mut m, mut e) = frexp_exact(x);
    if m < f(LOG_SQRTHF) {
        e -= 1;
        m = m + m - 1.0;
    } else {
        m -= 1.0;
    }

    let z = m * m;
    let mut poly = f(LOG_P[0]);
    for i in 1..=8 {
        poly = poly * m + f(LOG_P[i]);
    }
    let mut y = poly * m;
    y = y * z;
    let fe = e as f32;
    let t1 = fe * f(LOG_Q1);
    y = y + t1;
    let half_z = 0.5 * z;
    y = y - half_z;
    let mut result = m + y;
    let t2 = fe * f(LOG_Q2);
    result = result + t2;
    result
}

/// 6.6 table digest: SHA-256 over the little-endian bytes of every pinned
/// coefficient above, in exactly this declared order. Any reordering, any
/// omission, and any single-ULP edit to a coefficient moves this digest,
/// which is folded into the witness chain (12.1 item 4).
pub fn table_digest() -> [u8; 32] {
    let mut h = Sha256::new();
    for c in [EXP_LOG2E, EXP_C1, EXP_C2] {
        h.update(&f(c).to_le_bytes());
    }
    for c in EXP_P {
        h.update(&f(c).to_le_bytes());
    }
    h.update(&f64::from_bits(PI_2_F64).to_le_bytes());
    for c in [SIN_C0, SIN_C1, SIN_C2, COS_C0, COS_C1, COS_C2] {
        h.update(&f(c).to_le_bytes());
    }
    h.update(&f(LOG_SQRTHF).to_le_bytes());
    h.update(&f(LOG_Q1).to_le_bytes());
    h.update(&f(LOG_Q2).to_le_bytes());
    for c in LOG_P {
        h.update(&f(c).to_le_bytes());
    }
    h.finalize()
}
