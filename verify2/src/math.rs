// CIS-2 v0.1 pinned numeric primitives, implemented from
// docs/CIS2_SPEC_v0.1.md §5 and §6 alone (clean-room attempt #2, E15g).
// No FMA is ever used here: every "a*b + c" is written as two separate
// Rust expressions/statements so LLVM cannot contract them without an
// explicit fast-math flag (which this crate never passes).

/// §4.1 bf16 -> f32 widening: exact bit-shift, not a rounding conversion.
pub fn widen_bf16(b: u16) -> f32 {
    let bits: u32 = (b as u32) << 16;
    f32::from_bits(bits)
}

/// §5.1 sequential dot product: strict left-to-right, no pairwise tree.
pub fn dot_seq(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut acc: f32 = 0.0;
    for i in 0..a.len() {
        let p: f32 = a[i] * b[i];
        acc = acc + p;
    }
    acc
}

/// §5.3 sequential sum: strict left-to-right.
pub fn sum_seq(a: &[f32]) -> f32 {
    let mut acc: f32 = 0.0;
    for i in 0..a.len() {
        acc = acc + a[i];
    }
    acc
}

/// §6.1 rsqrt: correctly-rounded route (a), no table needed.
pub fn rsqrt(x: f32) -> f32 {
    1.0_f32 / x.sqrt()
}

// §6.2 pinned exp() coefficients (Cephes-pattern), f32 bit patterns.
const EXP_LOG2E: u32 = 0x3FB8AA3B;
const EXP_C1: u32 = 0x3F318000;
const EXP_C2: u32 = 0xB95E8083;
const EXP_P: [u32; 6] = [
    0x39506967, 0x3AB743CE, 0x3C088908, 0x3D2AA9C1, 0x3E2AAAAA, 0x3F000000,
];

/// §6.2.1 exact scale-by-power-of-2 via exponent-field bit manipulation.
fn ldexp_exact(x: f32, k: i32) -> f32 {
    if x == 0.0 {
        return x;
    }
    let bits = x.to_bits();
    let exp_bits: i32 = ((bits >> 23) & 0xFF) as i32;
    let new_exp = exp_bits + k;
    if new_exp <= 0 {
        return 0.0;
    }
    if new_exp >= 0xFF {
        return if x.is_sign_negative() {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
    }
    let new_exp_bits = (new_exp as u32) & 0xFF;
    let new_bits = (bits & !(0xFFu32 << 23)) | (new_exp_bits << 23);
    f32::from_bits(new_bits)
}

/// §6.2 exp(x) — pinned Cephes-pattern polynomial, no FMA, exact ldexp
/// reconstruction.
pub fn exp_pinned(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x > 88.0 {
        return f32::INFINITY;
    }
    if x < -88.0 {
        return 0.0;
    }

    let exp_log2e = f32::from_bits(EXP_LOG2E);
    let exp_c1 = f32::from_bits(EXP_C1);
    let exp_c2 = f32::from_bits(EXP_C2);

    let t: f32 = x * exp_log2e;
    let z0: f32 = t + 0.5;
    let k: f32 = z0.floor();
    let kc1: f32 = k * exp_c1;
    let r0: f32 = x - kc1;
    let kc2: f32 = k * exp_c2;
    let r: f32 = r0 - kc2;

    let r2: f32 = r * r;
    let mut poly: f32 = f32::from_bits(EXP_P[0]);
    for i in 1..=5 {
        poly = poly * r + f32::from_bits(EXP_P[i]);
    }
    let m1: f32 = poly * r2;
    let poly2: f32 = m1 + r;
    let result: f32 = poly2 + 1.0;

    let k_int: i32 = k as i32;
    ldexp_exact(result, k_int)
}

// §6.3 sin/cos pinned octant reduction (f64-staged, spec v0.3b) + SEPARATE
// minimax polynomials. Read from CIS2_SPEC_v0.3.md §6.3 -- NOT copied from
// src/math.rs -- per this crate's clean-room contract (CLEANROOM_LOG.md).
const PI_2_F64_BITS: u64 = 0x3FF921FB54442D18; // PI_2_F64 = pi/2, spec v0.3b §6.3

const SIN_C0_BITS: u32 = 0xBE2AAAA3;
const SIN_C1_BITS: u32 = 0x3C08839E;
const SIN_C2_BITS: u32 = 0xB94CA1F9;
const COS_C0_BITS: u32 = 0x3D2AAAA5;
const COS_C1_BITS: u32 = 0xBAB6061A;
const COS_C2_BITS: u32 = 0x37CCF5CE;

/// §6.3 (v0.3b) range reduction to r in [-pi/4, pi/4] + quadrant index k
/// mod 4, f64-staged: exact widening cast, f64 div+round+mul+sub (all
/// IEEE-754-mandatory correctly rounded or a deterministic bit-pattern
/// function), single correctly rounded narrowing cast back to f32. No FMA.
fn reduce_pi_2(x: f32) -> (f32, i64) {
    let pi_2_f64 = f64::from_bits(PI_2_F64_BITS);
    let xd: f64 = x as f64;
    let k_f64: f64 = (xd / pi_2_f64).round();
    let khi: f64 = k_f64 * pi_2_f64;
    let r64: f64 = xd - khi;
    (r64 as f32, k_f64 as i64)
}

/// §6.3 (v0.3b) sin(r) minimax polynomial for r in [-pi/4, pi/4], strict
/// left-to-right, no FMA.
fn sin_poly(r: f32) -> f32 {
    let r2 = r * r;
    let mut inner = f32::from_bits(SIN_C2_BITS);
    inner = inner * r2 + f32::from_bits(SIN_C1_BITS);
    inner = inner * r2 + f32::from_bits(SIN_C0_BITS);
    let r3 = r2 * r;
    let term = inner * r3;
    r + term
}

/// §6.3 (v0.3b) cos(r) minimax polynomial for r in [-pi/4, pi/4], strict
/// left-to-right, no FMA.
fn cos_poly(r: f32) -> f32 {
    let r2 = r * r;
    let mut inner = f32::from_bits(COS_C2_BITS);
    inner = inner * r2 + f32::from_bits(COS_C1_BITS);
    inner = inner * r2 + f32::from_bits(COS_C0_BITS);
    let r4 = r2 * r2;
    let term = inner * r4;
    let half_r2 = 0.5_f32 * r2;
    let step1 = 1.0_f32 - half_r2;
    step1 + term
}

#[inline]
fn quadrant(k: i64) -> i64 {
    ((k % 4) + 4) % 4
}

/// §6.3 (v0.3b) cos(x), octant reduction + minimax polynomial + quadrant
/// sign/swap.
pub fn cos_pinned(x: f32) -> f32 {
    let (r, k) = reduce_pi_2(x);
    let sr = sin_poly(r);
    let cr = cos_poly(r);
    match quadrant(k) {
        0 => cr,
        1 => -sr,
        2 => -cr,
        _ => sr,
    }
}

/// §6.3 (v0.3b) sin(x), octant reduction + minimax polynomial + quadrant
/// sign/swap.
pub fn sin_pinned(x: f32) -> f32 {
    let (r, k) = reduce_pi_2(x);
    let sr = sin_poly(r);
    let cr = cos_poly(r);
    match quadrant(k) {
        0 => sr,
        1 => cr,
        2 => -sr,
        _ => -cr,
    }
}

// §7.1 (v0.2) pinned ln() — Cephes-pattern logf: exact bit-manipulation
// frexp (mantissa in [0.5,1), power-of-two exponent), then a fixed-degree
// Horner polynomial in the reduced mantissa, then reassembly. Generalizes
// RoPE's inv_freq construction to any `rope_theta`, replacing v0.1's bare
// `LN_THETA` literal (valid only for rope_theta=100000.0).
const LOG_SQRTHF: u32 = 0x3F3504F3; // sqrt(0.5)
const LOG_Q1: u32 = 0xB95E8083; // ln2 lo (same split as EXP_C2)
const LOG_Q2: u32 = 0x3F318000; // ln2 hi (same split as EXP_C1)
const LOG_P: [u32; 9] = [
    0x3D9021BB, 0xBDEBD1B8, 0x3DEF251B, 0xBDFE5D4F, 0x3E11E9BF, 0xBE2AAE50,
    0x3E4CCEAD, 0xBE7FFFFC, 0x3EAAAAAA,
];

/// Exact frexp via bit manipulation: x = mantissa * 2^exp, mantissa in
/// [0.5, 1.0). No library call, exact (bit reinterpretation only).
fn frexp_exact(x: f32) -> (f32, i32) {
    let bits = x.to_bits();
    let exp_bits: i32 = ((bits >> 23) & 0xFF) as i32;
    let mantissa_bits = (bits & 0x807f_ffff) | (126u32 << 23);
    (f32::from_bits(mantissa_bits), exp_bits - 126)
}

/// §7.1 pinned ln(x), f32, x > 0 finite. Strict sequential Horner, no FMA.
pub fn ln_pinned(x: f32) -> f32 {
    if x.is_nan() || x < 0.0 {
        return f32::NAN;
    }
    if x == 0.0 {
        return f32::NEG_INFINITY;
    }
    let log_sqrthf = f32::from_bits(LOG_SQRTHF);
    let (mut m, mut e) = frexp_exact(x);
    if m < log_sqrthf {
        e -= 1;
        m = m + m - 1.0;
    } else {
        m = m - 1.0;
    }
    let z: f32 = m * m;
    let mut poly: f32 = f32::from_bits(LOG_P[0]);
    for i in 1..LOG_P.len() {
        let t: f32 = poly * m;
        poly = t + f32::from_bits(LOG_P[i]);
    }
    let mut y: f32 = poly * m;
    y = y * z;
    let fe: f32 = e as f32;
    let log_q1 = f32::from_bits(LOG_Q1);
    let t1: f32 = fe * log_q1;
    y = y + t1;
    let half_z: f32 = 0.5 * z;
    y = y - half_z;
    let mut result: f32 = m + y;
    let log_q2 = f32::from_bits(LOG_Q2);
    let t2: f32 = fe * log_q2;
    result = result + t2;
    result
}

/// §6.4 SiLU: x / (1 + exp_pinned(-x)).
pub fn silu_pinned(x: f32) -> f32 {
    let neg: f32 = -x;
    let e: f32 = exp_pinned(neg);
    let denom: f32 = 1.0 + e;
    x / denom
}

/// §6.5 table_digest: SHA-256 over the LE bytes of every pinned
/// coefficient, in the exact declared order. Informative only — not an
/// input to CIS2_REF.
pub fn table_digest() -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for w in [EXP_LOG2E, EXP_C1, EXP_C2] {
        h.update(f32::from_bits(w).to_le_bytes());
    }
    for w in EXP_P {
        h.update(f32::from_bits(w).to_le_bytes());
    }
    // v0.3: one 8-byte f64 constant replaces v0.2's two 4-byte f32 halves.
    h.update(f64::from_bits(PI_2_F64_BITS).to_le_bytes());
    for w in [SIN_C0_BITS, SIN_C1_BITS, SIN_C2_BITS] {
        h.update(f32::from_bits(w).to_le_bytes());
    }
    for w in [COS_C0_BITS, COS_C1_BITS, COS_C2_BITS] {
        h.update(f32::from_bits(w).to_le_bytes());
    }
    h.update(f32::from_bits(LOG_SQRTHF).to_le_bytes());
    h.update(f32::from_bits(LOG_Q1).to_le_bytes());
    h.update(f32::from_bits(LOG_Q2).to_le_bytes());
    for w in LOG_P {
        h.update(f32::from_bits(w).to_le_bytes());
    }
    h.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rsqrt_64_is_exact_eighth() {
        assert_eq!(rsqrt(64.0).to_bits(), 0x3E000000);
    }

    #[test]
    fn dot_seq_order_sensitive_regression() {
        let a = [1.0e8_f32, 1.0_f32, -1.0e8_f32];
        let b = [1.0_f32, 1.0_f32, 1.0_f32];
        assert_eq!(dot_seq(&a, &b), 0.0_f32);
    }

    #[test]
    fn ln_pinned_within_1ulp_on_spec_pinned_thetas() {
        fn ulp_key(v: f32) -> i64 {
            let bits = v.to_bits() as i64;
            if v.is_sign_negative() {
                -(bits & 0x7fff_ffff)
            } else {
                bits
            }
        }
        for &theta in &[10_000.0f64, 100_000.0, 500_000.0, 1_000_000.0] {
            let want = (theta.ln()) as f32;
            let got = ln_pinned(theta as f32);
            let ulp = (ulp_key(got) - ulp_key(want)).abs();
            assert!(ulp <= 1, "ln_pinned({theta}) got={got} want={want} ulp={ulp}");
        }
    }

    #[test]
    fn widen_bf16_exact() {
        // bf16 1.0 == 0x3F80, widened f32 1.0 == 0x3F800000
        assert_eq!(widen_bf16(0x3F80).to_bits(), 0x3F800000);
    }
}
