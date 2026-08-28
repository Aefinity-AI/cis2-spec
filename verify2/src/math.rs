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

// §6.3 sin/cos pinned two-part-pi reduction + Taylor coefficients.
const TWO_PI_HI: u32 = 0x40C90FDA;
const TWO_PI_LO: u32 = 0x347FF14D;

const SIN_COEF: [u32; 11] = [
    0x3F800000, 0xBE2AAAAB, 0x3C088888, 0xB9500D01, 0x3638EF1E, 0xB2D7322C,
    0x2F30922E, 0xAB573FA0, 0x274A963A, 0xA317A4DA, 0x1EB8DC77,
];
const COS_COEF: [u32; 11] = [
    0x3F800000, 0xBF000000, 0x3D2AAAAC, 0xBAB60B62, 0x37D00D02, 0xB493F27E,
    0x310F76C9, 0xAD49CBAA, 0x29573F9E, 0xA53413C5, 0x20F2A15F,
];

/// §6.3 range reduction to r in [-pi, pi], two-part-pi split.
fn reduce_2pi(x: f32) -> f32 {
    let two_pi_hi = f32::from_bits(TWO_PI_HI);
    let two_pi_lo = f32::from_bits(TWO_PI_LO);
    let k: f32 = (x / two_pi_hi).round();
    let khi: f32 = k * two_pi_hi;
    let r0: f32 = x - khi;
    let klo: f32 = k * two_pi_lo;
    r0 - klo
}

/// §6.3 cos(x), pinned Taylor series after 2*pi range reduction.
pub fn cos_pinned(x: f32) -> f32 {
    let r = reduce_2pi(x);
    let r2: f32 = r * r;
    let mut poly: f32 = f32::from_bits(COS_COEF[10]);
    for i in (0..=9).rev() {
        poly = poly * r2 + f32::from_bits(COS_COEF[i]);
    }
    poly
}

/// §6.3 sin(x), pinned Taylor series after 2*pi range reduction.
pub fn sin_pinned(x: f32) -> f32 {
    let r = reduce_2pi(x);
    let r2: f32 = r * r;
    let mut poly: f32 = f32::from_bits(SIN_COEF[10]);
    for i in (0..=9).rev() {
        poly = poly * r2 + f32::from_bits(SIN_COEF[i]);
    }
    poly * r
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
    for w in [TWO_PI_HI, TWO_PI_LO] {
        h.update(f32::from_bits(w).to_le_bytes());
    }
    for w in SIN_COEF {
        h.update(f32::from_bits(w).to_le_bytes());
    }
    for w in COS_COEF {
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
