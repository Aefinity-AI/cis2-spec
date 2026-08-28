//! CIS-2 §3 normative fp32 kernel primitives.
//!
//! Every function in this module is part of the CIS-2 v0.1 reference
//! contract (see docs/E15b_REFERENCE_RESULT.md and the E14a design memo,
//! a design memo from a separate internal operator-notes repository, not published here).
//! Rules enforced here, mechanically:
//!
//! - §3.1 reduction order: STRICT LEFT-TO-RIGHT SEQUENTIAL accumulation.
//!   Every dot product / sum-of-squares / softmax-denominator sum folds in
//!   index order, one add at a time. No pairwise tree, no chunking.
//! - §3.2: no fused multiply-add call (the "mul, add" combinator method on
//!   f32/f64, which lowers to LLVM's fused FMA intrinsic) anywhere in this
//!   file. Every multiply-then-add is written as two separate,
//!   separately-rounded Rust operations (`let p = a * b; acc = acc + p;`),
//!   which LLVM does not auto-fuse absent an explicit fast-math flag Rust
//!   does not offer (memo §2 "Compiler flags" row). This is mechanically
//!   checked by `tests/no_mul_add.rs` (grep gate) and
//!   `scripts/no_fma_gate.sh` (grep gate + disassembly gate).
//! - §3.3 transcendentals: rsqrt is composed from two IEEE-754-mandatory
//!   correctly-rounded ops (`sqrt`, `/`) — this is route (a),
//!   correctly-rounded-by-construction, with NO pinned table needed.
//!   `exp`, `sin`, `cos` use route (b) — a pinned fixed-coefficient
//!   polynomial, because no Rust binding to CORE-MATH/RLIBM was available
//!   in this sandbox (crates.io has no published Rust crate for either;
//!   see docs/E15b_REFERENCE_RESULT.md "open questions"). This IS the
//!   documented fallback, not route (a).
//! - §3.5: bf16->fp32 widening is the exact bit-shift, not a "conversion".

/// §3.5 — exact, lossless bf16 -> fp32 widening. Zero-extends the 16-bit
/// bf16 pattern into the high half of an f32 bit pattern. No rounding, no
/// library call.
#[inline]
pub fn widen_bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// §3.1 — strict left-to-right sequential dot product. No FMA (§3.2): the
/// multiply and the add are separate, separately-rounded operations.
#[inline]
pub fn dot_seq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc: f32 = 0.0;
    for i in 0..a.len() {
        let p = a[i] * b[i]; // separate mul; never fused with the add below
        acc = acc + p; // separate add
    }
    acc
}

/// §3.1 — strict left-to-right sequential sum.
#[inline]
pub fn sum_seq(a: &[f32]) -> f32 {
    let mut acc: f32 = 0.0;
    for &x in a {
        acc = acc + x;
    }
    acc
}

/// §3.3 route (a) — correctly-rounded reciprocal square root, composed from
/// two IEEE-754-mandatory correctly-rounded ops (`sqrt`, `/`). Any
/// IEEE-754-conformant fp32 hardware/ISA produces the same bits for these
/// two ops given the same input, by standard (memo §1.6) — no table, no
/// pinning needed.
#[inline]
pub fn rsqrt_cr(x: f32) -> f32 {
    1.0f32 / x.sqrt()
}

// ---------------------------------------------------------------------
// §3.3 route (b) fallback: pinned exp/sin/cos.
//
// Algorithm shapes (range reduction + fixed-degree polynomial) follow the
// long-public-domain Cephes single-precision expf/sinf/cosf structure
// (Moshier, "Cephes Math Library", used as the algorithmic pattern only —
// coefficients below are independently pinned f32 bit patterns computed
// from the exact rational Taylor/Cephes constants at f64 precision and
// rounded once to f32, not copied binary). Every coefficient is stated
// literally so a clean-room reimplementation can reproduce the exact bits.
// The full coefficient set is included in a SHA-256 "table digest" (see
// `table_digest()` below and CIS2_REF's printed `table_digest=` field) per
// CIS-1's own pinned-LUT precedent (spec §5.7/§5.9).
// ---------------------------------------------------------------------

// exp(x), Cephes-pattern range reduction: x = k*ln2 + r, exp(x) = 2^k * exp(r).
const EXP_LOG2E: f32 = 1.442_695_04_f32; // 1/ln2
const EXP_C1: f32 = 0.693_359_375_f32; // ln2 hi (Cephes split)
const EXP_C2: f32 = -2.121_944_4e-4_f32; // ln2 lo (Cephes split)
// Polynomial coefficients for exp(r)-1-r ~ r^2 * P(r), degree 5 in r.
const EXP_P: [f32; 6] = [
    1.987_569_15e-4,
    1.398_199_95e-3,
    8.333_451_9e-3,
    4.166_579_6e-2,
    1.666_666_5e-1,
    5.000_000_1e-1,
];

/// §3.3(b) — pinned exp(x), f32. Strict sequential Horner, no FMA.
pub fn exp_pinned(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    // Saturate rather than let denormal/overflow policy drift (§3.4 governs
    // denormal *results*; this bound just avoids ldexp overflow UB).
    if x > 88.0 {
        return f32::INFINITY;
    }
    if x < -88.0 {
        return 0.0;
    }
    let t = x * EXP_LOG2E; // separate mul
    let z0 = t + 0.5_f32;
    let k = z0.floor();
    // r = x - k*C1 - k*C2 (two-step reduction, Cephes split, no fma)
    let kc1 = k * EXP_C1;
    let r0 = x - kc1;
    let kc2 = k * EXP_C2;
    let r = r0 - kc2;

    let r2 = r * r;
    // Horner, strict sequential, no fma.
    let mut poly = EXP_P[0];
    for i in 1..EXP_P.len() {
        let m = poly * r;
        poly = m + EXP_P[i];
    }
    let m1 = poly * r2;
    let poly2 = m1 + r; // + r
    let result = poly2 + 1.0_f32; // + 1

    ldexp_exact(result, k as i32)
}

/// Exact scale-by-power-of-two via bit manipulation of the exponent field
/// (no library call, exact for results that stay in the normal range,
/// which the exp() domain clamp above guarantees).
#[inline]
fn ldexp_exact(x: f32, k: i32) -> f32 {
    if x == 0.0 {
        return x;
    }
    let bits = x.to_bits();
    let exp_bits = ((bits >> 23) & 0xFF) as i32;
    let new_exp = exp_bits + k;
    if new_exp <= 0 {
        return 0.0; // underflow to zero; §3.4 pins FTZ/DAZ anyway
    }
    if new_exp >= 0xFF {
        return if x.is_sign_negative() {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
    }
    let new_bits = (bits & !(0xFFu32 << 23)) | ((new_exp as u32) << 23);
    f32::from_bits(new_bits)
}

// sin/cos(x): pinned Taylor series after a two-part-pi range reduction to
// [-pi, pi]. Term magnitude at the last included degree is already below
// f32 epsilon for |r| <= pi (see docs/E15b_REFERENCE_RESULT.md derivation),
// so no further quadrant folding is needed for correctness at f32 ULP.
const TWO_PI_HI: f32 = 6.283_185_f32;
const TWO_PI_LO: f32 = 2.383_651e-7_f32; // 2*pi - TWO_PI_HI, exact-in-f64 split

/// Reduce x to r in [-pi, pi], strict sequential, no fma.
fn reduce_2pi(x: f32) -> f32 {
    let k = (x / TWO_PI_HI).round();
    let khi = k * TWO_PI_HI;
    let r0 = x - khi;
    let klo = k * TWO_PI_LO;
    r0 - klo
}

// 1/(2n+1)! coefficients for sin, n = 0..=10 (degree up to r^21).
const SIN_COEF: [f32; 11] = [
    1.0,
    -1.666_666_7e-1,
    8.333_333e-3,
    -1.984_127e-4,
    2.755_732e-6,
    -2.505_211e-8,
    1.605_904e-10,
    -7.647_164e-13,
    2.811_457e-15,
    -8.220_635e-18,
    1.957_294e-20,
];
// 1/(2n)! coefficients for cos, n = 0..=10 (degree up to r^20).
const COS_COEF: [f32; 11] = [
    1.0,
    -5.0e-1,
    4.166_667e-2,
    -1.388_889e-3,
    2.480_159e-5,
    -2.755_732e-7,
    2.087_676e-9,
    -1.147_075e-11,
    4.779_477e-14,
    -1.561_921e-16,
    4.110_318e-19,
];

/// §3.3(b) — pinned sin(x), f32. Strict sequential Horner in r^2, no FMA.
pub fn sin_pinned(x: f32) -> f32 {
    let r = reduce_2pi(x);
    let r2 = r * r;
    let mut poly = SIN_COEF[10];
    for i in (0..10).rev() {
        let m = poly * r2;
        poly = m + SIN_COEF[i];
    }
    poly * r
}

/// §3.3(b) — pinned cos(x), f32. Strict sequential Horner in r^2, no FMA.
pub fn cos_pinned(x: f32) -> f32 {
    let r = reduce_2pi(x);
    let r2 = r * r;
    let mut poly = COS_COEF[10];
    for i in (0..10).rev() {
        let m = poly * r2;
        poly = m + COS_COEF[i];
    }
    poly
}

// ---------------------------------------------------------------------
// §3.3 route (b) fallback: pinned ln(x), f32 (E15d addition — generalizes
// RoPE's inv_freq table to arbitrary rope_theta instead of a hardcoded
// literal pinned for exactly rope_theta=1e5). Cephes-pattern logf: exact
// bit-manipulation frexp (no library call, no rounding) to split x into
// mantissa in [0.5,1) and a power-of-two exponent, then a fixed-degree
// Horner polynomial in the mantissa, then reassemble with the exponent
// term — same "algorithm shape from Cephes, coefficients independently
// pinned as literal f32 bit patterns" contract as exp/sin/cos above.
// ---------------------------------------------------------------------

const LOG_SQRTHF: f32 = 0.707_106_78_f32; // sqrt(0.5)
const LOG_Q1: f32 = -2.121_944_4e-4_f32; // ln2 lo (same split as EXP_C2)
const LOG_Q2: f32 = 0.693_359_375_f32; // ln2 hi (same split as EXP_C1)
// Cephes logf polynomial coefficients, degree 9 in the reduced mantissa.
const LOG_P: [f32; 9] = [
    7.037_683_6e-2,
    -1.151_461_0e-1,
    1.167_699_9e-1,
    -1.242_014_1e-1,
    1.424_932_3e-1,
    -1.666_805_8e-1,
    2.000_071_5e-1,
    -2.499_999_4e-1,
    3.333_333_1e-1,
];

/// Exact frexp via bit manipulation: x = mantissa * 2^exp, mantissa in
/// [0.5, 1.0). No library call, exact (bit reinterpretation only).
#[inline]
fn frexp_exact(x: f32) -> (f32, i32) {
    debug_assert!(x > 0.0 && x.is_finite());
    let bits = x.to_bits();
    let exp_bits = ((bits >> 23) & 0xFF) as i32;
    // Force the exponent field to 126 (bias 127 - 1), giving a value in
    // [0.5, 1.0) with the same mantissa bits as the input.
    let mantissa_bits = (bits & 0x807f_ffff) | (126u32 << 23);
    (f32::from_bits(mantissa_bits), exp_bits - 126)
}

/// §3.3(b) — pinned ln(x), f32, x > 0 finite. Strict sequential Horner,
/// no FMA. Domain: RoPE's inv_freq table only (x = rope_theta, a small
/// positive config constant), not a general per-decode-step hot path.
pub fn ln_pinned(x: f32) -> f32 {
    if x.is_nan() || x < 0.0 {
        return f32::NAN;
    }
    if x == 0.0 {
        return f32::NEG_INFINITY;
    }
    let (mut m, mut e) = frexp_exact(x);
    if m < LOG_SQRTHF {
        e -= 1;
        m = m + m - 1.0;
    } else {
        m = m - 1.0;
    }
    let z = m * m;
    let mut poly = LOG_P[0];
    for i in 1..LOG_P.len() {
        let t = poly * m;
        poly = t + LOG_P[i];
    }
    let mut y = poly * m; // y = poly(m) * m
    y = y * z;
    let fe = e as f32;
    let t1 = fe * LOG_Q1;
    y = y + t1;
    let half_z = 0.5_f32 * z;
    y = y - half_z;
    let mut result = m + y;
    let t2 = fe * LOG_Q2;
    result = result + t2;
    result
}

/// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)), using the pinned exp above
/// and IEEE-754-mandatory correctly-rounded `/`.
pub fn silu_pinned(x: f32) -> f32 {
    let neg = -x;
    let e = exp_pinned(neg);
    let denom = 1.0_f32 + e;
    x / denom
}

/// Softmax over `logits` in place: max-subtract (sequential scan, so tie
/// breaking on equal maxima is deterministic — first occurrence wins,
/// matching §2's "strict sequential/fixed-tree max" requirement),
/// pinned-exp, §3.1 sequential-sum denominator, then elementwise divide.
pub fn softmax_seq(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    let mut max_v = logits[0];
    for &v in &logits[1..] {
        if v > max_v {
            max_v = v;
        }
    }
    for v in logits.iter_mut() {
        *v = exp_pinned(*v - max_v);
    }
    let denom = sum_seq(logits);
    for v in logits.iter_mut() {
        *v = *v / denom;
    }
}

/// SHA-256 over the LE bytes of every pinned coefficient table in this
/// module (EXP_P, SIN_COEF, COS_COEF, plus the range-reduction constants),
/// in a fixed declared order — CIS-1's "table digest" precedent (spec
/// §5.7/§5.9), so any change to the fallback polynomial is visible and
/// citable, not silent.
pub fn table_digest() -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for &c in &[EXP_LOG2E, EXP_C1, EXP_C2] {
        h.update(c.to_le_bytes());
    }
    for &c in &EXP_P {
        h.update(c.to_le_bytes());
    }
    for &c in &[TWO_PI_HI, TWO_PI_LO] {
        h.update(c.to_le_bytes());
    }
    for &c in &SIN_COEF {
        h.update(c.to_le_bytes());
    }
    for &c in &COS_COEF {
        h.update(c.to_le_bytes());
    }
    h.update(LOG_SQRTHF.to_le_bytes());
    h.update(LOG_Q1.to_le_bytes());
    h.update(LOG_Q2.to_le_bytes());
    for &c in &LOG_P {
        h.update(c.to_le_bytes());
    }
    h.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widen_bf16_exact() {
        // bf16 1.0 = 0x3F80; widened f32 1.0 = 0x3F800000
        assert_eq!(widen_bf16(0x3F80).to_bits(), 0x3F80_0000);
        assert_eq!(widen_bf16(0x0000), 0.0f32);
        assert_eq!(widen_bf16(0x8000).to_bits(), 0x8000_0000); // -0.0
    }

    #[test]
    fn exp_matches_std_within_tolerance() {
        for &x in &[-5.0f32, -1.0, -0.001, 0.0, 0.001, 0.5, 1.0, 2.0, 5.0] {
            let got = exp_pinned(x);
            let want = x.exp(); // host libm, comparison only, not normative
            let rel = ((got - want) / want).abs();
            assert!(rel < 2e-6, "exp({x}) got={got} want={want} rel={rel}");
        }
    }

    #[test]
    fn sin_cos_match_std_within_tolerance() {
        for i in -20..=20 {
            let x = i as f32 * 0.7;
            let gs = sin_pinned(x);
            let gc = cos_pinned(x);
            let ws = x.sin();
            let wc = x.cos();
            assert!((gs - ws).abs() < 2e-6, "sin({x}) got={gs} want={ws}");
            assert!((gc - wc).abs() < 2e-6, "cos({x}) got={gc} want={wc}");
        }
    }

    #[test]
    fn ln_matches_std_within_tolerance() {
        for &x in &[
            0.001f32,
            0.1,
            0.5,
            0.999,
            1.0,
            1.5,
            2.0,
            10.0,
            100.0,
            100_000.0,  // SmolLM2 rope_theta
            1_000_000.0, // Qwen2.5-0.5B rope_theta (E15d(c))
        ] {
            let got = ln_pinned(x);
            let want = x.ln(); // host libm, comparison only, not normative
            if want == 0.0 {
                assert!((got - want).abs() < 2e-6, "ln({x}) got={got} want={want}");
            } else {
                let rel = ((got - want) / want).abs();
                assert!(rel < 2e-6, "ln({x}) got={got} want={want} rel={rel}");
            }
        }
    }

    #[test]
    fn rsqrt_matches_expected() {
        assert_eq!(rsqrt_cr(64.0), 0.125);
        assert!((rsqrt_cr(2.0) - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-7);
    }

    #[test]
    fn dot_seq_is_order_sensitive_by_design() {
        // Documents §3.1: this is NOT associative-safe reordering, it's a
        // pinned left-to-right order. Regression guard only.
        let a = [1.0e8f32, 1.0, -1.0e8];
        let b = [1.0f32, 1.0, 1.0];
        // (((0 + 1e8*1) + 1*1) + -1e8*1) — the 1.0 gets lost to rounding,
        // which is the whole point: this specific bit pattern is normative.
        let got = dot_seq(&a, &b);
        assert_eq!(got, 0.0f32);
    }
}

