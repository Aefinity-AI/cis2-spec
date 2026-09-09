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

#[cfg(test)]
mod op_level_goldens {
    //! Tier-1 op-level goldens for the two spec clauses that the §13.1
    //! full-run test vector provably cannot exercise (E22 necessity matrix,
    //! rows M15 and M16).
    //!
    //! These values are computed here by this crate's *software* square root
    //! (`softfp::sqrt_cr`), independently of any hardware `sqrtf`, and match
    //! the C reference (`verify3/mathpin.c`, `cis2_rsqrt`) bit-for-bit.
    use super::*;

    /// §6.1's only stated conformance check is `rsqrt(64.0) == 0.125`. That is
    /// the one value in the plausible head_dim range where *every* plausible
    /// implementation route agrees, so it cannot detect a wrong route.
    #[test]
    fn the_spec_s_only_rsqrt_check_is_at_a_value_that_cannot_discriminate() {
        assert_eq!(rsqrt(64.0).to_bits(), 0x3E00_0000, "spec 6.1's stated check");
        // A double-precision route -- 1/sqrt computed in f64 and rounded once
        // -- is a different computation, but at 64 it lands on the same bits,
        // because 64 is a perfect square and 1/8 is exactly representable.
        assert_eq!(rsqrt_via_f64(64.0).to_bits(), 0x3E00_0000);
    }

    /// The head_dims at which the spec's composed-f32 route and a
    /// single-rounded f64 route differ by 1 ULP. 96 and 112 are head_dims that
    /// occur in shipping checkpoints, so this is not a synthetic corner.
    #[test]
    fn rsqrt_route_goldens_separate_the_f32_route_from_an_f64_route() {
        // (head_dim, spec-route bits, f64-route bits)
        const DIVERGE: &[(f32, u32, u32)] = &[
            (24.0, 0x3E51_05EB, 0x3E51_05EC),
            (72.0, 0x3DF1_5BF0, 0x3DF1_5BEF),
            (96.0, 0x3DD1_05EB, 0x3DD1_05EC),
            (112.0, 0x3DC1_8490, 0x3DC1_848F),
            (136.0, 0x3DAF_9D54, 0x3DAF_9D53),
        ];
        for &(d, spec, other) in DIVERGE {
            assert_eq!(rsqrt(d).to_bits(), spec, "spec 6.1 route at head_dim {d}");
            assert_ne!(spec, other, "golden must actually discriminate at {d}");
            assert_eq!(rsqrt_via_f64(d).to_bits(), other, "f64 route at {d}");
        }
        // Perfect squares in the same range agree on both routes -- these are
        // the values a conformance suite must NOT use as its only check.
        for d in [16.0f32, 64.0, 256.0] {
            assert_eq!(rsqrt(d).to_bits(), rsqrt_via_f64(d).to_bits());
        }
    }

    /// A non-conforming route computed with f64 intermediates. Present only so
    /// the goldens above are demonstrated to discriminate; never used by the
    /// verifier itself.
    fn rsqrt_via_f64(x: f32) -> f32 {
        // Newton on f64 is not needed: this crate has no libm, so the f64
        // reference square root is built from the software f32 root refined
        // once in f64, which is exact to well beyond f32 precision.
        let x64 = x as f64;
        let mut r = sqrt_cr(x) as f64;
        r = 0.5 * (r + x64 / r); // one Newton step: f64-accurate sqrt
        (1.0f64 / r) as f32
    }

    /// §11.2's `>` (first-maximal) vs `>=` (last-maximal). The §13.1 vector
    /// never reaches this branch -- no exact tie occurs in 16 argmaxes over
    /// 49152 fp32 logits -- so the clause needs its own golden.
    #[test]
    fn argmax_tie_golden_separates_first_maximal_from_last_maximal() {
        let logits: [f32; 6] = [1.0, 3.0, 3.0, 2.0, 3.0, -1.0];
        assert_eq!(argmax_first_maximal(&logits), 1);
        assert_eq!(argmax_last_maximal(&logits), 4);
        // A vector with no tie cannot tell the two rules apart, which is
        // exactly why the pinned full-run vector missed this.
        let untied: [f32; 4] = [1.0, 3.0, 2.0, -1.0];
        assert_eq!(argmax_first_maximal(&untied), argmax_last_maximal(&untied));
    }

    fn argmax_first_maximal(v: &[f32]) -> usize {
        let mut bi = 0;
        let mut bv = v[0];
        for (i, &x) in v.iter().enumerate() {
            if x > bv {
                bv = x;
                bi = i;
            }
        }
        bi
    }

    fn argmax_last_maximal(v: &[f32]) -> usize {
        let mut bi = 0;
        let mut bv = v[0];
        for (i, &x) in v.iter().enumerate() {
            if x >= bv {
                bv = x;
                bi = i;
            }
        }
        bi
    }
}

#[cfg(test)]
mod large_angle {
    //! Spec 14.4 said the trig polynomials were "only validated for
    //! `|x| ≲ 14`" and warned an implementer targeting a longer sequence not
    //! to assume the accuracy holds. E26 settled that exhaustively: every f32
    //! in `[0, 2^20)` -- 1,233,125,376 of them, which is every RoPE angle any
    //! context up to 1,048,576 positions can produce, since §7.1 makes
    //! `inv_freq[0]` exactly 1.0 and every other entry smaller -- lands within
    //! 9.4218e-8 absolute of the f64 value, i.e. under 0.8 ULP at 1.0.
    //!
    //! An exhaustive sweep is a measurement and cannot live in `cargo test`.
    //! What lives here is what a conformance suite needs: bit goldens at the
    //! arguments that sweep found worst, so a reduction that regresses at
    //! large angles fails loudly instead of quietly, plus the exact symmetry
    //! the sweep proved, which needs no oracle at all.
    //!
    //! Every operand is routed through `black_box` in both directions, per the
    //! house rule (CONTRIBUTING §3): `sin_pinned(f32::from_bits(K))` on a
    //! literal K is exactly the shape LLVM is entitled to fold, and a folded
    //! constant reports the compiler's arithmetic, not the pinned runtime
    //! FPU's.
    use super::*;
    use crate::fpenv;
    use core::hint::black_box;

    /// `(x, sin_pinned(x), cos_pinned(x))` as bits. The first seven rows are
    /// the arguments at which E26's exhaustive sweep of `[0, 2^20)` recorded
    /// its worst absolute or worst ULP error for one of the two functions; the
    /// rest span the range so a reduction that breaks in one octant or one
    /// decade cannot slip through.
    const GOLDENS: &[(u32, u32, u32)] = &[
        (0x4503_4E6F, 0x3F3B_C37A, 0xBF2E_0390), // 2100.902   worst |abs| sin
        (0x458B_E628, 0x33F7_F000, 0xBF80_0000), // 4476.7695  worst |ulp| sin < 2^16
        (0x467D_9F8D, 0x3F29_198C, 0xBF40_337A), // 16231.888  worst |abs| cos < 2^16
        (0x474D_246F, 0x3F80_0000, 0xB28B_6000), // 52516.434  worst |ulp| cos < 2^16
        (0x47CD_246F, 0xB30B_6000, 0xBF80_0000), // 105032.87  worst |ulp| sin < 2^20
        (0x47FF_31CE, 0x3F32_60D2, 0x3F37_9F5A), // 130659.61  worst |abs| cos < 2^20
        (0x4943_998D, 0x3F80_0000, 0x33DD_4000), // 801176.8   worst |ulp| cos < 2^20
        (0x3F80_0000, 0x3F57_6AA5, 0x3F0A_5140), // 1.0
        (0x4198_0000, 0x3E19_7969, 0x3F7D_1BBF), // 19.0   -- the old §14.4 edge
        (0x43C8_0000, 0xBF59_D5D9, 0xBF06_79D2), // 400.0
        (0x477F_FF00, 0x3F7B_3849, 0x3E44_F5D5), // 65535.0  -- a 64k context
        (0x497F_FFF0, 0xBF1D_9959, 0x3F49_BD22), // 1048575.0 -- a 1M context
    ];

    fn sin_at(bits: u32) -> u32 {
        let x = black_box(f32::from_bits(black_box(bits)));
        black_box(sin_pinned(x)).to_bits()
    }

    fn cos_at(bits: u32) -> u32 {
        let x = black_box(f32::from_bits(black_box(bits)));
        black_box(cos_pinned(x)).to_bits()
    }

    /// §6.3's Cody-Waite reduction, pinned at the angles E26 found hardest.
    #[test]
    fn reduction_goldens_hold_out_to_a_one_million_position_context() {
        fpenv::pin_and_selftest().expect("1.3 pin");
        for &(x, s, c) in GOLDENS {
            assert_eq!(sin_at(x), s, "sin_pinned at 0x{x:08X}");
            assert_eq!(cos_at(x), c, "cos_pinned at 0x{x:08X}");
        }
    }

    /// E26 checked every f32 with `2^-126 <= |x| < 2^31` and found
    /// `sin_pinned` exactly odd and `cos_pinned` exactly even in value, with
    /// no exceptions. That is what makes a positive-only sweep a statement
    /// about every finite f32 of that magnitude. Two caveats, both measured
    /// rather than assumed:
    ///
    ///  * below `2^-126` it does not hold and is not meant to -- §1.3's DAZ
    ///    flushes the input, so `sin_pinned` returns `+0.0` for both signs;
    ///  * at exactly three arguments in `[2^29, 2^31)` (620046660,
    ///    1175634300, 1240093300) the reduced argument underflows to zero and
    ///    only the *sign* of a zero result fails to mirror. Nothing below
    ///    `2^20` is affected, so no RoPE angle a 1M-position context can
    ///    produce is affected.
    #[test]
    fn sin_is_exactly_odd_and_cos_exactly_even_for_normal_inputs() {
        fpenv::pin_and_selftest().expect("1.3 pin");
        for &(x, _, _) in GOLDENS {
            let neg = x ^ 0x8000_0000;
            let s = f32::from_bits(sin_at(x));
            assert_eq!(sin_at(neg), (-s).to_bits(), "sin(-x) != -sin(x) at 0x{x:08X}");
            assert_eq!(cos_at(neg), cos_at(x), "cos(-x) != cos(x) at 0x{x:08X}");
        }
        // The documented exception, asserted rather than assumed: on the
        // zero/subnormal class the pinned environment does not carry the sign.
        for z in [0x0000_0000u32, 0x0000_0001, 0x007F_FFFF] {
            assert_eq!(sin_at(z), 0x0000_0000, "sin_pinned flushes 0x{z:08X}");
            assert_eq!(sin_at(z ^ 0x8000_0000), 0x0000_0000, "and its negation");
        }
    }
}
