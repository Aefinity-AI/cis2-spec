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
mod exp_ln_range {
    //! Spec 14.1 said `ln_pinned`'s "validated domain is a finite,
    //! explicitly-tested set of `x` values (§6.5), not a general accuracy
    //! proof", and §0's non-claims list said nothing was claimed for `exp` or
    //! `ln` outside the input ranges the checked decodes happen to exercise.
    //! E27 settled both exhaustively -- all 2^32 f32 bit patterns for
    //! `exp_pinned`, `ln_pinned` and `silu_pinned`, against an f64 accuracy
    //! oracle under the §1.3 pinned environment. Outside §6.2's two guard
    //! bands every one of the 3,257,925,634 comparable `exp` arguments and
    //! all 2,130,706,432 positive-normal `ln` arguments land within **one
    //! ULP** of the f64 value narrowed to f32, both functions are exactly
    //! monotone over the whole finite domain, and `silu_pinned` (§6.4) is
    //! within two ULP everywhere it is not sitting on §6.2's clip.
    //!
    //! An exhaustive sweep is a measurement and cannot live in `cargo test`
    //! (see `examples/exp_ln_exhaustive.rs`). What lives here is what a
    //! conformance suite needs: bit goldens at the arguments that sweep found
    //! worst and at every boundary of §6.2's and §6.5's guard logic, so a
    //! reimplementation that regresses there fails loudly instead of quietly.
    //!
    //! Every operand is routed through `black_box` in both directions, per
    //! the house rule (CONTRIBUTING §3): `exp_pinned(f32::from_bits(K))` on a
    //! literal K is exactly the shape LLVM is entitled to fold, and a folded
    //! constant reports the compiler's arithmetic, not the pinned runtime
    //! FPU's.
    use super::*;
    use crate::fpenv;
    use core::hint::black_box;

    /// `(x, exp_pinned(x))` as bits. The first block is where E27's sweep
    /// recorded its worst error; the second is every boundary §6.2's guard
    /// logic has, including the two arguments either side of the `x > 88.0`
    /// clip and the two either side of the `x < -88.0` flush.
    const EXP_GOLDENS: &[(u32, u32)] = &[
        (0x3380_0000, 0x3F80_0000), // 5.9604645e-8  worst |ulp|, all finite x
        (0xB420_0001, 0x3F7F_FFFE), // -1.4901163e-7 worst |ulp|, softmax half
        (0x42AE_AD6A, 0x7E80_46AE), // 87.3387       worst |abs| (1 ulp at 1e38)
        (0x3F00_0005, 0x3FD3_0950), // 0.5000003
        (0x3F80_0006, 0x402D_F85C), // 1.0000007
        (0xC280_0003, 0x114B_4D72), // -64.00002
        (0x3F80_0000, 0x402D_F854), // 1.0  -> e
        (0xBF80_0000, 0x3EBC_5AB2), // -1.0 -> 1/e
        // Spread across §6.2's reduction index `k`, so a reduction that is
        // right near zero and wrong far from it cannot pass. These span
        // k = +-7, +-14, +-29, +-58, +-115.
        (0x40A0_0000, 0x4314_69C5), // 5.0
        (0xC0A0_0000, 0x3BDC_C9FF), // -5.0
        (0x4120_0000, 0x46AC_14EE), // 10.0
        (0xC120_0000, 0x383E_6BCE), // -10.0
        (0x41A0_0000, 0x4DE7_5844), // 20.0
        (0xC1A0_0000, 0x310D_A433), // -20.0
        (0x4220_0000, 0x5C51_106A), // 40.0
        (0xC220_0000, 0x229C_BC92), // -40.0
        (0x42A0_0000, 0x792A_BBCE), // 80.0
        (0xC2A0_0000, 0x05BF_ECBA), // -80.0
        // Guard boundaries, exact.
        (0x0000_0000, 0x3F80_0000), // +0.0 -> 1.0 exactly: §7.1's inv_freq[0]
        (0x8000_0000, 0x3F80_0000), // -0.0 -> 1.0 exactly: the one §7.1 calls
        (0x0000_0001, 0x3F80_0000), // min subnormal: §1.3 DAZ makes it +0.0
        (0x42B0_0000, 0x7EF8_82B7), // 88.0        last argument the poly sees
        (0x42B0_0001, 0x7F80_0000), // 88.0000076  first clipped to +Inf
        (0x42B1_7217, 0x7F80_0000), // 88.722832   last x whose true exp is finite
        (0xC2B0_0000, 0x0000_0000), // -88.0       poly result underflows to 0
        (0xC2B0_0001, 0x0000_0000), // -88.0000076 first flushed by the guard
    ];

    /// `(x, ln_pinned(x))` as bits.
    const LN_GOLDENS: &[(u32, u32)] = &[
        (0x0080_0186, 0xC2AE_AC4A), // 1.175549e-38  worst |ulp| and |abs|
        (0x3F00_0841, 0xBF31_6196), // 0.50012594    worst inside the zero band
        (0x47C3_5000, 0x4138_34F1), // 100000.0   §0 model 1's rope_theta
        (0x4974_2400, 0x415D_0C55), // 1000000.0  §0 model 2's rope_theta
        (0x3F80_0000, 0x0000_0000), // 1.0 -> +0.0 exactly
        (0x4000_0000, 0x3F31_7218), // 2.0 -> ln 2
        (0x0080_0000, 0xC2AE_AC50), // min normal
        (0x7F7F_FFFF, 0x42B1_7218), // f32::MAX -> 88.72284, one ulp above the
        // largest argument whose exp is finite: the two guards are consistent
        // to within a rounding at the top of the range.
        // Guard domain, exact.
        (0x0000_0000, 0xFF80_0000), // +0.0 -> -Inf
        (0x8000_0000, 0xFF80_0000), // -0.0 -> -Inf (it takes the `== 0` branch)
        (0x0000_0001, 0xFF80_0000), // min subnormal -> -Inf, via §1.3's DAZ
        (0xBF80_0000, 0x7FC0_0000), // -1.0 -> NaN
    ];

    /// `(x, silu_pinned(x))` as bits. §6.4 is what actually consumes
    /// `exp_pinned`'s positive half, and a bound on `exp` is not by itself a
    /// bound on `x / (1 + exp(-x))`.
    const SILU_GOLDENS: &[(u32, u32)] = &[
        (0x3F00_0002, 0x3E9F_5981), // 0.5000001
        (0x3F80_0000, 0x3F3B_26A8), // 1.0
        (0xBF80_0000, 0xBE89_B2B1), // -1.0
        (0x4185_1592, 0x4185_1592), // 16.635532  worst |abs|; denom rounds to 1
        (0x4280_0000, 0x4280_0000), // 64.0       silu(x) == x exactly from here
        (0xC280_0000, 0x944B_4EA3), // -64.0
        (0xC2B0_0000, 0x8335_4DDC), // -88.0        last argument the poly sees
        (0xC2B0_0001, 0x8000_0000), // -88.0000076  §6.2's clip makes this -0.0
        (0x0000_0000, 0x0000_0000), // +0.0
        (0x8000_0000, 0x8000_0000), // -0.0
    ];

    fn at(f: fn(f32) -> f32, bits: u32) -> u32 {
        let x = black_box(f32::from_bits(black_box(bits)));
        black_box(f(x)).to_bits()
    }

    /// §6.2, §6.5 and §6.4 pinned at the arguments E27 found hardest and at
    /// every guard boundary the three functions have.
    #[test]
    fn exp_ln_silu_goldens_hold_across_the_whole_f32_domain() {
        fpenv::pin_and_selftest().expect("1.3 pin");
        for &(x, want) in EXP_GOLDENS {
            assert_eq!(at(exp_pinned, x), want, "exp_pinned at 0x{x:08X}");
        }
        for &(x, want) in LN_GOLDENS {
            assert_eq!(at(ln_pinned, x), want, "ln_pinned at 0x{x:08X}");
        }
        for &(x, want) in SILU_GOLDENS {
            assert_eq!(at(silu_pinned, x), want, "silu_pinned at 0x{x:08X}");
        }
    }

    /// The two exactness facts the rest of the specification quietly leans on.
    /// §7.1 builds `inv_freq[i] = exp_pinned(-((2i/head_dim) * ln_pinned(theta)))`,
    /// so `inv_freq[0] = exp_pinned(-0.0)`. E26's argument that "the largest
    /// RoPE angle a decode evaluates is its sequence length" holds only if
    /// that is exactly 1.0 and not one ULP below it.
    #[test]
    fn exp_of_signed_zero_is_exactly_one_and_ln_of_one_is_exactly_zero() {
        fpenv::pin_and_selftest().expect("1.3 pin");
        assert_eq!(at(exp_pinned, 0x8000_0000), 0x3F80_0000, "7.1 needs inv_freq[0] == 1.0");
        assert_eq!(at(exp_pinned, 0x0000_0000), 0x3F80_0000);
        assert_eq!(at(ln_pinned, 0x3F80_0000), 0x0000_0000, "6.5 must return +0.0, not -0.0");
    }

    /// Mutation control. A golden table is only worth having if it would
    /// actually fail on a wrong implementation, so this re-evaluates §6.2
    /// with the one substitution §6.2's two-part `ln 2` split exists to
    /// prevent -- a single-constant range reduction, `r = x - k*ln2` with
    /// `ln2` as one f32 -- and requires the goldens to disagree. This is the
    /// mutation §6.6's table digest cannot see: the coefficients are
    /// untouched, so `CIS2_REF` would not move, and only an op-level golden
    /// catches it. Without this test the table above could be vacuous and
    /// nothing would say so. Measured at E27: the mutation moves 9 of the 26
    /// `exp` goldens, and the bar is set at that number so a later edit that
    /// thins the table out fails here rather than silently.
    fn exp_with_single_constant_reduction(x: f32) -> f32 {
        if x.is_nan() {
            return x;
        }
        if x > 88.0 {
            return f32::INFINITY;
        }
        if x < -88.0 {
            return 0.0;
        }
        let t = black_box(x) * f(EXP_LOG2E);
        let k = floor_f32(t + 0.5);
        // The mutation: one rounded `ln 2` instead of §6.2's EXP_C1/EXP_C2 pair.
        let ln2 = f(EXP_C1) + f(EXP_C2);
        let r = x - k * ln2;
        let r2 = r * r;
        let mut poly = f(EXP_P[0]);
        for i in 1..=5 {
            poly = poly * r + f(EXP_P[i]);
        }
        let m1 = poly * r2;
        let poly2 = m1 + r;
        ldexp_exact(poly2 + 1.0, k as i32)
    }

    #[test]
    fn a_single_constant_range_reduction_is_caught_by_the_goldens() {
        fpenv::pin_and_selftest().expect("1.3 pin");
        let caught = EXP_GOLDENS
            .iter()
            .filter(|&&(x, want)| {
                let m = black_box(exp_with_single_constant_reduction(black_box(
                    f32::from_bits(x),
                )));
                m.to_bits() != want
            })
            .count();
        assert!(
            caught >= 9,
            "the one-constant reduction moved only {caught} of the {} exp goldens; \
             the table is too weak to be worth having",
            EXP_GOLDENS.len()
        );
    }

    /// The same control for §6.5. The mutation is the one an implementer
    /// reaching for a textbook `log` writes: drop the `m < LOG_SQRTHF`
    /// fix-up, which is what keeps the reduced mantissa in
    /// `[sqrt(0.5)-1, sqrt(2)-1]` rather than `[-0.5, 0]`. The coefficients
    /// are again untouched, so §6.6's table digest cannot see it. Measured at
    /// E27: it moves 5 of the 12 `ln` goldens -- 5 of the 8 that are not
    /// guard-domain entries, which the mutation does not reach.
    fn ln_without_the_mantissa_fixup(x: f32) -> f32 {
        if x.is_nan() || x < 0.0 {
            return f32::NAN;
        }
        if x == 0.0 {
            return f32::NEG_INFINITY;
        }
        let (mut m, e) = frexp_exact(black_box(x));
        m -= 1.0; // the mutation: no `if m < LOG_SQRTHF` branch
        let z = m * m;
        let mut poly = f(LOG_P[0]);
        for i in 1..=8 {
            poly = poly * m + f(LOG_P[i]);
        }
        let mut y = poly * m;
        y = y * z;
        let fe = e as f32;
        y = y + fe * f(LOG_Q1);
        y = y - 0.5 * z;
        m + y + fe * f(LOG_Q2)
    }

    #[test]
    fn a_missing_mantissa_fixup_is_caught_by_the_ln_goldens() {
        fpenv::pin_and_selftest().expect("1.3 pin");
        let caught = LN_GOLDENS
            .iter()
            .filter(|&&(x, want)| {
                let m = black_box(ln_without_the_mantissa_fixup(black_box(f32::from_bits(x))));
                m.to_bits() != want
            })
            .count();
        assert!(
            caught >= 5,
            "dropping the mantissa fix-up moved only {caught} of the {} ln goldens; \
             the table is too weak to be worth having",
            LN_GOLDENS.len()
        );
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
