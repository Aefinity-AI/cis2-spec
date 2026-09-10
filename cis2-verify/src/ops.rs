//! Spec 5 and 8: the numeric reduction primitives and RMSNorm.
//!
//! Spec 5.1 is the whole determinism argument in one function: strictly
//! left to right, index order 0..n-1, no pairwise tree, no chunking, no
//! reordering by magnitude, one separate RNE-rounded multiply and one
//! separate RNE-rounded add per element.

use crate::mathpin::{exp_pinned, rsqrt};
use alloc::vec;
use alloc::vec::Vec;

/// 5.1 sequential dot product.
#[inline]
pub fn dot_seq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        let p = a[i] * b[i];
        // E40 census. Compiled out entirely without the `census` feature, and
        // evaluated beside the f32 arithmetic, never in it, so no stored value
        // can depend on it.
        #[cfg(feature = "census")]
        {
            if census::maybe_tiny_mul(a[i], b[i]) {
                census::note_matvec(census::widen_exact(a[i]) * census::widen_exact(b[i]));
            }
            if census::maybe_tiny_add(acc, p) {
                census::note_matvec(census::widen_exact(acc) + census::widen_exact(p));
            }
        }
        acc = acc + p;
    }
    #[cfg(feature = "census")]
    census::note_matvec_bulk(2 * a.len() as u64);
    acc
}

/// 5.3 sequential sum.
#[inline]
pub fn sum_seq(a: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc = acc + a[i];
    }
    acc
}

/// 5.2 matvec over a row-major `[out_features, in_features]` weight.
/// Row order does not affect any single row's bits; each row's own
/// reduction uses 5.1's exact order.
pub fn matvec(w: &[f32], x: &[f32], out_features: usize, in_features: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), out_features * in_features);
    debug_assert_eq!(x.len(), in_features);
    let mut y = vec![0.0f32; out_features];
    for o in 0..out_features {
        y[o] = dot_seq(&w[o * in_features..(o + 1) * in_features], x);
    }
    y
}

/// 8 RMSNorm. The multiply order `(x[i] * inv) * weight[i]` is pinned
/// explicitly; `x[i] * (inv * weight[i])` is a different, non-conforming
/// association.
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut sq = vec![0.0f32; n];
    for i in 0..n {
        sq[i] = x[i] * x[i];
        #[cfg(feature = "census")]
        census::note_rms(census::widen_exact(x[i]) * census::widen_exact(x[i]));
    }
    let ss = sum_seq(&sq);
    let mean = ss / (n as f32);
    let inv = rsqrt(mean + eps);
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let scaled = x[i] * inv;
        out[i] = scaled * weight[i];
        #[cfg(feature = "census")]
        {
            census::note_rms(census::widen_exact(x[i]) * census::widen_exact(inv));
            census::note_rms(census::widen_exact(scaled) * census::widen_exact(weight[i]));
        }
        // 14.5 census. Compiled out entirely unless the `census` feature is
        // on, so no shipping build carries it and no digest can depend on it.
        #[cfg(feature = "census")]
        {
            let other = x[i] * (inv * weight[i]);
            census::note(out[i], other);
            if census::alt_order() {
                out[i] = other;
            }
        }
    }
    out
}

/// Spec 14.5 instrumentation: count how many of a real decode's RMSNorm
/// elements actually distinguish the two multiply associations, and
/// optionally run the whole decode in the non-conforming order so the
/// resulting digests can be compared.
///
/// This module exists only under `--features census`. It is not part of the
/// verifier: a census build is a measurement instrument, not a conforming
/// implementation, and it says so in its own output.
#[cfg(feature = "census")]
pub mod census {
    use core::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

    use core::sync::atomic::AtomicI64;

    static ELEMENTS: AtomicU64 = AtomicU64::new(0);
    static DIVERGENT: AtomicU64 = AtomicU64::new(0);
    static ALT: AtomicBool = AtomicBool::new(false);

    // 14.1 reach census. 6.2's high guard clips every `x > 88.0`, but
    // `ln(f32::MAX)` is 88.7228390520684, so the 94,743 arguments in
    // [0x42B00001, 0x42B17217] are clipped although their true `exp` is
    // finite. 10's softmax cannot reach that band -- its argument is always
    // <= 0 -- but 6.4's SiLU calls `exp_pinned(-x)`, so an FFN intermediate
    // in [-88.7228317, -88.0000076] does reach it, and there `silu_pinned`
    // returns -0.0 instead of a normal f32 near -5.3e-37. These counters
    // answer whether a real decode ever gets there. They only read.
    static SILU_N: AtomicU64 = AtomicU64::new(0);
    static SILU_CLIP_BAND: AtomicU64 = AtomicU64::new(0);
    static SILU_BELOW_88: AtomicU64 = AtomicU64::new(0);
    static SILU_MIN: AtomicI64 = AtomicI64::new(i64::MAX);
    static SILU_MAX: AtomicI64 = AtomicI64::new(i64::MIN);
    // 6.4 evaluates `exp_pinned(-x)`, so SiLU has its own subnormal window,
    // mirrored about zero: `-x` is in [-88.0, SUBNORM_TOP) exactly when `x`
    // is in (-SUBNORM_TOP, 88.0]. Above 88.0 the LOW guard fires on `-x` and
    // returns 0.0, so that side is clamped rather than computed. Counting
    // both windows is what makes E35 a statement about the whole `exp`
    // route rather than about softmax alone.
    static SILU_SUBNORM_BAND: AtomicU64 = AtomicU64::new(0);

    // 6.2's LOW guard, `x < -88.0 -> 0.0`, on the softmax path. Reachable by
    // construction (any score more than 88 below the row maximum), and
    // harmless by measurement (E27: `exp(-88)` is subnormal, so 1.3 flushes
    // it anyway). This counts how often a real decode exercises it.
    static SOFTMAX_N: AtomicU64 = AtomicU64::new(0);
    static SOFTMAX_LOW_GUARD: AtomicU64 = AtomicU64::new(0);

    // The window the low guard does NOT cover, and the one that actually
    // matters to 1.3. `ln(f32::MIN_POSITIVE)` is -87.3365447505531, so
    // `exp_pinned(a)` is **subnormal** for `a` in [-88.0, -87.3365447505531)
    // -- above the guard, so it is computed rather than clamped. With FTZ
    // pinned that subnormal is flushed to zero; without it, it survives and
    // contributes to the softmax denominator. That is precisely the case
    // E22's M01 mutant (FTZ/DAZ gutted) would have to hit to move the digest,
    // and no counter measured it: `SOFTMAX_LOW_GUARD` counts `a < -88.0`,
    // where the guard fires and the answer is 0.0 either way. E22 called this
    // out as unresolved -- "no denormal *changed a result*, which is slightly
    // weaker than no denormal ever arose" -- and this counter is what closes
    // it. The arg range says how far a real decode lands from the window.
    static SOFTMAX_SUBNORM_BAND: AtomicU64 = AtomicU64::new(0);
    // E39: the counters that measure where 1.3's FTZ is ACTUALLY decisive in
    // 10's softmax --- the elementwise division, not the exp. `exp_pinned`
    // cannot return a subnormal (ldexp_exact returns 0.0 whenever the
    // reconstructed exponent field would be <= 0), so no FTZ decision is ever
    // taken on its result. The quotient `exp / denom` can be subnormal.
    static WEIGHT_N: AtomicU64 = AtomicU64::new(0);
    static WEIGHT_TRUE_SUBNORM: AtomicU64 = AtomicU64::new(0);
    static WEIGHT_FTZ_DECISIVE: AtomicU64 = AtomicU64::new(0);
    static WEIGHT_MIN_NONZERO: AtomicI64 = AtomicI64::new(i64::MAX);
    // E40: the same question at 5.2/5.3/8, which no counter has ever reached.
    // MATVEC_* count intermediates of `dot_seq` (each product and each partial
    // sum); RMS_* count 8's `x*x`, `x*inv` and `scaled*weight`. `*_DECISIVE`
    // is the only one that means 1.3 changed the stored bits.
    static MV_N: AtomicU64 = AtomicU64::new(0);
    static MV_TRUE_SUBNORM: AtomicU64 = AtomicU64::new(0);
    static MV_FTZ_DECISIVE: AtomicU64 = AtomicU64::new(0);
    static RMS_N: AtomicU64 = AtomicU64::new(0);
    static RMS_TRUE_SUBNORM: AtomicU64 = AtomicU64::new(0);
    static RMS_FTZ_DECISIVE: AtomicU64 = AtomicU64::new(0);
    static SOFTMAX_MIN: AtomicI64 = AtomicI64::new(i64::MAX);
    static SOFTMAX_MAX: AtomicI64 = AtomicI64::new(i64::MIN);

    /// Exclusive top of the subnormal band: the largest f32 whose
    /// `exp_pinned` is still normal is `SUBNORM_TOP` itself, so `a <
    /// SUBNORM_TOP` holds exactly for the arguments whose `exp_pinned` is
    /// subnormal (`0xC2AE_AC4F` is -87.336_540_222_167_97, one ULP above
    /// `ln(f32::MIN_POSITIVE)` = -87.336_544_750_553_1 rounded to f32).
    /// `subnorm_band_edge_is_exp_pinneds_own_boundary` pins this against
    /// `exp_pinned` rather than against `libm`, so the band tracks the
    /// implementation the digest actually runs.
    pub const SUBNORM_TOP: f32 = f32::from_bits(0xC2AE_AC4F);

    /// Total order on f32 bits, so `fetch_min`/`fetch_max` mean what they say
    /// across the sign boundary.
    fn key(v: f32) -> i64 {
        let b = v.to_bits() as i64;
        if v.to_bits() & 0x8000_0000 != 0 { 0x8000_0000i64 - b } else { b }
    }
    fn unkey(k: i64) -> f32 {
        let b = if k < 0 { (0x8000_0000i64 - k) as u32 } else { k as u32 };
        f32::from_bits(b)
    }

    /// One SiLU argument (6.4's `x`, i.e. the FFN gate value).
    pub fn note_silu(x: f32) {
        SILU_N.fetch_add(1, Relaxed);
        let b = x.to_bits();
        if (0xC2B0_0001..=0xC2B1_7217).contains(&b) {
            SILU_CLIP_BAND.fetch_add(1, Relaxed);
        }
        if x < -88.0 {
            SILU_BELOW_88.fetch_add(1, Relaxed);
        } else if x > -SUBNORM_TOP {
            SILU_SUBNORM_BAND.fetch_add(1, Relaxed);
        }
        let k = key(x);
        SILU_MIN.fetch_min(k, Relaxed);
        SILU_MAX.fetch_max(k, Relaxed);
    }

    /// One softmax `exp_pinned` argument (`v - max_v`, always <= 0).
    pub fn note_softmax_exp_arg(a: f32) {
        SOFTMAX_N.fetch_add(1, Relaxed);
        if a < -88.0 {
            SOFTMAX_LOW_GUARD.fetch_add(1, Relaxed);
        } else if a < SUBNORM_TOP {
            SOFTMAX_SUBNORM_BAND.fetch_add(1, Relaxed);
        }
        SOFTMAX_MIN.fetch_min(key(a), Relaxed);
        SOFTMAX_MAX.fetch_max(key(a), Relaxed);
    }

    /// `(calls, in_clip_band, below_-88, in_subnormal_band, min_arg,
    /// max_arg)` for 6.4's SiLU.
    pub fn silu_counts() -> (u64, u64, u64, u64, f32, f32) {
        (
            SILU_N.load(Relaxed),
            SILU_CLIP_BAND.load(Relaxed),
            SILU_BELOW_88.load(Relaxed),
            SILU_SUBNORM_BAND.load(Relaxed),
            unkey(SILU_MIN.load(Relaxed)),
            unkey(SILU_MAX.load(Relaxed)),
        )
    }

    /// One softmax weight, at the elementwise division of 10's final step.
    /// `num` is `exp_pinned(v - max_v)`, `denom` the 5.1 sum. The exact
    /// quotient is formed in f64 --- both operands widen exactly --- so the
    /// census can see the value the f32 division *would* have produced before
    /// 1.3's FTZ had a say.
    ///
    /// E39: this is the place 1.3 can be decisive in 10, and neither E35 nor
    /// E38 counted it. They counted `exp_pinned` arguments whose *true* exp is
    /// subnormal, which is a different thing: `exp_pinned` itself returns
    /// either 0.0 or a NORMAL f32, never a subnormal, because `ldexp_exact`
    /// returns 0.0 whenever the reconstructed exponent field would be <= 0.
    pub fn note_softmax_weight(num: f32, denom: f32) {
        WEIGHT_N.fetch_add(1, Relaxed);
        let q = widen_exact(num) / widen_exact(denom);
        let a = q.abs();
        if q != 0.0 && a < f32::MIN_POSITIVE as f64 {
            WEIGHT_TRUE_SUBNORM.fetch_add(1, Relaxed);
            // Below half the smallest subnormal the f32 rounding is +0.0 with
            // or without FTZ, so FTZ decides nothing. At or above it, the
            // rounded value IS a nonzero subnormal and FTZ changes the bits.
            const HALF_MIN_SUBNORM: f64 = 7.006492321624085e-46;
            if a >= HALF_MIN_SUBNORM {
                WEIGHT_FTZ_DECISIVE.fetch_add(1, Relaxed);
            }
        }
        if q != 0.0 {
            WEIGHT_MIN_NONZERO.fetch_min(a.to_bits() as i64, Relaxed);
        }
    }

    /// Below half the smallest subnormal an f32 result rounds to `+0.0` with
    /// or without FTZ, so FTZ decides nothing there (erratum E-11's trap).
    pub const HALF_MIN_SUBNORM: f64 = 7.006492321624085e-46;

    /// Widen an f32 to f64 **without an SSE conversion**, by decoding the bit
    /// pattern with integer arithmetic and rebuilding the value from a normal
    /// double.
    ///
    /// E40: `x as f64` is `cvtss2sd`, and §1.3's DAZ makes it read a subnormal
    /// operand as zero — so a census built on `as f64` is blind to exactly the
    /// DAZ cases it exists to count. Its positive control caught this: an
    /// RMSNorm case whose stored bits differ pinned vs unpinned was scored 0.
    /// Every input here is an integer or a normal double, so no MXCSR bit can
    /// act on it.
    #[inline]
    pub fn widen_exact(x: f32) -> f64 {
        let b = x.to_bits();
        let sign = if b >> 31 == 1 { -1.0f64 } else { 1.0f64 };
        let exp = ((b >> 23) & 0xFF) as i32;
        let mant = (b & 0x007F_FFFF) as u64;
        if exp == 0 {
            // Subnormal (or zero): value = mant * 2^-149.
            sign * (mant as f64) * 1.401_298_464_324_817_1e-45
        } else if exp == 0xFF {
            if mant == 0 { sign * f64::INFINITY } else { f64::NAN }
        } else {
            // Normal: (2^23 + mant) * 2^(exp - 150).
            sign * ((mant + (1 << 23)) as f64) * exp2i(exp - 150)
        }
    }

    /// `2^n` for the exponent range an f32 can carry, built by integer bit
    /// assembly so it needs no `powi` and no table.
    #[inline]
    fn exp2i(n: i32) -> f64 {
        // f32's normal exponents run [-126, 127]; minus 150 puts n in
        // [-149, -23]... [104], all comfortably normal for f64.
        f64::from_bits((((n + 1023) as u64) & 0x7FF) << 52)
    }

    /// Classify one exact intermediate value. `exact` must be computed in f64
    /// from f32 operands, which is exact in the range that matters here, and
    /// is itself far from the f64 subnormal range so MXCSR cannot flush it.
    #[inline]
    fn note_exact(exact: f64, n: &AtomicU64, sub: &AtomicU64, dec: &AtomicU64) {
        n.fetch_add(1, Relaxed);
        let a = exact.abs();
        if exact != 0.0 && a < f32::MIN_POSITIVE as f64 {
            sub.fetch_add(1, Relaxed);
            if a >= HALF_MIN_SUBNORM {
                dec.fetch_add(1, Relaxed);
            }
        }
    }

    /// Integer pre-filter: could `a * b` be subnormal or below? Returns a
    /// conservative superset — never false when the product is subnormal.
    ///
    /// The product is at least `2^(ea + eb - 254)`, so `ea + eb >= 129`
    /// guarantees a magnitude of at least `2^-125`, comfortably normal. A zero
    /// exponent field (a subnormal or zero operand) always passes, so DAZ
    /// cases are never filtered out. This is integer arithmetic on the bit
    /// patterns and cannot itself be perturbed by MXCSR.
    #[inline]
    pub fn maybe_tiny_mul(a: f32, b: f32) -> bool {
        let ea = (a.to_bits() >> 23) & 0xFF;
        let eb = (b.to_bits() >> 23) & 0xFF;
        ea == 0 || eb == 0 || ea + eb <= 130
    }

    /// Integer pre-filter for `a + b`. `|a + b| <= 2 * max(|a|, |b|)`, so a
    /// subnormal sum requires the larger operand below `2^-125` — exponent
    /// field at most 2. The margin is generous on purpose.
    #[inline]
    pub fn maybe_tiny_add(a: f32, b: f32) -> bool {
        let ea = (a.to_bits() >> 23) & 0xFF;
        let eb = (b.to_bits() >> 23) & 0xFF;
        let m = if ea > eb { ea } else { eb };
        m <= 4
    }

    /// Add `k` to the 5.1 intermediate total without classifying each one.
    /// The classification is done only for intermediates the integer filter
    /// admits, so the per-element cost of the census is two integer compares.
    #[inline]
    pub fn note_matvec_bulk(k: u64) {
        MV_N.fetch_add(k, Relaxed);
    }

    /// One 5.1 intermediate: a product or a partial sum inside `dot_seq`.
    #[inline]
    pub fn note_matvec(exact: f64) {
        let a = exact.abs();
        if exact != 0.0 && a < f32::MIN_POSITIVE as f64 {
            MV_TRUE_SUBNORM.fetch_add(1, Relaxed);
            if a >= HALF_MIN_SUBNORM {
                MV_FTZ_DECISIVE.fetch_add(1, Relaxed);
            }
        }
    }

    /// One 8 RMSNorm intermediate.
    #[inline]
    pub fn note_rms(exact: f64) {
        note_exact(exact, &RMS_N, &RMS_TRUE_SUBNORM, &RMS_FTZ_DECISIVE);
    }

    /// `(intermediates, true_subnormal, ftz_decisive)` for 5.1 reductions.
    pub fn matvec_counts() -> (u64, u64, u64) {
        (MV_N.load(Relaxed), MV_TRUE_SUBNORM.load(Relaxed), MV_FTZ_DECISIVE.load(Relaxed))
    }

    /// `(intermediates, true_subnormal, ftz_decisive)` for 8 RMSNorm.
    pub fn rms_counts() -> (u64, u64, u64) {
        (RMS_N.load(Relaxed), RMS_TRUE_SUBNORM.load(Relaxed), RMS_FTZ_DECISIVE.load(Relaxed))
    }

    /// `(weights, true_subnormal_quotients, ftz_decisive, min_nonzero_|q|)`.
    /// `ftz_decisive > 0` is the only measurement that establishes 1.3 is
    /// digest-relevant to 10 on this vector.
    pub fn weight_counts() -> (u64, u64, u64, f64) {
        (
            WEIGHT_N.load(Relaxed),
            WEIGHT_TRUE_SUBNORM.load(Relaxed),
            WEIGHT_FTZ_DECISIVE.load(Relaxed),
            {
                let m = WEIGHT_MIN_NONZERO.load(Relaxed);
                if m == i64::MAX { f64::NAN } else { f64::from_bits(m as u64) }
            },
        )
    }

    /// `(calls, hit_low_guard, in_subnormal_band, min_arg, max_arg)` for 10's
    /// softmax `exp_pinned`. NOTE (E39): `in_subnormal_band` counts arguments
    /// whose *true* exp is subnormal. It does NOT mean a subnormal was
    /// computed and flushed --- `exp_pinned` never returns one. Use
    /// `weight_counts` for the FTZ question.
    pub fn softmax_counts() -> (u64, u64, u64, f32, f32) {
        (
            SOFTMAX_N.load(Relaxed),
            SOFTMAX_LOW_GUARD.load(Relaxed),
            SOFTMAX_SUBNORM_BAND.load(Relaxed),
            unkey(SOFTMAX_MIN.load(Relaxed)),
            unkey(SOFTMAX_MAX.load(Relaxed)),
        )
    }

    pub fn note(pinned: f32, other: f32) {
        ELEMENTS.fetch_add(1, Relaxed);
        if pinned.to_bits() != other.to_bits() {
            DIVERGENT.fetch_add(1, Relaxed);
        }
    }
    pub fn alt_order() -> bool {
        ALT.load(Relaxed)
    }
    pub fn set_alt_order(v: bool) {
        ALT.store(v, Relaxed);
    }
    pub fn reset() {
        ELEMENTS.store(0, Relaxed);
        DIVERGENT.store(0, Relaxed);
        SILU_N.store(0, Relaxed);
        SILU_CLIP_BAND.store(0, Relaxed);
        SILU_BELOW_88.store(0, Relaxed);
        SILU_SUBNORM_BAND.store(0, Relaxed);
        SILU_MIN.store(i64::MAX, Relaxed);
        SILU_MAX.store(i64::MIN, Relaxed);
        SOFTMAX_N.store(0, Relaxed);
        SOFTMAX_LOW_GUARD.store(0, Relaxed);
        SOFTMAX_SUBNORM_BAND.store(0, Relaxed);
        SOFTMAX_MIN.store(i64::MAX, Relaxed);
        SOFTMAX_MAX.store(i64::MIN, Relaxed);
    }
    /// `(elements, divergent)`
    pub fn counts() -> (u64, u64) {
        (ELEMENTS.load(Relaxed), DIVERGENT.load(Relaxed))
    }

    #[cfg(test)]
    mod tests {
        use super::SUBNORM_TOP;
        use crate::mathpin::exp_pinned;

        /// E40: the census must widen its operands without an SSE conversion.
        /// `x as f64` is `cvtss2sd`, and 1.3's DAZ reads a subnormal operand
        /// as zero, so a census built on it is blind to exactly the DAZ cases
        /// it exists to count --- which is how E39's counter shipped able to
        /// score a genuine FTZ-decisive RMSNorm case as zero.
        ///
        /// The assertion is on `widen_exact` alone and holds with or without
        /// the pin in the harness thread, so it is a property of the decoder
        /// rather than of the environment the test happens to run in.
        #[test]
        fn widen_exact_sees_subnormals_that_daz_would_hide() {
            // Smallest subnormal: 1 * 2^-149.
            let tiny = f32::from_bits(0x0000_0001);
            let w = super::widen_exact(tiny);
            assert!(w > 0.0, "widen_exact flattened the smallest subnormal");
            assert!(
                (w - 1.401_298_464_324_817_1e-45).abs() < 1e-60,
                "widen_exact({tiny:e}) = {w:e}"
            );
            // A mid subnormal, and its sign.
            let mid = f32::from_bits(0x0040_0000);
            assert!((super::widen_exact(mid) - 5.877_471_754_111_438e-39).abs() < 1e-54);
            assert!(super::widen_exact(-mid) < 0.0);
            // Normals, zeros and the boundary must widen exactly as `as f64`
            // does --- the two agree everywhere except on subnormal operands.
            for b in [0x0000_0000u32, 0x8000_0000, 0x0080_0000, 0x3F80_0000, 0xC2AE_AC4F] {
                let x = f32::from_bits(b);
                assert_eq!(
                    super::widen_exact(x).to_bits(),
                    (x as f64).to_bits(),
                    "widen_exact disagrees with `as f64` on the normal {b:#010x}"
                );
            }
        }

        /// The integer pre-filters must be conservative supersets: anything
        /// they reject is provably not subnormal, so a filtered intermediate
        /// can never be a missed FTZ-decisive one.
        #[test]
        fn tiny_filters_never_reject_a_subnormal_result() {
            // Products that ARE subnormal must be admitted.
            for (a, b) in [
                (f32::MIN_POSITIVE, 1.0e-2f32),
                (f32::MIN_POSITIVE, 1.0e-7f32),
                (f32::from_bits(0x0000_0001), 1.0f32),
                (1.0e-30f32, 1.0e-10f32),
            ] {
                assert!(super::maybe_tiny_mul(a, b), "filter rejected {a:e} * {b:e}");
            }
            // Sums that ARE subnormal must be admitted.
            for (a, b) in [
                (f32::MIN_POSITIVE, -f32::from_bits(0x007F_FFFF)),
                (f32::from_bits(0x0000_0002), f32::from_bits(0x0000_0001)),
            ] {
                assert!(super::maybe_tiny_add(a, b), "filter rejected {a:e} + {b:e}");
            }
            // And the filter must actually filter, or it buys nothing.
            assert!(!super::maybe_tiny_mul(1.0, 1.0));
            assert!(!super::maybe_tiny_add(1.0, -1.0));
        }

        /// The band's edge is pinned against `exp_pinned` itself, not against
        /// `libm`, because `exp_pinned` is what the digest runs. The assertion
        /// is written as `is_normal()` on both sides so it holds whether or
        /// not 1.3's FTZ pin is in effect in the harness thread: with the pin
        /// a band argument returns +0.0, without it a subnormal, and neither
        /// is normal, while an argument at or above the edge returns a normal
        /// f32 either way. That is exactly the asymmetry `SOFTMAX_SUBNORM_BAND`
        /// exists to count.
        #[test]
        fn subnorm_band_edge_is_exp_pinneds_own_boundary() {
            let top = SUBNORM_TOP;
            assert_eq!(top.to_bits(), 0xC2AE_AC4F);

            // At the edge and one ULP above it (toward zero): normal.
            assert!(exp_pinned(top).is_normal(), "exp_pinned({top}) should be normal");
            let above = f32::from_bits(top.to_bits() - 1);
            assert!(exp_pinned(above).is_normal(), "exp_pinned({above}) should be normal");

            // One ULP below the edge (further from zero): not normal, i.e.
            // subnormal unpinned and flushed to zero pinned.
            let below = f32::from_bits(top.to_bits() + 1);
            assert!(!exp_pinned(below).is_normal(), "exp_pinned({below}) should not be normal");

            // The LOW guard's own boundary sits inside the band: -88.0 is not
            // clamped (6.2's test is strict `< -88.0`) and its exp is
            // subnormal, so the band and the guard meet without a gap.
            assert!(!exp_pinned(-88.0).is_normal());
            assert!(-88.0f32 < top);
        }

        /// A counter that cannot fire proves nothing when it reads zero, so
        /// drive a score gap through the real `softmax_seq` and check that
        /// each of the three regions lands in the bucket it should. This is
        /// the positive control for E35's `n=0` sweep result: without it,
        /// "no decode reached the subnormal band" and "the instrument is
        /// dead" are the same observation.
        #[test]
        fn subnorm_band_counter_fires_on_a_constructed_gap() {
            // Inside the band: computed, not clamped, and subnormal.
            crate::ops::census::reset();
            let mut v = [0.0f32, -87.5];
            crate::ops::softmax_seq(&mut v);
            let (n, low, band, min, max) = crate::ops::census::softmax_counts();
            assert_eq!((n, low, band), (2, 0, 1));
            assert_eq!(min.to_bits(), (-87.5f32).to_bits());
            assert_eq!(max.to_bits(), 0.0f32.to_bits());

            // Below the guard: 6.2 clamps to 0.0 before `exp` is evaluated,
            // so this is NOT a subnormal-band case and must not be counted
            // as one. Conflating the two is what made E29's n=0 unreadable.
            crate::ops::census::reset();
            let mut v = [0.0f32, -90.0];
            crate::ops::softmax_seq(&mut v);
            assert_eq!(crate::ops::census::softmax_counts().0, 2);
            assert_eq!(crate::ops::census::softmax_counts().1, 1);
            assert_eq!(crate::ops::census::softmax_counts().2, 0);

            // Above the band: an ordinary normal `exp`, counted in neither.
            crate::ops::census::reset();
            let mut v = [0.0f32, -80.0];
            crate::ops::softmax_seq(&mut v);
            assert_eq!(crate::ops::census::softmax_counts().1, 0);
            assert_eq!(crate::ops::census::softmax_counts().2, 0);
        }

        /// 6.4's window is the same one mirrored, because SiLU evaluates
        /// `exp_pinned(-x)`. Checked against `exp_pinned` on the same
        /// arguments so the bucket boundary and the arithmetic cannot drift
        /// apart.
        #[test]
        fn silu_subnorm_band_is_the_softmax_band_mirrored() {
            use crate::ops::census;

            census::reset();
            census::note_silu(87.5); // -87.5 is inside the band
            census::note_silu(-90.0); // -(-90) = 90 -> LOW guard on -x
            census::note_silu(0.0); // ordinary
            let (n, _clip, below, band, _min, _max) = census::silu_counts();
            assert_eq!((n, below, band), (3, 1, 1));

            assert!(!exp_pinned(-87.5).is_normal());
            assert!(exp_pinned(-0.0).is_normal());

            // The mirror is exact: negation of an f32 is exact, so the two
            // windows share one constant rather than two rounded literals.
            assert_eq!((-SUBNORM_TOP).to_bits(), SUBNORM_TOP.to_bits() ^ 0x8000_0000);
            census::reset();
        }
    }
}

/// 9.3 softmax, in place. `>` is strict, so the FIRST occurrence of the
/// maximum wins ties; the final step is elementwise division, not
/// multiply-by-reciprocal.
pub fn softmax_seq(scores: &mut [f32]) {
    let mut max_v = scores[0];
    for &v in scores[1..].iter() {
        if v > max_v {
            max_v = v;
        }
    }
    for v in scores.iter_mut() {
        let arg = *v - max_v;
        #[cfg(feature = "census")]
        census::note_softmax_exp_arg(arg);
        *v = exp_pinned(arg);
    }
    let denom = sum_seq(scores);
    for v in scores.iter_mut() {
        #[cfg(feature = "census")]
        census::note_softmax_weight(*v, denom);
        *v = *v / denom;
    }
}

/// 11.2 argmax: strict left-to-right, first occurrence wins on exact ties.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = logits[0];
    for (idx, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = idx;
        }
    }
    best_idx as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fpenv;
    use core::hint::black_box;

    /// Spec 8 pins `(x[i] * inv) * weight[i]`. Spec 14.5 records that the
    /// bit-level *necessity* of that pin was unconfirmed: the other
    /// association had never been shown to produce different bits. These
    /// tests confirm it.
    ///
    /// Every operand goes through `black_box` in both directions, per the
    /// house rule — see `fpenv`'s test module for the failure that prevents.
    /// Here the risk is concrete: LLVM is entitled to fold
    /// `f32::from_bits(K1) * f32::from_bits(K2)` at compile time, and a
    /// folded constant would report the compiler's arithmetic rather than the
    /// pinned runtime FPU's.
    fn assoc_lr(xb: u32, ib: u32, wb: u32) -> u32 {
        let x = black_box(f32::from_bits(black_box(xb)));
        let inv = black_box(f32::from_bits(black_box(ib)));
        let w = black_box(f32::from_bits(black_box(wb)));
        let scaled = black_box(x * inv);
        black_box(scaled * w).to_bits()
    }

    fn assoc_rl(xb: u32, ib: u32, wb: u32) -> u32 {
        let x = black_box(f32::from_bits(black_box(xb)));
        let inv = black_box(f32::from_bits(black_box(ib)));
        let w = black_box(f32::from_bits(black_box(wb)));
        let folded = black_box(inv * w);
        black_box(x * folded).to_bits()
    }

    /// `(x, inv, weight, (x*inv)*w, x*(inv*w))` — triples at the magnitudes
    /// RMSNorm actually sees (unit-ish activations, `inv` in [0.2, 5],
    /// weights near 1) for which the two associations differ. Every pair here
    /// differs by exactly one ulp, which is the whole point: one ulp in a
    /// hidden state is enough to move an argmax later in the decode.
    const ASSOC_DIVERGES: &[(u32, u32, u32, u32, u32)] = &[
        (0x3EFF_136D, 0x409C_E023, 0x3F88_2890, 0x4026_45A5, 0x4026_45A6),
        (0xBFB4_8836, 0x3FFA_D914, 0x3F6E_2B49, 0xC024_93D5, 0xC024_93D6),
        (0x3E0D_F647, 0x3E71_F232, 0x3F8C_EF0D, 0x3D13_B9C5, 0x3D13_B9C6),
        (0xBF1F_9348, 0x3F42_71B8, 0x3F85_D672, 0xBEFD_7738, 0xBEFD_7737),
        (0x3E61_1015, 0x4039_8E47, 0x3F93_B432, 0x3F3C_3E5D, 0x3F3C_3E5C),
    ];

    /// Triples at the same magnitudes for which the two associations agree,
    /// kept so `orders_are_not_interchangeable` cannot be satisfied by a
    /// table that happens to diverge everywhere. The last three are exactly
    /// representable, where association is exact by construction.
    const ASSOC_AGREES: &[(u32, u32, u32, u32)] = &[
        (0xBF8B_948B, 0x3FCB_7C51, 0x3F70_99F5, 0xBFD0_8C45),
        (0xBE54_3C92, 0x4088_617F, 0x3F5B_C879, 0xBF42_242F),
        (0xBFB9_5B5C, 0x4017_DFB2, 0x3F51_03E3, 0xC033_9068),
        (0x3FC0_0000, 0x4000_0000, 0x4080_0000, 0x4140_0000), // 1.5, 2, 4
        (0xBF40_0000, 0x4100_0000, 0x3F00_0000, 0xC040_0000), // -0.75, 8, 0.5
        (0x4040_0000, 0x3E80_0000, 0x4180_0000, 0x4140_0000), // 3, 0.25, 16
    ];

    /// 14.5 witness, scalar half: the pinned association is not a notational
    /// preference. On these operands the two orders give different bits, and
    /// the bits the spec pins are the ones on the left.
    #[test]
    fn orders_are_not_interchangeable() {
        fpenv::pin_and_selftest().expect("1.3 pin");
        for &(x, i, w, lr, rl) in ASSOC_DIVERGES {
            assert_eq!(assoc_lr(x, i, w), lr, "pinned order moved: 0x{x:08X} 0x{i:08X} 0x{w:08X}");
            assert_eq!(assoc_rl(x, i, w), rl, "other order moved: 0x{x:08X} 0x{i:08X} 0x{w:08X}");
            assert_ne!(
                lr, rl,
                "0x{x:08X} 0x{i:08X} 0x{w:08X} is in the diverging table but does not diverge"
            );
        }
        for &(x, i, w, both) in ASSOC_AGREES {
            assert_eq!(assoc_lr(x, i, w), both, "0x{x:08X} 0x{i:08X} 0x{w:08X}");
            assert_eq!(assoc_rl(x, i, w), both, "0x{x:08X} 0x{i:08X} 0x{w:08X}");
        }
    }

    /// 14.5 witness, vector half: the divergence survives the real function.
    /// `rmsnorm` must agree with the pinned association at every index and
    /// disagree with the other one somewhere — otherwise 8's wording is
    /// describing something unobservable.
    #[test]
    fn rmsnorm_uses_the_pinned_order() {
        fpenv::pin_and_selftest().expect("1.3 pin");

        let x: Vec<f32> = [
            0x3EFF_136Eu32, 0xBFA2_FC70, 0x3DBB_F402, 0x4000_B9C1,
            0xBF24_6150, 0x3FB8_6C78, 0xBEAB_8D57, 0x3F45_DF80,
        ]
        .iter()
        .map(|&b| black_box(f32::from_bits(black_box(b))))
        .collect();
        let weight: Vec<f32> = [
            0x3F88_2890u32, 0x3F6B_1079, 0x3F98_6186, 0x3F5E_FD38,
            0x3F83_DB00, 0x3F74_8000, 0x3F9A_98D0, 0x3F67_4000,
        ]
        .iter()
        .map(|&b| black_box(f32::from_bits(black_box(b))))
        .collect();
        let eps = black_box(1.0e-5f32);

        let got = rmsnorm(&x, &weight, eps);

        // Re-derive `inv` exactly as 8 does, so the only difference between
        // the two candidate outputs below is the multiply association.
        let n = x.len();
        let mut sq = vec![0.0f32; n];
        for i in 0..n {
            sq[i] = x[i] * x[i];
        }
        let inv = rsqrt(sum_seq(&sq) / (n as f32) + eps);

        let mut diverged = 0usize;
        for i in 0..n {
            let pinned = black_box(black_box(x[i] * inv) * weight[i]);
            let other = black_box(x[i] * black_box(inv * weight[i]));
            assert_eq!(
                got[i].to_bits(),
                pinned.to_bits(),
                "rmsnorm index {i} does not use 8's pinned association"
            );
            if pinned.to_bits() != other.to_bits() {
                diverged += 1;
            }
        }
        assert!(
            diverged > 0,
            "this vector does not distinguish the two associations, so it \
             proves nothing about 8"
        );
    }
}

/// Spec 14.6 measurement only: a per-layer activation tap.
///
/// Section 14.6 records that the oracle checks of section 13.3 are
/// spot-checks on the *output* of the stack — greedy token ids and one
/// step's full logit vector — and that a compensating pair of errors inside
/// the layer stack, which happened to preserve step-0's logits and every
/// argmax decision, could not be ruled out by that evidence alone. Closing
/// that gap needs the *intermediate* tensors, compared against an
/// independent implementation.
///
/// This module exists only under `--features layerdump`. A layerdump build
/// writes named f32 tensors to a file and is not a conforming verifier; the
/// tap is write-only and touches no value the forward pass reads, which is
/// checked by the digests the dumping run reproduces.
///
/// Record format, little-endian, repeated until EOF:
///
/// ```text
///   u32  name_len
///   u8   name[name_len]        (ASCII)
///   u32  n                     (element count)
///   f32  data[n]
/// ```
#[cfg(feature = "layerdump")]
pub mod tap {
    use std::fs::File;
    use std::io::{BufWriter, Write};
    use std::sync::Mutex;

    static SINK: Mutex<Option<BufWriter<File>>> = Mutex::new(None);

    /// Open the dump file. Any previously open sink is flushed and dropped.
    pub fn open(path: &str) {
        let f = File::create(path).expect("layerdump: cannot create dump file");
        *SINK.lock().unwrap() = Some(BufWriter::new(f));
    }

    /// Flush and close, returning the number of bytes the file holds.
    pub fn close() {
        let mut g = SINK.lock().unwrap();
        if let Some(mut w) = g.take() {
            w.flush().expect("layerdump: flush failed");
        }
    }

    /// Record one named tensor. Read-only in `v`.
    pub fn emit(name: &str, v: &[f32]) {
        let mut g = SINK.lock().unwrap();
        let Some(w) = g.as_mut() else { return };
        let nb = name.as_bytes();
        w.write_all(&(nb.len() as u32).to_le_bytes()).unwrap();
        w.write_all(nb).unwrap();
        w.write_all(&(v.len() as u32).to_le_bytes()).unwrap();
        for x in v {
            w.write_all(&x.to_bits().to_le_bytes()).unwrap();
        }
    }
}
