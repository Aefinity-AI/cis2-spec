//! Bit-level floating-point helpers, all exact and all implemented without
//! `std`.
//!
//! Spec 1.6 says ordinary IEEE-754 `/` and `sqrt` are the one part of the
//! arithmetic the specification does not need to additionally pin, because
//! both are mandatory-correctly-rounded operations: *any* conformant
//! binary32 implementation gives identical bits. This crate takes that
//! claim at its word and tests it — `sqrt_cr` below is an integer software
//! square root that never touches the FPU's `sqrtss`/`fsqrt` instruction,
//! and it is proven correctly rounded by construction (the exact integer
//! remainder decides every rounding case, including ties-to-even). If the
//! specification's reasoning in 1.6 is sound, this implementation and a
//! hardware `sqrt` must agree on every input, and the whole decode must
//! land on the same digest either way.
//!
//! `/` is used as the ordinary operator throughout, which `core` provides.

/// Correctly-rounded IEEE-754 binary32 square root, computed with integer
/// arithmetic only.
///
/// Domain notes: negative inputs give NaN, `+0.0`/`-0.0` are returned
/// unchanged, `+inf` returns `+inf`, NaN returns NaN. Subnormal inputs are
/// handled by normalizing first; with the spec 1.3 FTZ/DAZ pin in effect
/// no decode-path value is ever subnormal, but correctness here must not
/// depend on that.
pub fn sqrt_cr(x: f32) -> f32 {
    let bits = x.to_bits();
    let sign = bits >> 31;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x007f_ffff;

    if exp == 0xff {
        // inf or NaN
        if man != 0 {
            return x; // NaN propagates
        }
        return if sign == 1 { f32::from_bits(0x7fc0_0000) } else { x };
    }
    if bits & 0x7fff_ffff == 0 {
        return x; // +/-0.0 -> itself, sign preserved
    }
    if sign == 1 {
        return f32::from_bits(0x7fc0_0000); // sqrt of a negative -> NaN
    }

    // Decompose into significand * 2^e with the significand an integer.
    let (mut sig, mut e) = if exp == 0 {
        // Subnormal: value = man * 2^-149.
        (man as u64, -149i32)
    } else {
        // Normal: value = (2^23 + man) * 2^(exp - 127 - 23).
        ((man | 0x0080_0000) as u64, exp - 150)
    };

    // sqrt(sig * 2^e) needs an even e. Shifting sig left by one and
    // decrementing e is exact.
    if e % 2 != 0 {
        sig <<= 1;
        e -= 1;
    }

    // Scale so the integer square root carries well over 24 significant
    // bits, then round from the exact remainder. sig < 2^25 here, so
    // sig << 52 < 2^77 and the u128 never overflows.
    let scaled = (sig as u128) << 52;
    let q = isqrt_u128(scaled);
    let r = scaled - q * q; // exact remainder; r == 0 iff the root is exact
    let half_e = e / 2 - 26; // the 52-bit scaling contributes 2^26 to q

    // q has 38 or 39 significant bits. Normalize to a 24-bit significand
    // with round-to-nearest-even, using r as the sticky bit.
    let nbits = 128 - q.leading_zeros() as i32;
    let drop = nbits - 24;
    debug_assert!(drop > 0);
    let mut hi = (q >> drop) as u64;
    let low = q & ((1u128 << drop) - 1);
    let half = 1u128 << (drop - 1);
    let mut out_e = half_e + drop;
    if low > half || (low == half && (r != 0 || (hi & 1) == 1)) {
        hi += 1;
        if hi == (1u64 << 24) {
            hi >>= 1;
            out_e += 1;
        }
    }

    // Reassemble. sqrt of any finite positive binary32 is in the normal
    // range (the smallest subnormal, 2^-149, has root 2^-74.5), so no
    // subnormal or overflow path is reachable.
    let biased = out_e + 23 + 127;
    debug_assert!(biased > 0 && biased < 0xff);
    f32::from_bits(((biased as u32) << 23) | ((hi as u32) & 0x007f_ffff))
}

/// Integer square root: the largest `q` with `q*q <= n`.
fn isqrt_u128(n: u128) -> u128 {
    if n == 0 {
        return 0;
    }
    // Bit-by-bit restoring square root: exact, no floating point anywhere.
    let mut rem: u128 = 0;
    let mut root: u128 = 0;
    let shift = (127 - n.leading_zeros()) & !1; // highest even bit index
    let mut i = shift as i32;
    while i >= 0 {
        rem = (rem << 2) | ((n >> i) & 3);
        root <<= 1;
        let trial = (root << 1) | 1;
        if rem >= trial {
            rem -= trial;
            root |= 1;
        }
        i -= 2;
    }
    root
}

/// `floor(x)` for f32, by bit manipulation (spec 6.2 step 4 wants the
/// value truncated toward negative infinity).
pub fn floor_f32(x: f32) -> f32 {
    let bits = x.to_bits();
    let exp = (((bits >> 23) & 0xff) as i32) - 127;
    if exp >= 23 {
        return x; // already integral (or inf/NaN)
    }
    if exp < 0 {
        // |x| < 1
        return if bits >> 31 == 1 {
            if bits & 0x7fff_ffff == 0 {
                x // -0.0
            } else {
                -1.0
            }
        } else {
            0.0
        };
    }
    let mask = (1u32 << (23 - exp)) - 1;
    if bits & mask == 0 {
        return x;
    }
    let truncated = f32::from_bits(bits & !mask);
    if bits >> 31 == 1 {
        truncated - 1.0
    } else {
        truncated
    }
}

/// `f64::round()` semantics: round to the nearest integer, ties away from
/// zero. Spec 6.3's reduction names this exact function.
pub fn round_ties_away_f64(x: f64) -> f64 {
    let bits = x.to_bits();
    let exp = (((bits >> 52) & 0x7ff) as i32) - 1023;
    if exp >= 52 {
        return x; // already integral (or inf/NaN)
    }
    let sign = if bits >> 63 == 1 { -1.0f64 } else { 1.0f64 };
    if exp < 0 {
        // |x| < 1: ties away from zero means 0.5 rounds to 1.
        let mag = f64::from_bits(bits & 0x7fff_ffff_ffff_ffff);
        return if mag >= 0.5 { sign } else { sign * 0.0 };
    }
    let mask = (1u64 << (52 - exp)) - 1;
    let frac = bits & mask;
    if frac == 0 {
        return x;
    }
    let truncated = f64::from_bits(bits & !mask); // toward zero, sign kept
    let half = 1u64 << (51 - exp);
    if frac >= half {
        truncated + sign // exact: |truncated| < 2^52
    } else {
        truncated
    }
}

/// `|x|` for f64 without `std`.
pub fn abs_f64(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & 0x7fff_ffff_ffff_ffff)
}

/// `|x|` for f32 without `std`.
pub fn abs_f32(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7fff_ffff)
}
