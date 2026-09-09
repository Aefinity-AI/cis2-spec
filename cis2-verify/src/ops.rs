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
        acc = acc + p;
    }
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
    }
    let ss = sum_seq(&sq);
    let mean = ss / (n as f32);
    let inv = rsqrt(mean + eps);
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let scaled = x[i] * inv;
        out[i] = scaled * weight[i];
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

    static ELEMENTS: AtomicU64 = AtomicU64::new(0);
    static DIVERGENT: AtomicU64 = AtomicU64::new(0);
    static ALT: AtomicBool = AtomicBool::new(false);

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
    }
    /// `(elements, divergent)`
    pub fn counts() -> (u64, u64) {
        (ELEMENTS.load(Relaxed), DIVERGENT.load(Relaxed))
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
        *v = exp_pinned(*v - max_v);
    }
    let denom = sum_seq(scores);
    for v in scores.iter_mut() {
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
