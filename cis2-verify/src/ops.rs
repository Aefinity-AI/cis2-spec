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
    }
    out
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
