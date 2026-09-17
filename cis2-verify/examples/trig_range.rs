//! Spec 14.4 range sweep: how far out does §6.3's Cody-Waite reduction hold?
//!
//! §14.4 records that the trig polynomials are "only validated for |x| ≲ 14"
//! because the unit tests sweep `x = i * 0.7`, `i` in `-20..=20`, and warns
//! that "a clean-room implementer targeting a longer sequence than this spec's
//! 20 positions should not assume this polynomial's accuracy holds unchanged".
//! RoPE's largest angle is `pos * inv_freq[0]`, and `inv_freq[0] == 1.0`, so
//! the angle a decode reaches is simply its sequence length.
//!
//! This sweeps `sin_pinned`/`cos_pinned` binade by binade against a f64
//! reference and prints the worst ULP error in each.
//!
//! SUPERSEDED by `examples/trig_exhaustive.rs`, and kept only because the way
//! it is wrong is worth keeping. 4096 samples per binade reported max|ulp| = 1
//! through 2^16 and suggested the reduction was near-perfect out to a 64k
//! context. The exhaustive sweep of the same range found max|ulp| sin = 66 and
//! cos = 2283: the sampler simply never landed near a zero of the function,
//! where a fixed absolute error is a large number of ULP. The conclusion the
//! exhaustive run supports is stronger than the sampled one, but it is a
//! different conclusion, stated in absolute error rather than ULP. Do not
//! quote this file's numbers; see E26.
//!
//!     cargo run --release --example trig_range
//!
//! Reference: `(x as f64).sin()` rounded to f32. The widening cast is exact
//! and the host's f64 `sin` carries a full Payne-Hanek reduction, so its error
//! is ~1e-16 relative -- some nine orders of magnitude below an f32 ULP. It is
//! an accuracy oracle only; nothing pinned depends on it.

use cis2_verify::{fpenv, mathpin};

/// Signed ULP distance between two finite f32s, using the standard
/// monotonic bits-to-integer mapping.
fn ulp_diff(a: f32, b: f32) -> i64 {
    let key = |v: f32| -> i64 {
        let b = v.to_bits() as i64;
        if v.to_bits() & 0x8000_0000 != 0 { 0x8000_0000i64 - b } else { b }
    };
    key(a) - key(b)
}

fn main() {
    fpenv::pin_and_selftest().expect("1.3 pin");
    println!("TRIG fp-env=pinned control={}", fpenv::control_name());
    println!("TRIG oracle=f64 sin/cos rounded to f32 (accuracy oracle, not a pin)");
    println!("TRIG binade  samples  max|ulp|sin  max|ulp|cos  max|abs|sin  worst-x(sin)");

    // 4096 samples per binade, spread by an odd stride over the binade's
    // mantissa space so the sample set is not a lattice aligned to pi/2.
    const N: u32 = 4096;
    let mut overall_sin = 0i64;
    let mut overall_cos = 0i64;

    for e in 0..31i32 {
        let lo = (2.0f32).powi(e);
        let hi = (2.0f32).powi(e + 1);
        let lo_b = lo.to_bits();
        let hi_b = hi.to_bits();
        let span = (hi_b - lo_b) as u64;
        let (mut ws, mut wc, mut wx) = (0i64, 0i64, lo);
        let mut wabs = 0.0f64;
        for i in 0..N {
            // 2654435761 is Knuth's 32-bit golden-ratio multiplier; it is used
            // here only to scatter the sample points, and nothing depends on it.
            let off = ((i as u64).wrapping_mul(2_654_435_761) % span) as u32;
            let x = f32::from_bits(lo_b + off);
            let sref = (x as f64).sin();
            let s = ulp_diff(mathpin::sin_pinned(x), sref as f32).abs();
            let c = ulp_diff(mathpin::cos_pinned(x), (x as f64).cos() as f32).abs();
            let a = (mathpin::sin_pinned(x) as f64 - sref).abs();
            if s > ws { ws = s; wx = x; }
            if c > wc { wc = c; }
            if a > wabs { wabs = a; }
        }
        if ws > overall_sin { overall_sin = ws; }
        if wc > overall_cos { overall_cos = wc; }
        println!("TRIG [2^{e:<2}, 2^{:<2})  {N}  {ws:>9}  {wc:>9}  {wabs:>11.3e}  {wx:e}", e + 1);
    }
    println!("TRIG overall max|ulp| sin={overall_sin} cos={overall_cos}");
}
