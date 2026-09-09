//! Spec 14.4, settled exhaustively: every f32 angle a long-context RoPE can
//! produce, checked one bit pattern at a time.
//!
//! §14.4 says the trig polynomials are "only validated for `|x| ≲ 14`" and
//! warns that "a clean-room implementer targeting a longer sequence than this
//! spec's 20 positions should not assume this polynomial's accuracy holds
//! unchanged". `examples/trig_range.rs` sampled binades and suggested the
//! caution is far too conservative. Sampling cannot license a MUST, so this
//! example replaces the sample with the whole set:
//!
//!   A. odd/even symmetry, exhaustively, so a positive-only sweep is a
//!      statement about every finite f32 and not only half of them;
//!   B. every f32 bit pattern `x` with `0 <= x < 2^16`, i.e. all 1,199,570,944
//!      of them, subnormals included, against a f64 oracle;
//!   C. the exact finite multiset of RoPE angles `(pos as f32) * inv_freq[i]`
//!      that each pinned model produces for a 131,072-position context.
//!
//! (B) bounds (C) for any context up to 65,536 positions, because RoPE's
//! largest angle is `pos * inv_freq[0]` and §7.1 makes `inv_freq[0]` exactly
//! 1.0 (`exp_pinned(-0.0)`), so the largest angle a decode reaches *is* its
//! sequence length. (C) is run anyway: it is the claim a reader actually
//! wants, and it is cheap.
//!
//!     cargo run --release --example trig_exhaustive
//!
//! Oracle: `(x as f64).sin()` rounded to f32. The widening cast is exact and
//! the host libm's f64 `sin` carries a full Payne-Hanek reduction, so its own
//! error is ~1e-16 relative -- some nine orders of magnitude below an f32 ULP.
//! It is an accuracy oracle only; nothing pinned depends on it, and a
//! disagreement here is a claim about accuracy, never about conformance.

use cis2_verify::config::Config;
use cis2_verify::model::build_inv_freq;
use cis2_verify::{fpenv, mathpin};
use std::thread;

/// Signed ULP distance between two finite f32s, using the standard monotonic
/// bits-to-integer mapping (so the count crosses zero and binade boundaries
/// correctly).
fn ulp_diff(a: f32, b: f32) -> i64 {
    let key = |v: f32| -> i64 {
        let b = v.to_bits() as i64;
        if v.to_bits() & 0x8000_0000 != 0 { 0x8000_0000i64 - b } else { b }
    };
    key(a) - key(b)
}

/// Worst case seen so far, and the argument that produced it.
#[derive(Clone, Copy)]
struct Worst {
    ulp: i64,
    at: f32,
    /// Worst *absolute* error, tracked separately. For a range reduction this
    /// is the figure of merit: near a zero of sin or cos the result itself is
    /// tiny, so a fixed absolute error is an unbounded number of ULP, and the
    /// ULP column alone would say the reduction falls apart when it has not.
    abs: f64,
    abs_at: f32,
}

impl Worst {
    const ZERO: Worst = Worst { ulp: 0, at: 0.0, abs: 0.0, abs_at: 0.0 };
    fn note(&mut self, ulp: i64, abs: f64, at: f32) {
        if ulp > self.ulp {
            self.ulp = ulp;
            self.at = at;
        }
        if abs > self.abs {
            self.abs = abs;
            self.abs_at = at;
        }
    }
    fn merge(&mut self, other: Worst) {
        if other.ulp > self.ulp { self.ulp = other.ulp; self.at = other.at; }
        if other.abs > self.abs { self.abs = other.abs; self.abs_at = other.abs_at; }
    }
}

/// Exponent field of an f32, 0..=254 for finite values. Used only to bucket
/// the exhaustive results into a readable table.
fn expfield(bits: u32) -> usize {
    ((bits >> 23) & 0xFF) as usize
}

/// Per-exponent-field accumulators for one worker's slice of the bit range.
struct Slice {
    sin: Vec<Worst>,
    cos: Vec<Worst>,
    /// Symmetry failures with |x| >= 2^-126 where the two sides differ in
    /// value. This one must stay 0.
    asym_normal: u64,
    /// Symmetry failures with |x| >= 2^-126 where both sides are zero and
    /// only the sign of that zero differs -- the reduced argument underflowed
    /// to +-0, so the polynomial returns a signed zero whose sign does not
    /// mirror. Recorded separately because "the value is wrong" and "the zero
    /// is signed the other way" are different claims.
    asym_zero_sign: u64,
    first_zero_sign: [f32; 4],
    /// Symmetry failures inside that class, counted separately because they
    /// are a documented consequence of the pinned FTZ/DAZ environment and not
    /// a property of the polynomial.
    asym_subnormal: u64,
    checked: u64,
}

fn sweep(lo_bits: u32, hi_bits: u32) -> Slice {
    // MXCSR / FPCR is per-thread state: a worker that does not pin is not
    // running the spec's arithmetic, so this is not optional here.
    fpenv::pin_and_selftest().expect("1.3 pin (worker)");
    let mut s = Slice {
        sin: vec![Worst::ZERO; 256],
        cos: vec![Worst::ZERO; 256],
        asym_normal: 0,
        asym_zero_sign: 0,
        first_zero_sign: [0.0; 4],
        asym_subnormal: 0,
        checked: 0,
    };
    for bits in lo_bits..hi_bits {
        let x = f32::from_bits(bits);
        let e = expfield(bits);

        let sp = mathpin::sin_pinned(x);
        let cp = mathpin::cos_pinned(x);
        let xd = x as f64;
        let (sref, cref) = (xd.sin(), xd.cos());
        s.sin[e].note(ulp_diff(sp, sref as f32).abs(), (sp as f64 - sref).abs(), x);
        s.cos[e].note(ulp_diff(cp, cref as f32).abs(), (cp as f64 - cref).abs(), x);

        // (A) Odd/even symmetry, on the same pass. `-x` here is the exact
        // sign-bit flip, and the comparison is on bits, not on value, so
        // `sin_pinned(-0.0) == -0.0` is required rather than merely `== 0.0`.
        let nx = f32::from_bits(bits ^ 0x8000_0000);
        let sn = mathpin::sin_pinned(nx);
        let cn = mathpin::cos_pinned(nx);
        let bad_sin = sn.to_bits() != (-sp).to_bits();
        let bad_cos = cn.to_bits() != cp.to_bits();
        if bad_sin || bad_cos {
            if e == 0 {
                s.asym_subnormal += 1;
            } else if (!bad_sin || (sn == 0.0 && sp == 0.0))
                && (!bad_cos || (cn == 0.0 && cp == 0.0))
            {
                // Both sides agree in value; only a zero's sign differs.
                if (s.asym_zero_sign as usize) < s.first_zero_sign.len() {
                    s.first_zero_sign[s.asym_zero_sign as usize] = x;
                }
                s.asym_zero_sign += 1;
            } else {
                s.asym_normal += 1;
            }
        }
        s.checked += 1;
    }
    s
}

fn main() {
    fpenv::pin_and_selftest().expect("1.3 pin");
    println!("TRIG-X fp-env=pinned control={}", fpenv::control_name());
    println!("TRIG-X oracle=f64 sin/cos rounded to f32 (accuracy oracle, not a pin)");

    // ---- (A)+(B) exhaustive over every f32 in [0, 2^E) ------------------
    // E defaults to 16 (65,536 -- a 64k context) and can be raised from the
    // command line; the cost is linear in 2^E.
    let e_hi: i32 = std::env::args()
        .nth(1)
        .map(|a| a.parse().expect("usage: trig_exhaustive [exponent]"))
        .unwrap_or(16);
    let hi_bits = (2.0f32).powi(e_hi).to_bits();
    let workers = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let step = hi_bits / workers as u32 + 1;
    let mut sin = vec![Worst::ZERO; 256];
    let mut cos = vec![Worst::ZERO; 256];
    let (mut asym_n, mut asym_s, mut checked) = (0u64, 0u64, 0u64);
    let mut asym_z = 0u64;
    let mut zero_sign_at: Vec<f32> = Vec::new();

    let parts: Vec<Slice> = thread::scope(|sc| {
        let mut hs = Vec::new();
        let mut lo = 0u32;
        while lo < hi_bits {
            let hi = (lo + step).min(hi_bits);
            hs.push(sc.spawn(move || sweep(lo, hi)));
            lo = hi;
        }
        hs.into_iter().map(|h| h.join().expect("worker")).collect()
    });
    for p in parts {
        for e in 0..256 {
            sin[e].merge(p.sin[e]);
            cos[e].merge(p.cos[e]);
        }
        asym_n += p.asym_normal;
        asym_s += p.asym_subnormal;
        for i in 0..(p.asym_zero_sign as usize).min(p.first_zero_sign.len()) {
            zero_sign_at.push(p.first_zero_sign[i]);
        }
        asym_z += p.asym_zero_sign;
        checked += p.checked;
    }

    println!("TRIG-X exhaustive over every f32 bit pattern in [0, 2^{e_hi}): checked={checked}");
    println!("TRIG-X symmetry, |x| >= 2^-126: value-differs={asym_n} (sin odd and cos even, in value)");
    println!(
        "TRIG-X symmetry, |x| >= 2^-126: zero-sign-only={asym_z} at {:?} \
         (reduced argument underflowed to +-0)",
        zero_sign_at
    );
    println!(
        "TRIG-X symmetry, zero/subnormal: violations={asym_s} (= 2^23; \
         1.3 DAZ+FTZ returns +0.0 for sin, so signed zero is not carried)"
    );
    println!("TRIG-X range                     max|ulp|sin  max|ulp|cos    max|abs|sin    max|abs|cos");

    // Everything below 2^-14 is one row: the argument reduction is the
    // identity there (k == 0) and the polynomial is dominated by its linear
    // term, so a per-binade table would be 112 identical lines.
    let small_hi = 127 - 14; // exponent field of 2^-14
    let (mut ss, mut sc_) = (Worst::ZERO, Worst::ZERO);
    for e in 0..small_hi {
        ss.merge(sin[e]);
        sc_.merge(cos[e]);
    }
    println!(
        "TRIG-X [0, 2^-14)                {:>11}  {:>11}    {:>11.4e}    {:>11.4e}",
        ss.ulp, sc_.ulp, ss.abs, sc_.abs
    );
    for e in small_hi..(127 + e_hi as usize) {
        let p = e as i32 - 127;
        println!(
            "TRIG-X [2^{p:<3}, 2^{:<3})            {:>11}  {:>11}    {:>11.4e}    {:>11.4e}",
            p + 1,
            sin[e].ulp,
            cos[e].ulp,
            sin[e].abs,
            cos[e].abs
        );
    }
    let all_sin = sin.iter().fold(Worst::ZERO, |mut a, w| { a.merge(*w); a });
    let all_cos = cos.iter().fold(Worst::ZERO, |mut a, w| { a.merge(*w); a });
    println!(
        "TRIG-X EXHAUSTIVE |x|<2^{e_hi} max|ulp| sin={} (at {:e})  cos={} (at {:e})",
        all_sin.ulp, all_sin.at, all_cos.ulp, all_cos.at
    );
    println!(
        "TRIG-X EXHAUSTIVE |x|<2^{e_hi} max|abs| sin={:.4e} (at {:e})  cos={:.4e} (at {:e})",
        all_sin.abs, all_sin.abs_at, all_cos.abs, all_cos.abs_at
    );

    // ---- (C) the exact RoPE angle sets --------------------------------
    // Both pinned models of §0, at a context far longer than either ships
    // with. `head_dim` and `rope_theta` are the only fields §7.1 reads;
    // the rest are filled in from the real configs so nothing here is a
    // stand-in for a field that matters.
    const L: usize = 131_072;
    for (name, hidden, heads, theta) in [
        ("SmolLM2-135M", 576usize, 9usize, 100000.0f32),
        ("Qwen2.5-0.5B", 896usize, 14usize, 1000000.0f32),
    ] {
        let cfg = Config {
            hidden_size: hidden,
            intermediate_size: 0,
            num_hidden_layers: 0,
            num_attention_heads: heads,
            num_key_value_heads: heads,
            vocab_size: 0,
            rms_norm_eps: 0.0,
            rope_theta: theta,
            tie_word_embeddings: true,
            rope_interleaved: false,
            torch_dtype: String::from("float32"),
            hidden_act: String::from("silu"),
        };
        let inv_freq = build_inv_freq(&cfg);
        let mut w_sin = Worst::ZERO;
        let mut w_cos = Worst::ZERO;
        let mut max_angle = 0.0f32;
        for pos in 0..L {
            for f in &inv_freq {
                let angle = (pos as f32) * *f;
                if angle > max_angle {
                    max_angle = angle;
                }
                let ad = angle as f64;
                let (sref, cref) = (ad.sin(), ad.cos());
                let sp = mathpin::sin_pinned(angle);
                let cp = mathpin::cos_pinned(angle);
                w_sin.note(ulp_diff(sp, sref as f32).abs(), (sp as f64 - sref).abs(), angle);
                w_cos.note(ulp_diff(cp, cref as f32).abs(), (cp as f64 - cref).abs(), angle);
            }
        }
        println!(
            "TRIG-X ROPE {name} head_dim={} theta={theta:e} L={L} angles={} \
             inv_freq[0]={} max_angle={max_angle:e} \
             max|ulp| sin={} cos={} max|abs| sin={:.4e} cos={:.4e}",
            cfg.head_dim(),
            L * inv_freq.len(),
            inv_freq[0],
            w_sin.ulp,
            w_cos.ulp,
            w_sin.abs,
            w_cos.abs
        );
    }
}
