//! Spec 14.1, settled exhaustively: every f32 `exp_pinned`, `ln_pinned` and
//! `silu_pinned` can be handed, checked one bit pattern at a time.
//!
//! §14.1 says `ln_pinned`'s "validated domain is a finite, explicitly-tested
//! set of `x` values (§6.5), not a general accuracy proof", and §0's
//! non-claims list says nothing is claimed for `exp` or `ln` outside the
//! input ranges the checked decodes happen to exercise. E26 showed what that
//! kind of clause is worth: for a single-argument f32 function the domain is
//! finite and enumerable, so a sampled statement should never stand where an
//! exhaustive one is available.
//!
//! It is also the wrong function that §14.1 worries about. `ln_pinned` is
//! called exactly once per model load (`model.rs`, on `cfg.rope_theta`).
//! `exp_pinned` is on the decode hot path twice over:
//!
//!   * `ops.rs`'s softmax evaluates `exp_pinned(v - max_v)` on **every
//!     attention score at every position**, so its argument is always `<= 0`;
//!   * `mathpin.rs`'s `silu_pinned(x)` evaluates `exp_pinned(-x)` on **every
//!     FFN intermediate**, so its argument is an arbitrary finite f32 of
//!     either sign.
//!
//! So this sweep covers, with no domain caveat left over:
//!
//!   A. `exp_pinned` on all 2^32 bit patterns, against an f64 oracle, with
//!      the two guard regions and the FTZ-flush region counted separately
//!      from the polynomial region;
//!   B. the softmax subdomain `x <= 0` reported on its own, since that is the
//!      only half `ops.rs` can reach;
//!   C. `ln_pinned` on all 2^32 bit patterns -- every positive f32 against an
//!      f64 oracle, and the negative/zero/NaN domain guard checked for the
//!      value §6.5 requires rather than assumed;
//!   D. `silu_pinned` on all 2^32 bit patterns, because §6.4 is what actually
//!      consumes `exp_pinned`'s positive half and a bound on `exp` is not by
//!      itself a bound on `x / (1 + exp(-x))`;
//!   E. monotonicity of both `exp_pinned` and `ln_pinned` across the whole
//!      finite domain -- an exactness property that needs no oracle at all,
//!      and the one a table-driven or range-reduced reimplementation is most
//!      likely to break;
//!   F. the two exactness facts §7.1 silently depends on:
//!      `exp_pinned(-0.0) == 1.0` bit-exactly (this is why `inv_freq[0]` is
//!      exactly 1.0, which is in turn why E26's "largest angle is the
//!      sequence length" argument holds) and `ln_pinned(1.0) == +0.0`.
//!
//!     cargo run --release --example exp_ln_exhaustive
//!
//! Oracle: `(x as f64).exp()` / `.ln()` narrowed to f32, with the narrowing
//! done under the §1.3 environment so a result in the subnormal range is
//! flushed the way §1.3 flushes it. The widening cast is exact and the host
//! libm's f64 `exp`/`ln` carry ~1e-16 relative error, some nine orders of
//! magnitude below an f32 ULP. It is an accuracy oracle only; nothing pinned
//! depends on it, and a disagreement here is a claim about accuracy, never
//! about conformance.

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
    /// Worst *relative* error, tracked separately. For `exp` this is the
    /// figure of merit: the result spans 2^-149 to 2^128, so an absolute
    /// error column would say nothing, while for `ln` near `x = 1` the result
    /// itself is tiny and the ULP column alone falsely reports collapse. Both
    /// are kept for both functions and the reader is told which one to read.
    rel: f64,
    rel_at: f32,
    /// Worst *absolute* error. The figure of merit for `ln` near its zero and
    /// for `silu` everywhere.
    abs: f64,
    abs_at: f32,
    n: u64,
}

impl Worst {
    const ZERO: Worst =
        Worst { ulp: 0, at: 0.0, rel: 0.0, rel_at: 0.0, abs: 0.0, abs_at: 0.0, n: 0 };
    fn note(&mut self, ulp: i64, rel: f64, abs: f64, at: f32) {
        if ulp > self.ulp {
            self.ulp = ulp;
            self.at = at;
        }
        if rel > self.rel {
            self.rel = rel;
            self.rel_at = at;
        }
        if abs > self.abs {
            self.abs = abs;
            self.abs_at = at;
        }
        self.n += 1;
    }
    fn merge(&mut self, o: Worst) {
        if o.ulp > self.ulp { self.ulp = o.ulp; self.at = o.at; }
        if o.rel > self.rel { self.rel = o.rel; self.rel_at = o.rel_at; }
        if o.abs > self.abs { self.abs = o.abs; self.abs_at = o.abs_at; }
        self.n += o.n;
    }
}

/// Exponent field of an f32, 0..=255. Used only to bucket the exhaustive
/// results into a readable table.
fn expfield(bits: u32) -> usize {
    ((bits >> 23) & 0xFF) as usize
}

/// §1.3 flush-to-zero applied to an oracle value that has been narrowed to
/// f32. The pinned environment cannot produce a subnormal result, so an
/// oracle that produces one is compared against the zero §1.3 would deliver.
fn ftz(v: f32) -> f32 {
    // Written on bits, not on comparisons: under §1.3's DAZ a subnormal
    // operand compares equal to zero, so `v != 0.0` would be false here and
    // the flush would silently not happen.
    let b = v.to_bits();
    if b & 0x7F80_0000 == 0 && b & 0x007F_FFFF != 0 { f32::from_bits(b & 0x8000_0000) } else { v }
}

/// Per-exponent-field accumulators for one worker's slice of the bit range.
/// Index 0..256 is the non-negative half, 256..512 the negative half, so a
/// single array carries both signs and the softmax subdomain is just the
/// second half.
struct Slice {
    exp: Vec<Worst>,
    ln: Vec<Worst>,
    silu: Vec<Worst>,
    /// `silu_pinned` on the arguments where §6.2's `x > 88.0` clip fires
    /// inside §6.4, i.e. `x < -88.0`. Kept out of the per-binade table so the
    /// table states a bound on the polynomial rather than a bound on the
    /// guard, and reported on its own line instead.
    silu_clip: Worst,

    /// `exp_pinned` returned +Inf where the true value is finite and
    /// representable in f32: the `x > 88.0` guard clipping below
    /// `ln(f32::MAX) = 88.7228...`. Deliberate and pinned; counted, with its
    /// exact extent, rather than hidden.
    exp_clip_hi: u64,
    exp_clip_lo_at: f32,
    exp_clip_hi_at: f32,
    /// `exp_pinned` returned 0.0 where the FTZ-narrowed oracle is nonzero.
    /// Must stay 0: `exp(-88) < f32::MIN_POSITIVE`, so the `x < -88.0` guard
    /// is inside the region §1.3 flushes anyway.
    exp_flush_bad: u64,
    exp_flush_bad_at: f32,
    /// `exp_pinned` on a NaN input returned something that is not a NaN.
    exp_nan_bad: u64,

    /// Monotonicity violations, counted over finite inputs in bit order.
    exp_nonmono: u64,
    exp_nonmono_at: f32,
    ln_nonmono: u64,
    ln_nonmono_at: f32,

    /// `ln_pinned` on the guard domain returned something other than what
    /// §6.5 requires (NaN for NaN and for `x < 0`, `-Inf` for zero, and --
    /// under §1.3's DAZ, which makes a subnormal compare equal to zero --
    /// `-Inf` for a subnormal too).
    ln_guard_bad: u64,
    ln_guard_bad_at: f32,
    ln_subnormal_neg_inf: u64,

    checked: u64,
}

impl Slice {
    fn new() -> Slice {
        Slice {
            exp: vec![Worst::ZERO; 512],
            ln: vec![Worst::ZERO; 512],
            silu: vec![Worst::ZERO; 512],
            silu_clip: Worst::ZERO,
            exp_clip_hi: 0,
            exp_clip_lo_at: f32::INFINITY,
            exp_clip_hi_at: f32::NEG_INFINITY,
            exp_flush_bad: 0,
            exp_flush_bad_at: 0.0,
            exp_nan_bad: 0,
            exp_nonmono: 0,
            exp_nonmono_at: 0.0,
            ln_nonmono: 0,
            ln_nonmono_at: 0.0,
            ln_guard_bad: 0,
            ln_guard_bad_at: 0.0,
            ln_subnormal_neg_inf: 0,
            checked: 0,
        }
    }
    fn merge(&mut self, o: &Slice) {
        for i in 0..512 {
            self.exp[i].merge(o.exp[i]);
            self.ln[i].merge(o.ln[i]);
            self.silu[i].merge(o.silu[i]);
        }
        self.silu_clip.merge(o.silu_clip);
        self.exp_clip_hi += o.exp_clip_hi;
        if o.exp_clip_lo_at < self.exp_clip_lo_at { self.exp_clip_lo_at = o.exp_clip_lo_at; }
        if o.exp_clip_hi_at > self.exp_clip_hi_at { self.exp_clip_hi_at = o.exp_clip_hi_at; }
        if o.exp_flush_bad > 0 && self.exp_flush_bad == 0 {
            self.exp_flush_bad_at = o.exp_flush_bad_at;
        }
        self.exp_flush_bad += o.exp_flush_bad;
        self.exp_nan_bad += o.exp_nan_bad;
        if o.exp_nonmono > 0 && self.exp_nonmono == 0 { self.exp_nonmono_at = o.exp_nonmono_at; }
        self.exp_nonmono += o.exp_nonmono;
        if o.ln_nonmono > 0 && self.ln_nonmono == 0 { self.ln_nonmono_at = o.ln_nonmono_at; }
        self.ln_nonmono += o.ln_nonmono;
        if o.ln_guard_bad > 0 && self.ln_guard_bad == 0 { self.ln_guard_bad_at = o.ln_guard_bad_at; }
        self.ln_guard_bad += o.ln_guard_bad;
        self.ln_subnormal_neg_inf += o.ln_subnormal_neg_inf;
        self.checked += o.checked;
    }
}

/// Bucket index: exponent field, offset by 256 for negative arguments.
fn bucket(bits: u32) -> usize {
    expfield(bits) + if bits & 0x8000_0000 != 0 { 256 } else { 0 }
}

/// Compare one pinned result against one oracle result, tolerating the
/// infinities and NaNs that neither ULP nor relative error is defined on.
/// Returns `None` when the pair is a non-finite agreement (nothing to
/// record) and `Some(..)` when both sides are finite.
fn compare(pinned: f32, oracle: f32) -> Option<(i64, f64, f64)> {
    if !pinned.is_finite() || !oracle.is_finite() {
        return None;
    }
    let d = (pinned as f64) - (oracle as f64);
    let rel = if oracle == 0.0 { 0.0 } else { (d / oracle as f64).abs() };
    Some((ulp_diff(pinned, oracle).abs(), rel, d.abs()))
}

fn sweep(lo_bits: u32, hi_bits: u32) -> Slice {
    // MXCSR / FPCR is per-thread state: a worker that does not pin is not
    // running the spec's arithmetic, so this is not optional here.
    fpenv::pin_and_selftest().expect("1.3 pin (worker)");
    let mut s = Slice::new();

    // Seed the monotonicity chain from the pattern *before* this slice, so
    // the slice boundaries are checked too and not silently skipped.
    let seed = |b: u32| -> (Option<f32>, Option<f32>) {
        let x = f32::from_bits(b);
        if !x.is_finite() {
            return (None, None);
        }
        (Some(mathpin::exp_pinned(x)), if x > 0.0 { Some(mathpin::ln_pinned(x)) } else { None })
    };
    let (mut prev_exp, mut prev_ln) = if lo_bits > 0 { seed(lo_bits - 1) } else { (None, None) };

    let mut bits = lo_bits;
    loop {
        let x = f32::from_bits(bits);
        let b = bucket(bits);
        let xd = x as f64;

        // ---- exp ------------------------------------------------------
        let ep = mathpin::exp_pinned(x);
        if x.is_nan() {
            if !ep.is_nan() {
                s.exp_nan_bad += 1;
            }
        } else {
            let eo = ftz(xd.exp() as f32);
            match compare(ep, eo) {
                Some((u, r, a)) => s.exp[b].note(u, r, a, x),
                None => {
                    if ep.is_infinite() && eo.is_finite() {
                        s.exp_clip_hi += 1;
                        if x < s.exp_clip_lo_at { s.exp_clip_lo_at = x; }
                        if x > s.exp_clip_hi_at { s.exp_clip_hi_at = x; }
                    }
                }
            }
            if ep == 0.0 && eo != 0.0 {
                if s.exp_flush_bad == 0 { s.exp_flush_bad_at = x; }
                s.exp_flush_bad += 1;
            }
            // Monotonicity, in bit order. The positive half ascends in value
            // as the bits ascend; the negative half descends.
            if x.is_finite() {
                if let Some(p) = prev_exp {
                    let ok = if bits & 0x8000_0000 == 0 { ep >= p } else { ep <= p };
                    if !ok {
                        if s.exp_nonmono == 0 { s.exp_nonmono_at = x; }
                        s.exp_nonmono += 1;
                    }
                }
                prev_exp = Some(ep);
            } else {
                prev_exp = None;
            }
        }

        // ---- ln -------------------------------------------------------
        let lp = mathpin::ln_pinned(x);
        let subnormal = expfield(bits) == 0 && (bits & 0x007F_FFFF) != 0;
        if x.is_nan() || (bits & 0x8000_0000) != 0 {
            // §6.5's guard: NaN, and every negative including -0.0 (which
            // compares `< 0.0` false, so -0.0 takes the `x == 0.0` branch).
            let want_neg_inf = bits == 0x8000_0000;
            let ok = if x.is_nan() {
                lp.is_nan()
            } else if want_neg_inf || subnormal {
                lp == f32::NEG_INFINITY
            } else {
                lp.is_nan()
            };
            if !ok {
                if s.ln_guard_bad == 0 { s.ln_guard_bad_at = x; }
                s.ln_guard_bad += 1;
            }
            if subnormal && lp == f32::NEG_INFINITY {
                s.ln_subnormal_neg_inf += 1;
            }
            prev_ln = None;
        } else if bits == 0 {
            if lp != f32::NEG_INFINITY {
                if s.ln_guard_bad == 0 { s.ln_guard_bad_at = x; }
                s.ln_guard_bad += 1;
            }
            prev_ln = Some(f32::NEG_INFINITY);
        } else if subnormal {
            // §1.3's DAZ makes `x == 0.0` true for a subnormal, so §6.5's
            // zero branch is taken. Recorded, not assumed.
            if lp == f32::NEG_INFINITY {
                s.ln_subnormal_neg_inf += 1;
            } else {
                if s.ln_guard_bad == 0 { s.ln_guard_bad_at = x; }
                s.ln_guard_bad += 1;
            }
        } else if x.is_finite() {
            let lo = xd.ln() as f32;
            if let Some((u, r, a)) = compare(lp, lo) {
                s.ln[b].note(u, r, a, x);
            }
            if let Some(p) = prev_ln {
                if lp < p {
                    if s.ln_nonmono == 0 { s.ln_nonmono_at = x; }
                    s.ln_nonmono += 1;
                }
            }
            prev_ln = Some(lp);
        } else {
            prev_ln = None;
        }

        // ---- silu -----------------------------------------------------
        if !x.is_nan() {
            let sp = mathpin::silu_pinned(x);
            let so = ftz((xd / (1.0 + (-xd).exp())) as f32);
            if let Some((u, r, a)) = compare(sp, so) {
                if x < -88.0 {
                    s.silu_clip.note(u, r, a, x);
                } else {
                    s.silu[b].note(u, r, a, x);
                }
            }
        }

        s.checked += 1;
        if bits == hi_bits - 1 {
            break;
        }
        bits += 1;
    }
    s
}

fn row(label: &str, w: Worst) {
    if w.n == 0 {
        println!("EXPLN-X {label:<26} n=0            (no finite pair to compare: guard region)");
        return;
    }
    println!(
        "EXPLN-X {label:<26} n={:<12} max|ulp|={:<9} @0x{:08X} max|rel|={:<11.4e} \
         max|abs|={:<11.4e} @0x{:08X}",
        w.n,
        w.ulp,
        w.at.to_bits(),
        w.rel,
        w.abs,
        w.abs_at.to_bits()
    );
}

fn fold(src: &[Worst], range: std::ops::Range<usize>) -> Worst {
    let mut a = Worst::ZERO;
    for i in range {
        a.merge(src[i]);
    }
    a
}

fn main() {
    fpenv::pin_and_selftest().expect("1.3 pin");
    println!("EXPLN-X fp-env=pinned control={}", fpenv::control_name());
    println!("EXPLN-X oracle=f64 exp/ln narrowed to f32 then 1.3-flushed (accuracy oracle, not a pin)");

    // ---- (F) the two exactness facts, checked before anything else ------
    // `inv_freq[0] = exp_pinned(-0.0 * ln_pinned(theta))` must be exactly
    // 1.0, or E26's "the largest RoPE angle is the sequence length" argument
    // does not hold. It is asserted here rather than assumed.
    let e_neg_zero = mathpin::exp_pinned(-0.0f32);
    let e_pos_zero = mathpin::exp_pinned(0.0f32);
    let ln_one = mathpin::ln_pinned(1.0f32);
    println!(
        "EXPLN-X exact exp(-0.0)=0x{:08X} exp(+0.0)=0x{:08X} (both must be 0x3F800000) \
         ln(1.0)=0x{:08X} (must be 0x00000000)",
        e_neg_zero.to_bits(),
        e_pos_zero.to_bits(),
        ln_one.to_bits()
    );
    assert_eq!(e_neg_zero.to_bits(), 0x3F80_0000, "7.1 depends on inv_freq[0] == 1.0 exactly");
    assert_eq!(e_pos_zero.to_bits(), 0x3F80_0000);
    assert_eq!(ln_one.to_bits(), 0x0000_0000, "6.5 must return +0.0 at 1.0");

    // ---- (A)-(E) exhaustive over all 2^32 bit patterns ------------------
    let workers = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let span = (1u64 << 32) / workers as u64;
    let mut all = Slice::new();
    let parts: Vec<Slice> = thread::scope(|sc| {
        let mut hs = Vec::new();
        let mut lo = 0u64;
        while lo < (1u64 << 32) {
            let hi = (lo + span).min(1u64 << 32);
            hs.push(sc.spawn(move || sweep(lo as u32, hi as u32)));
            lo = hi;
        }
        hs.into_iter().map(|h| h.join().expect("worker")).collect()
    });
    for p in &parts {
        all.merge(p);
    }

    println!("EXPLN-X exhaustive over every f32 bit pattern: checked={}", all.checked);
    assert_eq!(all.checked, 1u64 << 32, "the sweep must cover every pattern");

    // -- exp, per input binade, both signs ------------------------------
    // Below 2^-14 the polynomial is dominated by `1 + r` and every binade is
    // identical; above 2^7 every input is in a guard region. Everything in
    // between gets a row.
    println!("EXPLN-X --- (A) exp_pinned, by input binade ---");
    row("exp x in [0, 2^-14)", fold(&all.exp, 0..113));
    row("exp x in (-2^-14, -0]", fold(&all.exp, 256..369));
    for e in 113..135usize {
        let p = e as i32 - 127;
        if all.exp[e].n > 0 {
            row(&format!("exp x in [2^{p}, 2^{})", p + 1), all.exp[e]);
        }
        if all.exp[e + 256].n > 0 {
            row(&format!("exp x in (-2^{}, -2^{p}]", p + 1), all.exp[e + 256]);
        }
    }
    row("exp x >= 2^8 (guarded)", fold(&all.exp, 135..256));
    row("exp x <= -2^8 (guarded)", fold(&all.exp, 391..512));

    let exp_pos = fold(&all.exp, 0..256);
    let exp_neg = fold(&all.exp, 256..512);
    let exp_all = {
        let mut a = exp_pos;
        a.merge(exp_neg);
        a
    };
    println!("EXPLN-X --- (A) exp_pinned, totals ---");
    row("exp ALL finite", exp_all);
    row("exp x >= 0 (silu path)", exp_pos);
    println!("EXPLN-X --- (B) the softmax subdomain, x <= 0, ops.rs:126 ---");
    row("exp x <= 0 (softmax)", exp_neg);

    println!(
        "EXPLN-X exp guard: clipped-to-+Inf count={} over x in [{:e}, {:e}] \
         = [0x{:08X}, 0x{:08X}] (true value finite and representable; the x > 88.0 guard \
         cuts below ln(f32::MAX)=88.7228390520684)",
        all.exp_clip_hi,
        all.exp_clip_lo_at,
        all.exp_clip_hi_at,
        all.exp_clip_lo_at.to_bits(),
        all.exp_clip_hi_at.to_bits()
    );
    println!(
        "EXPLN-X exp guard: returned-0-where-oracle-nonzero={} (must be 0; exp(-88) is subnormal, \
         so the x < -88.0 guard sits inside the region 1.3 flushes anyway) first-at={:e}",
        all.exp_flush_bad, all.exp_flush_bad_at
    );
    println!("EXPLN-X exp NaN: non-NaN-for-NaN-input={} (must be 0)", all.exp_nan_bad);
    println!(
        "EXPLN-X exp monotone: violations={} (must be 0) first-at={:e}",
        all.exp_nonmono, all.exp_nonmono_at
    );

    // -- ln --------------------------------------------------------------
    println!("EXPLN-X --- (C) ln_pinned, by input binade (positive normals only) ---");
    // 16 groups of 16 exponent fields, so the table is readable and still
    // shows where the error lives.
    for g in 0..16usize {
        let lo = g * 16;
        let hi = lo + 16;
        let (flo, fhi) = (lo.max(1), hi.min(255));
        let w = fold(&all.ln, flo..fhi);
        if w.n == 0 {
            continue;
        }
        row(&format!("ln x in [2^{}, 2^{})", flo as i32 - 127, fhi as i32 - 127), w);
    }
    // The band where `ln` crosses its zero. §6.5's `frexp_exact` + LOG_SQRTHF
    // branch puts `m` near 0 here, the result is tiny, and the ULP column
    // stops meaning anything -- E26's lesson, restated on a different
    // function.
    row("ln x in [2^-1, 2^1)  ZERO", fold(&all.ln, 126..128));
    let ln_all = fold(&all.ln, 0..256);
    println!("EXPLN-X --- (C) ln_pinned, totals ---");
    row("ln ALL positive normals", ln_all);
    println!(
        "EXPLN-X ln guard: wrong-value-on-guard-domain={} (must be 0) first-at={:e}",
        all.ln_guard_bad, all.ln_guard_bad_at
    );
    println!(
        "EXPLN-X ln guard: subnormals-returning--Inf={} (= 2*(2^23-1) = 16777214; 1.3's DAZ makes \
         a subnormal compare equal to zero, so 6.5's zero branch is taken)",
        all.ln_subnormal_neg_inf
    );
    println!(
        "EXPLN-X ln monotone: violations={} (must be 0) first-at={:e}",
        all.ln_nonmono, all.ln_nonmono_at
    );
    for theta in [100000.0f32, 1000000.0f32] {
        let p = mathpin::ln_pinned(theta);
        let o = (theta as f64).ln();
        println!(
            "EXPLN-X ln theta={theta:e} pinned=0x{:08X} ({p:.9e}) oracle={o:.9e} \
             ulp={} rel={:.4e}",
            p.to_bits(),
            ulp_diff(p, o as f32).abs(),
            ((p as f64 - o) / o).abs()
        );
    }

    // -- silu -------------------------------------------------------------
    println!("EXPLN-X --- (D) silu_pinned (6.4), by input binade ---");
    row("silu x in [0, 2^-14)", fold(&all.silu, 0..113));
    row("silu x in (-2^-14, -0]", fold(&all.silu, 256..369));
    for e in 113..135usize {
        let p = e as i32 - 127;
        if all.silu[e].n > 0 {
            row(&format!("silu x in [2^{p}, 2^{})", p + 1), all.silu[e]);
        }
        if all.silu[e + 256].n > 0 {
            row(&format!("silu x in (-2^{}, -2^{p}]", p + 1), all.silu[e + 256]);
        }
    }
    row("silu x >= 2^8", fold(&all.silu, 135..256));
    row("silu x <= -2^8", fold(&all.silu, 391..512));
    let silu_all = {
        let mut a = fold(&all.silu, 0..256);
        a.merge(fold(&all.silu, 256..512));
        a
    };
    println!("EXPLN-X --- (D) silu_pinned, totals ---");
    row("silu ALL finite, x >= -88", silu_all);
    row("silu x < -88 (6.2 clip)", all.silu_clip);
}
