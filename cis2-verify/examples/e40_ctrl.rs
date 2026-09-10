//! E40 positive control. E35's rule — a counter that reads zero must be shown
//! able to fire — applied before the §8/§5.1 counters are believed. Ground
//! truth is measured, not asserted: the same reduction is evaluated pinned and
//! with the pin cleared and restored, and the stored bits compared.
use cis2_verify::{fpenv, ops};
use std::hint::black_box;

fn dot1(a: f32, b: f32) -> u32 {
    // black_box in both directions: LLVM folds constant f32 arithmetic at
    // compile time, where MXCSR does not exist. This is the trap that made
    // §1.3's own self-test inert (erratum E-11) and E39's first control inert.
    let va = vec![black_box(a)];
    let vb = vec![black_box(b)];
    black_box(ops::dot_seq(black_box(&va), black_box(&vb))).to_bits()
}

fn main() {
    fpenv::pin_and_selftest().expect("1.3 pin");
    println!("E40-CTRL pin={}", fpenv::is_pinned());

    // (numerator, multiplier, exact product, what the predicate should say)
    let cases: [(f32, f32, &str); 4] = [
        (f32::MIN_POSITIVE, 1.0, "normal product, not decisive"),
        (f32::MIN_POSITIVE, 1.0e-2, "subnormal product, DECISIVE"),
        (f32::MIN_POSITIVE, 1.0e-7, "subnormal product, DECISIVE"),
        (f32::MIN_POSITIVE, 1.0e-10, "below half min subnormal, not decisive"),
    ];
    for (a, b, note) in cases {
        let before = ops::census::matvec_counts();
        let pinned = dot1(a, b);
        let after = ops::census::matvec_counts();
        let unpinned = fpenv::probe_unpinned(|| dot1(a, b));
        let exact = (a as f64) * (b as f64);
        println!(
            "E40-CTRL a=MIN_POSITIVE b={b:e} exact={exact:e} pred_decisive={} \
             pinned={pinned:#010x} unpinned={unpinned:#010x} DIFFER={} [{note}]",
            after.2 > before.2,
            pinned != unpinned
        );
    }

    // Case R: RMSNorm. A vector whose scaled value times its weight is
    // subnormal. Weight tiny, x ordinary: `scaled * weight[i]` underflows.
    let before = ops::census::rms_counts();
    let x = vec![black_box(1.0f32), black_box(1.0f32)];
    let w = vec![black_box(1.0e-38f32), black_box(1.0f32)];
    let pinned = ops::rmsnorm(black_box(&x), black_box(&w), black_box(1.0e-6f32));
    let after = ops::census::rms_counts();
    let unpinned = fpenv::probe_unpinned(|| ops::rmsnorm(black_box(&x), black_box(&w), black_box(1.0e-6f32)));
    println!(
        "E40-CTRL case-R d_n={} d_true_sub={} d_ftz_decisive={} pinned[0]={:#010x} unpinned[0]={:#010x} DIFFER={}",
        after.0 - before.0,
        after.1 - before.1,
        after.2 - before.2,
        pinned[0].to_bits(),
        unpinned[0].to_bits(),
        pinned[0].to_bits() != unpinned[0].to_bits()
    );

    // Case S: a subnormal *partial sum* rather than a subnormal product —
    // cancellation drives the accumulator into the subnormal range.
    let before = ops::census::matvec_counts();
    let a = vec![black_box(f32::MIN_POSITIVE), black_box(f32::MIN_POSITIVE)];
    let b = vec![black_box(1.0f32), black_box(-0.9999999f32)];
    let pinned = black_box(ops::dot_seq(black_box(&a), black_box(&b))).to_bits();
    let after = ops::census::matvec_counts();
    let unpinned = fpenv::probe_unpinned(|| black_box(ops::dot_seq(black_box(&a), black_box(&b))).to_bits());
    println!(
        "E40-CTRL case-S(cancellation) d_ftz_decisive={} pinned={pinned:#010x} unpinned={unpinned:#010x} DIFFER={}",
        after.2 - before.2,
        pinned != unpinned
    );
}
