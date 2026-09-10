//! E39 positive control: the softmax-weight FTZ counter must be able to fire.
use cis2_verify::{fpenv, ops::{census, softmax_seq}};

fn main() {
    fpenv::pin_and_selftest().expect("1.3 pin");
    println!("E39-CTRL pin={}", fpenv::is_pinned());

    // Construct a softmax vector whose normalized weights land in the
    // subnormal range: one dominant entry sets max_v and the denominator,
    // the others sit just above the low guard so exp_pinned returns the
    // smallest normals, which the division then pushes subnormal.
    let mut v: Vec<f32> = vec![0.0];              // exp -> 1.0
    for _ in 0..8 { v.push(-87.34); }             // exp -> smallest normals
    // denominator ~ 1.0 + 8*1.2e-38 ~ 1.0, so q ~ 1.2e-38: NOT subnormal.
    // Push it: make the max entry huge so the sum is huge.
    let mut w: Vec<f32> = vec![0.0, -87.34, -87.34];
    // scale trick: softmax subtracts max, so instead give many entries at 0.0
    for _ in 0..200 { w.push(0.0); }
    let mut a = v.clone();
    softmax_seq(&mut a);
    let (n1, s1, f1, m1) = census::weight_counts();
    println!("E39-CTRL case-A n={n1} true_sub={s1} ftz_decisive={f1} min|q|={m1:e}");
    softmax_seq(&mut w);
    let (n2, s2, f2, m2) = census::weight_counts();
    println!("E39-CTRL case-B n={n2} true_sub={s2} ftz_decisive={f2} min|q|={m2:e}");

    // Case C: the counter must fire through softmax_seq itself, not only on
    // hand-built division. 300 entries at 0.0 give a denominator near 300;
    // one entry just above -87.33654 gives a smallest-NORMAL numerator, and
    // the quotient is then subnormal.
    let mut c: Vec<f32> = vec![-87.336525];
    for _ in 0..300 { c.push(0.0); }
    let before = census::weight_counts();
    softmax_seq(&mut c);
    let after = census::weight_counts();
    println!(
        "E39-CTRL case-C dn={} d_true_sub={} d_ftz_decisive={} stored_bits={:#010x}",
        after.0 - before.0, after.1 - before.1, after.2 - before.2, c[0].to_bits()
    );

    // Direct arithmetic control on the counter's own predicate.
    use std::hint::black_box;
    let num = black_box(f32::MIN_POSITIVE);
    for denom in [1.0f32, 100.0, 1.0e7, 1.0e30] {
        let denom = black_box(denom);
        let q = (num as f64) / (denom as f64);
        let pinned = black_box(black_box(num) / black_box(denom));
        let subnorm = q != 0.0 && q.abs() < f32::MIN_POSITIVE as f64;
        let decisive = q.abs() >= 7.006492321624085e-46;
        let unpinned =
            fpenv::probe_unpinned(|| black_box(black_box(num) / black_box(denom)).to_bits());
        println!(
            "E39-CTRL num=MIN_POSITIVE denom={denom:e} exact_q={q:e} true_sub={subnorm} \
             pred_decisive={} pinned_bits={:#010x} unpinned_bits={unpinned:#010x} DIFFER={}",
            subnorm && decisive, pinned.to_bits(), pinned.to_bits() != unpinned
        );
    }
}
