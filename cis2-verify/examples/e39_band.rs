//! E39: does exp_pinned EVER return a subnormal, and is FTZ what flushes the
//! [-88.0, -87.33654) band, or is it ldexp_exact's own new_exp<=0 branch?
use cis2_verify::{fpenv, mathpin::exp_pinned};
use std::hint::black_box;

fn show(tag: &str, x: f32) {
    let a = black_box(exp_pinned(black_box(x))).to_bits();
    let u = fpenv::probe_unpinned(|| black_box(exp_pinned(black_box(x))).to_bits());
    let subnormal_asis = a != 0 && (a >> 23) & 0xFF == 0;
    let subnormal_unpin = u != 0 && (u >> 23) & 0xFF == 0;
    println!(
        "{tag:>26} x={x:<14} pinned={a:#010x}{} unpinned={u:#010x}{}  DIFFER={}",
        if subnormal_asis { " SUBNORMAL" } else { "" },
        if subnormal_unpin { " SUBNORMAL" } else { "" },
        a != u
    );
}

fn main() {
    fpenv::pin_and_selftest().expect("1.3 pin");
    println!("E39 pin={} control={}", fpenv::is_pinned(), fpenv::control_name());
    // the band 6.2 computes in, whose TRUE exp is subnormal
    show("just inside band (top)", -87.34);
    show("mid band", -87.6);
    show("band bottom", -87.999999);
    show("exactly -88.0 (golden)", -88.0);
    // guarded side
    show("guard fires", -88.369385);
    // controls: normal results either side
    show("last normal (approx)", -87.33654);
    show("ordinary", -1.0);
    println!("E39 pin_after={}", fpenv::is_pinned());
    // what is the smallest nonzero exp_pinned can return?
    let mut smallest = 0u32;
    let mut xs = 0f32;
    let mut x = -87.0f32;
    while x > -88.5 {
        let b = exp_pinned(x).to_bits();
        if b != 0 { smallest = b; xs = x; }
        x -= 0.0001;
    }
    println!("E39 smallest nonzero exp_pinned over [-88.5,-87.0] = {smallest:#010x} at x={xs} (exp_field={})", (smallest>>23)&0xFF);
}

// (appended) positive control for the new weight counter — a counter that
// reads zero must be shown able to fire.
#[allow(dead_code)]
fn unused() {}
