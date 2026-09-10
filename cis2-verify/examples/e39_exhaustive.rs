//! E39: exhaustive check of the structural claim — `exp_pinned` never returns
//! a subnormal, over ALL 2^32 f32 arguments. Also reports the smallest nonzero
//! magnitude it can produce and the argument that produces it.
use cis2_verify::{fpenv, mathpin::exp_pinned};

fn main() {
    fpenv::pin_and_selftest().expect("1.3 pin");
    println!("E39-EXH pin={} control={}", fpenv::is_pinned(), fpenv::control_name());
    let mut subnormal_outputs: u64 = 0;
    let mut first_bad: Option<(u32, u32)> = None;
    let mut smallest: u32 = u32::MAX;
    let mut smallest_at: u32 = 0;
    let mut zeros: u64 = 0;
    let mut b: u64 = 0;
    while b <= u32::MAX as u64 {
        let x = f32::from_bits(b as u32);
        let y = exp_pinned(x).to_bits();
        let mag = y & 0x7FFF_FFFF;
        if mag == 0 {
            zeros += 1;
        } else if (y >> 23) & 0xFF == 0 {
            subnormal_outputs += 1;
            if first_bad.is_none() {
                first_bad = Some((b as u32, y));
            }
        } else if mag < smallest {
            smallest = mag;
            smallest_at = b as u32;
        }
        b += 1;
    }
    println!("E39-EXH arguments=4294967296 subnormal_outputs={subnormal_outputs} zeros={zeros}");
    println!(
        "E39-EXH smallest nonzero output={smallest:#010x} (exp_field={}) at x={:e} ({:#010x})",
        (smallest >> 23) & 0xFF,
        f32::from_bits(smallest_at),
        smallest_at
    );
    if let Some((xb, yb)) = first_bad {
        println!("E39-EXH FIRST SUBNORMAL OUTPUT x={xb:#010x} y={yb:#010x}");
    }
    // Positive control: the detector must be able to see a subnormal.
    let probe = f32::from_bits(0x0000_0001).to_bits();
    println!(
        "E39-EXH detector control: bits={probe:#010x} exp_field={} classified_subnormal={}",
        (probe >> 23) & 0xFF,
        probe & 0x7FFF_FFFF != 0 && (probe >> 23) & 0xFF == 0
    );
}
