//! E40 §5.3 positive control for the **layer-dump differential**.
//!
//! §5.3 reports that a pinned dump and an unpinned dump of the §13.1 decode are
//! byte-identical over 14,934 records. A null result of that shape is only
//! evidence if the comparison could have come out the other way, so this
//! exercises the same three mechanisms on an input that *does* underflow:
//!
//!   1. `fpenv::probe_unpinned` really clears MXCSR around the call,
//!   2. `ops::tap::emit` records the raw bit pattern, so a stored subnormal
//!      survives into the file instead of being normalised on the way out,
//!   3. comparing the two files detects the difference.
//!
//! The operation is the §7 RMSNorm of E40 §3.1 case R: `scaled * weight[i]`
//! underflows for a tiny weight, which FTZ flushes to `+0.0` and an unpinned
//! SSE multiply stores as a subnormal.
//!
//!     cargo run --release --features layerdump --example e40_ldctrl

use cis2_verify::{fpenv, ops};
use std::hint::black_box;

fn dump_one(path: &str, unpinned: bool) -> Vec<u8> {
    ops::tap::open(path);
    let x = vec![black_box(1.0f32), black_box(1.0f32)];
    let w = vec![black_box(1.0e-38f32), black_box(1.0f32)];
    let call = || ops::rmsnorm(black_box(&x), black_box(&w), black_box(1.0e-6f32));
    let y = if unpinned { fpenv::probe_unpinned(call) } else { call() };
    ops::tap::emit("e40_ldctrl/rmsnorm_out", &y);
    ops::tap::close();
    println!(
        "E40-LDCTRL {:<8} is_pinned_during={} y[0]={:#010x} y[1]={:#010x}",
        if unpinned { "UNPINNED" } else { "pinned" },
        if unpinned { "false (expected)" } else { "true (expected)" },
        y[0].to_bits(),
        y[1].to_bits()
    );
    std::fs::read(path).expect("read back dump")
}

fn main() {
    fpenv::pin_and_selftest().expect("1.3 pin");
    println!("E40-LDCTRL build=instrumented (NOT a conforming verifier)");

    let a = dump_one("/tmp/e40-ldctrl-pinned.bin", false);
    let b = dump_one("/tmp/e40-ldctrl-unpinned.bin", true);
    println!("E40-LDCTRL pin-restored-after={}", fpenv::is_pinned());

    // The byte the two files must disagree on is inside the emitted payload.
    let differing: Vec<usize> = a
        .iter()
        .zip(b.iter())
        .enumerate()
        .filter(|(_, (p, q))| p != q)
        .map(|(i, _)| i)
        .collect();
    println!(
        "E40-LDCTRL bytes_pinned={} bytes_unpinned={} differing_byte_offsets={:?} FILES-DIFFER={}",
        a.len(),
        b.len(),
        differing,
        a != b
    );

    // A subnormal must actually be present in the unpinned file, or the control
    // would pass for the wrong reason (any difference at all would satisfy it).
    let mut subnormals_in_unpinned = 0usize;
    let mut i = 0usize;
    while i < b.len() {
        let ln = u32::from_le_bytes(b[i..i + 4].try_into().unwrap()) as usize;
        i += 4 + ln;
        let k = u32::from_le_bytes(b[i..i + 4].try_into().unwrap()) as usize;
        i += 4;
        for j in 0..k {
            let bits = u32::from_le_bytes(b[i + 4 * j..i + 4 * j + 4].try_into().unwrap());
            if (bits >> 23) & 0xFF == 0 && bits & 0x007F_FFFF != 0 {
                subnormals_in_unpinned += 1;
            }
        }
        i += 4 * k;
    }
    println!("E40-LDCTRL subnormals_stored_in_unpinned_file={subnormals_in_unpinned} (must be >0)");

    assert!(a != b, "control inert: the differential cannot see a 1.3 difference");
    assert!(
        subnormals_in_unpinned > 0,
        "control inert: no subnormal reached the dump file"
    );
    println!("E40-LDCTRL OK --- the differential is sensitive to 1.3");
}
