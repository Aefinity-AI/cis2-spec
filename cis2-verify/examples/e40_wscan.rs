//! E40: do the *weights* a real decode multiplies ever contain subnormals?
//!
//! §1.3's DAZ treats a subnormal **input** as zero. E39 showed §1.3 takes no
//! decision on `exp_pinned`'s output and none on §10's weights on the measured
//! vector. §8 matvec and §7 RMSNorm remain uncounted, and the cheapest exact
//! probe there is static: every product in §8 has a weight as one operand, so
//! a single subnormal weight means DAZ acts on every product that uses it.
//!
//! Operands are scanned as the decode sees them — after §2.5 widening, which
//! is what `safetensors::load` produces.
use cis2_verify::safetensors;
use std::collections::BTreeMap;

fn main() {
    let path = std::env::args().nth(1).expect("usage: e40_wscan <model.safetensors>");
    let bytes = std::fs::read(&path).expect("read model");
    let st = safetensors::load(&bytes).expect("parse");
    let names: Vec<String> = st.names().map(|s| s.to_string()).collect();

    let (mut total, mut sub, mut zero) = (0u64, 0u64, 0u64);
    let mut smallest: u32 = u32::MAX;
    let mut smallest_at = String::new();
    let mut first_sub: Option<(String, usize, u32)> = None;
    let mut by_kind: BTreeMap<String, (u64, u64)> = BTreeMap::new();

    for n in &names {
        let t = st.get(n).expect("tensor");
        // Kind = the trailing component, so per-layer tensors aggregate.
        let kind = n.rsplit('.').next().unwrap_or(n).to_string();
        let e = by_kind.entry(kind).or_insert((0, 0));
        for (i, v) in t.data.iter().enumerate() {
            let b = v.to_bits();
            let mag = b & 0x7FFF_FFFF;
            total += 1;
            e.0 += 1;
            if mag == 0 {
                zero += 1;
            } else if (b >> 23) & 0xFF == 0 {
                sub += 1;
                e.1 += 1;
                if first_sub.is_none() {
                    first_sub = Some((n.clone(), i, b));
                }
            } else if mag < smallest {
                smallest = mag;
                smallest_at = n.clone();
            }
        }
    }

    println!("E40-W file={path}");
    println!("E40-W tensors={} elements={total}", names.len());
    println!("E40-W subnormal_operands={sub} exact_zeros={zero}");
    println!(
        "E40-W smallest_nonzero_NORMAL={smallest:#010x} ({:e}) in {smallest_at}",
        f32::from_bits(smallest)
    );
    match &first_sub {
        Some((n, i, b)) => println!("E40-W first_subnormal={n}[{i}] bits={b:#010x} ({:e})", f32::from_bits(*b)),
        None => println!("E40-W first_subnormal=NONE"),
    }
    println!("E40-W --- per tensor kind (only kinds with subnormals, then totals) ---");
    for (k, (n, s)) in &by_kind {
        if *s > 0 {
            println!("E40-W   {k}: {s} subnormal of {n}");
        }
    }
    for (k, (n, s)) in &by_kind {
        println!("E40-W   ALL {k}: {n} elements, {s} subnormal");
    }
    // Positive control: the classifier must be able to see a subnormal.
    let p = f32::from_bits(0x0000_0001);
    println!(
        "E40-W detector control: bits={:#010x} classified_subnormal={}",
        p.to_bits(),
        p.to_bits() & 0x7FFF_FFFF != 0 && (p.to_bits() >> 23) & 0xFF == 0
    );
}
