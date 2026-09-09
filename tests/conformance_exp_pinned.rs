//! Self-test for `tests/conformance/vectors/exp_pinned_v1.txt` (E21 third
//! slice: exp_pinned table). Runs the CIS-2 reference pinned exp(x)
//! (docs/CIS2_SPEC_v0.2.md §6.2, same code as `src/math.rs::exp_pinned`,
//! included verbatim by path below so this test exercises the actual
//! reference implementation, not a re-typed copy) against the pinned
//! fixture and checks both the per-element bit patterns and the output
//! digest against `tests/conformance/vectors/exp_pinned_v1.expected`.
//!
//! This proves the vector + expected digest pair in
//! `tests/conformance/` is correct and reproducible *before* any outside
//! implementer is asked to match it. It is deliberately independent of
//! `verify2`/`verify3` and of any model weights.

#[path = "../src/math.rs"]
mod math;

use std::fs;
use std::path::Path;

fn parse_hex_u32(s: &str) -> u32 {
    let s = s.trim();
    let s = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s);
    u32::from_str_radix(s, 16).unwrap()
}

fn parse_field<'a>(text: &'a str, key: &str) -> &'a str {
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix(&format!("{key}=")) {
            return rest;
        }
    }
    panic!("field {key} not found in vector file");
}

fn parse_bits_list(s: &str) -> Vec<f32> {
    s.split(',')
        .map(|tok| f32::from_bits(parse_hex_u32(tok)))
        .collect()
}

/// SHA-256 over each output element's little-endian f32 bytes, in index
/// order (same "feed_f32_le" convention as `src/math.rs::table_digest` /
/// `tests/conformance_rmsnorm.rs::digest_f32_le`).
fn digest_f32_le(v: &[f32]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for &f in v {
        h.update(f.to_le_bytes());
    }
    let out = h.finalize();
    out.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn exp_pinned_v1_reproduces_pinned_digest() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance/vectors");
    let vector_text = fs::read_to_string(dir.join("exp_pinned_v1.txt")).unwrap();
    let expected_text = fs::read_to_string(dir.join("exp_pinned_v1.expected")).unwrap();

    let n: usize = parse_field(&vector_text, "n").trim().parse().unwrap();
    let x = parse_bits_list(parse_field(&vector_text, "x_bits"));
    assert_eq!(x.len(), n);

    let out: Vec<f32> = x.iter().map(|&v| math::exp_pinned(v)).collect();
    let got_digest = digest_f32_le(&out);
    let got_bits: Vec<u32> = out.iter().map(|f| f.to_bits()).collect();

    let expected_bits: Vec<u32> = parse_field(&expected_text, "out_bits")
        .split(',')
        .map(parse_hex_u32)
        .collect();
    let expected_digest = parse_field(&expected_text, "out_sha256").trim().to_string();

    assert_eq!(
        got_bits, expected_bits,
        "exp_pinned_v1 output bit patterns diverged from tests/conformance/vectors/exp_pinned_v1.expected"
    );
    assert_eq!(
        got_digest, expected_digest,
        "exp_pinned_v1 digest diverged from pinned value in tests/conformance/vectors/exp_pinned_v1.expected"
    );
}

#[test]
#[ignore]
fn print_bits_for_generation() {
    // Not part of the pinned suite; run with
    // `cargo test --test conformance_exp_pinned print_bits -- --ignored --nocapture`
    // to regenerate the fixture if the reference math ever changes.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance/vectors");
    let vector_text = fs::read_to_string(dir.join("exp_pinned_v1.txt")).unwrap();
    let x = parse_bits_list(parse_field(&vector_text, "x_bits"));
    let out: Vec<f32> = x.iter().map(|&v| math::exp_pinned(v)).collect();
    let bits: Vec<String> = out.iter().map(|f| format!("0x{:08X}", f.to_bits())).collect();
    println!("out_bits={}", bits.join(","));
    println!("out_sha256={}", digest_f32_le(&out));
}
