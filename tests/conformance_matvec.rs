//! Self-test for `tests/conformance/vectors/matvec_v1.txt` (E21 fourth
//! slice: matvec). Runs the CIS-2 reference matvec
//! (docs/CIS2_SPEC_v0.2.md §5.2, which is `dot_seq` per row --
//! `src/math.rs::dot_seq`, included by path below so this test exercises
//! the actual reference implementation, not a re-typed copy) against the
//! pinned fixture and checks both the per-element bit patterns and the
//! output digest against `tests/conformance/vectors/matvec_v1.expected`.
//!
//! Row 0 of this vector is §5.1's own worked order-sensitivity example
//! (`[1e8, 1.0, -1e8]` dotted against `[1,1,1]` must give exactly `0.0`),
//! so this test also stands as a regression check for that specific claim.

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

/// §5.2 matvec, verbatim: y[o] = dot_seq(w[o,:], x) for o in 0..out_features.
fn matvec(w: &[f32], x: &[f32], out_features: usize, in_features: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out_features];
    for o in 0..out_features {
        let row = &w[o * in_features..(o + 1) * in_features];
        y[o] = math::dot_seq(row, x);
    }
    y
}

fn load_inputs(vector_text: &str) -> (usize, usize, Vec<f32>, Vec<f32>) {
    let out_features: usize = parse_field(vector_text, "out_features").trim().parse().unwrap();
    let in_features: usize = parse_field(vector_text, "in_features").trim().parse().unwrap();
    let w = parse_bits_list(parse_field(vector_text, "w_bits"));
    let x = parse_bits_list(parse_field(vector_text, "x_bits"));
    assert_eq!(w.len(), out_features * in_features);
    assert_eq!(x.len(), in_features);
    (out_features, in_features, w, x)
}

#[test]
fn matvec_v1_reproduces_pinned_digest() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance/vectors");
    let vector_text = fs::read_to_string(dir.join("matvec_v1.txt")).unwrap();
    let expected_text = fs::read_to_string(dir.join("matvec_v1.expected")).unwrap();

    let (out_features, in_features, w, x) = load_inputs(&vector_text);
    let out = matvec(&w, &x, out_features, in_features);

    // Row 0 is the §5.1 order-sensitivity worked example: must be exactly
    // 0.0_f32, not merely "close".
    assert_eq!(out[0], 0.0f32, "matvec_v1 row 0 (§5.1 worked example) must be exactly 0.0");

    let got_bits: Vec<u32> = out.iter().map(|f| f.to_bits()).collect();
    let got_digest = digest_f32_le(&out);

    let expected_bits: Vec<u32> = parse_field(&expected_text, "out_bits")
        .split(',')
        .map(parse_hex_u32)
        .collect();
    let expected_digest = parse_field(&expected_text, "out_sha256").trim().to_string();

    assert_eq!(
        got_bits, expected_bits,
        "matvec_v1 output bit patterns diverged from tests/conformance/vectors/matvec_v1.expected"
    );
    assert_eq!(
        got_digest, expected_digest,
        "matvec_v1 digest diverged from pinned value in tests/conformance/vectors/matvec_v1.expected"
    );
}

#[test]
#[ignore]
fn print_bits_for_generation() {
    // Not part of the pinned suite; run with
    // `cargo test --test conformance_matvec print_bits -- --ignored --nocapture`
    // to regenerate the fixture if the reference math ever changes.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance/vectors");
    let vector_text = fs::read_to_string(dir.join("matvec_v1.txt")).unwrap();
    let (out_features, in_features, w, x) = load_inputs(&vector_text);
    let out = matvec(&w, &x, out_features, in_features);
    let bits: Vec<String> = out.iter().map(|f| format!("0x{:08X}", f.to_bits())).collect();
    println!("out_bits={}", bits.join(","));
    println!("out_sha256={}", digest_f32_le(&out));
}
