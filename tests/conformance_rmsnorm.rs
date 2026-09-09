//! Self-test for `tests/conformance/vectors/rmsnorm_v1.txt` (E21 first
//! slice). Runs the CIS-2 reference RMSNorm (docs/CIS2_SPEC_v0.2.md §8,
//! same code as `src/math.rs::sum_seq`/`rsqrt_cr`, included verbatim by
//! path below so this test exercises the actual reference implementation,
//! not a re-typed copy) against the pinned fixture and checks the output
//! digest against `tests/conformance/vectors/rmsnorm_v1.expected`.
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

/// §8 RMSNorm, verbatim per spec (same operations/order as
/// `src/main.rs::rmsnorm`, reproduced here so this test only depends on
/// `src/math.rs` primitives, not on the model-loading binary).
fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut sq = vec![0.0f32; n];
    for i in 0..n {
        sq[i] = x[i] * x[i];
    }
    let ss = math::sum_seq(&sq);
    let mean = ss / (n as f32);
    let inv = math::rsqrt_cr(mean + eps);
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let scaled = x[i] * inv;
        out[i] = scaled * weight[i]; // (x[i]*inv)*weight[i] pinned association, §8
    }
    out
}

/// SHA-256 over each output element's little-endian f32 bytes, in index
/// order (same "feed_f32_le" convention as `src/math.rs::table_digest`).
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
fn rmsnorm_v1_reproduces_pinned_digest() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance/vectors");
    let vector_text = fs::read_to_string(dir.join("rmsnorm_v1.txt")).unwrap();
    let expected_text = fs::read_to_string(dir.join("rmsnorm_v1.expected")).unwrap();

    let n: usize = parse_field(&vector_text, "n").trim().parse().unwrap();
    let eps = f32::from_bits(parse_hex_u32(parse_field(&vector_text, "eps_bits")));
    let x = parse_bits_list(parse_field(&vector_text, "x_bits"));
    let gamma = parse_bits_list(parse_field(&vector_text, "gamma_bits"));
    assert_eq!(x.len(), n);
    assert_eq!(gamma.len(), n);

    let out = rmsnorm(&x, &gamma, eps);
    let got_digest = digest_f32_le(&out);

    let expected_digest = parse_field(&expected_text, "out_sha256").trim().to_string();
    let expected_bits: Vec<u32> = parse_field(&expected_text, "out_bits")
        .split(',')
        .map(|t| parse_hex_u32(t))
        .collect();
    let got_bits: Vec<u32> = out.iter().map(|f| f.to_bits()).collect();

    assert_eq!(
        got_bits, expected_bits,
        "rmsnorm output bit patterns diverged from tests/conformance/vectors/rmsnorm_v1.expected"
    );
    assert_eq!(
        got_digest, expected_digest,
        "rmsnorm_v1 digest diverged from pinned value in tests/conformance/vectors/rmsnorm_v1.expected"
    );
}
