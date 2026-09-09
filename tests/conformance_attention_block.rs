//! Self-test for `tests/conformance/vectors/attention_block_v1.txt` (E21
//! fifth slice: one attention block's score/softmax/V-mix). Runs the
//! CIS-2 reference §9.2 (score) / §9.3 (softmax) / §9.4 (V-mix) pipeline
//! for a single query head against a small causal KV cache
//! (`src/math.rs::dot_seq`/`rsqrt_cr`/`softmax_seq`, included by path
//! below so this test exercises the actual reference implementation, not
//! a re-typed copy) against the pinned fixture and checks both the
//! per-element bit patterns and the output digest against
//! `tests/conformance/vectors/attention_block_v1.expected`.
//!
//! GQA's `kv_head = qh / group` head-to-KV-head mapping (§9's opening
//! paragraph) is a pure indexing detail on top of this same per-head
//! arithmetic, not additional numeric behavior, so this vector covers
//! exactly one head and does not separately exercise the mapping.

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

struct Inputs {
    head_dim: usize,
    seq_len: usize,
    q_head: Vec<f32>,
    k: Vec<f32>, // seq_len * head_dim, row-major
    v: Vec<f32>, // seq_len * head_dim, row-major
}

fn load_inputs(vector_text: &str) -> Inputs {
    let head_dim: usize = parse_field(vector_text, "head_dim").trim().parse().unwrap();
    let seq_len: usize = parse_field(vector_text, "seq_len").trim().parse().unwrap();
    let q_head = parse_bits_list(parse_field(vector_text, "q_head_bits"));
    let k = parse_bits_list(parse_field(vector_text, "k_bits"));
    let v = parse_bits_list(parse_field(vector_text, "v_bits"));
    assert_eq!(q_head.len(), head_dim);
    assert_eq!(k.len(), seq_len * head_dim);
    assert_eq!(v.len(), seq_len * head_dim);
    Inputs { head_dim, seq_len, q_head, k, v }
}

/// §9.2 (score) + §9.3 (softmax) + §9.4 (V-mix), verbatim, for one query
/// head against a causal KV cache of `seq_len` positions (`pos = seq_len -
/// 1`, every cached position `j = 0..=pos` attended).
fn attention_block(inp: &Inputs) -> Vec<f32> {
    let scale = math::rsqrt_cr(inp.head_dim as f32); // §9.2: rsqrt(head_dim), computed once
    let mut scores = vec![0.0f32; inp.seq_len];
    for j in 0..inp.seq_len {
        let k_j = &inp.k[j * inp.head_dim..(j + 1) * inp.head_dim];
        let d = math::dot_seq(&inp.q_head, k_j); // §5.1
        scores[j] = d * scale; // separate multiply, applied AFTER the dot
    }
    math::softmax_seq(&mut scores); // §9.3, in place

    // §9.4 V-mix: for each output dim d, acc = sum_j scores[j] * v[j][d],
    // written as an explicit left-to-right loop per the spec (not dot_seq,
    // since the two operands are strided differently, but the same
    // separate-mul/separate-add rule applies).
    let mut out_head = vec![0.0f32; inp.head_dim];
    for d in 0..inp.head_dim {
        let mut acc = 0.0f32;
        for j in 0..inp.seq_len {
            let p = scores[j] * inp.v[j * inp.head_dim + d];
            acc = acc + p;
        }
        out_head[d] = acc;
    }
    out_head
}

#[test]
fn attention_block_v1_reproduces_pinned_digest() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance/vectors");
    let vector_text = fs::read_to_string(dir.join("attention_block_v1.txt")).unwrap();
    let expected_text = fs::read_to_string(dir.join("attention_block_v1.expected")).unwrap();

    let inp = load_inputs(&vector_text);
    let out = attention_block(&inp);

    let got_bits: Vec<u32> = out.iter().map(|f| f.to_bits()).collect();
    let got_digest = digest_f32_le(&out);

    let expected_bits: Vec<u32> = parse_field(&expected_text, "out_bits")
        .split(',')
        .map(parse_hex_u32)
        .collect();
    let expected_digest = parse_field(&expected_text, "out_sha256").trim().to_string();

    assert_eq!(
        got_bits, expected_bits,
        "attention_block_v1 output bit patterns diverged from tests/conformance/vectors/attention_block_v1.expected"
    );
    assert_eq!(
        got_digest, expected_digest,
        "attention_block_v1 digest diverged from pinned value in tests/conformance/vectors/attention_block_v1.expected"
    );
}

#[test]
#[ignore]
fn print_bits_for_generation() {
    // Not part of the pinned suite; run with
    // `cargo test --test conformance_attention_block print_bits -- --ignored --nocapture`
    // to regenerate the fixture if the reference math ever changes.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance/vectors");
    let vector_text = fs::read_to_string(dir.join("attention_block_v1.txt")).unwrap();
    let inp = load_inputs(&vector_text);
    let out = attention_block(&inp);
    let bits: Vec<String> = out.iter().map(|f| format!("0x{:08X}", f.to_bits())).collect();
    println!("out_bits={}", bits.join(","));
    println!("out_sha256={}", digest_f32_le(&out));
}
