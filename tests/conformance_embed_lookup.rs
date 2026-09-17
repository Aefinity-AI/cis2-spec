//! Self-test for `tests/conformance/vectors/embed_lookup_v1.txt` (E21
//! seventh slice: token-embedding lookup). Reimplements, verbatim, the
//! embedding-lookup expression from `src/main.rs`'s per-decode-step
//! forward pass (`let start = token_id as usize * hidden;
//! model.embed_tokens[start..start + hidden].to_vec()` -- see
//! `src/main.rs`'s `forward_step`, immediately before the layer loop /
//! §9 attention) and checks both the per-element bit patterns and the
//! output digest against
//! `tests/conformance/vectors/embed_lookup_v1.expected`.
//!
//! Unlike the other conformance_*.rs self-tests, this does not include
//! `src/math.rs` by path: the lookup itself is pure indexing/slicing of
//! an already-widened fp32 table, not a floating-point computation, so
//! there is no transcendental/reduction routine to share; the thing
//! being pinned is the row-major `start = token_id * hidden` indexing
//! convention from §2.5, reproduced here identically to `src/main.rs`.

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

fn parse_u32_list(s: &str) -> Vec<u32> {
    s.split(',').map(|tok| tok.trim().parse().unwrap()).collect()
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

/// §2.5 embed_tokens.weight row-major lookup, verbatim from
/// `src/main.rs`'s forward_step: `start = token_id * hidden`, then a
/// contiguous `hidden`-wide slice.
fn embed_lookup(embed: &[f32], hidden: usize, token_id: u32) -> Vec<f32> {
    let start = token_id as usize * hidden;
    embed[start..start + hidden].to_vec()
}

fn load_inputs(vector_text: &str) -> (usize, usize, Vec<f32>, Vec<u32>) {
    let vocab: usize = parse_field(vector_text, "vocab").trim().parse().unwrap();
    let hidden: usize = parse_field(vector_text, "hidden").trim().parse().unwrap();
    let embed = parse_bits_list(parse_field(vector_text, "embed_bits"));
    let token_ids = parse_u32_list(parse_field(vector_text, "token_ids"));
    assert_eq!(embed.len(), vocab * hidden);
    (vocab, hidden, embed, token_ids)
}

#[test]
fn embed_lookup_v1_reproduces_pinned_digest() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance/vectors");
    let vector_text = fs::read_to_string(dir.join("embed_lookup_v1.txt")).unwrap();
    let expected_text = fs::read_to_string(dir.join("embed_lookup_v1.expected")).unwrap();

    let (_vocab, hidden, embed, token_ids) = load_inputs(&vector_text);
    let mut out = Vec::with_capacity(token_ids.len() * hidden);
    for &id in &token_ids {
        out.extend(embed_lookup(&embed, hidden, id));
    }

    let got_bits: Vec<u32> = out.iter().map(|f| f.to_bits()).collect();
    let got_digest = digest_f32_le(&out);

    let expected_bits: Vec<u32> = parse_field(&expected_text, "out_bits")
        .split(',')
        .map(parse_hex_u32)
        .collect();
    let expected_digest = parse_field(&expected_text, "out_sha256").trim().to_string();

    assert_eq!(
        got_bits, expected_bits,
        "embed_lookup_v1 output bit patterns diverged from tests/conformance/vectors/embed_lookup_v1.expected"
    );
    assert_eq!(
        got_digest, expected_digest,
        "embed_lookup_v1 digest diverged from pinned value in tests/conformance/vectors/embed_lookup_v1.expected"
    );
}

#[test]
#[ignore]
fn print_bits_for_generation() {
    // Not part of the pinned suite; run with
    // `cargo test --test conformance_embed_lookup print_bits -- --ignored --nocapture`
    // to regenerate the fixture if the reference math ever changes.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance/vectors");
    let vector_text = fs::read_to_string(dir.join("embed_lookup_v1.txt")).unwrap();
    let (_vocab, hidden, embed, token_ids) = load_inputs(&vector_text);
    let mut out = Vec::with_capacity(token_ids.len() * hidden);
    for &id in &token_ids {
        out.extend(embed_lookup(&embed, hidden, id));
    }
    let bits: Vec<String> = out.iter().map(|f| format!("0x{:08X}", f.to_bits())).collect();
    println!("out_bits={}", bits.join(","));
    println!("out_sha256={}", digest_f32_le(&out));
}
