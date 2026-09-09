//! Self-test for `tests/conformance/vectors/rope_v1.txt` (E21, RoPE table
//! slice). Runs the CIS-2 reference RoPE table construction
//! (docs/CIS2_SPEC_v0.2.md §7.1 `inv_freq` + §7.2 table digest + §7.3
//! per-position cos/sin table, same code path as `src/main.rs`'s
//! `inv_freq`/`inv_freq_table_digest` computation and the §7.3 loop,
//! reimplemented verbatim here against `src/math.rs::ln_pinned`/
//! `exp_pinned`/`cos_pinned`/`sin_pinned` included by path below) against
//! the pinned fixture and checks both the per-element bit patterns and two
//! digests: the full-output digest (this suite's general convention, see
//! `tests/conformance/README.md`) and the §7.2-specific `inv_freq`-only
//! raw-bytes digest (so this vector can also be checked directly against
//! the spec's own `inv_freq_table_digest` definition).

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

fn parse_usize_list(s: &str) -> Vec<usize> {
    s.split(',').map(|tok| tok.trim().parse().unwrap()).collect()
}

/// §7.1 `inv_freq` construction, verbatim per spec (same operations/order
/// as `src/main.rs`'s inv_freq loop).
fn inv_freq_table(head_dim: usize, rope_theta: f32) -> Vec<f32> {
    let ln_theta = math::ln_pinned(rope_theta);
    let half = head_dim / 2;
    let mut inv_freq = vec![0.0f32; half];
    for i in 0..half {
        let frac = (2 * i) as f32 / (head_dim as f32);
        let neg_arg = {
            let a = frac * ln_theta;
            -a
        };
        inv_freq[i] = math::exp_pinned(neg_arg);
    }
    inv_freq
}

/// §7.2 — raw 32-byte SHA-256 over the LE bytes of each `inv_freq` value,
/// in index order (same construction as `src/main.rs`'s
/// `inv_freq_table_digest`).
fn inv_freq_table_digest(inv_freq: &[f32]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for &v in inv_freq {
        h.update(v.to_le_bytes());
    }
    h.finalize().into()
}

/// §7.3 — per-position cos/sin table, verbatim per spec.
fn cos_sin_table(inv_freq: &[f32], pos: usize) -> (Vec<f32>, Vec<f32>) {
    let half = inv_freq.len();
    let mut cos_v = vec![0.0f32; half];
    let mut sin_v = vec![0.0f32; half];
    for i in 0..half {
        let angle = (pos as f32) * inv_freq[i];
        cos_v[i] = math::cos_pinned(angle);
        sin_v[i] = math::sin_pinned(angle);
    }
    (cos_v, sin_v)
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

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn rope_v1_reproduces_pinned_digest() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance/vectors");
    let vector_text = fs::read_to_string(dir.join("rope_v1.txt")).unwrap();
    let expected_text = fs::read_to_string(dir.join("rope_v1.expected")).unwrap();

    let head_dim: usize = parse_field(&vector_text, "head_dim").trim().parse().unwrap();
    let rope_theta = f32::from_bits(parse_hex_u32(parse_field(&vector_text, "rope_theta_bits")));
    let positions = parse_usize_list(parse_field(&vector_text, "positions"));

    let inv_freq = inv_freq_table(head_dim, rope_theta);

    // Full output: inv_freq[0..half], then per position (in the order
    // listed in `positions=`): cos_v[0..half], sin_v[0..half].
    let mut out: Vec<f32> = Vec::new();
    out.extend_from_slice(&inv_freq);
    for &pos in &positions {
        let (cos_v, sin_v) = cos_sin_table(&inv_freq, pos);
        out.extend_from_slice(&cos_v);
        out.extend_from_slice(&sin_v);
    }

    let got_bits: Vec<u32> = out.iter().map(|f| f.to_bits()).collect();
    let got_digest = digest_f32_le(&out);
    let got_inv_freq_digest = hex(&inv_freq_table_digest(&inv_freq));

    let expected_bits: Vec<u32> = parse_field(&expected_text, "out_bits")
        .split(',')
        .map(parse_hex_u32)
        .collect();
    let expected_digest = parse_field(&expected_text, "out_sha256").trim().to_string();
    let expected_inv_freq_digest = parse_field(&expected_text, "inv_freq_table_digest")
        .trim()
        .to_string();

    assert_eq!(
        got_bits, expected_bits,
        "rope_v1 output bit patterns diverged from tests/conformance/vectors/rope_v1.expected"
    );
    assert_eq!(
        got_digest, expected_digest,
        "rope_v1 full-output digest diverged from pinned value"
    );
    assert_eq!(
        got_inv_freq_digest, expected_inv_freq_digest,
        "rope_v1 §7.2 inv_freq_table_digest diverged from pinned value"
    );
}

#[test]
#[ignore]
fn print_bits_for_generation() {
    // Not part of the pinned suite; run with
    // `cargo test --test conformance_rope print_bits -- --ignored --nocapture`
    // to regenerate the fixture if the reference math ever changes.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance/vectors");
    let vector_text = fs::read_to_string(dir.join("rope_v1.txt")).unwrap();
    let head_dim: usize = parse_field(&vector_text, "head_dim").trim().parse().unwrap();
    let rope_theta = f32::from_bits(parse_hex_u32(parse_field(&vector_text, "rope_theta_bits")));
    let positions = parse_usize_list(parse_field(&vector_text, "positions"));

    let inv_freq = inv_freq_table(head_dim, rope_theta);
    let mut out: Vec<f32> = Vec::new();
    out.extend_from_slice(&inv_freq);
    for &pos in &positions {
        let (cos_v, sin_v) = cos_sin_table(&inv_freq, pos);
        out.extend_from_slice(&cos_v);
        out.extend_from_slice(&sin_v);
    }
    let bits: Vec<String> = out.iter().map(|f| format!("0x{:08X}", f.to_bits())).collect();
    println!("out_bits={}", bits.join(","));
    println!("out_sha256={}", digest_f32_le(&out));
    println!("inv_freq_table_digest={}", hex(&inv_freq_table_digest(&inv_freq)));
}
