//! A reference "candidate" implementing the `tests/conformance/PROTOCOL.md`
//! stdin/stdout contract, built ONLY for smoke-testing
//! `cis2-conformance` itself (E21). This is NOT a third-party
//! implementation and is intentionally excluded from
//! `tests/conformance/README.md`'s vector list / advertised suite; it
//! exists so `cargo test` / CI can prove the checker binary actually
//! detects PASS and FAIL correctly, without relying on an external
//! binary being present.
//!
//! Reuses `src/math.rs` directly (unlike a real third-party candidate,
//! which per PROTOCOL.md must not depend on this repo's code at all) --
//! that is exactly why this file is not itself a conformance vector or
//! part of the advertised protocol surface.

#[path = "../math.rs"]
mod math;

use sha2::{Digest, Sha256};
use std::io::Read;

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
    panic!("field {key} not found in vector input");
}

fn parse_bits_list(s: &str) -> Vec<f32> {
    s.split(',')
        .map(|tok| f32::from_bits(parse_hex_u32(tok)))
        .collect()
}

fn digest_f32_le(v: &[f32]) -> String {
    let mut h = Sha256::new();
    for &f in v {
        h.update(f.to_le_bytes());
    }
    let out = h.finalize();
    out.iter().map(|b| format!("{b:02x}")).collect()
}

fn op_rmsnorm(text: &str) -> Vec<f32> {
    let n: usize = parse_field(text, "n").trim().parse().unwrap();
    let eps = f32::from_bits(parse_hex_u32(parse_field(text, "eps_bits")));
    let x = parse_bits_list(parse_field(text, "x_bits"));
    let gamma = parse_bits_list(parse_field(text, "gamma_bits"));
    assert_eq!(x.len(), n);
    let sq: Vec<f32> = x.iter().map(|&v| v * v).collect();
    let ss = math::sum_seq(&sq);
    let mean = ss / (n as f32);
    let inv = math::rsqrt_cr(mean + eps);
    (0..n).map(|i| (x[i] * inv) * gamma[i]).collect()
}

fn op_exp_pinned(text: &str) -> Vec<f32> {
    let x = parse_bits_list(parse_field(text, "x_bits"));
    x.iter().map(|&v| math::exp_pinned(v)).collect()
}

fn op_rope_table(text: &str) -> Vec<f32> {
    let head_dim: usize = parse_field(text, "head_dim").trim().parse().unwrap();
    let rope_theta = f32::from_bits(parse_hex_u32(parse_field(text, "rope_theta_bits")));
    let positions: Vec<usize> = parse_field(text, "positions")
        .split(',')
        .map(|t| t.trim().parse().unwrap())
        .collect();
    let half = head_dim / 2;
    let ln_theta = math::ln_pinned(rope_theta);
    let mut inv_freq = vec![0.0f32; half];
    for i in 0..half {
        let frac = (2 * i) as f32 / (head_dim as f32);
        let neg_arg = -(frac * ln_theta);
        inv_freq[i] = math::exp_pinned(neg_arg);
    }
    let mut out = inv_freq.clone();
    for &pos in &positions {
        for i in 0..half {
            let angle = (pos as f32) * inv_freq[i];
            out.push(math::cos_pinned(angle));
        }
        // NOTE: this reference candidate intentionally interleaves
        // cos/sin per the rope_v1 layout below, not per-i.
        let mut sins = vec![0.0f32; half];
        for i in 0..half {
            let angle = (pos as f32) * inv_freq[i];
            sins[i] = math::sin_pinned(angle);
        }
        out.extend_from_slice(&sins);
    }
    out
}

fn op_matvec(text: &str) -> Vec<f32> {
    let out_features: usize = parse_field(text, "out_features").trim().parse().unwrap();
    let in_features: usize = parse_field(text, "in_features").trim().parse().unwrap();
    let w = parse_bits_list(parse_field(text, "w_bits"));
    let x = parse_bits_list(parse_field(text, "x_bits"));
    (0..out_features)
        .map(|o| math::dot_seq(&w[o * in_features..(o + 1) * in_features], &x))
        .collect()
}

fn op_attention_block(text: &str) -> Vec<f32> {
    let head_dim: usize = parse_field(text, "head_dim").trim().parse().unwrap();
    let seq_len: usize = parse_field(text, "seq_len").trim().parse().unwrap();
    let q_head = parse_bits_list(parse_field(text, "q_head_bits"));
    let k = parse_bits_list(parse_field(text, "k_bits"));
    let v = parse_bits_list(parse_field(text, "v_bits"));
    let scale = math::rsqrt_cr(head_dim as f32);
    let mut scores = vec![0.0f32; seq_len];
    for j in 0..seq_len {
        let k_j = &k[j * head_dim..(j + 1) * head_dim];
        scores[j] = math::dot_seq(&q_head, k_j) * scale;
    }
    math::softmax_seq(&mut scores);
    let mut out_head = vec![0.0f32; head_dim];
    for d in 0..head_dim {
        let mut acc = 0.0f32;
        for j in 0..seq_len {
            acc = acc + scores[j] * v[j * head_dim + d];
        }
        out_head[d] = acc;
    }
    out_head
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let op_arg = args.get(1).cloned();

    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).expect("read stdin");
    let op_field = parse_field(&input, "op").to_string();
    if let Some(op_arg) = &op_arg {
        assert_eq!(op_arg, &op_field, "argv op and stdin op= disagree");
    }

    let out = match op_field.as_str() {
        "rmsnorm" => op_rmsnorm(&input),
        "exp_pinned" => op_exp_pinned(&input),
        "rope_table" => op_rope_table(&input),
        "matvec" => op_matvec(&input),
        "attention_block" => op_attention_block(&input),
        other => panic!("unsupported op: {other}"),
    };

    println!("out_sha256={}", digest_f32_le(&out));
}
