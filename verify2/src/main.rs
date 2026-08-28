// cis2-verify2: E15g clean-room attempt #2 at CIS-2 v0.1 conformance.
//
// Implemented ONLY from docs/CIS2_SPEC_v0.1.md (body + Appendix A/B, which
// are part of the normative/informative spec document itself). This crate
// does not read, and was not written by reading, anything under src/,
// tests/, or verify/ in this repository (see verify2/CLEANROOM_LOG.md for
// the exact list of files read while writing it).

mod fpenv;
mod math;

use sha2::{Digest, Sha256};

const EPS_F32_BITS: u32 = 0x3727C5AC;

const HIDDEN: usize = 576;
const INTER: usize = 1536;
const N_LAYERS: usize = 30;
const N_HEADS: usize = 9;
const N_KV_HEADS: usize = 3;
const HEAD_DIM: usize = 64;
const HALF: usize = HEAD_DIM / 2;
const GROUP: usize = N_HEADS / N_KV_HEADS;
const VOCAB: usize = 49152;

const WEIGHTS_SHA256_EXPECT: &str =
    "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1";
const TOKENIZER_SHA256_EXPECT: &str =
    "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c";
const CONFIG_SHA256_EXPECT: &str =
    "1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843";

/// §2.1 artifact digest — raw 32-byte SHA-256, per v0.2 §12.1 (the witness
/// header is fed these raw bytes, not the 64-char hex-ASCII rendering).
fn sha256_raw(bytes: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().into()
}

struct Layer {
    input_layernorm: Vec<f32>,
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    o_proj: Vec<f32>,
    post_attention_layernorm: Vec<f32>,
    gate_proj: Vec<f32>,
    up_proj: Vec<f32>,
    down_proj: Vec<f32>,
}

struct Model {
    embed_tokens: Vec<f32>, // [VOCAB, HIDDEN], row-major
    layers: Vec<Layer>,
    norm: Vec<f32>,
    eps: f32,
    inv_freq: [f32; HALF],
}

fn widen_tensor(data: &[u8]) -> Vec<f32> {
    assert_eq!(data.len() % 2, 0);
    let mut out = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        let b = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(math::widen_bf16(b));
    }
    out
}

fn load_tensor(st: &safetensors::SafeTensors, name: &str) -> Vec<f32> {
    let view = st
        .tensor(name)
        .unwrap_or_else(|_| panic!("missing tensor {name}"));
    assert_eq!(
        view.dtype(),
        safetensors::Dtype::BF16,
        "tensor {name} is not BF16 on disk per spec §2.5"
    );
    widen_tensor(view.data())
}

/// §7.1 (v0.2) — config-driven `ln_pinned(rope_theta)`, replacing v0.1's
/// bare `LN_THETA` literal (pinned only for rope_theta=100000.0).
fn build_inv_freq(rope_theta: f64) -> [f32; HALF] {
    let ln_theta = math::ln_pinned(rope_theta as f32);
    let mut inv_freq = [0.0f32; HALF];
    for i in 0..HALF {
        let frac: f32 = (2 * i) as f32 / HEAD_DIM as f32;
        let neg_arg: f32 = -(frac * ln_theta);
        inv_freq[i] = math::exp_pinned(neg_arg);
    }
    inv_freq
}

fn inv_freq_table_digest(inv_freq: &[f32; HALF]) -> [u8; 32] {
    let mut h = Sha256::new();
    for v in inv_freq {
        h.update(v.to_le_bytes());
    }
    h.finalize().into()
}

fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut sq = vec![0.0f32; n];
    for i in 0..n {
        sq[i] = x[i] * x[i];
    }
    let ss = math::sum_seq(&sq);
    let mean: f32 = ss / (n as f32);
    let inv = math::rsqrt(mean + eps);
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let scaled: f32 = x[i] * inv;
        out[i] = scaled * weight[i];
    }
    out
}

/// row-major [out_features, in_features] weight, per §5.2.
fn matvec(w: &[f32], x: &[f32], out_features: usize, in_features: usize) -> Vec<f32> {
    assert_eq!(w.len(), out_features * in_features);
    assert_eq!(x.len(), in_features);
    let mut y = vec![0.0f32; out_features];
    for o in 0..out_features {
        let row = &w[o * in_features..(o + 1) * in_features];
        y[o] = math::dot_seq(row, x);
    }
    y
}

fn elementwise_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// §7.4 rotate-half RoPE application, in place, for one head_dim-length
/// head slice.
fn apply_rope_head(head: &mut [f32], cos_v: &[f32; HALF], sin_v: &[f32; HALF]) {
    let mut out = [0.0f32; HEAD_DIM];
    for i in 0..HALF {
        let x1 = head[i];
        let x2 = head[i + HALF];
        let t1: f32 = x1 * cos_v[i];
        let t2: f32 = (-x2) * sin_v[i];
        out[i] = t1 + t2;
        let t3: f32 = x2 * cos_v[i];
        let t4: f32 = x1 * sin_v[i];
        out[i + HALF] = t3 + t4;
    }
    head.copy_from_slice(&out);
}

struct KvCache {
    // per layer, per position: [N_KV_HEADS][HEAD_DIM]
    k: Vec<Vec<[[f32; HEAD_DIM]; N_KV_HEADS]>>,
    v: Vec<Vec<[[f32; HEAD_DIM]; N_KV_HEADS]>>,
}

impl KvCache {
    fn new() -> Self {
        KvCache {
            k: (0..N_LAYERS).map(|_| Vec::new()).collect(),
            v: (0..N_LAYERS).map(|_| Vec::new()).collect(),
        }
    }
}

/// One forward pass for a single token at position `pos`, updating the KV
/// cache in place and returning the fp32 logit vector (§9, §10, §11).
fn forward_token(model: &Model, token_id: u32, pos: usize, cache: &mut KvCache) -> Vec<f32> {
    let tid = token_id as usize;
    let mut h: Vec<f32> = model.embed_tokens[tid * HIDDEN..(tid + 1) * HIDDEN].to_vec();

    // §7.3 per-position cos/sin table (shared by all heads/layers this step).
    let mut cos_v = [0.0f32; HALF];
    let mut sin_v = [0.0f32; HALF];
    for i in 0..HALF {
        let angle: f32 = (pos as f32) * model.inv_freq[i];
        cos_v[i] = math::cos_pinned(angle);
        sin_v[i] = math::sin_pinned(angle);
    }

    let scale = math::rsqrt(HEAD_DIM as f32);

    for (li, layer) in model.layers.iter().enumerate() {
        let ln1 = rmsnorm(&h, &layer.input_layernorm, model.eps);

        let mut q = matvec(&layer.q_proj, &ln1, HIDDEN, HIDDEN);
        let mut k = matvec(&layer.k_proj, &ln1, N_KV_HEADS * HEAD_DIM, HIDDEN);
        let v = matvec(&layer.v_proj, &ln1, N_KV_HEADS * HEAD_DIM, HIDDEN);

        for qh in 0..N_HEADS {
            apply_rope_head(&mut q[qh * HEAD_DIM..(qh + 1) * HEAD_DIM], &cos_v, &sin_v);
        }
        for kh in 0..N_KV_HEADS {
            apply_rope_head(&mut k[kh * HEAD_DIM..(kh + 1) * HEAD_DIM], &cos_v, &sin_v);
        }

        // Store this position's post-RoPE k, raw v, into the cache.
        let mut k_entry: [[f32; HEAD_DIM]; N_KV_HEADS] = [[0.0f32; HEAD_DIM]; N_KV_HEADS];
        let mut v_entry: [[f32; HEAD_DIM]; N_KV_HEADS] = [[0.0f32; HEAD_DIM]; N_KV_HEADS];
        for kh in 0..N_KV_HEADS {
            k_entry[kh].copy_from_slice(&k[kh * HEAD_DIM..(kh + 1) * HEAD_DIM]);
            v_entry[kh].copy_from_slice(&v[kh * HEAD_DIM..(kh + 1) * HEAD_DIM]);
        }
        cache.k[li].push(k_entry);
        cache.v[li].push(v_entry);

        let mut attn_out = vec![0.0f32; HIDDEN];
        for qh in 0..N_HEADS {
            let kv_head = qh / GROUP;
            let q_head = &q[qh * HEAD_DIM..(qh + 1) * HEAD_DIM];

            let mut scores = Vec::with_capacity(pos + 1);
            for j in 0..=pos {
                let k_j = &cache.k[li][j][kv_head];
                let d: f32 = math::dot_seq(q_head, k_j);
                scores.push(d * scale);
            }

            // §9.3 softmax_seq, in place.
            let mut max_v = scores[0];
            for &sv in &scores[1..] {
                if sv > max_v {
                    max_v = sv;
                }
            }
            for sv in scores.iter_mut() {
                *sv = math::exp_pinned(*sv - max_v);
            }
            let denom = math::sum_seq(&scores);
            for sv in scores.iter_mut() {
                *sv = *sv / denom;
            }

            // §9.4 V-mix.
            for d in 0..HEAD_DIM {
                let mut acc: f32 = 0.0;
                for j in 0..=pos {
                    let p: f32 = scores[j] * cache.v[li][j][kv_head][d];
                    acc = acc + p;
                }
                attn_out[qh * HEAD_DIM + d] = acc;
            }
        }

        let o = matvec(&layer.o_proj, &attn_out, HIDDEN, HIDDEN);
        h = elementwise_add(&h, &o);

        let ln2 = rmsnorm(&h, &layer.post_attention_layernorm, model.eps);
        let gate = matvec(&layer.gate_proj, &ln2, INTER, HIDDEN);
        let up = matvec(&layer.up_proj, &ln2, INTER, HIDDEN);
        let mut hid = vec![0.0f32; INTER];
        for i in 0..INTER {
            hid[i] = math::silu_pinned(gate[i]) * up[i];
        }
        let down = matvec(&layer.down_proj, &hid, HIDDEN, INTER);
        h = elementwise_add(&h, &down);
    }

    let hn = rmsnorm(&h, &model.norm, model.eps);
    matvec(&model.embed_tokens, &hn, VOCAB, HIDDEN)
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = logits[0];
    for (idx, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = idx;
        }
    }
    best_idx as u32
}

struct RunResult {
    witness_digest: [u8; 32],
    argmax_digest: [u8; 32],
    generated_token_ids: Vec<u32>,
}

#[allow(clippy::too_many_arguments)]
fn run_decode(
    model: &Model,
    prompt_token_ids: &[u32; 4],
    weights_digest: [u8; 32],
    tokenizer_digest: [u8; 32],
    config_digest: [u8; 32],
    table_digest: [u8; 32],
    inv_freq_digest: [u8; 32],
) -> RunResult {
    let mut cache = KvCache::new();
    // v0.2 §12.1 witness header: raw 32-byte digests (not hex-ASCII, v0.1's
    // §14.2 gap), now ALSO binding table_digest and inv_freq_table_digest
    // (v0.1's §14.3 gap — those two were computed/printed but never fed
    // into CIS2_REF).
    let mut witness = Sha256::new();
    witness.update(weights_digest);
    witness.update(tokenizer_digest);
    witness.update(config_digest);
    witness.update(table_digest);
    witness.update(inv_freq_digest);
    for &t in prompt_token_ids {
        witness.update(t.to_le_bytes());
    }

    // Prefill: positions 0..=3, prompt tokens in order.
    let mut last_logits = Vec::new();
    for (pos, &tok) in prompt_token_ids.iter().enumerate() {
        last_logits = forward_token(model, tok, pos, &mut cache);
    }

    let mut generated_token_ids: Vec<u32> = Vec::with_capacity(16);
    let mut cur_logits = last_logits;
    for step in 0..16usize {
        for v in cur_logits.iter() {
            witness.update(v.to_bits().to_le_bytes());
        }
        let next_token = argmax(&cur_logits);
        witness.update(next_token.to_le_bytes());
        generated_token_ids.push(next_token);

        if step + 1 < 16 {
            let pos = 4 + step;
            cur_logits = forward_token(model, next_token, pos, &mut cache);
        }
    }

    let witness_digest: [u8; 32] = witness.finalize().into();

    let mut argmax_h = Sha256::new();
    for &t in prompt_token_ids {
        argmax_h.update(t.to_le_bytes());
    }
    for &t in &generated_token_ids {
        argmax_h.update(t.to_le_bytes());
    }
    let argmax_digest: [u8; 32] = argmax_h.finalize().into();

    RunResult {
        witness_digest,
        argmax_digest,
        generated_token_ids,
    }
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn main() {
    fpenv::pin_ftz_daz();
    fpenv::adversarial_selftest();

    let weights_path = "weights/model.safetensors";
    let config_path = "weights/config.json";
    let tokenizer_path = "weights/tokenizer.json";

    let weights_bytes = std::fs::read(weights_path)
        .unwrap_or_else(|e| panic!("failed to read {weights_path}: {e}"));
    let config_bytes = std::fs::read(config_path)
        .unwrap_or_else(|e| panic!("failed to read {config_path}: {e}"));
    let tokenizer_bytes = std::fs::read(tokenizer_path)
        .unwrap_or_else(|e| panic!("failed to read {tokenizer_path}: {e}"));

    let weights_digest = sha256_raw(&weights_bytes);
    let config_digest = sha256_raw(&config_bytes);
    let tokenizer_digest = sha256_raw(&tokenizer_bytes);
    let weights_hex = hex(&weights_digest);
    let config_hex = hex(&config_digest);
    let tokenizer_hex = hex(&tokenizer_digest);

    assert_eq!(
        weights_hex, WEIGHTS_SHA256_EXPECT,
        "model.safetensors sha256 mismatch"
    );
    assert_eq!(
        config_hex, CONFIG_SHA256_EXPECT,
        "config.json sha256 mismatch"
    );
    assert_eq!(
        tokenizer_hex, TOKENIZER_SHA256_EXPECT,
        "tokenizer.json sha256 mismatch"
    );

    let config: serde_json::Value =
        serde_json::from_slice(&config_bytes).expect("config.json parse");
    // §2.4 (v0.2): rope_theta is no longer restricted to exactly 100000.0 —
    // `math::ln_pinned` is a general pinned ln(), so any positive finite
    // rope_theta is conformant (v0.1's restriction is lifted, see
    // CIS2_SPEC_v0.2.md §7.1/§16 changelog).
    let rope_theta = config["rope_theta"].as_f64().expect("rope_theta field");
    let hidden_size = config["hidden_size"].as_u64().unwrap() as usize;
    assert_eq!(hidden_size, HIDDEN);
    let num_attention_heads = config["num_attention_heads"].as_u64().unwrap() as usize;
    assert_eq!(num_attention_heads, N_HEADS);
    let num_key_value_heads = config["num_key_value_heads"].as_u64().unwrap() as usize;
    assert_eq!(num_key_value_heads, N_KV_HEADS);
    let intermediate_size = config["intermediate_size"].as_u64().unwrap() as usize;
    assert_eq!(intermediate_size, INTER);
    let num_hidden_layers = config["num_hidden_layers"].as_u64().unwrap() as usize;
    assert_eq!(num_hidden_layers, N_LAYERS);
    let vocab_size = config["vocab_size"].as_u64().unwrap() as usize;
    assert_eq!(vocab_size, VOCAB);

    let eps = f32::from_bits(EPS_F32_BITS);
    // Cross-check against the JSON value's f64->f32 RNE cast, per §2.3.
    let eps_json = config["rms_norm_eps"].as_f64().expect("rms_norm_eps field");
    assert_eq!((eps_json as f32).to_bits(), EPS_F32_BITS);

    let st = safetensors::SafeTensors::deserialize(&weights_bytes).expect("safetensors parse");

    let embed_tokens = load_tensor(&st, "model.embed_tokens.weight");
    let norm = load_tensor(&st, "model.norm.weight");

    let mut layers = Vec::with_capacity(N_LAYERS);
    for i in 0..N_LAYERS {
        let p = |suffix: &str| format!("model.layers.{i}.{suffix}");
        layers.push(Layer {
            input_layernorm: load_tensor(&st, &p("input_layernorm.weight")),
            q_proj: load_tensor(&st, &p("self_attn.q_proj.weight")),
            k_proj: load_tensor(&st, &p("self_attn.k_proj.weight")),
            v_proj: load_tensor(&st, &p("self_attn.v_proj.weight")),
            o_proj: load_tensor(&st, &p("self_attn.o_proj.weight")),
            post_attention_layernorm: load_tensor(&st, &p("post_attention_layernorm.weight")),
            gate_proj: load_tensor(&st, &p("mlp.gate_proj.weight")),
            up_proj: load_tensor(&st, &p("mlp.up_proj.weight")),
            down_proj: load_tensor(&st, &p("mlp.down_proj.weight")),
        });
    }

    let inv_freq = build_inv_freq(rope_theta);
    let inv_freq_digest = inv_freq_table_digest(&inv_freq);
    let tbl_digest = math::table_digest();

    let model = Model {
        embed_tokens,
        layers,
        norm,
        eps,
        inv_freq,
    };

    // §3.1-3.3 tokenizer + prompt, add_special_tokens=false.
    let tokenizer = tokenizers::Tokenizer::from_bytes(&tokenizer_bytes)
        .expect("tokenizer.json parse");
    let encoding = tokenizer
        .encode("Once upon a time", false)
        .expect("tokenize prompt");
    let ids: Vec<u32> = encoding.get_ids().to_vec();
    assert_eq!(ids, vec![6403u32, 1980, 253, 655], "prompt_token_ids mismatch");
    let prompt_token_ids: [u32; 4] = [ids[0], ids[1], ids[2], ids[3]];

    // §12.4 determinism check: run the whole decode twice.
    let run1 = run_decode(
        &model,
        &prompt_token_ids,
        weights_digest,
        tokenizer_digest,
        config_digest,
        tbl_digest,
        inv_freq_digest,
    );
    let run2 = run_decode(
        &model,
        &prompt_token_ids,
        weights_digest,
        tokenizer_digest,
        config_digest,
        tbl_digest,
        inv_freq_digest,
    );

    assert_eq!(
        run1.witness_digest, run2.witness_digest,
        "§12.4: witness digest differs across same-process runs"
    );
    assert_eq!(
        run1.argmax_digest, run2.argmax_digest,
        "§12.4: argmax digest differs across same-process runs"
    );
    assert_eq!(
        run1.generated_token_ids, run2.generated_token_ids,
        "§12.4: generated token ids differ across same-process runs"
    );

    println!(
        "CIS2_VERIFY2 digest={} prompt_toks=4 gen_toks=16 dtype=fp32",
        hex(&run1.witness_digest)
    );
    println!("argmax_digest={}", hex(&run1.argmax_digest));
    println!("table_digest={}", hex(&tbl_digest));
    println!("inv_freq table_digest={}", hex(&inv_freq_digest));
    println!(
        "generated_token_ids={:?}",
        run1.generated_token_ids
    );
}
