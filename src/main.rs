//! CIS-2 E15b milestone 1 — fp32 reference forward pass for
//! HuggingFaceTB/SmolLM2-135M. 
//! See docs/E15b_REFERENCE_RESULT.md for the full writeup, repro commands,
//! and honest open questions. 
//!
//! E15d additions (see docs/E15d_bc_RESULT.md): `--gen-toks N`,
//! `--prompt-set FILE` (JSON array of prompt strings) for milestone (b);
//! optional QKV bias + untied LM head + generic rope_theta (via the new
//! `math::ln_pinned`) for milestone (c)'s second model family
//! (Qwen2.5-0.5B-Instruct). With no CLI args, behavior is unchanged from
//! E15c (single default prompt, gen_toks=16) except that the printed
//! `table_digest=` value changed because `math::table_digest()` now also
//! covers the new `ln_pinned` coefficient table.

mod denormal;
mod math;

use math::{dot_seq, exp_pinned, rsqrt_cr, silu_pinned, softmax_seq, widen_bf16};
use sha2::{Digest, Sha256};
use std::path::Path;

struct LayerWeights {
    input_layernorm: Vec<f32>,
    post_attention_layernorm: Vec<f32>,
    q_proj: Vec<f32>,    // [576,576] row-major [out,in]
    k_proj: Vec<f32>,    // [192,576]
    v_proj: Vec<f32>,    // [192,576]
    o_proj: Vec<f32>,    // [576,576]
    gate_proj: Vec<f32>, // [1536,576]
    up_proj: Vec<f32>,   // [1536,576]
    down_proj: Vec<f32>, // [576,1536]
    // E15d(c): optional QKV bias (Qwen2-family checkpoints set
    // attention_bias=true; Llama-family SmolLM2 has none of these tensors
    // in the checkpoint, so these stay None and forward_token adds nothing).
    q_bias: Option<Vec<f32>>,
    k_bias: Option<Vec<f32>>,
    v_bias: Option<Vec<f32>>,
}

struct Model {
    embed_tokens: Vec<f32>, // [vocab*hidden]
    layers: Vec<LayerWeights>,
    final_norm: Vec<f32>,
    hidden: usize,
    inter: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    vocab: usize,
    eps: f32,
    inv_freq: Vec<f32>, // len head_dim/2
    // E15d(c): tied embeddings reuse embed_tokens for the LM head (as
    // SmolLM2-135M does); an untied checkpoint provides a separate
    // lm_head.weight tensor here instead.
    lm_head: Option<Vec<f32>>,
}

fn matvec(w: &[f32], x: &[f32], out_features: usize, in_features: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), out_features * in_features);
    debug_assert_eq!(x.len(), in_features);
    let mut y = vec![0.0f32; out_features];
    for o in 0..out_features {
        let row = &w[o * in_features..(o + 1) * in_features];
        y[o] = dot_seq(row, x);
    }
    y
}

fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut sq = vec![0.0f32; n];
    for i in 0..n {
        sq[i] = x[i] * x[i];
    }
    let ss = math::sum_seq(&sq);
    let mean = ss / (n as f32);
    let inv = rsqrt_cr(mean + eps);
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let scaled = x[i] * inv;
        out[i] = scaled * weight[i];
    }
    out
}

/// E15d(c) — add an optional per-output-feature bias vector in place
/// (strict left-to-right elementwise add, §3.1), a no-op if `bias` is None
/// (Llama-family checkpoints with no QKV bias tensors).
fn add_bias_opt(x: &mut [f32], bias: &Option<Vec<f32>>) {
    if let Some(b) = bias {
        debug_assert_eq!(x.len(), b.len());
        for i in 0..x.len() {
            x[i] = x[i] + b[i];
        }
    }
}

fn elementwise_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; a.len()];
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
    out
}

/// RoPE rotate-half convention (rope_interleaved=false, matching config),
/// applied in place to one head's 64-dim slice, using CIS-2 §3.3(b) pinned
/// sin/cos.
fn apply_rope_head(head: &mut [f32], pos: usize, inv_freq: &[f32]) {
    let half = head.len() / 2;
    let mut cos_v = vec![0.0f32; half];
    let mut sin_v = vec![0.0f32; half];
    for i in 0..half {
        let angle = (pos as f32) * inv_freq[i];
        cos_v[i] = math::cos_pinned(angle);
        sin_v[i] = math::sin_pinned(angle);
    }
    let mut out = vec![0.0f32; head.len()];
    for i in 0..half {
        let x1 = head[i];
        let x2 = head[i + half];
        // rotate_half(x) = [-x2, x1]; out = x*cos + rotate_half(x)*sin
        let t1 = x1 * cos_v[i];
        let t2 = (-x2) * sin_v[i];
        out[i] = t1 + t2;
        let t3 = x2 * cos_v[i];
        let t4 = x1 * sin_v[i];
        out[i + half] = t3 + t4;
    }
    head.copy_from_slice(&out);
}

struct KvCache {
    k: Vec<Vec<f32>>, // per position, len n_kv_heads*head_dim
    v: Vec<Vec<f32>>,
}

/// E15k debug-only helper: SHA-256 over the raw LE f32 bit patterns of a
/// slice. Used only by the `CIS2_DUMP_LAYERS`/`dump_pos` side-channel below
/// to localize a cross-language divergence; not part of any normative
/// digest and not read by default (env var gated).
fn hash_f32_slice(xs: &[f32]) -> [u8; 32] {
    let mut h = Sha256::new();
    for &v in xs {
        h.update(v.to_bits().to_le_bytes());
    }
    h.finalize().into()
}

/// E15k debug-only: print one `CIS2_DUMP` line to stderr in a format the
/// C clean-room's matching dump mode also produces, so the two can be
/// diffed line-for-line. Not part of CIS2_REF; gated on `dump_pos`.
fn dump_line(layer: &str, field: &str, pos: usize, xs: &[f32]) {
    let d = hash_f32_slice(xs);
    eprintln!("CIS2_DUMP layer={layer} field={field} pos={pos} digest={}", hex::encode(d));
}

fn forward_token(
    model: &Model,
    caches: &mut [KvCache],
    pos: usize,
    token_id: u32,
    dump_positions: &[usize],
) -> Vec<f32> {
    let dump = dump_positions.contains(&pos);
    let hidden = model.hidden;
    let head_dim = model.head_dim;
    let n_heads = model.n_heads;
    let n_kv_heads = model.n_kv_heads;
    let group = n_heads / n_kv_heads;
    let scale = rsqrt_cr(head_dim as f32);

    let mut h: Vec<f32> = {
        let start = token_id as usize * hidden;
        model.embed_tokens[start..start + hidden].to_vec()
    };
    if dump {
        dump_line("embed", "token_embedding", pos, &h);
    }

    for (li, layer) in model.layers.iter().enumerate() {
        let ln1 = rmsnorm(&h, &layer.input_layernorm, model.eps);
        let mut q = matvec(&layer.q_proj, &ln1, hidden, hidden);
        add_bias_opt(&mut q, &layer.q_bias);
        let mut k = matvec(&layer.k_proj, &ln1, n_kv_heads * head_dim, hidden);
        add_bias_opt(&mut k, &layer.k_bias);
        let mut v = matvec(&layer.v_proj, &ln1, n_kv_heads * head_dim, hidden);
        add_bias_opt(&mut v, &layer.v_bias);

        for qh in 0..n_heads {
            let s = qh * head_dim;
            apply_rope_head(&mut q[s..s + head_dim], pos, &model.inv_freq);
        }
        for kh in 0..n_kv_heads {
            let s = kh * head_dim;
            apply_rope_head(&mut k[s..s + head_dim], pos, &model.inv_freq);
        }

        caches[li].k.push(k);
        caches[li].v.push(v);

        let mut attn_out = vec![0.0f32; hidden];
        for qh in 0..n_heads {
            let kv_head = qh / group;
            let qs = qh * head_dim;
            let q_head = &q[qs..qs + head_dim];

            let mut scores = vec![0.0f32; pos + 1];
            for j in 0..=pos {
                let kj = &caches[li].k[j][kv_head * head_dim..kv_head * head_dim + head_dim];
                let d = dot_seq(q_head, kj);
                scores[j] = d * scale;
            }
            softmax_seq(&mut scores);

            let mut out_head = vec![0.0f32; head_dim];
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..=pos {
                    let vj = caches[li].v[j][kv_head * head_dim + d];
                    let p = scores[j] * vj;
                    acc = acc + p;
                }
                out_head[d] = acc;
            }
            attn_out[qs..qs + head_dim].copy_from_slice(&out_head);
        }

        let o = matvec(&layer.o_proj, &attn_out, hidden, hidden);
        h = elementwise_add(&h, &o);
        if dump {
            dump_line(&format!("block{li}"), "post_attention_hidden", pos, &h);
        }

        let ln2 = rmsnorm(&h, &layer.post_attention_layernorm, model.eps);
        let gate = matvec(&layer.gate_proj, &ln2, model.inter, hidden);
        let up = matvec(&layer.up_proj, &ln2, model.inter, hidden);
        let mut hid = vec![0.0f32; model.inter];
        for i in 0..model.inter {
            hid[i] = silu_pinned(gate[i]) * up[i];
        }
        let down = matvec(&layer.down_proj, &hid, hidden, model.inter);
        h = elementwise_add(&h, &down);
        if dump {
            dump_line(&format!("block{li}"), "post_mlp_hidden", pos, &h);
        }
    }

    let hn = rmsnorm(&h, &model.final_norm, model.eps);
    if dump {
        dump_line("final_norm", "final_norm_output", pos, &hn);
    }
    // LM head: tied to embed_tokens when the checkpoint has no separate
    // lm_head.weight tensor (SmolLM2-135M; confirmed by header inspection,
    // see docs/E15b_REFERENCE_RESULT.md), otherwise the untied tensor
    // loaded into model.lm_head (E15d(c) generalization).
    let lm_head_weights = model.lm_head.as_ref().unwrap_or(&model.embed_tokens);
    let logits = matvec(lm_head_weights, &hn, model.vocab, hidden);
    if dump {
        dump_line("logits", "pre_argmax_logits", pos, &logits);
    }
    logits
}

fn load_f32_tensor(st: &safetensors::SafeTensors, name: &str, expect_shape: &[usize]) -> Vec<f32> {
    let view = st
        .tensor(name)
        .unwrap_or_else(|e| panic!("tensor {name} missing: {e}"));
    assert_eq!(view.shape(), expect_shape, "tensor {name} shape mismatch");
    let raw = view.data();
    assert_eq!(raw.len() % 2, 0, "bf16 tensor {name} has odd byte length");
    let n = raw.len() / 2;
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let lo = raw[2 * i];
        let hi = raw[2 * i + 1];
        let bits = u16::from_le_bytes([lo, hi]);
        out[i] = widen_bf16(bits); // §3.5 exact bit-shift widening
    }
    out
}

/// §2.1 artifact digest — returns the raw 32-byte SHA-256 digest (CIS-2 v0.2
/// §12.1 pins that the witness header is fed these raw bytes, not the
/// 64-char hex-ASCII rendering; hex is used only for display/printing, via
/// `hex::encode` at the call site).
fn sha256_file(path: &Path) -> [u8; 32] {
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let mut h = Sha256::new();
    h.update(&data);
    h.finalize().into()
}

// tiny hex encoder (avoid pulling in the `hex` crate for one call site)
mod hex {
    pub fn encode(bytes: impl AsRef<[u8]>) -> String {
        let mut s = String::new();
        for b in bytes.as_ref() {
            s.push_str(&format!("{:02x}", b));
        }
        s
    }
}

/// E15d(b) — CLI options. Defaults reproduce E15c's invocation exactly
/// (single hardcoded prompt "Once upon a time", gen_toks=16, weights/).
struct Args {
    gen_toks: usize,
    prompt_set: Option<std::path::PathBuf>,
    weights_dir: String,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut gen_toks = 16usize;
    let mut prompt_set = None;
    let mut weights_dir = "weights".to_string();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--gen-toks" => {
                i += 1;
                gen_toks = argv[i].parse().unwrap_or_else(|e| {
                    panic!("--gen-toks: invalid integer {:?}: {e}", argv[i])
                });
            }
            "--prompt-set" => {
                i += 1;
                prompt_set = Some(std::path::PathBuf::from(&argv[i]));
            }
            "--weights-dir" => {
                i += 1;
                weights_dir = argv[i].clone();
            }
            other => panic!("unknown CLI argument: {other}"),
        }
        i += 1;
    }
    Args {
        gen_toks,
        prompt_set,
        weights_dir,
    }
}

fn main() {
    denormal::pin_ftz_daz();
    assert!(
        denormal::ftz_daz_on(),
        "FTZ/DAZ (or FPCR.FZ on aarch64) must be pinned before any decode-path fp32 arithmetic (§3.4)"
    );

    let args = parse_args();

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(&args.weights_dir);
    let weights_path = root.join("model.safetensors");
    let tokenizer_path = root.join("tokenizer.json");
    let config_path = root.join("config.json");

    let weights_sha256 = sha256_file(&weights_path);
    let tokenizer_sha256 = sha256_file(&tokenizer_path);
    let config_sha256 = sha256_file(&config_path);

    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
    let hidden = config["hidden_size"].as_u64().unwrap() as usize;
    let inter = config["intermediate_size"].as_u64().unwrap() as usize;
    let n_layers = config["num_hidden_layers"].as_u64().unwrap() as usize;
    let n_heads = config["num_attention_heads"].as_u64().unwrap() as usize;
    let n_kv_heads = config["num_key_value_heads"].as_u64().unwrap() as usize;
    let vocab = config["vocab_size"].as_u64().unwrap() as usize;
    let eps = config["rms_norm_eps"].as_f64().unwrap() as f32;
    let rope_theta = config["rope_theta"].as_f64().unwrap();
    let head_dim = hidden / n_heads;

    // §3.3(b)-pinned RoPE inv_freq (normative, per spec-change on top of
    // milestone 1): inv_freq[i] = exp_pinned( -(2*i/d) * ln_pinned(theta) ),
    // using ONLY the route-(b) pinned polynomials already in src/math.rs —
    // NOT host f64::powf, NOT std ln/exp. E15d(c) generalizes this from a
    // hardcoded literal (pinned for exactly rope_theta=1e5, SmolLM2) to a
    // config-driven `ln_pinned(rope_theta)` call, since Qwen2.5-0.5B uses
    // rope_theta=1e6, not 1e5. This is the SAME math for SmolLM2 (theta is
    // still read from config.json; `math::tests::ln_matches_std_within_tolerance`
    // checks ln_pinned(100_000.0) and ln_pinned(1_000_000.0) both agree with
    // host ln to the same 2e-6 relative tolerance used elsewhere in this
    // module), just no longer restricted to one hardcoded value.
    let ln_theta = math::ln_pinned(rope_theta as f32);
    let mut inv_freq = vec![0.0f32; head_dim / 2];
    for i in 0..head_dim / 2 {
        let frac = (2 * i) as f32 / (head_dim as f32);
        let neg_arg = {
            let a = frac * ln_theta;
            -a
        };
        inv_freq[i] = exp_pinned(neg_arg);
    }
    // §7.2 (v0.2) — raw 32-byte digest, folded into the witness header below
    // (v0.1 computed and printed this but never bound it into CIS2_REF;
    // v0.2 §12.1 closes that gap, see docs/CIS2_SPEC_v0.2.md changelog).
    let inv_freq_table_digest: [u8; 32] = {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        for &v in &inv_freq {
            h.update(v.to_le_bytes());
        }
        h.finalize().into()
    };

    let raw = std::fs::read(&weights_path).expect("read model.safetensors");
    let st = safetensors::SafeTensors::deserialize(&raw).expect("parse safetensors");

    eprintln!("loading + widening weights (bf16 -> fp32, exact bit-shift, §3.5)...");
    let embed_tokens = load_f32_tensor(&st, "model.embed_tokens.weight", &[vocab, hidden]);

    // E15d(c): optional bias load helper — returns Some(vec) if the tensor
    // exists in the checkpoint (Qwen2-family attention_bias=true), None if
    // absent (Llama-family SmolLM2, which has no QKV bias tensors at all).
    let load_bias_opt = |st: &safetensors::SafeTensors, name: &str, len: usize| {
        st.tensor(name)
            .ok()
            .map(|_| load_f32_tensor(st, name, &[len]))
    };

    let mut layers = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let p = |suffix: &str| format!("model.layers.{i}.{suffix}");
        layers.push(LayerWeights {
            input_layernorm: load_f32_tensor(&st, &p("input_layernorm.weight"), &[hidden]),
            post_attention_layernorm: load_f32_tensor(
                &st,
                &p("post_attention_layernorm.weight"),
                &[hidden],
            ),
            q_proj: load_f32_tensor(&st, &p("self_attn.q_proj.weight"), &[hidden, hidden]),
            k_proj: load_f32_tensor(
                &st,
                &p("self_attn.k_proj.weight"),
                &[n_kv_heads * head_dim, hidden],
            ),
            v_proj: load_f32_tensor(
                &st,
                &p("self_attn.v_proj.weight"),
                &[n_kv_heads * head_dim, hidden],
            ),
            o_proj: load_f32_tensor(&st, &p("self_attn.o_proj.weight"), &[hidden, hidden]),
            gate_proj: load_f32_tensor(&st, &p("mlp.gate_proj.weight"), &[inter, hidden]),
            up_proj: load_f32_tensor(&st, &p("mlp.up_proj.weight"), &[inter, hidden]),
            down_proj: load_f32_tensor(&st, &p("mlp.down_proj.weight"), &[hidden, inter]),
            q_bias: load_bias_opt(&st, &p("self_attn.q_proj.bias"), hidden),
            k_bias: load_bias_opt(&st, &p("self_attn.k_proj.bias"), n_kv_heads * head_dim),
            v_bias: load_bias_opt(&st, &p("self_attn.v_proj.bias"), n_kv_heads * head_dim),
        });
    }
    let final_norm = load_f32_tensor(&st, "model.norm.weight", &[hidden]);
    // E15d(c): untied LM head, if the checkpoint carries a separate
    // lm_head.weight tensor (config tie_word_embeddings=false).
    let tie_word_embeddings = config
        .get("tie_word_embeddings")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let lm_head = if tie_word_embeddings {
        None
    } else {
        Some(load_f32_tensor(&st, "lm_head.weight", &[vocab, hidden]))
    };
    drop(st);
    drop(raw);
    eprintln!("weights loaded.");

    let model = Model {
        embed_tokens,
        layers,
        final_norm,
        hidden,
        inter,
        n_layers,
        n_heads,
        n_kv_heads,
        head_dim,
        vocab,
        eps,
        inv_freq,
        lm_head,
    };

    let tok = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .unwrap_or_else(|e| panic!("load tokenizer: {e}"));

    // E15d(b): --prompt-set FILE loads a JSON array of prompt strings
    // (checked into the repo, see prompts/e15d_prompt_set.json); with no
    // --prompt-set flag, behavior is byte-for-byte unchanged from E15c
    // (single hardcoded prompt "Once upon a time").
    let prompts: Vec<String> = match &args.prompt_set {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .unwrap_or_else(|e| panic!("read --prompt-set {path:?}: {e}"));
            serde_json::from_str(&text)
                .unwrap_or_else(|e| panic!("--prompt-set {path:?} must be a JSON array of strings: {e}"))
        }
        None => vec!["Once upon a time".to_string()],
    };
    let gen_toks = args.gen_toks;

    for (prompt_idx, prompt) in prompts.iter().enumerate() {
        run_one_prompt(
            &model,
            &tok,
            prompt,
            prompt_idx,
            gen_toks,
            weights_sha256,
            tokenizer_sha256,
            config_sha256,
            inv_freq_table_digest,
        );
    }
}

/// Runs the full deterministic-twice-run reference pass for one prompt and
/// prints its CIS2_REF block. Factored out of `main()` in E15d(b) so the
/// same logic drives both the legacy single-prompt path and the new
/// `--prompt-set` multi-prompt path identically.
#[allow(clippy::too_many_arguments)]
fn run_one_prompt(
    model: &Model,
    tok: &tokenizers::Tokenizer,
    prompt: &str,
    prompt_idx: usize,
    gen_toks: usize,
    weights_sha256: [u8; 32],
    tokenizer_sha256: [u8; 32],
    config_sha256: [u8; 32],
    inv_freq_table_digest: [u8; 32],
) {
    let enc = tok.encode(prompt, false).expect("tokenize prompt");
    let prompt_ids: Vec<u32> = enc.get_ids().to_vec();
    eprintln!("[prompt {prompt_idx}] token ids ({} toks): {prompt_ids:?}", prompt_ids.len());

    let run = |label: &str| {
        let mut caches: Vec<KvCache> = (0..model.n_layers)
            .map(|_| KvCache {
                k: Vec::new(),
                v: Vec::new(),
            })
            .collect();

        // CIS-2 v0.2 §12.1 witness header: raw 32-byte digests (not the
        // 64-char hex-ASCII rendering, v0.1's §14.2 gap — v0.2 closes it),
        // now ALSO binding table_digest and inv_freq_table_digest (v0.1's
        // §14.3 gap: those two digests were computed and printed but never
        // fed into CIS2_REF, so a receipt holder could not detect a
        // different transcendental table producing the same logits).
        let mut witness = Sha256::new();
        witness.update(weights_sha256);
        witness.update(tokenizer_sha256);
        witness.update(config_sha256);
        witness.update(math::table_digest());
        witness.update(inv_freq_table_digest);
        for &t in &prompt_ids {
            witness.update(t.to_le_bytes());
        }

        let mut all_token_ids: Vec<u32> = prompt_ids.clone();
        let mut logits: Vec<f32> = Vec::new();

        // E15k debug-only: per-layer state dump for the forward pass that
        // produces step 0's logits (the last prefill position), gated on
        // env var CIS2_DUMP_LAYERS (any value) and only on run1/prompt 0
        // to avoid duplicate/noisy output. Not part of CIS2_REF.
        // E15k: dump both the forward pass that produces step 0's logits
        // (pos = prompt_ids.len()-1) and the one that produces step 1's
        // logits (pos = prompt_ids.len()) -- step 0 alone was found to
        // match bit-for-bit against verify3/, so step 1 is needed to
        // localize the actual divergence.
        let dump_positions: Vec<usize> = if label == "run1"
            && prompt_idx == 0
            && std::env::var("CIS2_DUMP_LAYERS").is_ok()
        {
            vec![prompt_ids.len() - 1, prompt_ids.len()]
        } else {
            Vec::new()
        };

        // Prefill: run every prompt position to populate the KV cache.
        for (pos, &tid) in prompt_ids.iter().enumerate() {
            logits = forward_token(model, &mut caches, pos, tid, &dump_positions);
        }

        // Debug-only (m1.5 correctness check): dump the step-0 full fp32
        // logit vector (raw LE bytes) to a file if requested via env var.
        // Not part of the CIS2_REF normative output; used only to diff
        // against an external oracle. Only on the first labeled run to
        // avoid overwriting with run2 (both runs are identical anyway).
        if label == "run1" && prompt_idx == 0 {
            if let Ok(path) = std::env::var("CIS2_DUMP_STEP0_LOGITS") {
                let mut bytes = Vec::with_capacity(logits.len() * 4);
                for &v in &logits {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                std::fs::write(&path, &bytes)
                    .unwrap_or_else(|e| panic!("write {path}: {e}"));
                eprintln!(
                    "[debug] wrote step-0 logits ({} f32 values) to {path}",
                    logits.len()
                );
            }
        }

        for step in 0..gen_toks {
            // logits currently correspond to the last processed position.
            let mut best_idx = 0usize;
            let mut best_val = logits[0];
            for (idx, &v) in logits.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = idx;
                }
            }
            let next_id = best_idx as u32;

            // Bind this step's full fp32 logit vector into the witness
            // chain (bit pattern, LE, mirroring CIS-1's per-step chain but
            // over fp32 bits instead of exact integers).
            let mut step_bytes = Vec::with_capacity(logits.len() * 4);
            for &v in &logits {
                step_bytes.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            witness.update(&step_bytes);
            witness.update(next_id.to_le_bytes());

            all_token_ids.push(next_id);
            let pos = prompt_ids.len() + step;
            logits = forward_token(model, &mut caches, pos, next_id, &dump_positions);
            eprintln!("[{label}] step {step}: token_id={next_id}");
        }

        // Debug-only (H1 512-tok oracle check): dump the FINAL step's
        // full fp32 logit vector, mirroring CIS2_DUMP_STEP0_LOGITS above
        // but at the last generated step instead of step 0. Not part of
        // the CIS2_REF normative output.
        if label == "run1" && prompt_idx == 0 {
            if let Ok(path) = std::env::var("CIS2_DUMP_LASTSTEP_LOGITS") {
                let mut bytes = Vec::with_capacity(logits.len() * 4);
                for &v in &logits {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                std::fs::write(&path, &bytes)
                    .unwrap_or_else(|e| panic!("write {path}: {e}"));
                eprintln!(
                    "[debug] wrote final-step logits ({} f32 values) to {path}",
                    logits.len()
                );
            }
        }

        let witness_digest = witness.finalize();
        let mut argmax_hasher = Sha256::new();
        for &t in &all_token_ids {
            argmax_hasher.update(t.to_le_bytes());
        }
        let argmax_digest = argmax_hasher.finalize();

        (
            hex::encode(witness_digest),
            hex::encode(argmax_digest),
            all_token_ids,
        )
    };

    let (w1, a1, toks1) = run("run1");
    let (w2, a2, toks2) = run("run2");

    println!("weights_sha256={}", hex::encode(weights_sha256));
    println!("tokenizer_sha256={}", hex::encode(tokenizer_sha256));
    println!("config_sha256={}", hex::encode(config_sha256));
    println!("table_digest={}", hex::encode(math::table_digest()));
    println!("inv_freq_table_digest={}", hex::encode(inv_freq_table_digest));
    println!("prompt_idx={prompt_idx} prompt_tokens={prompt_ids:?}");
    println!("prompt_idx={prompt_idx} run1_tokens={toks1:?}");
    println!("prompt_idx={prompt_idx} run2_tokens={toks2:?}");
    println!("prompt_idx={prompt_idx} run1_witness_digest={w1}");
    println!("prompt_idx={prompt_idx} run2_witness_digest={w2}");
    println!("prompt_idx={prompt_idx} run1_argmax_digest={a1}");
    println!("prompt_idx={prompt_idx} run2_argmax_digest={a2}");
    println!(
        "prompt_idx={prompt_idx} runs_identical={}",
        w1 == w2 && a1 == a2 && toks1 == toks2
    );
    println!(
        "CIS2_REF digest={w1} prompt_idx={prompt_idx} prompt_toks={} gen_toks={gen_toks} dtype=fp32",
        prompt_ids.len()
    );
}
