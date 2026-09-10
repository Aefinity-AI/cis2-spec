//! Spec 7-11: RoPE, RMSNorm placement, GQA attention, SwiGLU, the tied LM
//! head, and the decode protocol of spec 3.4.
//!
//! Every reduction in here goes through `ops.rs`, which is spec 5's
//! strictly-left-to-right order; every transcendental goes through
//! `mathpin.rs`, which is spec 6's pinned polynomials. This module supplies
//! only the arrangement, and the arrangement is where the remaining
//! freedom lives: which association, which position index, which head maps
//! to which KV head.

use crate::config::Config;
use crate::mathpin::{cos_pinned, exp_pinned, ln_pinned, rsqrt, silu_pinned};
use crate::ops::{argmax, dot_seq, matvec, rmsnorm, softmax_seq};
use crate::mathpin::sin_pinned;
use crate::safetensors::SafeTensors;
use crate::sha256::Sha256;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

struct LayerRefs<'a> {
    input_ln: &'a [f32],
    post_attn_ln: &'a [f32],
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    o: &'a [f32],
    gate: &'a [f32],
    up: &'a [f32],
    down: &'a [f32],
    // Spec 9.1 (new in v0.2): present only on checkpoints that carry them.
    // SmolLM2-135M does not, so these stay `None` and the bias add is a
    // no-op that cannot perturb the pinned test vector.
    q_bias: Option<&'a [f32]>,
    k_bias: Option<&'a [f32]>,
    v_bias: Option<&'a [f32]>,
}

pub struct Model<'a> {
    pub cfg: Config,
    embed: &'a [f32],
    layers: Vec<LayerRefs<'a>>,
    final_norm: &'a [f32],
    /// Spec 11.1: the tied embedding table, or a separate `lm_head.weight`
    /// when the config says the embeddings are not tied.
    lm_head: &'a [f32],
    /// Spec 7.1, computed once at load from `rope_theta`.
    pub inv_freq: Vec<f32>,
}

impl<'a> Model<'a> {
    pub fn load(st: &'a SafeTensors, cfg: Config) -> Result<Model<'a>, String> {
        let h = cfg.hidden_size;
        let kv = cfg.kv_width();
        let inter = cfg.intermediate_size;

        let embed = st.get_2d("model.embed_tokens.weight", cfg.vocab_size, h)?.data.as_slice();

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = alloc::format!("model.layers.{i}");
            let bias = |suffix: &str, n: usize| -> Result<Option<&'a [f32]>, String> {
                let name = alloc::format!("{p}.self_attn.{suffix}");
                if st.contains(&name) {
                    Ok(Some(st.get_1d(&name, n)?.data.as_slice()))
                } else {
                    Ok(None)
                }
            };
            layers.push(LayerRefs {
                input_ln: st.get_1d(&alloc::format!("{p}.input_layernorm.weight"), h)?.data.as_slice(),
                post_attn_ln: st
                    .get_1d(&alloc::format!("{p}.post_attention_layernorm.weight"), h)?
                    .data
                    .as_slice(),
                q: st.get_2d(&alloc::format!("{p}.self_attn.q_proj.weight"), h, h)?.data.as_slice(),
                k: st.get_2d(&alloc::format!("{p}.self_attn.k_proj.weight"), kv, h)?.data.as_slice(),
                v: st.get_2d(&alloc::format!("{p}.self_attn.v_proj.weight"), kv, h)?.data.as_slice(),
                o: st.get_2d(&alloc::format!("{p}.self_attn.o_proj.weight"), h, h)?.data.as_slice(),
                gate: st.get_2d(&alloc::format!("{p}.mlp.gate_proj.weight"), inter, h)?.data.as_slice(),
                up: st.get_2d(&alloc::format!("{p}.mlp.up_proj.weight"), inter, h)?.data.as_slice(),
                down: st.get_2d(&alloc::format!("{p}.mlp.down_proj.weight"), h, inter)?.data.as_slice(),
                q_bias: bias("q_proj.bias", h)?,
                k_bias: bias("k_proj.bias", kv)?,
                v_bias: bias("v_proj.bias", kv)?,
            });
        }

        let final_norm = st.get_1d("model.norm.weight", h)?.data.as_slice();

        // Spec 11.1. The tied case is what the pinned test vector exercises;
        // the untied branch is spec-described and, as spec 11.1 itself says,
        // less validated -- so it is taken only when the config asks for it
        // and the tensor is actually present, and refused loudly otherwise.
        let lm_head = if cfg.tie_word_embeddings {
            if st.contains("lm_head.weight") {
                return Err("safetensors: config sets tie_word_embeddings = true but the \
                            checkpoint also carries lm_head.weight; spec 11.1 does not say \
                            which wins, so this verifier refuses to guess"
                    .into());
            }
            embed
        } else {
            st.get_2d("lm_head.weight", cfg.vocab_size, h)?.data.as_slice()
        };

        let inv_freq = build_inv_freq(&cfg);

        Ok(Model {
            cfg,
            embed,
            layers,
            final_norm,
            lm_head,
            inv_freq,
        })
    }

    /// Spec 7.2: SHA-256 over the LE bytes of the `inv_freq` values in index
    /// order.
    pub fn inv_freq_table_digest(&self) -> [u8; 32] {
        let mut s = Sha256::new();
        for v in &self.inv_freq {
            s.update(&v.to_le_bytes());
        }
        s.finalize()
    }

    /// One decode step: run `token` at `pos`, extend the cache, and return
    /// the full fp32 logit vector (spec 11.1).
    pub fn forward(&self, token: u32, pos: usize, cache: &mut KvCache) -> Vec<f32> {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let half = cfg.half();
        let eps = cfg.rms_norm_eps;

        // Spec 7.3, once per step: the cos/sin tables at this position.
        let mut cos_v = vec![0.0f32; half];
        let mut sin_v = vec![0.0f32; half];
        for i in 0..half {
            let angle = (pos as f32) * self.inv_freq[i];
            cos_v[i] = cos_pinned(angle);
            sin_v[i] = sin_pinned(angle);
        }

        // Spec 9.2: computed once. For head_dim = 64 this is exactly 0.125.
        let scale = rsqrt(head_dim as f32);

        let mut hs: Vec<f32> = self.embed[token as usize * h..(token as usize + 1) * h].to_vec();
        // 14.6 activation tap. Write-only; compiled out without the feature.
        #[cfg(feature = "layerdump")]
        crate::ops::tap::emit(&format!("p{pos}.embed"), &hs);

        for (li, layer) in self.layers.iter().enumerate() {
            // ---- attention block ----
            let ln1 = rmsnorm(&hs, layer.input_ln, eps);
            let mut q = matvec(layer.q, &ln1, h, h);
            let mut k = matvec(layer.k, &ln1, cfg.kv_width(), h);
            let mut v = matvec(layer.v, &ln1, cfg.kv_width(), h);
            // Spec 9.1: the optional QKV bias, added elementwise before RoPE.
            add_bias(&mut q, layer.q_bias);
            add_bias(&mut k, layer.k_bias);
            add_bias(&mut v, layer.v_bias);
            // The tap is taken *after* the bias add, because the oracle's
            // `q_proj` is an `nn.Linear` whose output already includes it.
            // SmolLM2 has no QKV bias and Qwen2.5 does, so taking it before
            // would compare different quantities on the second model only.
            #[cfg(feature = "layerdump")]
            {
                crate::ops::tap::emit(&format!("p{pos}.L{li}.ln1"), &ln1);
                crate::ops::tap::emit(&format!("p{pos}.L{li}.q_proj"), &q);
                crate::ops::tap::emit(&format!("p{pos}.L{li}.k_proj"), &k);
                crate::ops::tap::emit(&format!("p{pos}.L{li}.v_proj"), &v);
            }

            // Spec 9.1 / 7.4: rotate every query head and every key head at
            // this step's position. `v` is never rotated.
            for qh in 0..cfg.num_attention_heads {
                rope_rotate(&mut q[qh * head_dim..(qh + 1) * head_dim], &cos_v, &sin_v, half);
            }
            for kh in 0..cfg.num_key_value_heads {
                rope_rotate(&mut k[kh * head_dim..(kh + 1) * head_dim], &cos_v, &sin_v, half);
            }

            // Spec 3.5: the cache stores post-RoPE K and un-rotated V.
            cache.push(li, &k, &v);
            let n_pos = cache.len(li);

            let mut attn_out = vec![0.0f32; h];
            for qh in 0..cfg.num_attention_heads {
                let kv_head = qh / cfg.group();
                let q_head = &q[qh * head_dim..(qh + 1) * head_dim];

                // Spec 9.2: causal by construction -- the cache holds exactly
                // positions 0..=pos, so there is no mask value to add.
                let mut scores = vec![0.0f32; n_pos];
                for (j, score) in scores.iter_mut().enumerate() {
                    let k_j = cache.key(li, j, kv_head, head_dim);
                    let d = dot_seq(q_head, k_j);
                    *score = d * scale;
                }

                // Spec 9.3.
                softmax_seq(&mut scores);

                // Spec 9.4, written as the explicit loop the spec writes.
                let out_head = &mut attn_out[qh * head_dim..(qh + 1) * head_dim];
                for (d, slot) in out_head.iter_mut().enumerate() {
                    let mut acc = 0.0f32;
                    for j in 0..n_pos {
                        let p = scores[j] * cache.value(li, j, kv_head, head_dim)[d];
                        acc = acc + p;
                    }
                    *slot = acc;
                }
            }

            // Spec 9.5.
            let o = matvec(layer.o, &attn_out, h, h);
            #[cfg(feature = "layerdump")]
            {
                crate::ops::tap::emit(&format!("p{pos}.L{li}.attn_out"), &attn_out);
                crate::ops::tap::emit(&format!("p{pos}.L{li}.o_proj"), &o);
            }
            for i in 0..h {
                hs[i] = hs[i] + o[i];
            }
            #[cfg(feature = "layerdump")]
            crate::ops::tap::emit(&format!("p{pos}.L{li}.resid_attn"), &hs);

            // ---- MLP block (spec 10) ----
            let ln2 = rmsnorm(&hs, layer.post_attn_ln, eps);
            let gate = matvec(layer.gate, &ln2, cfg.intermediate_size, h);
            let up = matvec(layer.up, &ln2, cfg.intermediate_size, h);
            let mut hid = vec![0.0f32; cfg.intermediate_size];
            for i in 0..cfg.intermediate_size {
                // 14.1 reach census. Reads only; compiled out without the feature.
                #[cfg(feature = "census")]
                crate::ops::census::note_silu(gate[i]);
                hid[i] = silu_pinned(gate[i]) * up[i];
            }
            let down = matvec(layer.down, &hid, h, cfg.intermediate_size);
            #[cfg(feature = "layerdump")]
            {
                crate::ops::tap::emit(&format!("p{pos}.L{li}.ln2"), &ln2);
                crate::ops::tap::emit(&format!("p{pos}.L{li}.gate_proj"), &gate);
                crate::ops::tap::emit(&format!("p{pos}.L{li}.up_proj"), &up);
                crate::ops::tap::emit(&format!("p{pos}.L{li}.mlp_act"), &hid);
                crate::ops::tap::emit(&format!("p{pos}.L{li}.down_proj"), &down);
            }
            for i in 0..h {
                hs[i] = hs[i] + down[i];
            }
            #[cfg(feature = "layerdump")]
            crate::ops::tap::emit(&format!("p{pos}.L{li}.resid_mlp"), &hs);
        }

        // Spec 11.1.
        let hn = rmsnorm(&hs, self.final_norm, eps);
        let logits = matvec(self.lm_head, &hn, self.cfg.vocab_size, h);
        #[cfg(feature = "layerdump")]
        {
            crate::ops::tap::emit(&format!("p{pos}.final_norm"), &hn);
            crate::ops::tap::emit(&format!("p{pos}.logits"), &logits);
        }
        logits
    }

    /// Spec 3.4's decode protocol: prefill the prompt, then greedily emit
    /// exactly `gen_toks` tokens with no EOS check.
    ///
    /// Returns the generated ids and, for each of the `gen_toks` steps, the
    /// logit vector argmax was computed against -- which is precisely what
    /// spec 12.1 item 7a folds into the witness chain.
    pub fn decode(&self, prompt: &[u32], gen_toks: usize) -> Decode {
        let mut cache = KvCache::new(self.cfg.num_hidden_layers);
        let mut tokens: Vec<u32> = prompt.to_vec();
        let mut generated: Vec<u32> = Vec::with_capacity(gen_toks);
        let mut step_logits: Vec<Vec<f32>> = Vec::with_capacity(gen_toks);

        // Prefill positions 0..prompt.len()-2; their logits are not part of
        // any digest and are not consulted.
        for pos in 0..prompt.len().saturating_sub(1) {
            let _ = self.forward(tokens[pos], pos, &mut cache);
        }
        // The first decode step runs the *last* prompt token, so its logits
        // are the ones the first generated token is picked from.
        let mut pos = prompt.len() - 1;
        for _ in 0..gen_toks {
            let logits = self.forward(tokens[pos], pos, &mut cache);
            let next = argmax(&logits);
            step_logits.push(logits);
            generated.push(next);
            tokens.push(next);
            pos += 1;
        }
        Decode {
            generated,
            step_logits,
        }
    }
}

pub struct Decode {
    pub generated: Vec<u32>,
    pub step_logits: Vec<Vec<f32>>,
}

fn add_bias(x: &mut [f32], bias: Option<&[f32]>) {
    if let Some(b) = bias {
        for i in 0..x.len() {
            x[i] = x[i] + b[i];
        }
    }
}

/// Spec 7.4, rotate-half, in place after all outputs are computed.
fn rope_rotate(head: &mut [f32], cos_v: &[f32], sin_v: &[f32], half: usize) {
    let mut out = vec![0.0f32; head.len()];
    for i in 0..half {
        let x1 = head[i];
        let x2 = head[i + half];
        let t1 = x1 * cos_v[i];
        let t2 = (-x2) * sin_v[i];
        out[i] = t1 + t2;
        let t3 = x2 * cos_v[i];
        let t4 = x1 * sin_v[i];
        out[i + half] = t3 + t4;
    }
    head.copy_from_slice(&out);
}

/// Spec 7.1: `inv_freq[i] = exp_pinned(-( (2i/head_dim) * ln_pinned(theta) ))`,
/// with both transcendentals the pinned ones -- never a host `powf`.
pub fn build_inv_freq(cfg: &Config) -> Vec<f32> {
    let half = cfg.half();
    let head_dim = cfg.head_dim();
    let ln_theta = ln_pinned(cfg.rope_theta);
    let mut inv_freq = vec![0.0f32; half];
    for i in 0..half {
        let frac = ((2 * i) as f32) / (head_dim as f32);
        let neg_arg = -(frac * ln_theta);
        inv_freq[i] = exp_pinned(neg_arg);
    }
    inv_freq
}

/// Spec 3.5's memoization. Layout is one flat `Vec<f32>` per layer, appended
/// position by position, so `key`/`value` are slices into it and nothing is
/// copied per step.
pub struct KvCache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    widths: Vec<usize>,
}

impl KvCache {
    pub fn new(n_layers: usize) -> KvCache {
        KvCache {
            keys: vec![Vec::new(); n_layers],
            values: vec![Vec::new(); n_layers],
            widths: vec![0; n_layers],
        }
    }
    fn push(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        self.widths[layer] = k.len();
        self.keys[layer].extend_from_slice(k);
        self.values[layer].extend_from_slice(v);
    }
    fn len(&self, layer: usize) -> usize {
        if self.widths[layer] == 0 {
            0
        } else {
            self.keys[layer].len() / self.widths[layer]
        }
    }
    fn key(&self, layer: usize, pos: usize, kv_head: usize, head_dim: usize) -> &[f32] {
        let base = pos * self.widths[layer] + kv_head * head_dim;
        &self.keys[layer][base..base + head_dim]
    }
    fn value(&self, layer: usize, pos: usize, kv_head: usize, head_dim: usize) -> &[f32] {
        let base = pos * self.widths[layer] + kv_head * head_dim;
        &self.values[layer][base..base + head_dim]
    }
}

/// Unused elsewhere, but the compiler needs `exp_pinned`/`sin_pinned` paths
/// referenced from this module's imports; keeping this here documents that
/// the RoPE table is the only consumer of `sin`/`cos` in the whole forward
/// pass, and softmax the only consumer of `exp`.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::parse_config;

    const SMOL: &[u8] = br#"{"hidden_size":576,"intermediate_size":1536,
        "num_hidden_layers":30,"num_attention_heads":9,"num_key_value_heads":3,
        "vocab_size":49152,"rms_norm_eps":1e-05,"rope_theta":100000.0,
        "tie_word_embeddings":true,"hidden_act":"silu","torch_dtype":"bfloat16"}"#;

    /// Spec 7.2's pinned digest, reachable without any weights: it depends
    /// only on `head_dim` and `rope_theta`, so it isolates spec 6.2/6.5 and
    /// spec 7.1 from every other stage.
    #[test]
    fn inv_freq_reproduces_the_pinned_table_digest() {
        let cfg = parse_config(SMOL).unwrap();
        let inv_freq = build_inv_freq(&cfg);
        assert_eq!(inv_freq.len(), 32);
        assert_eq!(inv_freq[0], 1.0f32, "theta^0 must be exactly 1.0");
        let mut s = Sha256::new();
        for v in &inv_freq {
            s.update(&v.to_le_bytes());
        }
        assert_eq!(
            crate::hex::encode(&s.finalize()),
            "da9f6dcfde0425588815509e874515cdcd3d6b8818b6d0136590052e7bbf6f12"
        );
    }

    /// Spec 7.1's stated cross-check on the v0.1 literal it replaced.
    #[test]
    fn ln_pinned_of_the_pinned_theta_agrees_with_the_v0_1_literal() {
        let got = ln_pinned(100_000.0f32);
        let v0_1_literal = f32::from_bits(0x413834F1);
        let rel = crate::softfp::abs_f32(got - v0_1_literal) / v0_1_literal;
        assert!(rel <= 2e-6, "relative difference {rel} exceeds spec 7.1's 2e-6");
    }

    /// Spec 9.2 states this value is exact, and the statement is load-bearing:
    /// an approximate rsqrt here would perturb every score in the model.
    #[test]
    fn the_attention_scale_is_exactly_one_eighth() {
        assert_eq!(rsqrt(64.0f32), 0.125f32);
        assert_eq!(rsqrt(64.0f32).to_bits(), 0.125f32.to_bits());
    }

    /// Spec 7.4 at position 0: cos(0) = 1, sin(0) = 0, so the rotation is
    /// the identity and any sign or index error shows up immediately.
    #[test]
    fn rope_at_position_zero_is_the_identity() {
        let half = 4;
        let cos_v = vec![1.0f32; half];
        let sin_v = vec![0.0f32; half];
        let mut head: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
        let before = head.clone();
        rope_rotate(&mut head, &cos_v, &sin_v, half);
        assert_eq!(head, before);
    }

    /// Spec 5.1's own regression check, restated where it matters: the
    /// V-mix and score loops are this order.
    #[test]
    fn spec_5_1s_cancellation_check_holds() {
        assert_eq!(dot_seq(&[1e8, 1.0, -1e8], &[1.0, 1.0, 1.0]), 0.0f32);
    }
}
