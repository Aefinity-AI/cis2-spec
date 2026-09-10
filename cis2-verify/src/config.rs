//! Spec 2: the model artifact's `config.json`, and the derived shapes the
//! rest of the crate works in.
//!
//! Everything here is read from the file rather than hardcoded, and then
//! checked against what spec 2.2 pins. The distinction matters: a verifier
//! that hardcodes `hidden = 576` cannot tell a wrong config from a right
//! one, and the spec's own conformance argument is that an implementation
//! reads the artifact and fails loudly when it is not the artifact the
//! receipt claims.

use crate::json::{parse, Json};
use alloc::string::String;
use alloc::vec::Vec;

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub rope_interleaved: bool,
    pub torch_dtype: String,
    pub hidden_act: String,
}

impl Config {
    /// `head_dim = hidden_size / num_attention_heads` (spec 2.2 derived).
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    /// GQA group size: how many query heads share one KV head (spec 9).
    pub fn group(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
    /// Half a head, the width of the RoPE rotation tables (spec 7.3).
    pub fn half(&self) -> usize {
        self.head_dim() / 2
    }
    /// Width of the K and V projections (spec 9.1).
    pub fn kv_width(&self) -> usize {
        self.num_key_value_heads * self.head_dim()
    }
}

fn req_usize(root: &Json, key: &str) -> Result<usize, String> {
    root.get(key)
        .and_then(|v| v.as_usize())
        .ok_or_else(|| alloc::format!("config.json: missing or non-integer field {key:?}"))
}

/// A numeric field this crate actually uses. A literal the JSON reader
/// declined to convert (see `json::Json::Num`) is reported as such rather
/// than treated as absent, because "we will not round this for you" and
/// "this field is missing" are different problems for whoever has to fix
/// the config.
fn req_f32(root: &Json, key: &str) -> Result<f32, String> {
    match root.get(key) {
        None => Err(alloc::format!("config.json: missing field {key:?}")),
        Some(v) => match (v.as_f32(), v.num_text()) {
            (Some(x), _) => Ok(x),
            (None, Some(text)) => Err(alloc::format!(
                "config.json: {key} = {text} has more precision than this verifier will \
                 round; write it with at most 16 significant digits"
            )),
            (None, None) => Err(alloc::format!("config.json: field {key:?} is not a number")),
        },
    }
}

pub fn parse_config(bytes: &[u8]) -> Result<Config, String> {
    let root = parse(bytes).map_err(|e| alloc::format!("config.json: {e}"))?;

    let hidden_size = req_usize(&root, "hidden_size")?;
    let intermediate_size = req_usize(&root, "intermediate_size")?;
    let num_hidden_layers = req_usize(&root, "num_hidden_layers")?;
    let num_attention_heads = req_usize(&root, "num_attention_heads")?;
    let num_key_value_heads = req_usize(&root, "num_key_value_heads")?;
    let vocab_size = req_usize(&root, "vocab_size")?;

    let rms_norm_eps = req_f32(&root, "rms_norm_eps")?;
    let rope_theta = req_f32(&root, "rope_theta")?;

    // Spec 2.2: absent means the HuggingFace default, which for this family
    // is `false` for `rope_interleaved` and `true` for `tie_word_embeddings`.
    // Both are stated explicitly by SmolLM2-135M's config; defaulting is for
    // configs that omit them, and the default chosen is the one spec 2.2
    // names.
    let tie_word_embeddings = root
        .get("tie_word_embeddings")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let rope_interleaved = root
        .get("rope_interleaved")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let torch_dtype = root
        .get("torch_dtype")
        .and_then(|v| v.as_str())
        .unwrap_or("bfloat16")
        .into();
    let hidden_act = root
        .get("hidden_act")
        .and_then(|v| v.as_str())
        .unwrap_or("silu")
        .into();

    let cfg = Config {
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        vocab_size,
        rms_norm_eps,
        rope_theta,
        tie_word_embeddings,
        rope_interleaved,
        torch_dtype,
        hidden_act,
    };
    cfg.check()?;
    Ok(cfg)
}

impl Config {
    /// The structural preconditions the rest of this crate relies on. These
    /// are not "is this SmolLM2-135M" checks (spec 2.1's artifact hashes do
    /// that job, in `artifacts.rs`); they are the invariants without which
    /// the spec's own formulas are undefined.
    pub fn check(&self) -> Result<(), String> {
        if self.num_attention_heads == 0 || self.num_key_value_heads == 0 {
            return Err("config.json: head counts must be non-zero".into());
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err("config.json: hidden_size is not a multiple of num_attention_heads, \
                        so head_dim (spec 2.2) is undefined"
                .into());
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err("config.json: num_attention_heads is not a multiple of \
                        num_key_value_heads, so the GQA group (spec 9) is undefined"
                .into());
        }
        if self.head_dim() % 2 != 0 {
            return Err("config.json: head_dim is odd, so RoPE's half-split (spec 7.4) \
                        is undefined"
                .into());
        }
        // Spec 1.1/4: this crate's whole loader is the bf16 widening of
        // spec 4.1. A checkpoint stored in any other dtype is out of scope,
        // and saying so is better than widening 16 bits of an f16 as if it
        // were bf16.
        if self.torch_dtype != "bfloat16" {
            return Err(alloc::format!(
                "config.json: torch_dtype is {:?}; spec 4 defines widening only for bfloat16",
                self.torch_dtype
            ));
        }
        // Spec 10 is SwiGLU with spec 6.4's pinned SiLU. Another activation
        // would be a different model, silently.
        if self.hidden_act != "silu" {
            return Err(alloc::format!(
                "config.json: hidden_act is {:?}; spec 10 pins silu",
                self.hidden_act
            ));
        }
        // Spec 7.4 is the rotate-half convention. The interleaved variant is
        // a different rotation and this crate does not implement it.
        if self.rope_interleaved {
            return Err("config.json: rope_interleaved is true; spec 7.4 pins the \
                        rotate-half convention (rope_interleaved = false)"
                .into());
        }
        Ok(())
    }

    /// Spec 2.2's pinned SmolLM2-135M shape, as a separate opt-in check so
    /// the parser above stays honest about what it read.
    pub fn expect_smollm2_135m(&self) -> Result<(), String> {
        let want = [
            ("hidden_size", self.hidden_size, 576usize),
            ("intermediate_size", self.intermediate_size, 1536),
            ("num_hidden_layers", self.num_hidden_layers, 30),
            ("num_attention_heads", self.num_attention_heads, 9),
            ("num_key_value_heads", self.num_key_value_heads, 3),
            ("vocab_size", self.vocab_size, 49152),
        ];
        let mut bad: Vec<String> = Vec::new();
        for (name, got, exp) in want {
            if got != exp {
                bad.push(alloc::format!("{name}={got} (spec 2.2 pins {exp})"));
            }
        }
        if self.rms_norm_eps.to_bits() != crate::mathpin::EPS_F32_BITS {
            bad.push(alloc::format!(
                "rms_norm_eps bits=0x{:08X} (spec 2.3 pins EPS_F32=0x{:08X})",
                self.rms_norm_eps.to_bits(),
                crate::mathpin::EPS_F32_BITS
            ));
        }
        if !self.tie_word_embeddings {
            bad.push("tie_word_embeddings=false (spec 2.2 pins true for this model)".into());
        }
        if bad.is_empty() {
            Ok(())
        } else {
            Err(alloc::format!("config.json does not match spec 2.2: {}", bad.join("; ")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMOL: &[u8] = br#"{"hidden_size":576,"intermediate_size":1536,
        "num_hidden_layers":30,"num_attention_heads":9,"num_key_value_heads":3,
        "vocab_size":49152,"rms_norm_eps":1e-05,"rope_theta":100000.0,
        "tie_word_embeddings":true,"hidden_act":"silu","torch_dtype":"bfloat16"}"#;

    #[test]
    fn reads_the_pinned_shape_and_derives_spec_2_2s_numbers() {
        let c = parse_config(SMOL).unwrap();
        c.expect_smollm2_135m().unwrap();
        assert_eq!(c.head_dim(), 64);
        assert_eq!(c.group(), 3);
        assert_eq!(c.half(), 32);
        assert_eq!(c.kv_width(), 192);
        assert_eq!(c.rms_norm_eps.to_bits(), 0x3727C5AC);
        assert_eq!(c.rope_theta, 100000.0f32);
    }

    #[test]
    fn a_shape_the_spec_leaves_undefined_is_refused() {
        let bad = br#"{"hidden_size":577,"intermediate_size":1536,"num_hidden_layers":30,
            "num_attention_heads":9,"num_key_value_heads":3,"vocab_size":49152,
            "rms_norm_eps":1e-05,"rope_theta":100000.0}"#;
        assert!(parse_config(bad).is_err());
    }

    #[test]
    fn a_non_bf16_or_non_silu_checkpoint_is_refused() {
        for bad in [
            br#"{"hidden_size":576,"intermediate_size":1536,"num_hidden_layers":30,"num_attention_heads":9,"num_key_value_heads":3,"vocab_size":49152,"rms_norm_eps":1e-05,"rope_theta":100000.0,"torch_dtype":"float16"}"#.as_slice(),
            br#"{"hidden_size":576,"intermediate_size":1536,"num_hidden_layers":30,"num_attention_heads":9,"num_key_value_heads":3,"vocab_size":49152,"rms_norm_eps":1e-05,"rope_theta":100000.0,"hidden_act":"gelu"}"#.as_slice(),
        ] {
            assert!(parse_config(bad).is_err());
        }
    }

    #[test]
    fn a_wrong_eps_is_reported_not_silently_accepted() {
        let bad = br#"{"hidden_size":576,"intermediate_size":1536,"num_hidden_layers":30,
            "num_attention_heads":9,"num_key_value_heads":3,"vocab_size":49152,
            "rms_norm_eps":1e-06,"rope_theta":100000.0}"#;
        let c = parse_config(bad).unwrap();
        assert!(c.expect_smollm2_135m().unwrap_err().contains("EPS_F32"));
    }
}
