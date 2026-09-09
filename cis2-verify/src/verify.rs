//! The top level: run the pinned decode from the artifacts, produce a
//! receipt, and check a receipt against a fresh run.
//!
//! This module does no file I/O -- it takes the three artifacts as byte
//! slices -- so the same code path serves the command-line tool, the tests,
//! and any embedding that has the bytes by other means.

use crate::config::{parse_config, Config};
use crate::hex;
use crate::mathpin;
use crate::model::Model;
use crate::receipt::Receipt;
use crate::safetensors;
use crate::sha256::sha256;
use crate::spec;
use crate::tokenizer::Tokenizer;
use crate::witness::{argmax_digest, witness_digest};
use alloc::string::{String, ToString};
use alloc::vec::Vec;

pub struct Artifacts<'a> {
    pub weights: &'a [u8],
    pub config: &'a [u8],
    pub tokenizer: &'a [u8],
}

/// Run the decode of spec 3.4 and return the receipt it justifies.
///
/// Spec 12.4 makes the same-host determinism check a MUST, so it is done
/// here rather than left to the caller: the whole decode runs twice in this
/// process and all three of the witness digest, the argmax digest and the
/// token ids must agree before anything is returned.
pub fn run(art: &Artifacts, prompt: &str, gen_toks: usize) -> Result<Receipt, String> {
    let weights_sha = sha256(art.weights);
    let config_sha = sha256(art.config);
    let tokenizer_sha = sha256(art.tokenizer);

    let cfg: Config = parse_config(art.config)?;
    let tok = Tokenizer::from_json(art.tokenizer)?;
    let prompt_token_ids = tok.encode(prompt)?;
    if prompt_token_ids.is_empty() {
        return Err("prompt tokenizes to zero tokens; there is nothing to decode".to_string());
    }

    let st = safetensors::load(art.weights)?;
    let model = Model::load(&st, cfg)?;

    let table = mathpin::table_digest();
    let inv_freq = model.inv_freq_table_digest();

    let d1 = model.decode(&prompt_token_ids, gen_toks);
    let w1 = witness_digest(
        &weights_sha, &tokenizer_sha, &config_sha, &table, &inv_freq,
        &prompt_token_ids, &d1.step_logits, &d1.generated,
    );
    let a1 = argmax_digest(&prompt_token_ids, &d1.generated);

    let d2 = model.decode(&prompt_token_ids, gen_toks);
    let w2 = witness_digest(
        &weights_sha, &tokenizer_sha, &config_sha, &table, &inv_freq,
        &prompt_token_ids, &d2.step_logits, &d2.generated,
    );
    let a2 = argmax_digest(&prompt_token_ids, &d2.generated);

    if w1 != w2 || a1 != a2 || d1.generated != d2.generated {
        return Err("spec 12.4 determinism check FAILED: two runs in one process disagreed"
            .to_string());
    }

    Ok(Receipt {
        spec_version: "0.3b".to_string(),
        weights_sha256: weights_sha,
        config_sha256: config_sha,
        tokenizer_sha256: tokenizer_sha,
        prompt: prompt.as_bytes().to_vec(),
        prompt_token_ids,
        gen_toks,
        dtype: "fp32".to_string(),
        table_digest: table,
        inv_freq_table_digest: inv_freq,
        generated_token_ids: d1.generated,
        argmax_digest: a1,
        witness_digest: w1,
    })
}

/// Check a receipt against the values CIS-2 v0.3b pins (spec 2.1, 3.3,
/// 6.6, 7.2, 12.1, 12.2, 13.1). Returns one line per disagreement.
///
/// This is a separate check from `verify`: it asks "is this the pinned test
/// vector", not "is this receipt self-consistent with these artifacts". A
/// run on a different prompt is a perfectly valid receipt and fails this.
pub fn check_against_pinned(r: &Receipt) -> Vec<String> {
    let mut bad = Vec::new();
    let mut cmp_hex = |name: &str, got: &[u8], want: &str| {
        let got = hex::encode(got);
        if got != want {
            bad.push(alloc::format!("{name}: got {got}, spec pins {want}"));
        }
    };
    cmp_hex("weights_sha256", &r.weights_sha256, spec::WEIGHTS_SHA256);
    cmp_hex("config_sha256", &r.config_sha256, spec::CONFIG_SHA256);
    cmp_hex("tokenizer_sha256", &r.tokenizer_sha256, spec::TOKENIZER_SHA256);
    cmp_hex("table_digest", &r.table_digest, spec::TABLE_DIGEST);
    cmp_hex("inv_freq_table_digest", &r.inv_freq_table_digest, spec::INV_FREQ_TABLE_DIGEST);
    cmp_hex("argmax_digest", &r.argmax_digest, spec::ARGMAX_DIGEST);
    cmp_hex("CIS2_REF", &r.witness_digest, spec::CIS2_REF);
    if r.prompt != spec::PROMPT.as_bytes() {
        bad.push(alloc::format!(
            "prompt: got {:?}, spec 3.2 pins {:?}",
            String::from_utf8_lossy(&r.prompt),
            spec::PROMPT
        ));
    }
    if r.prompt_token_ids != spec::PROMPT_TOKEN_IDS {
        bad.push(alloc::format!(
            "prompt_token_ids: got {:?}, spec 3.3 pins {:?}",
            r.prompt_token_ids,
            spec::PROMPT_TOKEN_IDS
        ));
    }
    if r.gen_toks != spec::GEN_TOKS {
        bad.push(alloc::format!(
            "gen_toks: got {}, spec 3.4 pins {}",
            r.gen_toks,
            spec::GEN_TOKS
        ));
    }
    if r.generated_token_ids != spec::GENERATED_TOKEN_IDS {
        bad.push(alloc::format!(
            "generated_token_ids: got {:?}, spec 13.1 pins {:?}",
            r.generated_token_ids,
            spec::GENERATED_TOKEN_IDS
        ));
    }
    if r.dtype != "fp32" {
        bad.push(alloc::format!("dtype: got {:?}, spec 1.1 pins \"fp32\"", r.dtype));
    }
    bad
}

/// Verify a claimed receipt by re-deriving it from the artifacts.
///
/// The receipt's own artifact hashes are checked against the bytes it was
/// handed *before* anything else runs, because a receipt that names
/// different artifacts than the ones on disk is answering a different
/// question -- and reporting a digest mismatch in that case would be
/// misleading about which half is wrong.
pub fn verify(art: &Artifacts, claimed: &Receipt) -> Vec<String> {
    let mut bad = Vec::new();
    let pairs: [(&str, [u8; 32], [u8; 32]); 3] = [
        ("model.safetensors", sha256(art.weights), claimed.weights_sha256),
        ("config.json", sha256(art.config), claimed.config_sha256),
        ("tokenizer.json", sha256(art.tokenizer), claimed.tokenizer_sha256),
    ];
    for (name, got, want) in pairs {
        if got != want {
            bad.push(alloc::format!(
                "{name}: on-disk sha256 {} does not match the receipt's {}",
                hex::encode(&got),
                hex::encode(&want)
            ));
        }
    }
    if !bad.is_empty() {
        return bad;
    }

    let prompt = match core::str::from_utf8(&claimed.prompt) {
        Ok(p) => p,
        Err(_) => {
            bad.push("receipt: prompt-hex is not valid UTF-8".to_string());
            return bad;
        }
    };
    let fresh = match run(art, prompt, claimed.gen_toks) {
        Ok(r) => r,
        Err(e) => {
            bad.push(e);
            return bad;
        }
    };

    let mut cmp = |name: &str, got: &[u8], want: &[u8]| {
        if got != want {
            bad.push(alloc::format!(
                "{name}: replay gives {}, receipt claims {}",
                hex::encode(got),
                hex::encode(want)
            ));
        }
    };
    cmp("table_digest", &fresh.table_digest, &claimed.table_digest);
    cmp(
        "inv_freq_table_digest",
        &fresh.inv_freq_table_digest,
        &claimed.inv_freq_table_digest,
    );
    cmp("argmax_digest", &fresh.argmax_digest, &claimed.argmax_digest);
    cmp("witness_digest", &fresh.witness_digest, &claimed.witness_digest);
    if fresh.prompt_token_ids != claimed.prompt_token_ids {
        bad.push(alloc::format!(
            "prompt_token_ids: replay gives {:?}, receipt claims {:?}",
            fresh.prompt_token_ids,
            claimed.prompt_token_ids
        ));
    }
    if fresh.generated_token_ids != claimed.generated_token_ids {
        bad.push(alloc::format!(
            "generated_token_ids: replay gives {:?}, receipt claims {:?}",
            fresh.generated_token_ids,
            claimed.generated_token_ids
        ));
    }
    if fresh.dtype != claimed.dtype {
        bad.push(alloc::format!(
            "dtype: replay is {:?}, receipt claims {:?}",
            fresh.dtype, claimed.dtype
        ));
    }
    bad
}
