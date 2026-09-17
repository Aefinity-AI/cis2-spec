//! The receipt: a canonical text rendering of everything spec 12 pins for
//! one run, and a strict parser for it.
//!
//! Strictness here is not fussiness. A receipt whose parser silently
//! ignores unknown keys, duplicate keys, or trailing junk is not canonical:
//! many distinct byte strings then share one verdict, and attacker-chosen
//! text can ride inside a file that reports PASS. So this parser rejects
//! any line whose key it does not know, any repeated key, any missing key,
//! and any value that is not exactly the shape its key requires. The tests
//! at the bottom are one mutation per rule.

use crate::hex;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

pub const HEADER: &str = "CIS-2-RECEIPT v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Receipt {
    pub spec_version: String,
    pub weights_sha256: [u8; 32],
    pub config_sha256: [u8; 32],
    pub tokenizer_sha256: [u8; 32],
    /// The prompt as raw bytes, hex-encoded in the file so that a prompt
    /// containing a newline or a space cannot change the line structure.
    pub prompt: Vec<u8>,
    pub prompt_token_ids: Vec<u32>,
    pub gen_toks: usize,
    pub dtype: String,
    pub table_digest: [u8; 32],
    pub inv_freq_table_digest: [u8; 32],
    pub generated_token_ids: Vec<u32>,
    pub argmax_digest: [u8; 32],
    pub witness_digest: [u8; 32],
}

const KEYS: [&str; 13] = [
    "spec-version",
    "weights-sha256",
    "config-sha256",
    "tokenizer-sha256",
    "prompt-hex",
    "prompt-token-ids",
    "gen-toks",
    "dtype",
    "table-digest",
    "inv-freq-table-digest",
    "generated-token-ids",
    "argmax-digest",
    "witness-digest",
];

impl Receipt {
    /// The canonical rendering. `parse(render(r)) == r` for every receipt,
    /// and `render(parse(text)) == text` for every text this parser
    /// accepts -- the two round-trip tests below pin both directions, which
    /// together are what "canonical" means.
    pub fn render(&self) -> String {
        let mut s = String::new();
        s.push_str(HEADER);
        s.push('\n');
        s.push_str(&alloc::format!("spec-version {}\n", self.spec_version));
        s.push_str(&alloc::format!("weights-sha256 {}\n", hex::encode(&self.weights_sha256)));
        s.push_str(&alloc::format!("config-sha256 {}\n", hex::encode(&self.config_sha256)));
        s.push_str(&alloc::format!(
            "tokenizer-sha256 {}\n",
            hex::encode(&self.tokenizer_sha256)
        ));
        s.push_str(&alloc::format!("prompt-hex {}\n", hex::encode(&self.prompt)));
        s.push_str(&alloc::format!("prompt-token-ids {}\n", csv(&self.prompt_token_ids)));
        s.push_str(&alloc::format!("gen-toks {}\n", self.gen_toks));
        s.push_str(&alloc::format!("dtype {}\n", self.dtype));
        s.push_str(&alloc::format!("table-digest {}\n", hex::encode(&self.table_digest)));
        s.push_str(&alloc::format!(
            "inv-freq-table-digest {}\n",
            hex::encode(&self.inv_freq_table_digest)
        ));
        s.push_str(&alloc::format!(
            "generated-token-ids {}\n",
            csv(&self.generated_token_ids)
        ));
        s.push_str(&alloc::format!("argmax-digest {}\n", hex::encode(&self.argmax_digest)));
        s.push_str(&alloc::format!("witness-digest {}\n", hex::encode(&self.witness_digest)));
        s
    }

    pub fn parse(text: &str) -> Result<Receipt, String> {
        let mut lines = text.lines();
        // The header must be the first line, byte for byte. A receipt that
        // can be relabelled is a receipt whose format is negotiable.
        match lines.next() {
            Some(h) if h == HEADER => {}
            Some(h) => {
                return Err(alloc::format!(
                    "receipt: first line is {h:?}, expected {HEADER:?}"
                ))
            }
            None => return Err("receipt: empty".to_string()),
        }

        let mut seen: Vec<&str> = Vec::new();
        let mut spec_version = None;
        let mut weights = None;
        let mut config = None;
        let mut tokenizer = None;
        let mut prompt = None;
        let mut prompt_ids = None;
        let mut gen_toks = None;
        let mut dtype = None;
        let mut table = None;
        let mut inv_freq = None;
        let mut generated = None;
        let mut argmax = None;
        let mut witness = None;

        for (i, line) in lines.enumerate() {
            let line_no = i + 2;
            if line.is_empty() {
                return Err(alloc::format!("receipt: blank line {line_no}"));
            }
            let (key, value) = line
                .split_once(' ')
                .ok_or_else(|| alloc::format!("receipt: line {line_no} has no value"))?;
            if !KEYS.contains(&key) {
                return Err(alloc::format!("receipt: unknown key {key:?} on line {line_no}"));
            }
            if seen.contains(&key) {
                return Err(alloc::format!("receipt: duplicate key {key:?} on line {line_no}"));
            }
            seen.push(key);
            match key {
                "spec-version" => spec_version = Some(word(key, value)?.to_string()),
                "weights-sha256" => weights = Some(digest(key, value)?),
                "config-sha256" => config = Some(digest(key, value)?),
                "tokenizer-sha256" => tokenizer = Some(digest(key, value)?),
                "prompt-hex" => {
                    prompt = Some(
                        hex::decode(word(key, value)?)
                            .map_err(|e| alloc::format!("receipt: prompt-hex: {e}"))?,
                    )
                }
                "prompt-token-ids" => prompt_ids = Some(ids(key, value)?),
                "gen-toks" => {
                    gen_toks = Some(
                        word(key, value)?
                            .parse::<usize>()
                            .map_err(|_| "receipt: gen-toks is not a non-negative integer")?,
                    )
                }
                "dtype" => dtype = Some(word(key, value)?.to_string()),
                "table-digest" => table = Some(digest(key, value)?),
                "inv-freq-table-digest" => inv_freq = Some(digest(key, value)?),
                "generated-token-ids" => generated = Some(ids(key, value)?),
                "argmax-digest" => argmax = Some(digest(key, value)?),
                "witness-digest" => witness = Some(digest(key, value)?),
                _ => unreachable!("key was allowlisted above"),
            }
        }

        let missing = |name: &str| alloc::format!("receipt: missing {name} line");
        let r = Receipt {
            spec_version: spec_version.ok_or_else(|| missing("spec-version"))?,
            weights_sha256: weights.ok_or_else(|| missing("weights-sha256"))?,
            config_sha256: config.ok_or_else(|| missing("config-sha256"))?,
            tokenizer_sha256: tokenizer.ok_or_else(|| missing("tokenizer-sha256"))?,
            prompt: prompt.ok_or_else(|| missing("prompt-hex"))?,
            prompt_token_ids: prompt_ids.ok_or_else(|| missing("prompt-token-ids"))?,
            gen_toks: gen_toks.ok_or_else(|| missing("gen-toks"))?,
            dtype: dtype.ok_or_else(|| missing("dtype"))?,
            table_digest: table.ok_or_else(|| missing("table-digest"))?,
            inv_freq_table_digest: inv_freq.ok_or_else(|| missing("inv-freq-table-digest"))?,
            generated_token_ids: generated.ok_or_else(|| missing("generated-token-ids"))?,
            argmax_digest: argmax.ok_or_else(|| missing("argmax-digest"))?,
            witness_digest: witness.ok_or_else(|| missing("witness-digest"))?,
        };
        // A receipt claiming 16 generated tokens and listing 15 is not a
        // receipt with a small error in it; it is a receipt about a
        // different run.
        if r.generated_token_ids.len() != r.gen_toks {
            return Err(alloc::format!(
                "receipt: gen-toks is {} but generated-token-ids lists {}",
                r.gen_toks,
                r.generated_token_ids.len()
            ));
        }
        Ok(r)
    }
}

fn csv(ids: &[u32]) -> String {
    let mut s = String::new();
    for (i, id) in ids.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&alloc::format!("{id}"));
    }
    s
}

/// A value that must be a single token with no interior or trailing
/// whitespace -- so `weights-sha256 <hex> extra` is rejected rather than
/// truncated to the hex.
fn word<'a>(key: &str, value: &'a str) -> Result<&'a str, String> {
    if value.is_empty() || value.contains(char::is_whitespace) {
        return Err(alloc::format!("receipt: {key} value must be a single token"));
    }
    Ok(value)
}

fn digest(key: &str, value: &str) -> Result<[u8; 32], String> {
    let bytes = hex::decode(word(key, value)?).map_err(|e| alloc::format!("receipt: {key}: {e}"))?;
    if bytes.len() != 32 {
        return Err(alloc::format!(
            "receipt: {key} is {} bytes, expected 32",
            bytes.len()
        ));
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
}

fn ids(key: &str, value: &str) -> Result<Vec<u32>, String> {
    let value = word(key, value)?;
    let mut out = Vec::new();
    for part in value.split(',') {
        // Rejecting an empty element also rejects a trailing comma, a
        // leading comma, and a doubled comma -- three more ways to write
        // "the same" list.
        if part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()) {
            return Err(alloc::format!("receipt: {key} has a malformed element {part:?}"));
        }
        if part.len() > 1 && part.starts_with('0') {
            return Err(alloc::format!(
                "receipt: {key} element {part:?} has a leading zero"
            ));
        }
        out.push(
            part.parse::<u32>()
                .map_err(|_| alloc::format!("receipt: {key} element {part:?} does not fit in u32"))?,
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn sample() -> Receipt {
        Receipt {
            spec_version: "0.3b".into(),
            weights_sha256: [1u8; 32],
            config_sha256: [2u8; 32],
            tokenizer_sha256: [3u8; 32],
            prompt: b"Once upon a time".to_vec(),
            prompt_token_ids: vec![6403, 1980, 253, 655],
            gen_toks: 2,
            dtype: "fp32".into(),
            table_digest: [4u8; 32],
            inv_freq_table_digest: [5u8; 32],
            generated_token_ids: vec![28, 665],
            argmax_digest: [6u8; 32],
            witness_digest: [7u8; 32],
        }
    }

    #[test]
    fn render_then_parse_is_the_identity() {
        let r = sample();
        assert_eq!(Receipt::parse(&r.render()).unwrap(), r);
    }

    #[test]
    fn parse_then_render_is_the_identity_so_the_form_is_canonical() {
        let text = sample().render();
        assert_eq!(Receipt::parse(&text).unwrap().render(), text);
    }

    /// One test per way a non-canonical parser leaks. Each mutation below
    /// is a distinct byte string that a lax parser would map to the same
    /// receipt.
    #[test]
    fn every_non_canonical_spelling_is_rejected() {
        let base = sample().render();
        let mutants: Vec<String> = vec![
            base.replace("CIS-2-RECEIPT v1", "CIS-2-RECEIPT v2"),
            base.replace("CIS-2-RECEIPT v1", "cis-2-receipt v1"),
            alloc::format!("junk\n{base}"),
            alloc::format!("{base}unknown-key value\n"),
            alloc::format!("{base}dtype fp32\n"),
            base.replace("dtype fp32\n", "dtype fp32 extra\n"),
            base.replace("gen-toks 2\n", "gen-toks 2x\n"),
            base.replace("prompt-token-ids 6403,1980,253,655", "prompt-token-ids 6403,1980,253,655,"),
            base.replace("prompt-token-ids 6403,1980,253,655", "prompt-token-ids 06403,1980,253,655"),
            base.replace("generated-token-ids 28,665", "generated-token-ids 28"),
            base.replace("dtype fp32\n", ""),
            base.replace("\ndtype fp32", "\n\ndtype fp32"),
            base.replace("table-digest 0404", "table-digest 04040404"),
        ];
        for m in mutants {
            assert!(
                Receipt::parse(&m).is_err(),
                "accepted a non-canonical receipt:\n{m}"
            );
        }
    }

    #[test]
    fn the_positive_control_still_parses() {
        assert!(Receipt::parse(&sample().render()).is_ok());
    }
}
