//! `check`: what a receipt can be held to **without the weights**.
//!
//! `verify` re-runs the decode, so it needs `model.safetensors` --- for the
//! pinned checkpoint, 270 MB and a full forward pass per generated token.
//! That is the right answer when the question is "did this run happen", and
//! the wrong tool when the question is "is this receipt worth running".
//!
//! Most of a CIS-2 receipt is not about the weights at all. Four of its
//! thirteen fields are recomputable from nothing (spec 6.6's table), from
//! `config.json` alone (spec 7.2's `inv_freq` table), or from
//! `tokenizer.json` alone (spec 3.3's prompt token ids); its canonical form
//! is checkable from the bytes on the page; and when the three artifact
//! hashes are the ones spec 2.1 pins, *every remaining field* is pinned by
//! spec 13.1 and needs no artifacts either.
//!
//! So `check` is a cheap tier, and its value depends entirely on it being
//! honest about its own limits. Every check returns one of three verdicts,
//! and a `Skip` is reported as loudly as a `Fail`: a receipt that "passes"
//! `check` while five of its fields were skipped has been told apart from a
//! receipt that passes with none skipped. `residual` names, field by field,
//! exactly what `check` did not establish --- so the caller knows what a
//! subsequent `verify` would still buy them.
//!
//! What `check` can never do, at any tier: establish that the logits behind
//! `witness-digest` were produced by running the model. Only `verify` does
//! that. A receipt whose artifact hashes are unknown to this crate and
//! whose weights are absent is, after `check`, a well-formed claim about
//! arithmetic --- nothing more.

use crate::config::parse_config;
use crate::hex;
use crate::mathpin;
use crate::model::build_inv_freq;
use crate::receipt::Receipt;
use crate::sha256::sha256;
use crate::spec;
use crate::tokenizer::Tokenizer;
use crate::verify::check_against_pinned;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// The optional side artifacts. Both are small: `config.json` is under a
/// kilobyte and `tokenizer.json` a few megabytes, against the weights'
/// 270 MB. Neither is required.
#[derive(Default)]
pub struct Evidence<'a> {
    pub config: Option<&'a [u8]>,
    pub tokenizer: Option<&'a [u8]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Recomputed and it agreed.
    Pass,
    /// Recomputed and it disagreed. The receipt is wrong.
    Fail,
    /// Not established here. Says nothing about whether it is true.
    Skip,
}

impl Status {
    pub fn label(self) -> &'static str {
        match self {
            Status::Pass => "PASS",
            Status::Fail => "FAIL",
            Status::Skip => "SKIP",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Check {
    pub name: &'static str,
    pub status: Status,
    /// Always populated, including on `Pass`: on a skip it says what would
    /// be needed, and on a pass it says what the agreement rests on.
    pub detail: String,
}

fn c(name: &'static str, status: Status, detail: String) -> Check {
    Check {
        name,
        status,
        detail,
    }
}

/// The receipt fields `check` can reach, in receipt order. Used by
/// `residual` to name what is left.
const CHECKABLE: [&str; 6] = [
    "config-sha256",
    "tokenizer-sha256",
    "prompt-token-ids",
    "table-digest",
    "inv-freq-table-digest",
    "weights-sha256",
];

/// Audit a receipt from its text plus whatever small artifacts are to hand.
///
/// Returns one `Check` per question asked, in a fixed order, whether or not
/// it could be answered. An empty return is impossible; a parse failure is
/// reported as a single failing check rather than as an error, so that a
/// caller rendering a report has one code path.
pub fn check(text: &str, ev: &Evidence) -> Vec<Check> {
    let mut out = Vec::new();

    let r = match Receipt::parse(text) {
        Ok(r) => r,
        Err(e) => {
            out.push(c("parse", Status::Fail, e));
            return out;
        }
    };
    out.push(c(
        "parse",
        Status::Pass,
        format!("13 keys, spec-version {}", r.spec_version),
    ));

    // Canonical form. The parser already rejects unknown keys, duplicates,
    // blank lines and malformed values; this catches the remaining way two
    // byte strings can mean one receipt --- a different key *order*, or a
    // missing trailing newline.
    let rendered = r.render();
    if rendered == text {
        out.push(c(
            "canonical-form",
            Status::Pass,
            "the file is byte-identical to the canonical rendering of what it parses to"
                .to_string(),
        ));
    } else {
        out.push(c(
            "canonical-form",
            Status::Fail,
            format!(
                "parses, but is not the canonical rendering ({} bytes on disk, {} canonical) \
                 --- two distinct files would share one verdict",
                text.len(),
                rendered.len()
            ),
        ));
    }

    // dtype is the one free-text field with a normative value (spec 1.1).
    if r.dtype == "fp32" {
        out.push(c(
            "dtype",
            Status::Pass,
            "fp32, as spec 1.1 requires".to_string(),
        ));
    } else {
        out.push(c(
            "dtype",
            Status::Fail,
            format!("{:?}; spec 1.1 pins \"fp32\"", r.dtype),
        ));
    }

    // The prompt must round-trip through UTF-8 for §3 to apply to it at all.
    let prompt_str = core::str::from_utf8(&r.prompt).ok();
    match prompt_str {
        Some(p) => out.push(c(
            "prompt-hex",
            Status::Pass,
            format!("{} bytes, valid UTF-8: {p:?}", r.prompt.len()),
        )),
        None => out.push(c(
            "prompt-hex",
            Status::Fail,
            "not valid UTF-8; spec 3.2 takes a text prompt".to_string(),
        )),
    }

    // Spec 6.6's table needs nothing at all: it is a property of the
    // arithmetic this binary implements, not of any checkpoint.
    let table = hex::encode(&mathpin::table_digest());
    let claimed_table = hex::encode(&r.table_digest);
    if claimed_table == table {
        out.push(c(
            "table-digest",
            Status::Pass,
            format!("recomputed from spec 6.6 with no artifacts: {table}"),
        ));
    } else {
        out.push(c(
            "table-digest",
            Status::Fail,
            format!("receipt claims {claimed_table}; this build recomputes {table}"),
        ));
    }

    out.push(check_config(&r, ev));
    out.push(check_tokenizer(&r, ev, prompt_str));
    out.push(check_pinned(&r));

    out
}

/// `config.json` decides the `inv_freq` table (spec 7.1/7.2) and nothing
/// else this receipt records. Two ways to reach it: the file itself, or the
/// observation that its hash is the one spec 2.1 pins, in which case spec
/// 7.2 already pins the answer.
fn check_config(r: &Receipt, ev: &Evidence) -> Check {
    let claimed = hex::encode(&r.inv_freq_table_digest);
    let cfg_sha = hex::encode(&r.config_sha256);
    match ev.config {
        Some(bytes) => {
            let on_disk = hex::encode(&sha256(bytes));
            if on_disk != cfg_sha {
                return c(
                    "inv-freq-table-digest",
                    Status::Fail,
                    format!(
                        "the config.json supplied hashes to {on_disk}, but the receipt names \
                         {cfg_sha} --- these are receipts about different checkpoints"
                    ),
                );
            }
            let cfg = match parse_config(bytes) {
                Ok(cfg) => cfg,
                Err(e) => {
                    return c(
                        "inv-freq-table-digest",
                        Status::Fail,
                        format!("config.json does not parse under spec 2.2: {e}"),
                    )
                }
            };
            let mut s = crate::sha256::Sha256::new();
            for v in build_inv_freq(&cfg) {
                s.update(&v.to_le_bytes());
            }
            let got = hex::encode(&s.finalize());
            if got == claimed {
                c(
                    "inv-freq-table-digest",
                    Status::Pass,
                    format!("rebuilt from the supplied config.json alone (spec 7.1/7.2): {got}"),
                )
            } else {
                c(
                    "inv-freq-table-digest",
                    Status::Fail,
                    format!(
                        "receipt claims {claimed}; rebuilt from its own config.json gives {got}"
                    ),
                )
            }
        }
        None if cfg_sha == spec::CONFIG_SHA256 => {
            if claimed == spec::INV_FREQ_TABLE_DIGEST {
                c(
                    "inv-freq-table-digest",
                    Status::Pass,
                    format!(
                        "no config.json supplied, but config-sha256 is the one spec 2.1 pins, \
                         and spec 7.2 pins its table: {claimed}"
                    ),
                )
            } else {
                c(
                    "inv-freq-table-digest",
                    Status::Fail,
                    format!(
                        "config-sha256 is the checkpoint spec 2.1 pins, whose table spec 7.2 \
                         pins as {}; the receipt claims {claimed}",
                        spec::INV_FREQ_TABLE_DIGEST
                    ),
                )
            }
        }
        None => c(
            "inv-freq-table-digest",
            Status::Skip,
            format!(
                "config-sha256 {cfg_sha} is not the checkpoint spec 2.1 pins and no config.json \
                 was supplied --- pass --config to establish this field"
            ),
        ),
    }
}

/// `tokenizer.json` decides `prompt-token-ids` (spec 3.3) and nothing else
/// here. Same two routes as the config.
fn check_tokenizer(r: &Receipt, ev: &Evidence, prompt: Option<&str>) -> Check {
    let tok_sha = hex::encode(&r.tokenizer_sha256);
    let prompt = match prompt {
        Some(p) => p,
        None => {
            return c(
                "prompt-token-ids",
                Status::Skip,
                "the prompt is not valid UTF-8, so spec 3 cannot be applied to it".to_string(),
            )
        }
    };
    match ev.tokenizer {
        Some(bytes) => {
            let on_disk = hex::encode(&sha256(bytes));
            if on_disk != tok_sha {
                return c(
                    "prompt-token-ids",
                    Status::Fail,
                    format!(
                        "the tokenizer.json supplied hashes to {on_disk}, but the receipt names \
                         {tok_sha}"
                    ),
                );
            }
            let t = match Tokenizer::from_json(bytes) {
                Ok(t) => t,
                Err(e) => {
                    return c(
                        "prompt-token-ids",
                        Status::Fail,
                        format!("tokenizer.json is not one spec 3.1 accepts: {e}"),
                    )
                }
            };
            match t.encode(prompt) {
                Ok(ids) if ids == r.prompt_token_ids => c(
                    "prompt-token-ids",
                    Status::Pass,
                    format!(
                        "re-encoded the prompt with the supplied tokenizer.json alone (spec 3.3): \
                         {} tokens",
                        ids.len()
                    ),
                ),
                Ok(ids) => c(
                    "prompt-token-ids",
                    Status::Fail,
                    format!(
                        "receipt claims {:?}; re-encoding gives {ids:?}",
                        r.prompt_token_ids
                    ),
                ),
                Err(e) => c(
                    "prompt-token-ids",
                    Status::Fail,
                    format!("re-encoding failed: {e}"),
                ),
            }
        }
        None if tok_sha == spec::TOKENIZER_SHA256 && prompt == spec::PROMPT => {
            if r.prompt_token_ids == spec::PROMPT_TOKEN_IDS {
                c(
                    "prompt-token-ids",
                    Status::Pass,
                    "no tokenizer.json supplied, but tokenizer-sha256 and the prompt are the ones \
                     spec 2.1/3.2 pin, and spec 3.3 pins their encoding"
                        .to_string(),
                )
            } else {
                c(
                    "prompt-token-ids",
                    Status::Fail,
                    format!(
                        "spec 3.3 pins {:?} for this tokenizer and prompt; the receipt claims {:?}",
                        spec::PROMPT_TOKEN_IDS,
                        r.prompt_token_ids
                    ),
                )
            }
        }
        None => c(
            "prompt-token-ids",
            Status::Skip,
            "no tokenizer.json supplied, and this tokenizer/prompt pair is not the one spec 3.3 \
             pins --- pass --tokenizer to establish this field"
                .to_string(),
        ),
    }
}

/// The one case where the weights are unnecessary even for the witness: if
/// the receipt names the three artifacts spec 2.1 pins and the prompt and
/// length spec 3.2/3.4 pin, then spec 13.1 already fixes every remaining
/// field, and a disagreement is decisive without running anything.
fn check_pinned(r: &Receipt) -> Check {
    let is_pinned_run = hex::encode(&r.weights_sha256) == spec::WEIGHTS_SHA256
        && hex::encode(&r.config_sha256) == spec::CONFIG_SHA256
        && hex::encode(&r.tokenizer_sha256) == spec::TOKENIZER_SHA256
        && r.prompt == spec::PROMPT.as_bytes()
        && r.gen_toks == spec::GEN_TOKS;
    if !is_pinned_run {
        return c(
            "pinned-vector",
            Status::Skip,
            "not the spec 13.1 run (artifact hashes, prompt or gen-toks differ), so witness-digest, \
             argmax-digest and generated-token-ids are not pinned by the document and can only be \
             established by replay"
                .to_string(),
        );
    }
    let bad = check_against_pinned(r);
    if bad.is_empty() {
        c(
            "pinned-vector",
            Status::Pass,
            "this receipt names the spec 13.1 artifacts, prompt and length, and every field it \
             carries is the value spec 13.1 pins --- established with no artifacts at all"
                .to_string(),
        )
    } else {
        c("pinned-vector", Status::Fail, bad.join("; "))
    }
}

/// The fields `check` did **not** establish, given its own results. This is
/// the honest half of the cheap tier: it is what a `verify` would still buy.
pub fn residual(checks: &[Check]) -> Vec<&'static str> {
    let pinned_passed = checks
        .iter()
        .any(|k| k.name == "pinned-vector" && k.status == Status::Pass);
    if pinned_passed {
        // Spec 13.1 fixed all thirteen fields; nothing is left to a replay
        // except the claim that the logits came from running the model,
        // which no field records and no digest comparison can supply.
        return Vec::new();
    }
    let mut left: Vec<&'static str> = Vec::new();
    for field in CHECKABLE {
        let covered = checks.iter().any(|k| {
            k.status == Status::Pass
                && match field {
                    "inv-freq-table-digest" | "config-sha256" => k.name == "inv-freq-table-digest",
                    "prompt-token-ids" | "tokenizer-sha256" => k.name == "prompt-token-ids",
                    "table-digest" => k.name == "table-digest",
                    _ => false,
                }
        });
        if !covered {
            left.push(field);
        }
    }
    for field in ["generated-token-ids", "argmax-digest", "witness-digest"] {
        left.push(field);
    }
    left
}

/// True when nothing failed. Skips are not failures --- they are the reason
/// `residual` exists.
pub fn passed(checks: &[Check]) -> bool {
    !checks.iter().any(|k| k.status == Status::Fail)
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    /// A receipt carrying exactly the spec 13.1 values, built from `spec`
    /// so the test cannot drift from the document.
    fn pinned_receipt() -> Receipt {
        let d = |s: &str| {
            let b = hex::decode(s).unwrap();
            let mut o = [0u8; 32];
            o.copy_from_slice(&b);
            o
        };
        Receipt {
            spec_version: "0.3b".into(),
            weights_sha256: d(spec::WEIGHTS_SHA256),
            config_sha256: d(spec::CONFIG_SHA256),
            tokenizer_sha256: d(spec::TOKENIZER_SHA256),
            prompt: spec::PROMPT.as_bytes().to_vec(),
            prompt_token_ids: spec::PROMPT_TOKEN_IDS.to_vec(),
            gen_toks: spec::GEN_TOKS,
            dtype: "fp32".into(),
            table_digest: d(spec::TABLE_DIGEST),
            inv_freq_table_digest: d(spec::INV_FREQ_TABLE_DIGEST),
            generated_token_ids: spec::GENERATED_TOKEN_IDS.to_vec(),
            argmax_digest: d(spec::ARGMAX_DIGEST),
            witness_digest: d(spec::CIS2_REF),
        }
    }

    fn status_of(checks: &[Check], name: &str) -> Status {
        checks.iter().find(|k| k.name == name).unwrap().status
    }

    #[test]
    fn the_pinned_receipt_passes_with_no_artifacts_and_leaves_no_residual() {
        let text = pinned_receipt().render();
        let checks = check(&text, &Evidence::default());
        assert!(passed(&checks), "{checks:?}");
        assert_eq!(status_of(&checks, "pinned-vector"), Status::Pass);
        assert_eq!(status_of(&checks, "table-digest"), Status::Pass);
        assert!(residual(&checks).is_empty(), "{:?}", residual(&checks));
    }

    /// The point of the tier: an off-vector receipt is *not* silently
    /// blessed. It passes what can be checked and names what it could not.
    #[test]
    fn an_unknown_checkpoint_skips_rather_than_passes_and_names_the_residual() {
        let mut r = pinned_receipt();
        r.weights_sha256 = [0xAB; 32];
        r.config_sha256 = [0xCD; 32];
        r.tokenizer_sha256 = [0xEF; 32];
        let checks = check(&r.render(), &Evidence::default());
        assert!(
            passed(&checks),
            "skips must not read as failures: {checks:?}"
        );
        assert_eq!(status_of(&checks, "pinned-vector"), Status::Skip);
        assert_eq!(status_of(&checks, "inv-freq-table-digest"), Status::Skip);
        assert_eq!(status_of(&checks, "prompt-token-ids"), Status::Skip);
        let left = residual(&checks);
        for must in [
            "witness-digest",
            "argmax-digest",
            "generated-token-ids",
            "weights-sha256",
        ] {
            assert!(left.contains(&must), "residual omitted {must}: {left:?}");
        }
        // The one thing that never needs an artifact is still established.
        assert_eq!(status_of(&checks, "table-digest"), Status::Pass);
        assert!(!left.contains(&"table-digest"));
    }

    #[test]
    fn a_moved_table_digest_fails_with_no_artifacts() {
        let mut r = pinned_receipt();
        r.table_digest[0] ^= 1;
        let checks = check(&r.render(), &Evidence::default());
        assert_eq!(status_of(&checks, "table-digest"), Status::Fail);
        assert!(!passed(&checks));
    }

    /// A receipt that claims the pinned artifacts but a forged witness is
    /// refuted without downloading 270 MB. This is the case the cheap tier
    /// exists for.
    #[test]
    fn a_forged_witness_on_the_pinned_vector_fails_with_no_artifacts() {
        let mut r = pinned_receipt();
        r.witness_digest[31] ^= 0xFF;
        let checks = check(&r.render(), &Evidence::default());
        assert_eq!(status_of(&checks, "pinned-vector"), Status::Fail);
        assert!(!passed(&checks));
    }

    #[test]
    fn a_non_canonical_but_parseable_file_is_caught() {
        // Key order is the one non-canonical spelling the parser accepts.
        let text = pinned_receipt().render();
        let mut lines: Vec<&str> = text.lines().collect();
        lines.swap(2, 3);
        let mut reordered = lines.join("\n");
        reordered.push('\n');
        let checks = check(&reordered, &Evidence::default());
        assert_eq!(status_of(&checks, "canonical-form"), Status::Fail);
    }

    #[test]
    fn a_missing_trailing_newline_is_caught() {
        let text = pinned_receipt().render();
        let checks = check(text.trim_end(), &Evidence::default());
        assert_eq!(status_of(&checks, "canonical-form"), Status::Fail);
    }

    #[test]
    fn garbage_is_one_failing_check_not_a_panic() {
        let checks = check("not a receipt\n", &Evidence::default());
        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].status, Status::Fail);
        assert!(!passed(&checks));
    }

    #[test]
    fn a_wrong_dtype_fails() {
        let mut r = pinned_receipt();
        r.dtype = "bf16".into();
        let checks = check(&r.render(), &Evidence::default());
        assert_eq!(status_of(&checks, "dtype"), Status::Fail);
    }
}
