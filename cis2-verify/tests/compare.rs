//! `compare`'s verdicts, and — more importantly — the verdicts it refuses
//! to reach.
//!
//! A tool whose whole value is that a disputing party can rely on its
//! accusation must never accuse falsely, so half of this file is negative:
//! agreements that look suspicious and are not, and differences that look
//! damning and are not.
//!
//! Every case is built by editing a receipt's text, because that is what a
//! real dispute consists of: two documents, no artifacts.

use cis2_verify::compare::{compare, is_contradiction, Part, Verdict};
use cis2_verify::receipt::Receipt;

/// A canonical receipt, built by rendering rather than by hand so that the
/// fixture cannot drift out of the format.
fn base() -> String {
    let r = Receipt {
        spec_version: "0.3b".into(),
        weights_sha256: [0x11; 32],
        config_sha256: [0x22; 32],
        tokenizer_sha256: [0x33; 32],
        prompt: b"Once upon a time".to_vec(),
        prompt_token_ids: alloc_ids(&[6403, 1980, 253, 655]),
        gen_toks: 4,
        dtype: "fp32".into(),
        table_digest: [0x44; 32],
        inv_freq_table_digest: [0x55; 32],
        generated_token_ids: alloc_ids(&[28, 665, 436, 253]),
        argmax_digest: [0x66; 32],
        witness_digest: [0x77; 32],
    };
    r.render()
}

fn alloc_ids(v: &[u32]) -> Vec<u32> {
    v.to_vec()
}

/// Replace the value of one key, keeping the line structure intact.
fn set(text: &str, key: &str, value: &str) -> String {
    let mut out = String::new();
    let mut hit = false;
    for line in text.lines() {
        if line.starts_with(&format!("{key} ")) {
            out.push_str(&format!("{key} {value}\n"));
            hit = true;
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    assert!(hit, "no line for key {key}");
    assert_ne!(out, text, "edit of {key} was a no-op");
    out
}

fn p(text: &str) -> Receipt {
    Receipt::parse(text).unwrap_or_else(|e| panic!("fixture does not parse: {e}"))
}

#[test]
fn a_receipt_compared_with_itself_is_identical_and_passes() {
    let t = base();
    let c = compare(&p(&t), &p(&t));
    assert_eq!(c.verdict, Verdict::Identical);
    assert!(!is_contradiction(&c));
    assert!(c.fields.iter().all(|f| f.same));
    assert_eq!(c.fields.len(), 13, "every receipt field must be compared");
}

#[test]
fn same_inputs_with_a_different_witness_is_a_contradiction() {
    let a = base();
    let b = set(&a, "witness-digest", &"88".repeat(32));
    let c = compare(&p(&a), &p(&b));
    assert_eq!(c.verdict, Verdict::Contradiction);
    assert!(is_contradiction(&c));
    assert!(c.differing(Part::Input).is_empty());
    assert_eq!(c.differing(Part::Output).len(), 1);
    // The finding must send the reader somewhere, not just label the state.
    assert!(c.finding.contains("cis2-verify verify"));
}

#[test]
fn same_inputs_with_a_different_derived_field_is_also_a_contradiction() {
    // inv-freq-table-digest is a function of config.json alone. Two
    // receipts naming the same config cannot disagree about it, and this
    // one is decidable with no weights at all.
    let a = base();
    let b = set(&a, "inv-freq-table-digest", &"99".repeat(32));
    let c = compare(&p(&a), &p(&b));
    assert_eq!(c.verdict, Verdict::Contradiction);
    assert_eq!(c.differing(Part::Derived).len(), 1);
    assert!(
        c.finding.contains("cis2-verify check"),
        "a derived-field contradiction is settled by check, not verify: {}",
        c.finding
    );
}

#[test]
fn different_weights_with_different_outputs_is_not_a_contradiction() {
    let a = base();
    let mut b = set(&a, "weights-sha256", &"aa".repeat(32));
    b = set(&b, "witness-digest", &"bb".repeat(32));
    b = set(&b, "argmax-digest", &"cc".repeat(32));
    b = set(&b, "generated-token-ids", "1,2,3,4");
    let c = compare(&p(&a), &p(&b));
    assert_eq!(c.verdict, Verdict::DifferentRun);
    assert!(!is_contradiction(&c), "two different runs must exit 0");
    assert!(c.finding.contains("model-substitution signature"));
}

#[test]
fn different_weights_sharing_the_full_logit_witness_is_flagged() {
    let a = base();
    let b = set(&a, "weights-sha256", &"aa".repeat(32));
    let c = compare(&p(&a), &p(&b));
    assert_eq!(c.verdict, Verdict::WitnessCollision);
    assert!(is_contradiction(&c));
    // It must offer the innocent reading too. An accusation with only one
    // reading is the thing that makes a tool untrustworthy in a dispute.
    assert!(c.finding.contains("serialise the same tensors"));
    assert!(c.finding.contains("copied"));
}

#[test]
fn different_weights_sharing_only_the_argmax_stream_is_not_flagged() {
    // Two related checkpoints agreeing on four greedy tokens is ordinary.
    // Flagging it would make every fine-tune comparison a false accusation.
    let a = base();
    let mut b = set(&a, "weights-sha256", &"aa".repeat(32));
    b = set(&b, "witness-digest", &"bb".repeat(32));
    let c = compare(&p(&a), &p(&b));
    assert_eq!(c.verdict, Verdict::DifferentRun);
    assert!(!is_contradiction(&c));
    assert!(
        c.finding.contains("observation, not a finding"),
        "the coincidence must still be surfaced: {}",
        c.finding
    );
}

#[test]
fn a_different_prompt_is_a_different_run_not_a_substitution() {
    let a = base();
    let mut b = set(&a, "prompt-hex", &hexed(b"Twice upon a time"));
    b = set(&b, "prompt-token-ids", "1,2,3,4,5");
    b = set(&b, "witness-digest", &"bb".repeat(32));
    let c = compare(&p(&a), &p(&b));
    assert_eq!(c.verdict, Verdict::DifferentRun);
    assert!(
        !c.finding.contains("model-substitution"),
        "the substitution signature requires the weights to be the only input that moved"
    );
}

#[test]
fn every_field_lands_in_exactly_one_part_and_the_partition_is_the_documented_one() {
    let t = base();
    let c = compare(&p(&t), &p(&t));
    let count = |part: Part| c.fields.iter().filter(|f| f.part == part).count();
    assert_eq!(count(Part::Input), 7);
    assert_eq!(count(Part::Derived), 3);
    assert_eq!(count(Part::Output), 3);
    // The partition is load-bearing: outputs are exactly the three fields
    // that require a forward pass.
    let outs: Vec<&str> = c
        .fields
        .iter()
        .filter(|f| f.part == Part::Output)
        .map(|f| f.name)
        .collect();
    assert_eq!(
        outs,
        vec!["generated-token-ids", "argmax-digest", "witness-digest"]
    );
}

fn hexed(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
