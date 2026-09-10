//! `compare`: can these two receipts both be telling the truth?
//!
//! `verify` answers "did this run happen" and needs the weights. `check`
//! answers "is this receipt worth running" and needs nothing. Neither
//! answers the question that actually starts a dispute, which is always
//! about two documents rather than one:
//!
//! > "The vendor's receipt says the model produced X. Mine says it produced
//! > Y. Which of us is wrong, and about what?"
//!
//! That question is answerable without any artifacts, because CIS-2 makes
//! the receipt's fields split cleanly into what was *run* and what the run
//! *produced*:
//!
//! - **inputs** --- `spec-version`, the three artifact hashes, `prompt-hex`,
//!   `gen-toks`, `dtype`. These say which computation was performed.
//! - **derived** --- `prompt-token-ids`, `table-digest`,
//!   `inv-freq-table-digest`. These are functions of the inputs alone; spec
//!   3.3, 6.6 and 7.2 fix them, and no run is involved.
//! - **outputs** --- `generated-token-ids`, `argmax-digest`,
//!   `witness-digest`. These are what the forward pass produced.
//!
//! Spec 1.4 is what gives the comparison teeth: a conforming implementation
//! run on identical inputs produces identical outputs, bit for bit, on any
//! conforming target. So if two receipts agree on every input and disagree
//! on any output, **at most one of them is conforming** --- and that is a
//! contradiction the holder of either document can establish from an email
//! attachment, with no weights, no GPU and no cooperation from the other
//! party.
//!
//! The converse case is not a contradiction and is not reported as one: if
//! the inputs differ, the outputs are under no obligation to agree, and
//! `compare` says which input differs instead of pretending to a verdict it
//! cannot reach. When the differing input is `weights-sha256` and
//! everything else matches, that is the model-substitution signature, and
//! it is named as such.
//!
//! One asymmetric case is worth more than the obvious ones. If two receipts
//! name **different weights** and still carry the **same `witness-digest`**,
//! `compare` raises it, because a forger's cheapest move is to take a real
//! receipt and edit the header. Spec 12.1 digests the full logit stream, so
//! this is not a coincidence two unrelated models produce.
//!
//! It is deliberately raised only for `witness-digest`, and deliberately
//! not called proof. Two points, both of which would make a broader rule
//! produce false accusations, and a tool that accuses falsely is worth
//! nothing in a dispute:
//!
//! - `generated-token-ids` and `argmax-digest` carry the *argmax* stream,
//!   not the logits. Two different models --- a fine-tune and its base, two
//!   quantisations, two checkpoints --- agreeing on sixteen greedy tokens
//!   for a short prompt is ordinary, not suspicious. `compare` reports such
//!   an agreement as an observation and reaches no verdict from it.
//! - Even for `witness-digest`, there is one innocent explanation:
//!   `weights-sha256` hashes the *file*, and two safetensors files can
//!   serialise the same tensors with different metadata and hash
//!   differently while computing identically. So the finding names both
//!   readings and the cheap way to tell them apart, rather than asserting
//!   the accusatory one.
//!
//! What `compare` cannot do: tell you *which* of two contradicting receipts
//! is the honest one. That needs the weights, and the answer is
//! `cis2-verify verify` on each. `compare` narrows a dispute to the field
//! it turns on and tells you whether paying for a replay would settle
//! anything.

use crate::hex;
use crate::receipt::Receipt;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Which part of the receipt a field belongs to. The whole comparison
/// rests on this partition, so it is data rather than a comment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Part {
    /// Names the computation that was performed.
    Input,
    /// A function of the inputs alone, fixed by the spec, no run involved.
    Derived,
    /// What the forward pass produced.
    Output,
}

impl Part {
    pub fn label(self) -> &'static str {
        match self {
            Part::Input => "input",
            Part::Derived => "derived",
            Part::Output => "output",
        }
    }
}

/// One field, and whether the two receipts agreed on it.
#[derive(Debug, Clone)]
pub struct FieldDiff {
    pub name: &'static str,
    pub part: Part,
    pub same: bool,
    /// The value in the first receipt, rendered as the file renders it.
    pub a: String,
    /// The value in the second receipt.
    pub b: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Every field agrees. The two documents make the same claim.
    Identical,
    /// Inputs agree, outputs (or derived fields) do not. At most one of
    /// these receipts is conforming.
    Contradiction,
    /// Inputs differ, so the outputs were never required to agree.
    DifferentRun,
    /// Inputs differ in the weights, yet the full-logit witness matches.
    /// Either the same tensors in two files, or a copied field.
    WitnessCollision,
}

impl Verdict {
    pub fn label(self) -> &'static str {
        match self {
            Verdict::Identical => "IDENTICAL",
            Verdict::Contradiction => "CONTRADICTION",
            Verdict::DifferentRun => "DIFFERENT-RUN",
            Verdict::WitnessCollision => "WITNESS-COLLISION",
        }
    }
}

pub struct Comparison {
    pub fields: Vec<FieldDiff>,
    pub verdict: Verdict,
    /// One sentence saying what the verdict means for the reader, and what
    /// (if anything) a replay would still buy them.
    pub finding: String,
}

impl Comparison {
    pub fn differing(&self, part: Part) -> Vec<&FieldDiff> {
        self.fields
            .iter()
            .filter(|f| f.part == part && !f.same)
            .collect()
    }
    pub fn agreeing(&self, part: Part) -> Vec<&FieldDiff> {
        self.fields
            .iter()
            .filter(|f| f.part == part && f.same)
            .collect()
    }
}

fn ids_str(v: &[u32]) -> String {
    let parts: Vec<String> = v.iter().map(|t| t.to_string()).collect();
    parts.join(",")
}

fn names(fs: &[&FieldDiff]) -> String {
    let parts: Vec<&str> = fs.iter().map(|f| f.name).collect();
    parts.join(",")
}

/// Compare two parsed receipts field by field.
///
/// Note what this deliberately does *not* take: no artifact directory, no
/// `Evidence`, nothing but the two documents.
pub fn compare(a: &Receipt, b: &Receipt) -> Comparison {
    let d = |name: &'static str, part: Part, x: String, y: String| FieldDiff {
        name,
        part,
        same: x == y,
        a: x,
        b: y,
    };
    let h = |x: &[u8; 32]| hex::encode(x);

    let fields = alloc::vec![
        d("spec-version", Part::Input, a.spec_version.clone(), b.spec_version.clone()),
        d("weights-sha256", Part::Input, h(&a.weights_sha256), h(&b.weights_sha256)),
        d("config-sha256", Part::Input, h(&a.config_sha256), h(&b.config_sha256)),
        d("tokenizer-sha256", Part::Input, h(&a.tokenizer_sha256), h(&b.tokenizer_sha256)),
        d("prompt-hex", Part::Input, hex::encode(&a.prompt), hex::encode(&b.prompt)),
        d("gen-toks", Part::Input, a.gen_toks.to_string(), b.gen_toks.to_string()),
        d("dtype", Part::Input, a.dtype.clone(), b.dtype.clone()),
        d("prompt-token-ids", Part::Derived, ids_str(&a.prompt_token_ids), ids_str(&b.prompt_token_ids)),
        d("table-digest", Part::Derived, h(&a.table_digest), h(&b.table_digest)),
        d("inv-freq-table-digest", Part::Derived, h(&a.inv_freq_table_digest), h(&b.inv_freq_table_digest)),
        d("generated-token-ids", Part::Output, ids_str(&a.generated_token_ids), ids_str(&b.generated_token_ids)),
        d("argmax-digest", Part::Output, h(&a.argmax_digest), h(&b.argmax_digest)),
        d("witness-digest", Part::Output, h(&a.witness_digest), h(&b.witness_digest)),
    ];

    let mut c = Comparison {
        fields,
        verdict: Verdict::Identical,
        finding: String::new(),
    };
    let in_diff = c.differing(Part::Input).len();
    let der_diff = c.differing(Part::Derived).len();
    let out_diff = c.differing(Part::Output).len();
    let weights_differ = c
        .fields
        .iter()
        .any(|f| f.name == "weights-sha256" && !f.same);
    let out_same = c.agreeing(Part::Output);
    let witness_same = c
        .fields
        .iter()
        .any(|f| f.name == "witness-digest" && f.same);

    let (verdict, finding) = if in_diff == 0 && der_diff == 0 && out_diff == 0 {
        (
            Verdict::Identical,
            "the two receipts make the same claim in every field; nothing is in dispute and a \
             replay would settle nothing that is contested"
                .to_string(),
        )
    } else if in_diff == 0 {
        // Same computation named, different result claimed.
        let mut which = Vec::new();
        if der_diff > 0 {
            which.push(format!(
                "derived field(s) {} --- these are functions of the inputs alone (spec 3.3/6.6/7.2), \
                 so `cis2-verify check` on each receipt names the wrong one with no artifacts at all",
                names(&c.differing(Part::Derived))
            ));
        }
        if out_diff > 0 {
            which.push(format!(
                "output field(s) {} --- spec 1.4 requires identical inputs to give identical \
                 outputs, so at most one of these receipts is conforming",
                names(&c.differing(Part::Output))
            ));
        }
        (
            Verdict::Contradiction,
            format!(
                "every input field agrees, but the receipts disagree on {}. Which one is honest \
                 is not decidable from the documents: run `cis2-verify verify <artifact-dir>` on \
                 each with the weights the inputs name",
                which.join("; and on ")
            ),
        )
    } else if weights_differ && witness_same {
        (
            Verdict::WitnessCollision,
            "the receipts name different weights, yet carry the same witness-digest. Spec 12.1 \
             digests the full logit stream, so this is not two different models coinciding. \
             There are exactly two readings and they are cheap to tell apart: either the two \
             files serialise the same tensors with different metadata --- weights-sha256 hashes \
             the file, not the parameters --- or one of these documents was copied from the \
             other and its header edited. Run `cis2-verify verify` on each against the weights \
             it names; the copied one fails"
                .to_string(),
        )
    } else {
        let inputs = names(&c.differing(Part::Input));
        let argmax_note = if weights_differ && !out_same.is_empty() {
            format!(
                " Output field(s) {} agree even though the weights differ; that is an \
                 observation, not a finding --- these fields carry the argmax stream, and two \
                 related checkpoints agreeing on a short greedy continuation is ordinary.",
                names(&out_same)
            )
        } else {
            String::new()
        };
        let sig = if weights_differ && in_diff == 1 {
            " Only the weights differ: this is the model-substitution signature --- same prompt, \
             same length, same dtype, a different model. Whether that substitution was permitted \
             is a contract question, not one these receipts can answer."
        } else {
            ""
        };
        (
            Verdict::DifferentRun,
            format!(
                "the receipts describe different computations (input field(s) {inputs} differ), so \
                 their outputs were never required to agree and no contradiction is \
                 established.{sig}{argmax_note}"
            ),
        )
    };
    c.verdict = verdict;
    c.finding = finding;
    c
}

/// The exit-status question. `compare` fails only on a state that cannot be
/// innocent: a contradiction, or an agreement that arithmetic cannot
/// produce. Two receipts describing different runs is a legitimate answer,
/// not a failure, and exits 0.
pub fn is_contradiction(c: &Comparison) -> bool {
    matches!(
        c.verdict,
        Verdict::Contradiction | Verdict::WitnessCollision
    )
}
