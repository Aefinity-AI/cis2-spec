//! End-to-end regression test over a committed 2.8 kB fixture model.
//!
//! Every other test in this crate covers one function, or is a golden that
//! only a 500 MB checkpoint can exercise. This one runs the whole pipeline
//! in CI with nothing to download: safetensors parse -> BF16 widening ->
//! config -> tokenizer -> forward pass (GQA, RoPE, RMSNorm, SwiGLU, an
//! untied lm_head) -> witness chain -> receipt render -> receipt parse ->
//! verify.
//!
//! What it is not: it is NOT a conformance vector, and its digests say
//! nothing about CIS-2 v0.3b's pinned values. The spec's real vector is
//! `spec::PINNED_*`, checked in `mathpin.rs` and by the binary against the
//! actual SmolLM2 checkpoint. This fixture is a tripwire: if a refactor
//! changes any stage's arithmetic, or breaks the receipt round trip, or
//! lets a tampered receipt through, one of the asserts below fails.
//!
//! Regenerate the fixture with `python3 tools/gen_tiny_fixture.py`. It takes
//! no RNG, so a regeneration that changes any byte is a real change and the
//! `fixture_bytes_are_the_ones_these_goldens_were_taken_from` test catches
//! it before the goldens silently become meaningless.

use cis2_verify::{hex, receipt::Receipt, sha256::sha256, verify};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/tiny/");
const PROMPT: &str = "abcd";
const GEN_TOKS: usize = 6;

// Taken from a run of this exact fixture. See the module doc for what these
// are and, more importantly, what they are not.
const WITNESS: &str = "1dfe8486aeed5be8528e0eeba843328b742f5c84bf16beeb30f124e84e9d511e";
const ARGMAX: &str = "f788b25305cd9bee125556cefc42f33e49d115e7730699c514d5cde1c19bc127";
const PROMPT_IDS: &[u32] = &[15, 3];
const GEN_IDS: &[u32] = &[6, 6, 1, 5, 1, 6];

const SHA_WEIGHTS: &str = "3f97fa1b70f07d558cceffb425c5d632be62fc035d219c55564c3fd37fc5692c";
const SHA_CONFIG: &str = "9aa26de5976dda7e8c4f494b008105528a8267b19efac9b152993da2e7165fb2";
const SHA_TOKENIZER: &str = "0906b35b6abd0a8c03aee568d6609fa66c56779ceac3a5ce80d0164afdd7dd80";

fn artifacts() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    (
        std::fs::read(format!("{DIR}model.safetensors")).expect("fixture weights"),
        std::fs::read(format!("{DIR}config.json")).expect("fixture config"),
        std::fs::read(format!("{DIR}tokenizer.json")).expect("fixture tokenizer"),
    )
}

fn art<'a>(w: &'a [u8], c: &'a [u8], t: &'a [u8]) -> verify::Artifacts<'a> {
    verify::Artifacts { weights: w, config: c, tokenizer: t }
}

#[test]
fn fixture_bytes_are_the_ones_these_goldens_were_taken_from() {
    // Without this, regenerating the fixture with a different generator would
    // move every golden below and the test file would still "pass" after a
    // mechanical update -- which is how a regression harness quietly stops
    // being one.
    let (w, c, t) = artifacts();
    assert_eq!(hex::encode(&sha256(&w)), SHA_WEIGHTS, "model.safetensors changed");
    assert_eq!(hex::encode(&sha256(&c)), SHA_CONFIG, "config.json changed");
    assert_eq!(hex::encode(&sha256(&t)), SHA_TOKENIZER, "tokenizer.json changed");
}

#[test]
fn the_whole_pipeline_reproduces_its_goldens() {
    let (w, c, t) = artifacts();
    let r = verify::run(&art(&w, &c, &t), PROMPT, GEN_TOKS).expect("run");
    assert_eq!(r.prompt_token_ids, PROMPT_IDS, "tokenizer output moved");
    assert_eq!(r.generated_token_ids, GEN_IDS, "decoded tokens moved");
    assert_eq!(hex::encode(&r.witness_digest), WITNESS);
    assert_eq!(hex::encode(&r.argmax_digest), ARGMAX);
    assert_eq!(r.gen_toks, GEN_TOKS);
    assert_eq!(r.prompt, PROMPT.as_bytes());
}

#[test]
fn two_runs_of_the_same_inputs_give_the_same_receipt() {
    // The property the whole estate rests on, checked at fixture scale.
    let (w, c, t) = artifacts();
    let a = verify::run(&art(&w, &c, &t), PROMPT, GEN_TOKS).expect("run a");
    let b = verify::run(&art(&w, &c, &t), PROMPT, GEN_TOKS).expect("run b");
    assert_eq!(a, b);
    assert_eq!(a.render(), b.render());
}

#[test]
fn a_receipt_written_out_and_read_back_verifies_against_its_own_artifacts() {
    // The real round trip: not `parse(render(r)) == r` over a synthetic
    // struct (that is unit-tested in receipt.rs) but a receipt produced by an
    // actual decode, rendered to text, parsed by the strict parser, and then
    // re-verified by recomputing the decode from the artifacts on disk.
    let (w, c, t) = artifacts();
    let produced = verify::run(&art(&w, &c, &t), PROMPT, GEN_TOKS).expect("run");
    let text = produced.render();
    let parsed = Receipt::parse(&text).expect("the verifier's own output must parse");
    assert_eq!(parsed, produced, "render/parse is not the identity on a real receipt");
    let complaints = verify::verify(&art(&w, &c, &t), &parsed);
    assert!(complaints.is_empty(), "honest receipt rejected: {complaints:?}");
}

#[test]
fn one_flipped_weight_bit_breaks_verification() {
    let (mut w, c, t) = artifacts();
    let honest = verify::run(&art(&w, &c, &t), PROMPT, GEN_TOKS).expect("run");

    // Flip the lowest mantissa bit of the first BF16 in the payload. The
    // payload starts after the 8-byte header length and the header JSON.
    let hlen = u64::from_le_bytes(w[0..8].try_into().unwrap()) as usize;
    let first = 8 + hlen;
    w[first] ^= 1;

    // First: the weights hash no longer matches, which is the cheap check.
    let complaints = verify::verify(&art(&w, &c, &t), &honest);
    assert!(
        complaints.iter().any(|s| s.contains("model.safetensors")),
        "a flipped weight bit did not trip the artifact hash: {complaints:?}"
    );

    // Then the one that actually matters: even if an attacker also rewrote
    // the receipt's artifact hashes to match the tampered file, recomputation
    // still lands on a different witness digest. One mantissa bit, 8 hidden
    // dimensions, 6 decoded tokens -- and the receipt stops matching.
    let tampered = verify::run(&art(&w, &c, &t), PROMPT, GEN_TOKS).expect("run tampered");
    assert_ne!(
        tampered.witness_digest, honest.witness_digest,
        "a flipped weight bit left the witness digest unchanged"
    );
}

#[test]
fn a_doctored_receipt_is_rejected_field_by_field() {
    let (w, c, t) = artifacts();
    let honest = verify::run(&art(&w, &c, &t), PROMPT, GEN_TOKS).expect("run");
    let text = honest.render();

    // Each mutation is one line of the rendered receipt, rewritten to a
    // value that is still well-formed -- so the parser cannot reject it on
    // shape and `verify` has to catch it by recomputing.
    let mutations: [(&str, &str, &str); 4] = [
        ("last generated token", &format!("generated-token-ids {}", csv(GEN_IDS)),
         &format!("generated-token-ids {}", csv(&flip_last(GEN_IDS)))),
        ("prompt token ids", &format!("prompt-token-ids {}", csv(PROMPT_IDS)),
         &format!("prompt-token-ids {}", csv(&flip_last(PROMPT_IDS)))),
        ("witness digest", &format!("witness-digest {WITNESS}"),
         &format!("witness-digest {}", flip_hex(WITNESS))),
        ("argmax digest", &format!("argmax-digest {ARGMAX}"),
         &format!("argmax-digest {}", flip_hex(ARGMAX))),
    ];

    for (what, from, to) in mutations {
        assert!(text.contains(from), "receipt has no {what} line to mutate: {from:?}");
        let doctored = text.replace(from, to);
        assert_ne!(doctored, text, "mutation for {what} was a no-op");
        let claimed = Receipt::parse(&doctored)
            .unwrap_or_else(|e| panic!("mutation for {what} broke the shape, not the claim: {e}"));
        let complaints = verify::verify(&art(&w, &c, &t), &claimed);
        assert!(
            !complaints.is_empty(),
            "a doctored {what} verified clean -- the receipt is not binding"
        );
    }
}

fn csv(ids: &[u32]) -> String {
    ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",")
}

fn flip_last(ids: &[u32]) -> Vec<u32> {
    let mut v = ids.to_vec();
    let n = v.len() - 1;
    v[n] = if v[n] == 0 { 1 } else { v[n] - 1 };
    v
}

fn flip_hex(h: &str) -> String {
    let mut s = h.to_string();
    let last = s.pop().unwrap();
    s.push(if last == '0' { '1' } else { '0' });
    s
}
