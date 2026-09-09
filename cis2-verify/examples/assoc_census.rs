//! Spec 14.5 census: how many RMSNorm elements in a real pinned decode
//! actually distinguish `(x*inv)*w` from `x*(inv*w)`, and does the
//! difference reach the published digests?
//!
//! Build and run:
//!
//!     cargo run --release --features census --example assoc_census -- <artifact-dir>
//!
//! A census build is instrumented and can run the decode in the
//! non-conforming order. It is a measurement instrument, not a conforming
//! verifier, and prints that on every line of its own output.

use cis2_verify::{fpenv, hex, ops::census, spec, verify};
use std::path::Path;

fn read(dir: &Path, name: &str) -> Vec<u8> {
    std::fs::read(dir.join(name))
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.join(name).display()))
}

fn main() {
    let dir = std::env::args().nth(1).expect("usage: assoc_census <artifact-dir>");
    let prompt = std::env::args().nth(2).unwrap_or_else(|| spec::PROMPT.to_string());
    let gen_toks: usize = std::env::args()
        .nth(3)
        .map(|v| v.parse().expect("gen-toks"))
        .unwrap_or(spec::GEN_TOKS);
    let dir = Path::new(&dir);

    fpenv::pin_and_selftest().expect("1.3 pin");
    println!("CENSUS fp-env=pinned control={}", fpenv::control_name());
    println!("CENSUS build=instrumented (NOT a conforming verifier)");
    println!("CENSUS artifacts={} prompt={prompt:?} gen_toks={gen_toks}", dir.display());

    let (w, c, t) = (read(dir, "model.safetensors"), read(dir, "config.json"), read(dir, "tokenizer.json"));
    let art = verify::Artifacts { weights: &w, config: &c, tokenizer: &t };

    // Pass 1: the pinned association (8), instrumented.
    census::reset();
    census::set_alt_order(false);
    let pinned = verify::run(&art, &prompt, gen_toks).expect("pinned run");
    let (elements, divergent) = census::counts();

    // Pass 2: the same decode, associated the other way throughout.
    census::reset();
    census::set_alt_order(true);
    let other = verify::run(&art, &prompt, gen_toks).expect("alternative-order run");
    census::set_alt_order(false);

    // `verify::run` decodes twice: spec 12.4 makes the same-host determinism
    // check a MUST, and both passes are instrumented. Report the per-decode
    // figure, which is what a reader will reconstruct from the architecture.
    const DECODES_PER_RUN: u64 = 2;
    let pct = 100.0 * (divergent as f64) / (elements as f64);
    println!("CENSUS decodes_per_run={DECODES_PER_RUN} (spec 12.4 determinism check)");
    println!("CENSUS rmsnorm_elements_total={elements}");
    println!("CENSUS rmsnorm_elements_per_decode={}", elements / DECODES_PER_RUN);
    println!(
        "CENSUS elements_per_decode_where_the_two_orders_differ={} ({pct:.4}%)",
        divergent / DECODES_PER_RUN
    );
    println!("CENSUS pinned   witness={} argmax={}", hex::encode(&pinned.witness_digest), hex::encode(&pinned.argmax_digest));
    println!("CENSUS altorder witness={} argmax={}", hex::encode(&other.witness_digest), hex::encode(&other.argmax_digest));
    println!("CENSUS witness_digest_moved={}", pinned.witness_digest != other.witness_digest);
    println!("CENSUS argmax_digest_moved={}", pinned.argmax_digest != other.argmax_digest);
    println!("CENSUS pinned   tokens={:?}", pinned.generated_token_ids);
    println!("CENSUS altorder tokens={:?}", other.generated_token_ids);
    println!("CENSUS tokens_moved={}", pinned.generated_token_ids != other.generated_token_ids);
}
