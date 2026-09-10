//! Spec 14.6 evidence: dump every intermediate activation of the pinned
//! forward pass, so it can be compared against an independent oracle.
//!
//! §14.6 records that §13.3's oracle checks are spot-checks on the *output*
//! of the layer stack — greedy token ids, plus one step's full logit vector
//! to ~4e-6 relative — and that a compensating pair of errors *inside* the
//! stack, which happened to preserve step-0's logits and every argmax
//! decision, is not excluded by that evidence. The way to exclude it is to
//! compare the intermediates, not the output.
//!
//! This writes, for every position of the §13.1 decode and every layer:
//! `ln1`, `q_proj`, `k_proj`, `v_proj`, `attn_out`, `o_proj`, `resid_attn`,
//! `ln2`, `gate_proj`, `up_proj`, `mlp_act`, `down_proj`, `resid_mlp`, plus
//! `embed`, `final_norm` and `logits` once per position. Ten of those
//! sixteen are the direct output of a named `torch.nn.Module` in the
//! reference `transformers` implementation and so are comparable without
//! any interpretation of what the oracle "meant".
//!
//! Build and run:
//!
//!     cargo run --release --features layerdump --example layer_dump -- <artifact-dir> <out.bin> [prompt] [gen-toks]
//!
//! A layerdump build is instrumented and is NOT a conforming verifier. The
//! tap is write-only, so the digests printed at the end must be the pinned
//! ones; that is the check that the instrumentation changed nothing.

use cis2_verify::{fpenv, hex, ops::tap, spec, verify};
use std::path::Path;

fn read(dir: &Path, name: &str) -> Vec<u8> {
    std::fs::read(dir.join(name))
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.join(name).display()))
}

fn main() {
    let dir = std::env::args().nth(1).expect("usage: layer_dump <artifact-dir> <out.bin>");
    let out_path = std::env::args().nth(2).expect("usage: layer_dump <artifact-dir> <out.bin>");
    let prompt = std::env::args().nth(3).unwrap_or_else(|| spec::PROMPT.to_string());
    let gen_toks: usize = std::env::args()
        .nth(4)
        .map(|v| v.parse().expect("gen-toks"))
        .unwrap_or(spec::GEN_TOKS);
    let dir = Path::new(&dir);

    fpenv::pin_and_selftest().expect("1.3 pin");
    println!("LAYERDUMP fp-env=pinned control={}", fpenv::control_name());
    println!("LAYERDUMP build=instrumented (NOT a conforming verifier)");
    println!("LAYERDUMP artifacts={} prompt={prompt:?} gen_toks={gen_toks}", dir.display());
    println!("LAYERDUMP out={out_path}");

    let (w, c, t) = (
        read(dir, "model.safetensors"),
        read(dir, "config.json"),
        read(dir, "tokenizer.json"),
    );
    let art = verify::Artifacts { weights: &w, config: &c, tokenizer: &t };

    // Optional 5th argument: comma-separated prompt token ids, for a
    // checkpoint whose `tokenizer.json` 3.1.3/3.1.4 refuse (Qwen2.5-0.5B,
    // see 14.8 / erratum E-3). Such a run exercises 4-11 only.
    let ids: Option<Vec<u32>> = std::env::args().nth(5).map(|v| {
        v.split(',').map(|p| p.trim().parse::<u32>().expect("token id")).collect()
    });
    match &ids {
        None => println!("LAYERDUMP tokenization=spec-3 (this crate derived the prompt token ids)"),
        Some(ids) => println!(
            "LAYERDUMP tokenization=SUPPLIED prompt_token_ids={ids:?} \
             (spec 3.1.3/3.1.4 refuse this tokenizer.json; spec 4-11 only, NOT a conformance run)"
        ),
    }

    // E40: with CIS2_E40_UNPINNED=1 in the environment, the whole decode runs
    // with 1.3's FTZ/DAZ cleared, and MXCSR is restored afterwards. An
    // environment variable rather than a positional argument so the flag can be
    // set without disturbing the prompt / gen-toks / token-id arguments. Dumping both ways and
    // comparing the two files answers "did 1.3 change ANY stored intermediate
    // anywhere in the stack" at per-tensor granularity, rather than only at the
    // two digests -- and it covers the operations no census counter reaches.
    let unpinned = std::env::var("CIS2_E40_UNPINNED").is_ok_and(|v| v == "1");
    println!(
        "LAYERDUMP fp-env-for-decode={}",
        if unpinned { "UNPINNED (1.3 cleared)" } else { "pinned" }
    );

    tap::open(&out_path);
    let decode = || {
        // Control for "a counter that reads zero must be shown able to fire":
        // this reads MXCSR from *inside* the decode closure, so an identical
        // pair of dumps cannot be explained by the unpinned mode never having
        // taken effect. pin_and_selftest() is called once in main and never
        // inside verify::run, so nothing re-establishes the pin under us.
        println!("LAYERDUMP mxcsr-inside-decode is_pinned={}", fpenv::is_pinned());
        match &ids {
        None => verify::run(&art, &prompt, gen_toks),
        Some(ids) => verify::run_with_token_ids(&art, &prompt, ids, gen_toks),
        }
    };
    let out = if unpinned {
        fpenv::probe_unpinned(decode)
    } else {
        decode()
    }
    .expect("decode");
    tap::close();
    println!("LAYERDUMP pin-restored-after={}", fpenv::is_pinned());

    let bytes = std::fs::metadata(&out_path).expect("stat dump").len();
    println!("LAYERDUMP bytes={bytes}");
    println!("LAYERDUMP prompt-token-ids={:?}", out.prompt_token_ids);
    println!("LAYERDUMP generated-token-ids={:?}", out.generated_token_ids);

    // The tap is write-only, so these must be the pinned values.
    println!("LAYERDUMP witness-digest {}", hex::encode(&out.witness_digest));
    println!("LAYERDUMP argmax-digest {}", hex::encode(&out.argmax_digest));
}
