//! Spec 14.1 reach census: does a real pinned decode ever put an FFN
//! intermediate inside §6.2's clip band, and how often does §10's softmax
//! exercise §6.2's low guard?
//!
//! E27 measured `exp_pinned`/`ln_pinned`/`silu_pinned` at all 2^32 f32
//! arguments and found that §6.2 step 2 clips at `x > 88.0` although
//! `ln(f32::MAX) = 88.7228390520684`: the 94,743 arguments in
//! `[0x42B00001, 0x42B17217]` return `+Infinity` where the true value is
//! finite and representable. §10's softmax cannot reach that band --- its
//! argument is `v - max_v`, always `<= 0`. §6.4's SiLU can: it evaluates
//! `exp_pinned(-x)`, so an FFN intermediate in
//! `[-88.7228317, -88.0000076]` lands inside it, and there `silu_pinned`
//! returns `-0.0` in place of a NORMAL f32 near `-5.328e-37`.
//!
//! E27 left that reachable-in-principle and unmeasured. This measures it.
//!
//! Build and run:
//!
//!     cargo run --release --features census --example silu_reach -- <artifact-dir> [prompt] [gen-toks] [ids]
//!
//! A census build is instrumented and is NOT a conforming verifier. It says
//! so on every line of its own output. It only reads the values it counts,
//! so the digests it reports are the pinned ones.

use cis2_verify::{fpenv, hex, ops::census, spec, verify};
use std::path::Path;

fn read(dir: &Path, name: &str) -> Vec<u8> {
    std::fs::read(dir.join(name))
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.join(name).display()))
}

fn main() {
    let dir = std::env::args().nth(1).expect("usage: silu_reach <artifact-dir>");
    let prompt = std::env::args().nth(2).unwrap_or_else(|| spec::PROMPT.to_string());
    let gen_toks: usize = std::env::args()
        .nth(3)
        .map(|v| v.parse().expect("gen-toks"))
        .unwrap_or(spec::GEN_TOKS);
    let dir = Path::new(&dir);

    fpenv::pin_and_selftest().expect("1.3 pin");
    println!("REACH fp-env=pinned control={}", fpenv::control_name());
    println!("REACH build=instrumented (NOT a conforming verifier)");
    println!("REACH artifacts={} prompt={prompt:?} gen_toks={gen_toks}", dir.display());

    let (w, c, t) = (
        read(dir, "model.safetensors"),
        read(dir, "config.json"),
        read(dir, "tokenizer.json"),
    );
    let art = verify::Artifacts { weights: &w, config: &c, tokenizer: &t };

    // Optional 4th argument: comma-separated prompt token ids, for a
    // checkpoint whose `tokenizer.json` §3.1.3/§3.1.4 refuse (Qwen2.5-0.5B).
    let ids: Option<Vec<u32>> = std::env::args().nth(4).map(|v| {
        v.split(',')
            .map(|p| p.trim().parse::<u32>().expect("token id"))
            .collect()
    });
    match &ids {
        None => println!("REACH tokenization=spec-3 (this crate derived the prompt token ids)"),
        Some(ids) => println!(
            "REACH tokenization=SUPPLIED prompt_token_ids={ids:?} \
             (spec 3.1.3/3.1.4 refuse this tokenizer.json; spec 4-11 only, NOT a conformance run)"
        ),
    }

    census::reset();
    let out = match &ids {
        None => verify::run(&art, &prompt, gen_toks),
        Some(ids) => verify::run_with_token_ids(&art, &prompt, ids, gen_toks),
    }
    .expect("decode");

    let (n, band, below, subband, min, max) = census::silu_counts();
    let top = census::SUBNORM_TOP;
    let (sn, low, subn, smin, smax) = census::softmax_counts();

    println!("REACH silu calls={n}");
    println!(
        "REACH silu arg range=[{min:e}, {max:e}]  bits=[0x{:08X}, 0x{:08X}]",
        min.to_bits(),
        max.to_bits()
    );
    println!(
        "REACH silu IN 6.2 CLIP BAND [-88.7228317, -88.0000076] n={band} \
         (>0 means a real decode reaches the clip; each such element returns -0.0 \
         instead of a normal f32 near -5.328e-37)"
    );
    println!("REACH silu arg < -88.0 (any) n={below}");
    println!(
        "REACH silu IN SUBNORMAL BAND ({}, 88.0] n={subband} \
         (6.4 evaluates exp_pinned(-x), so this is the softmax band mirrored)",
        -top
    );
    println!("REACH softmax exp args={sn}");
    println!(
        "REACH softmax hit 6.2 LOW guard (arg < -88.0 -> 0.0) n={low} \
         (this side is FTZ-independent: the guard returns 0.0 before exp runs, \
         and the unclamped exp would have been subnormal and flushed anyway. \
         The FTZ-dependent side is the SUBNORMAL BAND line below.)"
    );
    println!("REACH softmax arg range=[{smin:e}, {smax:e}]");
    println!(
        "REACH softmax IN SUBNORMAL BAND [-88.0, {top}) n={subn} \
         (>0 means this decode computed a subnormal exp that 1.3's FTZ flushed \
         to zero, so FTZ/DAZ is digest-relevant here and E22's M01 mutant \
         should move the digest; =0 means it is not, which is what E22 item 1 \
         asks and no counter previously answered)"
    );

    // The census only reads, so these must be the pinned values.
    println!("REACH witness-digest {}", hex::encode(&out.witness_digest));
    println!("REACH argmax-digest {}", hex::encode(&out.argmax_digest));
}
