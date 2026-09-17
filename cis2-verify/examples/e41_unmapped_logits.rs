//! E41 §2: 271 of Qwen's logit indices have no string in either the
//! `model.vocab` table or `added_tokens`. The lm_head is tied to
//! `model.embed_tokens.weight`, shape [vocab_size, d_model], so §12.1's argmax
//! ranges over all `vocab_size` indices --- including the unmapped ones. This
//! measures how close an unmapped index gets to winning on a real decode.
//!
//! "A counter that reads zero must be shown able to fire" (E35 §3): the probe
//! also runs with a DELIBERATELY WRONG unmapped set --- one that contains the
//! true argmax at every step --- and asserts it then reports rank 1. Without
//! that, `best-unmapped-rank` reading large proves nothing about the probe.
//!
//!   cargo run --release --features census --example e41_unmapped_logits \
//!       -- <artifact-dir> <prompt> <gen_toks> [comma,separated,prompt,ids]

use cis2_verify::json::{parse, Json};
use cis2_verify::{fpenv, model::Model, safetensors, config::parse_config};
use std::collections::BTreeSet;

/// Rank of the best member of `set` in `logits`, 1 = argmax. Also returns that
/// member's id and logit, and the argmax id and logit.
fn best_in_set(logits: &[f32], set: &BTreeSet<u32>) -> (usize, u32, f32, u32, f32) {
    let mut arg = 0u32;
    let mut argv = f32::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        if l > argv { argv = l; arg = i as u32; }
    }
    let mut best = u32::MAX;
    let mut bestv = f32::NEG_INFINITY;
    for &i in set {
        let l = logits[i as usize];
        if l > bestv { bestv = l; best = i; }
    }
    // rank = 1 + how many logits strictly exceed bestv
    let rank = 1 + logits.iter().filter(|&&l| l > bestv).count();
    (rank, best, bestv, arg, argv)
}

fn main() {
    fpenv::pin_and_selftest().expect("1.3 pin");
    let dir = std::env::args().nth(1).expect("artifact dir");
    let prompt = std::env::args().nth(2).expect("prompt");
    let gen: usize = std::env::args().nth(3).expect("gen_toks").parse().expect("gen_toks");
    let ids: Option<Vec<u32>> = std::env::args().nth(4).map(|s| {
        s.split(',').map(|t| t.trim().parse().expect("token id")).collect()
    });

    let cfg_bytes = std::fs::read(format!("{dir}/config.json")).expect("config.json");
    let tok_bytes = std::fs::read(format!("{dir}/tokenizer.json")).expect("tokenizer.json");
    let w_bytes = std::fs::read(format!("{dir}/model.safetensors")).expect("model.safetensors");
    let cfg = parse_config(&cfg_bytes).expect("parse config");
    let vocab_size = cfg.vocab_size as u32;

    // Every id that HAS a string: model.vocab values, union added_tokens ids.
    let root = parse(&tok_bytes).expect("parse tokenizer.json");
    let mut mapped: BTreeSet<u32> = BTreeSet::new();
    if let Some(Json::Obj(kv)) = root.get("model").and_then(|m| m.get("vocab")) {
        for (_, v) in kv { mapped.insert(v.as_u64().expect("vocab id") as u32); }
    }
    let mut n_added = 0usize;
    if let Some(arr) = root.get("added_tokens").and_then(|a| a.as_arr()) {
        for e in arr {
            if let Some(id) = e.get("id").and_then(|i| i.as_u64()) {
                mapped.insert(id as u32);
                n_added += 1;
            }
        }
    }
    let unmapped: BTreeSet<u32> = (0..vocab_size).filter(|i| !mapped.contains(i)).collect();
    println!(
        "E41-LOGITS vocab_size={vocab_size} mapped={} added_tokens={n_added} UNMAPPED={} \
         first_unmapped={:?}",
        mapped.len(), unmapped.len(),
        unmapped.iter().take(3).collect::<Vec<_>>()
    );

    let st = safetensors::load(&w_bytes).expect("safetensors");
    let model = Model::load(&st, cfg).expect("model");
    let pids = match ids {
        Some(v) => v,
        None => {
            use cis2_verify::tokenizer::Tokenizer;
            Tokenizer::from_json(&tok_bytes).expect("tokenizer").encode(&prompt).expect("encode")
        }
    };
    println!("E41-LOGITS prompt_token_ids={pids:?} gen={gen}");
    let d = model.decode(&pids, gen);

    if unmapped.is_empty() {
        println!("E41-LOGITS no unmapped ids in this checkpoint; nothing to measure");
    } else {
        let mut worst_rank = usize::MAX;
        let mut worst_step = 0usize;
        let mut worst_gap = f32::INFINITY;
        let mut ever_won = 0usize;
        for (s, lg) in d.step_logits.iter().enumerate() {
            assert_eq!(lg.len(), vocab_size as usize, "logit width != vocab_size");
            let (rank, _bid, bv, _aid, av) = best_in_set(lg, &unmapped);
            if rank == 1 { ever_won += 1; }
            if rank < worst_rank { worst_rank = rank; worst_step = s; }
            let gap = av - bv;
            if gap < worst_gap { worst_gap = gap; }
        }
        println!(
            "E41-LOGITS steps={} best-unmapped-rank={worst_rank} (at step {worst_step}) \
             smallest-argmax-minus-unmapped-logit={worst_gap:e} unmapped-won-argmax={ever_won}",
            d.step_logits.len()
        );
    }

    // Positive control: a set that contains the argmax at every step MUST come
    // back as rank 1. Same code path, deliberately wrong set.
    let mut ctrl_ok = true;
    for lg in d.step_logits.iter() {
        let mut arg = 0u32; let mut argv = f32::NEG_INFINITY;
        for (i, &l) in lg.iter().enumerate() { if l > argv { argv = l; arg = i as u32; } }
        let ctrl: BTreeSet<u32> = [arg].into_iter().collect();
        let (rank, _, _, _, _) = best_in_set(lg, &ctrl);
        if rank != 1 { ctrl_ok = false; }
    }
    assert!(ctrl_ok, "control inert: the rank probe cannot report rank 1");
    println!("E41-LOGITS control=PASS (probe reports rank 1 when the set holds the argmax)");
    println!("E41-LOGITS generated={:?}", d.generated);
}
