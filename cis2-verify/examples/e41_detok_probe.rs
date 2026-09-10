//! E41 probe: is the tokenizer's id -> token-string map even well-defined?
//!
//! Spec §3 specifies encoding (text -> ids) completely, and §3.1.5.a says the
//! *byte* map's inverse is used for decoding. Nothing specifies the id -> token
//! string step, and `Tokenizer` has no `decode`. Before calling that a spec
//! gap, establish whether the inverse exists at all in the shipped artifacts:
//!
//!   * is every id in `0..vocab_size` present exactly once?
//!   * do two distinct strings ever share an id (inverse not a function)?
//!   * does `added_tokens` collide with, or extend past, `model.vocab`?
//!   * is every vocab string decodable through the inverse byte map?
//!
//!     cargo run --release --example e41_detok_probe -- <tokenizer.json> <label>

use cis2_verify::json::{parse, Json};
use cis2_verify::tokenizer;
use std::collections::BTreeMap;

fn main() {
    let path = std::env::args().nth(1).expect("usage: <tokenizer.json> <label>");
    let label = std::env::args().nth(2).unwrap_or_else(|| "?".into());
    let bytes = std::fs::read(&path).expect("read tokenizer.json");
    let root = parse(&bytes).expect("parse tokenizer.json");

    let vocab = match root.get("model").and_then(|m| m.get("vocab")) {
        Some(Json::Obj(kv)) => kv.clone(),
        _ => panic!("no model.vocab object"),
    };

    // id -> the strings that claim it
    let mut by_id: BTreeMap<u64, Vec<String>> = BTreeMap::new();
    for (s, v) in &vocab {
        let id = v.as_u64().expect("vocab value is not an integer");
        by_id.entry(id).or_default().push(s.clone());
    }
    let n_pairs = vocab.len();
    let n_ids = by_id.len();
    let collisions: Vec<_> = by_id.iter().filter(|(_, v)| v.len() > 1).collect();
    let max_id = by_id.keys().copied().max().unwrap_or(0);
    let gaps: Vec<u64> = (0..=max_id).filter(|i| !by_id.contains_key(i)).collect();

    // added_tokens
    let added = root.get("added_tokens").and_then(|a| a.as_arr()).map(|a| a.to_vec()).unwrap_or_default();
    let mut added_ids = Vec::new();
    let mut added_overlap_vocab = 0usize;
    let mut added_beyond_vocab = 0usize;
    let mut added_content_differs = Vec::new();
    for t in &added {
        let id = t.get("id").and_then(|v| v.as_u64()).expect("added token id");
        let content = t.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string();
        added_ids.push((id, content.clone()));
        match by_id.get(&id) {
            Some(v) => {
                added_overlap_vocab += 1;
                if !v.contains(&content) {
                    added_content_differs.push((id, content, v.clone()));
                }
            }
            None => added_beyond_vocab += 1,
        }
    }

    // Every vocab string must be expressible through the inverse byte map:
    // each char of the token string must be in the 256-entry byte table.
    let table = tokenizer::bytes_char();
    let mut inv = BTreeMap::new();
    for (b, c) in table.iter().enumerate() {
        inv.insert(*c, b as u8);
    }
    let mut undecodable: Vec<(u64, String)> = Vec::new();
    for (s, v) in &vocab {
        if s.chars().any(|c| !inv.contains_key(&c)) {
            undecodable.push((v.as_u64().unwrap(), s.clone()));
        }
    }
    // The byte map itself must be a bijection or "the inverse map" is not one.
    let distinct_chars: std::collections::BTreeSet<char> = table.iter().copied().collect();

    println!("E41-PROBE {label} vocab-pairs={n_pairs} distinct-ids={n_ids} max-id={max_id}");
    println!("E41-PROBE {label} id-collisions={} gaps-in-0..=max={}", collisions.len(), gaps.len());
    if !collisions.is_empty() {
        for (id, v) in collisions.iter().take(5) {
            println!("E41-PROBE {label}   COLLISION id={id} strings={v:?}");
        }
    }
    if !gaps.is_empty() {
        println!("E41-PROBE {label}   first gaps={:?}", &gaps[..gaps.len().min(8)]);
    }
    println!(
        "E41-PROBE {label} added-tokens={} overlap-vocab={} beyond-vocab={} content-differs={}",
        added.len(),
        added_overlap_vocab,
        added_beyond_vocab,
        added_content_differs.len()
    );
    for (id, c, v) in added_content_differs.iter().take(5) {
        println!("E41-PROBE {label}   ADDED-DIFFERS id={id} added={c:?} vocab={v:?}");
    }
    println!(
        "E41-PROBE {label} byte-table-distinct-chars={} (must be 256) undecodable-vocab-strings={}",
        distinct_chars.len(),
        undecodable.len()
    );
    for (id, s) in undecodable.iter().take(5) {
        println!("E41-PROBE {label}   UNDECODABLE id={id} s={s:?}");
    }
    let well_defined = collisions.is_empty() && gaps.is_empty() && distinct_chars.len() == 256;
    println!("E41-PROBE {label} INVERSE-WELL-DEFINED={well_defined}");
}
