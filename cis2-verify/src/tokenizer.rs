//! Spec 3.1: the byte-level BPE tokenizer, written out from the
//! specification's own description of the algorithm rather than by calling
//! any tokenizer library.
//!
//! Spec 3.1.3 gives the pipeline (no normalizer -> pre-tokenizer -> per-word
//! BPE -> no post-processor); spec 3.1.4 gives the two pre-tokenizers and
//! the GPT-2 regex; spec 3.1.5 gives the byte->unicode bijection, the
//! initial symbol sequence, and the rank-ordered merge loop with its
//! leftmost tie-break. Spec 3.1.7's worked example is a test below, and it
//! is the check the spec itself recommends running before any model math.

use crate::json::{parse, Json};
use crate::unicode::{LETTER, NUMBER, WHITESPACE};
use alloc::collections::{BTreeMap, BinaryHeap};
use alloc::string::String;
use alloc::vec::Vec;
use core::cmp::Ordering;

// ---------------------------------------------------------------------------
// Unicode property tests (spec 3.1.4's \p{L}, \p{N}, \s and char::is_numeric)
// ---------------------------------------------------------------------------

fn in_ranges(table: &[(u32, u32)], c: char) -> bool {
    let cp = c as u32;
    table
        .binary_search_by(|&(lo, hi)| {
            if cp < lo {
                Ordering::Greater
            } else if cp > hi {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        })
        .is_ok()
}

pub fn is_letter(c: char) -> bool {
    in_ranges(&LETTER, c)
}
pub fn is_number(c: char) -> bool {
    in_ranges(&NUMBER, c)
}
pub fn is_whitespace(c: char) -> bool {
    in_ranges(&WHITESPACE, c)
}

// ---------------------------------------------------------------------------
// 3.1.5.a byte <-> unicode bijection
// ---------------------------------------------------------------------------

/// The 256-entry byte->codepoint table of spec 3.1.5.a, built by the exact
/// construction the spec gives: the 188 printable ranges map to themselves,
/// then the remaining 68 bytes are assigned `256 + n` in ascending byte
/// order. Building it (rather than transcribing 256 constants) is what makes
/// the two directions provably inverse; `byte_unicode_is_a_bijection` checks
/// it.
pub fn bytes_char() -> [char; 256] {
    let mut mapped = [false; 256];
    let mut table = ['\0'; 256];
    for b in 0x21u16..=0x7E {
        table[b as usize] = char::from_u32(b as u32).unwrap();
        mapped[b as usize] = true;
    }
    for b in 0xA1u16..=0xAC {
        table[b as usize] = char::from_u32(b as u32).unwrap();
        mapped[b as usize] = true;
    }
    for b in 0xAEu16..=0xFF {
        table[b as usize] = char::from_u32(b as u32).unwrap();
        mapped[b as usize] = true;
    }
    let mut n: u32 = 0;
    for b in 0usize..=255 {
        if !mapped[b] {
            table[b] = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }
    table
}

/// Spec 3.1.4.b's final step: remap every UTF-8 byte of a segment through
/// the table, one byte to one codepoint.
fn byte_remap(segment: &str, table: &[char; 256]) -> String {
    let mut out = String::with_capacity(segment.len());
    for b in segment.as_bytes() {
        out.push(table[*b as usize]);
    }
    out
}

// ---------------------------------------------------------------------------
// 3.1.4 pre-tokenizer
// ---------------------------------------------------------------------------

/// 3.1.4.a Digits, `individual_digits = true`: every numeric character
/// becomes its own one-character segment; non-digit runs pass through
/// unsplit.
fn split_digits(input: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut run_start = 0usize;
    for (i, c) in input.char_indices() {
        if is_number(c) {
            if run_start < i {
                out.push(&input[run_start..i]);
            }
            out.push(&input[i..i + c.len_utf8()]);
            run_start = i + c.len_utf8();
        }
    }
    if run_start < input.len() {
        out.push(&input[run_start..]);
    }
    out
}

/// 3.1.4.b ByteLevel's GPT-2 regex, hand-matched:
///
/// ```text
/// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
/// ```
///
/// This is a leftmost-first alternation, so the alternatives are tried in
/// the written order at each position and the first that matches wins --
/// the same semantics a backtracking engine gives, not the
/// longest-alternative semantics of a POSIX engine. The optional leading
/// space in ` ?\p{L}+` is greedy, so the space is consumed when the next
/// character is a letter and skipped when it is not. `\s+(?!\S)` is a
/// greedy run with a negative lookahead, which after backtracking means:
/// all but the last character of a whitespace run that is followed by a
/// non-whitespace character, or the whole run at end of input. Every
/// alternative is written out here rather than delegated, because the
/// segmentation it produces is part of the pinned token ids.
fn split_byte_level_regex(input: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = input.char_indices().collect();
    let n = chars.len();
    let byte_at = |i: usize| -> usize {
        if i < n {
            chars[i].0
        } else {
            input.len()
        }
    };
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < n {
        let end = match_one(&chars, n, i);
        // Every character is covered by one of the alternatives (a letter, a
        // number, whitespace, or "none of those" -- which is exactly the
        // third class), so a zero-width result would mean this matcher and
        // the pattern above had drifted apart. Advancing by one character
        // keeps that a wrong segmentation rather than an infinite loop; the
        // tests below pin the segmentation itself.
        let end = if end > i { end } else { i + 1 };
        out.push(&input[byte_at(i)..byte_at(end)]);
        i = end;
    }
    out
}

/// Returns the character index just past the match starting at `i`.
fn match_one(chars: &[(usize, char)], n: usize, i: usize) -> usize {
    let c = |k: usize| -> Option<char> { chars.get(k).map(|&(_, ch)| ch) };

    // 's | 't | 're | 've | 'm | 'll | 'd
    if c(i) == Some('\'') {
        for word in ["s", "t", "re", "ve", "m", "ll", "d"] {
            let w: Vec<char> = word.chars().collect();
            if i + 1 + w.len() <= n && (0..w.len()).all(|k| c(i + 1 + k) == Some(w[k])) {
                return i + 1 + w.len();
            }
        }
    }

    // ` ?\p{L}+` then ` ?\p{N}+` then ` ?[^\s\p{L}\p{N}]+`, in that order.
    for class in [Class::Letter, Class::Number, Class::Other] {
        // Greedy `?`: try consuming the leading space first.
        for lead_space in [true, false] {
            let mut k = i;
            if lead_space {
                if c(k) != Some(' ') {
                    continue;
                }
                k += 1;
            }
            let start_body = k;
            while let Some(ch) = c(k) {
                if class.holds(ch) {
                    k += 1;
                } else {
                    break;
                }
            }
            if k > start_body {
                return k;
            }
        }
    }

    // `\s+(?!\S)`: the maximal whitespace run, minus its last character when
    // that run is followed by a non-whitespace character.
    let mut run_end = i;
    while let Some(ch) = c(run_end) {
        if is_whitespace(ch) {
            run_end += 1;
        } else {
            break;
        }
    }
    if run_end > i {
        let followed_by_non_space = run_end < n;
        if followed_by_non_space {
            if run_end - 1 > i {
                return run_end - 1;
            }
            // A single whitespace character followed by a non-whitespace one:
            // the lookahead alternative cannot match at all, so fall through
            // to the final bare `\s+`.
        } else {
            return run_end;
        }
        // `\s+`
        return run_end;
    }
    i
}

#[derive(Clone, Copy)]
enum Class {
    Letter,
    Number,
    Other,
}

impl Class {
    fn holds(self, c: char) -> bool {
        match self {
            Class::Letter => is_letter(c),
            Class::Number => is_number(c),
            // `[^\s\p{L}\p{N}]`
            Class::Other => !is_whitespace(c) && !is_letter(c) && !is_number(c),
        }
    }
}

/// The whole of spec 3.1.4: Digits, then ByteLevel's regex, then the
/// byte->unicode remap, producing the "words" the BPE model consumes.
pub fn pre_tokenize(input: &str, table: &[char; 256]) -> Vec<String> {
    let mut words = Vec::new();
    for segment in split_digits(input) {
        for piece in split_byte_level_regex(segment) {
            words.push(byte_remap(piece, table));
        }
    }
    words
}

// ---------------------------------------------------------------------------
// 3.1.5.c merge loop
// ---------------------------------------------------------------------------

/// One queued merge. Ordering is spec 3.1.5.c.ii: lowest rank first, and on
/// equal rank the leftmost position first. `BinaryHeap` pops the *greatest*
/// element, so `cmp` is written inverted.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Queued {
    rank: u32,
    pos: usize,
    new_id: u32,
}

impl Ord for Queued {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.pos.cmp(&self.pos))
    }
}
impl PartialOrd for Queued {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy)]
struct Symbol {
    id: u32,
    prev: isize,
    next: isize,
    /// Byte length; zero marks a symbol that a merge has absorbed.
    len: usize,
}

pub struct Tokenizer {
    vocab: BTreeMap<String, u32>,
    /// `(left_id, right_id) -> (rank, merged_id)`.
    merges: BTreeMap<(u32, u32), (u32, u32)>,
    byte_table: [char; 256],
}

impl Tokenizer {
    pub fn vocab_len(&self) -> usize {
        self.vocab.len()
    }
    pub fn merges_len(&self) -> usize {
        self.merges.len()
    }
    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.vocab.get(s).copied()
    }

    /// Parse a HuggingFace `tokenizer.json`, checking the model
    /// configuration spec 3.1.5 pins. Anything that would change the
    /// algorithm -- dropout, an unk token, a subword prefix or suffix,
    /// `byte_fallback`, `ignore_merges` -- is refused rather than ignored.
    pub fn from_json(bytes: &[u8]) -> Result<Tokenizer, String> {
        let root = parse(bytes).map_err(|e| alloc::format!("tokenizer.json: {e}"))?;

        // 3.1.3: the normalizer and post-processor must both be absent, or
        // the pinned prompt would not tokenize to 4 ids.
        match root.get("normalizer") {
            Some(Json::Null) | None => {}
            _ => return Err("tokenizer.json: spec 3.1.3 requires a null normalizer".into()),
        }
        match root.get("post_processor") {
            Some(Json::Null) | None => {}
            _ => return Err("tokenizer.json: spec 3.1.3 requires a null post_processor".into()),
        }
        check_pre_tokenizer(&root)?;

        let model = root
            .get("model")
            .ok_or("tokenizer.json: no model section")?;
        if model.get("type").and_then(|v| v.as_str()) != Some("BPE") {
            return Err("tokenizer.json: spec 3.1.5 requires model.type == BPE".into());
        }
        for (field, why) in [
            ("dropout", "spec 3.1.5 requires dropout = null (a deterministic merge)"),
            ("unk_token", "spec 3.1.5 requires unk_token = null"),
            ("continuing_subword_prefix", "spec 3.1.5 requires no continuing_subword_prefix"),
            ("end_of_word_suffix", "spec 3.1.5 requires no end_of_word_suffix"),
        ] {
            match model.get(field) {
                Some(Json::Null) | None => {}
                _ => return Err(alloc::format!("tokenizer.json: {why}")),
            }
        }
        for (field, want) in [("fuse_unk", false), ("byte_fallback", false), ("ignore_merges", false)] {
            if let Some(v) = model.get(field) {
                if v.as_bool() != Some(want) && *v != Json::Null {
                    return Err(alloc::format!(
                        "tokenizer.json: spec 3.1.5 requires {field} = {want}"
                    ));
                }
            }
        }

        let vocab_obj = match model.get("vocab") {
            Some(Json::Obj(fields)) => fields,
            _ => return Err("tokenizer.json: model.vocab is missing or not an object".into()),
        };
        let mut vocab: BTreeMap<String, u32> = BTreeMap::new();
        for (tok, id) in vocab_obj {
            let id = id
                .as_u64()
                .ok_or("tokenizer.json: a vocab id is not an integer")?;
            let id: u32 = id
                .try_into()
                .map_err(|_| "tokenizer.json: a vocab id does not fit in u32")?;
            if vocab.insert(tok.clone(), id).is_some() {
                return Err(alloc::format!("tokenizer.json: duplicate vocab entry {tok:?}"));
            }
        }

        let merges_arr = model
            .get("merges")
            .and_then(|v| v.as_arr())
            .ok_or("tokenizer.json: model.merges is missing or not an array")?;
        let mut merges: BTreeMap<(u32, u32), (u32, u32)> = BTreeMap::new();
        for (rank, entry) in merges_arr.iter().enumerate() {
            // Both encodings the format has used: "L R" and ["L", "R"].
            let (left, right) = match entry {
                Json::Str(s) => {
                    let mut parts = s.split(' ');
                    let l = parts.next().unwrap_or("");
                    let r = parts.next().unwrap_or("");
                    if parts.next().is_some() || l.is_empty() || r.is_empty() {
                        return Err(alloc::format!(
                            "tokenizer.json: merge rank {rank} is not two space-separated symbols"
                        ));
                    }
                    (String::from(l), String::from(r))
                }
                Json::Arr(pair) if pair.len() == 2 => {
                    let l = pair[0].as_str().ok_or("tokenizer.json: non-string merge symbol")?;
                    let r = pair[1].as_str().ok_or("tokenizer.json: non-string merge symbol")?;
                    (String::from(l), String::from(r))
                }
                _ => return Err(alloc::format!("tokenizer.json: malformed merge at rank {rank}")),
            };
            let mut joined = left.clone();
            joined.push_str(&right);
            let (lid, rid, mid) = match (vocab.get(&left), vocab.get(&right), vocab.get(&joined)) {
                (Some(a), Some(b), Some(c)) => (*a, *b, *c),
                // 3.1.5.c.iv resolves a merge through `model.vocab`; a merge
                // naming a symbol the vocab does not contain has no defined
                // result, so refuse the file instead of dropping the rule.
                _ => {
                    return Err(alloc::format!(
                        "tokenizer.json: merge rank {rank} ({left:?} + {right:?}) names a symbol \
                         that is not in model.vocab"
                    ))
                }
            };
            let rank: u32 = rank
                .try_into()
                .map_err(|_| "tokenizer.json: more merges than fit in u32")?;
            // The list position is the rank (3.1.5.c), and ranks are a total
            // order over distinct pairs -- so a repeated pair would make the
            // tie-break of 3.1.5.c.ii observable. Keep the first (highest
            // priority) and say so, rather than let a later duplicate win.
            merges.entry((lid, rid)).or_insert((rank, mid));
        }

        Ok(Tokenizer {
            vocab,
            merges,
            byte_table: bytes_char(),
        })
    }

    /// Spec 3.1.5.b-d for one byte-remapped word.
    fn encode_word(&self, word: &str) -> Result<Vec<u32>, String> {
        // (b) initial symbol sequence: one symbol per character, which by
        // construction of 3.1.4.b/3.1.5.a is one original input byte.
        let mut symbols: Vec<Symbol> = Vec::new();
        for (i, ch) in word.chars().enumerate() {
            let mut buf = [0u8; 4];
            let s: &str = ch.encode_utf8(&mut buf);
            let id = *self.vocab.get(s).ok_or_else(|| {
                alloc::format!("tokenizer: character {ch:?} is not a base vocab entry")
            })?;
            symbols.push(Symbol {
                id,
                prev: i as isize - 1,
                next: i as isize + 1,
                len: 1,
            });
        }
        if let Some(last) = symbols.last_mut() {
            last.next = -1;
        }

        // (c.i) seed the queue with every adjacent mergeable pair.
        let mut queue: BinaryHeap<Queued> = BinaryHeap::new();
        for pos in 0..symbols.len().saturating_sub(1) {
            if let Some(&(rank, new_id)) = self.merges.get(&(symbols[pos].id, symbols[pos + 1].id)) {
                queue.push(Queued { rank, pos, new_id });
            }
        }

        // (c.ii-v)
        while let Some(top) = queue.pop() {
            // (c.iii) re-validate: the position may have been absorbed by an
            // earlier merge, may no longer have a right neighbour, or may now
            // hold a different pair than the one that was queued.
            if symbols[top.pos].len == 0 || symbols[top.pos].next == -1 {
                continue;
            }
            let right_pos = symbols[top.pos].next as usize;
            let pair = (symbols[top.pos].id, symbols[right_pos].id);
            match self.merges.get(&pair) {
                Some(&(_, new_id)) if new_id == top.new_id => {}
                _ => continue,
            }

            // (c.iv) apply.
            let right = symbols[right_pos];
            symbols[top.pos].id = top.new_id;
            symbols[top.pos].len += right.len;
            symbols[top.pos].next = right.next;
            symbols[right_pos].len = 0;
            if right.next >= 0 {
                symbols[right.next as usize].prev = top.pos as isize;
            }

            // (c.iv, second half) enqueue the pairs the merge just created.
            let cur = symbols[top.pos];
            if cur.prev >= 0 {
                let p = cur.prev as usize;
                if let Some(&(rank, new_id)) = self.merges.get(&(symbols[p].id, cur.id)) {
                    queue.push(Queued { rank, pos: p, new_id });
                }
            }
            if cur.next >= 0 {
                let nx = cur.next as usize;
                if let Some(&(rank, new_id)) = self.merges.get(&(cur.id, symbols[nx].id)) {
                    queue.push(Queued {
                        rank,
                        pos: top.pos,
                        new_id,
                    });
                }
            }
        }

        Ok(symbols
            .into_iter()
            .filter(|s| s.len > 0)
            .map(|s| s.id)
            .collect())
    }

    /// The full encode of spec 3.1.3, with `add_special_tokens = false`
    /// (spec 3.3): no BOS, no EOS, no template -- so this is exactly the
    /// concatenation of 3.1.5.d.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let mut ids = Vec::new();
        for word in pre_tokenize(text, &self.byte_table) {
            ids.extend(self.encode_word(&word)?);
        }
        Ok(ids)
    }
}

fn check_pre_tokenizer(root: &Json) -> Result<(), String> {
    let pt = root
        .get("pre_tokenizer")
        .ok_or("tokenizer.json: no pre_tokenizer section")?;
    if pt.get("type").and_then(|v| v.as_str()) != Some("Sequence") {
        return Err("tokenizer.json: spec 3.1.4 requires a Sequence pre_tokenizer".into());
    }
    let list = pt
        .get("pretokenizers")
        .and_then(|v| v.as_arr())
        .ok_or("tokenizer.json: pre_tokenizer has no pretokenizers list")?;
    if list.len() != 2 {
        return Err("tokenizer.json: spec 3.1.4 pins exactly two pre-tokenizers".into());
    }
    if list[0].get("type").and_then(|v| v.as_str()) != Some("Digits")
        || list[0].get("individual_digits").and_then(|v| v.as_bool()) != Some(true)
    {
        return Err("tokenizer.json: spec 3.1.4.a pins Digits with individual_digits = true".into());
    }
    if list[1].get("type").and_then(|v| v.as_str()) != Some("ByteLevel")
        || list[1].get("add_prefix_space").and_then(|v| v.as_bool()) != Some(false)
        || list[1].get("use_regex").and_then(|v| v.as_bool()) != Some(true)
    {
        return Err("tokenizer.json: spec 3.1.4.b pins ByteLevel with add_prefix_space = false \
                    and use_regex = true"
            .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn byte_unicode_is_a_bijection_with_the_landmarks_spec_3_1_5_a_names() {
        let t = bytes_char();
        assert_eq!(t[0x20], '\u{0120}'); // space -> G-with-dot
        assert_eq!(t[0x0A], '\u{010A}'); // newline
        assert_eq!(t[b'A' as usize], 'A');
        assert_eq!(t[0xFF], '\u{00FF}');
        let mut seen: Vec<char> = t.to_vec();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 256, "the map must be injective over all 256 bytes");
        let self_mapped = t.iter().enumerate().filter(|(b, c)| **c as usize == *b).count();
        assert_eq!(self_mapped, 188, "spec 3.1.5.a: 188 bytes map to themselves");
    }

    /// Spec 3.1.7's worked example, at the segmentation stage.
    #[test]
    fn the_pinned_prompt_segments_into_the_four_words_spec_3_1_7_names() {
        let t = bytes_char();
        let words = pre_tokenize("Once upon a time", &t);
        assert_eq!(words, vec!["Once", "\u{0120}upon", "\u{0120}a", "\u{0120}time"]);
    }

    #[test]
    fn the_regexs_alternatives_split_where_the_pattern_says() {
        let t = bytes_char();
        // Contractions come first in the alternation.
        assert_eq!(
            pre_tokenize("don't", &t),
            vec!["don", "'t"]
        );
        // A digit run is isolated character by character by 3.1.4.a.
        assert_eq!(pre_tokenize("a12", &t), vec!["a", "1", "2"]);
        // ` ?[^\s\p{L}\p{N}]+` picks up punctuation with its leading space.
        assert_eq!(pre_tokenize("a !!", &t), vec!["a", "\u{0120}!!"]);
        // `\s+(?!\S)` keeps all but the last space of a run before a word;
        // the last space joins the word through ` ?\p{L}+`.
        assert_eq!(pre_tokenize("a   b", &t), vec!["a", "\u{0120}\u{0120}", "\u{0120}b"]);
        // A trailing run has no following non-space, so the whole run matches.
        assert_eq!(pre_tokenize("a  ", &t), vec!["a", "\u{0120}\u{0120}"]);
    }

    #[test]
    fn a_single_space_before_a_word_is_carried_into_the_word() {
        let t = bytes_char();
        assert_eq!(pre_tokenize(" a", &t), vec!["\u{0120}a"]);
    }
}
