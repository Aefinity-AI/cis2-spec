//! Erratum E-14: CIS-2 v0.3b specifies text -> ids completely (3.1.3-3.1.5)
//! and says nothing about ids -> text beyond 3.1.5.a's remark that the *byte*
//! map's inverse is used. `Tokenizer::decode` is this crate's documented
//! extension, and these tests pin the three decisions it makes where the spec
//! is silent:
//!
//!   1. bytes are concatenated across tokens BEFORE UTF-8 validation, because
//!      a codepoint can straddle a token boundary;
//!   2. an id with no string in `model.vocab` is an error, not a substitution;
//!   3. a `model.vocab` that maps two strings to one id is refused outright,
//!      because it has no inverse.
//!
//! Each test carries a control that fails if the mechanism under test is not
//! actually doing the work (E35 3: a check that passes must be shown able to
//! fail).

use cis2_verify::tokenizer::{bytes_char, DetokError, Tokenizer};

/// A `tokenizer.json` with the 3.1.3/3.1.4/3.1.5 pins the spec requires and a
/// caller-chosen vocab and merge list.
fn tokenizer_json(vocab: &[(String, u32)], merges: &[&str]) -> Vec<u8> {
    let esc = |s: &str| {
        let mut o = String::new();
        for c in s.chars() {
            match c {
                '"' => o.push_str("\\\""),
                '\\' => o.push_str("\\\\"),
                c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
                c => o.push(c),
            }
        }
        o
    };
    let v: Vec<String> =
        vocab.iter().map(|(s, i)| format!("\"{}\":{}", esc(s), i)).collect();
    let m: Vec<String> = merges.iter().map(|s| format!("\"{}\"", esc(s))).collect();
    format!(
        r#"{{"decoder":{{"type":"ByteLevel"}},
           "model":{{"byte_fallback":false,"continuing_subword_prefix":null,"dropout":null,
                     "end_of_word_suffix":null,"fuse_unk":false,"ignore_merges":false,
                     "merges":[{}],"type":"BPE","unk_token":null,"vocab":{{{}}}}},
           "normalizer":null,"post_processor":null,
           "pre_tokenizer":{{"pretokenizers":[{{"individual_digits":true,"type":"Digits"}},
             {{"add_prefix_space":false,"trim_offsets":true,"type":"ByteLevel","use_regex":true}}],
             "type":"Sequence"}},
           "version":"1.0"}}"#,
        m.join(","),
        v.join(",")
    )
    .into_bytes()
}

/// One token per byte, no merges: the identity BPE. Enough to encode and
/// decode any byte string, so a round trip over it tests the maps and not the
/// merge search.
fn byte_identity() -> Tokenizer {
    let t = bytes_char();
    let vocab: Vec<(String, u32)> =
        t.iter().enumerate().map(|(b, &c)| (c.to_string(), b as u32)).collect();
    Tokenizer::from_json(&tokenizer_json(&vocab, &[])).expect("byte-identity tokenizer")
}

#[test]
fn byte_table_inverse_is_a_bijection() {
    let t = bytes_char();
    let mut seen = std::collections::BTreeSet::new();
    for &c in t.iter() {
        assert!(seen.insert(c), "byte table maps two bytes to {c:?}; it has no inverse");
    }
    assert_eq!(seen.len(), 256);
}

/// The decision that matters most, and the one a per-token decoder gets
/// wrong. "é" is U+00E9 = the two bytes C3 A9. Under 3.1.5.a those bytes map
/// to two separate codepoints, and with no merge covering them they encode as
/// two tokens. Neither token's bytes are valid UTF-8 alone.
#[test]
fn a_codepoint_split_across_tokens_decodes_only_after_concatenation() {
    let tok = byte_identity();
    let ids = tok.encode("é").expect("encode");
    assert_eq!(ids.len(), 2, "expected é to split into two byte tokens, got {ids:?}");
    assert_eq!(ids, vec![0xC3, 0xA9], "byte-identity ids should be the bytes themselves");

    // CONTROL: this is only a test of the concatenation rule if the pieces
    // really are individually undecodable. If a future change made single
    // tokens valid here, the assertion below would pass for the wrong reason.
    assert_eq!(tok.decode(&ids[..1]), Err(DetokError::NotUtf8), "control inert: piece 0 decodes alone");
    assert_eq!(tok.decode(&ids[1..]), Err(DetokError::NotUtf8), "control inert: piece 1 decodes alone");

    assert_eq!(tok.decode(&ids).expect("joint decode"), "é");
    assert_eq!(tok.decode_bytes(&ids).expect("bytes"), vec![0xC3u8, 0xA9]);
}

#[test]
fn round_trip_over_a_corpus() {
    let tok = byte_identity();
    let corpus = [
        "",
        "Once upon a time",
        "  leading and trailing  ",
        "digits 0123456789 split individually",
        "accents: café naïve Grüße",
        "cjk: 東京は晴れです",
        "emoji: 🜁🜂🜃🜄 and a family 👩‍👩‍👧",
        "punctuation!?;:'\"()[]{}<>@#$%^&*",
        "tabs\tand\nnewlines\r\n",
        "rtl: مرحبا بالعالم",
        "combining: e\u{0301} vs \u{00e9}",
    ];
    for text in corpus {
        let ids = tok.encode(text).expect("encode");
        let back = tok.decode(&ids).expect("decode");
        assert_eq!(back, text, "round trip changed {text:?} into {back:?}");
        // And the other direction: re-encoding the decoded text must give the
        // same ids, or the pair is not an inverse on this input.
        assert_eq!(tok.encode(&back).expect("re-encode"), ids, "encode(decode(ids)) != ids for {text:?}");
    }
}

/// The committed fixture's generated ids, decoded and re-encoded. Uses the
/// real merge table rather than the identity BPE, so a merge that decoded to
/// the wrong string would show up here.
#[test]
fn fixture_generated_ids_round_trip_through_the_merge_table() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/tiny/tokenizer.json");
    let bytes = std::fs::read(path).expect("fixture tokenizer");
    let tok = Tokenizer::from_json(&bytes).expect("fixture tokenizer parses");
    // From tests/end_to_end.rs: the ids that fixture decode generates.
    let gen: &[u32] = &[6, 6, 1, 5, 1, 6];
    let text = tok.decode(gen).expect("decode fixture ids");
    assert_eq!(text, "ggbfbg");
    assert_eq!(tok.encode(&text).expect("re-encode"), gen.to_vec());
    // A merged token really is stored as the joined string.
    assert_eq!(tok.token_str(15), Some("abc"));
    assert_eq!(tok.decode(&[15]).expect("merged token"), "abc");
    assert_eq!(tok.encode("abc").expect("encode abc"), vec![15]);
}

/// 12.1's argmax ranges over `config.vocab_size` logits. On
/// `Qwen/Qwen2.5-0.5B` that is 151936 against 151643 `model.vocab` entries
/// plus 22 `added_tokens` at 151643..=151664, leaving ids 151665..=151935
/// emittable with no string anywhere. Decoding one is an error here, not a
/// silent skip.
#[test]
fn an_id_with_no_string_is_an_error_not_a_substitution() {
    let tok = byte_identity();
    assert_eq!(tok.token_str(256), None);
    assert_eq!(tok.decode(&[256]), Err(DetokError::UnmappedId(256)));
    // CONTROL: the same call on a mapped id must succeed, or the error above
    // could be coming from anything.
    assert_eq!(tok.decode(&[0x41]).expect("mapped id"), "A");
    // An unmapped id anywhere in the sequence fails the whole decode.
    assert_eq!(tok.decode(&[0x41, 999, 0x42]), Err(DetokError::UnmappedId(999)));
}

#[test]
fn a_vocab_with_no_inverse_is_refused() {
    let t = bytes_char();
    let mut vocab: Vec<(String, u32)> =
        t.iter().enumerate().map(|(b, &c)| (c.to_string(), b as u32)).collect();
    // Two distinct strings, one id: the id -> string map is not a function.
    vocab.push(("Ab".to_string(), 0x41));
    let err = match Tokenizer::from_json(&tokenizer_json(&vocab, &[])) {
        Ok(_) => panic!("a vocab with two strings on one id was accepted"),
        Err(e) => e,
    };
    assert!(err.contains("not a function"), "unexpected refusal: {err}");
    // CONTROL: the same vocab without the collision is accepted, so the
    // refusal is the collision and not the extra entry per se.
    vocab.pop();
    vocab.push(("Ab".to_string(), 300));
    assert!(Tokenizer::from_json(&tokenizer_json(&vocab, &["A b"])).is_ok(),
        "control inert: the no-collision vocab was also refused");
}

/// The case with a consequence beyond rendering. 3.3 pins
/// `add_special_tokens = false`, so 3 never *produces* a special-token id --
/// but 11.2's argmax can emit one. Decoding it yields its literal text, and
/// re-encoding that text gives ordinary tokens, never the special id back.
/// Once a decode has been rendered to text, a model-emitted control token is
/// indistinguishable from generated text that merely spells it.
///
/// Measured on `HuggingFaceTB/SmolLM2-135M`: all 17 `added_tokens` are ids
/// 0..=16 and all 17 behave this way; id 2 (`<|im_end|>`) re-encodes to
/// `[44, 108, 306, 79, 486, 108, 46]`. Reproduced here on a synthetic vocab
/// so it runs in CI without the 2.1 MB checkpoint tokenizer.
#[test]
fn a_special_token_decodes_to_text_that_does_not_re_encode_to_it() {
    let t = bytes_char();
    let mut vocab: Vec<(String, u32)> =
        t.iter().enumerate().map(|(b, &c)| (c.to_string(), b as u32)).collect();
    // A special token whose *string* is spellable from ordinary byte tokens.
    vocab.push(("<|im_end|>".to_string(), 300));
    let tok = Tokenizer::from_json(&tokenizer_json(&vocab, &[])).expect("parse");

    let text = tok.decode(&[300]).expect("decode the special id");
    assert_eq!(text, "<|im_end|>");

    let back = tok.encode(&text).expect("re-encode");
    assert_ne!(back, vec![300], "encode returned the special id; 3.3 pins add_special_tokens = false");
    // It is exactly the byte spelling, which is the point: the text carries no
    // mark distinguishing "the model emitted the control token" from "the
    // model emitted these ten characters".
    let spelled: Vec<u32> = "<|im_end|>".bytes().map(|b| b as u32).collect();
    assert_eq!(back, spelled);

    // The two inputs that a text-only consumer cannot tell apart, and the ids
    // that do tell them apart.
    assert_eq!(tok.decode(&spelled).expect("decode the spelling"), text);
    assert_ne!(spelled, vec![300]);

    // CONTROL: an ordinary token round trips, so the assertion above is about
    // special tokens and not about decode being broken in general.
    let ids = tok.encode("hello").expect("encode");
    assert_eq!(tok.encode(&tok.decode(&ids).expect("decode")).expect("re-encode"), ids);
}
