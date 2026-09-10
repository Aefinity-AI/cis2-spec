//! E41: exercise `Tokenizer::decode` against a real 2.1 MB `tokenizer.json`
//! rather than the synthetic fixtures the CI tests use.
//!
//!   cargo run --release --example e41_detok_real -- <tokenizer.json> <label>

use cis2_verify::tokenizer::Tokenizer;

fn main() {
    let path = std::env::args().nth(1).expect("usage: <tokenizer.json> <label>");
    let label = std::env::args().nth(2).unwrap_or_else(|| "?".into());
    let bytes = std::fs::read(&path).expect("read tokenizer.json");
    let tok = Tokenizer::from_json(&bytes).expect("tokenizer parses");
    println!("E41-DETOK {label} vocab={} merges={}", tok.vocab_len(), tok.merges_len());

    // 1. Every id in the vocab decodes, and re-encodes to itself where the
    //    token is a whole pre-token. Count, do not assume.
    let mut decodable = 0usize;
    let mut undecodable = 0usize;
    let mut lone_roundtrip = 0usize;
    let mut lone_not_utf8 = 0usize;
    for id in 0..tok.vocab_len() as u32 {
        match tok.decode_bytes(&[id]) {
            Ok(b) => {
                decodable += 1;
                match core::str::from_utf8(&b) {
                    Ok(s) => {
                        if tok.encode(s).map(|v| v == vec![id]).unwrap_or(false) {
                            lone_roundtrip += 1;
                        }
                    }
                    Err(_) => lone_not_utf8 += 1,
                }
            }
            Err(_) => undecodable += 1,
        }
    }
    println!(
        "E41-DETOK {label} ids-decodable-to-bytes={decodable} undecodable={undecodable} \
         lone-token-is-valid-utf8-and-reencodes={lone_roundtrip} lone-token-not-utf8={lone_not_utf8} \
         (the last is >0 exactly when the vocab holds partial codepoints, which is why decode \
         concatenates bytes before validating)"
    );

    // 2. Round trip a corpus.
    let corpus = [
        "Once upon a time",
        "The quick brown fox jumps over the lazy dog 0123456789.",
        "accents: cafe\u{0301} café naïve Grüße",
        "cjk: 東京は晴れです",
        "emoji: 🜁🜂🜃🜄 👩‍👩‍👧",
        "rtl: مرحبا بالعالم",
        "\ttabs\n newlines \r\n and  double  spaces ",
        "code: fn main() { let x: u32 = 0xDEAD_BEEF; }",
    ];
    let mut ok = 0usize;
    for text in corpus {
        let ids = tok.encode(text).expect("encode");
        let back = tok.decode(&ids).expect("decode");
        assert_eq!(back, text, "round trip changed {text:?}");
        assert_eq!(tok.encode(&back).expect("re-encode"), ids, "encode(decode(ids)) != ids");
        ok += 1;
    }
    println!("E41-DETOK {label} corpus-round-trips={ok}/{}", corpus.len());

    // 3. The 13.1 reference decode's own ids, if supplied.
    if let Some(list) = std::env::args().nth(3) {
        let ids: Vec<u32> = list.split(',').map(|t| t.trim().parse().expect("id")).collect();
        let text = tok.decode(&ids).expect("decode generated ids");
        println!("E41-DETOK {label} generated-text={text:?}");
        println!(
            "E41-DETOK {label} reencode-matches={}",
            tok.encode(&text).expect("re-encode") == ids
        );
    }
}
