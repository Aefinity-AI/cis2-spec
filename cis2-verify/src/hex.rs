//! Lowercase hex rendering and parsing. Spec 12: digests are 32 raw bytes,
//! rendered as 64 lowercase hex characters for display only.

use alloc::string::String;
use alloc::vec::Vec;

pub fn encode(bytes: &[u8]) -> String {
    const D: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(D[(b >> 4) as usize] as char);
        s.push(D[(b & 0x0f) as usize] as char);
    }
    s
}

pub fn decode(s: &str) -> Result<Vec<u8>, &'static str> {
    let b = s.as_bytes();
    if b.len() % 2 != 0 {
        return Err("hex string has odd length");
    }
    let mut out = Vec::with_capacity(b.len() / 2);
    let nib = |c: u8| -> Result<u8, &'static str> {
        match c {
            b'0'..=b'9' => Ok(c - b'0'),
            b'a'..=b'f' => Ok(c - b'a' + 10),
            _ => Err("hex string contains a non-lowercase-hex character"),
        }
    };
    for pair in b.chunks(2) {
        out.push((nib(pair[0])? << 4) | nib(pair[1])?);
    }
    Ok(out)
}
